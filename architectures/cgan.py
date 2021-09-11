
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Lambda, ReLU, Subtract, Reshape, Permute, Activation
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.layers.merge import _Merge
from utils import optimizers_builtin, loss_functions, metrics
import numpy as np
import keras.backend as K


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


# define the combined generator and discriminator model, for updating the generator
def build_cgan(gen_conf, train_conf, g_model, d_model):

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    patch_shape = train_conf['patch_shape']
    # optimizer = train_conf['optimizer']
    # initial_lr = train_conf['initial_lr']
    loss_opt = train_conf['GAN']['generator']['loss']
    d_loss = train_conf['GAN']['discriminator']['loss']
    num_classes = train_conf['GAN']['generator']['num_classes']
    metric_opt = train_conf['GAN']['generator']['metric']
    multi_output = train_conf['GAN']['generator']['multi_output']
    output_name = train_conf['GAN']['generator']['output_name']
    attention_loss = train_conf['GAN']['generator']['attention_loss']
    overlap_penalty_loss = train_conf['GAN']['generator']['overlap_penalty_loss']
    lamda = train_conf['GAN']['generator']['lamda']
    adv_loss_weight = train_conf['GAN']['generator']['adv_loss_weight']
    optimizer = train_conf['GAN']['generator']['optimizer']
    initial_lr = train_conf['GAN']['generator']['initial_lr']

    input_shape = (num_modality,) + patch_shape

    # make weights in the discriminator not trainable
    d_model.trainable = False

    # define the source image
    in_src = Input(shape=input_shape)

    # connect the source image to the generator input
    gen_out = g_model(in_src)

    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out[0], gen_out[1]])

    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out[0], gen_out[1], gen_out[2], gen_out[3]])
    model.summary()

    dentate_prob_output = Lambda(lambda x: x[:, :, num_classes[0] - 1])(gen_out[0])
    interposed_prob_output = Lambda(lambda x: x[:, :, num_classes[1] - 1])(gen_out[1])

    # loss_multi = {
    #     g_output_name[0]: loss_functions.select(g_num_classes[0], loss_opt[0]),
    #     g_output_name[1]: loss_functions.select(g_num_classes[1], loss_opt[1]),
    #     'attention_maps': 'categorical_crossentropy',
    #     'overlap_dentate_interposed': loss_functions.dc_btw_dentate_interposed(dentate_prob_output,
    #                                                                            interposed_prob_output)
    # }
    # g_loss_weights = {g_output_name[0]: g_lamda[0], g_output_name[1]: g_lamda[1], 'attention_maps': g_lamda[2],
    #                 'overlap_dentate_interposed': g_lamda[3]}
    #metric_f = ['loss', metrics.select(g_metric_opt)]

    # compile model
    loss_f = [d_loss, loss_functions.select(num_classes[0], loss_opt[0]),
              loss_functions.select(num_classes[1], loss_opt[1]), 'categorical_crossentropy',
              loss_functions.dc_btw_dentate_interposed(dentate_prob_output, interposed_prob_output)]  # adversarial loss via discriminator output + L1 loss via the direct image output
    loss_weights_f = [adv_loss_weight, lamda[0], lamda[1], lamda[2], lamda[3]]
    #loss_f = ['binary_crossentropy', loss_functions.select(2, loss_opt[0]), loss_functions.select(2, loss_opt[1])]  # adversarial loss via discriminator output + L1 loss via the direct image output
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_f, optimizer=optimizers_builtin.select(optimizer, initial_lr, clipvalue=1.0),
                  loss_weights=loss_weights_f) #metrics=metrics.select(metric_opt)

    d_model.trainable = True

    return model


def build_cgan_sar(gen_conf, train_conf, g_model, d_model):

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    dimension = train_conf['dimension']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    patch_shape = train_conf['patch_shape']
    # optimizer = train_conf['optimizer']
    # initial_lr = train_conf['initial_lr']
    g_loss = train_conf['GAN']['generator']['loss']
    d_loss = train_conf['GAN']['discriminator']['loss']
    lamda = train_conf['GAN']['generator']['lamda']
    adv_loss_weight = train_conf['GAN']['generator']['adv_loss_weight']
    g_patch_shape = train_conf['GAN']['generator']['patch_shape']
    g_output_shape = train_conf['GAN']['generator']['output_shape']
    d_patch_shape = train_conf['GAN']['discriminator']['patch_shape']
    num_classes_g = train_conf['GAN']['generator']['num_classes']
    optimizer = train_conf['GAN']['generator']['optimizer']
    initial_lr = train_conf['GAN']['generator']['initial_lr']

    if dimension == 2:
        g_input_shape = (g_patch_shape[0], g_patch_shape[1], num_modality)
        d_input_shape = (d_patch_shape[0], d_patch_shape[1], num_modality)
    else:
        g_input_shape = (num_modality,) + g_patch_shape
        d_input_shape = (num_modality,) + d_patch_shape

    # make weights in the discriminator not trainable
    d_model.trainable = False

    # define the source image
    g_in_src = Input(shape=g_input_shape)
    if g_patch_shape != d_patch_shape:
        d_in_src = Input(shape=d_input_shape)

    output_shape_prod = (np.prod(g_output_shape), num_classes_g)
    g_sar_real = Input(shape=output_shape_prod)

    gen_out = g_model(g_in_src)

    print(g_sar_real.shape)
    print(gen_out.shape)

    tf_g_sar_real = Lambda(lambda x: x)(g_sar_real)
    tf_gen_out = Lambda(lambda x: x)(gen_out)

    print(tf_g_sar_real.shape)
    print(tf_gen_out.shape)

    # connect the source input and generator output to the discriminator input
    if g_patch_shape != d_patch_shape:
        [dis_out, dis_e_out, interm_f1_out, interm_f2_out, interm_f3_out] = d_model([d_in_src, gen_out])
        #[dis_out, interm_f1_out, interm_f2_out, interm_f3_out] = d_model([d_in_src, gen_out])
        model = Model([g_in_src, d_in_src, g_sar_real], [dis_out, dis_e_out, gen_out, interm_f1_out, interm_f2_out, interm_f3_out,
                                                         g_sar_real, g_sar_real, g_sar_real, g_sar_real]) # output g_sar_real is garbage for peak loss
    else:
        [dis_out, dis_e_out, interm_f1_out, interm_f2_out, interm_f3_out] = d_model([g_in_src, gen_out])
        #[dis_out, interm_f1_out, interm_f2_out, interm_f3_out] = d_model([g_in_src, gen_out])
        model = Model([g_in_src, g_sar_real], [dis_out, dis_e_out, gen_out, interm_f1_out, interm_f2_out, interm_f3_out,
                                               g_sar_real, g_sar_real, g_sar_real, g_sar_real])
    model.summary()

    #adversarial loss via discriminator output + L1 loss + L2 perceptual loss via the direct image output
    # loss_f = [d_loss, g_loss, 'mae', 'mae', 'mae',
    #           loss_functions.local_sar_peak_loss(tf_g_sar_real, tf_gen_out),
    #           loss_functions.local_sar_real_peak(tf_g_sar_real),
    #           loss_functions.local_sar_pred_peak(tf_gen_out),
    #           loss_functions.sar_tversky_focal_loss(tf_g_sar_real, tf_gen_out, thres=0.3, tv_alpha=0.3, tv_beta=0.7,
    #                                                 f_gamma=2.0, f_alpha=0.25, w_focal=0.5)]

    loss_f = [d_loss, d_loss, g_loss, 'mae', 'mae', 'mae',
              loss_functions.local_sar_peak_loss(tf_g_sar_real, tf_gen_out),
              loss_functions.local_sar_max(tf_g_sar_real),
              loss_functions.local_sar_max(tf_gen_out),
              loss_functions.sar_tversky_focal_loss(tf_g_sar_real, tf_gen_out, thres=0.0, tv_alpha=0.3, tv_beta=0.7,
                                                    f_gamma=2.0, f_alpha=0.25, w_focal=0.5)]
    #loss_weights_f = [adv_loss_weight, lamda, 0.5/3, 0.5/3, 0.5/3, 0.001, 0, 0, 0.001]
    #loss_weights_f = [adv_loss_weight, lamda[0], lamda[1], lamda[2], lamda[3], lamda[4], 0, 0,
    #                  lamda[5]]
    loss_weights_f = [adv_loss_weight, adv_loss_weight, lamda[0], lamda[1], lamda[2], lamda[3], lamda[4], 0, 0, lamda[5]]
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_f, optimizer=optimizers_builtin.select(optimizer, initial_lr, clipvalue=1.0), loss_weights=loss_weights_f)

    d_model.trainable = True

    return model


def build_cgan_sar_lse(gen_conf, train_conf, g_model, d_model, local_sar_model, g_encoder_ch):

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    dimension = train_conf['dimension']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    patch_shape = train_conf['patch_shape']
    # optimizer = train_conf['optimizer']
    # initial_lr = train_conf['initial_lr']
    g_loss = train_conf['GAN']['generator']['loss']
    d_loss = train_conf['GAN']['discriminator']['loss']
    lamda = train_conf['GAN']['generator']['lamda']
    adv_loss_weight = train_conf['GAN']['generator']['adv_loss_weight']
    g_patch_shape = train_conf['GAN']['generator']['patch_shape']
    g_output_shape = train_conf['GAN']['generator']['output_shape']
    d_patch_shape = train_conf['GAN']['discriminator']['patch_shape']
    num_classes_g = train_conf['GAN']['generator']['num_classes']
    optimizer = train_conf['GAN']['generator']['optimizer']
    initial_lr = train_conf['GAN']['generator']['initial_lr']

    if dimension == 2:
        g_input_shape = (g_patch_shape[0], g_patch_shape[1], num_modality)
        f_output_shape = (np.divide(g_patch_shape[0], 8).astype(int), np.divide(g_patch_shape[1], 8).astype(int),
                          g_encoder_ch)
    else:
        g_input_shape = (num_modality,) + g_patch_shape
        f_output_shape = (g_encoder_ch,) + np.divide(g_patch_shape, 8).astype(int)

    # make weights in the discriminator not trainable
    d_model.trainable = False

    # define the source image
    g_in_src = Input(shape=g_input_shape)
    output_shape_prod = (np.prod(g_output_shape), num_classes_g)
    g_sar_real_norm = Input(shape=output_shape_prod)  # normalized SAR
    g_sar_real = Input(shape=output_shape_prod)  # non-normalized SAR
    fb_output = Input(shape=f_output_shape)

    # [_, encoder_out] = g_model([g_in_src, fb_output])  # fb_output: zero
    # [gen_out, _] = g_model([g_in_src, encoder_out])

    [gen_out, encoder_out] = g_model([g_in_src, fb_output])  # fb_output: zero

    [dis_out, dis_e_out, interm_f1_out, interm_f2_out, interm_f3_out] = d_model([g_in_src, gen_out])

    [local_sar_min_output, local_sar_max_output, attn_map_low, attn_map_high] = local_sar_model([encoder_out])

    tf_g_sar_real_norm = Lambda(lambda x: x)(g_sar_real_norm)
    tf_g_sar_real = Lambda(lambda x: x)(g_sar_real)
    tf_gen_out = Lambda(lambda x: x)(gen_out)
    tf_local_sar_min_out = Lambda(lambda x: x)(local_sar_min_output)
    tf_local_sar_max_out = Lambda(lambda x: x)(local_sar_max_output)


    print(g_sar_real.shape)
    print(gen_out.shape)

    print(tf_g_sar_real.shape)

    model = Model([g_in_src, fb_output, g_sar_real_norm, g_sar_real], [dis_out, dis_e_out, gen_out, attn_map_low, attn_map_high,
                                                            interm_f1_out, interm_f2_out, interm_f3_out, g_sar_real,
                                                            g_sar_real, g_sar_real, g_sar_real, g_sar_real, g_sar_real,
                                                            g_sar_real, g_sar_real, g_sar_real])
    model.summary()

    loss_f = [d_loss, d_loss, g_loss, 'mae', 'mae', 'mae', 'mae', 'mae',
              loss_functions.local_sar_min_loss(tf_g_sar_real, tf_local_sar_min_out),
              loss_functions.local_sar_max_loss(tf_g_sar_real, tf_local_sar_max_out),
              loss_functions.local_sar_min(tf_g_sar_real),
              loss_functions.local_sar_max(tf_g_sar_real),
              loss_functions.local_sar_est(tf_local_sar_min_out),
              loss_functions.local_sar_est(tf_local_sar_max_out),
              loss_functions.local_sar_max(tf_g_sar_real_norm),
              loss_functions.local_sar_max(tf_gen_out),
              loss_functions.sar_tversky_focal_loss(tf_g_sar_real_norm, tf_gen_out, thres=0.0, tv_alpha=0.3, tv_beta=0.7,
                                                    f_gamma=2.0, f_alpha=0.25, w_focal=0.5)]

    loss_weights_f = [adv_loss_weight, adv_loss_weight, lamda[0], lamda[1], lamda[2], lamda[3], lamda[4], lamda[5],
                      lamda[6], lamda[7], 0, 0, 0, 0, 0, 0, lamda[8]]

    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_f, optimizer=optimizers_builtin.select(optimizer, initial_lr, clipvalue=1.0), loss_weights=loss_weights_f)

    d_model.trainable = True

    return model


def build_cgan_sar_fsmt_lse(gen_conf, train_conf, g_model, d_model, fsmt_model, local_sar_model, g_encoder_ch):

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    dimension = train_conf['dimension']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    patch_shape = train_conf['patch_shape']
    # optimizer = train_conf['optimizer']
    # initial_lr = train_conf['initial_lr']
    g_loss = train_conf['GAN']['generator']['loss']
    d_loss = train_conf['GAN']['discriminator']['loss']
    lamda = train_conf['GAN']['generator']['lamda']
    adv_loss_weight = train_conf['GAN']['generator']['adv_loss_weight']
    g_patch_shape = train_conf['GAN']['generator']['patch_shape']
    g_output_shape = train_conf['GAN']['generator']['output_shape']
    d_patch_shape = train_conf['GAN']['discriminator']['patch_shape']
    num_classes_g = train_conf['GAN']['generator']['num_classes']
    optimizer = train_conf['GAN']['generator']['optimizer']
    initial_lr = train_conf['GAN']['generator']['initial_lr']
    num_model_samples = train_conf['num_model_samples']

    if dimension == 2:
        g_input_shape = (g_patch_shape[0], g_patch_shape[1], num_modality)
        f_output_shape = (np.divide(g_patch_shape[0], 8).astype(int), np.divide(g_patch_shape[1], 8).astype(int),
                          g_encoder_ch)
    else:
        g_input_shape = (num_modality,) + g_patch_shape
        f_output_shape = (g_encoder_ch,) + np.divide(g_patch_shape, 8).astype(int)

    # make weights in the discriminator not trainable
    d_model.trainable = False

    # define the source image
    g_in_src = Input(shape=g_input_shape)
    output_shape_prod = (np.prod(g_output_shape), num_classes_g)
    g_sar_real_norm = Input(shape=output_shape_prod)  # normalized SAR
    g_sar_real = Input(shape=output_shape_prod)  # non-normalized SAR
    fb_output = Input(shape=f_output_shape)

    model_samples = []
    for i in range(num_model_samples):
        model_samples.append(Input(shape=output_shape_prod))

    # [_, encoder_out] = g_model([g_in_src, fb_output])  # fb_output: zero
    # [gen_out, _] = g_model([g_in_src, encoder_out])

    [_, encoder_out] = g_model([g_in_src, fb_output])  # fb_output: zero
    decoder_in = fsmt_model([g_in_src] + model_samples + [encoder_out])
    [gen_out, _] = g_model([g_in_src, decoder_in])  # fb_output: zero

    [dis_out, dis_e_out, interm_f1_out, interm_f2_out, interm_f3_out] = d_model([g_in_src, gen_out])

    [local_sar_min_output, local_sar_max_output, attn_map_low, attn_map_high] = local_sar_model([decoder_in])

    tf_g_sar_real_norm = Lambda(lambda x: x)(g_sar_real_norm)
    tf_g_sar_real = Lambda(lambda x: x)(g_sar_real)
    tf_gen_out = Lambda(lambda x: x)(gen_out)
    tf_local_sar_min_out = Lambda(lambda x: x)(local_sar_min_output)
    tf_local_sar_max_out = Lambda(lambda x: x)(local_sar_max_output)

    print(g_sar_real.shape)
    print(gen_out.shape)

    print(tf_g_sar_real.shape)

    model = Model([g_in_src, fb_output, g_sar_real_norm, g_sar_real] + model_samples,
                  [dis_out, dis_e_out, gen_out, attn_map_low, attn_map_high, interm_f1_out, interm_f2_out,
                   interm_f3_out, g_sar_real, g_sar_real, g_sar_real, g_sar_real, g_sar_real, g_sar_real,
                   g_sar_real, g_sar_real, g_sar_real])

    model.summary()

    loss_f = [d_loss, d_loss, g_loss, 'mae', 'mae', 'mae', 'mae', 'mae',
              loss_functions.local_sar_min_loss(tf_g_sar_real, tf_local_sar_min_out),
              loss_functions.local_sar_max_loss(tf_g_sar_real, tf_local_sar_max_out),
              loss_functions.local_sar_min(tf_g_sar_real),
              loss_functions.local_sar_max(tf_g_sar_real),
              loss_functions.local_sar_est(tf_local_sar_min_out),
              loss_functions.local_sar_est(tf_local_sar_max_out),
              loss_functions.local_sar_max(tf_g_sar_real_norm),
              loss_functions.local_sar_max(tf_gen_out),
              loss_functions.sar_tversky_focal_loss(tf_g_sar_real_norm, tf_gen_out, thres=0.0, tv_alpha=0.3, tv_beta=0.7,
                                                    f_gamma=2.0, f_alpha=0.25, w_focal=0.5)]

    loss_weights_f = [adv_loss_weight, adv_loss_weight, lamda[0], lamda[1], lamda[2], lamda[3], lamda[4], lamda[5],
                      lamda[6], lamda[7], 0, 0, 0, 0, 0, 0, lamda[8]]

    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_f, optimizer=optimizers_builtin.select(optimizer, initial_lr, clipvalue=1.0), loss_weights=loss_weights_f)

    d_model.trainable = True

    return model


def build_cgan_feedback_sar(gen_conf, train_conf, g_model, d_model, fb_model, num_fb_loop, g_encoder_ch):

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    dimension = train_conf['dimension']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    patch_shape = train_conf['patch_shape']
    # optimizer = train_conf['optimizer']
    # initial_lr = train_conf['initial_lr']
    g_loss = train_conf['GAN']['generator']['loss']
    d_loss = train_conf['GAN']['discriminator']['loss']
    lamda = train_conf['GAN']['generator']['lamda']
    adv_loss_weight = train_conf['GAN']['generator']['adv_loss_weight']
    g_patch_shape = train_conf['GAN']['generator']['patch_shape']
    g_output_shape = train_conf['GAN']['generator']['output_shape']
    d_patch_shape = train_conf['GAN']['discriminator']['patch_shape']
    num_classes_g = train_conf['GAN']['generator']['num_classes']
    optimizer = train_conf['GAN']['generator']['optimizer']
    initial_lr = train_conf['GAN']['generator']['initial_lr']

    if dimension == 2:
        g_input_shape = (g_patch_shape[0], g_patch_shape[1], num_modality)
        f_output_shape = (np.divide(g_patch_shape[0], 8).astype(int), np.divide(g_patch_shape[1], 8).astype(int), g_encoder_ch)
    else:
        g_input_shape = (num_modality,) + g_patch_shape
        f_output_shape = (g_encoder_ch,) + np.divide(g_patch_shape, 8).astype(int)

    # make weights in the discriminator not trainable
    d_model.trainable = False

    # define the source image
    g_in_src = Input(shape=g_input_shape)
    output_shape_prod = (np.prod(g_output_shape), num_classes_g)
    g_sar_real = Input(shape=output_shape_prod)
    fb_output = Input(shape=f_output_shape)

    # connect the source input and generator output to the discriminator input
    ## DO NOT USE "for loop" of if statement when connecting models (due to backpropagation)

    # [_, encoder_out] = g_model([g_in_src, fb_output])  # fb_output: zero
    # [gen_out, _] = g_model([g_in_src, encoder_out])

    [gen_out, encoder_out] = g_model([g_in_src, fb_output])  # fb_output: zero

    for n in range(num_fb_loop):
        [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
        decoder_in = fb_model([gen_out, dis_out, encoder_out])
        [gen_out, encoder_out] = g_model([g_in_src, decoder_in])

    # [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
    # decoder_in = fb_model([gen_out, dis_out, encoder_out])
    # [gen_out, encoder_out] = g_model([g_in_src, decoder_in])
    #
    # [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
    # decoder_in = fb_model([gen_out, dis_out, encoder_out])
    # [gen_out, encoder_out] = g_model([g_in_src, decoder_in])

    # [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
    # decoder_in = fb_model([gen_out, dis_out, encoder_out])
    # [gen_out, encoder_out] = g_model([g_in_src, decoder_in])
    #
    # [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
    # decoder_in = fb_model([gen_out, dis_out, encoder_out])
    # [gen_out, encoder_out] = g_model([g_in_src, decoder_in])
    #
    # [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
    # decoder_in = fb_model([gen_out, dis_out, encoder_out])
    # [gen_out, _] = g_model([g_in_src, decoder_in])

    [dis_out, dis_e_out, interm_f1_out, interm_f2_out, interm_f3_out] = d_model([g_in_src, gen_out])

    tf_g_sar_real = Lambda(lambda x: x)(g_sar_real)
    tf_gen_out = Lambda(lambda x: x)(gen_out)

    print(g_sar_real.shape)
    print(gen_out.shape)

    print(tf_g_sar_real.shape)
    print(tf_gen_out.shape)

    model = Model([g_in_src, fb_output, g_sar_real], [dis_out, dis_e_out, gen_out, interm_f1_out, interm_f2_out, interm_f3_out,
                                           g_sar_real, g_sar_real, g_sar_real, g_sar_real])
    model.summary()

    loss_f = [d_loss, d_loss, g_loss, 'mae', 'mae', 'mae',
              loss_functions.local_sar_peak_loss(tf_g_sar_real, tf_gen_out),
              loss_functions.local_sar_max(tf_g_sar_real),
              loss_functions.local_sar_max(tf_gen_out),
              loss_functions.sar_tversky_focal_loss(tf_g_sar_real, tf_gen_out, thres=0.0, tv_alpha=0.3, tv_beta=0.7,
                                                    f_gamma=2.0, f_alpha=0.25, w_focal=0.5)]

    loss_weights_f = [adv_loss_weight, adv_loss_weight, lamda[0], lamda[1], lamda[2], lamda[3], lamda[4], 0, 0, lamda[5]]

    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_f, optimizer=optimizers_builtin.select(optimizer, initial_lr, clipvalue=1.0),
                  loss_weights=loss_weights_f)

    d_model.trainable = True

    return model


def build_cgan_feedback_lse_sar(gen_conf, train_conf, g_model, d_model, fb_model, local_sar_model, num_fb_loop,
                                g_encoder_ch):

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    dimension = train_conf['dimension']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    patch_shape = train_conf['patch_shape']
    # optimizer = train_conf['optimizer']
    # initial_lr = train_conf['initial_lr']
    batch_size = train_conf['batch_size']
    g_loss = train_conf['GAN']['generator']['loss']
    d_loss = train_conf['GAN']['discriminator']['loss']
    lamda = train_conf['GAN']['generator']['lamda']
    adv_loss_weight = train_conf['GAN']['generator']['adv_loss_weight']
    g_patch_shape = train_conf['GAN']['generator']['patch_shape']
    g_output_shape = train_conf['GAN']['generator']['output_shape']
    d_patch_shape = train_conf['GAN']['discriminator']['patch_shape']
    num_classes_g = train_conf['GAN']['generator']['num_classes']
    num_classes_d = train_conf['GAN']['discriminator']['num_classes']
    optimizer = train_conf['GAN']['generator']['optimizer']
    initial_lr = train_conf['GAN']['generator']['initial_lr']

    if dimension == 2:
        g_input_shape = (g_patch_shape[0], g_patch_shape[1], num_modality)
        f_output_shape = (np.divide(g_patch_shape[0], 8).astype(int), np.divide(g_patch_shape[1], 8).astype(int), g_encoder_ch)
    else:
        g_input_shape = (num_modality,) + g_patch_shape
        f_output_shape = (g_encoder_ch,) + np.divide(g_patch_shape, 8).astype(int)

    # make weights in the discriminator not trainable
    d_model.trainable = False

    # define the source image
    g_in_src = Input(shape=g_input_shape)
    output_shape_prod = (np.prod(g_output_shape), num_classes_g)
    g_sar_real_norm = Input(shape=output_shape_prod)  # normalized SAR
    g_sar_real = Input(shape=output_shape_prod) # non-normalized SAR
    fb_output = Input(shape=f_output_shape)

    # connect the source input and generator output to the discriminator input

    # [_, encoder_out] = g_model([g_in_src, fb_output])  # fb_output: zero
    # [gen_out, _] = g_model([g_in_src, encoder_out])

    [gen_out, encoder_out] = g_model([g_in_src, fb_output]) # fb_output: zero

    decoder_in=encoder_out
    for n in range(num_fb_loop):
        [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
        decoder_in = fb_model([gen_out, dis_out, encoder_out])
        [gen_out, encoder_out] = g_model([g_in_src, decoder_in])

    # [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
    # decoder_in = fb_model([gen_out, dis_out, encoder_out])
    # [gen_out, encoder_out] = g_model([g_in_src, decoder_in])
    #
    # [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
    # decoder_in = fb_model([gen_out, dis_out, encoder_out])
    # [gen_out, encoder_out] = g_model([g_in_src, decoder_in])

    # [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
    # decoder_in = fb_model([gen_out, dis_out, encoder_out])
    # [gen_out, encoder_out] = g_model([g_in_src, decoder_in])
    #
    # [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
    # decoder_in = fb_model([gen_out, dis_out, encoder_out])
    # [gen_out, encoder_out] = g_model([g_in_src, decoder_in])
    #
    # [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
    # decoder_in = fb_model([gen_out, dis_out, encoder_out])
    # [gen_out, _] = g_model([g_in_src, decoder_in])

    [dis_out, dis_e_out, interm_f1_out, interm_f2_out, interm_f3_out] = d_model([g_in_src, gen_out])

    [local_sar_min_output, local_sar_max_output, attn_map_low, attn_map_high] = local_sar_model([decoder_in])

    tf_g_sar_real_norm = Lambda(lambda x: x)(g_sar_real_norm)
    tf_g_sar_real = Lambda(lambda x: x)(g_sar_real)
    tf_gen_out = Lambda(lambda x: x)(gen_out)

    tf_local_sar_min_out = Lambda(lambda x: x)(local_sar_min_output)
    tf_local_sar_max_out = Lambda(lambda x: x)(local_sar_max_output)

    print(g_sar_real_norm.shape)
    print(tf_g_sar_real_norm.shape)
    print(tf_local_sar_min_out.shape)
    print(tf_local_sar_max_out.shape)

    model = Model([g_in_src, fb_output, g_sar_real_norm, g_sar_real], [dis_out, dis_e_out, gen_out, attn_map_low,
                                                                       attn_map_high, interm_f1_out, interm_f2_out,
                                                                       interm_f3_out, g_sar_real, g_sar_real, g_sar_real,
                                                                       g_sar_real, g_sar_real, g_sar_real, g_sar_real,
                                                                       g_sar_real, g_sar_real])
    model.summary()

    loss_f = [d_loss, d_loss,
              g_loss, 'mae', 'mae', 'mae', 'mae', 'mae',
              loss_functions.local_sar_min_loss(tf_g_sar_real, tf_local_sar_min_out),
              loss_functions.local_sar_max_loss(tf_g_sar_real, tf_local_sar_max_out),
              loss_functions.local_sar_min(tf_g_sar_real),
              loss_functions.local_sar_max(tf_g_sar_real),
              loss_functions.local_sar_est(tf_local_sar_min_out),
              loss_functions.local_sar_est(tf_local_sar_max_out),
              loss_functions.local_sar_max(tf_g_sar_real_norm),
              loss_functions.local_sar_max(tf_gen_out),
              loss_functions.sar_tversky_focal_loss(tf_g_sar_real_norm, tf_gen_out, thres=0.0, tv_alpha=0.3, tv_beta=0.7,
                                                    f_gamma=2.0, f_alpha=0.25, w_focal=0.5)]

    loss_weights_f = [adv_loss_weight, adv_loss_weight, lamda[0], lamda[1], lamda[2], lamda[3], lamda[4], lamda[5],
                      lamda[6], lamda[7], 0, 0, 0, 0, 0, 0, lamda[8]]

    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    # clipvalue: Gradients will be clipped when their absolute value exceeds this value
    model.compile(loss=loss_f, optimizer=optimizers_builtin.select(optimizer, initial_lr, clipvalue=1.0), #clipvalue=1.0, clipnorm=1.0
                  loss_weights=loss_weights_f)

    d_model.trainable = True

    return model


def build_cgan_fb_fsmt_lse_sar(gen_conf, train_conf, g_model, d_model, fb_model, fsmt_model, local_sar_model, num_fb_loop,
                                g_encoder_ch):

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    dimension = train_conf['dimension']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    patch_shape = train_conf['patch_shape']
    # optimizer = train_conf['optimizer']
    # initial_lr = train_conf['initial_lr']
    batch_size = train_conf['batch_size']
    g_loss = train_conf['GAN']['generator']['loss']
    d_loss = train_conf['GAN']['discriminator']['loss']
    lamda = train_conf['GAN']['generator']['lamda']
    adv_loss_weight = train_conf['GAN']['generator']['adv_loss_weight']
    g_patch_shape = train_conf['GAN']['generator']['patch_shape']
    g_output_shape = train_conf['GAN']['generator']['output_shape']
    d_patch_shape = train_conf['GAN']['discriminator']['patch_shape']
    num_classes_g = train_conf['GAN']['generator']['num_classes']
    num_classes_d = train_conf['GAN']['discriminator']['num_classes']
    optimizer = train_conf['GAN']['generator']['optimizer']
    initial_lr = train_conf['GAN']['generator']['initial_lr']
    num_model_samples = train_conf['num_model_samples']

    if dimension == 2:
        g_input_shape = (g_patch_shape[0], g_patch_shape[1], num_modality)
        f_output_shape = (np.divide(g_patch_shape[0], 8).astype(int), np.divide(g_patch_shape[1], 8).astype(int), g_encoder_ch)
    else:
        g_input_shape = (num_modality,) + g_patch_shape
        f_output_shape = (g_encoder_ch,) + np.divide(g_patch_shape, 8).astype(int)

    # make weights in the discriminator not trainable
    d_model.trainable = False

    # define the source image
    g_in_src = Input(shape=g_input_shape)
    output_shape_prod = (np.prod(g_output_shape), num_classes_g)
    g_sar_real_norm = Input(shape=output_shape_prod)  # normalized SAR
    g_sar_real = Input(shape=output_shape_prod) # non-normalized SAR
    fb_output = Input(shape=f_output_shape)

    model_samples_norm = []
    model_samples = []
    for i in range(num_model_samples):
        model_samples_norm.append(Input(shape=output_shape_prod))
        model_samples.append(Input(shape=output_shape_prod))

    # connect the source input and generator output to the discriminator input

    # [_, encoder_out] = g_model([g_in_src, fb_output])  # fb_output: zero
    # [gen_out, _] = g_model([g_in_src, encoder_out])

    [gen_out, encoder_out] = g_model([g_in_src, fb_output]) # fb_output: zero

    decoder_in=encoder_out
    if num_fb_loop >= 1:
        for n in range(num_fb_loop):
            [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
            decoder_in = fb_model([gen_out, dis_out, encoder_out])
            decoder_in = fsmt_model([g_in_src] + model_samples_norm + [decoder_in])
            [gen_out, encoder_out] = g_model([g_in_src, decoder_in])
    else:
        decoder_in = fsmt_model([g_in_src] + model_samples_norm + [decoder_in])
        [gen_out, _] = g_model([g_in_src, decoder_in])

    # [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
    # decoder_in = fb_model([gen_out, dis_out, encoder_out])
    # [gen_out, encoder_out] = g_model([g_in_src, decoder_in])
    #
    # [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
    # decoder_in = fb_model([gen_out, dis_out, encoder_out])
    # [gen_out, encoder_out] = g_model([g_in_src, decoder_in])

    # [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
    # decoder_in = fb_model([gen_out, dis_out, encoder_out])
    # [gen_out, encoder_out] = g_model([g_in_src, decoder_in])
    #
    # [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
    # decoder_in = fb_model([gen_out, dis_out, encoder_out])
    # [gen_out, encoder_out] = g_model([g_in_src, decoder_in])
    #
    # [dis_out, _, _, _, _] = d_model([g_in_src, gen_out])
    # decoder_in = fb_model([gen_out, dis_out, encoder_out])
    # [gen_out, _] = g_model([g_in_src, decoder_in])

    [dis_out, dis_e_out, interm_f1_out, interm_f2_out, interm_f3_out] = d_model([g_in_src, gen_out])

    #[local_sar_min_output, local_sar_max_output, attn_map_low, attn_map_high] = local_sar_model([decoder_in])
    [local_sar_min_output, local_sar_max_output, attn_map_low, attn_map_high] = local_sar_model(
        [g_in_src, decoder_in] + model_samples)

    tf_g_sar_real_norm = Lambda(lambda x: x)(g_sar_real_norm)
    tf_g_sar_real = Lambda(lambda x: x)(g_sar_real)
    tf_gen_out = Lambda(lambda x: x)(gen_out)

    tf_local_sar_min_out = Lambda(lambda x: x)(local_sar_min_output)
    tf_local_sar_max_out = Lambda(lambda x: x)(local_sar_max_output)

    print(g_sar_real_norm.shape)
    print(tf_g_sar_real_norm.shape)
    print(tf_local_sar_min_out.shape)
    print(tf_local_sar_max_out.shape)

    model = Model([g_in_src, fb_output, g_sar_real_norm, g_sar_real] + model_samples_norm + model_samples, [dis_out, dis_e_out, gen_out, attn_map_low,
                                                                       attn_map_high, interm_f1_out, interm_f2_out,
                                                                       interm_f3_out, g_sar_real, g_sar_real, g_sar_real,
                                                                       g_sar_real, g_sar_real, g_sar_real, g_sar_real,
                                                                       g_sar_real, g_sar_real])
    model.summary()

    loss_f = [d_loss, d_loss,
              g_loss, 'mae', 'mae', 'mae', 'mae', 'mae',
              loss_functions.local_sar_min_loss(tf_g_sar_real, tf_local_sar_min_out),
              loss_functions.local_sar_max_loss(tf_g_sar_real, tf_local_sar_max_out),
              loss_functions.local_sar_min(tf_g_sar_real),
              loss_functions.local_sar_max(tf_g_sar_real),
              loss_functions.local_sar_est(tf_local_sar_min_out),
              loss_functions.local_sar_est(tf_local_sar_max_out),
              loss_functions.local_sar_max(tf_g_sar_real_norm),
              loss_functions.local_sar_max(tf_gen_out),
              loss_functions.sar_tversky_focal_loss(tf_g_sar_real_norm, tf_gen_out, thres=0.0, tv_alpha=0.3, tv_beta=0.7,
                                                    f_gamma=2.0, f_alpha=0.25, w_focal=0.5)]

    loss_weights_f = [adv_loss_weight, adv_loss_weight, lamda[0], lamda[1], lamda[2], lamda[3], lamda[4], lamda[5],
                      lamda[6], lamda[7], 0, 0, 0, 0, 0, 0, lamda[8]]

    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    # clipvalue: Gradients will be clipped when their absolute value exceeds this value
    model.compile(loss=loss_f, optimizer=optimizers_builtin.select(optimizer, initial_lr, clipvalue=1.0), #clipvalue=1.0, clipnorm=1.0
                  loss_weights=loss_weights_f)

    d_model.trainable = True

    return model