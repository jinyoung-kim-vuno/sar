
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Lambda, ReLU, Subtract, Reshape, Permute, Activation
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from utils import optimizers_builtin, loss_functions, metrics
import numpy as np
import keras.backend as K


# define the combined generator and discriminator model, for updating the generator
def build_cyclegan_sar(gen_conf, train_conf, g_model_1, d_model_1, g_model_2, d_model_2):

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    dimension = train_conf['dimension']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    num_classes = train_conf['GAN']['generator']['num_classes']
    # optimizer = train_conf['optimizer']
    # initial_lr = train_conf['initial_lr']
    g_loss = train_conf['GAN']['generator']['loss']
    d_loss = train_conf['GAN']['discriminator']['loss']
    lamda = train_conf['GAN']['generator']['lamda']
    adv_loss_weight = train_conf['GAN']['generator']['adv_loss_weight']
    g_patch_shape = train_conf['GAN']['generator']['patch_shape']
    g_output_shape = train_conf['GAN']['generator']['output_shape']
    optimizer = train_conf['GAN']['generator']['optimizer']
    initial_lr = train_conf['GAN']['generator']['initial_lr']

    if dimension == 2:
        g_input_shape_in_src = (g_patch_shape[0], g_patch_shape[1], num_modality)
        g_input_shape_in_sar = (g_patch_shape[0], g_patch_shape[1], num_classes)
    else:
        g_input_shape_in_src = (num_modality,) + g_patch_shape
        g_input_shape_in_sar = (1,) + g_patch_shape

    # ensure the model we're updating is trainable
    g_model_1.trainable = True
    # mark other generator model as not trainable
    g_model_2.trainable = True
    # mark discriminator as not trainable
    d_model_1.trainable = False
    # mark discriminator as not trainable
    d_model_2.trainable = False

    # forward cycle
    g_in_src = Input(shape=g_input_shape_in_src)
    gen1_out = g_model_1(g_in_src) # A (input) -> B (sar)
    [output_d1, output_d1_e, interm_d1_f1_out, interm_d1_f2_out, interm_d1_f3_out] = d_model_1([g_in_src, gen1_out])
    # identity element
    #output_id = g_model_1(g_in_sar) # B -> B   # REVISE g_model_1: 3ch -> 1ch => 1ch -> 1ch
    output_f = g_model_2(Reshape(g_input_shape_in_sar)(gen1_out)) # B -> A

    # backward cycle
    g_in_sar = Input(shape=g_input_shape_in_sar)
    gen2_out = g_model_2(g_in_sar) # B -> A
    [output_d2, output_d2_e, interm_d2_f1_out, interm_d2_f2_out, interm_d2_f3_out] = d_model_2([g_in_sar, gen2_out])
    output_b = g_model_1(Reshape(g_input_shape_in_src)(gen2_out)) # A -> B

    # define model graph
    output_shape_prod = (np.prod(g_output_shape), 1)
    g_out_sar_real = Input(shape=output_shape_prod)
    tf_g_sar_real = Lambda(lambda x: x)(g_out_sar_real)
    #tf_gen_out = Lambda(lambda x: x[:, :, 0])(gen1_out)
    tf_gen_out = Lambda(lambda x: x)(output_b)

    model = Model([g_in_src, g_in_sar, g_out_sar_real], [output_d1, output_d1_e, output_d2, output_d2_e, output_f, output_b,
                                                         interm_d1_f1_out, interm_d1_f2_out, interm_d1_f3_out,
                                                         interm_d2_f1_out, interm_d2_f2_out, interm_d2_f3_out, g_out_sar_real,
                                                         g_out_sar_real, g_out_sar_real, g_out_sar_real])

    model.summary()

    # compile model with weighting of least squares loss and L1 loss
    loss_f = [d_loss, d_loss, d_loss, d_loss, g_loss, g_loss, 'mae', 'mae', 'mae', 'mae', 'mae', 'mae',
              loss_functions.local_sar_peak_loss(tf_g_sar_real, tf_gen_out),
              loss_functions.local_sar_max(tf_g_sar_real),
              loss_functions.local_sar_max(tf_gen_out),
              loss_functions.sar_tversky_focal_loss(tf_g_sar_real, tf_gen_out, thres=0.0, tv_alpha=0.3, tv_beta=0.7,
                                                    f_gamma=2.0, f_alpha=0.25, w_focal=0.5)]  # adversarial loss via discriminator output + L1 loss + L2 perceptual loss via the direct image output
    #loss_weights_f = [adv_loss_weight, lamda, 0.5/3, 0.5/3, 0.5/3, 0.001, 0, 0, 0.001]
    loss_weights_f = [adv_loss_weight, adv_loss_weight, adv_loss_weight, adv_loss_weight, lamda[0], lamda[1], lamda[2], lamda[3], lamda[4], lamda[5],
                      lamda[6], lamda[7], lamda[8], 0, 0, lamda[9]]

    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_f, optimizer=optimizers_builtin.select(optimizer, initial_lr, clipvalue=1.0), loss_weights=loss_weights_f)

    d_model_1.trainable = True
    d_model_2.trainable = True

    return model


# define the combined generator and discriminator model, for updating the generator
def build_multi_cyclegan_sar_fwd(gen_conf, train_conf, g_model_1, d_model_1, g_model_2):

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    dimension = train_conf['dimension']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    num_classes = train_conf['GAN']['generator']['num_classes']
    # optimizer = train_conf['optimizer']
    # initial_lr = train_conf['initial_lr']
    g_loss = train_conf['GAN']['generator']['loss']
    d_loss = train_conf['GAN']['discriminator']['loss']
    lamda = train_conf['GAN']['generator']['lamda']
    adv_loss_weight = train_conf['GAN']['generator']['adv_loss_weight']
    g_patch_shape = train_conf['GAN']['generator']['patch_shape']
    g_output_shape = train_conf['GAN']['generator']['output_shape']
    optimizer = train_conf['GAN']['generator']['optimizer']
    initial_lr = train_conf['GAN']['generator']['initial_lr']

    if dimension == 2:
        g_input_shape_in_src = (g_patch_shape[0], g_patch_shape[1], num_modality)
        g_input_shape_in_sar = (g_patch_shape[0], g_patch_shape[1], num_classes)
    else:
        g_input_shape_in_src = (num_modality,) + g_patch_shape
        g_input_shape_in_sar = (num_classes,) + g_patch_shape

    # ensure the model we're updating is trainable
    g_model_1.trainable = True
    # mark discriminator as not trainable
    d_model_1.trainable = False
    # mark other generator model as not trainable
    g_model_2.trainable = False

    # discriminator element
    g_in_src = Input(shape=g_input_shape_in_src)
    gen1_out = g_model_1(g_in_src) # A (input) -> B (sar)
    [output_d, interm_f1_out, interm_f2_out, interm_f3_out] = d_model_1([g_in_src, gen1_out])
    # identity element
    g_in_sar = Input(shape=g_input_shape_in_sar)
    output_id = g_model_1(g_in_sar) # B -> B   # REVISE g_model_1: 3ch -> 1ch => 1ch -> 1ch
    # forward cycle
    output_f = g_model_2(Reshape(g_input_shape_in_sar)(gen1_out)) # B -> A
    # backward cycle
    gen2_out = g_model_2(g_in_sar) # B -> A
    output_b = g_model_1(Reshape(g_input_shape_in_src)(gen2_out)) # A -> B

    # define model graph
    output_shape_prod = (np.prod(g_output_shape), 2)
    #output_shape_prod = (np.prod(g_output_shape), 5)
    g_out_sar_real = Input(shape=output_shape_prod)

    tf_g_sar_real = Lambda(lambda x: x)(g_out_sar_real)
    #tf_gen_out = Lambda(lambda x: x[:, :, 0])(gen1_out)
    tf_gen_out = Lambda(lambda x: x)(output_b)

    model = Model([g_in_src, g_in_sar, g_out_sar_real], [output_d, output_id, output_f, output_b, interm_f1_out,
                                                     interm_f2_out, interm_f3_out, g_out_sar_real, g_out_sar_real,
                                                         g_out_sar_real, g_out_sar_real, g_out_sar_real])

    model.summary()

    # compile model with weighting of least squares loss and L1 loss
    loss_f = [d_loss, g_loss, g_loss, g_loss, 'mae', 'mae', 'mae',
              loss_functions.local_sar_peak_loss(tf_g_sar_real, tf_gen_out),
              loss_functions.local_sar_max(tf_g_sar_real),
              loss_functions.local_sar_max(tf_gen_out),
              loss_functions.local_sar_mean_loss(tf_g_sar_real, tf_gen_out),
              loss_functions.local_sar_neg_loss(tf_gen_out)]  # adversarial loss via discriminator output + L1 loss + L2 perceptual loss via the direct image output
    #loss_weights_f = [adv_loss_weight, lamda, 0.5/3, 0.5/3, 0.5/3, 0.001, 0, 0, 0.001]
    loss_weights_f = [adv_loss_weight, lamda[0], lamda[1], lamda[2], lamda[3], lamda[4], lamda[5], lamda[6], 0, 0, lamda[7], lamda[8]]
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_f, optimizer=optimizers_builtin.select(optimizer, initial_lr, clipvalue=1.0), loss_weights=loss_weights_f)

    d_model_1.trainable = True
    g_model_2.trainable = True

    return model


def build_multi_cyclegan_sar_bwd(gen_conf, train_conf, g_model_2, d_model_2, g_model_1):

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    dimension = train_conf['dimension']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    num_classes = train_conf['GAN']['generator']['num_classes']
    # optimizer = train_conf['optimizer']
    # initial_lr = train_conf['initial_lr']
    g_loss = train_conf['GAN']['generator']['loss']
    d_loss = train_conf['GAN']['discriminator']['loss']
    lamda = train_conf['GAN']['generator']['lamda']
    adv_loss_weight = train_conf['GAN']['generator']['adv_loss_weight']
    g_patch_shape = train_conf['GAN']['generator']['patch_shape']
    g_output_shape = train_conf['GAN']['generator']['output_shape']
    optimizer = train_conf['GAN']['generator']['optimizer']
    initial_lr = train_conf['GAN']['generator']['initial_lr']

    if dimension == 2:
        g_input_shape_in_sar = (g_patch_shape[0], g_patch_shape[1], num_classes)
        g_input_shape_in_src = (g_patch_shape[0], g_patch_shape[1], num_modality)
    else:
        g_input_shape_in_sar = (num_classes,) + g_patch_shape
        g_input_shape_in_src = (num_modality,) + g_patch_shape

    # ensure the model we're updating is trainable
    g_model_2.trainable = True
    # mark discriminator as not trainable
    d_model_2.trainable = False
    # mark other generator model as not trainable
    g_model_1.trainable = False

    # discriminator element
    g_in_sar = Input(shape=g_input_shape_in_sar)
    gen2_out = g_model_2(g_in_sar) # B (sar) -> A (input)
    [output_d, interm_f1_out, interm_f2_out, interm_f3_out] = d_model_2([g_in_sar, gen2_out])
    # identity element
    g_in_src = Input(shape=g_input_shape_in_src)
    output_id = g_model_2(g_in_src) # A -> A   # REVISE g_model_2: 1ch -> 3ch => 3ch -> 3ch
    # forward cycle
    output_f = g_model_1(Reshape(g_input_shape_in_src)(gen2_out)) # A -> B
    # backward cycle
    gen1_out = g_model_1(g_in_src) # A -> B
    output_b = g_model_2(Reshape(g_input_shape_in_sar)(gen1_out)) # B -> A

    # define model graph

    output_shape_prod = (np.prod(g_output_shape), 2)
    #output_shape_prod = (np.prod(g_output_shape), 5)
    g_out_sar_real = Input(shape=output_shape_prod)

    tf_g_sar_real = Lambda(lambda x: x)(g_out_sar_real)
    #tf_gen_out = Lambda(lambda x: x[:, :, 0])(gen1_out)
    tf_gen_out = Lambda(lambda x: x)(output_f)

    model = Model([g_in_sar, g_in_src, g_out_sar_real], [output_d, output_id, output_f, output_b, interm_f1_out, interm_f2_out,
                                         interm_f3_out, g_out_sar_real, g_out_sar_real, g_out_sar_real,
                                         g_out_sar_real, g_out_sar_real])

    model.summary()

    # compile model with weighting of least squares loss and L1 loss
    loss_f = [d_loss, g_loss, g_loss, g_loss, 'mae', 'mae', 'mae',
              loss_functions.local_sar_peak_loss(tf_g_sar_real, tf_gen_out),
              loss_functions.local_sar_max(tf_g_sar_real),
              loss_functions.local_sar_max(tf_gen_out),
              loss_functions.local_sar_mean_loss(tf_g_sar_real, tf_gen_out),
              loss_functions.local_sar_neg_loss(tf_gen_out)] # adversarial loss via discriminator output + L1 loss + L2 perceptual loss via the direct image output
    #loss_weights_f = [adv_loss_weight, lamda, 0.5/3, 0.5/3, 0.5/3, 0.001, 0, 0, 0.001]
    loss_weights_f = [adv_loss_weight, lamda[9], lamda[10], lamda[11], lamda[12], lamda[13], lamda[14], lamda[15], 0, 0, lamda[16], lamda[17]]
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_f, optimizer=optimizers_builtin.select(optimizer, initial_lr, clipvalue=1.0), loss_weights=loss_weights_f)

    d_model_2.trainable = True
    g_model_1.trainable = True

    return model


# define the combined generator and discriminator model, for updating the generator
def build_single_cyclegan_sar_fwd(gen_conf, train_conf, g_model_1, d_model_1, g_model_2):

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    dimension = train_conf['dimension']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    # optimizer = train_conf['optimizer']
    # initial_lr = train_conf['initial_lr']
    g_loss = train_conf['GAN']['generator']['loss']
    d_loss = train_conf['GAN']['discriminator']['loss']
    lamda = train_conf['GAN']['generator']['lamda']
    adv_loss_weight = train_conf['GAN']['generator']['adv_loss_weight']
    g_patch_shape = train_conf['GAN']['generator']['patch_shape']
    g_output_shape = train_conf['GAN']['generator']['output_shape']
    optimizer = train_conf['GAN']['generator']['optimizer']
    initial_lr = train_conf['GAN']['generator']['initial_lr']

    if dimension == 2:
        g_input_shape_in_src = (g_patch_shape[0], g_patch_shape[1], num_modality)
        g_input_shape_in_sar = (g_patch_shape[0], g_patch_shape[1], 1)
    else:
        g_input_shape_in_src = (num_modality,) + g_patch_shape
        g_input_shape_in_sar = (1,) + g_patch_shape

    # ensure the model we're updating is trainable
    g_model_1.trainable = True
    # mark discriminator as not trainable
    d_model_1.trainable = False
    # mark other generator model as not trainable
    g_model_2.trainable = False

    # discriminator element
    g_in_src = Input(shape=g_input_shape_in_src)
    gen1_out = g_model_1(g_in_src) # A (input) -> B (sar)
    [output_d, interm_f1_out, interm_f2_out, interm_f3_out] = d_model_1([g_in_src, gen1_out])
    # identity element
    g_in_sar = Input(shape=g_input_shape_in_sar)
    output_id = g_model_1(g_in_sar) # B -> B   # REVISE g_model_1: 3ch -> 1ch => 1ch -> 1ch
    # forward cycle
    output_f = g_model_2(Reshape(g_input_shape_in_sar)(gen1_out)) # B -> A
    # backward cycle
    gen2_out = g_model_2(g_in_sar) # B -> A
    output_b = g_model_1(Reshape(g_input_shape_in_src)(gen2_out)) # A -> B

    # define model graph
    output_shape_prod = (np.prod(g_output_shape), 1)
    g_out_sar_real = Input(shape=output_shape_prod)

    tf_g_sar_real = Lambda(lambda x: x)(g_out_sar_real)
    #tf_gen_out = Lambda(lambda x: x[:, :, 0])(gen1_out)
    tf_gen_out = Lambda(lambda x: x)(output_b)

    model = Model([g_in_src, g_in_sar, g_out_sar_real], [output_d, output_id, output_f, output_b, interm_f1_out,
                                                     interm_f2_out, interm_f3_out, g_out_sar_real, g_out_sar_real,
                                                         g_out_sar_real, g_out_sar_real])

    model.summary()

    # compile model with weighting of least squares loss and L1 loss
    loss_f = [d_loss, g_loss, g_loss, g_loss, 'mae', 'mae', 'mae',
              loss_functions.local_sar_peak_loss(tf_g_sar_real, tf_gen_out),
              loss_functions.local_sar_max(tf_g_sar_real),
              loss_functions.local_sar_max(tf_gen_out),
              loss_functions.local_sar_neg_loss(tf_gen_out)]  # adversarial loss via discriminator output + L1 loss + L2 perceptual loss via the direct image output
    #loss_weights_f = [adv_loss_weight, lamda, 0.5/3, 0.5/3, 0.5/3, 0.001, 0, 0, 0.001]
    loss_weights_f = [adv_loss_weight, lamda[0], lamda[1], lamda[2], lamda[3], lamda[4], lamda[5], lamda[6], 0, 0, lamda[7]]
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_f, optimizer=optimizers_builtin.select(optimizer, initial_lr, clipvalue=1.0), loss_weights=loss_weights_f)

    d_model_1.trainable = True
    g_model_2.trainable = True

    return model


def build_single_cyclegan_sar_bwd(gen_conf, train_conf, g_model_2, d_model_2, g_model_1):

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    dimension = train_conf['dimension']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    # optimizer = train_conf['optimizer']
    # initial_lr = train_conf['initial_lr']
    g_loss = train_conf['GAN']['generator']['loss']
    d_loss = train_conf['GAN']['discriminator']['loss']
    lamda = train_conf['GAN']['generator']['lamda']
    adv_loss_weight = train_conf['GAN']['generator']['adv_loss_weight']
    g_patch_shape = train_conf['GAN']['generator']['patch_shape']
    g_output_shape = train_conf['GAN']['generator']['output_shape']
    optimizer = train_conf['GAN']['generator']['optimizer']
    initial_lr = train_conf['GAN']['generator']['initial_lr']

    if dimension == 2:
        g_input_shape_in_sar = (g_patch_shape[0], g_patch_shape[1], 1)
        g_input_shape_in_src = (g_patch_shape[0], g_patch_shape[1], num_modality)
    else:
        g_input_shape_in_sar = (1,) + g_patch_shape
        g_input_shape_in_src = (num_modality,) + g_patch_shape

    # ensure the model we're updating is trainable
    g_model_2.trainable = True
    # mark discriminator as not trainable
    d_model_2.trainable = False
    # mark other generator model as not trainable
    g_model_1.trainable = False

    # discriminator element
    g_in_sar = Input(shape=g_input_shape_in_sar)
    gen2_out = g_model_2(g_in_sar) # B (sar) -> A (input)
    [output_d, interm_f1_out, interm_f2_out, interm_f3_out] = d_model_2([g_in_sar, gen2_out])
    # identity element
    g_in_src = Input(shape=g_input_shape_in_src)
    output_id = g_model_2(g_in_src) # A -> A   # REVISE g_model_2: 1ch -> 3ch => 3ch -> 3ch
    # forward cycle
    output_f = g_model_1(Reshape(g_input_shape_in_src)(gen2_out)) # A -> B
    # backward cycle
    gen1_out = g_model_1(g_in_src) # A -> B
    output_b = g_model_2(Reshape(g_input_shape_in_sar)(gen1_out)) # B -> A

    # define model graph
    output_shape_prod = (np.prod(g_output_shape), 1)
    g_out_sar_real = Input(shape=output_shape_prod)

    tf_g_sar_real = Lambda(lambda x: x)(g_out_sar_real)
    # tf_gen_out = Lambda(lambda x: x[:, :, 0])(gen1_out)
    tf_gen_out = Lambda(lambda x: x)(output_f)

    model = Model([g_in_sar, g_in_src, g_out_sar_real], [output_d, output_id, output_f, output_b, interm_f1_out,
                                                         interm_f2_out, interm_f3_out, g_out_sar_real, g_out_sar_real,
                                                         g_out_sar_real, g_out_sar_real])

    model.summary()

    # compile model with weighting of least squares loss and L1 loss
    loss_f = [d_loss, g_loss, g_loss, g_loss, 'mae', 'mae', 'mae',
              loss_functions.local_sar_peak_loss(tf_g_sar_real, tf_gen_out),
              loss_functions.local_sar_max(tf_g_sar_real),
              loss_functions.local_sar_max(tf_gen_out),
              loss_functions.local_sar_neg_loss(tf_gen_out)]  # adversarial loss via discriminator output + L1 loss + L2 perceptual loss via the direct image output
    #loss_weights_f = [adv_loss_weight, lamda, 0.5/3, 0.5/3, 0.5/3, 0.001, 0, 0, 0.001]
    loss_weights_f = [adv_loss_weight, lamda[8], lamda[9], lamda[10], lamda[11], lamda[12], lamda[13], lamda[14], 0, 0, lamda[15]]
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_f, optimizer=optimizers_builtin.select(optimizer, initial_lr, clipvalue=1.0), loss_weights=loss_weights_f)

    d_model_2.trainable = True
    g_model_1.trainable = True

    return model