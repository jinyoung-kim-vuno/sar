
import numpy as np
from keras import backend as K

from keras.models import Model
from keras.layers import Input, Dropout, Reshape, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose, concatenate, BatchNormalization, Activation, \
    AveragePooling2D, AveragePooling3D, Concatenate, MaxPooling2D, MaxPooling3D, Cropping2D, Cropping3D, Lambda, Multiply, Add
from keras.layers.core import Permute
from keras.initializers import he_normal
from keras.layers.advanced_activations import LeakyReLU
from .SpectralNormalizationKeras import ConvSN2D, ConvSN3D, ConvSN2DTranspose
from keras.utils import multi_gpu_model
from keras.constraints import Constraint
from .squeeze_excitation import spatial_and_channel_squeeze_excite_block2D, spatial_and_channel_squeeze_excite_block3D
from .few_shot_model_transfer import few_shot_sar_distribution_transfer


class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


def build_fs_feedback_net(gen_conf, train_conf, g_encoder_ch):

    dimension = train_conf['dimension']
    is_set_random_seed = train_conf['is_set_random_seed']
    random_seed_num = train_conf['random_seed_num']
    dataset = train_conf['dataset']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)

    patch_shape = train_conf['GAN']['discriminator']['patch_shape']

    num_g_output = train_conf['GAN']['generator']['num_classes']
    num_d_output = train_conf['GAN']['discriminator']['num_classes']

    spectral_norm = train_conf['GAN']['feedback']['spectral_norm']
    num_model_samples = train_conf['num_model_samples']

    if dimension == 2:
        K.set_image_data_format('channels_last')

    padding = 'same'

    if dimension == 2:
        input_shape = [(patch_shape[0], patch_shape[1], num_g_output),
                       (patch_shape[0], patch_shape[1], num_d_output),
                       (np.prod(patch_shape), num_g_output),
                       (np.prod(patch_shape), num_d_output),
                       (np.divide(patch_shape[0], 8).astype(int), np.divide(patch_shape[1], 8).astype(int),
                        g_encoder_ch),
                       (patch_shape[0], patch_shape[1], num_modality)] # output of generator
        output_shape_prod = (np.prod(patch_shape), num_g_output)
        print(input_shape)
    else:
        input_shape = [(num_g_output, ) + patch_shape,
                       (num_d_output, ) + patch_shape,
                       (num_g_output, np.prod(patch_shape)),
                       (num_d_output, np.prod(patch_shape)),
                       (g_encoder_ch, (np.divide(patch_shape[0], 8).astype(int),
                                                       np.divide(patch_shape[1], 8).astype(int))),
                       (num_modality, ) + patch_shape] # output of generator
        output_shape_prod = (np.prod(patch_shape), num_g_output)
        print(input_shape)


    if input_shape is None:
        raise ValueError('For fully convolutional models, input shape must be supplied.')

    assert dimension in [2, 3]

    if is_set_random_seed == 1:
        kernel_init = he_normal(seed=random_seed_num)
    else:
        kernel_init = he_normal()

    weight_clip = 0
    if weight_clip == 1:
        const = ClipConstraint(0.01) # weight clipping: default 0.01
    else:
        const = 0

    g_output = Input(shape=input_shape[2])
    d_output = Input(shape=input_shape[3])
    g_encoder_output = Input(shape=input_shape[4])

    g_output_reshape = Reshape(input_shape[0])(Permute((2, 1))(g_output))
    d_output_reshape = Reshape(input_shape[1])(Permute((2, 1))(d_output))

    g_in_src = Input(shape=input_shape[5])

    print(g_output_reshape)
    print(d_output_reshape)
    print(g_encoder_output.shape)

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    merged_input = concatenate([g_output_reshape, d_output_reshape], axis=concat_axis)

    #fb_output = __generate_fb_model(dimension, merged_input, num_filters, spectral_norm, kernel_init, padding, concat_axis)
    fb_output = __generate_fb_res_model(dimension, merged_input, g_encoder_ch, spectral_norm, kernel_init, padding, const,
                                        concat_axis)

    #ht = spade_v2(dimension, fb_output, g_encoder_output, g_encoder_ch, spectral_norm, kernel_init, padding, const)
    #ht = spade_v3(dimension, fb_output, g_encoder_output, g_encoder_ch, spectral_norm, kernel_init, padding, const)
    ht = spade_v4(dimension, fb_output, g_encoder_output, g_encoder_ch, spectral_norm, kernel_init, padding, const)

    #print(ht)
    #ht = LeakyReLU(alpha=0.2)(ht)
    #ht = Activation('relu')(ht)
    ht = Activation('elu')(ht)

    model_samples = []
    for i in range(num_model_samples):
        model_in = Input(shape=output_shape_prod)
        model_samples.append(model_in)

    adain_encoded = few_shot_sar_distribution_transfer(gen_conf, train_conf, g_in_src, model_samples, ht, g_encoder_ch)

    fb_model = Model(inputs=[g_in_src, g_output, d_output, g_encoder_output] + model_samples, outputs=adain_encoded,
                     name='GAN_FS_Feedback')

    fb_model.summary()

    return fb_model


def build_feedback_net(gen_conf, train_conf, g_encoder_ch):

    dimension = train_conf['dimension']
    is_set_random_seed = train_conf['is_set_random_seed']
    random_seed_num = train_conf['random_seed_num']
    patch_shape = train_conf['GAN']['discriminator']['patch_shape']

    num_g_output = train_conf['GAN']['generator']['num_classes']
    num_d_output = train_conf['GAN']['discriminator']['num_classes']

    spectral_norm = train_conf['GAN']['feedback']['spectral_norm']

    if dimension == 2:
        K.set_image_data_format('channels_last')

    padding = 'same'

    if dimension == 2:
        input_shape = [(patch_shape[0], patch_shape[1], num_g_output),
                       (patch_shape[0], patch_shape[1], num_d_output),
                       (np.prod(patch_shape), num_g_output),
                       (np.prod(patch_shape), num_d_output),
                       (np.divide(patch_shape[0], 8).astype(int), np.divide(patch_shape[1], 8).astype(int),
                        g_encoder_ch)] # output of generator
        print(input_shape)
    else:
        input_shape = [(num_g_output, ) + patch_shape, (num_d_output, ) + patch_shape,
                       (num_g_output, np.prod(patch_shape)), (num_d_output, np.prod(patch_shape)),
                       (g_encoder_ch, (np.divide(patch_shape[0], 8).astype(int),
                                                       np.divide(patch_shape[1], 8).astype(int)))] # output of generator
        print(input_shape)


    if input_shape is None:
        raise ValueError('For fully convolutional models, input shape must be supplied.')

    assert dimension in [2, 3]

    if is_set_random_seed == 1:
        kernel_init = he_normal(seed=random_seed_num)
    else:
        kernel_init = he_normal()

    weight_clip = 0
    if weight_clip == 1:
        const = ClipConstraint(0.01) # weight clipping: default 0.01
    else:
        const = 0

    g_output = Input(shape=input_shape[2])
    d_output = Input(shape=input_shape[3])
    g_encoder_output = Input(shape=input_shape[4])

    g_output_reshape = Reshape(input_shape[0])(Permute((2, 1))(g_output))
    d_output_reshape = Reshape(input_shape[1])(Permute((2, 1))(d_output))

    print(g_output_reshape)
    print(d_output_reshape)
    print(g_encoder_output.shape)

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    merged_input = concatenate([g_output_reshape, d_output_reshape], axis=concat_axis)

    #fb_output = __generate_fb_model(dimension, merged_input, num_filters, spectral_norm, kernel_init, padding, concat_axis)
    fb_output = __generate_fb_res_model(dimension, merged_input, g_encoder_ch, spectral_norm, kernel_init, padding, const,
                                        concat_axis)

    #ht = spade_v2(dimension, fb_output, g_encoder_output, g_encoder_ch, spectral_norm, kernel_init, padding, const)
    #ht = spade_v3(dimension, fb_output, g_encoder_output, g_encoder_ch, spectral_norm, kernel_init, padding, const)
    ht = spade_v4(dimension, fb_output, g_encoder_output, g_encoder_ch, spectral_norm, kernel_init, padding, const)

    #print(ht)
    #ht = LeakyReLU(alpha=0.2)(ht)
    #ht = Activation('relu')(ht)
    ht = Activation('elu')(ht)

    fb_model = Model(inputs=[g_output, d_output, g_encoder_output], outputs=ht, name='GAN_Feedback')

    fb_model.summary()

    return fb_model


def spade(dimension, fb_output, g_encoder_output, num_filters, spectral_norm, kernel_init, padding):

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    fb_output = BatchNormalization(axis=concat_axis)(fb_output)

    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    #num_filters = g_encoder_output._keras_shape[concat_axis]

    if dimension == 2:
        if spectral_norm:
            x = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(g_encoder_output)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)
            gamma = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            beta = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
        else:
            x = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(g_encoder_output)
            x = BatchNormalization(axis=concat_axis)(x)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)
            gamma = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            beta = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
    else:
        if spectral_norm:
            x = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(g_encoder_output)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)
            gamma = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            beta = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
        else:
            x = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(g_encoder_output)
            x = BatchNormalization(axis=concat_axis)(x)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)
            gamma = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            beta = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)

    fb_output = Add()([fb_output, Multiply()([fb_output, gamma]), beta]) #
    #fb_output = fb_output * (1 + gamma) + beta = fb_output + fb_output * gamma + beta

    return fb_output


def spade_v2(dimension, fb_output, g_encoder_out, num_filters, spectral_norm, kernel_init, padding, const):

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    #num_filters = g_encoder_out._keras_shape[concat_axis]

    g_encoder_out = BatchNormalization(axis=concat_axis)(g_encoder_out)

    x = Concatenate(axis=concat_axis)([fb_output, g_encoder_out])

    if dimension == 2:
        if spectral_norm:
            if const == 0:
                x = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                beta = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            else:
                x = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)
                beta = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)
        else:
            if const == 0:
                x = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                x = BatchNormalization(axis=concat_axis)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                beta = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            else:
                x = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                           kernel_constraint=const)(x)
                x = BatchNormalization(axis=concat_axis)(x)
                # x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                               kernel_constraint=const)(x)
                beta = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                              kernel_constraint=const)(x)
    else:
        if spectral_norm:
            if const == 0:
                x = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                beta = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)

            else:
                x = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)
                beta = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)
        else:
            if const == 0:
                x = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                x = BatchNormalization(axis=concat_axis)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                beta = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            else:
                x = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)
                x = BatchNormalization(axis=concat_axis)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)
                beta = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)

    g_encoder_out_update = Add()([g_encoder_out, Multiply()([g_encoder_out, gamma]), beta]) #
    #g_encoder_out = g_encoder_out * (1 + gamma) + beta = g_encoder_out + g_encoder_out * gamma + beta

    #return Activation('linear')(g_encoder_out_update)
    return g_encoder_out_update


def spade_v3(dimension, fb_output, g_encoder_out, num_filters, spectral_norm, kernel_init, padding, const):

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)

    g_encoder_out = BatchNormalization(axis=concat_axis)(g_encoder_out)

    glam_out = spatial_and_channel_squeeze_excite_block2D(fb_output, kernel_init, arrange_type='two_way_sequential',
                                                          input_short_cut=True, final_conv=False)
    if dimension == 2:
        if spectral_norm:
            if const == 0:
                x = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(fb_output)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                beta = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            else:
                x = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                              kernel_constraint=const)(fb_output)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                                 kernel_constraint=const)(x)
                beta = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                                kernel_constraint=const)(x)
        else:
            if const == 0:
                x = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(fb_output)
                x = BatchNormalization(axis=concat_axis)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                beta = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            else:
                x = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                           kernel_constraint=const)(fb_output)
                x = BatchNormalization(axis=concat_axis)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                               kernel_constraint=const)(x)
                beta = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                              kernel_constraint=const)(x)
    else:
        if spectral_norm:
            if const == 0:
                x = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(fb_output)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                beta = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)

            else:
                x = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                             kernel_constraint=const)(fb_output)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                                 kernel_constraint=const)(x)
                beta = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                                kernel_constraint=const)(x)
        else:
            if const == 0:
                x = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(fb_output)
                x = BatchNormalization(axis=concat_axis)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                beta = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            else:
                x = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                           kernel_constraint=const)(fb_output)
                x = BatchNormalization(axis=concat_axis)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                               kernel_constraint=const)(x)
                beta = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                              kernel_constraint=const)(x)

    g_encoder_by_glam = Multiply()([g_encoder_out, glam_out])
    g_encoder_out_update = Add()([g_encoder_out, Multiply()([g_encoder_by_glam, gamma]), beta]) #
    #g_encoder_out = g_encoder_out * (1 + gamma) + beta = g_encoder_out + g_encoder_out * gamma + beta

    return g_encoder_out_update


def spade_v4(dimension, fb_output, g_encoder_out, num_filters, spectral_norm, kernel_init, padding, const):

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    #num_filters = g_encoder_out._keras_shape[concat_axis]

    g_encoder_out = BatchNormalization(axis=concat_axis)(g_encoder_out)

    x = Concatenate(axis=concat_axis)([fb_output, g_encoder_out])

    if dimension == 2:
        if spectral_norm:
            if const == 0:
                x = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                beta = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            else:
                x = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)
                beta = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)
        else:
            if const == 0:
                x = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                x = BatchNormalization(axis=concat_axis)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                beta = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            else:
                x = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                           kernel_constraint=const)(x)
                x = BatchNormalization(axis=concat_axis)(x)
                # x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                               kernel_constraint=const)(x)
                beta = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                              kernel_constraint=const)(x)
    else:
        if spectral_norm:
            if const == 0:
                x = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                beta = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)

            else:
                x = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)
                beta = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)
        else:
            if const == 0:
                x = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                x = BatchNormalization(axis=concat_axis)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
                beta = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            else:
                x = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)
                x = BatchNormalization(axis=concat_axis)(x)
                #x = LeakyReLU(alpha=0.2)(x)
                x = Activation('elu')(x)
                gamma = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)
                beta = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init, kernel_constraint=const)(x)


    glam_out = spatial_and_channel_squeeze_excite_block2D(g_encoder_out, kernel_init, arrange_type='two_way_sequential',
                                                          input_short_cut=True, final_conv=False)

    g_encoder_out_update = Add()([g_encoder_out, Multiply()([glam_out, gamma]), beta]) #
    #g_encoder_out = g_encoder_out * (1 + gamma) + beta = g_encoder_out + g_encoder_out * gamma + beta

    #return Activation('linear')(g_encoder_out_update)
    return g_encoder_out_update


def __generate_fb_model(dimension, merged_input, num_filters, spectral_norm, kernel_init, padding, concat_axis):

    conv1 = get_conv_core(dimension, merged_input, spectral_norm, int(num_filters/8), padding, kernel_init, concat_axis)
    pool1 = get_max_pooling_layer(dimension, conv1)

    conv2 = get_conv_core(dimension, pool1, spectral_norm, int(num_filters/4), padding, kernel_init, concat_axis)
    pool2 = get_max_pooling_layer(dimension, conv2)

    conv3 = get_conv_core(dimension, pool2, spectral_norm, int(num_filters/2), padding, kernel_init, concat_axis)
    pool3 = get_max_pooling_layer(dimension, conv3)

    # encoder output
    output = get_conv_core(dimension, pool3, spectral_norm, num_filters, padding, kernel_init, concat_axis)

    return output


def __generate_fb_res_model(dimension, merged_input, num_filters, spectral_norm, kernel_init, padding, const, concat_axis):

    #conv1 = get_res_conv_core(dimension, merged_input, spectral_norm, int(num_filters/8), padding, kernel_init, concat_axis)
    if const == 0:
        conv1 = get_res_conv_core_v2(dimension, merged_input, spectral_norm, int(num_filters/8), padding, kernel_init, concat_axis)
    else:
        conv1 = get_res_conv_core_clipconst(dimension, merged_input, spectral_norm, int(num_filters/8), padding, kernel_init, const, concat_axis)
    pool1 = get_max_pooling_layer(dimension, conv1)

    #conv2 = get_res_conv_core(dimension, pool1, spectral_norm, int(num_filters/4), padding, kernel_init, concat_axis)
    if const == 0:
        conv2 = get_res_conv_core_v2(dimension, pool1, spectral_norm, int(num_filters/4), padding, kernel_init, concat_axis)
    else:
        conv2 = get_res_conv_core_clipconst(dimension, pool1, spectral_norm, int(num_filters/4), padding, kernel_init, const, concat_axis)
    pool2 = get_max_pooling_layer(dimension, conv2)

    #conv3 = get_res_conv_core(dimension, pool2, spectral_norm, int(num_filters/2), padding, kernel_init, concat_axis)
    if const == 0:
        conv3 = get_res_conv_core_v2(dimension, pool2, spectral_norm, int(num_filters/2), padding, kernel_init, concat_axis)
    else:
        conv3 = get_res_conv_core_clipconst(dimension, pool2, spectral_norm, int(num_filters/2), padding, kernel_init, const, concat_axis)
    pool3 = get_max_pooling_layer(dimension, conv3)

    # encoder output
    if const == 0:
        # output = get_res_conv_core(dimension, pool3, spectral_norm, num_filters, padding, kernel_init,
        #                           concat_axis)
        output = get_res_conv_core_v2(dimension, pool3, spectral_norm, num_filters, padding, kernel_init, concat_axis)
    else:
        output = get_res_conv_core_clipconst(dimension, pool3, spectral_norm, num_filters, padding, kernel_init, const, concat_axis)

    return output


def get_conv_core(dimension, input, spectral_norm, num_filters, padding, kernel_init, concat_axis):

    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    if dimension == 2 :
        if spectral_norm:
            x = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(input)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)
            x = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)
        else:
            x = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(input)
            x = BatchNormalization(axis=concat_axis)(x)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)
            x = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            x = BatchNormalization(axis=concat_axis)(x)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)
    else :
        if spectral_norm:
            x = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(input)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)
            x = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)

        else:
            x = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(input)
            x = BatchNormalization(axis=concat_axis)(x)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)
            x = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            x = BatchNormalization(axis=concat_axis)(x)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)

    return x


def get_res_conv_core(dimension, input, spectral_norm, num_filters, padding, kernel_init, concat_axis) :

    kernel_size_a = (3, 3) if dimension == 2 else (3, 3, 3)
    kernel_size_b = (1, 1) if dimension == 2 else (1, 1, 1)

    if dimension == 2:
        if spectral_norm:
            a = ConvSN2D(num_filters, kernel_size=kernel_size_a, padding=padding, kernel_initializer=kernel_init)(input)
            b = ConvSN2D(num_filters, kernel_size=kernel_size_b, padding=padding, kernel_initializer=kernel_init)(input)
        else:
            a = Conv2D(num_filters, kernel_size=kernel_size_a, padding=padding, kernel_initializer=kernel_init)(input)
            a = BatchNormalization(axis=concat_axis)(a)
            b = Conv2D(num_filters, kernel_size=kernel_size_b, padding=padding, kernel_initializer=kernel_init)(input)
            b = BatchNormalization(axis=concat_axis)(b)
    else :
        if spectral_norm:
            a = ConvSN3D(num_filters, kernel_size=kernel_size_a, padding=padding, kernel_initializer=kernel_init)(input)
            b = ConvSN3D(num_filters, kernel_size=kernel_size_b, padding=padding, kernel_initializer=kernel_init)(input)
        else:
            a = Conv3D(num_filters, kernel_size=kernel_size_a, padding=padding, kernel_initializer=kernel_init)(input)
            a = BatchNormalization(axis=concat_axis)(a)
            b = Conv3D(num_filters, kernel_size=kernel_size_b, padding=padding, kernel_initializer=kernel_init)(input)
            b = BatchNormalization(axis=concat_axis)(b)

    c = Add()([a, b])
    if not spectral_norm:
        c = BatchNormalization(axis=concat_axis)(c)
    #return LeakyReLU(alpha=0.2)(c)
    return Activation('elu')(c)


def get_res_conv_core_v2(dimension, input, spectral_norm, num_filters, padding, kernel_init, concat_axis):

    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    merge_input = input
    # check if the number of filters needs to be increase, assumes channels last format
    if dimension == 2:
        if spectral_norm:
            if input.shape[-1] != num_filters:
                merge_input = ConvSN2D(num_filters, (1, 1), padding=padding, activation='elu', kernel_initializer=kernel_init)(input)
            conv1 = ConvSN2D(num_filters, kernel_size, padding=padding, activation='elu', kernel_initializer=kernel_init)(input)
            conv2 = ConvSN2D(num_filters, kernel_size, padding=padding, activation='linear', kernel_initializer=kernel_init)(conv1)
        else:
            if input.shape[-1] != num_filters:
                merge_input = Conv2D(num_filters, (1,1), padding=padding, activation='elu', kernel_initializer=kernel_init)(input)
            conv1 = Conv2D(num_filters, kernel_size, padding=padding, activation='elu', kernel_initializer=kernel_init)(input)
            conv1 = BatchNormalization(axis=concat_axis)(conv1)
            conv2 = Conv2D(num_filters, kernel_size, padding=padding, activation='linear', kernel_initializer=kernel_init)(conv1)
            conv2 = BatchNormalization(axis=concat_axis)(conv2)
    else:
        if spectral_norm:
            if input.shape[-1] != num_filters:
                merge_input = ConvSN3D(num_filters, (1,1,1), padding=padding, activation='elu', kernel_initializer=kernel_init)(input)
            conv1 = ConvSN3D(num_filters, kernel_size, padding=padding, activation='elu', kernel_initializer=kernel_init)(input)
            conv2 = ConvSN3D(num_filters, kernel_size, padding=padding, activation='linear', kernel_initializer=kernel_init)(conv1)
        else:
            if input.shape[-1] != num_filters:
                merge_input = Conv3D(num_filters, (1,1,1), padding=padding, activation='elu', kernel_initializer=kernel_init)(input)
            conv1 = Conv3D(num_filters, kernel_size, padding=padding, activation='elu', kernel_initializer=kernel_init)(input)
            conv1 = BatchNormalization(axis=concat_axis)(conv1)
            conv2 = Conv3D(num_filters, kernel_size, padding=padding, activation='linear', kernel_initializer=kernel_init)(conv1)
            conv2 = BatchNormalization(axis=concat_axis)(conv2)

    # add filters, assumes filters/channels last
    layer_out = Add()([conv2, merge_input])
    # activation function
    #layer_out = LeakyReLU(alpha=0.2)(layer_out)
    layer_out = Activation('elu')(layer_out)
    return layer_out


def get_res_conv_core_clipconst(dimension, input, spectral_norm, num_filters, padding, kernel_init, const, concat_axis):

    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    merge_input = input
    # check if the number of filters needs to be increase, assumes channels last format
    if dimension == 2:
        if spectral_norm:
            if input.shape[-1] != num_filters:
                merge_input = ConvSN2D(num_filters, (1, 1), padding=padding, activation='elu', kernel_initializer=kernel_init, kernel_constraint=const)(input)
            conv1 = ConvSN2D(num_filters, kernel_size, padding=padding, activation='elu', kernel_initializer=kernel_init, kernel_constraint=const)(input)
            conv2 = ConvSN2D(num_filters, kernel_size, padding=padding, activation='linear', kernel_initializer=kernel_init, kernel_constraint=const)(conv1)
        else:
            if input.shape[-1] != num_filters:
                merge_input = Conv2D(num_filters, (1,1), padding=padding, activation='elu', kernel_initializer=kernel_init, kernel_constraint=const)(input)
            conv1 = Conv2D(num_filters, kernel_size, padding=padding, activation='elu', kernel_initializer=kernel_init, kernel_constraint=const)(input)
            conv1 = BatchNormalization(axis=concat_axis)(conv1)
            conv2 = Conv2D(num_filters, kernel_size, padding=padding, activation='linear', kernel_initializer=kernel_init, kernel_constraint=const)(conv1)
            conv2 = BatchNormalization(axis=concat_axis)(conv2)
    else:
        if spectral_norm:
            if input.shape[-1] != num_filters:
                merge_input = ConvSN3D(num_filters, (1,1,1), padding=padding, activation='elu', kernel_initializer=kernel_init, kernel_constraint=const)(input)
            conv1 = ConvSN3D(num_filters, kernel_size, padding=padding, activation='elu', kernel_initializer=kernel_init, kernel_constraint=const)(input)
            conv2 = ConvSN3D(num_filters, kernel_size, padding=padding, activation='linear', kernel_initializer=kernel_init, kernel_constraint=const)(conv1)
        else:
            if input.shape[-1] != num_filters:
                merge_input = Conv3D(num_filters, (1,1,1), padding=padding, activation='elu', kernel_initializer=kernel_init, kernel_constraint=const)(input)
            conv1 = Conv3D(num_filters, kernel_size, padding=padding, activation='elu', kernel_initializer=kernel_init, kernel_constraint=const)(input)
            conv1 = BatchNormalization(axis=concat_axis)(conv1)
            conv2 = Conv3D(num_filters, kernel_size, padding=padding, activation='linear', kernel_initializer=kernel_init, kernel_constraint=const)(conv1)
            conv2 = BatchNormalization(axis=concat_axis)(conv2)

    # add filters, assumes filters/channels last
    layer_out = Add()([conv2, merge_input])
    # activation function
    #layer_out = LeakyReLU(alpha=0.2)(layer_out)
    layer_out = Activation('elu')(layer_out)
    return layer_out


def get_max_pooling_layer(dimension, input) :
    pool_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return MaxPooling2D(pool_size=pool_size)(input)
    else :
        return MaxPooling3D(pool_size=pool_size)(input)