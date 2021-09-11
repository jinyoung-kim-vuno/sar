
import numpy as np
import tensorflow as tf
from keras import backend as K

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Reshape, Conv2D, Conv3D, Conv2DTranspose, \
    Conv3DTranspose, UpSampling2D, UpSampling3D, MaxPooling2D, MaxPooling3D, AveragePooling2D, AveragePooling3D, \
    GlobalMaxPooling2D, GlobalMaxPooling3D, GlobalAveragePooling2D, GlobalAveragePooling3D, concatenate, \
    BatchNormalization, Activation, Add, Multiply, Cropping2D, Cropping3D, Lambda, Concatenate, LeakyReLU
from keras.regularizers import l2
from keras.layers.core import Permute
from keras.engine.topology import get_source_inputs
from keras.initializers import he_normal
from .squeeze_excitation import spatial_and_channel_squeeze_excite_block2D, spatial_and_channel_squeeze_excite_block3D
from .few_shot_model_transfer import few_shot_sar_peak_transfer


def build_fs_local_sar_estimator(gen_conf, train_conf, g_encoder_ch):

    dimension = train_conf['dimension']
    is_set_random_seed = train_conf['is_set_random_seed']
    random_seed_num = train_conf['random_seed_num']
    dataset = train_conf['dataset']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    patch_shape = train_conf['GAN']['generator']['patch_shape']
    num_g_output = train_conf['GAN']['generator']['num_classes']
    num_model_samples = train_conf['num_model_samples']

    if dimension == 2:
        K.set_image_data_format('channels_last')

    if is_set_random_seed == 1:
        kernel_init = he_normal(seed=random_seed_num)
    else:
        kernel_init = he_normal()

    padding = 'same'
    weight_decay = 1E-4

    if dimension == 2:
        input_shape = [(patch_shape[0], patch_shape[1], num_modality),
                       (np.divide(patch_shape[0], 8).astype(int), np.divide(patch_shape[1], 8).astype(int),
                        g_encoder_ch)] # output of generator
        output_shape_prod = (np.prod(patch_shape), num_g_output)
        print(input_shape)
    else:
        input_shape = [(num_modality, ) + patch_shape,
                       (g_encoder_ch, ) + np.divide(patch_shape, 8).astype(int)] # output of generator
        output_shape_prod = (np.prod(patch_shape), num_g_output)
        print(input_shape)

    assert dimension in [2, 3]

    with K.name_scope('FS_Local_SAR_Estimator'):

        g_in_src = Input(shape=input_shape[0])
        decoder_input = Input(shape=input_shape[1])

        model_samples = []
        for i in range(num_model_samples):
            model_in = Input(shape=output_shape_prod)
            model_samples.append(model_in)

        attn_map_low, attn_map_high = attention(decoder_input, dimension, kernel_init, padding, weight_decay)

        # local SAR lower/upper estimator
        encoder_out = local_sar_estimator_encoder(decoder_input, dimension, kernel_init, padding)
        adain_encoded = few_shot_sar_peak_transfer(gen_conf, train_conf, g_in_src, model_samples, encoder_out,
                                                   int(g_encoder_ch/4))

        local_sar_min_output, local_sar_max_output = local_sar_estimator_decoder(adain_encoded, attn_map_low,
                                                                                 attn_map_high, dimension, kernel_init)

        model = Model(inputs=[g_in_src, decoder_input] + model_samples,
                      outputs=[local_sar_min_output, local_sar_max_output, attn_map_low, attn_map_high],
                      name='FS_Local_SAR_Estimator')
        model.summary()

    return model


def build_local_sar_estimator(gen_conf, train_conf, g_encoder_ch):

    dimension = train_conf['dimension']
    is_set_random_seed = train_conf['is_set_random_seed']
    random_seed_num = train_conf['random_seed_num']

    patch_shape = train_conf['GAN']['generator']['patch_shape']

    if dimension == 2:
        K.set_image_data_format('channels_last')

    if is_set_random_seed == 1:
        kernel_init = he_normal(seed=random_seed_num)
    else:
        kernel_init = he_normal()

    padding = 'same'
    input_tensor = None
    weight_decay = 1E-4

    if dimension == 2:
        input_shape = [(np.divide(patch_shape[0], 8).astype(int), np.divide(patch_shape[1], 8).astype(int),
                        g_encoder_ch)]

    else:
        input_shape = [(g_encoder_ch, ) + np.divide(patch_shape, 8).astype(int)]

    assert dimension in [2, 3]

    with K.name_scope('Local_SAR_Estimator'):

        if input_tensor is None:
            decoder_input = Input(shape=input_shape[0])
        else:
            if not K.is_keras_tensor(input_tensor):
                decoder_input = Input(tensor=input_tensor, shape=input_shape[0])
            else:
                decoder_input = input_tensor

        attn_map_low, attn_map_high = attention(decoder_input, dimension, kernel_init, padding, weight_decay)

        # local SAR lower/upper estimator
        local_sar_min_output, local_sar_max_output = local_sar_estimator(decoder_input, attn_map_low, attn_map_high, dimension, kernel_init,
                                               padding, weight_decay)

        model = Model(inputs=decoder_input, outputs=[local_sar_min_output, local_sar_max_output, attn_map_low, attn_map_high], name='Local_SAR_Estimator')
        model.summary()

    return model


def attention(input, dimension, kernel_init, padding, weight_decay):

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)
    num_filters = input._keras_shape[concat_axis]

    input = spatial_and_channel_squeeze_excite_block2D(input, kernel_init, arrange_type='two_way_sequential',
                                                   input_short_cut=True, final_conv=False)

    if dimension == 2:
        x = Conv2D(int(num_filters/2), kernel_size, kernel_initializer=kernel_init, padding=padding)(input)
        low = Conv2D(1, kernel_size, kernel_initializer=kernel_init, padding=padding)(x)
        high = Conv2D(1, kernel_size, kernel_initializer=kernel_init, padding=padding)(x)
    else:
        x = Conv3D(int(num_filters/2), kernel_size, kernel_initializer=kernel_init, padding=padding)(input)
        low = Conv3D(1, kernel_size, kernel_initializer=kernel_init, padding=padding)(x)
        high = Conv3D(1, kernel_size, kernel_initializer=kernel_init, padding=padding)(x)

    attn_map_low = Activation('softmax')(low)
    attn_map_high = Activation('softmax')(high)

    return attn_map_low, attn_map_high


def local_sar_estimator(input, attn_map_low, attn_map_high, dimension, kernel_init, padding, weight_decay):

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    num_filters = input._keras_shape[concat_axis]

    if dimension == 2:
        #output_shape = (1, 1, int(num_filters/4))
        x = Conv2D(int(num_filters/2), kernel_size, kernel_initializer=kernel_init, padding=padding)(input)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        #x = LeakyReLU(alpha=0.2)(x)
        x = Activation('elu')(x)
        x = Conv2D(int(num_filters/4), kernel_size, kernel_initializer=kernel_init, padding=padding)(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        #x = LeakyReLU(alpha=0.2)(x)
        x = Activation('elu')(x)

        low_input = GlobalAveragePooling2D()(Multiply()([attn_map_low, x]))
        high_input = GlobalAveragePooling2D()(Multiply()([attn_map_high, x]))

    else:
        #output_shape = (1, 1, 1, int(num_filters/4))
        x = Conv3D(int(num_filters/2), kernel_size, kernel_initializer=kernel_init, padding=padding)(input)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        #x = LeakyReLU(alpha=0.2)(x)
        x = Activation('elu')(x)
        x = Conv3D(int(num_filters/4), kernel_size, kernel_initializer=kernel_init, padding=padding)(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        #x = LeakyReLU(alpha=0.2)(x)
        x = Activation('elu')(x)

        low_input = GlobalAveragePooling3D()(Multiply()([attn_map_low, x]))
        high_input = GlobalAveragePooling3D()(Multiply()([attn_map_high, x]))

    print(x._keras_shape)

    #x = Reshape(output_shape)(x)
    l = Dense(32, activation='elu', kernel_initializer=kernel_init, use_bias=False)(low_input)
    l = Dense(32, activation='elu', kernel_initializer=kernel_init, use_bias=False)(l)
    #local_sar_output = Dense(2, activation='linear', kernel_initializer=kernel_init, use_bias=False)(x)
    local_sar_min_output = Dense(1, activation='linear', kernel_initializer=kernel_init, use_bias=False)(l)

    #y = Reshape(output_shape)(y)
    h = Dense(32, activation='elu', kernel_initializer=kernel_init, use_bias=False)(high_input)
    h = Dense(32, activation='elu', kernel_initializer=kernel_init, use_bias=False)(h)
    local_sar_max_output = Dense(1, activation='linear', kernel_initializer=kernel_init, use_bias=False)(h)

    print(local_sar_min_output._keras_shape)
    print(local_sar_max_output._keras_shape)

    # if dimension == 2:
    #     output_shape = (1, 1, int(num_filters/4))
    #     y = Conv2D(int(num_filters/2), kernel_size, kernel_initializer=kernel_init, padding=padding)(input)
    #     y = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(y)
    #     y = Activation('elu')(y)
    #     y = Conv2D(int(num_filters/4), kernel_size, kernel_initializer=kernel_init, padding=padding)(y)
    #     y = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(y)
    #     y = Activation('elu')(y)
    #     y = GlobalAveragePooling2D()(y)
    # else:
    #     output_shape = (1, 1, 1, int(num_filters/4))
    #     y = Conv3D(int(num_filters/2), kernel_size, kernel_initializer=kernel_init, padding=padding)(input)
    #     y = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(y)
    #     y = Activation('elu')(y)
    #     y = Conv3D(int(num_filters/4), kernel_size, kernel_initializer=kernel_init, padding=padding)(y)
    #     y = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(y)
    #     y = Activation('elu')(y)
    #     y = GlobalAveragePooling3D()(y)

    # y = Reshape(output_shape)(y)
    # y = Dense(32, activation='elu', kernel_initializer=kernel_init, use_bias=False)(y)
    # y = Dense(32, activation='elu', kernel_initializer=kernel_init, use_bias=False)(y)
    # local_sar_max_output = Dense(1, activation='linear', kernel_initializer=kernel_init, use_bias=False)(y)

    return local_sar_min_output, local_sar_max_output


def local_sar_estimator_encoder(input, dimension, kernel_init, padding):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    num_filters = input._keras_shape[concat_axis]

    if dimension == 2:
        # output_shape = (1, 1, int(num_filters/4))
        x = Conv2D(int(num_filters / 2), kernel_size, kernel_initializer=kernel_init, padding=padding)(input)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        # x = LeakyReLU(alpha=0.2)(x)
        x = Activation('elu')(x)
        x = Conv2D(int(num_filters / 4), kernel_size, kernel_initializer=kernel_init, padding=padding)(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        # x = LeakyReLU(alpha=0.2)(x)
        x = Activation('elu')(x)

    else:
        # output_shape = (1, 1, 1, int(num_filters/4))
        x = Conv3D(int(num_filters / 2), kernel_size, kernel_initializer=kernel_init, padding=padding)(input)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        # x = LeakyReLU(alpha=0.2)(x)
        x = Activation('elu')(x)
        x = Conv3D(int(num_filters / 4), kernel_size, kernel_initializer=kernel_init, padding=padding)(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        # x = LeakyReLU(alpha=0.2)(x)
        x = Activation('elu')(x)

    print(x._keras_shape)

    return x


def local_sar_estimator_decoder(input, attn_map_low, attn_map_high, dimension, kernel_init):

    if dimension == 2:
        low_input = GlobalAveragePooling2D()(Multiply()([attn_map_low, input]))
        high_input = GlobalAveragePooling2D()(Multiply()([attn_map_high, input]))

    else:
        low_input = GlobalAveragePooling3D()(Multiply()([attn_map_low, input]))
        high_input = GlobalAveragePooling3D()(Multiply()([attn_map_high, input]))

    print(input._keras_shape)

    # x = Reshape(output_shape)(x)
    l = Dense(32, activation='elu', kernel_initializer=kernel_init, use_bias=False)(low_input)
    l = Dense(32, activation='elu', kernel_initializer=kernel_init, use_bias=False)(l)
    # local_sar_output = Dense(2, activation='linear', kernel_initializer=kernel_init, use_bias=False)(x)
    local_sar_min_output = Dense(1, activation='linear', kernel_initializer=kernel_init, use_bias=False)(l)

    # y = Reshape(output_shape)(y)
    h = Dense(32, activation='elu', kernel_initializer=kernel_init, use_bias=False)(high_input)
    h = Dense(32, activation='elu', kernel_initializer=kernel_init, use_bias=False)(h)
    local_sar_max_output = Dense(1, activation='linear', kernel_initializer=kernel_init, use_bias=False)(h)

    print(local_sar_min_output._keras_shape)
    print(local_sar_max_output._keras_shape)

    return local_sar_min_output, local_sar_max_output