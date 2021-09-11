
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dropout, Reshape, Dense, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose, concatenate, BatchNormalization, Activation, \
    AveragePooling2D, AveragePooling3D, Concatenate, GlobalAveragePooling2D, GlobalAveragePooling3D, MaxPooling2D, MaxPooling3D, \
    Cropping2D, Cropping3D, Lambda, Average, Multiply, Add, Subtract
from keras.layers.core import Permute
from keras.initializers import he_normal
from keras.layers.advanced_activations import LeakyReLU
from .SpectralNormalizationKeras import ConvSN2D, ConvSN3D, ConvSN2DTranspose
from keras.utils import multi_gpu_model
from keras.constraints import Constraint
from .squeeze_excitation import spatial_and_channel_squeeze_excite_block2D, spatial_and_channel_squeeze_excite_block3D


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


class AdaIN(keras.layers.Layer):
    def __init__(self, alpha=1.0, **kwargs):
        self.alpha = alpha
        super(AdaIN, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        #assert input_shape[0] == input_shape[1]
        return input_shape[0]

    def call(self, x, **kwargs):
        assert isinstance(x, list)
        # Todo : args
        content_features, style_features = x[0], x[1]
        style_mean, style_variance = tf.nn.moments(tf.expand_dims(style_features, 2), [1,2], keep_dims=True)
        content_mean, content_variance = tf.nn.moments(content_features, [1,2], keep_dims=True)
        epsilon = 1e-5
        normalized_content_features = tf.nn.batch_normalization(content_features, content_mean,
                                                                content_variance, style_mean,
                                                                tf.sqrt(style_variance), epsilon)
        normalized_content_features = self.alpha * normalized_content_features + (1 - self.alpha) * content_features
        return normalized_content_features


def build_few_shot_model_transfer(gen_conf, train_conf, g_encoder_ch):

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

    padding = 'same'

    if dimension == 2:
        input_shape = [(patch_shape[0], patch_shape[1], num_modality),
                       (patch_shape[0], patch_shape[1], num_g_output),
                       (np.divide(patch_shape[0], 8).astype(int), np.divide(patch_shape[1], 8).astype(int),
                        g_encoder_ch)] # output of generator
        output_shape_prod = (np.prod(patch_shape), num_g_output)
        print(input_shape)
    else:
        input_shape = [(num_modality, ) + patch_shape,
                       (num_g_output, ) + patch_shape,
                       (g_encoder_ch, (np.divide(patch_shape[0], 8).astype(int),
                                                       np.divide(patch_shape[1], 8).astype(int)))] # output of generator
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

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    content_in = Input(shape=input_shape[0])
    content_out = __generate_res_model(dimension, content_in, g_encoder_ch * 2, False, kernel_init, padding, const, concat_axis)
    content_out = GlobalAveragePooling2D()(content_out)
    content_out = Reshape((1, content_out._keras_shape[concat_axis]))(content_out)
    content_fc = Dense(g_encoder_ch, activation='relu', kernel_initializer=kernel_init, use_bias=False)(content_out)

    model_in_total = []
    model_out_total = []
    for i in range(num_model_samples):
        model_in = Input(shape=output_shape_prod)
        model_in_total.append(model_in)
        model_reshape = Reshape(input_shape[1])(Permute((2, 1))(model_in))
        model_out = __generate_res_model(dimension, model_reshape, g_encoder_ch * 2, False, kernel_init, padding, const, concat_axis)
        model_out = GlobalAveragePooling2D()(model_out)
        model_out = Reshape((1, model_out._keras_shape[concat_axis]))(model_out)
        model_out_total.append(model_out)

    model_avg = Average()(model_out_total)
    model_fc = Dense(g_encoder_ch, activation='relu', kernel_initializer=kernel_init, use_bias=False)(model_avg)

    model_transfer_fc = Dense(g_encoder_ch, activation='relu', kernel_initializer=kernel_init,
                              use_bias=False)(Multiply()([content_fc, model_fc]))
    if K.image_data_format() == 'channels_first':
        model_transfer_fc = Permute((2, 1))(model_transfer_fc)

    print(model_transfer_fc._keras_shape)
    encoder_out = Input(shape=input_shape[2])
    print(encoder_out._keras_shape)

    adain_encoded = Activation('elu')(AdaIN(alpha=0.5)([encoder_out, model_transfer_fc]))

    fsmt_model = Model(inputs=[content_in] + model_in_total + [encoder_out],
                       outputs=adain_encoded, name='GAN_FSMT')
    fsmt_model.summary()

    return fsmt_model


def few_shot_sar_distribution_transfer(gen_conf, train_conf, content_in, model_samples, encoder_out, g_encoder_ch):

    dimension = train_conf['dimension']
    is_set_random_seed = train_conf['is_set_random_seed']
    random_seed_num = train_conf['random_seed_num']
    patch_shape = train_conf['GAN']['generator']['patch_shape']
    num_g_output = train_conf['GAN']['generator']['num_classes']

    if dimension == 2:
        K.set_image_data_format('channels_last')

    padding = 'same'

    if dimension == 2:
        input_shape = (patch_shape[0], patch_shape[1], num_g_output)
        print(input_shape)
    else:
        input_shape = (num_g_output, ) + patch_shape
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

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    content_out = __generate_res_model(dimension, content_in, g_encoder_ch * 2, False, kernel_init, padding, const, concat_axis)
    content_out = GlobalAveragePooling2D()(content_out)
    content_out = Reshape((1, content_out._keras_shape[concat_axis]))(content_out)
    content_fc = Dense(g_encoder_ch, activation='relu', kernel_initializer=kernel_init, use_bias=False)(content_out)

    model_out_total = []
    for model_in in model_samples:
        model_reshape = Reshape(input_shape)(Permute((2, 1))(model_in))
        model_out = __generate_res_model(dimension, model_reshape, g_encoder_ch * 2, False, kernel_init, padding, const, concat_axis)
        model_out = GlobalAveragePooling2D()(model_out)
        model_out = Reshape((1, model_out._keras_shape[concat_axis]))(model_out)
        model_out_total.append(model_out)

    model_avg = Average()(model_out_total)
    model_fc = Dense(g_encoder_ch, activation='relu', kernel_initializer=kernel_init, use_bias=False)(model_avg)

    model_transfer_fc = Dense(g_encoder_ch, activation='relu', kernel_initializer=kernel_init,
                              use_bias=False)(Multiply()([content_fc, model_fc]))
    if K.image_data_format() == 'channels_first':
        model_transfer_fc = Permute((2, 1))(model_transfer_fc)

    print(model_transfer_fc._keras_shape)
    print(encoder_out._keras_shape)

    adain_encoded = Activation('elu')(AdaIN(alpha=0.5)([encoder_out, model_transfer_fc]))

    return adain_encoded


def few_shot_sar_peak_transfer(gen_conf, train_conf, content_in, model_samples, encoder_out, g_encoder_ch):

    dimension = train_conf['dimension']
    is_set_random_seed = train_conf['is_set_random_seed']
    random_seed_num = train_conf['random_seed_num']
    patch_shape = train_conf['GAN']['generator']['patch_shape']
    num_g_output = train_conf['GAN']['generator']['num_classes']


    if dimension == 2:
        K.set_image_data_format('channels_last')

    padding = 'same'

    if dimension == 2:
        input_shape = (patch_shape[0], patch_shape[1], num_g_output) # output of generator
        print(input_shape)
    else:
        input_shape = (num_g_output, ) + patch_shape # output of generator
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

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    content_out = __generate_res_model(dimension, content_in, g_encoder_ch * 2, False, kernel_init, padding, const, concat_axis)
    content_out = GlobalAveragePooling2D()(content_out)
    content_out = Reshape((1, content_out._keras_shape[concat_axis]))(content_out)
    content_fc = Dense(g_encoder_ch, activation='relu', kernel_initializer=kernel_init, use_bias=False)(content_out)

    model_out_total = []
    for model_in in model_samples:
        model_reshape = Reshape(input_shape)(Permute((2, 1))(model_in))
        model_out = __generate_res_model(dimension, model_reshape, g_encoder_ch * 2, False, kernel_init, padding, const, concat_axis)
        model_out = GlobalAveragePooling2D()(model_out)
        model_out = Reshape((1, model_out._keras_shape[concat_axis]))(model_out)
        model_out_total.append(model_out)

    model_avg = Average()(model_out_total)
    model_fc = Dense(g_encoder_ch, activation='relu', kernel_initializer=kernel_init, use_bias=False)(model_avg)

    model_transfer_fc = Dense(g_encoder_ch, activation='relu', kernel_initializer=kernel_init,
                              use_bias=False)(Multiply()([content_fc, model_fc]))
    if K.image_data_format() == 'channels_first':
        model_transfer_fc = Permute((2, 1))(model_transfer_fc)

    # print(model_transfer_fc._keras_shape)
    # encoder_out = Input(shape=input_shape[2])
    # print(encoder_out._keras_shape)

    adain_encoded = Activation('elu')(AdaIN(alpha=0.5)([encoder_out, model_transfer_fc]))

    return adain_encoded


def __generate_res_model(dimension, input, num_filters, spectral_norm, kernel_init, padding, const, concat_axis):

    if const == 0:
        conv1 = get_res_conv_core(dimension, input, spectral_norm, int(num_filters/8), padding, kernel_init, concat_axis)
    else:
        conv1 = get_res_conv_core_clipconst(dimension, input, spectral_norm, int(num_filters/8), padding, kernel_init, const, concat_axis)
    pool1 = get_max_pooling_layer(dimension, conv1)

    if const == 0:
        conv2 = get_res_conv_core(dimension, pool1, spectral_norm, int(num_filters/4), padding, kernel_init, concat_axis)
    else:
        conv2 = get_res_conv_core_clipconst(dimension, pool1, spectral_norm, int(num_filters/4), padding, kernel_init, const, concat_axis)
    pool2 = get_max_pooling_layer(dimension, conv2)

    if const == 0:
        conv3 = get_res_conv_core(dimension, pool2, spectral_norm, int(num_filters/2), padding, kernel_init, concat_axis)
    else:
        conv3 = get_res_conv_core_clipconst(dimension, pool2, spectral_norm, int(num_filters/2), padding, kernel_init, const, concat_axis)
    pool3 = get_max_pooling_layer(dimension, conv3)

    # encoder output
    if const == 0:
        output = get_res_conv_core(dimension, pool3, spectral_norm, num_filters, padding, kernel_init, concat_axis)
    else:
        output = get_res_conv_core_clipconst(dimension, pool3, spectral_norm, num_filters, padding, kernel_init, const, concat_axis)

    return output


def get_res_conv_core(dimension, input, spectral_norm, num_filters, padding, kernel_init, concat_axis):

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