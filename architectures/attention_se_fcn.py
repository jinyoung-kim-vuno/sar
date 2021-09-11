import numpy as np
from keras import backend as K
from keras.layers import Activation, Input, Add, Multiply
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D
from keras.layers.core import Permute, Reshape, RepeatVector
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import multi_gpu_model
from utils import loss_functions, metrics, optimizers_builtin
from .attention_gates import grid_attention, get_gate_signal
from .squeeze_excitation import squeeze_excite_block2D,squeeze_excite_block3D

K.set_image_dim_ordering('th')


def generate_attention_se_fcn_model(gen_conf, train_conf) :
    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    activation = train_conf['activation']
    dimension = train_conf['dimension']
    num_classes = gen_conf['num_classes']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    expected_output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']

    loss_opt = train_conf['loss']
    metric_opt = train_conf['metric']
    optimizer = train_conf['optimizer']
    initial_lr = train_conf['initial_lr']

    input_shape = (num_modality, ) + patch_shape
    output_shape = (num_classes, np.prod(expected_output_shape))

    assert dimension in [2, 3]

    model = __generate_attention_se_fcn_model(
        dimension, num_classes, input_shape, output_shape, activation, downsize_factor=1)

    model.summary()
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_functions.select(num_classes, loss_opt),
                  optimizer=optimizers_builtin.select(optimizer, initial_lr),
                  metrics=metrics.select(metric_opt))

    return model

def __generate_attention_se_fcn_model(
    dimension, num_classes, input_shape, output_shape, activation, downsize_factor=1):

    input = Input(shape=input_shape)

    conv1 = get_conv_core(dimension, input, int(64/downsize_factor))
    pool1 = get_max_pooling_layer(dimension, conv1)

    conv2 = get_conv_core(dimension, pool1, int(128/downsize_factor))
    pool2 = get_max_pooling_layer(dimension, conv2)

    # Attention Mechanism
    center = get_conv_core(dimension, pool2, int(256/downsize_factor))
    gating1 = get_gate_signal(dimension, center, int(128/downsize_factor))

    g_conv1, att1 = grid_attention(dimension, conv2, gating1, int(128/downsize_factor), int(128/downsize_factor),
                                   'concatenation')

    up1 = get_deconv_layer(dimension, center, int(128/downsize_factor))
    up1 = concatenate([up1, g_conv1], axis=1)

    conv3 = get_conv_core(dimension, up1, int(128/downsize_factor))
    gating3 = get_gate_signal(dimension, conv3, int(64/downsize_factor))

    g_conv2, att2 = grid_attention(dimension, conv1, gating3, int(64/downsize_factor), int(64/downsize_factor),
                                   'concatenation')

    up2 = get_deconv_layer(dimension, conv3, int(64/downsize_factor))
    up2 = concatenate([up2, g_conv2], axis=1)

    conv4 = get_conv_core(dimension, up2, int(64/downsize_factor))

    pred = get_conv_fc(dimension, conv4, num_classes)
    pred = organise_output(pred, output_shape, activation)

    print(input.shape)
    print(pred.shape)
    return Model(inputs=[input], outputs=[pred])


def get_conv_core(dimension, input, num_filters) :
    x = None
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)

    if dimension == 2:
        x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(input)
        x = BatchNormalization(axis=1)(x)
        x = Activation('elu')(x)
        x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation('elu')(x)
        x = squeeze_excite_block2D(x)

    else :
        x = Conv3D(num_filters, kernel_size=kernel_size, padding='same')(input)
        x = BatchNormalization(axis=1)(x)
        x = Activation('elu')(x)
        x = Conv3D(num_filters, kernel_size=kernel_size, padding='same')(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation('elu')(x)
        x = squeeze_excite_block3D(x)

    return x


def get_max_pooling_layer(dimension, input) :
    pool_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return MaxPooling2D(pool_size=pool_size)(input)
    else :
        return MaxPooling3D(pool_size=pool_size)(input)


def get_deconv_layer(dimension, input, num_filters) :
    strides = (2, 2) if dimension == 2 else (2, 2, 2)
    kernel_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return Conv2DTranspose(num_filters, kernel_size=kernel_size, strides=strides)(input)
    else :
        return Conv3DTranspose(num_filters, kernel_size=kernel_size, strides=strides)(input)


def get_conv_fc(dimension, input, num_filters) :
    fc = None
    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)

    if dimension == 2 :
        fc = Conv2D(num_filters, kernel_size=kernel_size)(input)
    else :
        fc = Conv3D(num_filters, kernel_size=kernel_size)(input)

    return Activation('elu')(fc)


def organise_output(input, output_shape, activation) :
    pred = Reshape(output_shape)(input)
    pred = Permute((2, 1))(pred)
    return Activation(activation)(pred)