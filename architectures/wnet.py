import numpy as np

from keras import backend as K
from keras.layers import Activation, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D
from keras.layers.core import Permute, Reshape
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import multi_gpu_model
from utils import loss_functions, metrics, optimizers_builtin

K.set_image_dim_ordering('th')


def generate_wunet_model(gen_conf, train_conf) :
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
    exclusive_train = train_conf['exclusive_train']
    if exclusive_train == 1:
        num_classes -= 1

    input_shape = (num_modality, ) + patch_shape
    output_shape = (num_classes, np.prod(expected_output_shape))

    assert dimension in [2, 3]

    model = __generate_wnet_model(
        dimension, num_classes, input_shape, output_shape, activation)

    model.summary()
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_functions.select(num_classes, loss_opt),
                  optimizer=optimizers_builtin.select(optimizer, initial_lr),
                  metrics=metrics.select(metric_opt))

    return model


def __generate_wnet_model(
    dimension, num_classes, input_shape, output_shape, activation):
    input = Input(shape=input_shape)

    # unet preprocessor (light unet)
    downsize_factor = 4
    pre_conv1 = get_conv_core_pre(dimension, input, int(64/downsize_factor), 2)
    pre_pool1 = get_max_pooling_layer(dimension, pre_conv1)

    pre_conv2 = get_conv_core_pre(dimension, pre_pool1, int(128/downsize_factor), 2)
    pre_pool2 = get_max_pooling_layer(dimension, pre_conv2)

    pre_conv3 = get_conv_core_pre(dimension, pre_pool2, int(256/downsize_factor), 2)
    pre_pool3 = get_max_pooling_layer(dimension, pre_conv3)

    pre_conv4 = get_conv_core_pre(dimension, pre_pool3, int(512/downsize_factor), 2)

    pre_up5 = get_deconv_layer(dimension, pre_conv4, int(256/downsize_factor))
    pre_up5 = concatenate([pre_up5, pre_conv3], axis=1)

    pre_conv5 = get_conv_core_pre(dimension, pre_up5, int(256/downsize_factor), 2)

    pre_up6 = get_deconv_layer(dimension, pre_conv5, int(128/downsize_factor))
    pre_up6 = concatenate([pre_up6, pre_conv2], axis=1)

    pre_conv6 = get_conv_core_pre(dimension, pre_up6, int(128/downsize_factor), 2)

    pre_up7 = get_deconv_layer(dimension, pre_conv6, int(64/downsize_factor))
    pre_up7 = concatenate([pre_up7, pre_conv1], axis=1)

    pre_conv7 = get_conv_core_pre(dimension, pre_up7, int(64/downsize_factor), 2)

    preprocess_out = get_conv_core_pre(dimension, pre_conv7, 1, 1)

    #main_in = concatenate([preprocess_out, input], axis=1)
    #main_in = add([preprocess_out, input])

    # unet main

    downsize_factor = 2
    conv1 = get_conv_core(dimension, preprocess_out, int(64 / downsize_factor), 2)
    #conv1 = get_conv_core(dimension, main_in, int(64/downsize_factor), 2)
    pool1 = get_max_pooling_layer(dimension, conv1)

    conv2 = get_conv_core(dimension, pool1, int(128/downsize_factor), 2)
    pool2 = get_max_pooling_layer(dimension, conv2)

    conv3 = get_conv_core(dimension, pool2, int(256/downsize_factor), 2)
    pool3 = get_max_pooling_layer(dimension, conv3)

    conv4 = get_conv_core(dimension, pool3, int(512/downsize_factor), 2)

    up5 = get_deconv_layer(dimension, conv4, int(256/downsize_factor))
    up5 = concatenate([up5, conv3], axis=1)

    conv5 = get_conv_core(dimension, up5, int(256/downsize_factor), 2)

    up6 = get_deconv_layer(dimension, conv5, int(128/downsize_factor))
    up6 = concatenate([up6, conv2], axis=1)

    conv6 = get_conv_core(dimension, up6, int(128/downsize_factor), 2)

    up7 = get_deconv_layer(dimension, conv6, int(64/downsize_factor))
    up7 = concatenate([up7, conv1], axis=1)

    conv7 = get_conv_core(dimension, up7, int(64/downsize_factor), 2)

    pred = get_conv_fc(dimension, conv7, num_classes)
    pred = organise_output(pred, output_shape, activation)

    return Model(inputs=[input], outputs=[pred])


def get_conv_core_pre(dimension, input, num_filters, num_conv):
    #x = None
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    x = input
    for i in range(num_conv):
        if dimension == 2 :
            x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(x)
            #x = BatchNormalization(axis=1)(x)
            x = Activation('relu')(x)
        else :
            x = Conv3D(num_filters, kernel_size=kernel_size, padding='same')(x)
            #x = BatchNormalization(axis=1)(x)
            x = Activation('relu')(x)
    return x


def get_conv_core(dimension, input, num_filters, num_conv):
    #x = None
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    x = input
    for i in range(num_conv):
        if dimension == 2 :
            x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(x)
            x = BatchNormalization(axis=1)(x)
            x = Activation('relu')(x)
        else :
            x = Conv3D(num_filters, kernel_size=kernel_size, padding='same')(x)
            x = BatchNormalization(axis=1)(x)
            x = Activation('relu')(x)
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

    return Activation('relu')(fc)

def organise_output(input, output_shape, activation) :
    pred = Reshape(output_shape)(input)
    pred = Permute((2, 1))(pred)
    return Activation(activation)(pred)