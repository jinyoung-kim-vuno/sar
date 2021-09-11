import numpy as np

from keras import backend as K
from keras.layers import Activation, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D
from keras.layers.core import Permute, Reshape
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import multi_gpu_model
from utils import loss_functions, metrics, optimizers_builtin


def generate_cc_3d_fcn_model(gen_conf, train_conf):
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

    model = __generate_cc_3d_fcn_model(
        dimension, num_classes, input_shape, output_shape, activation)

    model.summary()
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_functions.select(num_classes, loss_opt),
                  optimizer=optimizers_builtin.select(optimizer, initial_lr),
                  metrics=metrics.select(metric_opt))

    return model


def __generate_cc_3d_fcn_model(dimension, num_classes, input_shape, output_shape, activation):
    input = Input(shape=input_shape)

    conv1 = get_conv_core(dimension, input, int(96), 3, 0)
    print(np.shape(conv1))
    pool1 = get_max_pooling_layer(dimension, conv1)
    print(np.shape(pool1))
    conv1_tr = get_conv_core(dimension, conv1, int(32), 1, 1)
    print(np.shape(conv1_tr))

    conv2 = get_conv_core(dimension, pool1, int(128), 2, 0)
    print(np.shape(conv2))
    pool2 = get_max_pooling_layer(dimension, conv2)
    print(np.shape(pool2))
    conv2_tr = get_conv_core(dimension, conv2, int(32), 1, 1)
    print(np.shape(conv2_tr))

    conv3 = get_conv_core(dimension, pool2, int(128), 1, 0)
    print(np.shape(conv3))
    pool3 = get_max_pooling_layer(dimension, conv3)
    print(np.shape(pool3))
    conv3_tr = get_conv_core(dimension, conv3, int(32), 1, 1)
    print (np.shape(conv3_tr))
    up4 = get_deconv_layer(dimension, pool3, int(32))
    print(np.shape(up4))
    up4 = concatenate([up4, conv3_tr], axis=1)

    up5 = get_deconv_layer(dimension, up4, int(32))
    up5 = concatenate([up5, conv2_tr], axis=1)
    conv5 = get_conv_core(dimension, up5, int(32), 1, 0)

    up6 = get_deconv_layer(dimension, conv5, int(32))
    up6 = concatenate([up6, conv1_tr], axis=1)
    conv6 = get_conv_core(dimension, up6, int(32), 1, 0)
    pred = get_conv_fc(dimension, conv6, num_classes)

    pred = organise_output(pred, output_shape, activation)

    return Model(inputs=[input], outputs=[pred])


def get_conv_core(dimension, input, num_filters, num_conv, is_trans_module):
    #x = None
    if is_trans_module == 0:
        kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    else:
        kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)
    print (kernel_size)
    x = input
    for i in range(num_conv):
        if dimension == 2 :
            # x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(input)
            # x = BatchNormalization(axis=1)(x)
            # x = Activation('relu')(x)
            x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(x)
            x = BatchNormalization(axis=1)(x)
            x = Activation('relu')(x)
        else :
            # x = Conv3D(num_filters, kernel_size=kernel_size, padding='same')(input)
            # x = BatchNormalization(axis=1)(x)
            # x = Activation('relu')(x)
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


def get_deconv_layer(dimension, input, num_filters):
    strides = (2, 2) if dimension == 2 else (2, 2, 2)
    #strides = (1, 1) if dimension == 2 else (1, 1, 1)
    kernel_size = (2, 2) if dimension == 2 else (2, 2, 2)
    #kernel_size = (4, 4) if dimension == 2 else (4, 4, 4)

    if dimension == 2:
        return Conv2DTranspose(num_filters, kernel_size=kernel_size, strides=strides)(input)
    else :
        return Conv3DTranspose(num_filters, kernel_size=kernel_size, strides=strides)(input)


def get_conv_fc(dimension, input, num_filters):
    fc = None
    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)

    if dimension == 2 :
        fc = Conv2D(num_filters, kernel_size=kernel_size)(input)
    else :
        fc = Conv3D(num_filters, kernel_size=kernel_size)(input)

    return Activation('relu')(fc)


def organise_output(input, output_shape, activation):
    pred = Reshape(output_shape)(input)
    pred = Permute((2, 1))(pred)
    return Activation(activation)(pred)