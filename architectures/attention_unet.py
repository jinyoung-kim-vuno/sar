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
from keras.initializers import he_normal
from utils import loss_functions, metrics, optimizers_builtin
from .attention_gates import grid_attention, get_gate_signal

K.set_image_dim_ordering('th')


def generate_attention_unet_model(gen_conf, train_conf) :
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
    is_set_random_seed = train_conf['is_set_random_seed']
    random_seed_num = train_conf['random_seed_num']
    exclusive_train = train_conf['exclusive_train']
    if exclusive_train == 1:
        num_classes -= 1

    input_shape = (num_modality, ) + patch_shape
    output_shape = (num_classes, np.prod(expected_output_shape))

    assert dimension in [2, 3]

    if is_set_random_seed == 1:
        kernel_init = he_normal(seed=random_seed_num)
    else:
        kernel_init = he_normal()

    model = __generate_attention_unet_model(
        dimension, num_classes, input_shape, output_shape, activation, kernel_init, downsize_factor=2)

    model.summary()
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_functions.select(num_classes, loss_opt),
                  optimizer=optimizers_builtin.select(optimizer, initial_lr),
                  metrics=metrics.select(metric_opt))

    return model

def __generate_attention_unet_model(
    dimension, num_classes, input_shape, output_shape, activation, kernel_init, downsize_factor=2):

    input = Input(shape=input_shape)

    conv1 = get_conv_core(dimension, input, int(64/downsize_factor), kernel_init)
    pool1 = get_max_pooling_layer(dimension, conv1)

    conv2 = get_conv_core(dimension, pool1, int(128/downsize_factor), kernel_init)
    pool2 = get_max_pooling_layer(dimension, conv2)

    conv3 = get_conv_core(dimension, pool2, int(256/downsize_factor), kernel_init)
    pool3 = get_max_pooling_layer(dimension, conv3)

    # conv4 = get_conv_core(dimension, pool3, int(512/downsize_factor), kernel_init)
    # pool4 = get_max_pooling_layer(dimension, conv4)

    center = get_conv_core(dimension, pool3, int(512/downsize_factor), kernel_init)
    gating1 = get_gate_signal(dimension, center, int(256/downsize_factor), kernel_init)
    # Attention Mechanism
    g_conv3, att4 = grid_attention(dimension, conv3, gating1, int(256/downsize_factor), int(256/downsize_factor),
                                   'concatenation', kernel_init)

    up5 = get_deconv_layer(dimension, center, int(256/downsize_factor), kernel_init)
    up5 = concatenate([up5, g_conv3], axis=1)

    conv5 = get_conv_core(dimension, up5, int(256/downsize_factor), kernel_init)
    gating2 = get_gate_signal(dimension, conv5, int(128/downsize_factor), kernel_init)

    g_conv2, att3 = grid_attention(dimension, conv2, gating2, int(128/downsize_factor), int(128/downsize_factor),
                                   'concatenation', kernel_init)

    up6 = get_deconv_layer(dimension, conv5, int(128/downsize_factor), kernel_init)
    up6 = concatenate([up6, g_conv2], axis=1)

    conv6 = get_conv_core(dimension, up6, int(128/downsize_factor), kernel_init)
    gating3 = get_gate_signal(dimension, conv6, int(64/downsize_factor), kernel_init)

    g_conv1, att2 = grid_attention(dimension, conv1, gating3, int(64/downsize_factor), int(64/downsize_factor),
                                   'concatenation', kernel_init)

    up7 = get_deconv_layer(dimension, conv6, int(64/downsize_factor), kernel_init)
    up7 = concatenate([up7, g_conv1], axis=1)

    conv7 = get_conv_core(dimension, up7, int(64/downsize_factor), kernel_init)

    # up8 = get_deconv_layer(dimension, conv7, int(64/downsize_factor))
    # up8 = concatenate([up8, conv1], axis=1)
    #
    # conv8 = get_conv_core(dimension, up8, int(64/downsize_factor))

    pred = get_conv_fc(dimension, conv7, num_classes, kernel_init)
    pred = organise_output(pred, output_shape, activation)

    print(input.shape)
    print(pred.shape)
    return Model(inputs=[input], outputs=[pred])


def get_conv_core(dimension, input, num_filters, kernel_init) :
    x = None
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)

    if dimension == 2 :
        x = Conv2D(num_filters, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(input)
        x = BatchNormalization(axis=1)(x)
        x = Activation('relu')(x)
        x = Conv2D(num_filters, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation('relu')(x)
    else :
        x = Conv3D(num_filters, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(input)
        x = BatchNormalization(axis=1)(x)
        x = Activation('relu')(x)
        x = Conv3D(num_filters, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation('relu')(x)

    return x


def get_max_pooling_layer(dimension, input) :
    pool_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return MaxPooling2D(pool_size=pool_size)(input)
    else :
        return MaxPooling3D(pool_size=pool_size)(input)


def get_deconv_layer(dimension, input, num_filters, kernel_init) :
    strides = (2, 2) if dimension == 2 else (2, 2, 2)
    kernel_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return Conv2DTranspose(num_filters, kernel_size=kernel_size, strides=strides, kernel_initializer=kernel_init)(input)
    else :
        return Conv3DTranspose(num_filters, kernel_size=kernel_size, strides=strides, kernel_initializer=kernel_init)(input)


def get_conv_fc(dimension, input, num_filters, kernel_init) :
    fc = None
    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)

    if dimension == 2 :
        fc = Conv2D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(input)
    else :
        fc = Conv3D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(input)

    return Activation('relu')(fc)


def organise_output(input, output_shape, activation) :
    pred = Reshape(output_shape)(input)
    pred = Permute((2, 1))(pred)
    return Activation(activation)(pred)