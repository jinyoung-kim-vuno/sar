import numpy as np

from keras import backend as K
from keras.layers import Activation, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D
from keras.layers.core import Permute, Reshape
from keras.layers.merge import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.initializers import he_normal
from utils import loss_functions, metrics, optimizers_builtin

K.set_image_dim_ordering('th')


# Gu et al., TMI19, "CE-Net: Context Encoder Network for 2D Medical Image Segmentation"

def generate_cenet_model(gen_conf, train_conf) :
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

    model = __generate_cenet_model(dimension, num_classes, kernel_init, input_shape, output_shape, activation)

    model.summary()
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_functions.select(num_classes, loss_opt),
                  optimizer=optimizers_builtin.select(optimizer, initial_lr),
                  metrics=metrics.select(metric_opt))

    return model

def __generate_cenet_model(dimension, num_classes, kernel_init, input_shape, output_shape, activation):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    input = Input(shape=input_shape)
    kernel_size = (7, 7) if dimension == 2 else (7, 7, 7)
    if dimension == 2:
        conv0 = Conv2D(32, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(input)
    else :
        conv0 = Conv3D(32, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(input)

    conv1 = get_res_conv_core(dimension, conv0, 32, kernel_init)
    pool1 = get_max_pooling_layer(dimension, conv1)

    conv2 = get_res_conv_core(dimension, pool1, 64, kernel_init)
    pool2 = get_max_pooling_layer(dimension, conv2)

    # conv3 = get_res_conv_core(dimension, pool2, 128)
    # pool3 = get_max_pooling_layer(dimension, conv3)

    DAC_out = DAC_block(dimension, pool2, 128, kernel_init)
    RMP_out = RMP_block(dimension, DAC_out, kernel_init)

    up1 = get_deconv_layer(dimension, RMP_out, 64, kernel_init)
    concate31 = concatenate([conv2, up1], axis=concat_axis)

    up2 = get_deconv_layer(dimension, concate31, 32, kernel_init)
    concate22 = concatenate([conv1, up2], axis=concat_axis)
    #
    # up3 = get_deconv_layer(dimension, concate22, 32)
    # concate13 = concatenate([conv1, up3], axis=concat_axis)

    conv4 = get_res_conv_core(dimension, concate22, 32, kernel_init)

    pred = get_conv_fc(dimension, conv4, num_classes, kernel_init)
    pred = organise_output(pred, output_shape, activation)

    return Model(inputs=[input], outputs=[pred])

def get_res_conv_core(dimension, input, num_filters, kernel_init) :
    a = None
    b = None
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)

    if dimension == 2:
        a = Conv2D(num_filters, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(input)
        b = Conv2D(num_filters, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(a)
        input = Conv2D(num_filters, kernel_size=(1, 1), padding='same', kernel_initializer=kernel_init)(input)
    else:
        a = Conv3D(num_filters, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(input)
        b = Conv3D(num_filters, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(a)
        input = Conv3D(num_filters, kernel_size=(1, 1, 1), padding='same', kernel_initializer=kernel_init)(input)

    input = BatchNormalization(axis=1)(input)
    b = BatchNormalization(axis=1)(b)

    c = add([input, b])
    c = BatchNormalization(axis=1)(c)
    return Activation('relu')(c)


def get_max_pooling_layer(dimension, input) :
    pool_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return MaxPooling2D(pool_size=pool_size)(input)
    else :
        return MaxPooling3D(pool_size=pool_size)(input)


def DAC_block(dimension, input, num_filters, kernel_init):

    if dimension == 2:
        conv1 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(1, 1), padding='same',
                       kernel_initializer=kernel_init)(input)

        conv2 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(2, 2), padding='same',
                       kernel_initializer=kernel_init)(input)
        conv2 = Conv2D(num_filters, kernel_size=(1, 1), dilation_rate=(1, 1), padding='same',
                       kernel_initializer=kernel_init)(conv2)

        conv3 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(1, 1), padding='same',
                       kernel_initializer=kernel_init)(input)
        conv3 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(2, 2), padding='same',
                       kernel_initializer=kernel_init)(conv3)
        conv3 = Conv2D(num_filters, kernel_size=(1, 1), dilation_rate=(1, 1), padding='same',
                       kernel_initializer=kernel_init)(conv3)

        conv4 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(1, 1), padding='same',
                       kernel_initializer=kernel_init)(input)
        conv4 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(2, 2), padding='same',
                       kernel_initializer=kernel_init)(conv4)
        conv4 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(4, 4), padding='same',
                       kernel_initializer=kernel_init)(conv4)
        conv4 = Conv2D(num_filters, kernel_size=(1, 1), dilation_rate=(1, 1),
                       padding='same', kernel_initializer=kernel_init)(conv4)

    else :
        conv1 = Conv3D(num_filters, kernel_size=(3, 3, 3), dilation_rate=(1, 1, 1), padding='same',
                       kernel_initializer=kernel_init)(input)

        conv2 = Conv3D(num_filters, kernel_size=(3, 3, 3), dilation_rate=(2, 2, 2), padding='same',
                       kernel_initializer=kernel_init)(input)
        conv2 = Conv3D(num_filters, kernel_size=(1, 1, 1), dilation_rate=(1, 1, 1), padding='same',
                       kernel_initializer=kernel_init)(conv2)

        conv3 = Conv3D(num_filters, kernel_size=(3, 3, 3), dilation_rate=(1, 1, 1), padding='same',
                       kernel_initializer=kernel_init)(input)
        conv3 = Conv3D(num_filters, kernel_size=(3, 3, 3), dilation_rate=(2, 2, 2), padding='same',
                       kernel_initializer=kernel_init)(conv3)
        conv3 = Conv3D(num_filters, kernel_size=(1, 1, 1), dilation_rate=(1, 1, 1), padding='same',
                       kernel_initializer=kernel_init)(conv3)

        conv4 = Conv3D(num_filters, kernel_size=(3, 3, 3), dilation_rate=(1, 1, 1), padding='same',
                       kernel_initializer=kernel_init)(input)
        conv4 = Conv3D(num_filters, kernel_size=(3, 3, 3), dilation_rate=(2, 2, 2), padding='same',
                       kernel_initializer=kernel_init)(conv4)
        conv4 = Conv3D(num_filters, kernel_size=(3, 3, 3), dilation_rate=(4, 4, 4), padding='same',
                       kernel_initializer=kernel_init)(conv4)
        conv4 = Conv3D(num_filters, kernel_size=(1, 1, 1), dilation_rate=(1, 1, 1),
                       padding='same', kernel_initializer=kernel_init)(conv4)

    conv1 = BatchNormalization(axis=1)(conv1)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv4 = BatchNormalization(axis=1)(conv4)

    add_conv = add([conv1, conv2, conv3, conv4])
    add_conv = BatchNormalization(axis=1)(add_conv)

    return Activation('relu')(add_conv)


def RMP_block(dimension, input, kernel_init):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    if dimension == 2:
        p2 = MaxPooling2D(pool_size=(2, 2))(input)
        p4 = MaxPooling2D(pool_size=(4, 4))(input)
        p8 = MaxPooling2D(pool_size=(8, 8))(input)
        #p6 = MaxPooling2D(pool_size=(6, 6), strides=(2, 2))(input)

        p2_conv = Conv2D(1, kernel_size=(1, 1), padding='same', kernel_initializer=kernel_init)(p2)
        p2_up = UpSampling2D(size=(2, 2), interpolation='bilinear')(p2_conv)
        p4_conv = Conv2D(1, kernel_size=(1, 1), padding='same', kernel_initializer=kernel_init)(p4)
        p4_up = UpSampling2D(size=(4, 4), interpolation='bilinear')(p4_conv)
        p8_conv = Conv2D(1, kernel_size=(1, 1), padding='same', kernel_initializer=kernel_init)(p8)
        p8_up = UpSampling2D(size=(8, 8), interpolation='bilinear')(p8_conv)
        # p6_conv = Conv2D(1, kernel_size=(1, 1), padding='same', kernel_initializer=kernel_init)(p6)
        # p6_up = UpSampling2D()(p6_conv)
    else:
        p2 = MaxPooling3D(pool_size=(2, 2, 2))(input)
        p4 = MaxPooling3D(pool_size=(4, 4, 4))(input)
        p8 = MaxPooling3D(pool_size=(8, 8, 8))(input)
        #p6 = MaxPooling3D(pool_size=(6, 6, 6), strides=(2, 2, 2))(input)

        p2_conv = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer=kernel_init)(p2)
        p2_up = UpSampling3D(size=(2, 2, 2))(p2_conv)
        p4_conv = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer=kernel_init)(p4)
        p4_up = UpSampling3D(size=(4, 4, 4))(p4_conv)
        p8_conv = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer=kernel_init)(p8)
        p8_up = UpSampling3D(size=(8, 8, 8))(p8_conv)
        # p6_conv = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer=kernel_init)(p6)
        # p6_up = UpSampling3D(size=sub_sample_factor, interpolation='bilinear')(p6_conv)

    output = concatenate([p2_up, p4_up, p8_up, input], axis=concat_axis)

    return output


def get_deconv_layer(dimension, input, num_filters, kernel_init) :
    strides = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        conv1 = Conv2D(num_filters, kernel_size=(1, 1), padding='same', kernel_initializer=kernel_init)(input)
        conv1 = BatchNormalization(axis=1)(conv1)
        conv1 = Activation('relu')(conv1)

        deconv = Conv2DTranspose(num_filters, kernel_size=(3, 3), padding='same', strides=strides,
                                 kernel_initializer=kernel_init)(conv1)
        deconv = BatchNormalization(axis=1)(deconv)
        deconv = Activation('relu')(deconv)

        conv2 = Conv2D(num_filters, kernel_size=(1, 1), padding='same', kernel_initializer=kernel_init)(deconv)
        conv2 = BatchNormalization(axis=1)(conv2)
        conv2 = Activation('relu')(conv2)

    else :
        conv1 = Conv3D(num_filters, kernel_size=(1, 1, 1), padding='same', kernel_initializer=kernel_init)(input)
        conv1 = BatchNormalization(axis=1)(conv1)
        conv1 = Activation('relu')(conv1)

        deconv = Conv3DTranspose(num_filters, kernel_size=(3, 3, 3), padding='same', strides=strides,
                                 kernel_initializer=kernel_init)(conv1)
        deconv = BatchNormalization(axis=1)(deconv)
        deconv = Activation('relu')(deconv)

        conv2 = Conv3D(num_filters, kernel_size=(1, 1, 1), padding='same', kernel_initializer=kernel_init)(deconv)
        conv2 = BatchNormalization(axis=1)(conv2)
        conv2 = Activation('relu')(conv2)

    return conv2


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