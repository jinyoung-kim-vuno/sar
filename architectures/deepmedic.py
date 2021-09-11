import numpy as np

from keras import backend as K
from keras.layers import Activation, Input, AveragePooling2D, AveragePooling3D
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Cropping2D
from keras.layers.convolutional import Conv3D, Conv3DTranspose, Cropping3D
from keras.layers.core import Permute, Reshape
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.initializers import he_normal
from utils import loss_functions, metrics, optimizers_builtin

K.set_image_dim_ordering('th')

# Kamnitsas et al., MedIA17, "Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation"

def generate_deepmedic_model(gen_conf, train_conf) :
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

    model = __generate_deepmedic_model(dimension, num_classes, kernel_init, input_shape, output_shape, activation)

    model.summary()
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_functions.select(num_classes, loss_opt),
                  optimizer=optimizers_builtin.select(optimizer, initial_lr),
                  metrics=metrics.select(metric_opt))

    return model


def __generate_deepmedic_model(dimension, num_classes, kernel_init, input_shape, output_shape, activation) :
    original_input = Input(shape=input_shape)

    normal_res_input = get_cropping_layer(dimension, original_input, crop_size=(8, 8))
    low_res_input = get_low_res_layer(dimension, original_input)

    normal_res = get_conv_core(dimension, normal_res_input, 30, kernel_init)
    normal_res = get_conv_core(dimension, normal_res, 40, kernel_init)
    normal_res = get_conv_core(dimension, normal_res, 40, kernel_init)
    normal_res = get_conv_core(dimension, normal_res, 50, kernel_init)

    low_res = get_conv_core(dimension, low_res_input, 30, kernel_init)
    low_res = get_conv_core(dimension, low_res, 40, kernel_init)
    low_res = get_conv_core(dimension, low_res, 40, kernel_init)
    low_res = get_conv_core(dimension, low_res, 50, kernel_init)
    low_res = get_deconv_layer(dimension, low_res, 50, kernel_init)

    concat = concatenate([normal_res, low_res], axis=1)

    fc = get_conv_fc(dimension, concat, 150, kernel_init)
    fc = get_conv_fc(dimension, fc, 150, kernel_init)

    pred = get_conv_fc(dimension, fc, num_classes, kernel_init)
    pred = organise_output(pred, output_shape, activation)

    return Model(inputs=[original_input], outputs=[pred])


def get_conv_core(dimension, input, num_filters, kernel_init) :
    x = None
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)

    if dimension == 2 :
        x = Conv2D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(input)
        x = PReLU()(x)
        x = Conv2D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(x)
        x = PReLU()(x)
    else :
        x = Conv3D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(input)
        x = PReLU()(x)
        x = Conv3D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(x)
        x = PReLU()(x)

    return x


def get_cropping_layer(dimension, input, crop_size=(6, 6)) :
    cropping_param = (crop_size, crop_size) if dimension == 2 else (crop_size, crop_size, crop_size)

    if dimension == 2 :
        return Cropping2D(cropping=cropping_param)(input)
    else :
        return Cropping3D(cropping=cropping_param)(input)


def get_low_res_layer(dimension, input) :
    if dimension == 2 :
        return AveragePooling2D()(input)
    else :
        return AveragePooling3D()(input)


def get_conv_fc(dimension, input, num_filters, kernel_init) :
    fc = None
    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)

    if dimension == 2 :
        fc = Conv2D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(input)
    else :
        fc = Conv3D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(input)

    return PReLU()(fc)

def get_deconv_layer(dimension, input, num_filters, kernel_init):
    kernel_size = (2, 2) if dimension == 2 else (2, 2, 2)
    strides = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return Conv2DTranspose(num_filters, kernel_size=kernel_size, strides=strides, kernel_initializer=kernel_init)(input)
    else :
        return Conv3DTranspose(num_filters, kernel_size=kernel_size, strides=strides, kernel_initializer=kernel_init)(input)


def organise_output(input, output_shape, activation) :
    pred = Reshape(output_shape)(input)
    pred = Permute((2, 1))(pred)
    return Activation(activation)(pred)