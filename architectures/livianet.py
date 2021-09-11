import numpy as np

from keras import backend as K
from keras.layers import Activation, Input
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, Conv3D, Cropping2D, Cropping3D
from keras.layers.core import Permute, Reshape
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.initializers import he_normal
from utils import loss_functions, metrics, optimizers_builtin

K.set_image_dim_ordering('th')

# Dolz et al., NeuroImage18, "3D fully convolutional networks for subcortical segmentation in MRI: A large-scale study"

def generate_livianet_model(gen_conf, train_conf) :
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

    model = __generate_livianet_model(dimension, num_classes, kernel_init, input_shape, output_shape, activation)

    model.summary()
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_functions.select(num_classes, loss_opt),
                  optimizer=optimizers_builtin.select(optimizer, initial_lr),
                  metrics=metrics.select(metric_opt))

    return model

def __generate_livianet_model(dimension, num_classes, kernel_init, input_shape, output_shape, activation) :
    init_input = Input(shape=input_shape)

    x = get_conv_core(dimension, init_input, 25, kernel_init)
    y = get_conv_core(dimension, x, 50, kernel_init)
    z = get_conv_core(dimension, y, 75, kernel_init)

    print(x.shape)
    print(y.shape)
    print(z.shape)

    x_crop = get_cropping_layer(dimension, x, crop_size=(6, 6))
    y_crop = get_cropping_layer(dimension, y, crop_size=(3, 3))

    print(x_crop.shape)
    print(y_crop.shape)

    concat = concatenate([x_crop, y_crop, z], axis=1)

    fc = get_conv_fc(dimension, concat, 400, kernel_init)
    fc = get_conv_fc(dimension, fc, 200, kernel_init)
    fc = get_conv_fc(dimension, fc, 150, kernel_init)

    pred = get_conv_fc(dimension, fc, num_classes, kernel_init)
    pred = organise_output(pred, output_shape, activation)

    return Model(inputs=[init_input], outputs=[pred])


def get_conv_core(dimension, input, num_filters, kernel_init) :
    x = None
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)

    if dimension == 2 :
        x = Conv2D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(input)
        x = PReLU()(x)
        x = Conv2D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(x)
        x = PReLU()(x)
        x = Conv2D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(x)
        x = PReLU()(x)
    else :
        x = Conv3D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(input)
        x = PReLU()(x)
        x = Conv3D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(x)
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


def get_conv_fc(dimension, input, num_filters, kernel_init) :
    fc = None
    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)

    if dimension == 2 :
        fc = Conv2D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(input)
    else :
        fc = Conv3D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(input)

    return PReLU()(fc)


def organise_output(input, output_shape, activation) :
    pred = Reshape(output_shape)(input)
    pred = Permute((2, 1))(pred)
    return Activation(activation)(pred)