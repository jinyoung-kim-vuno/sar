import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Activation, Input, Bidirectional, GRU, TimeDistributed, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D
from keras.layers.core import Permute, Reshape, RepeatVector
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import multi_gpu_model
from utils import loss_functions, metrics, optimizers_builtin
from .attention_gates_4d import grid_attention, get_gate_signal
import time

K.set_image_dim_ordering('th')

# Fully Convolutional Recurrent Net with Attention Gates (FC-RNA) and its application to joint 4d mri segmentation
use_attention = 1
downsize_factor = 32
#BC_GRU_batch_size = 32

def generate_fc_rna_model(gen_conf, train_conf):
    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    activation = train_conf['activation']
    dimension = train_conf['dimension']
    num_classes = gen_conf['num_classes']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality) # time_points
    expected_output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']

    loss_opt = train_conf['loss']
    metric_opt = train_conf['metric']
    optimizer = train_conf['optimizer']
    initial_lr = train_conf['initial_lr']
    exclusive_train = train_conf['exclusive_train']
    if exclusive_train == 1:
        num_classes -= 1

    #time_points = 3
    #input_shape = (num_modality, 1) + patch_shape
    #input_shape = (BC_GRU_batch_size, num_modality, ) + patch_shape
    input_shape = (num_modality,) + patch_shape
    # previous: modality was integrated into channel, current: that should be split
    output_shape = (num_modality, num_classes, np.prod(expected_output_shape))

    print(input_shape)
    print(output_shape)

    assert dimension in [2, 3]

    # how to process in parallel below?
    input, pred = __generate_fc_rna_model(
        dimension, num_classes, num_modality, input_shape, output_shape, activation, downsize_factor)

    model = Model(inputs=[input], outputs=[pred])

    model.summary()
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_functions.select(num_classes, loss_opt),
                  optimizer=optimizers_builtin.select(optimizer, initial_lr),
                  metrics=metrics.select(metric_opt))

    return model

def __generate_fc_rna_model(
    dimension, num_classes, num_modality, input_shape, output_shape, activation, downsize_factor):

    input = Input(shape=input_shape)
    #input = Input(batch_shape=input_shape)
    input_expand_dims = Lambda(lambda x: K.expand_dims(x, axis=2))(input)
    print ('input (expanded): ', input_expand_dims._keras_shape)

    conv1 = get_conv_core(dimension, input_expand_dims, int(64/downsize_factor))
    pool1 = get_max_pooling_layer(dimension, conv1)
    print('pool1 output: ', pool1._keras_shape)
    # Conv3d output shape: None (batch size) + (num_modality, channel) + patch_shape (32, 32, 32) => (None, num_modality, channels, height, width, axis)
    # LSTM/GRU input shape: (None, num_modality, channels x height x width x axis)

    pool1_features_size = np.prod((int(64 / downsize_factor),) + (pool1._keras_shape[3], pool1._keras_shape[4],
                                                                  pool1._keras_shape[5]))
    gru1_input_shape = (num_modality, pool1_features_size)
    pool1_reshape = Reshape(gru1_input_shape)(pool1)

    # this works with keras == 2.1.2, tensorflow == 1.2.1
    gru1_start_time = time.time()
    gru1 = Bidirectional(GRU(pool1_features_size, return_sequences=True, return_state=False, stateful=False))(pool1_reshape)
    gru1_elapsed_time = time.time() - gru1_start_time
    print('BC_GRU1 elapsed time: ', gru1_elapsed_time)
    print('BC_GRU1 output: ', gru1._keras_shape)
    # LSTM/GRU output shape: (None, num_modality, channels x height x width x axis)
    # Conv3d input shape: None (batch size) + (num_modality, channel) + patch_shape (32, 32, 32) => (None, num_modality, channels, height, width, axis)

    conv2_input_shape = (num_modality, int(64/downsize_factor) * 2, pool1._keras_shape[3], pool1._keras_shape[4],
                         pool1._keras_shape[5]) # bidirectional returns features_size * 2 with merge_mode='concat' option

    #(batch_size, timeSteps, channels)

    gru1_reshape = Reshape(conv2_input_shape)(gru1)

    conv2 = get_conv_core(dimension, gru1_reshape, int(128/downsize_factor))
    pool2 = get_max_pooling_layer(dimension, conv2)
    print('pool2 output: ', pool2._keras_shape)

    pool2_features_size = np.prod((int(128 / downsize_factor),) + (pool2._keras_shape[3], pool2._keras_shape[4], pool2._keras_shape[5]))
    gru2_input_shape = (num_modality, pool2_features_size)
    pool2_reshape = Reshape(gru2_input_shape)(pool2)

    gru2_start_time = time.time()
    gru2 = Bidirectional(GRU(pool2_features_size, return_sequences=True, return_state=False, stateful=False))(pool2_reshape)
    gru2_elapsed_time = time.time() - gru2_start_time
    print('BC_GRU2 elapsed time: ', gru2_elapsed_time)
    print('BC_GRU2 output: ', gru2._keras_shape)
    conv3_input_shape = (num_modality, int(128/downsize_factor) * 2, pool2._keras_shape[3], pool2._keras_shape[4],
                         pool2._keras_shape[5])
    gru2_reshape = Reshape(conv3_input_shape)(gru2)

    conv3 = get_conv_core(dimension, gru2_reshape, int(256/downsize_factor))
    pool3 = get_max_pooling_layer(dimension, conv3)
    print('pool3 output: ', pool3._keras_shape)

    pool3_features_size = np.prod((int(256 / downsize_factor),) + (pool3._keras_shape[3], pool3._keras_shape[4],
                                                                   pool3._keras_shape[5]))
    gru3_input_shape = (num_modality, pool3_features_size)
    pool3_reshape = Reshape(gru3_input_shape)(pool3)

    gru3_start_time = time.time()
    gru3 = Bidirectional(GRU(pool3_features_size, return_sequences=True, return_state=False, stateful=False))(pool3_reshape)
    gru3_elapsed_time = time.time() - gru3_start_time
    print('BC_GRU3 elapsed time: ', gru3_elapsed_time)
    print('BC_GRU3 output: ', gru3._keras_shape)
    center_input_shape = (num_modality, int(256/downsize_factor) * 2, pool3._keras_shape[3], pool3._keras_shape[4],
                          pool3._keras_shape[5])
    gru3_reshape = Reshape(center_input_shape)(gru3)

    # conv4 = get_conv_core(dimension, pool3, int(512/downsize_factor))
    # pool4 = get_max_pooling_layer(dimension, conv4)

    center = get_conv_core(dimension, gru3_reshape, int(512/downsize_factor))
    print('center out: ', center._keras_shape)

    if use_attention == 1: # Attention Mechanism
        gating1 = get_gate_signal(dimension, center, int(256/downsize_factor))
        print('conv3 out: ', conv3._keras_shape)
        print('gating1 out: ', gating1._keras_shape)
        g_conv3, att4 = grid_attention(dimension, conv3, gating1, int(256/downsize_factor), int(256/downsize_factor),
                                   'concatenation')

    up5 = get_deconv_layer(dimension, center, int(256/downsize_factor))
    print('up5 out: ', up5._keras_shape)

    if use_attention == 1:  # Attention Mechanism
        up5 = concatenate([up5, g_conv3], axis=2) # axis of channel was changed from 1 to 2 due to num_modality (time)
    else:
        up5 = concatenate([up5, conv3], axis=2)

    print('up5 out (concatenated): ', up5._keras_shape)

    conv5 = get_conv_core(dimension, up5, int(256/downsize_factor))
    print('conv5 out: ', conv5._keras_shape)

    if use_attention == 1:  # Attention Mechanism
        gating2 = get_gate_signal(dimension, conv5, int(128/downsize_factor))
        print('conv2 out: ', conv2._keras_shape)
        print('gating2 out: ', gating2._keras_shape)
        g_conv2, att3 = grid_attention(dimension, conv2, gating2, int(128/downsize_factor), int(128/downsize_factor),
                                   'concatenation')

    up6 = get_deconv_layer(dimension, conv5, int(128/downsize_factor))
    print('up6 out: ', up6._keras_shape)

    if use_attention == 1:  # Attention Mechanism
        up6 = concatenate([up6, g_conv2], axis=2)
    else:
        up6 = concatenate([up6, conv2], axis=2)

    print('up6 out (concatenated): ', up6._keras_shape)

    conv6 = get_conv_core(dimension, up6, int(128/downsize_factor))
    print('conv6 out: ', conv6._keras_shape)

    if use_attention == 1:  # Attention Mechanism
        gating3 = get_gate_signal(dimension, conv6, int(64/downsize_factor))
        print('conv1 out: ', conv1._keras_shape)
        print('gating3 out: ', gating3._keras_shape)
        g_conv1, att2 = grid_attention(dimension, conv1, gating3, int(64/downsize_factor), int(64/downsize_factor),
                                   'concatenation')

    up7 = get_deconv_layer(dimension, conv6, int(64/downsize_factor))
    print('up7 out: ', up7._keras_shape)

    if use_attention == 1:  # Attention Mechanism
        up7 = concatenate([up7, g_conv1], axis=2)
    else:
        up7 = concatenate([up7, conv1], axis=2)

    print('up7 out (concatenated): ', up7._keras_shape)

    conv7 = get_conv_core(dimension, up7, int(64/downsize_factor))
    print('conv7 out: ', conv7._keras_shape)

    # up8 = get_deconv_layer(dimension, conv7, int(64/downsize_factor))
    # up8 = concatenate([up8, conv1], axis=1)
    #
    # conv8 = get_conv_core(dimension, up8, int(64/downsize_factor))

    pred = get_conv_fc(dimension, conv7, num_classes)
    print ('conv_fc output:', pred._keras_shape)

    pred = organise_output_4d(pred, output_shape, activation)

    print(input.shape)
    print(pred.shape)
    return input, pred


def get_conv_core(dimension, input, num_filters):
    x = None
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)

    if dimension == 2:
        x = TimeDistributed(Conv2D(num_filters, kernel_size=kernel_size, padding='same'))(input)
        x = TimeDistributed(BatchNormalization(axis=2))(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(Conv2D(num_filters, kernel_size=kernel_size, padding='same'))(x)
        x = TimeDistributed(BatchNormalization(axis=2))(x)
        x = TimeDistributed(Activation('relu'))(x)
    else :
        x = TimeDistributed(Conv3D(num_filters, kernel_size=kernel_size, padding='same'))(input)
        x = TimeDistributed(BatchNormalization(axis=2))(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(Conv3D(num_filters, kernel_size=kernel_size, padding='same'))(x)
        x = TimeDistributed(BatchNormalization(axis=2))(x)
        x = TimeDistributed(Activation('relu'))(x)

    return x


def get_max_pooling_layer(dimension, input) :
    pool_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return TimeDistributed(MaxPooling2D(pool_size=pool_size))(input)
    else :
        return TimeDistributed(MaxPooling3D(pool_size=pool_size))(input)


def get_deconv_layer(dimension, input, num_filters) :
    strides = (2, 2) if dimension == 2 else (2, 2, 2)
    kernel_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return TimeDistributed(Conv2DTranspose(num_filters, kernel_size=kernel_size, strides=strides))(input)
    else :
        return TimeDistributed(Conv3DTranspose(num_filters, kernel_size=kernel_size, strides=strides))(input)


def get_conv_fc(dimension, input, num_filters) :
    fc = None
    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)

    if dimension == 2:
        fc = TimeDistributed(Conv2D(num_filters, kernel_size=kernel_size))(input)
    else:
        fc = TimeDistributed(Conv3D(num_filters, kernel_size=kernel_size))(input)

    return TimeDistributed(Activation('relu'))(fc)


def organise_output_4d(input, output_shape, activation) :
    pred = Reshape(output_shape)(input)
    print('conv_fc output (reshape):', pred._keras_shape)
    pred = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 3, 2]))(pred)
    print('conv_fc output (permuted):', pred)
    return TimeDistributed(Activation(activation))(pred)


# def organise_output(input, output_shape, activation) :
#     pred = Reshape(output_shape)(input)
#     pred = Permute((2, 1))(pred)
#     return TimeDistributed(Activation(activation))(pred)