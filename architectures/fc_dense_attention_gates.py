from keras import backend as K
from keras.layers import Activation, Add, Multiply, UpSampling2D, UpSampling3D, Cropping2D, Cropping3D, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np

K.set_image_dim_ordering('th')


def grid_attention(dimension, input, g_f, gate, attn_opt, mode):

    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)
    sub_sample_factor = (2, 2) if dimension == 2 else (2, 2, 2)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    gate_channels = gate.shape[channel_axis]
    #input_channels = input.shape[channel_axis]
    #g_f_channels = g_f.shape[channel_axis]
    g_f_crop_size = np.int((g_f.shape[2] * sub_sample_factor[0] - g_f._keras_shape[2]) / 2)

    # Define the operation
    if mode == 'concatenation':
        return _concatenation(dimension, input, g_f, gate, gate_channels, g_f_crop_size,
                              kernel_size, sub_sample_factor, 'relu', attn_opt)
    # elif mode == 'concatenation_debug':
    #     return _concatenation(dimension, input, g_f, in_channels, out_channels, kernel_size, sub_sample_factor,
    #                           sub_sample_kernel_size, 'softplus')
    # elif mode == 'concatenation_residual':
    #     return _concatenation_residual(dimension, input, g_f, in_channels, out_channels, kernel_size,
    #                                    sub_sample_factor, sub_sample_kernel_size, 'relu')
    else:
        raise NotImplementedError('Unknown operation function.')


def _concatenation(dimension, input, g_f, gate, gate_channels, g_f_crop_size, kernel_size, sub_sample_factor,
                   activation, attn_opt):

    if dimension == 2:
        if attn_opt == 1:
            # Theta^T * x_ij + Phi^T * gating_signal + bias
            theta_x = Conv2D(gate_channels, kernel_size=kernel_size, strides=kernel_size, padding='same')(input)
            phi_g_f = Conv2D(gate_channels, kernel_size=kernel_size, strides=kernel_size, padding='same')(g_f)
            phi_g_f_up = UpSampling2D(size=sub_sample_factor, interpolation='bilinear')(phi_g_f)
            phi_g_f_up_crop = get_cropping_layer(dimension, phi_g_f_up, crop_size=(g_f_crop_size, g_f_crop_size))

            f = Activation(activation)(Add()([theta_x, phi_g_f_up_crop]))

        elif attn_opt == 2:
            theta_x = Conv2D(gate_channels, kernel_size=kernel_size, strides=kernel_size, padding='same')(input)
            phi_g_f = Conv2D(gate_channels, kernel_size=kernel_size, strides=kernel_size, padding='same')(g_f)
            phi_gate = Conv2D(gate_channels, kernel_size=kernel_size, strides=kernel_size, padding='same')(gate)
            phi_g_f_up = UpSampling2D(size=sub_sample_factor, interpolation='bilinear')(phi_g_f)
            phi_g_f_up_crop = get_cropping_layer(dimension, phi_g_f_up, crop_size=(g_f_crop_size, g_f_crop_size))
            phi_gate_up = UpSampling2D(size=sub_sample_factor, interpolation='bilinear')(phi_gate)

            f = Activation(activation)(Add()([theta_x, phi_g_f_up_crop, phi_gate_up]))

        psi_f = Conv2D(1, kernel_size=kernel_size, strides=kernel_size, padding='same')(f)
        sigm_psi_f = Activation('sigmoid')(psi_f)

        #y = Multiply()([sigm_psi_f, input])
        y = Multiply()([sigm_psi_f, theta_x])

    elif dimension == 3:
        if attn_opt == 1:
            # Theta^T * x_ij + Phi^T * gating_signal + bias
            theta_x = Conv3D(gate_channels, kernel_size=kernel_size, strides=kernel_size, padding='same')(input)
            phi_g_f = Conv3D(gate_channels, kernel_size=kernel_size, strides=kernel_size, padding='same')(g_f)
            phi_g_f_up = UpSampling3D(size=sub_sample_factor)(phi_g_f)
            phi_g_f_up_crop = get_cropping_layer(dimension, phi_g_f_up, crop_size=(g_f_crop_size, g_f_crop_size))

            f = Activation(activation)(Add()([theta_x, phi_g_f_up_crop]))

        elif attn_opt == 2:
            theta_x = Conv3D(gate_channels, kernel_size=kernel_size, strides=kernel_size, padding='same')(input)
            phi_g_f = Conv3D(gate_channels, kernel_size=kernel_size, strides=kernel_size, padding='same')(g_f)
            phi_gate = Conv3D(gate_channels, kernel_size=kernel_size, strides=kernel_size, padding='same')(gate)
            phi_g_f_up = UpSampling3D(size=sub_sample_factor)(phi_g_f)
            phi_g_f_up_crop = get_cropping_layer(dimension, phi_g_f_up, crop_size=(g_f_crop_size, g_f_crop_size))
            phi_gate_up = UpSampling3D(size=sub_sample_factor)(phi_gate)

            f = Activation(activation)(Add()([theta_x, phi_g_f_up_crop, phi_gate_up]))

        psi_f = Conv3D(1, kernel_size=kernel_size, strides=kernel_size, padding='same')(f)
        sigm_psi_f = Activation('sigmoid')(psi_f)

        #y = Multiply()([sigm_psi_f, input])
        y = Multiply()([sigm_psi_f, theta_x])


    else:
        raise NotImplemented

    return y
#
#
# def UpSampling3DBicubic(stride, **kwargs):
#     def layer(x):
#         input_shape = K.int_shape(x)
#         output_shape = (stride[0] * input_shape[1], stride[1] * input_shape[2], stride[2] * input_shape[3])
#         return tf.image.resize_bicubic(x, output_shape, align_corners=True)
#     return Lambda(layer, **kwargs)


def get_cropping_layer(dimension, input, crop_size=(16, 16)):
    cropping_param = (crop_size, crop_size) if dimension == 2 else (crop_size, crop_size, crop_size)

    if dimension == 2 :
        return Cropping2D(cropping=cropping_param)(input)
    else :
        return Cropping3D(cropping=cropping_param)(input)