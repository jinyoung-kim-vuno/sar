from keras import backend as K
from keras.layers import Activation, Add, Multiply, TimeDistributed
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.normalization import BatchNormalization

K.set_image_dim_ordering('th')


def grid_attention(dimension, input, gate, in_channels, out_channels, mode):

    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)
    sub_sample_factor = (2, 2) if dimension == 2 else (2, 2, 2)
    sub_sample_kernel_size = sub_sample_factor

    # Define the operation
    if mode == 'concatenation':
        return _concatenation(dimension, input, gate, in_channels, out_channels, kernel_size, sub_sample_factor,
                              sub_sample_kernel_size, 'relu')
    elif mode == 'concatenation_debug':
        return _concatenation(dimension, input, gate, in_channels, out_channels, kernel_size, sub_sample_factor,
                              sub_sample_kernel_size, 'softplus')
    elif mode == 'concatenation_residual':
        return _concatenation_residual(dimension, input, gate, in_channels, out_channels, kernel_size,
                                       sub_sample_factor, sub_sample_kernel_size, 'relu')
    else:
        raise NotImplementedError('Unknown operation function.')


def _concatenation(dimension, input, gate, in_channels, out_channels, kernel_size, sub_sample_factor,
                   sub_sample_kernel_size, activation):

    if dimension == 2:
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        theta_x = TimeDistributed(Conv2D(out_channels, kernel_size=sub_sample_kernel_size, strides=sub_sample_factor,
                         padding='same'))(input)
        phi_g = TimeDistributed(Conv2D(out_channels, kernel_size=kernel_size, strides=kernel_size, padding='same'))(gate)
        f = TimeDistributed(Activation(activation))(Add()([theta_x, phi_g]))

        psi_f = TimeDistributed(Conv2D(1, kernel_size=kernel_size, strides=kernel_size, padding='same'))(f)
        sigm_psi_f = TimeDistributed(Activation('sigmoid'))(psi_f)

        # upsample the attentions and multiply
        sigm_psi_f_up = TimeDistributed(Conv2DTranspose(1, kernel_size=kernel_size, strides=sub_sample_factor))(sigm_psi_f)

        y = Multiply()([sigm_psi_f_up, input])
        print('y: ', y)

        # worked here

        # Output transform
        w = TimeDistributed(Conv2D(in_channels, kernel_size=kernel_size, padding='same'))(y)
        w = TimeDistributed(BatchNormalization(axis=2))(w)
        print(w)

    elif dimension == 3:
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        theta_x = TimeDistributed(Conv3D(out_channels, kernel_size=sub_sample_kernel_size, strides=sub_sample_factor,
                         padding='same'))(input)
        print ('theta_x: ', theta_x._keras_shape)
        phi_g = TimeDistributed(Conv3D(out_channels, kernel_size=kernel_size, strides=kernel_size, padding='same'))(gate)
        print('phi_g: ', phi_g._keras_shape)
        f = TimeDistributed(Activation(activation))(Add()([theta_x, phi_g]))

        psi_f = TimeDistributed(Conv3D(1, kernel_size=kernel_size, strides=kernel_size, padding='same'))(f)
        sigm_psi_f = TimeDistributed(Activation('sigmoid'))(psi_f)

        # upsample the attentions and multiply
        sigm_psi_f_up = TimeDistributed(Conv3DTranspose(1, kernel_size=kernel_size, strides=sub_sample_factor))(sigm_psi_f)

        y = Multiply()([sigm_psi_f_up, input])
        print('y: ', y)

        # Output transform
        w = TimeDistributed(Conv3D(in_channels, kernel_size=kernel_size, padding='same'))(y)
        w = TimeDistributed(BatchNormalization(axis=2))(w)
        print(w)

    else:
        raise NotImplemented

    return w, sigm_psi_f_up


def _concatenation_residual(dimension, input, gate, in_channels, out_channels, kernel_size, sub_sample_factor,
                            sub_sample_kernel_size, activation):

    if dimension == 2:
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        theta_x = TimeDistributed(Conv2D(out_channels, kernel_size=sub_sample_kernel_size, strides=sub_sample_factor,
                         padding='same'))(input)
        phi_g = TimeDistributed(Conv2D(out_channels, kernel_size=kernel_size, strides=kernel_size, padding='same'))(gate)
        f = TimeDistributed(Activation(activation))(Add()([theta_x, phi_g]))

        psi_f = TimeDistributed(Conv2D(1, kernel_size=kernel_size, strides=kernel_size, padding='same'))(f)
        sigm_psi_f = TimeDistributed(Activation('softmax'))(psi_f)

        # upsample the attentions and multiply
        sigm_psi_f_up = TimeDistributed(Conv2DTranspose(1, kernel_size=kernel_size, strides=sub_sample_factor))(sigm_psi_f)

        y = TimeDistributed(Multiply())([sigm_psi_f_up, input])
        print('y: ', y)

        # worked here

        # Output transform
        w = TimeDistributed(Conv2D(in_channels, kernel_size=kernel_size, padding='same'))(y)
        w = TimeDistributed(BatchNormalization(axis=2))(w)
        print(w)

    elif dimension == 3:
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        theta_x = TimeDistributed(Conv3D(out_channels, kernel_size=sub_sample_kernel_size, strides=sub_sample_factor,
                         padding='same'))(input)
        phi_g = TimeDistributed(Conv3D(out_channels, kernel_size=kernel_size, strides=kernel_size, padding='same'))(gate)
        f = TimeDistributed(Activation(activation))(Add()([theta_x, phi_g]))

        psi_f = TimeDistributed(Conv3D(1, kernel_size=kernel_size, strides=kernel_size, padding='same'))(f)
        sigm_psi_f = TimeDistributed(Activation('softmax'))(psi_f)

        # upsample the attentions and multiply
        sigm_psi_f_up = TimeDistributed(Conv3DTranspose(1, kernel_size=kernel_size, strides=sub_sample_factor))(sigm_psi_f)

        y = TimeDistributed(Multiply())([sigm_psi_f_up, input])
        print('y: ', y)

        # worked here

        # Output transform
        w = TimeDistributed(Conv3D(in_channels, kernel_size=kernel_size, padding='same'))(y)
        w = TimeDistributed(BatchNormalization(axis=2))(w)
        print(w)

    else:
        raise NotImplemented

    return w, sigm_psi_f_up


def get_gate_signal(dimension, input, num_filters) :
    x = None
    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)

    if dimension == 2 :
        x = TimeDistributed(Conv2D(num_filters, kernel_size=kernel_size, padding='same'))(input)
        x = TimeDistributed(BatchNormalization(axis=2))(x)
        x = TimeDistributed(Activation('relu'))(x)

    else :
        x = TimeDistributed(Conv3D(num_filters, kernel_size=kernel_size, padding='same'))(input)
        x = TimeDistributed(BatchNormalization(axis=2))(x)
        x = TimeDistributed(Activation('relu'))(x)

    return x
