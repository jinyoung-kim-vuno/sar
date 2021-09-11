'''
3D FC-CapsNet Based on codes written by: Rodney LaLonde (lalonde@knights.ucf.edu)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at aeneas.kim@gmail.com

Code written by: Jinyoung Kim

This file contains the network definitions for the fully convolutional capsule network architecture.
'''

import numpy as np

from keras.models import Model
from keras import backend as K
from keras.layers import Activation, Input, Concatenate, Add
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv3D
from keras.layers.core import Permute, Reshape
from keras.utils import multi_gpu_model

from utils import loss_functions, metrics, optimizers_builtin
from .capsule_layers import ConvCapsuleLayer, ConvCapsuleLayer3D, DeconvCapsuleLayer, DeconvCapsuleLayer3D, Mask, \
    Mask_3d, Length, Length_3d

K.set_image_dim_ordering('th')
#K.set_image_data_format('channels_last')

def generate_fc_capsnet_model(gen_conf, train_conf, mode, downsize_factor=2):
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

    input_shape = (num_modality,) + patch_shape
    #output_shape = (num_classes, np.prod(expected_output_shape))
    output_shape = (np.prod(expected_output_shape), num_classes)

    assert dimension in [2, 3]

    model = __generate_fc_capsnet_model(dimension, num_classes, input_shape, output_shape, activation, mode)

    model.summary()
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_functions.select(num_classes, loss_opt),
                  optimizer=optimizers_builtin.select(optimizer, initial_lr),
                  metrics=metrics.select(metric_opt))

    return model


def __generate_fc_capsnet_model(dimension, num_classes, input_shape, output_shape, activation, mode):
    input = Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D or Conv3D layer
    if dimension == 2:
        conv1 = Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(input)
        # Reshape layer to be 1 capsule x [filters] atoms
        _, H, W, C = conv1.get_shape()
        conv1_reshaped = Reshape((H.value, W.value, 1, C.value))(conv1)

        # Layer 1: Primary Capsule: Conv cap with routing 1
        primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                        routings=1, name='primarycaps')(conv1_reshaped)

        # Layer 2: Convolutional Capsule
        conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                        routings=3, name='conv_cap_2_1')(primary_caps)

        # Layer 2: Convolutional Capsule
        conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                        routings=3, name='conv_cap_2_2')(conv_cap_2_1)

        # Layer 3: Convolutional Capsule
        conv_cap_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                        routings=3, name='conv_cap_3_1')(conv_cap_2_2)

        # Layer 3: Convolutional Capsule
        conv_cap_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, strides=2, padding='same',
                                        routings=3, name='conv_cap_3_2')(conv_cap_3_1)

        # Layer 4: Convolutional Capsule
        conv_cap_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                        routings=3, name='conv_cap_4_1')(conv_cap_3_2)

        # Layer 1 Up: Deconvolutional Capsule
        deconv_cap_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=32, upsamp_type='deconv',
                                            scaling=2, padding='same', routings=3,
                                            name='deconv_cap_1_1')(conv_cap_4_1)

        # Skip connection
        up_1 = Concatenate(axis=-2, name='up_1')([deconv_cap_1_1, conv_cap_3_1])

        # Layer 1 Up: Deconvolutional Capsule
        deconv_cap_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=1,
                                          padding='same', routings=3, name='deconv_cap_1_2')(up_1)

        # Layer 2 Up: Deconvolutional Capsule
        deconv_cap_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv',
                                            scaling=2, padding='same', routings=3,
                                            name='deconv_cap_2_1')(deconv_cap_1_2)

        # Skip connection
        up_2 = Concatenate(axis=-2, name='up_2')([deconv_cap_2_1, conv_cap_2_1])

        # Layer 2 Up: Deconvolutional Capsule
        deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1,
                                          padding='same', routings=3, name='deconv_cap_2_2')(up_2)

        # Layer 3 Up: Deconvolutional Capsule
        deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=16, upsamp_type='deconv',
                                            scaling=2, padding='same', routings=3,
                                            name='deconv_cap_3_1')(deconv_cap_2_2)

        # Skip connection
        up_3 = Concatenate(axis=-2, name='up_3')([deconv_cap_3_1, conv1_reshaped])

        # Layer 4: Convolutional Capsule: 1x1
        seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                    routings=3, name='seg_caps')(up_3)

        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        out_seg = Length(num_classes=num_classes, seg=True, name='out_seg')(seg_caps)

        # Decoder network.
        _, H, W, C, A = seg_caps.get_shape()
        noise = Input(shape=((H.value, W.value, C.value, A.value)))
        y = Input(shape=input_shape[:-1] + (1,))
        masked_by_y = Mask()([seg_caps, y])  # The true label is used to mask the output of capsule layer. For training
        masked = Mask()(seg_caps)  # Mask using the capsule with maximal length. For prediction

        def shared_decoder(mask_layer):
            recon_remove_dim = Reshape((H.value, W.value, A.value))(mask_layer)

            recon_1 = Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                             activation='relu', name='recon_1')(recon_remove_dim)

            recon_2 = Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
                             activation='relu', name='recon_2')(recon_1)

            out_recon = Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                               activation='sigmoid', name='out_recon')(recon_2)

            return out_recon

    else:
        downsize_factor_cap = 1
        downsize_factor_ch = 8
        conv_kernal_size = 3
        deconv_kernal_size = 2
        strides_1 = 1
        strides_2 = 2

        print(input.shape)
        conv1 = Conv3D(filters=int(16/downsize_factor_ch),
                       kernel_size=(conv_kernal_size, conv_kernal_size, conv_kernal_size),
                       strides=(strides_1, strides_1, strides_1),
                       padding='same',
                       activation='relu',
                       name='conv1')(input)

        # Reshape layer to be 1 capsule x [filters] atoms
        _, C, H, W, Z = conv1.get_shape()
        print(conv1.get_shape())
        conv1_reshaped = Reshape((H.value, W.value, Z.value, 1, C.value))(conv1)
        print('conv1_reshaped: ', conv1_reshaped.shape)
        # Layer 1: Primary Capsule: Conv cap with routing 1
        primary_caps = ConvCapsuleLayer3D(kernel_size=conv_kernal_size, num_capsule=int(2/downsize_factor_cap),
                                          num_atoms=int(16/downsize_factor_ch),
                                          strides=strides_2, padding='same',
                                          routings=1, name='primarycaps')(conv1_reshaped)
        print('primary_caps: ', primary_caps.shape)
        # Layer 2: Convolutional Capsule
        conv_cap_2_1 = ConvCapsuleLayer3D(kernel_size=conv_kernal_size, num_capsule=int(4/downsize_factor_cap),
                                          num_atoms=int(16/downsize_factor_ch), strides=strides_1, padding='same',
                                          routings=3, name='conv_cap_2_1')(primary_caps)
        print('conv_cap_2_1: ', conv_cap_2_1.shape)
        # Layer 2: Convolutional Capsule
        conv_cap_2_2 = ConvCapsuleLayer3D(kernel_size=conv_kernal_size, num_capsule=int(4/downsize_factor_cap),
                                          num_atoms=int(32/downsize_factor_ch), strides=strides_2, padding='same',
                                          routings=3, name='conv_cap_2_2')(conv_cap_2_1)
        print('conv_cap_2_2: ', conv_cap_2_2.shape)
        # Layer 3: Convolutional Capsule
        conv_cap_3_1 = ConvCapsuleLayer3D(kernel_size=conv_kernal_size, num_capsule=int(8/downsize_factor_cap),
                                          num_atoms=int(32/downsize_factor_ch), strides=strides_1, padding='same',
                                          routings=3, name='conv_cap_3_1')(conv_cap_2_2)
        print('conv_cap_3_1: ', conv_cap_3_1.shape)
        # Layer 3: Convolutional Capsule
        conv_cap_3_2 = ConvCapsuleLayer3D(kernel_size=conv_kernal_size, num_capsule=int(8/downsize_factor_cap),
                                          num_atoms=int(64/downsize_factor_ch), strides=strides_2, padding='same',
                                          routings=3, name='conv_cap_3_2')(conv_cap_3_1)
        print('conv_cap_3_2: ', conv_cap_3_2.shape)
        # Layer 4: Convolutional Capsule
        conv_cap_4_1 = ConvCapsuleLayer3D(kernel_size=conv_kernal_size, num_capsule=int(8/downsize_factor_cap),
                                          num_atoms=int(32/downsize_factor_ch), strides=strides_1, padding='same',
                                          routings=3, name='conv_cap_4_1')(conv_cap_3_2)
        print('conv_cap_4_1: ', conv_cap_4_1.shape)

        # Layer 1 Up: Deconvolutional Capsule
        deconv_cap_1_1 = DeconvCapsuleLayer3D(kernel_size=deconv_kernal_size, num_capsule=int(8/downsize_factor_cap),
                                              num_atoms=int(32/downsize_factor_ch), upsamp_type='deconv', scaling=2,
                                              padding='same', routings=3, name='deconv_cap_1_1')(conv_cap_4_1)
        print('deconv_cap_1_1: ', deconv_cap_1_1.shape)
        # Skip connection
        up_1 = Concatenate(axis=-2, name='up_1')([deconv_cap_1_1, conv_cap_3_1])

        # Layer 1 Up: Deconvolutional Capsule
        deconv_cap_1_2 = ConvCapsuleLayer3D(kernel_size=conv_kernal_size, num_capsule=int(4/downsize_factor_cap),
                                            num_atoms=int(32/downsize_factor_ch), strides=strides_1, padding='same',
                                            routings=3, name='deconv_cap_1_2')(up_1)
        print('deconv_cap_1_2: ', deconv_cap_1_2.shape)
        # Layer 2 Up: Deconvolutional Capsule
        deconv_cap_2_1 = DeconvCapsuleLayer3D(kernel_size=deconv_kernal_size, num_capsule=int(4/downsize_factor_cap),
                                              num_atoms=int(16/downsize_factor_ch), upsamp_type='deconv', scaling=2,
                                              padding='same', routings=3, name='deconv_cap_2_1')(deconv_cap_1_2)
        print('deconv_cap_2_1: ', deconv_cap_2_1.shape)
        # Skip connection
        up_2 = Concatenate(axis=-2, name='up_2')([deconv_cap_2_1, conv_cap_2_1])

        # Layer 2 Up: Deconvolutional Capsule
        deconv_cap_2_2 = ConvCapsuleLayer3D(kernel_size=conv_kernal_size, num_capsule=int(4/downsize_factor_cap),
                                            num_atoms=int(16/downsize_factor_ch), strides=strides_1, padding='same',
                                            routings=3, name='deconv_cap_2_2')(up_2)

        print('deconv_cap_2_2: ', deconv_cap_2_2.shape)
        # Layer 3 Up: Deconvolutional Capsule
        deconv_cap_3_1 = DeconvCapsuleLayer3D(kernel_size=deconv_kernal_size, num_capsule=int(4/downsize_factor_cap),
                                              num_atoms=int(16/downsize_factor_ch), upsamp_type='deconv', scaling=2,
                                              padding='same', routings=3, name='deconv_cap_3_1')(deconv_cap_2_2)
        print('deconv_cap_3_1: ', deconv_cap_3_1.shape)

        # Skip connection
        up_3 = Concatenate(axis=-2, name='up_3')([deconv_cap_3_1, conv1_reshaped])


        conv_cap_5_1 = ConvCapsuleLayer3D(kernel_size=conv_kernal_size, num_capsule=int(4/downsize_factor_cap),
                                          num_atoms=int(16/downsize_factor_ch), strides=strides_1,
                                          padding='same', routings=3, name='conv_cap_5_1')(up_3)

        # conv_cap_6_1 = ConvCapsuleLayer3D(kernel_size=conv_kernal_size, num_capsule=int(4/downsize_factor_cap),
        #                                   num_atoms=int(16/downsize_factor_ch), strides=strides_1,
        #                                   padding='same', routings=3, name='conv_cap_6_1')(conv_cap_5_1)


        # Layer 4: Convolutional Capsule: 1x1
        seg_caps = ConvCapsuleLayer3D(kernel_size=1, num_capsule=int(4/downsize_factor_cap),
                                      num_atoms=int(16/downsize_factor_ch), strides=strides_1, padding='same',
                                      routings=3, name='seg_caps')(conv_cap_5_1)
        print('seg_caps: ', seg_caps.shape)

        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        out_seg = Length_3d(num_classes=num_classes, seg=True, name='out_seg')(seg_caps)

        print('out_seg: ', out_seg.shape)
        print(out_seg.shape)

        # # Decoder network.
        # _, H, W, Z, C, A = seg_caps.get_shape()
        # noise = Input(shape=((H.value, W.value, Z.value, C.value, A.value)))
        # y = Input(shape=input_shape[:-1]+(1,))
        # masked_by_y = Mask_3d()([seg_caps, y])  # The true label is used to mask the output of capsule layer. For training
        # masked = Mask_3d()(seg_caps)  # Mask using the capsule with maximal length. For prediction
        #
        # def shared_decoder(mask_layer):
        #     recon_remove_dim = Reshape((H.value, W.value, Z.value, A.value))(mask_layer)
        #
        #     recon_1 = Conv3D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
        #                      activation='relu', name='recon_1')(recon_remove_dim)
        #
        #     recon_2 = Conv3D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
        #                      activation='relu', name='recon_2')(recon_1)
        #
        #     out_recon = Conv3D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal',
        #                        activation='sigmoid', name='out_recon')(recon_2)
        #
        #     return out_recon

    pred = Reshape(output_shape)(out_seg)
    #pred = Permute((2, 1))(pred)
    print(pred.shape)

    # Models for training and evaluation (prediction)
    #if mode is '0' or '1':
    #     model = Model(inputs=[input, y], outputs=[out_seg, shared_decoder(masked_by_y)])
    #     # manipulate model
    #     noised_seg_caps = Add()([seg_caps, noise])
    #     masked_noised_y = Mask_3d()([noised_seg_caps, y])
    #     manipulate_model = Model(inputs=[input, y, noise], outputs=shared_decoder(masked_noised_y))
    # elif mode is 2:
    #     model = Model(inputs=input, outputs=[out_seg, shared_decoder(masked)])
    # else:
    #     raise NotImplementedError('A mode (train or test) to run should be designated')

    model = Model(inputs=[input], outputs=[pred])

    return model
