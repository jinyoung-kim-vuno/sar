import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dropout, Reshape, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose, concatenate, BatchNormalization, Activation, \
    AveragePooling2D, AveragePooling3D, Concatenate, MaxPooling2D, MaxPooling3D, Cropping2D, Cropping3D, Lambda
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.regularizers import l2
from keras.layers.core import Permute
from keras.layers.advanced_activations import LeakyReLU, PReLU, ReLU
from keras.activations import elu
from keras.initializers import he_normal
from keras.engine.topology import get_source_inputs, Network
from keras.utils import multi_gpu_model
from utils import loss_functions, metrics, optimizers_builtin
from .squeeze_excitation import spatial_and_channel_squeeze_excite_block2D, spatial_and_channel_squeeze_excite_block3D
from .SpectralNormalizationKeras import ConvSN2D, ConvSN3D, ConvSN2DTranspose
from keras.constraints import Constraint

K.set_image_dim_ordering('th')

# Ref.
# Jegou et al., CVPRW 17, "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation"
# Kamnitsas et al., MedIA17, "Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation"
# Chen et al., ECCV18, "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"


class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


def build_patch_gan(gen_conf, train_conf):

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    dimension = train_conf['dimension']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    # optimizer = train_conf['optimizer']
    # initial_lr = train_conf['initial_lr']
    is_set_random_seed = train_conf['is_set_random_seed']
    random_seed_num = train_conf['random_seed_num']
    exclusive_train = train_conf['exclusive_train']

    optimizer = train_conf['GAN']['discriminator']['optimizer']
    initial_lr = train_conf['GAN']['discriminator']['initial_lr']
    num_classes = train_conf['GAN']['discriminator']['num_classes']
    activation = train_conf['GAN']['discriminator']['activation']
    loss_opt = train_conf['GAN']['discriminator']['loss']
    metric_opt = train_conf['GAN']['discriminator']['metric']
    patch_shape = train_conf['GAN']['discriminator']['patch_shape']
    d_output_shape = train_conf['GAN']['discriminator']['output_shape']
    g_output_shape = train_conf['GAN']['generator']['output_shape']

    num_g_output = train_conf['GAN']['generator']['num_classes']

    if train_conf['GAN']['discriminator']['model_name'] == []:
        model_name = 'discriminator_patch_gan'
    else:
        model_name = train_conf['GAN']['discriminator']['model_name']

    if dimension == 2:
        K.set_image_data_format('channels_last')

    if is_set_random_seed == 1:
        kernel_init = he_normal(seed=random_seed_num)
    else:
        kernel_init = he_normal()

    if exclusive_train == 1:
        num_classes -= 1

    # if type(num_classes) is list:
    #     num_g_output = len(num_classes)
    # else:
    #     num_g_output = num_classes

    if dimension == 2:
        input_shape = [(patch_shape[0], patch_shape[1], num_modality), (g_output_shape[0], g_output_shape[1], num_g_output),
                       (np.prod(g_output_shape), num_g_output)]  # output of generator
        print(input_shape)
        output_shape = (np.prod(d_output_shape), num_classes)
    else:
        input_shape = [(num_modality, ) + patch_shape, (num_g_output, ) + g_output_shape, (np.prod(g_output_shape), num_g_output)] # output of generator
        print(input_shape)
        output_shape = (num_classes, np.prod(d_output_shape))

    input = []
    input.append(Input(shape=input_shape[0]))
    # if num_g_output == 2:
    #     input.append(Input(shape=input_shape[2]))
    #     input.append(Input(shape=input_shape[2]))
    # elif num_g_output in (1, 3, 5):
    input.append(Input(shape=input_shape[2]))
    # else:
    #     raise ValueError('The number of generator output should be 1, 2, 3, 5')

    # concatenate images channel-wise
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # if num_g_output == 2:
    #     img_target = [Reshape(input_shape[1])(Permute((2, 1))(input[1])),
    #                   Reshape(input_shape[1])(Permute((2, 1))(input[2]))]
    #     print(img_target[0].shape)
    #     print(img_target[1].shape)
    #     merged_input = concatenate([input[0], img_target[0], img_target[1]], axis=concat_axis)
    # elif num_g_output in (1, 3, 5):
    img_target = [Reshape(input_shape[1])(Permute((2, 1))(input[1]))]
    print(img_target[0].shape)
    merged_input = concatenate([input[0], img_target[0]], axis=concat_axis)
    # else:
    #     raise ValueError('The number of generator output should be 1, 2, 3, or 5')

    if activation not in ['softmax', 'sigmoid', 'linear']:
        raise ValueError('activation must be one of "softmax", "sigmoid", or "linear"')

    if loss_opt == 'mse':
        if activation != 'linear':
            raise ValueError('For mse loss, activation must be "linear" (LSGAN(ICCV17))')


    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    stride_size = (2, 2) if dimension == 2 else (2, 2, 2)

    padding_opt="same" # 'valid'

    interm_f_lst = []

    # C64
    if dimension == 2:
        d = Conv2D(64, kernel_size, strides=stride_size, padding=padding_opt, kernel_initializer=kernel_init)(
                merged_input)
    else:
        d = Conv3D(64, kernel_size, strides=stride_size, padding=padding_opt, kernel_initializer=kernel_init)(merged_input)
    #d = LeakyReLU(alpha=0.2)(d)
    d = Activation('elu')(d)

    if dimension == 2:
        interm_f = Conv2D(1, (1, 1), activation='linear', padding=padding_opt, kernel_initializer=kernel_init,
                              use_bias=False)(d)
    else:
        interm_f = Conv3D(1, (1, 1, 1), activation='linear', padding=padding_opt, kernel_initializer=kernel_init,
                          use_bias=False)(d)
    interm_f_lst.append(Activation('sigmoid')(interm_f))  # for perceptual loss

    # C128
    if dimension == 2:
        d = Conv2D(128, kernel_size, strides=stride_size, padding=padding_opt, kernel_initializer=kernel_init)(d)
    else:
        d = Conv3D(128, kernel_size, strides=stride_size, padding=padding_opt, kernel_initializer=kernel_init)(d)
    d = InstanceNormalization(axis=-1)(d)
    #d = LeakyReLU(alpha=0.2)(d)
    d = Activation('elu')(d)

    if dimension == 2:
        interm_f = Conv2D(1, (1, 1), activation='linear', padding=padding_opt, kernel_initializer=kernel_init,
                          use_bias=False)(d)
    else:
        interm_f = Conv3D(1, (1, 1, 1), activation='linear', padding=padding_opt, kernel_initializer=kernel_init,
                          use_bias=False)(d)
    interm_f_lst.append(Activation('sigmoid')(interm_f))  # for perceptual loss

    # C256
    if dimension == 2:
        d = Conv2D(256, kernel_size, strides=stride_size, padding=padding_opt, kernel_initializer=kernel_init)(d)
    else:
        d = Conv3D(256, kernel_size, strides=stride_size, padding=padding_opt, kernel_initializer=kernel_init)(d)
    d = InstanceNormalization(axis=-1)(d)
    #d = LeakyReLU(alpha=0.2)(d)
    d = Activation('elu')(d)

    if dimension == 2:
        interm_f = Conv2D(1, (1, 1), activation='linear', padding=padding_opt, kernel_initializer=kernel_init,
                          use_bias=False)(d)
    else:
        interm_f = Conv3D(1, (1, 1, 1), activation='linear', padding=padding_opt, kernel_initializer=kernel_init,
                          use_bias=False)(d)
    interm_f_lst.append(Activation('sigmoid')(interm_f))  # for perceptual loss

    # # C512
    # if dimension == 2:
    #     d = Conv2D(512, kernel_size, strides=stride_size, padding=padding_opt, kernel_initializer=kernel_init)(d)
    # else:
    #     d = Conv3D(512, kernel_size, strides=stride_size, padding=padding_opt, kernel_initializer=kernel_init)(d)
    # d = BatchNormalization()(d)
    ## d = LeakyReLU(alpha=0.2)(d)
    # d = Activation('elu')(d)

    #second last output layer
    if dimension == 2:
        d = Conv2D(128, kernel_size, padding=padding_opt, kernel_initializer=kernel_init)(d)
    else:
        d = Conv3D(128, kernel_size, padding=padding_opt, kernel_initializer=kernel_init)(d)
    d = InstanceNormalization(axis=-1)(d)
    #d = LeakyReLU(alpha=0.2)(d)
    d = Activation('elu')(d)

    # patch output
    if dimension == 2:
        d = Conv2D(num_classes, (1, 1), padding=padding_opt, kernel_initializer=kernel_init)(d)
    else:
        d = Conv3D(num_classes, (1, 1, 1), padding=padding_opt, kernel_initializer=kernel_init)(d)

    print(d._keras_shape)

    d = Reshape(output_shape)(d)
    if dimension == 3:
        d = Permute((2, 1))(d)
    output = Activation(activation)(d)

    loss_weights_f = [1.0, 0, 0, 0]

    # define model
    model = Model(inputs=input, outputs=[output, interm_f_lst[0], interm_f_lst[1], interm_f_lst[2]],
                  name=model_name)
    model.summary()
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=[loss_functions.select(num_classes, loss_opt), 'mae', 'mae', 'mae'],
                  optimizer=optimizers_builtin.select(optimizer, initial_lr),
                  loss_weights=loss_weights_f,
                  metrics=metrics.select(metric_opt))

    return model


def build_d_dilated_densenet(gen_conf, train_conf):

    '''Instantiate the DenseNet FCN architecture.
        Note that when using TensorFlow,
        for best performance you should set
        `image_data_format='channels_last'` in your Keras config
        at ~/.keras/keras.json.
        # Arguments
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_layers_per_block: number of layers in each dense block.
                Can be a positive integer or a list.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
            reduction: reduction factor of transition blocks.
                Note : reduction value is inverted to compute compression.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            init_conv_filters: number of layers in the initial convolution layer
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization) or
                'cifar10' (pre-training on CIFAR-10)..
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `channels_last` dim ordering)
                or `(3, 32, 32)` (with `channels_first` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
            activation: Type of activation at the top layer. Can be one of 'softmax'
                or 'sigmoid'. Note that if sigmoid is used, classes must be 1.
            upsampling_type: Can be one of 'deconv', 'upsampling' and
                'subpixel'. Defines type of upsampling algorithm used.
            batchsize: Fixed batch size. This is a temporary requirement for
                computation of output shape in the case of Deconvolution2D layers.
                Parameter will be removed in next iteration of Keras, which infers
                output shape of deconvolution layers automatically.
            early_transition: Start with an extra initial transition down and end with
                an extra transition up to reduce the network size.
            initial_kernel_size: The first Conv2D kernel might vary in size based on the
                application, this parameter makes it configurable.
        # Returns
            A Keras model instance.
    '''

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    dimension = train_conf['dimension']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    # optimizer = train_conf['optimizer']
    # initial_lr = train_conf['initial_lr']
    is_set_random_seed = train_conf['is_set_random_seed']
    random_seed_num = train_conf['random_seed_num']
    exclusive_train = train_conf['exclusive_train']

    optimizer = train_conf['GAN']['discriminator']['optimizer']
    initial_lr = train_conf['GAN']['discriminator']['initial_lr']
    num_classes = train_conf['GAN']['discriminator']['num_classes']
    activation = train_conf['GAN']['discriminator']['activation']
    spectral_norm = train_conf['GAN']['discriminator']['spectral_norm']
    loss_opt = train_conf['GAN']['discriminator']['loss']
    metric_opt = train_conf['GAN']['discriminator']['metric']
    patch_shape = train_conf['GAN']['discriminator']['patch_shape']
    expected_output_shape = train_conf['GAN']['discriminator']['output_shape']

    num_g_output = train_conf['GAN']['generator']['num_classes']

    if train_conf['GAN']['discriminator']['model_name'] == []:
        model_name = 'discriminator_dilated_densenet'
    else:
        model_name = train_conf['GAN']['discriminator']['model_name']

    if dimension == 2:
        K.set_image_data_format('channels_last')

    if is_set_random_seed == 1:
        kernel_init = he_normal(seed=random_seed_num)
    else:
        kernel_init = he_normal()

    padding = 'same'

    nb_dense_block = 3 # for dentate 3 #for thalamus
    nb_layers_per_block = (3, 4, 5, 6) #[3,4,5,6] # for dentate[3,4,5,6] #for thalamus #[3,4,5,6] # 4 #
    growth_rate = 8
    dropout_rate = 0.2

    is_densenet = True # densenet or conventional convnet
    is_dilated_init_conv = False  # dilated convolution in initial convolution block
    is_dilated_conv = False #True # dilated densenet db or densenet db

    if is_densenet:
        if is_dilated_conv:
            #dilation_rate_per_block = [(1, 1, 2), (1, 1, 2, 4), (1, 1, 2, 4, 8)]
            dilation_rate_per_block = [(1, 2, 4), (1, 2, 4, 8), (1, 2, 4, 8, 16)]
            #dilation_rate_per_block = [(1, 1, 2, 4, 8)]
        else:
            dilation_rate_per_block = [(1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1, 1)]
    else:
        if is_dilated_conv:
            dilation_rate_per_block = [(2, 4, 8)]
        else:
            dilation_rate_per_block = [(1, 1, 1)]

    # Global-local attention module
    glam = False #True
    glam_arrange_type = 'two_way_sequential' #'two_way_sequential' #concurrent_scSE
    glam_input_short_cut = False #True #False
    glam_final_conv = False # False: addition # True: applying fully connected conv layer (1x1x1) after concatenation instead of adding up
    glam_position = 'before_shortcut' #before_shortcut #after_shortcut

    reduction = 0.0
    weight_decay = 1E-4
    init_conv_filters = 16 #16 #32
    transition_pooling = 'max'
    weights = None
    input_tensor = None

    if dimension == 2:
        initial_kernel_size = (3, 3)
    else:
        initial_kernel_size = (3, 3, 3)

    if exclusive_train == 1:
        num_classes -= 1

    # if type(num_classes) is list:
    #     num_g_output = len(num_classes)
    # else:
    #     num_g_output = num_classes

    if dimension == 2:
        input_shape = [(patch_shape[0], patch_shape[1], num_modality), (patch_shape[0], patch_shape[1], num_g_output),
                       (np.prod(patch_shape), num_g_output)] # output of generator
        print(input_shape)
        output_shape = (np.prod(expected_output_shape), num_classes)
    else:
        input_shape = [(num_modality, ) + patch_shape, (num_g_output, ) + patch_shape, (np.prod(patch_shape), num_g_output)] # output of generator
        print(input_shape)
        output_shape = (num_classes, np.prod(expected_output_shape))

    if weights not in {None}:
        raise ValueError('The `weights` argument should be `None` (random initialization) as no '
                         'model weights are provided.')

    if input_shape is None:
        raise ValueError('For fully convolutional models, input shape must be supplied.')

    if type(nb_layers_per_block) is not list and nb_dense_block < 1:
        raise ValueError('Number of dense layers per block must be greater than 1. '
                         'Argument value was %s.' % nb_layers_per_block)

    if activation not in ['softmax', 'sigmoid', 'linear']:
        raise ValueError('activation must be one of "softmax", "sigmoid", or "linear"')

    if loss_opt == 'mse':
        if activation != 'linear':
            raise ValueError('For mse loss, activation must be "linear" (LSGAN(ICCV17))')

    # if activation == 'sigmoid' and num_classes != 1:
    #     raise ValueError('sigmoid activation can only be used when classes = 1')

    assert dimension in [2, 3]

    model, d_static_model = __generate_dilated_densenet(model_name, dimension, num_g_output, num_classes, kernel_init,
                                                        padding, random_seed_num, input_shape, output_shape,
                                                        input_tensor, nb_dense_block, nb_layers_per_block, growth_rate,
                                                        reduction, dropout_rate, weight_decay, init_conv_filters,
                                                        activation, spectral_norm, transition_pooling, initial_kernel_size,
                                                        is_densenet, is_dilated_conv, is_dilated_init_conv,
                                                        dilation_rate_per_block, glam, glam_arrange_type,
                                                        glam_input_short_cut, glam_final_conv, glam_position)

    model.summary()
    loss_weights_f = [1.0, 0, 0, 0]

    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=[loss_functions.select(num_classes, loss_opt), 'mae', 'mae', 'mae'],
                  optimizer=optimizers_builtin.select(optimizer, initial_lr),
                  loss_weights=loss_weights_f,
                  metrics=metrics.select(metric_opt))

    return model, d_static_model


def __generate_dilated_densenet(model_name, dimension, num_g_output, num_classes, kernel_init, padding, random_seed_num, input_shape=None,
                                output_shape=None, input_tensor=None, nb_dense_block=3,
                                nb_layers_per_block=(3, 4, 5, 6), growth_rate=12, reduction=0.0,
                                dropout_rate=None, weight_decay=1e-4, init_conv_filters=48, activation='sigmoid', spectral_norm=0,
                                transition_pooling='max', initial_kernel_size=(3, 3, 3), is_densenet=True,
                                is_dilated_conv=True, is_dilated_init_conv=False,
                                dilation_rate_per_block=(2, 4, 8), glam=True,
                                glam_arrange_type='two_way_sequential', glam_input_short_cut=True,
                                glam_final_conv=False, glam_position='before_shortcut'):

    ''' Build the DenseNet-FCN model
    # Arguments
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        reduction: reduction factor of transition blocks. Note : reduction value
            is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay
        nb_layers_per_block: number of layers in each dense block.
            Can be a positive integer or a list.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be (nb_dense_block + 1)
        upsampling_type: Can be one of 'upsampling', 'deconv' and 'subpixel'. Defines
            type of upsampling algorithm used.
        input_shape: Only used for shape inference in fully convolutional networks.
        activation: Type of activation at the top layer. Can be one of 'softmax' or
            'sigmoid'. Note that if sigmoid is used, classes must be 1.
        early_transition: Start with an extra initial transition down and end with an
            extra transition up to reduce the network size.
        transition_pooling: 'max' for max pooling (default), 'avg' for average pooling,
            None for no pooling. Please note that this default differs from the DenseNet
            paper in accordance with the DenseNetFCN paper.
        initial_kernel_size: The first Conv2D kernel might vary in size based on the
            application, this parameter makes it configurable.
    # Returns
        a keras tensor
    # Raises
        ValueError: in case of invalid argument for `reduction`,
            `nb_dense_block`.
    '''

    with K.name_scope('DilatedDenseNet'):

        # Determine proper input shape
        min_size = 2 ** nb_dense_block

        if K.image_data_format() == 'channels_first':

            if dimension == 2:
                if input_shape is not None:
                    if ((input_shape[0][1] is not None and input_shape[0][1] < min_size) or
                            (input_shape[0][2] is not None and input_shape[0][2] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         ', got `input_shape=' + str(input_shape) + '`')
            else:
                if input_shape is not None:
                    if ((input_shape[0][1] is not None and input_shape[0][1] < min_size) or
                            (input_shape[0][2] is not None and input_shape[0][2] < min_size) or
                            (input_shape[0][3] is not None and input_shape[0][3] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         ', got `input_shape=' + str(input_shape) + '`')
        else:
            if dimension == 2:
                if input_shape is not None:
                    if ((input_shape[0][0] is not None and input_shape[0][0] < min_size) or
                            (input_shape[0][1] is not None and input_shape[0][1] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         ', got `input_shape=' + str(input_shape) + '`')
            else:
                if input_shape is not None:
                    if ((input_shape[0][0] is not None and input_shape[0][0] < min_size) or
                            (input_shape[0][1] is not None and input_shape[0][1] < min_size) or
                            (input_shape[0][2] is not None and [0][2] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         ', got `input_shape=' + str(input_shape) + '`')
        input = []
        if input_tensor is None:
            input.append(Input(shape=input_shape[0]))
            # if num_g_output == 2:
            #     input.append(Input(shape=input_shape[2]))
            #     input.append(Input(shape=input_shape[2]))
            # elif num_g_output in (1, 3, 5):
            input.append(Input(shape=input_shape[2]))
            # else:
            #     raise ValueError('The number of generator output should be 1, 2, 3, or 5')
            # #img_input = Input(shape=input_shape[0])
            # #img_target = [Input(shape=input_shape[1]), Input(shape=input_shape[1])]
        else:
            if not K.is_keras_tensor(input_tensor):
                input.append(Input(tensor=input_tensor, shape=input_shape[0]))
                # if num_g_output == 2:
                #     input.append(Input(tensor=input_tensor, shape=input_shape[2]))
                #     input.append(Input(tensor=input_tensor, shape=input_shape[2]))
                #elif num_g_output in (1, 3, 5):
                input.append(Input(tensor=input_tensor, shape=input_shape[2]))
                # else:
                #     raise ValueError('The number of generator output should be 1, 2, or 3')
            else:
                # if num_g_output == 2:
                #     input.append(input_tensor)
                #     input.append(input_tensor)
                #     input.append(input_tensor)
                # elif num_g_output in (1, 3, 5):
                input.append(input_tensor)
                input.append(input_tensor)
                # else:
                #     raise ValueError('The number of generator output should be 1, 2, or 3')

        # concatenate images channel-wise
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        # if num_g_output == 2:
        #     img_target = [Reshape(input_shape[1])(Permute((2, 1))(input[1])),
        #                   Reshape(input_shape[1])(Permute((2, 1))(input[2]))]
        #     print(img_target[0].shape)
        #     print(img_target[1].shape)
        #     merged_input = concatenate([input[0], img_target[0], img_target[1]], axis=concat_axis)
        #
        # elif num_g_output in (1, 3, 5):
        img_target = [Reshape(input_shape[1])(Permute((2, 1))(input[1]))]
        print(img_target[0].shape)
        merged_input = concatenate([input[0], img_target[0]], axis=concat_axis)
        # else:
        #     raise ValueError('The number of generator output should be 1, 2, or 3')

        if concat_axis == 1:  # channels_first dim ordering
            if dimension == 2:
                _, rows, cols = input_shape[0]
            else:
                _, rows, cols, axes = input_shape[0]
        else:
            if dimension == 2:
                rows, cols, _ = input_shape[0]
            else:
                rows, cols, axes, _ = input_shape[0]

        if reduction != 0.0:
            if not (reduction <= 1.0 and reduction > 0.0):
                raise ValueError('`reduction` value must lie between 0.0 and 1.0')

        # layers in each dense block
        if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            if len(nb_layers) != (nb_dense_block + 1):
                raise ValueError('If `nb_dense_block` is a list, its length must be '
                                 '(`nb_dense_block` + 1)')

            bottleneck_nb_layers = nb_layers[-1]
            rev_layers = nb_layers[::-1]
            nb_layers.extend(rev_layers[1:])
        else:
            bottleneck_nb_layers = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * (2 * nb_dense_block + 1)

        # compute compression factor
        compression = 1.0 - reduction

        # dilated densenet
        output_temp, interm_f = conv_dense_transition_down(dimension,
                                                           num_classes,
                                                           merged_input,
                                                           init_conv_filters,
                                                           initial_kernel_size,
                                                           kernel_init,
                                                           padding,
                                                           random_seed_num,
                                                           weight_decay,
                                                           concat_axis,
                                                           compression,
                                                           spectral_norm,
                                                           transition_pooling,
                                                           nb_dense_block,
                                                           nb_layers,
                                                           growth_rate,
                                                           dropout_rate,
                                                           is_densenet,
                                                           is_dilated_init_conv,
                                                           is_dilated_conv,
                                                           dilation_rate_per_block,
                                                           glam,
                                                           glam_arrange_type,
                                                           glam_input_short_cut,
                                                           glam_final_conv,
                                                           glam_position,
                                                           block_prefix_num=0)


        output = organise_output(dimension, output_temp, output_shape, activation)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        # if input_tensor is not None:
        #     img_input = get_source_inputs(img_input)
        #     img_target = [get_source_inputs(img_target), get_source_inputs(img_target)]


        print(merged_input.shape)
        print(output.shape)
        print(interm_f[0].shape)
        print(interm_f[1].shape)
        print(interm_f[2].shape)

        d_model = Model(inputs=input, outputs=[output, interm_f[0], interm_f[1], interm_f[2]], name=model_name)

        # Use containers to avoid falsy keras error about weight descripancies
        d_static_model = Network(inputs=input, outputs=[output, interm_f[0], interm_f[1], interm_f[2]],
                               name=model_name + '_static_model')

        # d_model = Model(inputs=input, outputs=[output, interm_f[0], interm_f[1], interm_f[2]], name=model_name)
        #
        # # Use containers to avoid falsy keras error about weight descripancies
        # d_static_model = Network(inputs=input, outputs=[output, interm_f[0], interm_f[1], interm_f[2]],
        #                        name=model_name + '_static_model')

        return d_model, d_static_model


def conv_dense_transition_down(dimension, num_classes, img_input, init_conv_filters, initial_kernel_size, kernel_init, padding,
                               random_seed_num, weight_decay, concat_axis, compression, spectral_norm,
                               transition_pooling, nb_dense_block, nb_layers, growth_rate, dropout_rate,
                               is_densenet, is_dilated_init_conv, is_dilated_conv,
                               dilation_rate_per_block, glam, glam_arrange_type, glam_input_short_cut, glam_final_conv,
                               glam_position, block_prefix_num):

    print(is_dilated_conv)
    # Initial convolution
    if dimension == 2:
        if spectral_norm:
            if is_dilated_init_conv:
                x = ConvSN2D(init_conv_filters, initial_kernel_size, dilation_rate=(2, 2),
                           kernel_initializer=kernel_init, padding=padding,
                           name='initial_conv2D_%i' % block_prefix_num, use_bias=False,
                           kernel_regularizer=l2(weight_decay))(img_input)
            else:
                x = ConvSN2D(init_conv_filters, initial_kernel_size,
                           kernel_initializer=kernel_init, padding=padding,
                           name='initial_conv2D_%i' % block_prefix_num, use_bias=False,
                           kernel_regularizer=l2(weight_decay))(img_input)
        else:
            if is_dilated_init_conv:
                x = Conv2D(init_conv_filters, initial_kernel_size, dilation_rate=(2, 2),
                           kernel_initializer=kernel_init, padding=padding,
                           name='initial_conv2D_%i' % block_prefix_num, use_bias=False,
                           kernel_regularizer=l2(weight_decay))(img_input)
            else:
                x = Conv2D(init_conv_filters, initial_kernel_size,
                           kernel_initializer=kernel_init, padding=padding,
                           name='initial_conv2D_%i' % block_prefix_num, use_bias=False,
                           kernel_regularizer=l2(weight_decay))(img_input)

    else:
        if spectral_norm:
            if is_dilated_init_conv:
                x = ConvSN3D(init_conv_filters, initial_kernel_size, dilation_rate=(2, 2, 2),
                           kernel_initializer=kernel_init, padding=padding,
                           name='initial_conv3D_%i' % block_prefix_num, use_bias=False,
                           kernel_regularizer=l2(weight_decay))(img_input)
            else:
                x = ConvSN3D(init_conv_filters, initial_kernel_size,
                           kernel_initializer=kernel_init, padding=padding,
                           name='initial_conv3D_%i' % block_prefix_num, use_bias=False,
                           kernel_regularizer=l2(weight_decay))(img_input)
        else:
            if is_dilated_init_conv:
                x = Conv3D(init_conv_filters, initial_kernel_size, dilation_rate=(2, 2, 2),
                           kernel_initializer=kernel_init, padding=padding,
                           name='initial_conv3D_%i' % block_prefix_num, use_bias=False,
                           kernel_regularizer=l2(weight_decay))(img_input)
            else:
                x = Conv3D(init_conv_filters, initial_kernel_size,
                           kernel_initializer=kernel_init, padding=padding,
                           name='initial_conv3D_%i' % block_prefix_num, use_bias=False,
                           kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5, name='initial_bn_%i' % block_prefix_num)(x)
    #x = LeakyReLU(alpha=0.2)(x)
    x = Activation('elu')(x)

    nb_filter = init_conv_filters
    interm_f_lst = []
    if is_densenet:

        # dense blocks and transition down blocks
        for block_idx in range(nb_dense_block):
            if glam:
                org_input = x
            else:
                org_input = None

            kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
            if is_dilated_conv:
                dilation_rate_lst = []
                for d in dilation_rate_per_block[block_idx]:
                    dilation_rate = (d, d) if dimension == 2 else (d, d, d)
                    dilation_rate_lst.append(dilation_rate)
                x, nb_filter, concat_list = __dilated_dense_block(dimension, x, kernel_size, kernel_init, padding, random_seed_num,
                                                                  nb_layers[block_idx], dilation_rate_lst, nb_filter,
                                                                  growth_rate, spectral_norm, dropout_rate=dropout_rate,
                                                                  weight_decay=weight_decay, return_concat_list=True,
                                                                  block_prefix='dilated_dense_%i_%i' % (block_idx,
                                                                                                        block_prefix_num))
            else:
                dilation_rate = (1, 1) if dimension == 2 else (1, 1, 1)
                x, nb_filter, concat_list = __dense_block(dimension, x, kernel_size, kernel_init, padding, random_seed_num,
                                                          nb_layers[block_idx], dilation_rate, nb_filter, growth_rate,
                                                          spectral_norm, dropout_rate=dropout_rate, weight_decay=weight_decay,
                                                          return_concat_list=True,
                                                          block_prefix='dense_%i_%i' % (block_idx, block_prefix_num))

            if glam:
                if glam_position == 'before_shortcut':
                    l = concatenate(concat_list[1:], axis=concat_axis)
                    if dimension == 2:
                        l = spatial_and_channel_squeeze_excite_block2D(l, kernel_init, arrange_type=glam_arrange_type,
                                                                       input_short_cut=glam_input_short_cut,
                                                                       final_conv=glam_final_conv)
                    else:
                        l = spatial_and_channel_squeeze_excite_block3D(l, kernel_init, arrange_type=glam_arrange_type,
                                                                       input_short_cut=glam_input_short_cut,
                                                                       final_conv=glam_final_conv)
                    x = concatenate([org_input, l], axis=concat_axis)
                else:
                    if dimension == 2:
                        x = spatial_and_channel_squeeze_excite_block2D(x, kernel_init, arrange_type=glam_arrange_type,
                                                                       input_short_cut=glam_input_short_cut,
                                                                       final_conv=glam_final_conv)
                    else:
                        x = spatial_and_channel_squeeze_excite_block3D(x, kernel_init, arrange_type=glam_arrange_type,
                                                                       input_short_cut=glam_input_short_cut,
                                                                       final_conv=glam_final_conv)

            if not is_dilated_conv:

                # add transition_block
                x = __transition_down_block(dimension, x, nb_filter, kernel_init, padding, random_seed_num, compression=compression,
                                            weight_decay=weight_decay, dropout_rate=dropout_rate, spectral_norm=spectral_norm,
                                            block_prefix='tr_%i_%i' % (block_idx, block_prefix_num),
                                            transition_pooling=transition_pooling)

            interm_f_lst.append(perceptual_f_map(x, dimension, spectral_norm, padding, kernel_init)) #for perceptual loss


    else:
        kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
        dilation_rate_lst = dilation_rate_per_block[0]
        filter_num_lst = [64, 128, 256]
        i = 0
        for d, f in zip(dilation_rate_lst, filter_num_lst):
            i += 1
            dilation_rate = (d, d) if dimension == 2 else (d, d, d)
            x = __conv_block(dimension, x, kernel_size, kernel_init, padding, random_seed_num, f, dropout_rate, spectral_norm, False,
                             weight_decay, dilation_rate, block_prefix=name_or_none('dilated_conv', '_%i' % i))

            if glam:
                if dimension == 2:
                    x = spatial_and_channel_squeeze_excite_block2D(x, kernel_init, arrange_type=glam_arrange_type,
                                                                   input_short_cut=glam_input_short_cut,
                                                                   final_conv=glam_final_conv)
                else:
                    x = spatial_and_channel_squeeze_excite_block3D(x, kernel_init, arrange_type=glam_arrange_type,
                                                                   input_short_cut=glam_input_short_cut,
                                                                   final_conv=glam_final_conv)

            # add transition_block
            x = __transition_down_block(dimension, x, f, kernel_init, padding, random_seed_num, compression=compression,
                                        weight_decay=weight_decay, dropout_rate=dropout_rate,
                                        block_prefix='tr_%i_%i' % (i, block_prefix_num),
                                        transition_pooling=transition_pooling)

            interm_f_lst.append(perceptual_f_map(x, dimension, spectral_norm, padding, kernel_init)) # for perceptual loss



    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = x._keras_shape[channel_axis]
    # x = AveragePooling3D((2, 2, 2), strides=(8, 8, 8), padding='valid', data_format=K.image_data_format(), name='')(x)

    if dimension == 2:
        if spectral_norm:
            x = ConvSN2D(int(filters / 2), (1, 1), activation='linear', padding=padding, kernel_initializer=kernel_init,
                       use_bias=False)(x)
            x = ConvSN2D(int(filters / 4), (1, 1), activation='linear', padding=padding, kernel_initializer=kernel_init,
                       use_bias=False)(x)
            output = ConvSN2D(num_classes, (1, 1), activation='linear', padding=padding, kernel_initializer=kernel_init,
                            use_bias=False)(x)

        else:
            x = Conv2D(int(filters / 2), (1, 1), activation='linear', padding=padding, kernel_initializer=kernel_init,
                       use_bias=False)(x)
            x = Conv2D(int(filters / 4), (1, 1), activation='linear', padding=padding, kernel_initializer=kernel_init,
                       use_bias=False)(x)
            output = Conv2D(num_classes, (1, 1), activation='linear', padding=padding, kernel_initializer=kernel_init,
                            use_bias=False)(x)
    else:
        if spectral_norm:
            x = ConvSN3D(int(filters / 2), (1, 1, 1), activation='linear', padding=padding,
                       kernel_initializer=kernel_init,
                       use_bias=False)(x)
            x = ConvSN3D(int(filters / 4), (1, 1, 1), activation='linear', padding=padding,
                       kernel_initializer=kernel_init,
                       use_bias=False)(x)
            output = ConvSN3D(num_classes, (1, 1, 1), activation='linear', padding=padding,
                            kernel_initializer=kernel_init,
                            use_bias=False)(x)
        else:
            x = Conv3D(int(filters / 2), (1, 1, 1), activation='linear', padding=padding, kernel_initializer=kernel_init,
                       use_bias=False)(x)
            x = Conv3D(int(filters / 4), (1, 1, 1), activation='linear', padding=padding, kernel_initializer=kernel_init,
                       use_bias=False)(x)
            output = Conv3D(num_classes, (1, 1, 1), activation='linear', padding=padding, kernel_initializer=kernel_init,
                            use_bias=False)(x)

    return output, interm_f_lst


def organise_output(dimension, input, output_shape, activation):
    pred = Reshape(output_shape)(input)
    if dimension == 3:
        pred = Permute((2, 1))(pred)
    return Activation(activation)(pred)


def __conv_block(dimension, ip, kernel_size, kernel_init, padding, random_seed_num, nb_filter, bottleneck=False,
                 dropout_rate=None, spectral_norm=0, weight_decay=1e-4, dilation_rate=(1, 1, 1), block_prefix=None):
    '''
    Adds a convolution layer (with batch normalization and elu),
    and optionally a bottleneck layer.
    # Arguments
        ip: Input tensor
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        bottleneck: if True, adds a bottleneck convolution block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        block_prefix: str, for unique layer naming
     # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        output tensor of block
    '''
    with K.name_scope('ConvBlock'):

        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                               name=name_or_none(block_prefix, '_bn'))(ip)
        #x = LeakyReLU(alpha=0.2)(x)
        x = Activation('elu')(x)

        if bottleneck:
            inter_channel = nb_filter * 4

            if dimension == 2:
                if spectral_norm:
                    x = ConvSN2D(inter_channel, (1, 1), kernel_initializer=kernel_init,
                               padding=padding, use_bias=False, dilation_rate=dilation_rate,
                               kernel_regularizer=l2(weight_decay),
                               name=name_or_none(block_prefix, '_bottleneck_conv2D'))(x)
                else:
                    x = Conv2D(inter_channel, (1, 1), kernel_initializer=kernel_init,
                               padding=padding, use_bias=False, dilation_rate=dilation_rate,
                               kernel_regularizer=l2(weight_decay),
                               name=name_or_none(block_prefix, '_bottleneck_conv2D'))(x)
            else:
                if spectral_norm:
                    x = ConvSN3D(inter_channel, (1, 1, 1), kernel_initializer=kernel_init,
                               padding=padding, use_bias=False, dilation_rate=dilation_rate,
                               kernel_regularizer=l2(weight_decay),
                               name=name_or_none(block_prefix, '_bottleneck_conv3D'))(x)
                else:
                    x = Conv3D(inter_channel, (1, 1, 1), kernel_initializer=kernel_init,
                               padding=padding, use_bias=False, dilation_rate=dilation_rate,
                               kernel_regularizer=l2(weight_decay),
                               name=name_or_none(block_prefix, '_bottleneck_conv3D'))(x)
            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                   name=name_or_none(block_prefix, '_bottleneck_bn'))(x)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)

        if dimension == 2:
            if spectral_norm:
                x = ConvSN2D(nb_filter, kernel_size, kernel_initializer=kernel_init, padding=padding,
                           use_bias=False, dilation_rate=dilation_rate, name=name_or_none(block_prefix, '_conv2D'))(x)
            else:
                x = Conv2D(nb_filter, kernel_size, kernel_initializer=kernel_init, padding=padding,
                           use_bias=False, dilation_rate=dilation_rate, name=name_or_none(block_prefix, '_conv2D'))(x)
        else:
            if spectral_norm:
                x = ConvSN3D(nb_filter, kernel_size, kernel_initializer=kernel_init, padding=padding,
                           use_bias=False, dilation_rate=dilation_rate, name=name_or_none(block_prefix, '_conv3D'))(x)
            else:
                x = Conv3D(nb_filter, kernel_size, kernel_initializer=kernel_init, padding=padding,
                           use_bias=False, dilation_rate=dilation_rate, name=name_or_none(block_prefix, '_conv3D'))(x)
        if dropout_rate:
            x = Dropout(dropout_rate, seed=random_seed_num)(x)

    return x


def __dense_block(dimension, x, kernel_size, kernel_init, padding, random_seed_num, nb_layers, dilation_rate, nb_filter,
                  growth_rate, spectral_norm, bottleneck=False, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True,
                  return_concat_list=False, block_prefix=None):
    '''
    Build a dense_block where the output of each conv_block is fed
    to subsequent ones
    # Arguments
        x: input keras tensor
        nb_layers: the number of conv_blocks to append to the model
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        growth_rate: growth rate of the dense block
        bottleneck: if True, adds a bottleneck convolution block to
            each conv_block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: if True, allows number of filters to grow
        return_concat_list: set to True to return the list of
            feature maps along with the actual output
        block_prefix: str, for block unique naming
    # Return
        If return_concat_list is True, returns a list of the output
        keras tensor, the number of filters and a list of all the
        dense blocks added to the keras tensor
        If return_concat_list is False, returns a list of the output
        keras tensor and the number of filters
    '''
    with K.name_scope('DenseBlock'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x_list = [x]

        for i in range(nb_layers):
            cb = __conv_block(dimension, x, kernel_size, kernel_init, padding, random_seed_num, growth_rate, bottleneck,
                              dropout_rate, spectral_norm, weight_decay, dilation_rate, block_prefix=name_or_none(block_prefix, '_%i' % i))
            x_list.append(cb)

            x = concatenate([x, cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter


def __dilated_dense_block(dimension, x, kernel_size, kernel_init, padding, random_seed_num, nb_layers, dilation_rate_lst,
                          nb_filter, growth_rate, spectral_norm, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                          grow_nb_filters=True, return_concat_list=False, block_prefix=None):
    '''
    Build a dense_block where the output of each conv_block is fed
    to subsequent ones
    # Arguments
        x: input keras tensor
        nb_layers: the number of conv_blocks to append to the model
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        growth_rate: growth rate of the dense block
        bottleneck: if True, adds a bottleneck convolution block to
            each conv_block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: if True, allows number of filters to grow
        return_concat_list: set to True to return the list of
            feature maps along with the actual output
        block_prefix: str, for block unique naming
    # Return
        If return_concat_list is True, returns a list of the output
        keras tensor, the number of filters and a list of all the
        dense blocks added to the keras tensor
        If return_concat_list is False, returns a list of the output
        keras tensor and the number of filters
    '''
    with K.name_scope('DilatedDenseBlock'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x_list = [x]

        for i in range(nb_layers):
            dilation_rate = dilation_rate_lst[i]
            cb = __conv_block(dimension, x, kernel_size, kernel_init, padding, random_seed_num, growth_rate, bottleneck,
                              dropout_rate, spectral_norm, weight_decay, dilation_rate,
                              block_prefix=name_or_none(block_prefix, '_%i' % i))
            x_list.append(cb)

            x = concatenate([x, cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter



def __transition_down_block(dimension, ip, nb_filter, kernel_init, padding, random_seed_num, compression=1.0, weight_decay=1e-4,
                            dropout_rate=None, spectral_norm=0, block_prefix=None, transition_pooling='max', stride=2):
    '''
    Adds a pointwise convolution layer (with batch normalization and elu),
    and an average pooling layer. The number of output convolution filters
    can be reduced by appropriately reducing the compression parameter.
    # Arguments
        ip: input keras tensor
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        compression: calculated as 1 - reduction. Reduces the number
            of feature maps in the transition block.
        weight_decay: weight decay factor
        block_prefix: str, for block unique naming
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, nb_filter * compression, rows / 2, cols / 2)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows / 2, cols / 2, nb_filter * compression)`
        if data_format='channels_last'.
    # Returns
        a keras tensor
    '''
    with K.name_scope('Transition'):
        strides = (stride, stride) if dimension == 2 else (stride, stride, stride)
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                               name=name_or_none(block_prefix, '_bn'))(ip)
        #x = LeakyReLU(alpha=0.2)(x)
        x = Activation('elu')(x)

        if dimension == 2:
            if spectral_norm:
                x = ConvSN2D(int(nb_filter * compression), (1, 1), kernel_initializer=kernel_init,
                           padding=padding, strides=strides, use_bias=False, kernel_regularizer=l2(weight_decay),
                           name=name_or_none(block_prefix, '_conv2D'))(x)
            else:
                x = Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer=kernel_init,
                           padding=padding, strides=strides, use_bias=False, kernel_regularizer=l2(weight_decay),
                           name=name_or_none(block_prefix, '_conv2D'))(x)
            if dropout_rate:
                x = Dropout(dropout_rate, seed=random_seed_num)(x)
            if stride == 1:
                if transition_pooling == 'avg':
                    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
                elif transition_pooling == 'max':
                    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        else:
            if spectral_norm:
                x = ConvSN3D(int(nb_filter * compression), (1, 1, 1), kernel_initializer=kernel_init,
                           padding=padding, strides=strides, use_bias=False, kernel_regularizer=l2(weight_decay),
                           name=name_or_none(block_prefix, '_conv3D'))(x)
            else:
                x = Conv3D(int(nb_filter * compression), (1, 1, 1), kernel_initializer=kernel_init,
                           padding=padding, strides=strides, use_bias=False, kernel_regularizer=l2(weight_decay),
                           name=name_or_none(block_prefix, '_conv3D'))(x)
            if dropout_rate:
                x = Dropout(dropout_rate, seed=random_seed_num)(x)
            if stride == 1:
                if transition_pooling == 'avg':
                    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
                elif transition_pooling == 'max':
                    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(x)

        return x


def name_or_none(prefix, name):
    return prefix + name if (prefix is not None and name is not None) else None


def build_d_unet(gen_conf, train_conf):

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    dimension = train_conf['dimension']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    # optimizer = train_conf['optimizer']
    # initial_lr = train_conf['initial_lr']
    is_set_random_seed = train_conf['is_set_random_seed']
    random_seed_num = train_conf['random_seed_num']

    optimizer = train_conf['GAN']['discriminator']['optimizer']
    initial_lr = train_conf['GAN']['discriminator']['initial_lr']
    num_classes = train_conf['GAN']['discriminator']['num_classes']
    activation = train_conf['GAN']['discriminator']['activation']
    spectral_norm = train_conf['GAN']['discriminator']['spectral_norm']
    loss_opt = train_conf['GAN']['discriminator']['loss']
    metric_opt = train_conf['GAN']['discriminator']['metric']
    patch_shape = train_conf['GAN']['discriminator']['patch_shape']
    expected_output_shape = train_conf['GAN']['discriminator']['output_shape']

    num_g_output = train_conf['GAN']['generator']['num_classes']

    if train_conf['GAN']['discriminator']['model_name'] == []:
        model_name = 'discriminator_unet'
    else:
        model_name = train_conf['GAN']['discriminator']['model_name']

    if dimension == 2:
        K.set_image_data_format('channels_last')

    padding = 'same'

    if dimension == 2:
        input_shape = [(patch_shape[0], patch_shape[1], num_modality), (patch_shape[0], patch_shape[1], num_g_output),
                       (np.prod(patch_shape), num_g_output)] # output of generator
        print(input_shape)
        output_shape1 = (np.prod(expected_output_shape), num_classes)
        output_shape2 = (np.prod(np.divide(expected_output_shape, 8).astype(int)), num_classes)
    else:
        input_shape = [(num_modality, ) + patch_shape, (num_g_output, ) + patch_shape, (np.prod(patch_shape), num_g_output)] # output of generator
        print(input_shape)
        output_shape1 = (num_classes, np.prod(expected_output_shape))
        output_shape2 = (num_classes, np.prod(np.divide(expected_output_shape, 8).astype(int)))


    if input_shape is None:
        raise ValueError('For fully convolutional models, input shape must be supplied.')

    if activation not in ['softmax', 'sigmoid', 'linear']:
        raise ValueError('activation must be one of "softmax", "sigmoid", or "linear"')

    if loss_opt == 'mse':
        if activation != 'linear':
            raise ValueError('For mse loss, activation must be "linear" (LSGAN(ICCV17))')

    assert dimension in [2, 3]

    if is_set_random_seed == 1:
        kernel_init = he_normal(seed=random_seed_num)
    else:
        kernel_init = he_normal()

    #const = ClipConstraint(0.01)

    d_model, d_static_model = __generate_unet_model(model_name, dimension, num_classes, input_shape, output_shape1, output_shape2,
                                                    activation, spectral_norm, kernel_init, padding, input_tensor=None,
                                                    downsize_factor=2)

    d_model.summary()
    # loss_weights_f = [1.0, 1.0, 0, 0, 0]
    #
    # if num_of_gpu > 1:
    #     d_model = multi_gpu_model(d_model, gpus=num_of_gpu)
    #
    # d_model.compile(loss=[loss_opt, loss_opt, 'mae', 'mae', 'mae'],
    #                 optimizer=optimizers_builtin.select(optimizer, initial_lr, clipvalue=1.0),  #clipvalue=1.0 #clipnorm=1.0
    #                 loss_weights=loss_weights_f, metrics=metrics.select(metric_opt))

    return d_model, d_static_model


def __generate_unet_model(model_name, dimension, num_classes, input_shape, output_shape1, output_shape2, activation, spectral_norm,
                          kernel_init, padding, input_tensor=None, downsize_factor=2):

    with K.name_scope('U-Net'):
        input = []
        if input_tensor is None:
            input.append(Input(shape=input_shape[0]))
            input.append(Input(shape=input_shape[2]))
        else:
            if not K.is_keras_tensor(input_tensor):
                input.append(Input(tensor=input_tensor, shape=input_shape[2]))
            else:
                input.append(input_tensor)
                input.append(input_tensor)
        # concatenate images channel-wise
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        img_target = [Reshape(input_shape[1])(Permute((2, 1))(input[1]))]
        print(img_target[0].shape)
        merged_input = concatenate([input[0], img_target[0]], axis=concat_axis)

        conv1 = get_conv_core(dimension, merged_input, spectral_norm, int(64/downsize_factor), padding, kernel_init, concat_axis)
        #conv1 = get_conv_core_clipconst(dimension, merged_input, spectral_norm, int(64 / downsize_factor), padding,
        #                                kernel_init, const, concat_axis)
        pool1 = get_max_pooling_layer(dimension, conv1)
        interm_f1 = perceptual_f_map(dimension, pool1, spectral_norm, padding, kernel_init)

        conv2 = get_conv_core(dimension, pool1, spectral_norm, int(128/downsize_factor), padding, kernel_init, concat_axis)
        #conv2 = get_conv_core_clipconst(dimension, pool1, spectral_norm, int(128 / downsize_factor), padding,
        #                                kernel_init, const, concat_axis)
        pool2 = get_max_pooling_layer(dimension, conv2)
        interm_f2 = perceptual_f_map(dimension, pool2, spectral_norm, padding, kernel_init)

        conv3 = get_conv_core(dimension, pool2, spectral_norm, int(256/downsize_factor), padding, kernel_init, concat_axis)
        #conv3 = get_conv_core_clipconst(dimension, pool2, spectral_norm, int(256 / downsize_factor), padding,
        #                                kernel_init, const, concat_axis)
        pool3 = get_max_pooling_layer(dimension, conv3)
        interm_f3 = perceptual_f_map(dimension, pool3, spectral_norm, padding, kernel_init)

        # encoder output
        conv4 = get_conv_core(dimension, pool3, spectral_norm, int(512/downsize_factor), padding, kernel_init, concat_axis)
        #conv4 = get_conv_core_clipconst(dimension, pool3, spectral_norm, int(512 / downsize_factor), padding,
        #                                kernel_init, const, concat_axis)

        up5 = get_deconv_layer(dimension, conv4, spectral_norm, int(256/downsize_factor), kernel_init)
        up5 = concatenate([up5, conv3], axis=concat_axis)

        conv5 = get_conv_core(dimension, up5, spectral_norm, int(256/downsize_factor), padding, kernel_init, concat_axis)
        #conv5 = get_conv_core_clipconst(dimension, up5, spectral_norm, int(256 / downsize_factor), padding, kernel_init,
        #                                const, concat_axis)

        up6 = get_deconv_layer(dimension, conv5, spectral_norm, int(128/downsize_factor), kernel_init)
        up6 = concatenate([up6, conv2], axis=concat_axis)

        conv6 = get_conv_core(dimension, up6, spectral_norm, int(128/downsize_factor), padding, kernel_init, concat_axis)
        #conv6 = get_conv_core_clipconst(dimension, up6, spectral_norm, int(128 / downsize_factor), padding, kernel_init,
        #                                const, concat_axis)

        up7 = get_deconv_layer(dimension, conv6, spectral_norm, int(64/downsize_factor), kernel_init)
        up7 = concatenate([up7, conv1], axis=concat_axis)

        conv7 = get_conv_core(dimension, up7, spectral_norm, int(64/downsize_factor), padding, kernel_init, concat_axis)
        # conv7 = get_conv_core_clipconst(dimension, up7, spectral_norm, int(64 / downsize_factor), padding, kernel_init,
        #                                 const, concat_axis)

        pred1 = get_conv_fc(dimension, conv7, spectral_norm, num_classes, kernel_init)
        pred2 = get_conv_fc(dimension, conv4, spectral_norm, num_classes, kernel_init)

        print(pred1.shape)
        print(pred2.shape)
        print(output_shape1)
        print(output_shape2)

        output1 = organise_output(dimension, pred1, output_shape1, activation)
        output2 = organise_output(dimension, pred2, output_shape2, activation)

        print(merged_input.shape)
        print(output1.shape)
        print(output2.shape)
        print(interm_f1.shape)
        print(interm_f2.shape)
        print(interm_f3.shape)

        d_model = Model(inputs=input, outputs=[output1, output2, interm_f1, interm_f2, interm_f3], name=model_name)

        d_static_model = Network(inputs=input, outputs=[output1, output2, interm_f1, interm_f2, interm_f3],
                                 name=model_name + '_static_model')


        return d_model, d_static_model


def build_d_unet_cr(gen_conf, train_conf, d_model):

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    dimension = train_conf['dimension']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)

    optimizer = train_conf['GAN']['discriminator']['optimizer']
    initial_lr = train_conf['GAN']['discriminator']['initial_lr']
    loss_opt = train_conf['GAN']['discriminator']['loss']
    metric_opt = train_conf['GAN']['discriminator']['metric']
    patch_shape = train_conf['GAN']['discriminator']['patch_shape']

    num_g_output = train_conf['GAN']['generator']['num_classes']

    if dimension == 2:
        K.set_image_data_format('channels_last')

    if dimension == 2:
        input_shape = (patch_shape[0], patch_shape[1], num_modality)
        output_shape_prod = (np.prod(patch_shape), num_g_output)
    else:
        input_shape = (num_modality, ) + patch_shape
        output_shape_prod = (np.prod(patch_shape), num_g_output)

    # define the source image
    g_in_src = Input(shape=input_shape)
    g_sar_real = Input(shape=output_shape_prod) # real SAR
    g_sar_real_am = Input(shape=output_shape_prod)  # augmixed real SAR
    g_sar_pred = Input(shape=output_shape_prod) # real SAR

    [dis_real_am_out, dis_real_am_e_out, interm_f1_real_am_out, interm_f2_real_am_out, interm_f3_real_am_out] = \
        d_model([g_in_src, g_sar_real_am])

    [dis_real_out, dis_real_e_out, interm_f1_real_out, interm_f2_real_out, interm_f3_real_out] = \
        d_model([g_in_src, g_sar_real])
    [dis_pred_out, dis_pred_e_out, interm_f1_pred_out, interm_f2_pred_out, interm_f3_pred_out] = \
        d_model([g_in_src, g_sar_pred])

    tf_dis_real_out = Lambda(lambda x: x)(dis_real_out)
    tf_dis_real_am_out = Lambda(lambda x: x)(dis_real_am_out)

    tf_dis_real_e_out = Lambda(lambda x: x)(dis_real_e_out)
    tf_dis_real_am_e_out = Lambda(lambda x: x)(dis_real_am_e_out)

    print(tf_dis_real_out.shape)
    print(tf_dis_real_am_out.shape)
    print(tf_dis_real_e_out.shape)
    print(tf_dis_real_am_e_out.shape)


    d_cr_model = Model(inputs=[g_in_src, g_sar_real, g_sar_real_am, g_sar_pred],
                    outputs=[dis_real_out, dis_real_e_out, dis_pred_out, dis_pred_e_out, g_sar_real, g_sar_real,
                             interm_f1_real_out, interm_f2_real_out, interm_f3_real_out,
                             interm_f1_real_am_out, interm_f2_real_am_out, interm_f3_real_am_out,
                             interm_f1_pred_out, interm_f2_pred_out, interm_f3_pred_out])

    d_cr_model.summary()

    loss_d_cr = [loss_opt, loss_opt, loss_opt, loss_opt,
                 loss_functions.mse(tf_dis_real_out, tf_dis_real_am_out),
                 loss_functions.mse(tf_dis_real_e_out, tf_dis_real_am_e_out),
                 'mae', 'mae', 'mae', 'mae', 'mae', 'mae', 'mae', 'mae', 'mae']

    loss_d_cr_weights = [0.5, 0.5, 0.5, 0.5, 10.0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    if num_of_gpu > 1:
        d_cr_model = multi_gpu_model(d_cr_model, gpus=num_of_gpu)

    d_cr_model.compile(loss=loss_d_cr,
                    optimizer=optimizers_builtin.select(optimizer, initial_lr, clipvalue=1.0),  #clipvalue=1.0 #clipnorm=1.0
                    loss_weights=loss_d_cr_weights, metrics=metrics.select(metric_opt))

    return d_cr_model



def get_conv_core(dimension, input, spectral_norm, num_filters, padding, kernel_init, concat_axis):

    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    if dimension == 2 :
        if spectral_norm:
            x = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(input)
            x = Activation('elu')(x)
            x = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            x = Activation('elu')(x)
        else:
            x = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(input)
            x = BatchNormalization(axis=concat_axis)(x)
            x = Activation('elu')(x)
            x = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            x = BatchNormalization(axis=concat_axis)(x)
            x = Activation('elu')(x)
    else :
        if spectral_norm:
            x = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(input)
            x = Activation('elu')(x)
            x = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            x = Activation('elu')(x)

        else:
            x = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(input)
            x = BatchNormalization(axis=concat_axis)(x)
            x = Activation('elu')(x)
            x = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init)(x)
            x = BatchNormalization(axis=concat_axis)(x)
            x = Activation('elu')(x)

    return x


def get_conv_core_clipconst(dimension, input, spectral_norm, num_filters, padding, kernel_init, const, concat_axis):

    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    if dimension == 2 :
        if spectral_norm:
            x = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                         kernel_constraint=const)(input)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)
            x = ConvSN2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                         kernel_constraint=const)(x)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)
        else:
            x = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                       kernel_constraint=const)(input)
            x = BatchNormalization(axis=concat_axis)(x)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)
            x = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                       kernel_constraint=const)(x)
            x = BatchNormalization(axis=concat_axis)(x)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)
    else :
        if spectral_norm:
            x = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                         kernel_constraint=const)(input)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)
            x = ConvSN3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                         kernel_constraint=const)(x)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)

        else:
            x = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                       kernel_constraint=const)(input)
            x = BatchNormalization(axis=concat_axis)(x)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)
            x = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_init,
                       kernel_constraint=const)(x)
            x = BatchNormalization(axis=concat_axis)(x)
            #x = LeakyReLU(alpha=0.2)(x)
            x = Activation('elu')(x)

    return x


def get_max_pooling_layer(dimension, input) :
    pool_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return MaxPooling2D(pool_size=pool_size)(input)
    else :
        return MaxPooling3D(
            pool_size=pool_size)(input)


def get_deconv_layer(dimension, input, spectral_norm, num_filters, kernel_init):
    strides = (2, 2) if dimension == 2 else (2, 2, 2)
    kernel_size = (2, 2) if dimension == 2 else (2, 2, 2)
    if dimension == 2:
        if spectral_norm:
            return ConvSN2DTranspose(num_filters, kernel_size=kernel_size, strides=strides,
                                     kernel_initializer=kernel_init)(input)
        else:
            return Conv2DTranspose(num_filters, kernel_size=kernel_size, strides=strides, kernel_initializer=kernel_init)(input)
    else :
        return Conv3DTranspose(num_filters, kernel_size=kernel_size, strides=strides, kernel_initializer=kernel_init)(input)


def get_conv_fc(dimension, input, spectral_norm, num_filters, kernel_init):
    fc = None
    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)
    if dimension == 2 :
        if spectral_norm:
            fc = ConvSN2D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(input)
        else:
            fc = Conv2D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(input)
    else :
        if spectral_norm:
            fc = ConvSN3D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(input)
        else:
            fc = Conv3D(num_filters, kernel_size=kernel_size, kernel_initializer=kernel_init)(input)

    return Activation('elu')(fc)


def perceptual_f_map(dimension, input, spectral_norm, padding, kernel_init):
    if dimension == 2:
        if spectral_norm:
            interm_f = ConvSN2D(1, (1, 1), activation='linear', padding=padding, kernel_initializer=kernel_init,
                                use_bias=False)(input)
        else:
            interm_f = Conv2D(1, (1, 1), activation='linear', padding=padding, kernel_initializer=kernel_init,
                              use_bias=False)(input)
    else:
        if spectral_norm:
            interm_f = ConvSN3D(1, (1, 1, 1), activation='linear', padding=padding,
                                kernel_initializer=kernel_init,
                                use_bias=False)(input)
        else:
            interm_f = Conv3D(1, (1, 1, 1), activation='linear', padding=padding, kernel_initializer=kernel_init,
                              use_bias=False)(input)

    return Activation('sigmoid')(interm_f)