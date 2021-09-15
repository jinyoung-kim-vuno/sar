import numpy as np
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Reshape, Conv2D, Conv3D, Conv2DTranspose, \
    Conv3DTranspose, UpSampling2D, UpSampling3D, MaxPooling2D, MaxPooling3D, AveragePooling2D, AveragePooling3D, \
    GlobalMaxPooling2D, GlobalMaxPooling3D, GlobalAveragePooling2D, GlobalAveragePooling3D, concatenate, \
    BatchNormalization, Activation, Add, Multiply, Permute
from tensorflow.keras.regularizers import l2
# from keras.layers.core import Permute
#from keras.engine.topology import get_source_inputs
from tensorflow.keras.utils import get_source_inputs
#from keras_contrib.layers import SubPixelUpscaling
from tensorflow.keras.utils import multi_gpu_model

from utils import loss_functions, metrics, optimizers_builtin
from .attention_gates import grid_attention, get_gate_signal
from .squeeze_excitation import squeeze_excite_block2D,squeeze_excite_block3D

K.set_image_data_format('channels_first')
#K.set_image_dim_ordering('th')

# Jegou et al., CVPRW 17, "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation"


def generate_fc_densenet_model(gen_conf, train_conf) :

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
            upsampling_conv: number of convolutional layers in upsampling via subpixel
                convolution
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

    nb_dense_block = 3
    nb_layers_per_block = 4 # 4 # [4,4,6,8]
    growth_rate = 8 #8 #16
    reduction = 0.0
    dropout_rate = 0.0 # 0.2
    weight_decay = 1E-4
    init_conv_filters = 16 #16 #32
    include_top = True
    weights = None
    input_tensor = None
    upsampling_type = 'deconv'
    transition_pooling = 'max'
    early_transition = False

    if dimension == 2:
        initial_kernel_size = (3, 3)
    else:
        initial_kernel_size = (3, 3, 3)

    if exclusive_train == 1:
        num_classes -= 1

    input_shape = (num_modality, ) + patch_shape
    output_shape = (num_classes, np.prod(expected_output_shape))

    if weights not in {None}:
        raise ValueError('The `weights` argument should be '
                         '`None` (random initialization) as no '
                         'model weights are provided.')

    upsampling_type = upsampling_type.lower()

    if upsampling_type not in ['upsampling', 'deconv', 'subpixel']:
        raise ValueError('Parameter "upsampling_type" must be one of "upsampling", '
                         '"deconv" or "subpixel".')

    if input_shape is None:
        raise ValueError('For fully convolutional models, input shape must be supplied.')

    if type(nb_layers_per_block) is not list and nb_dense_block < 1:
        raise ValueError('Number of dense layers per block must be greater than 1. '
                         'Argument value was %d.' % nb_layers_per_block)

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')

    if activation == 'sigmoid' and num_classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')

    assert dimension in [2, 3]

    model = __generate_fc_densenet_model(dimension, num_classes, input_shape, output_shape, input_tensor,
                                         nb_dense_block, nb_layers_per_block, growth_rate, reduction,
                                         dropout_rate, weight_decay, init_conv_filters, include_top, upsampling_type,
                                         activation, early_transition, transition_pooling, initial_kernel_size)

    model.summary()
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_functions.select(num_classes, loss_opt),
                  optimizer=optimizers_builtin.select(optimizer, initial_lr),
                  metrics=metrics.select(metric_opt))

    return model


def __generate_fc_densenet_model(dimension, num_classes, input_shape=None, output_shape=None, input_tensor=None,
                                 nb_dense_block=5, nb_layers_per_block=4, growth_rate=12, reduction=0.0,
                                 dropout_rate=None, weight_decay=1e-4, init_conv_filters=48,
                                 include_top=None, upsampling_type='deconv', activation='softmax',
                                 early_transition=False, transition_pooling='max', initial_kernel_size=(3, 3, 3)):

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
    with K.name_scope('DenseNetFCN'):

        # Determine proper input shape
        min_size = 2 ** nb_dense_block

        if K.image_data_format() == 'channels_first':

            if dimension == 2:
                if input_shape is not None:
                    if ((input_shape[1] is not None and input_shape[1] < min_size) or
                            (input_shape[2] is not None and input_shape[2] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         ', got `input_shape=' + str(input_shape) + '`')
            else:
                if input_shape is not None:
                    if ((input_shape[1] is not None and input_shape[1] < min_size) or
                            (input_shape[2] is not None and input_shape[2] < min_size) or
                            (input_shape[3] is not None and input_shape[3] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         ', got `input_shape=' + str(input_shape) + '`')
        else:
            if dimension == 2:
                if input_shape is not None:
                    if ((input_shape[0] is not None and input_shape[0] < min_size) or
                            (input_shape[1] is not None and input_shape[1] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         ', got `input_shape=' + str(input_shape) + '`')
            else:
                if input_shape is not None:
                    if ((input_shape[0] is not None and input_shape[0] < min_size) or
                            (input_shape[1] is not None and input_shape[1] < min_size) or
                            (input_shape[2] is not None and input_shape[2] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         ', got `input_shape=' + str(input_shape) + '`')

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if concat_axis == 1:  # channels_first dim ordering
            if dimension == 2:
                _, rows, cols = input_shape
            else:
                _, rows, cols, axes = input_shape
        else:
            if dimension == 2:
                rows, cols, _ = input_shape
            else:
                rows, cols, axes, _ = input_shape

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

        # Initial convolution
        if dimension == 2:
            x = Conv2D(init_conv_filters, initial_kernel_size,
                       kernel_initializer='he_normal', padding='same',
                       name='initial_conv2D', use_bias=False,
                       kernel_regularizer=l2(weight_decay))(img_input)
        else:
            x = Conv3D(init_conv_filters, initial_kernel_size,
                       kernel_initializer='he_normal', padding='same',
                       name='initial_conv3D', use_bias=False,
                       kernel_regularizer=l2(weight_decay))(img_input)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5, name='initial_bn')(x)
        x = Activation('relu')(x)

        nb_filter = init_conv_filters

        skip_list = []

        if early_transition:
            x = __transition_block(dimension, x, nb_filter, compression=compression,
                                   weight_decay=weight_decay, block_prefix='tr_early',
                                   transition_pooling=transition_pooling)

        # Add dense blocks and transition down block
        for block_idx in range(nb_dense_block):
            x, nb_filter = __dense_block(dimension, x, nb_layers[block_idx], nb_filter,
                                         growth_rate, dropout_rate=dropout_rate,
                                         weight_decay=weight_decay,
                                         block_prefix='dense_%i' % block_idx)

            # Skip connection
            skip_list.append(x)

            # add transition_block
            x = __transition_block(dimension, x, nb_filter, compression=compression,
                                   weight_decay=weight_decay,
                                   block_prefix='tr_%i' % block_idx,
                                   transition_pooling=transition_pooling)

            # this is calculated inside transition_down_block
            nb_filter = int(nb_filter * compression)

        # The last dense_block does not have a transition_down_block
        # return the concatenated feature maps without the concatenation of the input
        block_prefix = 'dense_%i' % nb_dense_block
        _, nb_filter, concat_list = __dense_block(dimension, x, bottleneck_nb_layers, nb_filter,
                                                  growth_rate,
                                                  dropout_rate=dropout_rate,
                                                  weight_decay=weight_decay,
                                                  return_concat_list=True,
                                                  block_prefix=block_prefix)

        skip_list = skip_list[::-1]  # reverse the skip list

        # Add dense blocks and transition up block
        for block_idx in range(nb_dense_block):
            n_filters_keep = growth_rate * nb_layers[nb_dense_block + block_idx]

            # upsampling block must upsample only the feature maps (concat_list[1:]),
            # not the concatenation of the input with the feature maps (concat_list[0].
            l = concatenate(concat_list[1:], axis=concat_axis)

            t = __transition_up_block(dimension, l, nb_filters=n_filters_keep,
                                      type=upsampling_type, weight_decay=weight_decay,
                                      block_prefix='tr_up_%i' % block_idx)

            # concatenate the skip connection with the transition block
            x = concatenate([t, skip_list[block_idx]], axis=concat_axis)

            # Dont allow the feature map size to grow in upsampling dense blocks
            block_layer_index = nb_dense_block + 1 + block_idx
            block_prefix = 'dense_%i' % (block_layer_index)
            x_up, nb_filter, concat_list = __dense_block(dimension, x,
                                                         nb_layers[block_layer_index],
                                                         nb_filter=growth_rate,
                                                         growth_rate=growth_rate,
                                                         dropout_rate=dropout_rate,
                                                         weight_decay=weight_decay,
                                                         return_concat_list=True,
                                                         grow_nb_filters=False,
                                                         block_prefix=block_prefix)

        if early_transition:
            x_up = __transition_up_block(dimension, x_up, nb_filters=nb_filter,
                                         type=upsampling_type,
                                         weight_decay=weight_decay,
                                         block_prefix='tr_up_early')
        if include_top:
            if dimension == 2:
                x = Conv2D(num_classes, (1, 1), activation='linear', padding='same',
                           use_bias=False)(x_up)
            else:
                x = Conv3D(num_classes, (1, 1, 1), activation='linear', padding='same',
                           use_bias=False)(x_up)
            output_temp = x
        else:
            output_temp = x_up

        output = organise_output(output_temp, output_shape, activation)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            input = get_source_inputs(input_tensor)
        else:
            input = img_input

        print(input.shape)
        print(output.shape)

        return Model(inputs=[input], outputs=[output], name='fcn-densenet')


def organise_output(input, output_shape, activation) :
    pred = Reshape(output_shape)(input)
    pred = Permute((2, 1))(pred)
    return Activation(activation)(pred)


def name_or_none(prefix, name):
    return prefix + name if (prefix is not None and name is not None) else None


def __conv_block(dimension, ip, nb_filter, bottleneck=False, dropout_rate=None,
                 weight_decay=1e-4, block_prefix=None):
    '''
    Adds a convolution layer (with batch normalization and relu),
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
        x = Activation('relu')(x)

        if bottleneck:
            inter_channel = nb_filter * 4

            if dimension == 2:
                x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal',
                           padding='same', use_bias=False,
                           kernel_regularizer=l2(weight_decay),
                           name=name_or_none(block_prefix, '_bottleneck_conv2D'))(x)
            else:
                x = Conv3D(inter_channel, (1, 1, 1), kernel_initializer='he_normal',
                           padding='same', use_bias=False,
                           kernel_regularizer=l2(weight_decay),
                           name=name_or_none(block_prefix, '_bottleneck_conv3D'))(x)
            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                   name=name_or_none(block_prefix, '_bottleneck_bn'))(x)
            x = Activation('relu')(x)

        if dimension == 2:
            x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same',
                       use_bias=False, name=name_or_none(block_prefix, '_conv2D'))(x)
        else:
            x = Conv3D(nb_filter, (3, 3, 3), kernel_initializer='he_normal', padding='same',
                       use_bias=False, name=name_or_none(block_prefix, '_conv3D'))(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

    return x


def __dense_block(dimension, x, nb_layers, nb_filter, growth_rate, bottleneck=False,
                  dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True,
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
            cb = __conv_block(dimension, x, growth_rate, bottleneck, dropout_rate, weight_decay,
                              block_prefix=name_or_none(block_prefix, '_%i' % i))
            x_list.append(cb)

            x = concatenate([x, cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter


def __transition_block(dimension, ip, nb_filter, compression=1.0, weight_decay=1e-4,
                       block_prefix=None, transition_pooling='max'):
    '''
    Adds a pointwise convolution layer (with batch normalization and relu),
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
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                               name=name_or_none(block_prefix, '_bn'))(ip)
        x = Activation('relu')(x)

        if dimension == 2:
            x = Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='he_normal',
                       padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                       name=name_or_none(block_prefix, '_conv2D'))(x)
            if transition_pooling == 'avg':
                x = AveragePooling2D((2, 2), strides=(2, 2))(x)
            elif transition_pooling == 'max':
                x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        else:
            x = Conv3D(int(nb_filter * compression), (1, 1, 1), kernel_initializer='he_normal',
                       padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                       name=name_or_none(block_prefix, '_conv3D'))(x)
            if transition_pooling == 'avg':
                x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
            elif transition_pooling == 'max':
                x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(x)

        return x


def __transition_up_block(dimension, ip, nb_filters, type='deconv', weight_decay=1E-4,
                          block_prefix=None):
    '''Adds an upsampling block. Upsampling operation relies on the the type parameter.
    # Arguments
        ip: input keras tensor
        nb_filters: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        type: can be 'upsampling', 'subpixel', 'deconv'. Determines
            type of upsampling performed
        weight_decay: weight decay factor
        block_prefix: str, for block unique naming
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, rows * 2, cols * 2)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows * 2, cols * 2, nb_filter)` if data_format='channels_last'.
    # Returns
        a keras tensor
    '''
    with K.name_scope('TransitionUp'):

        if dimension == 2:
            if type == 'upsampling':
                x = UpSampling2D(name=name_or_none(block_prefix, '_upsampling'))(ip)
            elif type == 'subpixel':
                x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=l2(weight_decay), use_bias=False,
                           kernel_initializer='he_normal',
                           name=name_or_none(block_prefix, '_conv2D'))(ip)
                x = SubPixelUpscaling(scale_factor=2,
                                      name=name_or_none(block_prefix, '_subpixel'))(x)
                x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=l2(weight_decay), use_bias=False,
                           kernel_initializer='he_normal',
                           name=name_or_none(block_prefix, '_conv2D'))(x)
            else:
                x = Conv2DTranspose(nb_filters, (3, 3), activation='relu', padding='same',
                                    strides=(2, 2), kernel_initializer='he_normal',
                                    kernel_regularizer=l2(weight_decay),
                                    name=name_or_none(block_prefix, '_conv2DT'))(ip)

        else:
            if type == 'upsampling':
                x = UpSampling3D(name=name_or_none(block_prefix, '_upsampling'))(ip)
            elif type == 'subpixel':
                x = Conv3D(nb_filters, (3, 3, 3), activation='relu', padding='same',
                           kernel_regularizer=l2(weight_decay), use_bias=False,
                           kernel_initializer='he_normal',
                           name=name_or_none(block_prefix, '_conv3D'))(ip)
                x = SubPixelUpscaling(scale_factor=2,
                                      name=name_or_none(block_prefix, '_subpixel'))(x)
                x = Conv3D(nb_filters, (3, 3, 3), activation='relu', padding='same',
                           kernel_regularizer=l2(weight_decay), use_bias=False,
                           kernel_initializer='he_normal',
                           name=name_or_none(block_prefix, '_conv3D'))(x)
            else:
                x = Conv3DTranspose(nb_filters, (3, 3, 3), activation='relu', padding='same',
                                    strides=(2, 2, 2), kernel_initializer='he_normal',
                                    kernel_regularizer=l2(weight_decay),
                                    name=name_or_none(block_prefix, '_conv3DT'))(ip)

        return x