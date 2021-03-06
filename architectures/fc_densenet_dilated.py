import numpy as np
from keras import backend as K

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Reshape, Conv2D, Conv3D, Conv2DTranspose, \
    Conv3DTranspose, UpSampling2D, UpSampling3D, MaxPooling2D, MaxPooling3D, AveragePooling2D, AveragePooling3D, \
    GlobalMaxPooling2D, GlobalMaxPooling3D, GlobalAveragePooling2D, GlobalAveragePooling3D, concatenate, \
    BatchNormalization, Activation, Add, Multiply, Cropping2D, Cropping3D, Lambda
from keras.regularizers import l2
from keras.layers.core import Permute
from keras.engine.topology import get_source_inputs
from keras_contrib.layers import SubPixelUpscaling
from keras.initializers import he_normal
from keras.utils import multi_gpu_model
from utils import loss_functions, metrics, optimizers_builtin
from .squeeze_excitation import spatial_and_channel_squeeze_excite_block2D, spatial_and_channel_squeeze_excite_block3D
from .attention_gates_dilated import grid_attention
from .attention_gates import _concatenation

K.set_image_dim_ordering('th')

# Ref.
# Jegou et al., CVPRW 17, "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation"
# Kamnitsas et al., MedIA17, "Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation"
# Chen et al., ECCV18, "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"

def generate_fc_densenet_dilated(gen_conf, train_conf):

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
    activation = train_conf['activation']
    dimension = train_conf['dimension']
    num_classes = gen_conf['num_classes']
    multi_output = gen_conf['multi_output']
    output_name = gen_conf['output_name']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    expected_output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']

    attention_loss = train_conf['attention_loss']
    overlap_penalty_loss = train_conf['overlap_penalty_loss']
    loss_opt = train_conf['loss']
    metric_opt = train_conf['metric']
    lamda = train_conf['lamda']
    optimizer = train_conf['optimizer']
    initial_lr = train_conf['initial_lr']
    is_set_random_seed = train_conf['is_set_random_seed']
    random_seed_num = train_conf['random_seed_num']
    exclusive_train = train_conf['exclusive_train']

    if is_set_random_seed == 1:
        kernel_init = he_normal(seed=random_seed_num)
    else:
        kernel_init = he_normal()
    # # dentate
    # nb_dense_block = 2 # for dentate, 3 #for thalamus
    # nb_layers_per_block = (4, 5, 6) # for dentate [3,4,5,6] #for thalamus #[3,4,5,6] # 4 #
    # growth_rate = 16 # 8 for thalamus
    # dropout_rate = 0.2

    # nb_dense_block = 2  # for dentate 3 #for thalamus
    # nb_layers_per_block = (3, 4, 5) #[3,4,5,6] # for dentate[3,4,5,6] #for thalamus #[3,4,5,6] # 4 #

    nb_dense_block = 3  # for dentate 3 #for thalamus
    nb_layers_per_block = (3, 4, 5, 6) #[3,4,5,6] # for dentate[3,4,5,6] #for thalamus #[3,4,5,6] # 4 #
    growth_rate = 8
    dropout_rate = 0.2 #None, #0.2

    global_path = True # dilated db path or db path
    dilated_db_fp_input = True # feature pyramid input in dilated db path
    local_path = False # only db path
    db_fp_input = False #True   # feature pyramid input in db path

    # dilated denseblock
    low_res_input = False #False # True: avg_pooling of globalsegmen encoder input for thalamus case
    is_dilated_init_conv = False # dilated convolution in initial convolution block
    is_dilated_conv = True #True

    # atrous spatial pyramid pooling
    is_aspp = False # use atrous spatial pyramid pooling (in deeplab v3+) instead of bottleneck block

    if low_res_input:
        dilation_rate_per_block = [(1, 1, 1), (1, 1, 1, 2), (1, 1, 1, 2, 4)]
        #dilation_rate_per_block = [(1, 1, 1), (2, 2, 2, 2), (4, 4, 4, 4, 4)]
    else:
        dilation_rate_per_block = [(1, 1, 2), (1, 1, 2, 4), (1, 1, 2, 4, 8)]
        #dilation_rate_per_block = [(2, 2, 2), (4, 4, 4, 4), (8, 8, 8, 8, 8)]

    # attention gate in the skip-connection
    attn_gates = True # True
    attn_gates_org_ver = False
    attn_dilation_rate_lst = (1, 1, 1) # for nth scale

    # Global-local attention module
    glam = False #False #True
    glam_arrange_type = 'two_way_sequential' #'two_way_sequential' #concurrent_scSE
    glam_input_short_cut = True #True #False
    glam_final_conv = False # False: addition # True: applying fully connected conv layer (1x1x1) after concatenation instead of adding up
    glam_position = 'before_shortcut' #before_shortcut #after_shortcut

    reduction = 0.0
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

    input_shape = (num_modality, ) + patch_shape

    if multi_output == 1:
        if attention_loss == 1:
            output_shape = [(num_classes[0], np.prod(expected_output_shape)), (num_classes[1], np.prod(expected_output_shape)),
                            (2, np.prod(expected_output_shape))]
        else:
            output_shape = [(num_classes[0], np.prod(expected_output_shape)), (num_classes[1], np.prod(expected_output_shape))]
    else:
        if attention_loss == 1:
            output_shape = [(num_classes, np.prod(expected_output_shape)), (2, np.prod(expected_output_shape))]
        else:
            output_shape = (num_classes, np.prod(expected_output_shape))

    if exclusive_train == 1 and multi_output == 0:
        num_classes -= 1

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
                         'Argument value was %s.' % nb_layers_per_block)

    if multi_output == 0:
        if activation not in ['softmax', 'sigmoid']:
            raise ValueError('activation must be one of "softmax" or "sigmoid"')

    # if activation == 'sigmoid' and num_classes != 1:
    #     raise ValueError('sigmoid activation can only be used when classes = 1')

    assert dimension in [2, 3]

    model = __generate_fc_densenet_ms(num_of_gpu, dimension, num_classes, multi_output, output_name, attention_loss,
                                      overlap_penalty_loss, lamda, kernel_init, random_seed_num, input_shape, output_shape,
                                      expected_output_shape, input_tensor, nb_dense_block, nb_layers_per_block,
                                      growth_rate, reduction, loss_opt, metric_opt, optimizer, initial_lr, dropout_rate,
                                      weight_decay, init_conv_filters, include_top, upsampling_type, activation,
                                      early_transition, transition_pooling, initial_kernel_size, is_aspp, global_path,
                                      local_path, dilated_db_fp_input, db_fp_input, low_res_input, is_dilated_init_conv,
                                      is_dilated_conv, dilation_rate_per_block, attn_gates, attn_gates_org_ver,
                                      attn_dilation_rate_lst, glam, glam_arrange_type, glam_input_short_cut,
                                      glam_final_conv, glam_position)

    # model.summary()
    # model = multi_gpu_model(model, gpus=num_of_gpu)
    # model.compile(loss=loss_functions.select(num_classes, loss_opt), optimizer=optimizer,
    #               metrics=metrics.select(metric_opt))

    return model


def __generate_fc_densenet_ms(num_of_gpu, dimension, num_classes, multi_output, output_name, attention_loss, overlap_penalty_loss,
                              lamda, kernel_init, random_seed_num, input_shape=None, output_shape=None,
                              expected_output_shape = None, input_tensor=None, nb_dense_block=3,
                              nb_layers_per_block=(3, 4, 5, 6), growth_rate=12, reduction=0.0, loss_opt='Dice',
                              metric_opt='acc', optimizer='Adam', initial_lr=0.001, dropout_rate=None,
                              weight_decay=1e-4, init_conv_filters=48, include_top=None, upsampling_type='deconv',
                              activation='softmax', early_transition=False, transition_pooling='max',
                              initial_kernel_size=(3, 3, 3), is_aspp=False, global_path=True, local_path=True,
                              dilated_db_fp_input=True, db_fp_input=True, low_res_input=True, is_dilated_init_conv=True,
                              is_dilated_conv=True, dilation_rate_per_block=(2, 4, 8), attn_gates=True,
                              attn_gates_org_ver=False, attn_dilation_rate_lst=(8, 4, 2), glam=True,
                              glam_arrange_type='two_way_sequential', glam_input_short_cut=True, glam_final_conv=False,
                              glam_position='before_shortcut'):

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

        if global_path and local_path:
            # img_input 64x64x64 patches for thalamus, 32x32x32 patches for dentate
            if input_shape[2] != expected_output_shape[2]:
                input_crop_size = np.int(img_input._keras_shape[2] / 4)
                local_input = get_cropping_layer(dimension, img_input,
                                                 crop_size=(input_crop_size, input_crop_size))  # local features
            else:
                local_input = img_input
            if low_res_input:
                global_input = get_low_res_layer(dimension, img_input, type='average') # global contextual features
            else:
                global_input = img_input
        elif not global_path and local_path:
            local_input = img_input
            global_input = None
        elif global_path and not local_path:
            local_input = None
            if low_res_input:
                global_input = get_low_res_layer(dimension, img_input, type='average') # global contextual features
            else:
                global_input = img_input
        else:
            raise NotImplementedError('One of the paths should exist')


        # local feature encoder
        if local_path:
            local_f, nb_filter, block_prefix, skip_list_local = conv_dense_transition_down(dimension,
                                                                                           local_input,
                                                                                           init_conv_filters,
                                                                                           initial_kernel_size,
                                                                                           kernel_init,
                                                                                           random_seed_num,
                                                                                           weight_decay,
                                                                                           concat_axis,
                                                                                           early_transition,
                                                                                           compression,
                                                                                           transition_pooling,
                                                                                           nb_dense_block,
                                                                                           nb_layers,
                                                                                           growth_rate,
                                                                                           dropout_rate,
                                                                                           dilated_db_fp_input,
                                                                                           db_fp_input,
                                                                                           False,
                                                                                           False,
                                                                                           dilation_rate_per_block,
                                                                                           glam,
                                                                                           glam_arrange_type,
                                                                                           glam_input_short_cut,
                                                                                           glam_final_conv,
                                                                                           glam_position,
                                                                                           block_prefix_num=0)
        else:
            local_f = None
            skip_list_local = None

        # global feature encoder
        if global_path:
            global_f, nb_filter, block_prefix, skip_list_global = conv_dense_transition_down(dimension,
                                                                                              global_input,
                                                                                              init_conv_filters,
                                                                                              initial_kernel_size,
                                                                                              kernel_init,
                                                                                              random_seed_num,
                                                                                              weight_decay,
                                                                                              concat_axis,
                                                                                              early_transition,
                                                                                              compression,
                                                                                              transition_pooling,
                                                                                              nb_dense_block,
                                                                                              nb_layers,
                                                                                              growth_rate,
                                                                                              dropout_rate,
                                                                                              dilated_db_fp_input,
                                                                                              db_fp_input,
                                                                                              is_dilated_init_conv,
                                                                                              is_dilated_conv,
                                                                                              dilation_rate_per_block,
                                                                                              glam,
                                                                                              glam_arrange_type,
                                                                                              glam_input_short_cut,
                                                                                              glam_final_conv,
                                                                                              glam_position,
                                                                                              block_prefix_num=1)
        else:
            global_f = None
            skip_list_global = None

        if is_aspp:
            bottleneck_out = ASPP(dimension, local_f, global_f, weight_decay, nb_filter, kernel_init, random_seed_num,
                                  growth_rate, bottleneck_nb_layers, dropout_rate, block_prefix, global_path, local_path,
                                  low_res_input)
        else:
            # bottleneck block
            bottleneck_out = bottleneck_block(dimension, local_f, global_f, weight_decay, concat_axis,
                                              bottleneck_nb_layers, nb_filter, kernel_init, random_seed_num,
                                              growth_rate, dropout_rate, block_prefix, global_path, local_path,
                                              low_res_input, glam, glam_arrange_type, glam_input_short_cut,
                                              glam_final_conv)

        # decoder
        skip_list = [skip_list_local, skip_list_global]
        output_temp, c_score_up = transition_up_dense_conv(dimension, multi_output, bottleneck_out, weight_decay,
                                                           concat_axis, early_transition, nb_dense_block, nb_layers,
                                                           kernel_init, random_seed_num, growth_rate, upsampling_type,
                                                           skip_list, dropout_rate, include_top, num_classes,
                                                           global_path, local_path, low_res_input, attn_gates,
                                                           attn_gates_org_ver, attn_dilation_rate_lst, glam,
                                                           glam_arrange_type, glam_input_short_cut, glam_final_conv,
                                                           glam_position)


        output = organise_output(dimension, kernel_init, concat_axis, output_temp, c_score_up, output_shape,
                                 activation, output_name, multi_output, attention_loss)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            input = get_source_inputs(input_tensor)
        else:
            input = img_input

        if multi_output == 1:
            if overlap_penalty_loss == 1:
                dentate_prob_output = Lambda(lambda x: x[:, :, num_classes[0]-1])(output[0])
                interposed_prob_output = Lambda(lambda x: x[:, :, num_classes[1]-1])(output[1])
                dentate_interposed_prob_output = Multiply(name='overlap_dentate_interposed')([dentate_prob_output,
                                                                                              interposed_prob_output])
                output.append(dentate_interposed_prob_output)

        print(input.shape)
        if multi_output == 1:
            for o in output:
                print(o.shape)
        else:
            if attention_loss == 1:
                for o in output:
                    print(o.shape)
            else:
                print(output[0].shape)

        model = Model(inputs=input, outputs=output, name='fc_densenet_dilated')
        model.summary()

        if multi_output == 1:
            if attention_loss == 1:
                if overlap_penalty_loss == 1:
                    loss_multi = {
                        output_name[0]: loss_functions.select(num_classes[0], loss_opt[0]),
                        output_name[1]: loss_functions.select(num_classes[1], loss_opt[1]),
                        'attention_maps': 'categorical_crossentropy',
                        'overlap_dentate_interposed': loss_functions.dc_btw_dentate_interposed(dentate_prob_output,
                                                                                   interposed_prob_output)
                    }
                    loss_weights = {output_name[0]: lamda[0], output_name[1]: lamda[1], 'attention_maps': lamda[2],
                                    'overlap_dentate_interposed': lamda[3]}
                else:
                    loss_multi = {
                        output_name[0]: loss_functions.select(num_classes[0], loss_opt[0]),
                        output_name[1]: loss_functions.select(num_classes[1], loss_opt[1]),
                        'attention_maps': 'categorical_crossentropy'
                    }
                    loss_weights = {output_name[0]: lamda[0], output_name[1]: lamda[1], 'attention_maps': lamda[2]}
            else:
                if overlap_penalty_loss == 1:
                    loss_multi = {
                        output_name[0]: loss_functions.select(num_classes[0], loss_opt[0]),
                        output_name[1]: loss_functions.select(num_classes[1], loss_opt[1]),
                        'overlap_dentate_interposed': loss_functions.dc_btw_dentate_interposed(dentate_prob_output,
                                                                                   interposed_prob_output)
                    }
                    loss_weights = {output_name[0]: lamda[0], output_name[1]: lamda[1], 'overlap_dentate_interposed':
                        lamda[3]}
                else:
                    loss_multi = {
                        output_name[0]: loss_functions.select(num_classes[0], loss_opt[0]),
                        output_name[1]: loss_functions.select(num_classes[1], loss_opt[1])
                    }
                    loss_weights = {output_name[0]: lamda[0], output_name[1]: lamda[1]}
        else:
            if attention_loss == 1:
                loss_multi = {
                    output_name: loss_functions.select(num_classes, loss_opt),
                    'attention_maps': 'categorical_crossentropy'
                }
                loss_weights = {output_name: lamda[0], 'attention_maps': lamda[2]}
            else:
                loss_multi = loss_functions.select(num_classes, loss_opt)
                loss_weights = None
        if num_of_gpu > 1:
            model = multi_gpu_model(model, gpus=num_of_gpu)
        model.compile(loss=loss_multi, loss_weights=loss_weights,
                      optimizer=optimizers_builtin.select(optimizer, initial_lr),
                      metrics=metrics.select(metric_opt))

        return model


def organise_output(dimension, kernel_init, concat_axis, input, c_score_up, output_shape, activation, output_name,
                    multi_output, attention_loss):
    output = []
    if multi_output == 1:
        for i, p, a, name in zip(input, output_shape, activation, output_name):
            pred = Reshape(p)(i)
            pred = Permute((2, 1))(pred)
            output.append(Activation(a, name=name)(pred))
    else:
        if attention_loss == 1:
            pred = Reshape(output_shape[0])(input)
        else:
            pred = Reshape(output_shape)(input)
        pred = Permute((2, 1))(pred)
        output.append(Activation(activation, name=output_name)(pred))

    if attention_loss == 1:
        c_concate = concatenate(c_score_up, axis=concat_axis)
        if dimension == 2:
            c_concate_fc = Conv2D(2, (1, 1), padding='same', kernel_initializer=kernel_init)(c_concate)
        elif dimension == 3:
            c_concate_fc = Conv3D(2, (1, 1, 1), padding='same', kernel_initializer=kernel_init)(c_concate)
        else:
            raise NotImplemented

        c_reshape = Reshape(output_shape[-1])(c_concate_fc)
        c_reshape = Permute((2, 1))(c_reshape)
        output.append(Activation('softmax', name='attention_maps')(c_reshape))

    return output


def conv_dense_transition_down(dimension, img_input, init_conv_filters, initial_kernel_size, kernel_init,
                               random_seed_num, weight_decay, concat_axis, early_transition, compression,
                               transition_pooling, nb_dense_block, nb_layers, growth_rate, dropout_rate,
                               dilated_db_fp_input, db_fp_input, is_dilated_init_conv, is_dilated_conv,
                               dilation_rate_per_block, glam, glam_arrange_type, glam_input_short_cut, glam_final_conv,
                               glam_position, block_prefix_num):

    # Initial convolution
    if dimension == 2:
        if is_dilated_init_conv:
            x = Conv2D(init_conv_filters, initial_kernel_size, dilation_rate=(2, 2),
                       kernel_initializer=kernel_init, padding='same',
                       name='initial_conv2D_%i' % block_prefix_num, use_bias=False,
                       kernel_regularizer=l2(weight_decay))(img_input)
        else:
            x = Conv2D(init_conv_filters, initial_kernel_size,
                       kernel_initializer=kernel_init, padding='same',
                       name='initial_conv2D_%i' % block_prefix_num, use_bias=False,
                       kernel_regularizer=l2(weight_decay))(img_input)
    else:
        if is_dilated_init_conv:
            x = Conv3D(init_conv_filters, initial_kernel_size, dilation_rate=(2, 2, 2),
                       kernel_initializer=kernel_init, padding='same',
                       name='initial_conv3D_%i' % block_prefix_num, use_bias=False,
                       kernel_regularizer=l2(weight_decay))(img_input)
        else:
            x = Conv3D(init_conv_filters, initial_kernel_size,
                       kernel_initializer=kernel_init, padding='same',
                       name='initial_conv3D_%i' % block_prefix_num, use_bias=False,
                       kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5, name='initial_bn_%i' % block_prefix_num)(x)
    x = Activation('elu')(x)

    nb_filter = init_conv_filters

    skip_list = []

    if early_transition:
        if not is_dilated_conv:
            x = __transition_down_block(dimension, x, nb_filter, kernel_init, random_seed_num, compression=compression,
                                        weight_decay=weight_decay, dropout_rate=dropout_rate,
                                        block_prefix='tr_early_%i' % block_prefix_num, transition_pooling=transition_pooling)

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
            x, nb_filter, concat_list = __dilated_dense_block(dimension, x, kernel_size, kernel_init, random_seed_num,
                                                              nb_layers[block_idx], dilation_rate_lst, nb_filter,
                                                              growth_rate, dropout_rate=dropout_rate,
                                                              weight_decay=weight_decay, return_concat_list=True,
                                                              block_prefix='dilated_dense_%i_%i' % (block_idx,
                                                                                                    block_prefix_num))
        else:
            dilation_rate = (1, 1) if dimension == 2 else (1, 1, 1)
            x, nb_filter, concat_list = __dense_block(dimension, x, kernel_size, kernel_init, random_seed_num,
                                                      nb_layers[block_idx], dilation_rate, nb_filter, growth_rate,
                                                      dropout_rate=dropout_rate, weight_decay=weight_decay,
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
        if is_dilated_conv:
            p = pow(2, block_idx)
            pooling_size = (p, p) if dimension == 2 else (p, p, p)
            if dimension == 2:
                x_pool = MaxPooling2D(pooling_size)(x)
            else:
                x_pool = MaxPooling3D(pooling_size)(x)

            if dilated_db_fp_input and block_idx != 0:
                channel_axis = 1 if K.image_data_format() == "channels_first" else -1
                ch_x_pool = x_pool._keras_shape[channel_axis]
                img_input, img_input_activation = multi_scale_input(dimension, img_input, ch_x_pool, initial_kernel_size,
                                                                    kernel_init, block_idx, block_prefix_num,
                                                                    weight_decay, concat_axis)
                skip_list.append(concatenate([x_pool, img_input_activation], axis=concat_axis))
            else:
                # Skip connection
                skip_list.append(x_pool)

            if block_idx == nb_dense_block-1:
                p = pow(2, block_idx + 1)
                pooling_size = (p, p) if dimension == 2 else (p, p, p)
                if dimension == 2:
                    x = MaxPooling2D(pooling_size)(x)
                else:
                    x = MaxPooling3D(pooling_size)(x)
                if dilated_db_fp_input:
                    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
                    ch_x_pool = x._keras_shape[channel_axis]
                    img_input, img_input_activation = multi_scale_input(dimension, img_input, ch_x_pool,
                                                                        initial_kernel_size, kernel_init,
                                                                        block_idx + 1, block_prefix_num, weight_decay,
                                                                        concat_axis)
                    x = concatenate([x, img_input_activation], axis=concat_axis)

        else:
            # Skip connection
            skip_list.append(x)

            # add transition_block
            x = __transition_down_block(dimension, x, nb_filter, kernel_init, random_seed_num, compression=compression,
                                   weight_decay=weight_decay, dropout_rate=dropout_rate,
                                   block_prefix='tr_%i_%i' % (block_idx, block_prefix_num),
                                   transition_pooling=transition_pooling)

        if db_fp_input and not is_dilated_conv:

            img_input, img_input_activation = multi_scale_input(dimension, img_input, nb_filter, initial_kernel_size,
                                                                kernel_init, block_idx, block_prefix_num, weight_decay,
                                                                concat_axis)
            x = concatenate([x, img_input_activation], axis=concat_axis)
            nb_filter += nb_filter

        # this is calculated inside transition_down_block
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_down_block
    # return the concatenated feature maps without the concatenation of the input
    block_prefix = 'dense_%i_%i' % (nb_dense_block, block_prefix_num)

    skip_list = skip_list[::-1]  # reverse the skip list

    return x, nb_filter, block_prefix, skip_list


def bottleneck_block(dimension, local_f, global_f, weight_decay, concat_axis, bottleneck_nb_layers, nb_filter,
                     kernel_init, random_seed_num, growth_rate, dropout_rate, block_prefix, global_path, local_path,
                     low_res_input, glam, glam_arrange_type, glam_input_short_cut, glam_final_conv):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    #input_channels = x_local._keras_shape[channel_axis]

    if global_path and local_path:
        db_input = __transition_block(dimension, local_f, global_f, nb_filter, kernel_init, random_seed_num,
                                      low_res_input, weight_decay=weight_decay, dropout_rate=dropout_rate,
                                      block_prefix='bottleneck_input')
        bottleneck = True
    elif not global_path and local_path:
        db_input = local_f
        bottleneck = False
    elif global_path and not local_path:
        if low_res_input:
            db_input = __transition_block(dimension, local_f, global_f, nb_filter, kernel_init, random_seed_num,
                                          low_res_input, weight_decay=weight_decay,
                                          dropout_rate=dropout_rate, block_prefix='bottleneck_input')
        else:
            db_input = global_f
        bottleneck = False
    else:
        raise NotImplementedError('One of the paths should exist')

    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    _, _, concat_list = __dense_block(dimension, db_input, kernel_size, kernel_init, random_seed_num,
                                      bottleneck_nb_layers, (1, 1, 1), nb_filter, growth_rate, bottleneck=bottleneck,
                                      dropout_rate=dropout_rate, weight_decay=weight_decay, return_concat_list=True,
                                      block_prefix=block_prefix)

    l = concatenate(concat_list[1:], axis=concat_axis)
    if glam:
        if dimension == 2:
            l = spatial_and_channel_squeeze_excite_block2D(l, kernel_init, arrange_type=glam_arrange_type,
                                                           input_short_cut=glam_input_short_cut,
                                                           final_conv=glam_final_conv)
        else:
            l = spatial_and_channel_squeeze_excite_block3D(l, kernel_init, arrange_type=glam_arrange_type,
                                                           input_short_cut=glam_input_short_cut,
                                                           final_conv=glam_final_conv)

    return l


def ASPP(dimension, local_f, global_f, weight_decay, nb_filter, kernel_init, random_seed_num, growth_rate,
         bottleneck_nb_layers, dropout_rate, block_prefix, global_path, local_path, low_res_input):

    if global_path and local_path:
        db_input = __transition_block(dimension, local_f, global_f, nb_filter, kernel_init, random_seed_num,
                                      low_res_input, weight_decay=weight_decay, dropout_rate=dropout_rate,
                                      block_prefix='ASPP_input')
    elif not global_path and local_path:
        db_input = local_f
    elif global_path and not local_path:
        if low_res_input:
            db_input = __transition_block(dimension, local_f, global_f, nb_filter, kernel_init, random_seed_num,
                                          low_res_input, weight_decay=weight_decay, dropout_rate=dropout_rate,
                                          block_prefix='ASPP_input')
        else:
            db_input = global_f
    else:
        raise NotImplementedError('One of the paths should exist')

    with K.name_scope('ASPP'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
        kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)

        if dimension == 2:
            x1 = Conv2D(nb_filter, (1, 1), kernel_initializer=kernel_init, padding='same',
                       use_bias=False, dilation_rate=(1, 1), name=name_or_none(block_prefix, '_conv2D_1'))(db_input)
            x2 = Conv2D(nb_filter, kernel_size, kernel_initializer=kernel_init, padding='same',
                       use_bias=False, dilation_rate=(2, 2), name=name_or_none(block_prefix, '_conv2D_2'))(db_input)
            x3 = Conv2D(nb_filter, kernel_size, kernel_initializer=kernel_init, padding='same',
                       use_bias=False, dilation_rate=(4, 4), name=name_or_none(block_prefix, '_conv2D_3'))(db_input)
        else:
            x1 = Conv3D(nb_filter, (1, 1, 1), kernel_initializer=kernel_init, padding='same',
                       use_bias=False, dilation_rate=(1, 1, 1), name=name_or_none(block_prefix, '_conv3D_1'))(db_input)
            x2 = Conv3D(nb_filter, kernel_size, kernel_initializer=kernel_init, padding='same',
                       use_bias=False, dilation_rate=(2, 2, 2), name=name_or_none(block_prefix, '_conv3D_2'))(db_input)
            x3 = Conv3D(nb_filter, kernel_size, kernel_initializer=kernel_init, padding='same',
                       use_bias=False, dilation_rate=(4, 4, 4), name=name_or_none(block_prefix, '_conv3D_3'))(db_input)

        concat = concatenate([x1, x2, x3, db_input], axis=concat_axis)

        out_channel = growth_rate * bottleneck_nb_layers
        if dimension == 2:
            out = Conv2D(out_channel, (1, 1), kernel_initializer=kernel_init, padding='same',
                       use_bias=False, dilation_rate=(1, 1, 1), kernel_regularizer=l2(weight_decay),
                         name=name_or_none(block_prefix, '_conv2D_4'))(concat)
        else:
            out = Conv3D(out_channel, (1, 1, 1), kernel_initializer=kernel_init, padding='same',
                       use_bias=False, dilation_rate=(1, 1, 1), kernel_regularizer=l2(weight_decay),
                        name=name_or_none(block_prefix, '_conv3D_4'))(concat)

    return out


def transition_up_dense_conv(dimension, multi_output, l, weight_decay, concat_axis, early_transition, nb_dense_block,
                             nb_layers, kernel_init, random_seed_num, growth_rate, upsampling_type, skip_list,
                             dropout_rate, include_top, num_classes, global_path, local_path, low_res_input, attn_gates,
                             attn_gates_org_ver, attn_dilation_rate_lst, glam, glam_arrange_type, glam_input_short_cut,
                             glam_final_conv, glam_position):

    # Add dense blocks and transition up block
    compatibility_score_up_lst = []
    for block_idx in range(nb_dense_block):
        n_filters_keep = growth_rate * nb_layers[nb_dense_block + block_idx]

        # upsampling block must upsample only the feature maps (concat_list[1:]),
        # not the concatenation of the input with the feature maps (concat_list[0].
        # if block_idx == 0:
        #     l = concatenate(concat_list[1:], axis=concat_axis)

        t = __transition_up_block(dimension, l, kernel_init, nb_filters=n_filters_keep, type=upsampling_type,
                                  weight_decay=weight_decay, block_prefix='tr_up_%i' % block_idx, stride=2)

        if global_path and local_path:
            skip_connection = __transition_block(dimension, skip_list[0][block_idx], skip_list[1][block_idx],
                                                 n_filters_keep, kernel_init, random_seed_num, low_res_input,
                                                 weight_decay=weight_decay, dropout_rate=dropout_rate,
                                                 block_prefix='skip_tr_%i' % block_idx)
            bottleneck = True
        elif not global_path and local_path:
            skip_connection = skip_list[0][block_idx]
            bottleneck = False

        elif global_path and not local_path:
            if low_res_input:
                skip_connection = __transition_block(dimension, None, skip_list[1][block_idx], n_filters_keep,
                                                     kernel_init, random_seed_num, low_res_input,
                                                     weight_decay=weight_decay, dropout_rate=dropout_rate,
                                                     block_prefix='skip_tr_%i' % block_idx)

            else:
                skip_connection = skip_list[1][block_idx]
            bottleneck = False
        else:
            raise NotImplementedError('One of the paths should exist')

        if attn_gates:
            if attn_gates_org_ver:
                attn_gates_kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)
                attn_gates_sub_sample_size = (2, 2) if dimension == 2 else (2, 2, 2)
                skip_connection, compatibility_score = _concatenation(dimension, skip_connection, l, n_filters_keep, n_filters_keep,
                                                    attn_gates_kernel_size, attn_gates_sub_sample_size,
                                                    attn_gates_sub_sample_size, 'elu', kernel_init)
            else:
                skip_connection, compatibility_score = grid_attention(dimension, skip_connection, l,
                                                                      attn_dilation_rate_lst, nb_dense_block - block_idx,
                                                                      n_filters_keep, kernel_init, weight_decay=weight_decay)

            s = pow(2, (nb_dense_block-1)-block_idx)
            sub_sample_factor = (s, s) if dimension == 2 else (s, s, s)
            if dimension == 2:
                compatibility_score_up = UpSampling2D(size=sub_sample_factor)(compatibility_score)
            elif dimension == 3:
                compatibility_score_up = UpSampling3D(size=sub_sample_factor)(compatibility_score)
            else:
                raise NotImplemented
            compatibility_score_up_lst.append(compatibility_score_up)
        else:
            if dimension == 2:
                skip_connection = Conv2D(n_filters_keep, (1, 1), kernel_initializer=kernel_init, padding='same',
                                         use_bias=False, kernel_regularizer=l2(weight_decay))(skip_connection)
            elif dimension == 3:
                skip_connection = Conv3D(n_filters_keep, (1, 1, 1), kernel_initializer=kernel_init, padding='same',
                                         use_bias=False, kernel_regularizer=l2(weight_decay))(skip_connection)
            else:
                raise NotImplemented

        db_input = concatenate([t, skip_connection], axis=concat_axis)

        kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
        # Dont allow the feature map size to grow in upsampling dense blocks
        block_layer_index = nb_dense_block + 1 + block_idx
        block_prefix = 'dense_%i' % (block_layer_index)
        x_up, nb_filter, concat_list = __dense_block(dimension,
                                                     db_input,
                                                     kernel_size,
                                                     kernel_init,
                                                     random_seed_num,
                                                     nb_layers[block_layer_index],
                                                     (1, 1, 1),
                                                     nb_filter=growth_rate,
                                                     growth_rate=growth_rate,
                                                     bottleneck=bottleneck,
                                                     dropout_rate=dropout_rate,
                                                     weight_decay=weight_decay,
                                                     return_concat_list=True,
                                                     grow_nb_filters=False,
                                                     block_prefix=block_prefix)

        l = concatenate(concat_list[1:], axis=concat_axis)
        if glam:
            if block_idx < nb_dense_block - 1:
                if dimension == 2:
                    l = spatial_and_channel_squeeze_excite_block2D(l, kernel_init, arrange_type=glam_arrange_type,
                                                                   input_short_cut=glam_input_short_cut,
                                                                   final_conv=glam_final_conv)
                else:
                    l = spatial_and_channel_squeeze_excite_block3D(l, kernel_init, arrange_type=glam_arrange_type,
                                                                   input_short_cut=glam_input_short_cut,
                                                                   final_conv=glam_final_conv)

            else:
                if glam_position == 'before_shortcut':
                    if dimension == 2:
                        l = spatial_and_channel_squeeze_excite_block2D(l, kernel_init, arrange_type=glam_arrange_type,
                                                                       input_short_cut=glam_input_short_cut,
                                                                       final_conv=glam_final_conv)
                    else:
                        l = spatial_and_channel_squeeze_excite_block3D(l, kernel_init, arrange_type=glam_arrange_type,
                                                                       input_short_cut=glam_input_short_cut,
                                                                       final_conv=glam_final_conv)
                    x_up = concatenate([db_input, l], axis=concat_axis)
                else:
                    if dimension == 2:
                        x_up = spatial_and_channel_squeeze_excite_block2D(x_up, kernel_init, arrange_type=glam_arrange_type,
                                                                       input_short_cut=glam_input_short_cut,
                                                                       final_conv=glam_final_conv)
                    else:
                        x_up = spatial_and_channel_squeeze_excite_block3D(x_up, kernel_init, arrange_type=glam_arrange_type,
                                                                       input_short_cut=glam_input_short_cut,
                                                                       final_conv=glam_final_conv)

    if early_transition:
        x_up = __transition_up_block(dimension, x_up, kernel_init, nb_filters=nb_filter, type=upsampling_type,
                                     weight_decay=weight_decay, block_prefix='tr_up_early', stride=2)
    if include_top:
        if dimension == 2:
            if multi_output == 1:
                x1 = Conv2D(num_classes[0], (1, 1), activation='linear', padding='same', use_bias=False,
                            kernel_initializer=kernel_init)(x_up)
                x2 = Conv2D(num_classes[1], (1, 1), activation='linear', padding='same', use_bias=False,
                            kernel_initializer=kernel_init)(x_up)
                x = [x1, x2]
            else:
                x = Conv2D(num_classes, (1, 1), activation='linear', padding='same', use_bias=False,
                           kernel_initializer=kernel_init)(x_up)
        else:
            if multi_output == 1:
                x1 = Conv3D(num_classes[0], (1, 1, 1), activation='linear', padding='same', use_bias=False,
                            kernel_initializer=kernel_init)(x_up)
                x2 = Conv3D(num_classes[1], (1, 1, 1), activation='linear', padding='same', use_bias=False,
                            kernel_initializer=kernel_init)(x_up)
                x = [x1, x2]
            else:
                x = Conv3D(num_classes, (1, 1, 1), activation='linear', padding='same', use_bias=False,
                           kernel_initializer=kernel_init)(x_up)
        output = x
    else:
        output = x_up

    return output, compatibility_score_up_lst


def __conv_block(dimension, ip, kernel_size, kernel_init, random_seed_num, nb_filter, bottleneck=False, dropout_rate=None,
                 weight_decay=1e-4, dilation_rate=(1, 1, 1), block_prefix=None):
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
        x = Activation('elu')(x)

        if bottleneck:
            inter_channel = nb_filter * 4
            if dimension == 2:
                x = Conv2D(inter_channel, (1, 1), kernel_initializer=kernel_init,
                           padding='same', use_bias=False, dilation_rate=dilation_rate,
                           kernel_regularizer=l2(weight_decay),
                           name=name_or_none(block_prefix, '_bottleneck_conv2D'))(x)
            else:
                x = Conv3D(inter_channel, (1, 1, 1), kernel_initializer=kernel_init,
                           padding='same', use_bias=False, dilation_rate=dilation_rate,
                           kernel_regularizer=l2(weight_decay),
                           name=name_or_none(block_prefix, '_bottleneck_conv3D'))(x)
            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                   name=name_or_none(block_prefix, '_bottleneck_bn'))(x)
            x = Activation('elu')(x)

        if dimension == 2:
            x = Conv2D(nb_filter, kernel_size, kernel_initializer=kernel_init, padding='same',
                       use_bias=False, dilation_rate=dilation_rate, name=name_or_none(block_prefix, '_conv2D'))(x)
        else:
            x = Conv3D(nb_filter, kernel_size, kernel_initializer=kernel_init, padding='same',
                       use_bias=False, dilation_rate=dilation_rate, name=name_or_none(block_prefix, '_conv3D'))(x)
        if dropout_rate:
            x = Dropout(dropout_rate, seed=random_seed_num)(x)

    return x


def __dense_block(dimension, x, kernel_size, kernel_init, random_seed_num, nb_layers, dilation_rate, nb_filter,
                  growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True,
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
            cb = __conv_block(dimension, x, kernel_size, kernel_init, random_seed_num, growth_rate, bottleneck,
                              dropout_rate, weight_decay, dilation_rate,
                              block_prefix=name_or_none(block_prefix, '_%i' % i))
            x_list.append(cb)

            x = concatenate([x, cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter


def __dilated_dense_block(dimension, x, kernel_size, kernel_init, random_seed_num, nb_layers, dilation_rate_lst,
                          nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
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
    with K.name_scope('DenseBlock'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x_list = [x]

        for i in range(nb_layers):
            dilation_rate = dilation_rate_lst[i]
            cb = __conv_block(dimension, x, kernel_size, kernel_init, random_seed_num, growth_rate, bottleneck,
                              dropout_rate, weight_decay, dilation_rate,
                              block_prefix=name_or_none(block_prefix, '_%i' % i))
            x_list.append(cb)

            x = concatenate([x, cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter


def __transition_down_block(dimension, ip, nb_filter, kernel_init, random_seed_num, compression=1.0, weight_decay=1e-4,
                            dropout_rate=None, block_prefix=None, transition_pooling='max'):
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
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                               name=name_or_none(block_prefix, '_bn'))(ip)
        x = Activation('elu')(x)

        if dimension == 2:
            x = Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer=kernel_init,
                       padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                       name=name_or_none(block_prefix, '_conv2D'))(x)
            if dropout_rate:
                x = Dropout(dropout_rate, seed=random_seed_num)(x)
            if transition_pooling == 'avg':
                x = AveragePooling2D((2, 2), strides=(2, 2))(x)
            elif transition_pooling == 'max':
                x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        else:
            x = Conv3D(int(nb_filter * compression), (1, 1, 1), kernel_initializer=kernel_init,
                       padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                       name=name_or_none(block_prefix, '_conv3D'))(x)
            if dropout_rate:
                x = Dropout(dropout_rate, seed=random_seed_num)(x)
            if transition_pooling == 'avg':
                x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
            elif transition_pooling == 'max':
                x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(x)

        return x


def __transition_up_block(dimension, ip, kernel_init, nb_filters, type='deconv', weight_decay=1E-4,
                          block_prefix=None, stride=2):
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
        strides = (stride, stride) if dimension == 2 else (stride, stride, stride)

        if dimension == 2:
            if type == 'upsampling':
                x = UpSampling2D(name=name_or_none(block_prefix, '_upsampling'))(ip)
            elif type == 'subpixel':
                x = Conv2D(nb_filters, (3, 3), activation='elu', padding='same',
                           kernel_regularizer=l2(weight_decay), use_bias=False,
                           kernel_initializer=kernel_init,
                           name=name_or_none(block_prefix, '_conv2D'))(ip)
                x = SubPixelUpscaling(scale_factor=2,
                                      name=name_or_none(block_prefix, '_subpixel'))(x)
                x = Conv2D(nb_filters, (3, 3), activation='elu', padding='same',
                           kernel_regularizer=l2(weight_decay), use_bias=False,
                           kernel_initializer=kernel_init,
                           name=name_or_none(block_prefix, '_conv2D'))(x)
            else:
                x = Conv2DTranspose(nb_filters, (3, 3), activation='elu', padding='same',
                                    strides=strides, kernel_initializer=kernel_init,
                                    kernel_regularizer=l2(weight_decay),
                                    name=name_or_none(block_prefix, '_conv2DT'))(ip)

        else:
            if type == 'upsampling':
                x = UpSampling3D(name=name_or_none(block_prefix, '_upsampling'))(ip)
            elif type == 'subpixel':
                x = Conv3D(nb_filters, (3, 3, 3), activation='elu', padding='same',
                           kernel_regularizer=l2(weight_decay), use_bias=False,
                           kernel_initializer=kernel_init,
                           name=name_or_none(block_prefix, '_conv3D'))(ip)
                x = SubPixelUpscaling(scale_factor=2,
                                      name=name_or_none(block_prefix, '_subpixel'))(x)
                x = Conv3D(nb_filters, (3, 3, 3), activation='elu', padding='same',
                           kernel_regularizer=l2(weight_decay), use_bias=False,
                           kernel_initializer=kernel_init,
                           name=name_or_none(block_prefix, '_conv3D'))(x)
            else:
                x = Conv3DTranspose(nb_filters, (3, 3, 3), activation='elu', padding='same',
                                    strides=strides, kernel_initializer=kernel_init,
                                    kernel_regularizer=l2(weight_decay),
                                    name=name_or_none(block_prefix, '_conv3DT'))(ip)

        return x


def __transition_block(dimension, local_f, global_f, nb_filter, kernel_init, random_seed_num, low_res_input,
                       weight_decay=1e-4, dropout_rate=None, block_prefix=None):

    with K.name_scope('Transition'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
        #strides = (2, 2) if dimension == 2 else (2, 2, 2)
        # x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
        #                        name=name_or_none(block_prefix, '_bn'))(ip)
        # x = Activation('elu')(x)

        if local_f is not None:
            local_f = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(local_f)
            local_f = Activation('elu')(local_f)

        global_f = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(global_f)
        global_f = Activation('elu')(global_f)

        if low_res_input:
            sub_sample_factor = (2, 2) if dimension == 2 else (2, 2, 2)
        else:
            sub_sample_factor = (1, 1) if dimension == 2 else (1, 1, 1)
        global_f_crop_size = np.int((global_f._keras_shape[2] * sub_sample_factor[0] - global_f._keras_shape[2]) / 2)

        if dimension == 2:
            if local_f is not None:
                local_f = Conv2D(nb_filter, (1, 1), kernel_initializer=kernel_init,
                           padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                           name=name_or_none(block_prefix, '_conv2D_normal'))(local_f)
            global_f = Conv2D(nb_filter, (1, 1), kernel_initializer=kernel_init,
                       padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                       name=name_or_none(block_prefix, '_conv2D_low_res'))(global_f)
            global_f_up = UpSampling2D(size=sub_sample_factor, interpolation='bilinear')(global_f)
            global_f_crop = get_cropping_layer(dimension, global_f_up, crop_size=(global_f_crop_size, global_f_crop_size))

            # low_res_up = Conv2DTranspose(nb_filter, (1, 1), activation='elu', padding='same', kernel_initializer=kernel_init,
            #                     strides=strides, kernel_initializer='he_normal',
            #                     kernel_regularizer=l2(weight_decay),
            #                     name=name_or_none(block_prefix, '_conv2DT'))(low_res)
            # if dropout_rate:
            #     x = Dropout(dropout_rate, seed=random_seed_num)(x)
        else:
            if local_f is not None:
                local_f = Conv3D(nb_filter, (1, 1, 1), kernel_initializer=kernel_init,
                           padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                           name=name_or_none(block_prefix, '_conv3D_normal'))(local_f)
            global_f = Conv3D(nb_filter, (1, 1, 1), kernel_initializer=kernel_init,
                       padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                       name=name_or_none(block_prefix, '_conv3D_low_res'))(global_f)
            global_f_up = UpSampling3D(size=sub_sample_factor)(global_f)
            global_f_crop = get_cropping_layer(dimension, global_f_up, crop_size=(global_f_crop_size, global_f_crop_size))

            # low_res_up = Conv3DTranspose(nb_filter, (1, 1, 1), activation='elu', padding='same', kernel_initializer=kernel_init,
            #                     strides=strides, kernel_initializer='he_normal',
            #                     kernel_regularizer=l2(weight_decay),
            #                     name=name_or_none(block_prefix, '_conv3DT'))(low_res)

        if local_f is not None:
            x = Add()([local_f, global_f_crop])
        else:
            x = global_f_crop

            # if dropout_rate:
            #     x = Dropout(dropout_rate, seed=random_seed_num)(x)
        # x = BatchNormalization(axis=channel_axis, epsilon=1.1e-5,
        #                        name=name_or_none(block_prefix, '_bn'))(x)
        # x = Activation('elu')(x)

        return x


def get_cropping_layer(dimension, input, crop_size=(16, 16)) :
    cropping_param = (crop_size, crop_size) if dimension == 2 else (crop_size, crop_size, crop_size)

    if dimension == 2 :
        return Cropping2D(cropping=cropping_param)(input)
    else :
        return Cropping3D(cropping=cropping_param)(input)


def get_low_res_layer(dimension, input, type='max'):
    if dimension == 2 :
        if type == 'average':
            return AveragePooling2D()(input)
        else:
            return MaxPooling2D()(input)
    else :
        if type == 'average':
            return AveragePooling3D()(input)
        else:
            return MaxPooling3D()(input)

def multi_scale_input(dimension, input, nb_filter, initial_kernel_size, kernel_init, block_idx, block_prefix_num,
                      weight_decay, concat_axis):
    input = get_low_res_layer(dimension, input, type='max')
    if dimension == 2:
        input_conv = Conv2D(nb_filter, initial_kernel_size,
                   kernel_initializer=kernel_init, padding='same',
                   name='multi_input_conv2D_%i_%i' % (block_idx, block_prefix_num), use_bias=False,
                   kernel_regularizer=l2(weight_decay))(input)
    else:
        input_conv = Conv3D(nb_filter, initial_kernel_size,
                   kernel_initializer=kernel_init, padding='same',
                   name='multi_input_conv3D_%i_%i' % (block_idx, block_prefix_num), use_bias=False,
                   kernel_regularizer=l2(weight_decay))(input)
    input_bn = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                      name='multi_input_bn_%i_%i' % (block_idx, block_prefix_num))(input_conv)
    input_activation = Activation('elu')(input_bn)

    return input, input_activation


def name_or_none(prefix, name):
    return prefix + name if (prefix is not None and name is not None) else None

