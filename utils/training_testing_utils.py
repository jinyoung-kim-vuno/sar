import numpy as np

from keras.utils import np_utils
from operator import itemgetter
from .extraction import extract_patches
from .general_utils import pad_both_sides
from utils.image import normalize_image


def split_train_val(train_indexes, validation_split) :
    N = len(train_indexes)
    val_volumes = np.int32(np.ceil(N * validation_split))
    train_volumes = N - val_volumes

    return train_indexes[:train_volumes], train_indexes[train_volumes:]


def build_training_set(gen_conf, train_conf, input_data, labels):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    bg_discard_percentage = train_conf['bg_discard_percentage']
    dimension = train_conf['dimension']
    extraction_step = train_conf['extraction_step']
    output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']
    normalized_range_min = dataset_info['normalized_range_min']
    normalized_range_max = dataset_info['normalized_range_max']
    normalized_range = [normalized_range_min, normalized_range_max]
    bg_value = normalized_range_min
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    target = dataset_info['target']
    if type(target) == list:
        num_targets = len(target) + 1  # include bg
    else:
        num_targets = 2

    #label_selector = determine_label_selector(patch_shape, output_shape)
    minimum_non_bg = bg_discard_percentage * np.prod(output_shape)

    data_patch_shape = (num_modality, ) + patch_shape
    data_extraction_step = (num_modality, ) + extraction_step
    output_patch_shape = (np.prod(output_shape), num_targets)

    x = np.zeros((0, ) + data_patch_shape)
    y = np.zeros((0, ) + output_patch_shape)
    for idx in range(len(input_data)):
        y_length = len(y)

        # pad_size = ()
        # for dim in range(dimension) :
        #     pad_size += (patch_shape[dim] // 2, )

        data_pad_size = ()
        for dim in range(dimension) :
            data_pad_size += (patch_shape[dim] // 2, )

        label_pad_size = ()
        for dim in range(dimension) :
            label_pad_size += (output_shape[dim] // 2, )

        print(labels[idx][0,0].shape)
        print(input_data[idx][0].shape)
        label_vol = pad_both_sides(dimension, labels[idx][0,0], label_pad_size, bg_value)
        input_vol = pad_both_sides(dimension, input_data[idx][0], data_pad_size, bg_value)

        label_patches = extract_patches(dimension, label_vol, output_shape, extraction_step)
        #label_patches = extract_patches(dimension, label_vol, patch_shape, extraction_step)
        #label_patches = label_patches[tuple(label_selector)]

        sum_axis = (1, 2, 3) if dimension == 3 else (1, 2)

        valid_idxs = np.where(np.sum(label_patches != bg_value, axis=sum_axis) >= minimum_non_bg)
        label_patches = label_patches[valid_idxs]

        N = len(label_patches)

        x = np.vstack((x, np.zeros((N, ) + data_patch_shape)))
        y = np.vstack((y, np.zeros((N, ) + output_patch_shape)))

        # one-hot encoding (if sparse_categorical_crossentropy is used, don't do this and just leave integer target)
        for i in range(N) :
            tmp = np_utils.to_categorical(label_patches[i].flatten(), num_targets)
            y[i + y_length] = tmp

        del label_patches

        data_train = extract_patches(dimension, input_vol, data_patch_shape, data_extraction_step)
        x[y_length:] = data_train[valid_idxs]

        del data_train

    # debug
    # w = np.zeros((num_classes,))
    # print(y.shape)
    # for index in range(num_classes):
    #     y_true_i_class = np.ndarray.flatten(y[:,:,index])
    #     print(y_true_i_class)
    #     w[index] = np.sum(np.asarray(y_true_i_class == 1, np.int8))
    # print(w)

    return x, y


def build_training_set_sar(gen_conf, train_conf, train_src_lst, train_sar_lst, g_d):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    bg_discard_percentage = train_conf['bg_discard_percentage']
    normalized_range_min = dataset_info['normalized_range_min']
    normalized_range_max = dataset_info['normalized_range_max']
    normalized_range = [normalized_range_min, normalized_range_max]
    bg_value = normalized_range_min
    dimension = train_conf['dimension']
    #extraction_step = train_conf['extraction_step']
    # output_shape = train_conf['output_shape']
    # patch_shape = train_conf['patch_shape']
    modality = dataset_info['image_modality']
    num_modality = len(modality)

    if g_d == 'g':
        patch_shape = train_conf['GAN']['generator']['patch_shape']
        output_shape = train_conf['GAN']['generator']['output_shape']
        extraction_step = train_conf['GAN']['generator']['extraction_step']
    else:
        patch_shape = train_conf['GAN']['discriminator']['patch_shape']
        output_shape = train_conf['GAN']['discriminator']['output_shape']
        extraction_step = train_conf['GAN']['discriminator']['extraction_step']

    minimum_non_bg = bg_discard_percentage * np.prod(output_shape)

    data_patch_shape = (num_modality, ) + patch_shape
    data_extraction_step = (num_modality, ) + extraction_step
    output_patch_shape = (np.prod(output_shape), 1)

    x = np.zeros((0, ) + data_patch_shape)
    y = np.zeros((0, ) + output_patch_shape)
    for idx in range(len(train_src_lst)):
        y_length = len(y)

        # pad_size = ()
        # for dim in range(dimension) :
        #     pad_size += (patch_shape[dim] // 2, )

        data_pad_size = ()
        for dim in range(dimension) :
            data_pad_size += (patch_shape[dim] // 2, )

        label_pad_size = ()
        for dim in range(dimension) :
            label_pad_size += (output_shape[dim] // 2, )

        print(train_sar_lst[idx][0,0].shape)
        print(train_src_lst[idx][0].shape)
        sar_vol = pad_both_sides(dimension, train_sar_lst[idx][0,0], label_pad_size, bg_value)
        src_vol = pad_both_sides(dimension, train_src_lst[idx][0], data_pad_size, bg_value)

        sar_patches = extract_patches(dimension, sar_vol, output_shape, extraction_step)

        sum_axis = (1, 2, 3) if dimension == 3 else (1, 2)

        valid_idxs = np.where(np.sum(sar_patches != bg_value, axis=sum_axis) > minimum_non_bg)
        #previous ver.: valid_idxs = np.where(np.sum(sar_patches != bg_value, axis=sum_axis) >= minimum_non_bg)
        sar_patches = sar_patches[valid_idxs]

        N = len(sar_patches)

        x = np.vstack((x, np.zeros((N, ) + data_patch_shape)))
        y = np.vstack((y, np.zeros((N, ) + output_patch_shape)))

        # one-hot encoding (if sparse_categorical_crossentropy is used, don't do this and just leave integer target)
        for i in range(N) :
            y[i + y_length][:, 0] = sar_patches[i].flatten()

        del sar_patches

        data_train = extract_patches(dimension, src_vol, data_patch_shape, data_extraction_step)
        x[y_length:] = data_train[valid_idxs]

        del data_train


    # debug
    # w = np.zeros((num_classes,))
    # print(y.shape)
    # for index in range(num_classes):
    #     y_true_i_class = np.ndarray.flatten(y[:,:,index])
    #     print(y_true_i_class)
    #     w[index] = np.sum(np.asarray(y_true_i_class == 1, np.int8))
    # print(w)

    return x, y


def build_training_set_sar_2_5D_MIMO(gen_conf, train_conf, train_src_lst, train_sar_lst, patch_shape, output_shape):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    bg_discard_percentage = train_conf['bg_discard_percentage']
    normalized_range_min = dataset_info['normalized_range_min']
    normalized_range_max = dataset_info['normalized_range_max']
    normalized_range = [normalized_range_min, normalized_range_max]
    bg_value = normalized_range_min
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    num_classes = train_conf['GAN']['generator']['num_classes']

    data_train_total = [] # training data in each dimension
    for i, j, k in zip([0,0,1], [1,2,2], [2,1,0]):
        patch_shape_2d = (patch_shape[i], patch_shape[j])
        output_shape_2d = (output_shape[i], output_shape[j])

        minimum_non_bg = bg_discard_percentage * np.prod(output_shape_2d)

        data_patch_shape = (num_modality, ) + patch_shape_2d
        output_patch_shape = (np.prod(output_shape_2d), num_classes)
        #output_patch_shape = (np.prod(output_shape_2d), 1)

        x = np.zeros((0, ) + data_patch_shape)
        y = np.zeros((0, ) + output_patch_shape)

        for idx in range(len(train_src_lst)):
            y_length = len(y)


            print(train_sar_lst[idx][0,0].shape)
            print(train_src_lst[idx][0].shape)
            sar_vol = train_sar_lst[idx][0,0]
            sar_vol_norm = normalize_image(sar_vol, normalized_range)  # tanh : -1~1
            src_vol = train_src_lst[idx][0]
            src_vol_norm = np.zeros(src_vol.shape)
            for m in range(num_modality):
                src_vol_norm[m, :, :, :] = normalize_image(src_vol[m, :, :, :], normalized_range)  # tanh : -1~1

            #print(train_sar_lst[idx][0,0].shape)
            #print(train_src_lst[idx][0].shape)
            #sar_vol = train_sar_lst[idx][0,0]
            #src_vol = train_src_lst[idx][0]

            sar_patches = np.zeros((sar_vol_norm.shape[k], ) + patch_shape_2d)
            src_patches = np.zeros((sar_vol_norm.shape[k], ) + data_patch_shape)

            for n in range(sar_vol.shape[k]):
                if k == 2: # axial
                    sar_patches[n, :, :] = sar_vol_norm[:, :, n]
                    src_patches[n, :, :, :] = src_vol_norm[:, :, :, n]
                    # sar_patches.append(sar_vol[:, :, n])
                    # src_patches.append(src_vol[:, :, :, n])
                elif k == 1: # sagittal
                    sar_patches[n, :, :] = sar_vol_norm[:, n, :]
                    src_patches[n, :, :, :] = src_vol_norm[:, :, n, :]
                    # sar_patches.append(sar_vol[:, n, :])
                    # src_patches.append(src_vol[:, :, n, :])
                elif k == 0: # coronal
                    sar_patches[n, :, :] = sar_vol_norm[n, :, :]
                    src_patches[n, :, :, :] = src_vol_norm[:, n, :, :]
                    # sar_patches.append(sar_vol[n, :, :])
                    # src_patches.append(src_vol[:, n, :, :])

            print(sar_patches.shape)
            print(src_patches.shape)

            print(len(sar_patches))
            valid_idxs = np.where(np.sum(sar_patches != bg_value, axis=(1, 2)) > minimum_non_bg)
            sar_patches = sar_patches[valid_idxs]

            N = len(sar_patches)

            x = np.vstack((x, np.zeros((N, ) + data_patch_shape)))
            y = np.vstack((y, np.zeros((N, ) + output_patch_shape)))

            # one-hot encoding (if sparse_categorical_crossentropy is used, don't do this and just leave integer target)
            for i in range(N):
                for c in range(num_classes):
                    y[i + y_length][:, c] = sar_patches[i].flatten()

            del sar_patches

            x[y_length:] = src_patches[valid_idxs]

            del src_patches


        data_train_total.append([x, y])

    return data_train_total


# for local sar lower/upper estimator
def build_training_set_sar_2_5D_MIMO_multi_task(gen_conf, train_conf, train_src_lst, train_sar_lst, patch_shape, output_shape):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    bg_discard_percentage = train_conf['bg_discard_percentage']
    normalized_range_min = dataset_info['normalized_range_min']
    normalized_range_max = dataset_info['normalized_range_max']
    normalized_range = [normalized_range_min, normalized_range_max]
    bg_value = normalized_range_min
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    num_classes = train_conf['GAN']['generator']['num_classes']

    data_train_total = [] # training data in each dimension
    for i, j, k in zip([0,0,1], [1,2,2], [2,1,0]):
        patch_shape_2d = (patch_shape[i], patch_shape[j])
        output_shape_2d = (output_shape[i], output_shape[j])

        minimum_non_bg = bg_discard_percentage * np.prod(output_shape_2d)

        data_patch_shape = (num_modality, ) + patch_shape_2d
        output_patch_shape = (np.prod(output_shape_2d), num_classes)
        #output_patch_shape = (np.prod(output_shape_2d), 1)

        x_norm = np.zeros((0, ) + data_patch_shape)
        y_norm = np.zeros((0, ) + output_patch_shape)
        y = np.zeros((0,) + output_patch_shape)
        y_tran_id = np.zeros((0,2))

        for idx in range(len(train_src_lst)):
            y_length = len(y)

            print(train_sar_lst[idx][0,0].shape)
            print(train_src_lst[idx][0].shape)
            sar_vol = train_sar_lst[idx][0,0]
            sar_vol_norm = normalize_image(sar_vol, normalized_range)  # tanh : -1~1
            src_vol = train_src_lst[idx][0]
            src_vol_norm = np.zeros(src_vol.shape)
            for m in range(num_modality):
                src_vol_norm[m, :, :, :] = normalize_image(src_vol[m, :, :, :], normalized_range)  # tanh : -1~1

            sar_patches = np.zeros((sar_vol.shape[k], ) + patch_shape_2d)
            sar_norm_patches = np.zeros((sar_vol_norm.shape[k],) + patch_shape_2d)
            src_norm_patches = np.zeros((sar_vol_norm.shape[k], ) + data_patch_shape)

            for n in range(sar_vol.shape[k]):
                if k == 2: # axial
                    sar_patches[n, :, :] = sar_vol[:, :, n]
                    sar_norm_patches[n, :, :] = sar_vol_norm[:, :, n]
                    src_norm_patches[n, :, :, :] = src_vol_norm[:, :, :, n]
                    # sar_patches.append(sar_vol[:, :, n])
                    # src_patches.append(src_vol[:, :, :, n])
                elif k == 1: # sagittal
                    sar_patches[n, :, :] = sar_vol[:, n, :]
                    sar_norm_patches[n, :, :] = sar_vol_norm[:, n, :]
                    src_norm_patches[n, :, :, :] = src_vol_norm[:, :, n, :]
                    # sar_patches.append(sar_vol[:, n, :])
                    # src_patches.append(src_vol[:, :, n, :])
                elif k == 0: # coronal
                    sar_patches[n, :, :] = sar_vol[n, :, :]
                    sar_norm_patches[n, :, :] = sar_vol_norm[n, :, :]
                    src_norm_patches[n, :, :, :] = src_vol_norm[:, n, :, :]
                    # sar_patches.append(sar_vol[n, :, :])
                    # src_patches.append(src_vol[:, n, :, :])

            print(sar_norm_patches.shape)
            print(src_norm_patches.shape)

            #valid_idxs = np.where(np.sum(sar_norm_patches != bg_value, axis=(1, 2)) > minimum_non_bg)

            # make all the volume same size wo removing black slices for few-shot learning
            valid_idxs = np.where(np.sum(sar_norm_patches != bg_value, axis=(1, 2)) >= minimum_non_bg)
            sar_patches = sar_patches[valid_idxs]
            sar_norm_patches = sar_norm_patches[valid_idxs]

            print(len(sar_norm_patches))

            N = len(sar_norm_patches)

            x_norm = np.vstack((x_norm, np.zeros((N, ) + data_patch_shape)))
            y_norm = np.vstack((y_norm, np.zeros((N,) + output_patch_shape)))
            y = np.vstack((y, np.zeros((N, ) + output_patch_shape)))
            y_tran_id = np.vstack((y_tran_id, np.zeros((N, 2))))

            # one-hot encoding (if sparse_categorical_crossentropy is used, don't do this and just leave integer target)
            for l in range(N):
                for c in range(num_classes):
                    y_norm[l + y_length][:, c] = sar_norm_patches[l].flatten()
                    y[l + y_length][:, c] = sar_patches[l].flatten()
                y_tran_id[l + y_length][0] = idx
                y_tran_id[l + y_length][1] = l

            del sar_patches, sar_norm_patches

            x_norm[y_length:] = src_norm_patches[valid_idxs]

            del src_norm_patches


        data_train_total.append([x_norm, y_norm, y, y_tran_id])

    return data_train_total


def build_training_set_sar_2_5D(gen_conf, train_conf, train_src_lst, train_sar_lst, patch_shape, output_shape):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    bg_discard_percentage = train_conf['bg_discard_percentage']
    normalized_range_min = dataset_info['normalized_range_min']
    normalized_range_max = dataset_info['normalized_range_max']
    normalized_range = [normalized_range_min, normalized_range_max]
    bg_value = normalized_range_min
    modality = dataset_info['image_modality']
    num_modality = len(modality)

    data_train_total = [] # training data in each dimension
    for i, j, k in zip([0,0,1], [1,2,2], [2,1,0]):
        patch_shape_2d = (patch_shape[i], patch_shape[j])
        output_shape_2d = (output_shape[i], output_shape[j])

        minimum_non_bg = bg_discard_percentage * np.prod(output_shape_2d)

        data_patch_shape = (num_modality, ) + patch_shape_2d
        output_patch_shape = (np.prod(output_shape_2d), 1)

        x = np.zeros((0, ) + data_patch_shape)
        y = np.zeros((0, ) + output_patch_shape)

        for idx in range(len(train_src_lst)):
            y_length = len(y)

            print(train_sar_lst[idx][0,0].shape)
            print(train_src_lst[idx][0].shape)
            sar_vol = train_sar_lst[idx][0,0]
            src_vol = train_src_lst[idx][0]

            sar_patches = np.zeros((sar_vol.shape[k], ) + patch_shape_2d)
            src_patches = np.zeros((sar_vol.shape[k], ) + data_patch_shape)

            for n in range(sar_vol.shape[k]):
                if k == 2: # axial
                    sar_patches[n, :, :] = sar_vol[:, :, n]
                    src_patches[n, :, :, :] = src_vol[:, :, :, n]
                    # sar_patches.append(sar_vol[:, :, n])
                    # src_patches.append(src_vol[:, :, :, n])
                elif k == 1: # sagittal
                    sar_patches[n, :, :] = sar_vol[:, n, :]
                    src_patches[n, :, :, :] = src_vol[:, :, n, :]
                    # sar_patches.append(sar_vol[:, n, :])
                    # src_patches.append(src_vol[:, :, n, :])
                elif k == 0: # coronal
                    sar_patches[n, :, :] = sar_vol[n, :, :]
                    src_patches[n, :, :, :] = src_vol[:, n, :, :]
                    # sar_patches.append(sar_vol[n, :, :])
                    # src_patches.append(src_vol[:, n, :, :])

            print(sar_patches.shape)
            print(src_patches.shape)

            print(len(sar_patches))
            valid_idxs = np.where(np.sum(sar_patches != bg_value, axis=(1, 2)) > minimum_non_bg)
            sar_patches = sar_patches[valid_idxs]

            N = len(sar_patches)

            x = np.vstack((x, np.zeros((N, ) + data_patch_shape)))
            y = np.vstack((y, np.zeros((N, ) + output_patch_shape)))

            # one-hot encoding (if sparse_categorical_crossentropy is used, don't do this and just leave integer target)
            for i in range(N):
                y[i + y_length][:, 0] = sar_patches[i].flatten()

            del sar_patches

            x[y_length:] = src_patches[valid_idxs]

            del src_patches


        data_train_total.append([x, y])

    return data_train_total



def build_training_set_4d(gen_conf, train_conf, input_data, labels):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    bg_discard_percentage = train_conf['bg_discard_percentage']
    normalized_range_min = dataset_info['normalized_range_min']
    normalized_range_max = dataset_info['normalized_range_max']
    normalized_range = [normalized_range_min, normalized_range_max]
    bg_value = normalized_range_min
    dimension = train_conf['dimension']
    extraction_step = train_conf['extraction_step']
    num_classes = gen_conf['num_classes']
    output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']

    modality = dataset_info['image_modality']
    num_modality = len(modality)

    label_selector = determine_label_selector(patch_shape, output_shape)
    minimum_non_bg = bg_discard_percentage * np.prod(output_shape)

    data_patch_shape = (num_modality,) + patch_shape
    data_extraction_step = (num_modality,) + extraction_step
    # output_patch_shape = (np.prod(output_shape), num_classes)
    output_patch_shape = (num_modality,) + (np.prod(output_shape), num_classes)

    x = np.zeros((0,) + data_patch_shape)
    y = np.zeros((0,) + output_patch_shape)
    for idx in range(len(input_data)):
        y_length = len(y)

        pad_size = ()
        for dim in range(dimension):
            pad_size += (patch_shape[dim] // 2,)

        # print(labels[idx][0,0].shape)
        print(labels[idx][0].shape)
        print(input_data[idx][0].shape)

        # label_vol = pad_both_sides(dimension, labels[idx][0,0], pad_size)
        label_vol = pad_both_sides(dimension, labels[idx][0], pad_size, bg_value)
        input_vol = pad_both_sides(dimension, input_data[idx][0], pad_size, bg_value)

        # label_patches = extract_patches(dimension, label_vol, patch_shape, extraction_step)
        label_patches = extract_patches(dimension, label_vol, data_patch_shape, data_extraction_step)
        label_patches = label_patches[label_selector]

        label_patches = np.transpose(label_patches, [1, 0, 2, 3, 4])

        sum_axis = (1, 2, 3) if dimension == 3 else (1, 2)

        # select useful patches based on labels
        valid_idxs_list = []
        n_patches = []
        for t in range(num_modality):
            valid_idxs = np.where(np.sum(label_patches[t] != bg_value, axis=sum_axis) >= minimum_non_bg)
            valid_idxs_list.append(valid_idxs[0])
            n_patches.append(len(valid_idxs[0]))

        min_n_patches = min(np.array(n_patches))
        min_valid_idxs_list = []
        label_patches_list = []
        for t in range(num_modality):
            min_valid_idxs = np.random.choice(valid_idxs_list[t], min_n_patches, False)
            min_valid_idxs_list.append(min_valid_idxs)
            label_patches_list.append([label_patches[t][i] for i in min_valid_idxs])

        N = min_n_patches
        x = np.vstack((x, np.zeros((N,) + data_patch_shape)))
        y = np.vstack((y, np.zeros((N,) + output_patch_shape)))

        data_train = extract_patches(dimension, input_vol, data_patch_shape, data_extraction_step)
        data_train = np.transpose(data_train, [1, 0, 2, 3, 4])

        for t in range(num_modality):
            for i in range(N):
                tmp = np_utils.to_categorical(label_patches_list[t][i].flatten(), num_classes)
                y[i + y_length][t] = tmp
                x[y_length + i][t] = data_train[t][min_valid_idxs_list[t][i]]

        print(x.shape)
        print(y.shape)

        del label_patches
        del data_train

    return x, y


def build_testing_set(gen_conf, test_conf, input_data) :
    dataset = test_conf['dataset']
    dimension = test_conf['dimension']
    extraction_step = test_conf['extraction_step']
    patch_shape = test_conf['patch_shape']
    dataset_info = gen_conf['dataset_info'][dataset]
    modality = dataset_info['image_modality']
    num_modality = len(modality)

    data_patch_shape = (num_modality, ) + patch_shape
    data_extraction_step = (num_modality, ) + extraction_step

    return extract_patches(dimension, input_data[0], data_patch_shape, data_extraction_step)


# def build_testing_set_2_5D(gen_conf, test_conf, input_data):
#     dataset = test_conf['dataset']
#     dimension = test_conf['dimension']
#     patch_shape = test_conf['GAN']['generator']['output_shape']
#     dataset_info = gen_conf['dataset_info'][dataset]
#     modality = dataset_info['image_modality']
#     num_modality = len(modality)
#
#     print(input_data[0].shape)
#
#     test_patches_total = []
#     for i, j, k in zip([0,0,1], [1,2,2], [2,1,0]):
#         patch_shape_2d = (patch_shape[i], patch_shape[j])
#         data_patch_shape = (num_modality,) + patch_shape_2d
#         N = input_data[0].shape[k+1]
#         test_patches = np.zeros((N,) + data_patch_shape)
#         for n in range(N):
#             if k == 2:
#                 test_patches[n, :, :, :] = input_data[0][:, :, :, n]
#                 #test_patches.append(input_data[0][:, :, n])
#             elif k == 1:
#                 test_patches[n, :, :, :] = input_data[0][:, :, n, :]
#                 #test_patches.append(input_data[0][:, n, :])
#             elif k == 0:
#                 test_patches[n, :, :, :] = input_data[0][:, n, :, :]
#                 #test_patches.append(input_data[0][n, :, :])
#
#         print(test_patches.shape)
#         test_patches_total.append(test_patches)
#
#     return test_patches_total


def build_testing_set_2_5D(gen_conf, test_conf, input_data):
    dataset = test_conf['dataset']
    dimension = test_conf['dimension']
    patch_shape = test_conf['GAN']['generator']['patch_shape']
    dataset_info = gen_conf['dataset_info'][dataset]
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    normalized_range_min = dataset_info['normalized_range_min']
    normalized_range_max = dataset_info['normalized_range_max']
    normalized_range = [normalized_range_min, normalized_range_max]

    print(input_data[0].shape)
    test_src_vol = input_data[0]
    test_src_vol_norm = np.zeros(test_src_vol.shape)
    for m in range(num_modality):
        test_src_vol_norm[m, :, :, :] = normalize_image(test_src_vol[m, :, :, :], normalized_range)  # tanh : -1~1

    test_patches_total = []
    for i, j, k in zip([0,0,1], [1,2,2], [2,1,0]):
        patch_shape_2d = (patch_shape[i], patch_shape[j])
        data_patch_shape = (num_modality,) + patch_shape_2d
        N = test_src_vol_norm.shape[k+1]
        test_patches = np.zeros((N,) + data_patch_shape)
        for n in range(N):
            if k == 2:
                test_patches[n, :, :, :] = test_src_vol_norm[:, :, :, n]
                #test_patches.append(input_data[0][:, :, n])
            elif k == 1:
                test_patches[n, :, :, :] = test_src_vol_norm[:, :, n, :]
                #test_patches.append(input_data[0][:, n, :])
            elif k == 0:
                test_patches[n, :, :, :] = test_src_vol_norm[:, n, :, :]
                #test_patches.append(input_data[0][n, :, :])

        print(test_patches.shape)
        test_patches_total.append(test_patches)

    return test_patches_total


def build_testing_fs_samples_2_5D(gen_conf, test_conf, input_data, patch_shape, output_shape):
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    normalized_range_min = dataset_info['normalized_range_min']
    normalized_range_max = dataset_info['normalized_range_max']
    normalized_range = [normalized_range_min, normalized_range_max]
    num_classes = test_conf['GAN']['generator']['num_classes']

    sar_patches_out_total = []
    sar_norm_patches_out_total = [] # training data in each dimension
    for i, j, k in zip([0,0,1], [1,2,2], [2,1,0]):
        patch_shape_2d = (patch_shape[i], patch_shape[j])
        output_shape_2d = (output_shape[i], output_shape[j])
        output_patch_shape = (np.prod(output_shape_2d), num_classes)

        sar_vol = input_data[0, 0]
        sar_vol_norm = normalize_image(sar_vol, normalized_range)  # tanh : -1~1

        sar_patches = np.zeros((sar_vol.shape[k],) + patch_shape_2d)
        sar_norm_patches = np.zeros((sar_vol_norm.shape[k],) + patch_shape_2d)

        sar_patches_out = np.zeros((sar_vol.shape[k],) + output_patch_shape)
        sar_norm_patches_out = np.zeros((sar_vol_norm.shape[k],) + output_patch_shape)
        for n in range(sar_vol.shape[k]):
            if k == 2: # axial
                sar_patches[n, :, :] = sar_vol[:, :, n]
                sar_norm_patches[n, :, :] = sar_vol_norm[:, :, n]
            elif k == 1: # sagittal
                sar_patches[n, :, :] = sar_vol[:, n, :]
                sar_norm_patches[n, :, :] = sar_vol_norm[:, n, :]
            elif k == 0: # coronal
                sar_patches[n, :, :] = sar_vol[n, :, :]
                sar_norm_patches[n, :, :] = sar_vol_norm[n, :, :]

            sar_patches_out[n][:, 0] = sar_patches[n].flatten()
            sar_norm_patches_out[n][:, 0] = sar_norm_patches[n].flatten()

        sar_patches_out_total.append(sar_patches_out)
        sar_norm_patches_out_total.append(sar_norm_patches_out)

    return sar_norm_patches_out_total, sar_patches_out_total


def determine_label_selector(patch_shape, output_shape) :
    ndim = len(patch_shape)
    patch_shape_equal_output_shape = patch_shape == output_shape

    slice_none = slice(None)
    if not patch_shape_equal_output_shape :
        return [slice_none] + [slice(output_shape[i], patch_shape[i] - output_shape[i]) for i in range(ndim)]
    else :
        return [slice_none for i in range(ndim)]