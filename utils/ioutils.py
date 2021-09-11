
from scipy.ndimage import zoom, label
from scipy.io import loadmat
import scipy.io as sio
import h5py
import nibabel as nib
import numpy as np
import os
import glob
import h5py
from utils.BrainImage import BrainImage, _remove_ending
import subprocess
from utils.image import find_crop_mask, compute_crop_mask, compute_crop_mask_manual, crop_image, \
    compute_side_mask, postprocess, normalize_image, generate_structures_surface, apply_image_orientation_to_stl, \
    write_stl, __smooth_stl, __smooth_binary_img, check_empty_vol
from utils.image import remove_outliers
#from nipype.interfaces.slicer.filtering import histogrammatching
from utils.callbacks import generate_output_filename
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims


def read_dataset(gen_conf, train_conf):
    root_path = gen_conf['root_path']
    dataset = train_conf['dataset']
    dataset_path = root_path + gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    preprocess = train_conf['preprocess']
    modality = dataset_info['image_modality']
    num_modality = len(modality)

    if dataset == 'iSeg2017' :
        return read_iSeg2017_dataset(dataset_path, dataset_info)
    if dataset == 'IBSR18' :
        return read_IBSR18_dataset(dataset_path, dataset_info)
    if dataset == 'MICCAI2012' :
        return read_MICCAI2012_dataset(dataset_path, dataset_info)
    if dataset in ['3T7T', '3T7T_real', '3T7T_total', '3T+7T']:
        if num_modality < 2:
            return read_3T7T_dataset(dataset_path, dataset_info, preprocess)
        else:
            return read_4d_images(dataset_path, dataset_info, preprocess)
    if dataset == 'ADNI':
        return read_ADNI_dataset(dataset_path, dataset_info, preprocess)
    if dataset in ['CBCT16', 'CBCT57', 'CT30']:
        return read_cbct_dataset(dataset_path, dataset_info, preprocess)


def save_dataset(gen_conf, train_conf, test_conf, volume, filename_ext, case_idx) :
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    modality = dataset_info['image_modality']
    num_modality = len(modality)

    if dataset in ['3T7T', '3T7T_real', '3T7T_total', '3T+7T']:
        if num_modality < 2:
            return save_volume_3T7T(gen_conf, train_conf, test_conf, volume, filename_ext, case_idx)
        else:
            return save_volume_4d(gen_conf, train_conf, test_conf, volume, filename_ext, case_idx)
    if dataset in ['CBCT16', 'CBCT57', 'CT30']:
        return save_volume_cbct(gen_conf, train_conf, test_conf, volume, filename_ext, case_idx)


def read_iSeg2017_dataset(dataset_path, dataset_info) :
    num_volumes = dataset_info['num_volumes']
    size = dataset_info['size']
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    inputs = dataset_info['inputs']

    image_data = np.zeros((num_volumes, num_modality) + size)
    labels = np.zeros((num_volumes, 1) + size)

    for img_idx in range(num_volumes):
        filename = dataset_path + path + pattern.format(str(img_idx + 1), inputs[0])
        image_data[img_idx, 0] = read_volume(filename)#[:, :, :, 0]
        
        filename = dataset_path + path + pattern.format(str(img_idx + 1), inputs[1])
        image_data[img_idx, 1] = read_volume(filename)#[:, :, :, 0]

        filename = dataset_path + path + pattern.format(img_idx + 1, inputs[1])
        labels[img_idx, 0] = read_volume(filename)[:, :, :, 0]

        image_data[img_idx, 1] = labels[img_idx, 0] != 0

    label_mapper = {0 : 0, 10 : 1, 150 : 2, 250 : 3}
    for key in label_mapper.keys() :
        labels[labels == key] = label_mapper[key]

    return image_data, labels

def read_IBSR18_dataset(dataset_path, dataset_info) :
    num_volumes = dataset_info['num_volumes']
    size = dataset_info['size']
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    inputs = dataset_info['inputs']

    image_data = np.zeros((num_volumes, num_modality) + size)
    labels = np.zeros((num_volumes, 1) + size)

    for img_idx in range(num_volumes) :
        filename = dataset_path + path + pattern.format(img_idx + 1, inputs[0])
        image_data[img_idx, 0] = read_volume(filename)[:, :, :, 0]

        filename = dataset_path + path + pattern.format(img_idx + 1, inputs[1])
        labels[img_idx, 0] = read_volume(filename)[:, :, :, 0]

    return image_data, labels

def read_MICCAI2012_dataset(dataset_path, dataset_info) :
    num_volumes = dataset_info['num_volumes']
    size = dataset_info['size']
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    folder_names = dataset_info['folder_names']

    image_data = np.zeros((np.sum(num_volumes), num_modality) + size)
    labels = np.zeros((np.sum(num_volumes), 1) + size)

    training_set = [1000, 1006, 1009, 1012, 1015, 1001, 1007,
        1010, 1013, 1017, 1002, 1008, 1011, 1014, 1036]

    testing_set = [1003, 1019, 1038, 1107, 1119, 1004, 1023, 1039, 1110, 1122, 1005,
        1024, 1101, 1113, 1125, 1018, 1025, 1104, 1116, 1128]

    for img_idx, image_name in enumerate(training_set) :
        filename = dataset_path + path + pattern[0].format(folder_names[0], image_name)
        image_data[img_idx, 0] = read_volume(filename)

        filename = dataset_path + path + pattern[1].format(folder_names[1], image_name)
        labels[img_idx, 0] = read_volume(filename)

        image_data[img_idx, 0] = np.multiply(image_data[img_idx, 0], labels[img_idx, 0] != 0)
        image_data[img_idx, 1] = labels[img_idx, 0] != 0

    for img_idx, image_name in enumerate(testing_set) :
        idx = img_idx + num_volumes[0]
        filename = dataset_path + path + pattern[0].format(folder_names[2], image_name)
        image_data[idx, 0] = read_volume(filename)

        filename = dataset_path + path + pattern[1].format(folder_names[3], image_name)
        labels[idx, 0] = read_volume(filename)

        image_data[idx, 0] = np.multiply(image_data[idx, 0], labels[idx, 0] != 0)
        image_data[idx, 1] = labels[idx, 0] != 0

    labels[labels > 4] = 0

    return image_data, labels


def read_4d_images(dataset_path, dataset_info, preprocess):
    num_volumes = dataset_info['num_volumes']
    size = dataset_info['size']
    modality = dataset_info['image_modality']
    num_modality = len(modality) # time_points
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    folder_names= dataset_info['folder_names']

    image_list = []
    label_lst = []
    data_filename_ext_list = []
    label_filename_ext_list = []

    label_mapper = {0: 0, 10: 1, 150: 2, 250: 3}

    for img_idx in range(num_volumes):
        filename = dataset_path + path + pattern[0].format(folder_names[0], str(img_idx + 1))
        print(filename)
        vol = read_volume(filename)
        data_filename_ext = os.path.basename(filename)
        data_filename_ext_list.append(data_filename_ext)

        if preprocess == 2 or preprocess == 3 or preprocess == 5:
            vol = normalize_image(vol, [0, 2**8])

        image_data = np.zeros((1, num_modality) + vol.shape[0:3])
        if np.size(np.shape(vol)) == 4:
            image_data[0, 0] = vol[:, :, :, 0]
            image_data[0, 1] = vol[:, :, :, 0]
            image_data[0, 2] = vol[:, :, :, 0]
        else:
            image_data[0, 0] = vol
            image_data[0, 1] = vol
            image_data[0, 2] = vol
        image_list.append(image_data)

        filename = dataset_path + path + pattern[1].format(folder_names[2], str(img_idx + 1))
        print(filename)
        vol = read_volume(filename)
        label_filename_ext = os.path.basename(filename)
        label_filename_ext_list.append(label_filename_ext)

        labels = np.zeros((1, num_modality) + vol.shape[0:3])
        if np.size(np.shape(vol)) == 4:
            labels[0, 0] = vol[:, :, :, 0]
            labels[0, 1] = vol[:, :, :, 0]
            labels[0, 2] = vol[:, :, :, 0]
            for key in label_mapper.keys():
                labels[labels == key] = label_mapper[key]
        else:
            labels[0, 0] = vol
            labels[0, 1] = vol
            labels[0, 2] = vol
            for key in label_mapper.keys():
                labels[labels == key] = label_mapper[key]
        label_lst.append(labels)

    return image_list, label_lst, np.array(data_filename_ext_list), np.array(label_filename_ext_list)


def read_3T7T_dataset(dataset_path, dataset_info, preprocess):
    num_volumes = dataset_info['num_volumes']
    size = dataset_info['size']
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    folder_names= dataset_info['folder_names']

    image_list = []
    label_lst = []
    data_filename_ext_list = []
    label_filename_ext_list = []

    label_mapper = {0: 0, 10: 1, 150: 2, 250: 3}

    for img_idx in range(num_volumes):
        filename = dataset_path + path + pattern[0].format(folder_names[0], str(img_idx + 1))
        print(filename)
        vol = read_volume(filename)
        data_filename_ext = os.path.basename(filename)
        data_filename_ext_list.append(data_filename_ext)

        if preprocess == 2 or preprocess == 3 or preprocess == 5:
            vol = normalize_image(vol, [0, 2**8])
            # out_filename = dataset_path + path + pattern[0].format(folder_names[0], str(img_idx + 1) +
            #                                                    '_normalized_0_1')
            # __save_volume(vol, read_volume_data(filename), out_filename, dataset_info['format'])

        image_data = np.zeros((1, num_modality) + vol.shape[0:3])
        if np.size(np.shape(vol)) == 4:
            image_data[0, 0] = vol[:, :, :, 0]
        else:
            image_data[0, 0] = vol
        image_list.append(image_data)

        filename = dataset_path + path + pattern[1].format(folder_names[2], str(img_idx + 1))
        print(filename)
        vol = read_volume(filename)
        label_filename_ext = os.path.basename(filename)
        label_filename_ext_list.append(label_filename_ext)

        labels = np.zeros((1, 1) + vol.shape[0:3])
        if np.size(np.shape(vol)) == 4:
            labels[0, 0] = vol[:, :, :, 0]
            for key in label_mapper.keys():
                labels[labels == key] = label_mapper[key]
        else:
            labels[0, 0] = vol
            for key in label_mapper.keys():
                labels[labels == key] = label_mapper[key]
        label_lst.append(labels)

    return image_list, label_lst, np.array(data_filename_ext_list), np.array(label_filename_ext_list)


def read_ADNI_dataset(dataset_path, dataset_info, preprocess):
    #num_volumes = dataset_info['num_volumes']
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    path = dataset_info['path']
    folder_names= dataset_info['folder_names']

    ext = '*.hdr'
    file_path = dataset_path + path + folder_names
    print(file_path)
    file_lst = glob.glob(file_path + '/' + ext)
    #vol_sample= read_volume(file_lst[0])
    #num_volumes = np.size (file_lst) # update # of images
    image_list = []
    filename_ext_list = []
    img_idx = 0
    for file_path_name in file_lst:
        vol = read_volume(file_path_name)

        if preprocess == 2 or preprocess == 3 or preprocess == 5:
           vol = normalize_image(vol, [0, 2**8])

        image_data = np.zeros((1, num_modality) + vol.shape[0:3])
        if np.size(np.shape(vol)) == 4:
            image_data[0, 0] = vol[:, :, :, 0]
        else:
            image_data[0, 0] = vol
        image_list.append(image_data)

        filename_ext = os.path.basename(file_path_name)
        filename_ext_list.append(filename_ext)
        img_idx += 1

    return image_list, filename_ext_list


def read_cbct_dataset(dataset_path, dataset_info, preprocess):
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    path = dataset_info['path']
    folder_names = dataset_info['folder_names']

    file_path = dataset_path + path + folder_names
    print (file_path)

    img_file_lst = glob.glob(file_path + '/' + '*origin.nii.gz') # image
    lb1_file_lst = glob.glob(file_path + '/' + '*mandible.nii.gz')# mandible label
    lb2_file_lst = glob.glob(file_path + '/' + '*midface.nii.gz') # midface label

    img_file_lst.sort()
    lb1_file_lst.sort()
    lb2_file_lst.sort()

    # print(img_file_lst)
    # print(lb1_file_lst)
    # print(lb2_file_lst)

    #vol_sample = read_volume(img_file_lst[0])
    #num_volumes = np.size(img_file_lst)  # update # of images
    image_list = []
    label_list = []
    img_filename_ext_list = []
    lb_filename_ext_list = []
    multimodal_data = []
    label_mapper = {0: 0, 150: 1, 250: 2}

    img_idx = 0
    for img_path, lb1_path, lb2_path in zip(img_file_lst, lb1_file_lst, lb2_file_lst):
        print(img_path)
        print(lb1_path)
        print(lb2_path)
        img_vol = read_volume(img_path)
        label1_vol = read_volume(lb1_path)
        label2_vol = read_volume(lb2_path)

        if preprocess == 2 or preprocess == 3 or preprocess == 5:
            img_vol = normalize_image(img_vol, [0, 2 ** 8])

        image_data = np.zeros((1, num_modality) + img_vol.shape[0:3])
        if np.size(np.shape(img_vol)) == 4:
            image_data[0, 0] = img_vol[:, :, :, 0]
        else:
            image_data[0, 0] = img_vol
        image_list.append(image_data)

        if np.size(np.shape(label1_vol)) == 4:
            label1_data = label1_vol[:, :, :, 0]
        else:
            label1_data = label1_vol

        if np.size(np.shape(label2_vol)) == 4:
            label2_data = label2_vol[:, :, :, 0]
        else:
            label2_data = label2_vol

        label_data = np.zeros((1, 1) + label1_vol.shape[0:3])
        label1_data[label1_data == np.max(label1_data)] = 150 # mandible label
        label2_data[label2_data == np.max(label2_data)] = 250 # midface label
        label_data[0, 0] = label1_data + label2_data
        for key in label_mapper.keys():
            label_data[label_data == key] = label_mapper[key]

        label_list.append(label_data)

        img_filename_ext = os.path.basename(img_path)
        lb1_filename_ext = os.path.basename(lb1_path)
        lb2_filename_ext = os.path.basename(lb2_path)
        img_filename_ext_list.append(img_filename_ext)
        lb_filename_ext_list.append([lb1_filename_ext, lb2_filename_ext])

        img_idx += 1

    print ("# of volumes: ", img_idx)

    return image_list, label_list, np.array(img_filename_ext_list), np.array(lb_filename_ext_list)


def read_tha_dataset(gen_conf, train_conf, train_patient_lst, test_patient_lst, preprocess_trn, preprocess_tst,
                         file_output_dir):
    dataset = train_conf['dataset']
    num_epochs = train_conf['num_epochs']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    mode = gen_conf['validation_mode']
    img_pattern = dataset_info['image_name_pattern']
    label_pattern = dataset_info['label_pattern']
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    init_mask_pattern = dataset_info['initial_mask_pattern']

    set_new_roi_mask = dataset_info['set_new_roi_mask']
    margin_crop_mask = dataset_info['margin_crop_mask']

    crop_trn_img_pattern = dataset_info['crop_trn_image_name_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    crop_trn_label_pattern = dataset_info['crop_trn_label_pattern']
    crop_tst_label_pattern = dataset_info['crop_tst_label_pattern']

    crop_init_mask_pattern = dataset_info['crop_initial_mask_pattern']

    train_roi_mask_pattern = dataset_info['train_roi_mask_pattern']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']

    modality_idx = []
    for m in modality: # the order of modality images should be fixed due to the absolute path
        if m == 'T1':
            modality_idx.append(0)
        if m == 'B0':
            modality_idx.append(1)
        if m == 'FA':
            modality_idx.append(2)

    # load multi-modal images in train_patient_lst and setting roi
    train_img_lst = []
    label_lst = []
    train_fname_lst = []
    label_fname_lst = []
    if train_patient_lst[0] is not None and num_epochs != 0 and mode != '2':
        for train_patient_id in train_patient_lst:

            file_train_patient_dir = os.path.join(file_output_dir, train_patient_id)
            if not os.path.exists(file_train_patient_dir):
                os.makedirs(os.path.join(file_output_dir, train_patient_id))

            train_roi_mask_file = os.path.join(file_train_patient_dir, train_roi_mask_pattern)
            if os.path.exists(train_roi_mask_file) and set_new_roi_mask is True:
                os.remove(train_roi_mask_file)

            # load label images in train_patient_lst and setting roi
            label_vol_lst = []
            label_fname_side_lst = []
            for side in ['left', 'right']:  # left + right
                for ver in ['2','3']:
                    label_filepath = os.path.join(dataset_path, train_patient_id, 'gt', label_pattern.format(side, ver))
                    if os.path.exists(label_filepath):
                        break
                    else:
                        if ver == '3':
                            print('No Found %s ground truth label' % side)
                            exit()
                print(label_filepath)
                label_fname_side_lst.append(label_filepath)
                label_data = read_volume_data(label_filepath)
                label_vol = label_data.get_data()
                if np.size(np.shape(label_vol)) == 4:
                    label_vol_lst.append(label_vol[:, :, :, 0])
                else:
                    label_vol_lst.append(label_vol)
            label_fname_lst.append(label_fname_side_lst)
            label_merge_vol = label_vol_lst[0] + label_vol_lst[1]

            # crop the roi of the image
            train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                       train_roi_mask_file,
                                                                                       label_data, margin_crop_mask)
            label_crop_vol = train_crop_mask.crop(label_merge_vol)

            # save the cropped labels (train_roi)
            for side in ['left', 'right']:
                for ver in ['2', '3']:
                    label_filepath = os.path.join(dataset_path, train_patient_id, 'gt', label_pattern.format(side, ver))
                    if os.path.exists(label_filepath):
                        break
                    else:
                        if ver == '3':
                            print('No Found %s ground truth label' % side)
                            exit()
                cropped_trn_label_file = os.path.join(file_output_dir, train_patient_id,
                                                      crop_trn_label_pattern.format(side, ver))
                if (not os.path.exists(cropped_trn_label_file)) or set_new_roi_mask is True:
                    crop_image(label_filepath, train_crop_mask, cropped_trn_label_file)

            label_crop_array = np.zeros((1, 1) + label_crop_vol.shape[0:3])
            label_crop_array[0, 0] = label_crop_vol
            label_lst.append(label_crop_array)

            # load training images in train_patient_lst and setting roi
            train_image_modal_lst= []
            train_fname_modality_lst = []
            for idx in modality_idx: #T1, B0, or FA
                training_img_path = os.path.join(dataset_path, train_patient_id, 'images', img_pattern[idx])
                print(training_img_path)
                train_fname_modality_lst.append(training_img_path)
                training_vol = read_volume(training_img_path)
                if preprocess_trn == 2 or preprocess_trn == 3 or preprocess_trn == 5:
                    training_vol = normalize_image(training_vol, [0, 2 ** 8])
                if np.size(np.shape(training_vol)) == 4:
                    train_crop_vol = train_crop_mask.crop(training_vol[:, :, :, 0])
                else:
                    train_crop_vol = train_crop_mask.crop(training_vol)
                train_image_modal_lst.append(train_crop_vol)
            train_fname_lst.append(train_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                train_img_path = os.path.join(dataset_path, train_patient_id, 'images', img_pattern[idx])
                crop_train_img_path = os.path.join(file_output_dir, train_patient_id, crop_trn_img_pattern[idx])
                if (not os.path.exists(crop_train_img_path)) or set_new_roi_mask is True:
                    crop_image(train_img_path, train_crop_mask, crop_train_img_path)
            train_crop_array = np.zeros((1, num_modality) + train_crop_vol.shape[0:3])
            for idx in range(num_modality):
                train_crop_array[0, idx] = train_image_modal_lst[idx]
            train_img_lst.append(train_crop_array)

    # load multi-modal images in test_patient_lst and setting roi
    test_img_lst = []
    test_fname_lst = []
    if test_patient_lst[0] is not None:
        for test_patient_id in test_patient_lst:

            file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
            if not os.path.exists(file_test_patient_dir):
                os.makedirs(os.path.join(file_output_dir, test_patient_id))

            test_roi_mask_file = os.path.join(file_test_patient_dir, test_roi_mask_pattern)
            if os.path.exists(test_roi_mask_file) and set_new_roi_mask is True:
                os.remove(test_roi_mask_file)

            # load initial mask (use fusion output or later linearly registered from ref. training image) for setting roi
            init_mask_vol_lst = []
            for side in ['left', 'right']:  # left + right
                init_mask_filepath = os.path.join(dataset_path, test_patient_id, 'fusion', init_mask_pattern.format(side))
                print(init_mask_filepath)
                init_mask_data = read_volume_data(init_mask_filepath)
                init_mask_vol = init_mask_data.get_data()
                if np.size(np.shape(init_mask_vol)) == 4:
                    init_mask_vol_lst.append(init_mask_vol[:, :, :, 0])
                else:
                    init_mask_vol_lst.append(init_mask_vol)
            init_mask_merge_vol = init_mask_vol_lst[0] + init_mask_vol_lst[1]

            # crop the roi of the image
            test_crop_mask = find_crop_mask(test_roi_mask_file) or compute_crop_mask(init_mask_merge_vol,
                                                                                     test_roi_mask_file,
                                                                                     init_mask_data,
                                                                                     margin_crop_mask)

            # save the cropped initial masks
            for side in ['left', 'right']:
                init_mask_filepath = os.path.join(dataset_path, test_patient_id, 'fusion',
                                                  init_mask_pattern.format(side))
                cropped_init_mask_file = os.path.join(file_test_patient_dir,
                                                      crop_init_mask_pattern.format(side))
                if (not os.path.exists(cropped_init_mask_file)) or set_new_roi_mask is True:
                    crop_image(init_mask_filepath, test_crop_mask, cropped_init_mask_file)

            # save the cropped labels (test_roi)
            for side in ['left', 'right']:
                for ver in ['2', '3']:
                    label_filepath = os.path.join(dataset_path, test_patient_id, 'gt', label_pattern.format(side, ver))
                    if os.path.exists(label_filepath):
                        break
                    else:
                        if ver == '3':
                            print ('No Found %s ground truth label' % side)
                            ver = 'unknown'
                cropped_tst_label_file = os.path.join(file_test_patient_dir,
                                                      crop_tst_label_pattern.format(side, ver))
                if (not os.path.exists(cropped_tst_label_file)) or set_new_roi_mask is True:
                    crop_image(label_filepath, test_crop_mask, cropped_tst_label_file)

            # load test images in test_patient_lst and setting roi
            test_image_modal_lst = []
            test_fname_modality_lst = []
            for idx in modality_idx:
                test_img_path = os.path.join(dataset_path, test_patient_id, 'images', img_pattern[idx])
                print(test_img_path)
                test_fname_modality_lst.append(test_img_path)
                test_vol = read_volume(test_img_path)
                if preprocess_tst == 2 or preprocess_tst == 3 or preprocess_tst == 5:
                    test_vol = normalize_image(test_vol, [0, 2 ** 8])
                if np.size(np.shape(test_vol)) == 4:
                    test_crop_vol = test_crop_mask.crop(test_vol[:, :, :, 0])
                else:
                    test_crop_vol = test_crop_mask.crop(test_vol)
                test_image_modal_lst.append(test_crop_vol)
            test_fname_lst.append(test_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                test_img_path = os.path.join(dataset_path, test_patient_id, 'images', img_pattern[idx])
                crop_test_img_path = os.path.join(file_test_patient_dir, crop_tst_img_pattern[idx])
                if (not os.path.exists(crop_test_img_path)) or set_new_roi_mask is True:
                    crop_image(test_img_path, test_crop_mask, crop_test_img_path)
            test_crop_array = np.zeros((1, num_modality) + test_crop_vol.shape[0:3])
            for idx in range(num_modality):
                test_crop_array[0, idx] = test_image_modal_lst[idx]
            test_img_lst.append(test_crop_array)

    return train_img_lst, label_lst, test_img_lst, train_fname_lst, label_fname_lst, test_fname_lst


def read_tha_dataset_unseen(gen_conf, train_conf, train_patient_lst, test_patient_lst, preprocess_trn, preprocess_tst,
                         file_output_dir, is_scaling, is_reg, roi_pos):
    dataset = train_conf['dataset']
    num_epochs = train_conf['num_epochs']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    mode = gen_conf['validation_mode']
    img_pattern = dataset_info['image_name_pattern']
    img_resampled_name_pattern = dataset_info['image_resampled_name_pattern']

    label_pattern = dataset_info['label_pattern']
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    init_mask_pattern = dataset_info['initial_mask_pattern']
    init_reg_mask_pattern = dataset_info['initial_reg_mask_pattern']

    set_new_roi_mask = dataset_info['set_new_roi_mask']
    margin_crop_mask = dataset_info['margin_crop_mask']

    crop_trn_img_pattern = dataset_info['crop_trn_image_name_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    crop_trn_label_pattern = dataset_info['crop_trn_label_pattern']
    crop_tst_label_pattern = dataset_info['crop_tst_label_pattern']

    crop_init_mask_pattern = dataset_info['crop_initial_mask_pattern']
    crop_init_reg_mask_pattern = dataset_info['crop_initial_reg_mask_pattern']
    train_roi_mask_pattern = dataset_info['train_roi_mask_pattern']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']

    modality_idx = []
    for m in modality: # the order of modality images should be fixed due to the absolute path
        if m == 'T1':
            modality_idx.append(0)
        if m == 'B0':
            modality_idx.append(1)
        if m == 'FA':
            modality_idx.append(2)

    is_low_res = False # initial setting
    # load multi-modal images in train_patient_lst and setting roi
    train_img_lst = []
    label_lst = []
    train_fname_lst = []
    label_fname_lst = []
    if train_patient_lst[0] is not None and num_epochs != 0 and mode != '2':
        for train_patient_id in train_patient_lst:

            file_train_patient_dir = os.path.join(file_output_dir, train_patient_id)
            if not os.path.exists(file_train_patient_dir):
                os.makedirs(os.path.join(file_output_dir, train_patient_id))

            train_roi_mask_file = os.path.join(file_train_patient_dir, train_roi_mask_pattern)
            if os.path.exists(train_roi_mask_file) and set_new_roi_mask is True:
                os.remove(train_roi_mask_file)

            # load label images in train_patient_lst and setting roi
            label_vol_lst = []
            label_fname_side_lst = []
            for side in ['left', 'right']:  # left + right
                for ver in ['2','3']:
                    label_filepath = os.path.join(dataset_path, train_patient_id, 'gt', label_pattern.format(side, ver))
                    if os.path.exists(label_filepath):
                        break
                    else:
                        if ver == '3':
                            print('No Found %s ground truth label' % side)
                            exit()
                print(label_filepath)
                label_fname_side_lst.append(label_filepath)
                label_data = read_volume_data(label_filepath)
                label_vol = label_data.get_data()
                if np.size(np.shape(label_vol)) == 4:
                    label_vol_lst.append(label_vol[:, :, :, 0])
                else:
                    label_vol_lst.append(label_vol)
            label_fname_lst.append(label_fname_side_lst)
            label_merge_vol = label_vol_lst[0] + label_vol_lst[1]

            # crop the roi of the image
            train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                       train_roi_mask_file,
                                                                                       label_data, margin_crop_mask)
            label_crop_vol = train_crop_mask.crop(label_merge_vol)

            # save the cropped labels (train_roi)
            for side in ['left', 'right']:
                for ver in ['2', '3']:
                    label_filepath = os.path.join(dataset_path, train_patient_id, 'gt', label_pattern.format(side, ver))
                    if os.path.exists(label_filepath):
                        break
                    else:
                        if ver == '3':
                            print('No Found %s ground truth label' % side)
                            exit()
                cropped_trn_label_file = os.path.join(file_output_dir, train_patient_id,
                                                      crop_trn_label_pattern.format(side, ver))
                if (not os.path.exists(cropped_trn_label_file)) or set_new_roi_mask is True:
                    crop_image(label_filepath, train_crop_mask, cropped_trn_label_file)

            label_crop_array = np.zeros((1, 1) + label_crop_vol.shape[0:3])
            label_crop_array[0, 0] = label_crop_vol
            label_lst.append(label_crop_array)

            # load training images in train_patient_lst and setting roi
            train_image_modal_lst= []
            train_fname_modality_lst = []
            for idx in modality_idx: #T1, B0, or FA
                training_img_path = os.path.join(dataset_path, train_patient_id, 'images', img_pattern[idx])
                print(training_img_path)
                train_fname_modality_lst.append(training_img_path)
                training_vol = read_volume(training_img_path)
                if preprocess_trn == 2 or preprocess_trn == 3 or preprocess_trn == 5:
                    training_vol = normalize_image(training_vol, [0, 2 ** 8])
                if np.size(np.shape(training_vol)) == 4:
                    train_crop_vol = train_crop_mask.crop(training_vol[:, :, :, 0])
                else:
                    train_crop_vol = train_crop_mask.crop(training_vol)
                train_image_modal_lst.append(train_crop_vol)
            train_fname_lst.append(train_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                train_img_path = os.path.join(dataset_path, train_patient_id, 'images', img_pattern[idx])
                crop_train_img_path = os.path.join(file_output_dir, train_patient_id, crop_trn_img_pattern[idx])
                if (not os.path.exists(crop_train_img_path)) or set_new_roi_mask is True:
                    crop_image(train_img_path, train_crop_mask, crop_train_img_path)
            train_crop_array = np.zeros((1, num_modality) + train_crop_vol.shape[0:3])
            for idx in range(num_modality):
                train_crop_array[0, idx] = train_image_modal_lst[idx]
            train_img_lst.append(train_crop_array)

    # load multi-modal images in test_patient_lst and setting roi
    test_img_lst = []
    test_fname_lst = []
    if test_patient_lst[0] is not None:
        for test_patient_id in test_patient_lst:

            file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
            if not os.path.exists(file_test_patient_dir):
                os.makedirs(os.path.join(file_output_dir, test_patient_id))

            test_roi_mask_file = os.path.join(file_test_patient_dir, test_roi_mask_pattern)
            if os.path.exists(test_roi_mask_file) and set_new_roi_mask is True:
                os.remove(test_roi_mask_file)

            test_fname_modality_lst_for_init_reg = []
            for idx in modality_idx:
                test_img_path = os.path.join(dataset_path, test_patient_id, 'images', img_pattern[idx])
                test_fname_modality_lst_for_init_reg.append(test_img_path)
            test_t1_img_path = test_fname_modality_lst_for_init_reg[0]

            # load initial mask (use fusion output or later linearly registered from ref. training image) for setting roi
            init_mask_path = os.path.join(dataset_path, test_patient_id, 'fusion')
            if os.path.exists(init_mask_path):
                init_mask_vol_lst = []
                for side in ['left', 'right']:  # left + right
                    init_mask_filepath = os.path.join(dataset_path, test_patient_id, 'fusion',
                                                      init_mask_pattern.format(side))
                    print(init_mask_filepath)
                    init_mask_data = read_volume_data(init_mask_filepath)
                    init_mask_vol = init_mask_data.get_data()
                    if np.size(np.shape(init_mask_vol)) == 4:
                        init_mask_vol_lst.append(init_mask_vol[:, :, :, 0])
                    else:
                        init_mask_vol_lst.append(init_mask_vol)
                init_mask_merge_vol = init_mask_vol_lst[0] + init_mask_vol_lst[1]

            else: # localize target via initial registration
                label_fname_lst_for_init_reg = []
                train_fname_lst_for_init_reg = []
                for train_patient_id in train_patient_lst:
                    train_fname_modality_lst_for_init_reg = []
                    for idx in modality_idx:  # T1, B0, or FA
                        training_img_path = os.path.join(dataset_path, train_patient_id, 'images', img_pattern[idx])
                        train_fname_modality_lst_for_init_reg.append(training_img_path)
                    train_fname_lst_for_init_reg.append(train_fname_modality_lst_for_init_reg)

                    label_fname_side_lst_for_init_reg = []
                    for side in ['left', 'right']:  # left + right
                        for ver in ['2', '3']:
                            label_filepath = os.path.join(dataset_path, train_patient_id, 'gt',
                                                          label_pattern.format(side, ver))
                            if os.path.exists(label_filepath):
                                break
                            else:
                                if ver == '3':
                                    print('No Found %s ground truth label' % side)
                                    exit()
                        label_fname_side_lst_for_init_reg.append(label_filepath)
                    label_fname_lst_for_init_reg.append(label_fname_side_lst_for_init_reg)
                init_mask_merge_vol, init_mask_data, test_t1_img_path, is_low_res, roi_pos = \
                    localize_target(file_output_dir,
                                    modality_idx,
                                    train_patient_lst,
                                    train_fname_lst_for_init_reg,
                                    test_patient_id,
                                    test_fname_modality_lst_for_init_reg,
                                    label_fname_lst_for_init_reg,
                                    img_pattern[0],
                                    img_resampled_name_pattern,
                                    init_reg_mask_pattern,
                                    'tha',
                                    is_low_res,
                                    roi_pos,
                                    is_scaling,
                                    is_reg)

            if init_mask_merge_vol.tolist():
                # crop the roi of the image
                test_crop_mask = find_crop_mask(test_roi_mask_file) or compute_crop_mask(init_mask_merge_vol,
                                                                                         test_roi_mask_file,
                                                                                         init_mask_data,
                                                                                        margin_crop_mask)
            else:
                test_t1_img_data = read_volume_data(test_t1_img_path)
                test_t1_image = BrainImage(test_t1_img_path, None)
                test_vol = test_t1_image.nii_data_normalized(bits=8)
                test_crop_mask = compute_crop_mask_manual(test_vol, test_roi_mask_file, test_t1_img_data,
                                                          roi_pos[0], roi_pos[1]) # PD091 for tha seg

            # save the cropped initial masks
            for side in ['left', 'right']:
                init_mask_path = os.path.join(dataset_path, test_patient_id, 'fusion')
                if os.path.exists(init_mask_path):
                    init_mask_filepath = os.path.join(dataset_path, test_patient_id, 'fusion',
                                                      init_mask_pattern.format(side))
                    if os.path.exists(init_mask_filepath):
                        cropped_init_mask_file = os.path.join(file_test_patient_dir,
                                                              crop_init_mask_pattern.format(side))
                        if (not os.path.exists(cropped_init_mask_file)) or set_new_roi_mask is True:
                            crop_image(init_mask_filepath, test_crop_mask, cropped_init_mask_file)
                else:
                    init_mask_filepath = os.path.join(file_test_patient_dir, 'init_reg',
                                                          init_reg_mask_pattern.format(side, 'tha'))
                    if os.path.exists(init_mask_filepath):
                        cropped_init_reg_mask_file = os.path.join(file_test_patient_dir,
                                                              crop_init_reg_mask_pattern.format(side))
                        if (not os.path.exists(cropped_init_reg_mask_file)) or set_new_roi_mask is True:
                            crop_image(init_mask_filepath, test_crop_mask, cropped_init_reg_mask_file)

            # save the cropped labels (test_roi)
            for side in ['left', 'right']:
                for ver in ['2', '3']:
                    label_filepath = os.path.join(dataset_path, test_patient_id, 'gt', label_pattern.format(side, ver))
                    if os.path.exists(label_filepath) and os.path.exists(init_mask_filepath):
                        cropped_tst_label_file = os.path.join(file_test_patient_dir,
                                                              crop_tst_label_pattern.format(side, ver))
                        if (not os.path.exists(cropped_tst_label_file)) or set_new_roi_mask is True:
                            crop_image(label_filepath, test_crop_mask, cropped_tst_label_file)
                        break
                    else:
                        if ver == '3':
                            print ('No found %s ground truth label' % side)

            # load test images in test_patient_lst and setting roi
            test_image_modal_lst = []
            test_fname_modality_lst = []
            for idx in modality_idx:
                if is_low_res is True:
                    test_img_path = os.path.join(file_output_dir, test_patient_id, img_resampled_name_pattern[idx])
                else:
                    test_img_path = os.path.join(dataset_path, test_patient_id, 'images', img_pattern[idx])
                print(test_img_path)
                test_fname_modality_lst.append(test_img_path)
                test_vol = read_volume(test_img_path)
                if preprocess_tst == 2 or preprocess_tst == 3 or preprocess_tst == 5:
                    test_vol = normalize_image(test_vol, [0, 2 ** 8])
                if np.size(np.shape(test_vol)) == 4:
                    test_crop_vol = test_crop_mask.crop(test_vol[:, :, :, 0])
                else:
                    test_crop_vol = test_crop_mask.crop(test_vol)
                test_image_modal_lst.append(test_crop_vol)
            test_fname_lst.append(test_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                if is_low_res is True:
                    test_img_path = os.path.join(file_output_dir, test_patient_id, img_resampled_name_pattern[idx])
                else:
                    test_img_path = os.path.join(dataset_path, test_patient_id, 'images', img_pattern[idx])
                crop_test_img_path = os.path.join(file_test_patient_dir, crop_tst_img_pattern[idx])
                if (not os.path.exists(crop_test_img_path)) or set_new_roi_mask is True:
                    crop_image(test_img_path, test_crop_mask, crop_test_img_path)
            test_crop_array = np.zeros((1, num_modality) + test_crop_vol.shape[0:3])
            for idx in range(num_modality):
                test_crop_array[0, idx] = test_image_modal_lst[idx]
            test_img_lst.append(test_crop_array)

    return train_img_lst, label_lst, test_img_lst, train_fname_lst, label_fname_lst, test_fname_lst, is_low_res

# processing smaller size
def read_tha_dataset_v2(gen_conf, train_conf, train_patient_lst, test_patient_lst, preprocess_trn, preprocess_tst,
                         file_output_dir):
    dataset = train_conf['dataset']
    num_epochs = train_conf['num_epochs']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    mode = gen_conf['validation_mode']
    img_pattern = dataset_info['image_name_pattern']
    label_pattern = dataset_info['label_pattern']
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    init_mask_pattern = dataset_info['initial_mask_pattern']

    set_new_roi_mask = dataset_info['set_new_roi_mask']
    margin_crop_mask = dataset_info['margin_crop_mask']

    crop_trn_img_pattern = dataset_info['crop_trn_image_name_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']

    crop_trn_img_down_pattern = dataset_info['crop_trn_image_downsampled_name_pattern']
    crop_tst_img_down_pattern = dataset_info['crop_tst_image_downsampled_name_pattern']

    crop_trn_label_pattern = dataset_info['crop_trn_label_pattern']
    crop_tst_label_pattern = dataset_info['crop_tst_label_pattern']

    crop_trn_label_down_pattern = dataset_info['crop_trn_label_downsampled_pattern']
    crop_tst_label_down_pattern = dataset_info['crop_tst_label_downsampled_pattern']

    crop_init_mask_pattern = dataset_info['crop_initial_mask_pattern']

    train_roi_mask_pattern = dataset_info['train_roi_mask_pattern']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']
    file_format = dataset_info['format']

    scale = 0.5

    modality_idx = []
    for m in modality:
        if m == 'T1':
            modality_idx.append(0)
        if m == 'B0':
            modality_idx.append(1)
        if m == 'FA':
            modality_idx.append(2)

    # load multi-modal images in train_patient_lst and setting roi
    train_img_lst = []
    label_lst = []
    train_fname_lst = []
    label_fname_lst = []
    if train_patient_lst[0] is not None and num_epochs != 0 and mode != '2':
        for train_patient_id in train_patient_lst:

            file_train_patient_dir = os.path.join(file_output_dir, train_patient_id)
            if not os.path.exists(file_train_patient_dir):
                os.makedirs(os.path.join(file_output_dir, train_patient_id))

            train_roi_mask_file = os.path.join(file_train_patient_dir, train_roi_mask_pattern)
            if os.path.exists(train_roi_mask_file) and set_new_roi_mask is True:
                os.remove(train_roi_mask_file)

            # load label images in train_patient_lst and setting roi
            label_vol_lst = []
            label_fname_side_lst = []
            for side in ['left', 'right']:  # left + right
                for ver in ['2','3']:
                    label_filepath = os.path.join(dataset_path, train_patient_id, 'gt', label_pattern.format(side, ver))
                    if os.path.exists(label_filepath):
                        break
                    else:
                        if ver == '3':
                            print('No Found %s ground truth label' % side)
                            exit()
                print(label_filepath)
                label_fname_side_lst.append(label_filepath)
                label_data = read_volume_data(label_filepath)
                label_vol = label_data.get_data()
                if np.size(np.shape(label_vol)) == 4:
                    label_vol_lst.append(label_vol[:, :, :, 0])
                else:
                    label_vol_lst.append(label_vol)
            label_fname_lst.append(label_fname_side_lst)
            label_merge_vol = label_vol_lst[0] + label_vol_lst[1]

            # crop the roi of the image
            train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                       train_roi_mask_file,
                                                                                       label_data, margin_crop_mask)
            label_crop_vol = train_crop_mask.crop(label_merge_vol)

            # downsample the cropped labels
            label_crop_vol = zoom(label_crop_vol, scale)
            print(label_crop_vol.shape)
            label_crop_array = np.zeros((1, 1) + label_crop_vol.shape[0:3])
            label_crop_array[0, 0] = label_crop_vol
            label_lst.append(label_crop_array)

            # save the cropped labels (train_roi)
            for side in ['left', 'right']:
                for ver in ['2','3']:
                    label_filepath = os.path.join(dataset_path, train_patient_id, 'gt', label_pattern.format(side, ver))
                    if os.path.exists(label_filepath):
                        break
                    else:
                        if ver == '3':
                            print('No Found %s ground truth label' % side)
                            exit()
                cropped_trn_label_file = os.path.join(file_output_dir, train_patient_id,
                                                      crop_trn_label_pattern.format(side, ver))
                if (not os.path.exists(cropped_trn_label_file)) or set_new_roi_mask is True:
                    crop_image(label_filepath, train_crop_mask, cropped_trn_label_file)

            # save downsampled labels
            crop_trn_label_down_filepath = os.path.join(file_output_dir, train_patient_id,
                                                        crop_trn_label_down_pattern.format(ver))
            __save_volume(label_crop_vol, label_data, crop_trn_label_down_filepath, file_format, is_compressed=True)

            # load training images in train_patient_lst and setting roi
            train_image_modal_lst= []
            train_fname_modality_lst = []
            for idx in modality_idx: #T1, B0, or FA
                training_img_path = os.path.join(dataset_path, train_patient_id, 'images', img_pattern[idx])
                print(training_img_path)
                train_fname_modality_lst.append(training_img_path)
                training_vol_data = read_volume_data(training_img_path)
                training_vol = training_vol_data.get_data()
                #training_vol = read_volume(training_img_path)
                if preprocess_trn == 2 or preprocess_trn == 3 or preprocess_trn == 5:
                    training_vol = normalize_image(training_vol, [0, 2 ** 8])
                if np.size(np.shape(training_vol)) == 4:
                    train_crop_vol = train_crop_mask.crop(training_vol[:, :, :, 0])
                else:
                    train_crop_vol = train_crop_mask.crop(training_vol)

                # downsample the cropped training images
                train_crop_vol = zoom(train_crop_vol, scale)
                train_image_modal_lst.append(train_crop_vol)

                # save downsampled training images
                crop_trn_image_down_filepath = os.path.join(file_output_dir, train_patient_id,
                                                            crop_trn_img_down_pattern[idx])
                __save_volume(train_crop_vol, training_vol_data, crop_trn_image_down_filepath, file_format,
                              is_compressed=True)

            # save the cropped image
            for idx in modality_idx:
                train_img_path = os.path.join(dataset_path, train_patient_id, 'images', img_pattern[idx])
                crop_train_img_path = os.path.join(file_output_dir, train_patient_id, crop_trn_img_pattern[idx])
                if (not os.path.exists(crop_train_img_path)) or set_new_roi_mask is True:
                    crop_image(train_img_path, train_crop_mask, crop_train_img_path)

            train_fname_lst.append(train_fname_modality_lst)
            train_crop_array = np.zeros((1, num_modality) + train_crop_vol.shape[0:3])
            for idx in range(num_modality):
                train_crop_array[0, idx] = train_image_modal_lst[idx]
            train_img_lst.append(train_crop_array)

    # load multi-modal images in test_patient_lst and setting roi
    test_img_lst = []
    test_fname_lst = []
    if test_patient_lst[0] is not None:
        for test_patient_id in test_patient_lst:

            file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
            if not os.path.exists(file_test_patient_dir):
                os.makedirs(os.path.join(file_output_dir, test_patient_id))

            test_roi_mask_file = os.path.join(file_test_patient_dir, test_roi_mask_pattern)
            if os.path.exists(test_roi_mask_file) and set_new_roi_mask is True:
                os.remove(test_roi_mask_file)

            # load initial mask (use fusion output or later linearly registered from ref. training image) for setting roi
            init_mask_vol_lst = []
            for side in ['left', 'right']:  # left + right
                init_mask_filepath = os.path.join(dataset_path, test_patient_id, 'fusion', init_mask_pattern.format(side))
                print(init_mask_filepath)
                init_mask_data = read_volume_data(init_mask_filepath)
                init_mask_vol = init_mask_data.get_data()
                if np.size(np.shape(init_mask_vol)) == 4:
                    init_mask_vol_lst.append(init_mask_vol[:, :, :, 0])
                else:
                    init_mask_vol_lst.append(init_mask_vol)
            init_mask_merge_vol = init_mask_vol_lst[0] + init_mask_vol_lst[1]

            # crop the roi of the image
            test_crop_mask = find_crop_mask(test_roi_mask_file) or compute_crop_mask(init_mask_merge_vol,
                                                                                     test_roi_mask_file,
                                                                                     init_mask_data,
                                                                                     margin_crop_mask)

            # save the cropped initial masks
            for side in ['left', 'right']:
                init_mask_filepath = os.path.join(dataset_path, test_patient_id, 'fusion',
                                                  init_mask_pattern.format(side))
                cropped_init_mask_file = os.path.join(file_test_patient_dir,
                                                      crop_init_mask_pattern.format(side))
                if (not os.path.exists(cropped_init_mask_file)) or set_new_roi_mask is True:
                    crop_image(init_mask_filepath, test_crop_mask, cropped_init_mask_file)

            # save the cropped labels (test_roi)
            for side in ['left', 'right']:
                for ver in ['2','3']:
                    label_filepath = os.path.join(dataset_path, test_patient_id, 'gt', label_pattern.format(side, ver))
                    if os.path.exists(label_filepath):
                        continue
                    else:
                        if ver == '3':
                            print('No Found %s ground truth label' % side)
                            ver = 'unknown'
                cropped_tst_label_file = os.path.join(file_test_patient_dir,
                                                      crop_tst_label_pattern.format(side, ver))
                if (not os.path.exists(cropped_tst_label_file)) or set_new_roi_mask is True:
                    crop_image(label_filepath, test_crop_mask, cropped_tst_label_file)

            # load test images in test_patient_lst and setting roi
            test_image_modal_lst = []
            test_fname_modality_lst = []
            for idx in modality_idx:
                test_img_path = os.path.join(dataset_path, test_patient_id, 'images', img_pattern[idx])
                print(test_img_path)
                test_fname_modality_lst.append(test_img_path)
                #test_vol = read_volume(test_img_path)
                test_vol_data = read_volume_data(test_img_path)
                test_vol = test_vol_data.get_data()
                if preprocess_tst == 2 or preprocess_tst == 3 or preprocess_tst == 5:
                    test_vol = normalize_image(test_vol, [0, 2 ** 8])
                if np.size(np.shape(test_vol)) == 4:
                    test_crop_vol = test_crop_mask.crop(test_vol[:, :, :, 0])
                else:
                    test_crop_vol = test_crop_mask.crop(test_vol)

                # downsample the cropped test images
                test_crop_vol = zoom(test_crop_vol, scale)
                test_image_modal_lst.append(test_crop_vol)

                # save downsampled training images
                crop_tst_image_down_filepath = os.path.join(file_output_dir, test_patient_id,
                                                            crop_tst_img_down_pattern[idx])
                __save_volume(test_crop_vol, test_vol_data, crop_tst_image_down_filepath, file_format,
                              is_compressed=True)

            test_fname_lst.append(test_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                test_img_path = os.path.join(dataset_path, test_patient_id, 'images', img_pattern[idx])
                crop_test_img_path = os.path.join(file_test_patient_dir, crop_tst_img_pattern[idx])
                if (not os.path.exists(crop_test_img_path)) or set_new_roi_mask is True:
                    crop_image(test_img_path, test_crop_mask, crop_test_img_path)

            test_crop_array = np.zeros((1, num_modality) + test_crop_vol.shape[0:3])
            for idx in range(num_modality):
                test_crop_array[0, idx] = test_image_modal_lst[idx]
            test_img_lst.append(test_crop_array)

    return train_img_lst, label_lst, test_img_lst, train_fname_lst, label_fname_lst, test_fname_lst


def read_dentate_dataset(gen_conf, train_conf, train_patient_lst, test_patient_lst, preprocess_trn, preprocess_tst,
                         file_output_dir):
    dataset = train_conf['dataset']
    num_epochs = train_conf['num_epochs']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    mode = gen_conf['validation_mode']
    img_pattern = dataset_info['image_name_pattern']
    label_pattern = dataset_info['manual_corrected_pattern']
    label_v2_pattern = dataset_info['manual_corrected_dentate_v2_pattern']
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    init_mask_pattern = dataset_info['initial_mask_pattern']

    set_new_roi_mask = dataset_info['set_new_roi_mask']
    margin_crop_mask = dataset_info['margin_crop_mask']

    crop_trn_img_pattern = dataset_info['crop_trn_image_name_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    crop_trn_label_pattern = dataset_info['crop_trn_manual_corrected_pattern']
    crop_tst_label_pattern = dataset_info['crop_tst_manual_corrected_pattern']
    crop_trn_label_v2_pattern = dataset_info['crop_trn_manual_dentate_v2_corrected_pattern']
    crop_tst_label_v2_pattern = dataset_info['crop_tst_manual_dentate_v2_corrected_pattern']
    crop_init_mask_pattern = dataset_info['crop_initial_mask_pattern']

    train_roi_mask_pattern = dataset_info['train_roi_mask_pattern']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']

    modality_idx = []
    for m in modality:
        if m == 'B0':
            modality_idx.append(0)
        if m in ['T1', 'QSM']:
            modality_idx.append(1)
        if m == 'LP':
            modality_idx.append(2)
        if m == 'FA':
            modality_idx.append(3)

    # load multi-modal images in train_patient_lst and setting roi
    train_img_lst = []
    label_lst = []
    train_fname_lst = []
    label_fname_lst = []
    if train_patient_lst[0] is not None and num_epochs != 0 and mode != '2':
        for train_patient_id in train_patient_lst:

            file_patient_dir = file_output_dir + '/' + train_patient_id
            if not os.path.exists(file_patient_dir):
                os.makedirs(os.path.join(file_output_dir, train_patient_id))

            train_roi_mask_file = file_output_dir + train_roi_mask_pattern.format(train_patient_id)
            if os.path.exists(train_roi_mask_file) and set_new_roi_mask is True:
                os.remove(train_roi_mask_file)

            # load label images in train_patient_lst and setting roi
            label_vol_lst = []
            label_fname_side_lst = []
            for side in ['left', 'right']:  # left + right
                # check patiets with updated labels
                updated_label_filepath = dataset_path + train_patient_id + label_v2_pattern.format(train_patient_id,
                                                                                                   side)
                if os.path.exists(updated_label_filepath):
                    label_filepath = updated_label_filepath
                else:
                    label_filepath = dataset_path + train_patient_id + label_pattern.format(train_patient_id, side)
                print(label_filepath)
                label_fname_side_lst.append(label_filepath)
                label_data = read_volume_data(label_filepath)
                label_vol = label_data.get_data()
                if np.size(np.shape(label_vol)) == 4:
                    label_vol_lst.append(label_vol[:, :, :, 0])
                else:
                    label_vol_lst.append(label_vol)
            label_fname_lst.append(label_fname_side_lst)
            label_merge_vol = label_vol_lst[0] + label_vol_lst[1]

            # crop the roi of the image
            train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                       train_roi_mask_file,
                                                                                       label_data, margin_crop_mask)
            label_crop_vol = train_crop_mask.crop(label_merge_vol)

            # save the cropped labels (train_roi)
            for side in ['left', 'right']:
                updated_label_filepath = dataset_path + train_patient_id + label_v2_pattern.format(train_patient_id,
                                                                                                   side)
                if os.path.exists(updated_label_filepath):
                    label_filepath = updated_label_filepath
                    cropped_trn_label_file = file_output_dir + '/' + train_patient_id + crop_trn_label_v2_pattern.format(
                    train_patient_id, side)
                else:
                    label_filepath = dataset_path + train_patient_id + label_pattern.format(train_patient_id, side)
                    cropped_trn_label_file = file_output_dir + '/' + train_patient_id + crop_trn_label_pattern.format(
                    train_patient_id, side)
                if (not os.path.exists(cropped_trn_label_file)) or set_new_roi_mask is True:
                    crop_image(label_filepath, train_crop_mask, cropped_trn_label_file)

            label_crop_array = np.zeros((1, 1) + label_crop_vol.shape[0:3])
            label_crop_array[0, 0] = label_crop_vol
            label_lst.append(label_crop_array)

            # load training images in train_patient_lst and setting roi
            train_image_modal_lst= []
            train_fname_modality_lst = []
            for idx in modality_idx:
                training_img_path = dataset_path + train_patient_id + '/' + img_pattern[idx].format(train_patient_id)
                print(training_img_path)
                train_fname_modality_lst.append(training_img_path)
                training_vol = read_volume(training_img_path)
                if preprocess_trn == 2 or preprocess_trn == 3 or preprocess_trn == 5:
                    training_vol = normalize_image(training_vol, [0, 2 ** 8])
                if np.size(np.shape(training_vol)) == 4:
                    train_crop_vol = train_crop_mask.crop(training_vol[:, :, :, 0])
                else:
                    train_crop_vol = train_crop_mask.crop(training_vol)
                train_image_modal_lst.append(train_crop_vol)
            train_fname_lst.append(train_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                train_img_path = dataset_path + train_patient_id + '/' + img_pattern[idx].format(train_patient_id)
                crop_train_img_path = file_output_dir + '/' + train_patient_id + '/' + \
                                           crop_trn_img_pattern[idx].format(train_patient_id)
                if (not os.path.exists(crop_train_img_path)) or set_new_roi_mask is True:
                    crop_image(train_img_path, train_crop_mask, crop_train_img_path)

            train_crop_array = np.zeros((1, num_modality) + train_crop_vol.shape[0:3])
            for idx in range(num_modality):
                train_crop_array[0, idx] = train_image_modal_lst[idx]
            train_img_lst.append(train_crop_array)

    # load multi-modal images in test_patient_lst and setting roi
    test_img_lst = []
    test_fname_lst = []
    if test_patient_lst[0] is not None:
        for test_patient_id in test_patient_lst:

            file_patient_dir = file_output_dir + '/' + test_patient_id
            if not os.path.exists(file_patient_dir):
                os.makedirs(os.path.join(file_output_dir, test_patient_id))

            test_roi_mask_file = file_output_dir + test_roi_mask_pattern.format(test_patient_id)
            if os.path.exists(test_roi_mask_file) and set_new_roi_mask is True:
                os.remove(test_roi_mask_file)

            # load initial mask (SUIT output or linearly registered from ref. training image) for setting roi
            init_mask_vol_lst = []
            for side in ['left', 'right']:  # left + right
                init_mask_filepath = dataset_path + test_patient_id + init_mask_pattern.format(test_patient_id, side)
                print(init_mask_filepath)
                init_mask_data = read_volume_data(init_mask_filepath)
                init_mask_vol = init_mask_data.get_data()
                if np.size(np.shape(init_mask_vol)) == 4:
                    init_mask_vol_lst.append(init_mask_vol[:, :, :, 0])
                else:
                    init_mask_vol_lst.append(init_mask_vol)
            init_mask_merge_vol = init_mask_vol_lst[0] + init_mask_vol_lst[1]

            # crop the roi of the image
            test_crop_mask = find_crop_mask(test_roi_mask_file) or compute_crop_mask(init_mask_merge_vol,
                                                                                     test_roi_mask_file,
                                                                                     init_mask_data,
                                                                                     margin_crop_mask)

            # save the cropped initial masks
            for side in ['left', 'right']:
                init_mask_filepath = dataset_path + test_patient_id + init_mask_pattern.format(test_patient_id, side)
                cropped_init_mask_file = file_output_dir + '/' + test_patient_id + \
                                         crop_init_mask_pattern.format(test_patient_id, side)
                if (not os.path.exists(cropped_init_mask_file)) or set_new_roi_mask is True:
                    crop_image(init_mask_filepath, test_crop_mask, cropped_init_mask_file)

            # save the cropped labels (test_roi)
            for side in ['left', 'right']:
                updated_label_filepath = dataset_path + test_patient_id + label_v2_pattern.format(test_patient_id, side)
                if os.path.exists(updated_label_filepath):
                    label_filepath = updated_label_filepath
                    cropped_tst_label_file = file_output_dir + '/' + test_patient_id + \
                                             crop_tst_label_v2_pattern.format(test_patient_id, side)
                else:
                    label_filepath = dataset_path + test_patient_id + label_pattern.format(test_patient_id, side)
                    cropped_tst_label_file = file_output_dir + '/'  + test_patient_id + \
                                         crop_tst_label_pattern.format(test_patient_id, side)
                if (not os.path.exists(cropped_tst_label_file)) or set_new_roi_mask is True:
                    crop_image(label_filepath, test_crop_mask, cropped_tst_label_file)

            # load test images in test_patient_lst and setting roi
            test_image_modal_lst = []
            test_fname_modality_lst = []
            for idx in modality_idx:
                test_img_path = dataset_path + test_patient_id + '/' + img_pattern[idx].format(test_patient_id)
                print(test_img_path)
                test_fname_modality_lst.append(test_img_path)
                test_vol = read_volume(test_img_path)
                if preprocess_tst == 2 or preprocess_tst == 3 or preprocess_tst == 5:
                    test_vol = normalize_image(test_vol, [0, 2 ** 8])
                if np.size(np.shape(test_vol)) == 4:
                    test_crop_vol = test_crop_mask.crop(test_vol[:, :, :, 0])
                else:
                    test_crop_vol = test_crop_mask.crop(test_vol)
                test_image_modal_lst.append(test_crop_vol)
            test_fname_lst.append(test_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                test_img_path = dataset_path + test_patient_id + '/' + img_pattern[idx].format(test_patient_id)
                crop_test_img_path = file_output_dir + '/' + test_patient_id + '/' + \
                                     crop_tst_img_pattern[idx].format(test_patient_id)
                if (not os.path.exists(crop_test_img_path)) or set_new_roi_mask is True:
                    crop_image(test_img_path, test_crop_mask, crop_test_img_path)

            test_crop_array = np.zeros((1, num_modality) + test_crop_vol.shape[0:3])
            for idx in range(num_modality):
                test_crop_array[0, idx] = test_image_modal_lst[idx]
            test_img_lst.append(test_crop_array)

    return train_img_lst, label_lst, test_img_lst, train_fname_lst, label_fname_lst, test_fname_lst


def read_dentate_interposed_dataset(gen_conf, train_conf, train_patient_lst, test_patient_lst, preprocess_trn,
                                    preprocess_tst, file_output_dir, target):
    dataset = train_conf['dataset']
    num_epochs = train_conf['num_epochs']
    approach = train_conf['approach']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    mode = gen_conf['validation_mode']
    img_pattern = dataset_info['image_name_pattern']
    dentate_label_pattern = dataset_info['manual_corrected_dentate_v2_pattern']
    interposed_label_pattern = dataset_info['manual_corrected_interposed_v2_pattern']

    modality = dataset_info['image_modality']
    num_modality = len(modality)
    init_dentate_mask_pattern = dataset_info['initial_mask_pattern']
    init_interposed_mask_pattern_thres = dataset_info['initial_interposed_mask_pattern_thres']
    init_interposed_mask_pattern_mask = dataset_info['initial_interposed_mask_pattern_mask']

    set_new_roi_mask = dataset_info['set_new_roi_mask']
    margin_crop_mask = dataset_info['margin_crop_mask']

    crop_trn_img_pattern = dataset_info['crop_trn_image_name_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']

    crop_trn_dentate_label_pattern = dataset_info['crop_trn_manual_dentate_v2_corrected_pattern']
    crop_tst_dentate_label_pattern = dataset_info['crop_tst_manual_dentate_v2_corrected_pattern']
    crop_trn_interposed_label_pattern = dataset_info['crop_trn_manual_interposed_v2_pattern']
    crop_tst_interposed_label_pattern = dataset_info['crop_tst_manual_interposed_v2_pattern']

    crop_init_dentate_mask_pattern = dataset_info['crop_initial_mask_pattern']
    crop_init_interposed_mask_pattern = dataset_info['crop_initial_interposed_mask_pattern']

    train_roi_mask_pattern = dataset_info['train_roi_mask_pattern']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']

    patient_id = dataset_info['patient_id']

    is_new_trn_label = train_conf['is_new_trn_label']
    if is_new_trn_label in [1, 2, 3]:  # 1: segmentation using a proposed network, 2: suit labels
        new_label_path = train_conf['new_label_path']
        dentate_new_label_pattern = dataset_info['trn_new_label_dentate_pattern']
        interposed_new_label_pattern = dataset_info['trn_new_label_interposed_pattern']
        crop_trn_dentate_new_label_pattern = dataset_info['crop_trn_new_label_dentate_pattern']
        crop_trn_interposed_new_label_pattern = dataset_info['crop_trn_new_label_interposed_pattern']

    modality_idx = []
    for m in modality:
        if m == 'B0':
            modality_idx.append(0)
        if m in ['T1', 'QSM']:
            modality_idx.append(1)
        if m == 'LP':
            modality_idx.append(2)
        if m == 'FA':
            modality_idx.append(3)

    #label_mapper = {0: 0, 10: 1, 150: 2}

    if len(target) == 2:
        target = 'both'

    # load multi-modal images in train_patient_lst and setting roi
    train_img_lst = []
    label_lst = []
    train_fname_lst = []
    dentate_label_fname_lst = []
    interposed_label_fname_lst = []
    if train_patient_lst[0] is not None and num_epochs != 0 and mode != '2':
        for train_patient_id in train_patient_lst:

            file_trn_patient_dir = os.path.join(file_output_dir, train_patient_id)
            if not os.path.exists(file_trn_patient_dir):
                os.makedirs(file_trn_patient_dir)

            train_roi_mask_file = os.path.join(file_output_dir, train_roi_mask_pattern.format(train_patient_id))
            if os.path.exists(train_roi_mask_file) and set_new_roi_mask is True:
                os.remove(train_roi_mask_file)

            # load label images in train_patient_lst and setting roi
            dentate_label_vol_lst = []
            dentate_label_fname_side_lst = []
            interposed_label_vol_lst = []
            interposed_label_fname_side_lst = []
            for side in ['left', 'right']:  # left + right

                if target == 'dentate':
                    if is_new_trn_label == 1:
                        dentate_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                              dentate_new_label_pattern.format(side, approach))
                    elif is_new_trn_label == 2:
                        dentate_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                              dentate_new_label_pattern.format(train_patient_id, side))
                    elif is_new_trn_label == 3:
                        if train_patient_id in patient_id:
                            dentate_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                  dentate_label_pattern.format(train_patient_id, side))
                        else:
                            dentate_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                                  dentate_new_label_pattern.format(side, approach))
                    else:
                        dentate_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                              dentate_label_pattern.format(train_patient_id, side))
                    print(dentate_label_filepath)
                    dentate_label_fname_side_lst.append(dentate_label_filepath)
                    dentate_label_data = read_volume_data(dentate_label_filepath)
                    dentate_label_vol = dentate_label_data.get_data()
                    if np.size(np.shape(dentate_label_vol)) == 4:
                        dentate_label_vol_lst.append(dentate_label_vol[:, :, :, 0])
                    else:
                        dentate_label_vol_lst.append(dentate_label_vol)

                elif target == 'interposed':
                    if is_new_trn_label == 1:
                        interposed_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                                 interposed_new_label_pattern.format(side, approach))
                    elif is_new_trn_label == 2:
                        interposed_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                                 interposed_new_label_pattern.format(train_patient_id,
                                                                                                     side))
                    elif is_new_trn_label == 3:
                        if train_patient_id in patient_id:
                            interposed_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                     interposed_label_pattern.format(train_patient_id,
                                                                                                     side))
                        else:
                            interposed_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                                     interposed_new_label_pattern.format(side, approach))
                    else:
                        interposed_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                 interposed_label_pattern.format(train_patient_id,
                                                                                                 side))
                    print(interposed_label_filepath)
                    interposed_label_fname_side_lst.append(interposed_label_filepath)
                    interposed_label_data = read_volume_data(interposed_label_filepath)
                    interposed_label_vol = interposed_label_data.get_data()
                    if np.size(np.shape(interposed_label_vol)) == 4:
                        interposed_label_vol_lst.append(interposed_label_vol[:, :, :, 0])
                    else:
                        interposed_label_vol_lst.append(interposed_label_vol)

                else:
                    if is_new_trn_label == 1:
                        dentate_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                              dentate_new_label_pattern.format(side, approach))
                        interposed_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                                 interposed_new_label_pattern.format(side, approach))
                    elif is_new_trn_label == 2:
                        dentate_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                              dentate_new_label_pattern.format(train_patient_id, side))
                        interposed_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                                 interposed_new_label_pattern.format(train_patient_id,
                                                                                                     side))
                    elif is_new_trn_label == 3:
                        if train_patient_id in patient_id:
                            dentate_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                  dentate_label_pattern.format(train_patient_id, side))
                            interposed_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                     interposed_label_pattern.format(train_patient_id,
                                                                                                     side))
                        else:
                            dentate_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                                  dentate_new_label_pattern.format(side, approach))
                            interposed_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                                     interposed_new_label_pattern.format(side, approach))

                    else:
                        dentate_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                              dentate_label_pattern.format(train_patient_id, side))
                        interposed_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                 interposed_label_pattern.format(train_patient_id,
                                                                                                 side))
                    print(dentate_label_filepath)
                    dentate_label_fname_side_lst.append(dentate_label_filepath)
                    dentate_label_data = read_volume_data(dentate_label_filepath)
                    dentate_label_vol = dentate_label_data.get_data()
                    if np.size(np.shape(dentate_label_vol)) == 4:
                        dentate_label_vol_lst.append(dentate_label_vol[:, :, :, 0])
                    else:
                        dentate_label_vol_lst.append(dentate_label_vol)

                    print(interposed_label_filepath)
                    interposed_label_fname_side_lst.append(interposed_label_filepath)
                    interposed_label_data = read_volume_data(interposed_label_filepath)
                    interposed_label_vol = interposed_label_data.get_data()
                    if np.size(np.shape(interposed_label_vol)) == 4:
                        interposed_label_vol_lst.append(interposed_label_vol[:, :, :, 0])
                    else:
                        interposed_label_vol_lst.append(interposed_label_vol)

            if target == 'dentate':
                dentate_label_fname_lst.append(dentate_label_fname_side_lst)
                dentate_label_merge_vol = dentate_label_vol_lst[0] + dentate_label_vol_lst[1]
                label_merge_vol = dentate_label_merge_vol

                # crop the roi of the image
                train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                           train_roi_mask_file,
                                                                                           dentate_label_data,
                                                                                           margin_crop_mask)

            elif target == 'interposed':
                interposed_label_fname_lst.append(interposed_label_fname_side_lst)
                interposed_label_merge_vol = interposed_label_vol_lst[0] + interposed_label_vol_lst[1]
                label_merge_vol = interposed_label_merge_vol

                # crop the roi of the image
                train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                           train_roi_mask_file,
                                                                                           interposed_label_data,
                                                                                           margin_crop_mask)
            else:
                dentate_label_fname_lst.append(dentate_label_fname_side_lst)
                dentate_label_merge_vol = dentate_label_vol_lst[0] + dentate_label_vol_lst[1]

                interposed_label_fname_lst.append(interposed_label_fname_side_lst)
                interposed_label_merge_vol = interposed_label_vol_lst[0] + interposed_label_vol_lst[1]

                label_merge_vol = dentate_label_merge_vol + interposed_label_merge_vol
                label_merge_vol[label_merge_vol == np.max(label_merge_vol)] = 1

                # crop the roi of the image
                train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                           train_roi_mask_file,
                                                                                           dentate_label_data,
                                                                                           margin_crop_mask)

                # assign integers, later it should be encoded into one-hot in build_training_set (to_categorical)
                dentate_label_merge_vol[dentate_label_merge_vol == np.max(dentate_label_merge_vol)] = 1
                interposed_label_merge_vol[interposed_label_merge_vol == np.max(interposed_label_merge_vol)] = 2
                label_merge_vol = dentate_label_merge_vol + interposed_label_merge_vol
                label_merge_vol[label_merge_vol == np.max(label_merge_vol)] = 2  # assign overlaps to interposed

            label_crop_vol = train_crop_mask.crop(label_merge_vol)
            # for key in label_mapper.keys():
            #     label_crop_vol[label_crop_vol == key] = label_mapper[key]

            # save the cropped labels (train_roi)
            for side, idx in zip(['left', 'right'], [0, 1]):
                if target == 'dentate':
                    dentate_label_filepath = dentate_label_fname_side_lst[idx]
                    if is_new_trn_label == 1:
                        cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                      crop_trn_dentate_new_label_pattern.format(side, approach))
                    elif is_new_trn_label == 2:
                        cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                      crop_trn_dentate_new_label_pattern.format(
                                                                          train_patient_id, side))
                    elif is_new_trn_label == 3:
                        if train_patient_id in patient_id:
                            cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                          crop_trn_dentate_label_pattern.format(
                                                                              train_patient_id, side))
                        else:
                            cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                          crop_trn_dentate_new_label_pattern.format(
                                                                              side, approach))
                    else:
                        cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                      crop_trn_dentate_label_pattern.format(
                                                                          train_patient_id, side))
                    if (not os.path.exists(cropped_trn_dentate_label_file)) or set_new_roi_mask is True:
                        crop_image(dentate_label_filepath, train_crop_mask, cropped_trn_dentate_label_file)

                elif target == 'interposed':
                    interposed_label_filepath = interposed_label_fname_side_lst[idx]
                    if is_new_trn_label == 1:
                        cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                         crop_trn_interposed_new_label_pattern.format(
                                                                             side, approach))
                    elif is_new_trn_label == 2:
                        cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                         crop_trn_interposed_new_label_pattern.format(
                                                                             train_patient_id, side))
                    elif is_new_trn_label == 3:
                        if train_patient_id in patient_id:
                            cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                             crop_trn_interposed_label_pattern.format(
                                                                                 train_patient_id, side))
                        else:
                            cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                             crop_trn_interposed_new_label_pattern.format(
                                                                                 side, approach))
                    else:
                        cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                         crop_trn_interposed_label_pattern.format(
                                                                             train_patient_id, side))
                    if (not os.path.exists(cropped_trn_interposed_label_file)) or set_new_roi_mask is True:
                        crop_image(interposed_label_filepath, train_crop_mask, cropped_trn_interposed_label_file)

                else:
                    if is_new_trn_label == 1:
                        cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                      crop_trn_dentate_new_label_pattern.format(side, approach))
                        cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                         crop_trn_interposed_new_label_pattern.format(
                                                                             side, approach))
                    elif is_new_trn_label == 2:
                        cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                      crop_trn_dentate_new_label_pattern.format(
                                                                          train_patient_id, side))
                        cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                         crop_trn_interposed_new_label_pattern.format(
                                                                             train_patient_id, side))
                    elif is_new_trn_label == 3:
                        if train_patient_id in patient_id:
                            cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                          crop_trn_dentate_label_pattern.format(
                                                                              train_patient_id, side))
                            cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                             crop_trn_interposed_label_pattern.format(
                                                                                 train_patient_id, side))
                        else:
                            cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                          crop_trn_dentate_new_label_pattern.format(
                                                                              side, approach))
                            cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                             crop_trn_interposed_new_label_pattern.format(
                                                                                 side, approach))
                    else:
                        cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                      crop_trn_dentate_label_pattern.format(
                                                                          train_patient_id, side))
                        cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                         crop_trn_interposed_label_pattern.format(
                                                                             train_patient_id, side))

                    dentate_label_filepath = dentate_label_fname_side_lst[idx]
                    if (not os.path.exists(cropped_trn_dentate_label_file)) or set_new_roi_mask is True:
                        crop_image(dentate_label_filepath, train_crop_mask, cropped_trn_dentate_label_file)

                    interposed_label_filepath = interposed_label_fname_side_lst[idx]
                    if (not os.path.exists(cropped_trn_interposed_label_file)) or set_new_roi_mask is True:
                        crop_image(interposed_label_filepath, train_crop_mask, cropped_trn_interposed_label_file)

            label_crop_array = np.zeros((1, 1) + label_crop_vol.shape[0:3])
            label_crop_array[0, 0] = label_crop_vol
            label_lst.append(label_crop_array)

            # load training images in train_patient_lst and setting roi
            train_image_modal_lst = []
            train_fname_modality_lst = []
            for idx in modality_idx:
                training_img_path = os.path.join(dataset_path, train_patient_id,
                                                 img_pattern[idx].format(train_patient_id))
                print(training_img_path)
                train_fname_modality_lst.append(training_img_path)
                training_vol = read_volume(training_img_path)
                if preprocess_trn == 2 or preprocess_trn == 3 or preprocess_trn == 5:
                    training_vol = normalize_image(training_vol, [0, 2 ** 8])
                if np.size(np.shape(training_vol)) == 4:
                    train_crop_vol = train_crop_mask.crop(training_vol[:, :, :, 0])
                else:
                    train_crop_vol = train_crop_mask.crop(training_vol)
                train_image_modal_lst.append(train_crop_vol)
            train_fname_lst.append(train_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                training_img_path = os.path.join(dataset_path, train_patient_id,
                                                 img_pattern[idx].format(train_patient_id))
                crop_train_img_path = os.path.join(file_output_dir, train_patient_id,
                                                   crop_trn_img_pattern[idx].format(train_patient_id))
                if (not os.path.exists(crop_train_img_path)) or set_new_roi_mask is True:
                    crop_image(training_img_path, train_crop_mask, crop_train_img_path)

            train_crop_array = np.zeros((1, num_modality) + train_crop_vol.shape[0:3])
            for idx_modal in range(num_modality):
                train_crop_array[0, idx_modal] = train_image_modal_lst[idx_modal]
            train_img_lst.append(train_crop_array)

    # load multi-modal images in test_patient_lst and setting roi
    test_img_lst = []
    test_fname_lst = []
    if test_patient_lst[0] is not None:
        for test_patient_id in test_patient_lst:

            file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
            if not os.path.exists(file_test_patient_dir):
                os.makedirs(file_test_patient_dir)

            test_roi_mask_file = os.path.join(file_output_dir, test_roi_mask_pattern.format(test_patient_id))
            if os.path.exists(test_roi_mask_file) and set_new_roi_mask is True:
                os.remove(test_roi_mask_file)

            # load initial mask (SUIT output or linearly registered from ref. training image) for setting roi
            init_dentate_mask_vol_lst = []
            init_interposed_mask_vol_lst = []

            for side in ['left', 'right']:  # left + right
                if target == 'dentate':
                    init_dentate_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                              init_dentate_mask_pattern.format(test_patient_id, side))
                    print(init_dentate_mask_filepath)
                    init_dentate_mask_data = read_volume_data(init_dentate_mask_filepath)
                    init_dentate_mask_vol = init_dentate_mask_data.get_data()
                    if np.size(np.shape(init_dentate_mask_vol)) == 4:
                        init_dentate_mask_vol_lst.append(init_dentate_mask_vol[:, :, :, 0])
                    else:
                        init_dentate_mask_vol_lst.append(init_dentate_mask_vol)
                    init_mask_data = init_dentate_mask_data

                elif target == 'interposed':
                    init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                 init_interposed_mask_pattern_thres.format(
                                                                     test_patient_id,
                                                                     side))
                    init_interposed_mask_data = read_volume_data(init_interposed_mask_filepath)
                    init_interposed_mask_vol = init_interposed_mask_data.get_data()
                    is_empty_vol = check_empty_vol(init_interposed_mask_vol)
                    is_init_interposed_mask_thres = True
                    if is_empty_vol:
                        print('There is no label in the initial mask in %s' % init_interposed_mask_filepath)
                        init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                     init_interposed_mask_pattern_mask.format(
                                                                         test_patient_id,
                                                                         side))
                        init_interposed_mask_data = read_volume_data(init_interposed_mask_filepath)
                        init_interposed_mask_vol = init_interposed_mask_data.get_data()
                        is_empty_vol = check_empty_vol(init_interposed_mask_vol)
                        is_init_interposed_mask_thres = False
                        if is_empty_vol:
                            print('There is no label in the initial mask in %s' % init_interposed_mask_filepath)
                            exit()

                    print(init_interposed_mask_filepath)

                    if np.size(np.shape(init_interposed_mask_vol)) == 4:
                        init_interposed_mask_vol_lst.append(init_interposed_mask_vol[:, :, :, 0])
                    else:
                        init_interposed_mask_vol_lst.append(init_interposed_mask_vol)
                    init_mask_data = init_interposed_mask_data

                else:
                    init_dentate_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                              init_dentate_mask_pattern.format(test_patient_id, side))
                    print(init_dentate_mask_filepath)
                    init_dentate_mask_data = read_volume_data(init_dentate_mask_filepath)
                    init_dentate_mask_vol = init_dentate_mask_data.get_data()
                    if np.size(np.shape(init_dentate_mask_vol)) == 4:
                        init_dentate_mask_vol_lst.append(init_dentate_mask_vol[:, :, :, 0])
                    else:
                        init_dentate_mask_vol_lst.append(init_dentate_mask_vol)

                    init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                 init_interposed_mask_pattern_thres.format(
                                                                     test_patient_id,
                                                                     side))
                    init_interposed_mask_data = read_volume_data(init_interposed_mask_filepath)
                    init_interposed_mask_vol = init_interposed_mask_data.get_data()
                    is_empty_vol = check_empty_vol(init_interposed_mask_vol)
                    is_init_interposed_mask_thres = True
                    if is_empty_vol:
                        print('There is no label in the initial mask in %s' % init_interposed_mask_filepath)
                        init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                     init_interposed_mask_pattern_mask.format(
                                                                         test_patient_id,
                                                                         side))
                        init_interposed_mask_data = read_volume_data(init_interposed_mask_filepath)
                        init_interposed_mask_vol = init_interposed_mask_data.get_data()
                        is_empty_vol = check_empty_vol(init_interposed_mask_vol)
                        is_init_interposed_mask_thres = False
                        if is_empty_vol:
                            print('There is no label in the initial mask in %s' % init_interposed_mask_filepath)
                            exit()

                    print(init_interposed_mask_filepath)

                    if np.size(np.shape(init_interposed_mask_vol)) == 4:
                        init_interposed_mask_vol_lst.append(init_interposed_mask_vol[:, :, :, 0])
                    else:
                        init_interposed_mask_vol_lst.append(init_interposed_mask_vol)
                    init_mask_data = init_dentate_mask_data

            if target == 'dentate':
                init_dentate_mask_merge_vol = init_dentate_mask_vol_lst[0] + init_dentate_mask_vol_lst[1]
                init_mask_merge_vol = init_dentate_mask_merge_vol
            elif target == 'interposed':
                init_interposed_mask_merge_vol = init_interposed_mask_vol_lst[0] + init_interposed_mask_vol_lst[1]
                init_mask_merge_vol = init_interposed_mask_merge_vol
            else:
                init_dentate_mask_merge_vol = init_dentate_mask_vol_lst[0] + init_dentate_mask_vol_lst[1]
                init_interposed_mask_merge_vol = init_interposed_mask_vol_lst[0] + init_interposed_mask_vol_lst[1]
                init_mask_merge_vol = init_dentate_mask_merge_vol + init_interposed_mask_merge_vol

            # crop the roi of the image
            test_crop_mask = find_crop_mask(test_roi_mask_file) or compute_crop_mask(init_mask_merge_vol,
                                                                                     test_roi_mask_file,
                                                                                     init_mask_data,
                                                                                     margin_crop_mask)
            # save the cropped initial masks
            for side in ['left', 'right']:
                if target == 'dentate':
                    init_dentate_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                              init_dentate_mask_pattern.format(test_patient_id, side))
                    cropped_init_dentate_mask_file = os.path.join(file_test_patient_dir,
                                                                  crop_init_dentate_mask_pattern.format(test_patient_id,
                                                                                                        side))
                    if (not os.path.exists(cropped_init_dentate_mask_file)) or set_new_roi_mask is True:
                        crop_image(init_dentate_mask_filepath, test_crop_mask, cropped_init_dentate_mask_file)

                elif target == 'interposed':
                    if is_init_interposed_mask_thres:
                        init_interposed_mask_pattern = init_interposed_mask_pattern_thres
                    else:
                        init_interposed_mask_pattern = init_interposed_mask_pattern_mask
                    init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                 init_interposed_mask_pattern.format(test_patient_id,
                                                                                                     side))
                    cropped_init_interposed_mask_file = os.path.join(file_test_patient_dir,
                                                                     crop_init_interposed_mask_pattern.format(
                                                                         test_patient_id, side))
                    if (not os.path.exists(cropped_init_interposed_mask_file)) or set_new_roi_mask is True:
                        crop_image(init_interposed_mask_filepath, test_crop_mask, cropped_init_interposed_mask_file)

                else:
                    init_dentate_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                              init_dentate_mask_pattern.format(test_patient_id, side))
                    cropped_init_dentate_mask_file = os.path.join(file_test_patient_dir,
                                                                  crop_init_dentate_mask_pattern.format(test_patient_id,
                                                                                                        side))
                    if (not os.path.exists(cropped_init_dentate_mask_file)) or set_new_roi_mask is True:
                        crop_image(init_dentate_mask_filepath, test_crop_mask, cropped_init_dentate_mask_file)

                    if is_init_interposed_mask_thres:
                        init_interposed_mask_pattern = init_interposed_mask_pattern_thres
                    else:
                        init_interposed_mask_pattern = init_interposed_mask_pattern_mask
                    init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                 init_interposed_mask_pattern.format(test_patient_id,
                                                                                                     side))
                    cropped_init_interposed_mask_file = os.path.join(file_test_patient_dir,
                                                                     crop_init_interposed_mask_pattern.format(
                                                                         test_patient_id, side))
                    if (not os.path.exists(cropped_init_interposed_mask_file)) or set_new_roi_mask is True:
                        crop_image(init_interposed_mask_filepath, test_crop_mask, cropped_init_interposed_mask_file)

            # save the cropped labels (test_roi)
            for side in ['left', 'right']:
                if target == 'dentate':
                    dentate_label_filepath = os.path.join(dataset_path, test_patient_id,
                                                          dentate_label_pattern.format(test_patient_id,
                                                                                       side))
                    if os.path.exists(dentate_label_filepath):
                        cropped_tst_dentate_label_file = os.path.join(file_test_patient_dir,
                                                                      crop_tst_dentate_label_pattern.format(
                                                                          test_patient_id, side))
                        if (not os.path.exists(cropped_tst_dentate_label_file)) or set_new_roi_mask is True:
                            crop_image(dentate_label_filepath, test_crop_mask,
                                       cropped_tst_dentate_label_file)

                elif target == 'interposed':
                    interposed_label_filepath = os.path.join(dataset_path, test_patient_id,
                                                             interposed_label_pattern.format(
                                                                 test_patient_id, side))
                    if os.path.exists(interposed_label_filepath):
                        cropped_tst_interposed_label_file = os.path.join(file_test_patient_dir,
                                                                         crop_tst_interposed_label_pattern.format(
                                                                             test_patient_id, side))
                        if (not os.path.exists(cropped_tst_interposed_label_file)) or set_new_roi_mask is True:
                            crop_image(interposed_label_filepath, test_crop_mask,
                                       cropped_tst_interposed_label_file)
                else:
                    dentate_label_filepath = os.path.join(dataset_path, test_patient_id,
                                                          dentate_label_pattern.format(test_patient_id,
                                                                                       side))
                    if os.path.exists(dentate_label_filepath):
                        cropped_tst_dentate_label_file = os.path.join(file_test_patient_dir,
                                                                      crop_tst_dentate_label_pattern.format(
                                                                          test_patient_id, side))
                        if (not os.path.exists(cropped_tst_dentate_label_file)) or set_new_roi_mask is True:
                            crop_image(dentate_label_filepath, test_crop_mask,
                                       cropped_tst_dentate_label_file)

                    interposed_label_filepath = os.path.join(dataset_path, test_patient_id,
                                                             interposed_label_pattern.format(
                                                                 test_patient_id, side))
                    if os.path.exists(interposed_label_filepath):
                        cropped_tst_interposed_label_file = os.path.join(file_test_patient_dir,
                                                                         crop_tst_interposed_label_pattern.format(
                                                                             test_patient_id, side))
                        if (not os.path.exists(cropped_tst_interposed_label_file)) or set_new_roi_mask is True:
                            crop_image(interposed_label_filepath, test_crop_mask,
                                       cropped_tst_interposed_label_file)

            # load test images in test_patient_lst and setting roi
            test_image_modal_lst = []
            test_fname_modality_lst = []
            for idx in modality_idx:
                test_img_path = os.path.join(dataset_path, test_patient_id, img_pattern[idx].format(test_patient_id))
                print(test_img_path)
                test_fname_modality_lst.append(test_img_path)
                test_vol = read_volume(test_img_path)
                if preprocess_tst == 2 or preprocess_tst == 3 or preprocess_tst == 5:
                    test_vol = normalize_image(test_vol, [0, 2 ** 8])
                if np.size(np.shape(test_vol)) == 4:
                    test_crop_vol = test_crop_mask.crop(test_vol[:, :, :, 0])
                else:
                    test_crop_vol = test_crop_mask.crop(test_vol)
                test_image_modal_lst.append(test_crop_vol)
            test_fname_lst.append(test_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                test_img_path = os.path.join(dataset_path, test_patient_id, img_pattern[idx].format(test_patient_id))
                crop_test_img_path = os.path.join(file_test_patient_dir,
                                                  crop_tst_img_pattern[idx].format(test_patient_id))
                if (not os.path.exists(crop_test_img_path)) or set_new_roi_mask is True:
                    crop_image(test_img_path, test_crop_mask, crop_test_img_path)

            test_crop_array = np.zeros((1, num_modality) + test_crop_vol.shape[0:3])
            for idx in range(num_modality):
                test_crop_array[0, idx] = test_image_modal_lst[idx]
            test_img_lst.append(test_crop_array)

    return train_img_lst, label_lst, test_img_lst, train_fname_lst, dentate_label_fname_lst, test_fname_lst


def read_dentate_interposed_dataset_unseen(gen_conf, train_conf, train_patient_lst, test_patient_lst, preprocess_trn,
                                    preprocess_tst, file_output_dir, target, is_scaling, is_reg, roi_pos):
    root_path = gen_conf['root_path']
    dataset = train_conf['dataset']
    num_epochs = train_conf['num_epochs']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    mode = gen_conf['validation_mode']
    img_pattern = dataset_info['image_name_pattern']
    image_new_name_pattern = dataset_info['image_new_name_pattern']
    img_resampled_name_pattern = dataset_info['image_resampled_name_pattern']

    dentate_label_pattern = dataset_info['manual_corrected_dentate_v2_pattern']
    interposed_label_pattern = dataset_info['manual_corrected_interposed_v2_pattern']
    dentate_interposed_label_pattern = dataset_info['manual_corrected_dentate_interposed_v2_pattern']

    modality = dataset_info['image_modality']
    num_modality = len(modality)

    file_format = dataset_info['format']

    init_dentate_mask_pattern = dataset_info['initial_mask_pattern']
    init_interposed_mask_pattern = dataset_info['initial_interposed_mask_pattern_thres']
    init_reg_mask_pattern = dataset_info['initial_reg_mask_pattern']

    set_new_roi_mask = dataset_info['set_new_roi_mask']
    margin_crop_mask = dataset_info['margin_crop_mask']

    crop_trn_img_pattern = dataset_info['crop_trn_image_name_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    crop_trn_dentate_label_pattern = dataset_info['crop_trn_manual_dentate_v2_corrected_pattern']
    crop_tst_dentate_label_pattern = dataset_info['crop_tst_manual_dentate_v2_corrected_pattern']
    crop_trn_interposed_label_pattern = dataset_info['crop_trn_manual_interposed_v2_pattern']
    crop_tst_interposed_label_pattern = dataset_info['crop_tst_manual_interposed_v2_pattern']

    crop_init_dentate_mask_pattern = dataset_info['crop_initial_mask_pattern']
    crop_init_interposed_mask_pattern = dataset_info['crop_initial_interposed_mask_pattern']
    crop_init_reg_mask_pattern = dataset_info['crop_initial_reg_mask_pattern']

    train_roi_mask_pattern = dataset_info['train_roi_mask_pattern']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']

    modality_idx = []
    for m in modality:
        if m == 'B0':
            modality_idx.append(0)
        if m in ['T1', 'QSM']:
            modality_idx.append(1)
        if m == 'LP':
            modality_idx.append(2)
        if m == 'FA':
            modality_idx.append(3)

    #label_mapper = {0: 0, 10: 1, 150: 2}

    is_res_diff = False # initial setting

    if len(target) == 2:
        target = 'both'

    # load multi-modal images in train_patient_lst and setting roi
    train_img_lst = []
    label_lst = []
    train_fname_lst = []
    dentate_label_fname_lst = []
    interposed_label_fname_lst = []
    if train_patient_lst[0] is not None and num_epochs != 0 and mode != '2':
        for train_patient_id in train_patient_lst:

            file_trn_patient_dir = os.path.join(file_output_dir, train_patient_id)
            if not os.path.exists(file_trn_patient_dir):
                os.makedirs(file_trn_patient_dir)

            train_roi_mask_file = os.path.join(file_output_dir, train_roi_mask_pattern.format(train_patient_id))
            if os.path.exists(train_roi_mask_file) and set_new_roi_mask is True:
                os.remove(train_roi_mask_file)

            # load label images in train_patient_lst and setting roi
            dentate_label_vol_lst = []
            dentate_label_fname_side_lst = []
            interposed_label_vol_lst = []
            interposed_label_fname_side_lst = []
            for side in ['left', 'right']:  # left + right
                if target == 'dentate':
                    dentate_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                          dentate_label_pattern.format(train_patient_id, side))
                    print(dentate_label_filepath)
                    dentate_label_fname_side_lst.append(dentate_label_filepath)
                    dentate_label_data = read_volume_data(dentate_label_filepath)
                    dentate_label_vol = dentate_label_data.get_data()
                    if np.size(np.shape(dentate_label_vol)) == 4:
                        dentate_label_vol_lst.append(dentate_label_vol[:, :, :, 0])
                    else:
                        dentate_label_vol_lst.append(dentate_label_vol)

                elif target == 'interposed':
                    interposed_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                             interposed_label_pattern.format(train_patient_id, side))
                    print(interposed_label_filepath)
                    interposed_label_fname_side_lst.append(interposed_label_filepath)
                    interposed_label_data = read_volume_data(interposed_label_filepath)
                    interposed_label_vol = interposed_label_data.get_data()
                    if np.size(np.shape(interposed_label_vol)) == 4:
                        interposed_label_vol_lst.append(interposed_label_vol[:, :, :, 0])
                    else:
                        interposed_label_vol_lst.append(interposed_label_vol)

                else:
                    dentate_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                          dentate_label_pattern.format(train_patient_id, side))
                    print(dentate_label_filepath)
                    dentate_label_fname_side_lst.append(dentate_label_filepath)
                    dentate_label_data = read_volume_data(dentate_label_filepath)
                    dentate_label_vol = dentate_label_data.get_data()
                    if np.size(np.shape(dentate_label_vol)) == 4:
                        dentate_label_vol_lst.append(dentate_label_vol[:, :, :, 0])
                    else:
                        dentate_label_vol_lst.append(dentate_label_vol)

                    interposed_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                             interposed_label_pattern.format(train_patient_id, side))
                    print(interposed_label_filepath)
                    interposed_label_fname_side_lst.append(interposed_label_filepath)
                    interposed_label_data = read_volume_data(interposed_label_filepath)
                    interposed_label_vol = interposed_label_data.get_data()
                    if np.size(np.shape(interposed_label_vol)) == 4:
                        interposed_label_vol_lst.append(interposed_label_vol[:, :, :, 0])
                    else:
                        interposed_label_vol_lst.append(interposed_label_vol)

            if target == 'dentate':
                dentate_label_fname_lst.append(dentate_label_fname_side_lst)
                dentate_label_merge_vol = dentate_label_vol_lst[0] + dentate_label_vol_lst[1]
                label_merge_vol = dentate_label_merge_vol

                # crop the roi of the image
                train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                       train_roi_mask_file,
                                                                                       dentate_label_data,
                                                                                       margin_crop_mask)

            elif target == 'interposed':
                interposed_label_fname_lst.append(interposed_label_fname_side_lst)
                interposed_label_merge_vol = interposed_label_vol_lst[0] + interposed_label_vol_lst[1]
                label_merge_vol = interposed_label_merge_vol

                # crop the roi of the image
                train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                       train_roi_mask_file,
                                                                                       interposed_label_data,
                                                                                       margin_crop_mask)
            else:
                dentate_label_fname_lst.append(dentate_label_fname_side_lst)
                dentate_label_merge_vol = dentate_label_vol_lst[0] + dentate_label_vol_lst[1]

                interposed_label_fname_lst.append(interposed_label_fname_side_lst)
                interposed_label_merge_vol = interposed_label_vol_lst[0] + interposed_label_vol_lst[1]

                label_merge_vol = dentate_label_merge_vol + interposed_label_merge_vol
                label_merge_vol[label_merge_vol == np.max(label_merge_vol)] = 1

                # crop the roi of the image
                train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                       train_roi_mask_file,
                                                                                       dentate_label_data,
                                                                                       margin_crop_mask)

                # assign integers, later it should be encoded into one-hot in build_training_set (to_categorical)
                dentate_label_merge_vol[dentate_label_merge_vol == np.max(dentate_label_merge_vol)] = 1
                interposed_label_merge_vol[interposed_label_merge_vol == np.max(interposed_label_merge_vol)] = 2
                label_merge_vol = dentate_label_merge_vol + interposed_label_merge_vol
                label_merge_vol[label_merge_vol == np.max(label_merge_vol)] = 2  # assign overlaps to interposed

            label_crop_vol = train_crop_mask.crop(label_merge_vol)
            # for key in label_mapper.keys():
            #     label_crop_vol[label_crop_vol == key] = label_mapper[key]

            # save the cropped labels (train_roi)
            for side, idx in zip(['left', 'right'], [0, 1]):
                if target == 'dentate':
                    dentate_label_filepath = dentate_label_fname_side_lst[idx]
                    cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                  crop_trn_dentate_label_pattern.format(train_patient_id, side))
                    if (not os.path.exists(cropped_trn_dentate_label_file)) or set_new_roi_mask is True:
                        crop_image(dentate_label_filepath, train_crop_mask, cropped_trn_dentate_label_file)

                elif target == 'interposed':
                    interposed_label_filepath = interposed_label_fname_side_lst[idx]
                    cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                     crop_trn_interposed_label_pattern.format(train_patient_id, side))
                    if (not os.path.exists(cropped_trn_interposed_label_file)) or set_new_roi_mask is True:
                        crop_image(interposed_label_filepath, train_crop_mask, cropped_trn_interposed_label_file)

                else:
                    dentate_label_filepath = dentate_label_fname_side_lst[idx]
                    cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                  crop_trn_dentate_label_pattern.format(train_patient_id, side))
                    if (not os.path.exists(cropped_trn_dentate_label_file)) or set_new_roi_mask is True:
                        crop_image(dentate_label_filepath, train_crop_mask, cropped_trn_dentate_label_file)

                    interposed_label_filepath = interposed_label_fname_side_lst[idx]
                    cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                     crop_trn_interposed_label_pattern.format(train_patient_id, side))
                    if (not os.path.exists(cropped_trn_interposed_label_file)) or set_new_roi_mask is True:
                        crop_image(interposed_label_filepath, train_crop_mask, cropped_trn_interposed_label_file)

            label_crop_array = np.zeros((1, 1) + label_crop_vol.shape[0:3])
            label_crop_array[0, 0] = label_crop_vol
            label_lst.append(label_crop_array)

            # load training images in train_patient_lst and setting roi
            train_image_modal_lst= []
            train_fname_modality_lst = []
            for idx in modality_idx:
                training_img_path = os.path.join(dataset_path, train_patient_id, img_pattern[idx].format(train_patient_id))
                print(training_img_path)
                train_fname_modality_lst.append(training_img_path)
                training_vol = read_volume(training_img_path)
                if preprocess_trn == 2 or preprocess_trn == 3 or preprocess_trn == 5:
                    training_vol = normalize_image(training_vol, [0, 2 ** 8])
                if np.size(np.shape(training_vol)) == 4:
                    train_crop_vol = train_crop_mask.crop(training_vol[:, :, :, 0])
                else:
                    train_crop_vol = train_crop_mask.crop(training_vol)
                train_image_modal_lst.append(train_crop_vol)
            train_fname_lst.append(train_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                train_img_path = train_fname_modality_lst[idx]
                crop_train_img_path = os.path.join(file_output_dir, train_patient_id,
                                                   crop_trn_img_pattern[idx].format(train_patient_id))
                if (not os.path.exists(crop_train_img_path)) or set_new_roi_mask is True:
                    crop_image(train_img_path, train_crop_mask, crop_train_img_path)

            train_crop_array = np.zeros((1, num_modality) + train_crop_vol.shape[0:3])
            for idx in range(num_modality):
                train_crop_array[0, idx] = train_image_modal_lst[idx]
            train_img_lst.append(train_crop_array)

    # load multi-modal images in test_patient_lst and setting roi
    test_img_lst = []
    test_fname_lst = []
    if test_patient_lst[0] is not None:
        for test_patient_id in test_patient_lst:

            file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
            if not os.path.exists(file_test_patient_dir):
                os.makedirs(file_test_patient_dir)

            test_roi_mask_file = os.path.join(file_output_dir, test_roi_mask_pattern.format(test_patient_id))
            if os.path.exists(test_roi_mask_file) and set_new_roi_mask is True:
                os.remove(test_roi_mask_file)

            test_fname_modality_lst_for_init_reg = []
            for idx in modality_idx:
                #test_img_path = os.path.join(dataset_path, test_patient_id, img_pattern[idx].format(test_patient_id))
                test_img_path = ''
                if not os.path.exists(test_img_path):
                    test_img_path = os.path.join(root_path, 'datasets', 'dcn', test_patient_id, 'image',
                                                 image_new_name_pattern[idx].format(test_patient_id))
                print(test_img_path)
                test_fname_modality_lst_for_init_reg.append(test_img_path)
            test_ref_img_path = test_fname_modality_lst_for_init_reg[0]

            # load initial mask (SUIT output or linearly registered from ref. training image) for setting roi
            init_dentate_mask_vol_lst = []
            init_interposed_mask_vol_lst = []
            #init_mask_path = os.path.join(dataset_path, test_patient_id, 'DCN_masks', 'thresh35')
            init_mask_path = ''
            if os.path.exists(init_mask_path):
                for side in ['left', 'right']:  # left + right
                    if target == 'dentate':
                        init_dentate_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                  init_dentate_mask_pattern.format(test_patient_id, side))
                        print(init_dentate_mask_filepath)
                        init_dentate_mask_data = read_volume_data(init_dentate_mask_filepath)
                        init_dentate_mask_vol = init_dentate_mask_data.get_data()
                        if np.size(np.shape(init_dentate_mask_vol)) == 4:
                            init_dentate_mask_vol_lst.append(init_dentate_mask_vol[:, :, :, 0])
                        else:
                            init_dentate_mask_vol_lst.append(init_dentate_mask_vol)
                        init_mask_data = init_dentate_mask_data

                    elif target == 'interposed':
                        init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                     init_interposed_mask_pattern.format(test_patient_id, side))
                        print(init_interposed_mask_filepath)
                        init_interposed_mask_data = read_volume_data(init_interposed_mask_filepath)
                        init_interposed_mask_vol = init_interposed_mask_data.get_data()
                        if np.size(np.shape(init_interposed_mask_vol)) == 4:
                            init_interposed_mask_vol_lst.append(init_interposed_mask_vol[:, :, :, 0])
                        else:
                            init_interposed_mask_vol_lst.append(init_interposed_mask_vol)
                        init_mask_data = init_interposed_mask_data

                    else:
                        init_dentate_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                  init_dentate_mask_pattern.format(test_patient_id, side))
                        print(init_dentate_mask_filepath)
                        init_dentate_mask_data = read_volume_data(init_dentate_mask_filepath)
                        init_dentate_mask_vol = init_dentate_mask_data.get_data()
                        if np.size(np.shape(init_dentate_mask_vol)) == 4:
                            init_dentate_mask_vol_lst.append(init_dentate_mask_vol[:, :, :, 0])
                        else:
                            init_dentate_mask_vol_lst.append(init_dentate_mask_vol)

                        init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                     init_interposed_mask_pattern.format(test_patient_id, side))
                        print(init_interposed_mask_filepath)
                        init_interposed_mask_data = read_volume_data(init_interposed_mask_filepath)
                        init_interposed_mask_vol = init_interposed_mask_data.get_data()
                        if np.size(np.shape(init_interposed_mask_vol)) == 4:
                            init_interposed_mask_vol_lst.append(init_interposed_mask_vol[:, :, :, 0])
                        else:
                            init_interposed_mask_vol_lst.append(init_interposed_mask_vol)
                        init_mask_data = init_dentate_mask_data

                if target == 'dentate':
                    init_dentate_mask_merge_vol = init_dentate_mask_vol_lst[0] + init_dentate_mask_vol_lst[1]
                    init_mask_merge_vol = init_dentate_mask_merge_vol
                elif target == 'interposed':
                    init_interposed_mask_merge_vol = init_interposed_mask_vol_lst[0] + init_interposed_mask_vol_lst[1]
                    init_mask_merge_vol = init_interposed_mask_merge_vol
                else:
                    init_dentate_mask_merge_vol = init_dentate_mask_vol_lst[0] + init_dentate_mask_vol_lst[1]
                    init_interposed_mask_merge_vol = init_interposed_mask_vol_lst[0] + init_interposed_mask_vol_lst[1]
                    init_mask_merge_vol = init_dentate_mask_merge_vol + init_interposed_mask_merge_vol

            else:  # localize target via initial registration
                train_fname_lst_for_init_reg = []
                label_fname_lst_for_init_reg = []

                file_test_patient_label_dir = os.path.join(file_test_patient_dir, 'training_labels')
                if not os.path.exists(file_test_patient_label_dir):
                    os.makedirs(file_test_patient_label_dir)

                for train_patient_id in train_patient_lst:
                    train_fname_modality_lst_for_init_reg = []
                    for idx in modality_idx:
                        training_img_path = os.path.join(dataset_path, train_patient_id, img_pattern[idx].format(
                            train_patient_id))
                        train_fname_modality_lst_for_init_reg.append(training_img_path)
                    train_fname_lst_for_init_reg.append(train_fname_modality_lst_for_init_reg)
                    train_image_data = read_volume_data(train_fname_modality_lst_for_init_reg[0])

                    dentate_label_fname_side_lst_for_init_reg = []
                    interposed_label_fname_side_lst_for_init_reg = []
                    dentate_interposed_label_fname_side_lst_for_init_reg = []
                    for side in ['left', 'right']:  # left + right
                        if target == 'dentate':
                            dentate_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                  dentate_label_pattern.format(train_patient_id, side))
                            dentate_label_fname_side_lst_for_init_reg.append(dentate_label_filepath)
                        elif target == 'interposed':
                            interposed_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                     interposed_label_pattern.format(train_patient_id,
                                                                                                     side))
                            interposed_label_fname_side_lst_for_init_reg.append(interposed_label_filepath)
                        else:
                            dentate_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                  dentate_label_pattern.format(train_patient_id, side))

                            dentate_label_fname_side_lst_for_init_reg.append(dentate_label_filepath)
                            dentate_label_data = read_volume_data(dentate_label_filepath)
                            dentate_label_vol = dentate_label_data.get_data()
                            if np.size(np.shape(dentate_label_vol)) == 4:
                                _dentate_label_vol = dentate_label_vol[:, :, :, 0]
                            else:
                                _dentate_label_vol = dentate_label_vol

                            interposed_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                     interposed_label_pattern.format(train_patient_id,
                                                                                                     side))
                            interposed_label_fname_side_lst_for_init_reg.append(interposed_label_filepath)
                            interposed_label_data = read_volume_data(interposed_label_filepath)
                            interposed_label_vol = interposed_label_data.get_data()
                            if np.size(np.shape(interposed_label_vol)) == 4:
                                _interposed_label_vol = interposed_label_vol[:, :, :, 0]
                            else:
                                _interposed_label_vol = interposed_label_vol

                            dentate_interposed_label_filepath = os.path.join(file_test_patient_label_dir,
                                                                     dentate_interposed_label_pattern.format(train_patient_id,
                                                                                                     side))
                            dentate_interposed_label_fname_side_lst_for_init_reg.append(dentate_interposed_label_filepath)

                            label_merge_vol = _dentate_label_vol + _interposed_label_vol
                            label_merge_vol[label_merge_vol == np.max(label_merge_vol)] = 1

                            __save_volume(label_merge_vol, train_image_data, dentate_interposed_label_filepath,
                                          file_format, is_compressed=True)

                    if target == 'dentate':
                        label_fname_lst_for_init_reg.append(dentate_label_fname_side_lst_for_init_reg)
                    elif target == 'interposed':
                        label_fname_lst_for_init_reg.append(interposed_label_fname_side_lst_for_init_reg)
                    else:
                        label_fname_lst_for_init_reg.append(dentate_interposed_label_fname_side_lst_for_init_reg)

                init_mask_merge_vol, init_mask_data, test_ref_img_path, is_res_diff, roi_pos = \
                    localize_target(file_output_dir,
                                    modality_idx,
                                    train_patient_lst,
                                    train_fname_lst_for_init_reg,
                                    test_patient_id,
                                    test_fname_modality_lst_for_init_reg,
                                    label_fname_lst_for_init_reg,
                                    _remove_ending(img_pattern[0].format(''), '.nii')+'.nii.gz',
                                    img_resampled_name_pattern,
                                    init_reg_mask_pattern,
                                    target,
                                    is_res_diff,
                                    roi_pos,
                                    is_scaling,
                                    is_reg)

            if init_mask_merge_vol.tolist():
                # crop the roi of the image
                test_crop_mask = find_crop_mask(test_roi_mask_file) or compute_crop_mask(init_mask_merge_vol,
                                                                                         test_roi_mask_file,
                                                                                         init_mask_data,
                                                                                         margin_crop_mask=(5, 5, 10))
            else:
                print (roi_pos)
                test_ref_img_data = read_volume_data(test_ref_img_path)
                test_ref_image = BrainImage(test_ref_img_path, None)
                test_vol = test_ref_image.nii_data_normalized(bits=8)
                test_crop_mask = compute_crop_mask_manual(test_vol, test_roi_mask_file, test_ref_img_data,
                                                          roi_pos[0], roi_pos[1])

            # save the cropped initial masks
            for side in ['left', 'right']:
                if os.path.exists(init_mask_path):
                    if target == 'dentate':
                        init_dentate_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                  init_dentate_mask_pattern.format(test_patient_id, side))
                        cropped_init_dentate_mask_file = os.path.join(file_test_patient_dir,
                                                                      crop_init_dentate_mask_pattern.format(test_patient_id, side))
                        if (not os.path.exists(cropped_init_dentate_mask_file)) or set_new_roi_mask is True:
                            crop_image(init_dentate_mask_filepath, test_crop_mask, cropped_init_dentate_mask_file)

                    elif target == 'interposed':
                        init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                     init_interposed_mask_pattern.format(test_patient_id, side))
                        cropped_init_interposed_mask_file = os.path.join(file_test_patient_dir,
                                                                         crop_init_interposed_mask_pattern.format(test_patient_id, side))
                        if (not os.path.exists(cropped_init_interposed_mask_file)) or set_new_roi_mask is True:
                            crop_image(init_interposed_mask_filepath, test_crop_mask, cropped_init_interposed_mask_file)

                    else:
                        init_dentate_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                  init_dentate_mask_pattern.format(test_patient_id, side))
                        cropped_init_dentate_mask_file = os.path.join(file_test_patient_dir,
                                                                      crop_init_dentate_mask_pattern.format(test_patient_id, side))
                        if (not os.path.exists(cropped_init_dentate_mask_file)) or set_new_roi_mask is True:
                            crop_image(init_dentate_mask_filepath, test_crop_mask, cropped_init_dentate_mask_file)

                        init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                     init_interposed_mask_pattern.format(test_patient_id, side))
                        cropped_init_interposed_mask_file = os.path.join(file_test_patient_dir,
                                                                         crop_init_interposed_mask_pattern.format(test_patient_id, side))
                        if (not os.path.exists(cropped_init_interposed_mask_file)) or set_new_roi_mask is True:
                            crop_image(init_interposed_mask_filepath, test_crop_mask, cropped_init_interposed_mask_file)

                else:
                    init_mask_filepath = os.path.join(file_test_patient_dir, 'init_reg',
                                                      init_reg_mask_pattern.format(side, target))
                    if os.path.exists(init_mask_filepath):
                        cropped_init_reg_mask_file = os.path.join(file_test_patient_dir,
                                                                  crop_init_reg_mask_pattern.format(side, target))
                        if (not os.path.exists(cropped_init_reg_mask_file)) or set_new_roi_mask is True:
                            crop_image(init_mask_filepath, test_crop_mask, cropped_init_reg_mask_file)

            # save the cropped labels (test_roi)
            for side in ['left', 'right']:
                if target == 'dentate':
                    dentate_label_filepath = os.path.join(dataset_path, test_patient_id,
                                                          dentate_label_pattern.format(test_patient_id,
                                                                                       side))
                    if os.path.exists(dentate_label_filepath):
                        cropped_tst_dentate_label_file = os.path.join(file_test_patient_dir,
                                                                      crop_tst_dentate_label_pattern.format(
                                                                          test_patient_id, side))
                        if (not os.path.exists(cropped_tst_dentate_label_file)) or set_new_roi_mask is True:
                            crop_image(dentate_label_filepath, test_crop_mask,
                                       cropped_tst_dentate_label_file)

                elif target == 'interposed':
                    interposed_label_filepath = os.path.join(dataset_path, test_patient_id,
                                                             interposed_label_pattern.format(
                                                                 test_patient_id, side))
                    if os.path.exists(interposed_label_filepath):
                        cropped_tst_interposed_label_file = os.path.join(file_test_patient_dir,
                                                                         crop_tst_interposed_label_pattern.format(
                                                                             test_patient_id, side))
                        if (not os.path.exists(cropped_tst_interposed_label_file)) or set_new_roi_mask is True:
                            crop_image(interposed_label_filepath, test_crop_mask,
                                       cropped_tst_interposed_label_file)
                else:
                    dentate_label_filepath = os.path.join(dataset_path, test_patient_id,
                                                          dentate_label_pattern.format(test_patient_id,
                                                                                       side))
                    if os.path.exists(dentate_label_filepath):
                        cropped_tst_dentate_label_file = os.path.join(file_test_patient_dir,
                                                                      crop_tst_dentate_label_pattern.format(
                                                                          test_patient_id, side))
                        if (not os.path.exists(cropped_tst_dentate_label_file)) or set_new_roi_mask is True:
                            crop_image(dentate_label_filepath, test_crop_mask,
                                       cropped_tst_dentate_label_file)

                    interposed_label_filepath = os.path.join(dataset_path, test_patient_id,
                                                             interposed_label_pattern.format(
                                                                 test_patient_id, side))
                    if os.path.exists(interposed_label_filepath):
                        cropped_tst_interposed_label_file = os.path.join(file_test_patient_dir,
                                                                         crop_tst_interposed_label_pattern.format(
                                                                             test_patient_id, side))
                        if (not os.path.exists(cropped_tst_interposed_label_file)) or set_new_roi_mask is True:
                            crop_image(interposed_label_filepath, test_crop_mask,
                                       cropped_tst_interposed_label_file)

            # load test images in test_patient_lst and setting roi
            test_image_modal_lst = []
            test_fname_modality_lst = []
            for idx in modality_idx:
                if is_res_diff is True:
                    test_img_path = os.path.join(file_output_dir, test_patient_id,
                                                 img_resampled_name_pattern[idx].format(test_patient_id))
                else:
                    # test_img_path = os.path.join(dataset_path, test_patient_id,
                    #                              img_pattern[idx].format(test_patient_id))
                    test_img_path = ''
                    if not os.path.exists(test_img_path):
                        test_img_path = os.path.join(root_path, 'datasets', 'dcn', test_patient_id, 'image',
                                                     image_new_name_pattern[idx].format(test_patient_id))
                print(test_img_path)

                test_fname_modality_lst.append(test_img_path)
                test_vol = read_volume(test_img_path)
                if preprocess_tst == 2 or preprocess_tst == 3 or preprocess_tst == 5:
                    test_vol = normalize_image(test_vol, [0, 2 ** 8])
                if np.size(np.shape(test_vol)) == 4:
                    test_crop_vol = test_crop_mask.crop(test_vol[:, :, :, 0])
                else:
                    test_crop_vol = test_crop_mask.crop(test_vol)
                test_image_modal_lst.append(test_crop_vol)
            test_fname_lst.append(test_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                if is_res_diff is True:
                    test_img_path = os.path.join(file_output_dir, test_patient_id,
                                                 img_resampled_name_pattern[idx].format(test_patient_id))
                else:
                    # test_img_path = os.path.join(dataset_path, test_patient_id,
                    #                              img_pattern[idx].format(test_patient_id))
                    test_img_path = ''
                    if not os.path.exists(test_img_path):
                        test_img_path = os.path.join(root_path, 'datasets', 'dcn', test_patient_id, 'image',
                                                     image_new_name_pattern[idx].format(test_patient_id))
                crop_test_img_path = os.path.join(file_output_dir, test_patient_id,
                                                  crop_tst_img_pattern[idx].format(test_patient_id))
                if (not os.path.exists(crop_test_img_path)) or set_new_roi_mask is True:
                    crop_image(test_img_path, test_crop_mask, crop_test_img_path)

            test_crop_array = np.zeros((1, num_modality) + test_crop_vol.shape[0:3])
            for idx in range(num_modality):
                test_crop_array[0, idx] = test_image_modal_lst[idx]
            test_img_lst.append(test_crop_array)

    return train_img_lst, label_lst, test_img_lst, train_fname_lst, dentate_label_fname_lst, test_fname_lst, is_res_diff


def read_interposed_fastigial_dataset(gen_conf, train_conf, train_patient_lst, test_patient_lst, preprocess_trn,
                                      preprocess_tst, file_output_dir):
    # suit dentate + b0 -> suit interposed/fastigial learning => manual dentate + b0 -> estimation of manual interposed/fastigial

    dataset = train_conf['dataset']
    num_epochs = train_conf['num_epochs']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    mode = gen_conf['validation_mode']
    img_pattern = dataset_info['image_name_pattern']
    dentate_label_pattern = dataset_info['manual_corrected_pattern']
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    init_mask_pattern = dataset_info['initial_mask_pattern']

    suit_dentate_mask_pattern = dataset_info['suit_dentate_mask_pattern']
    suit_interposed_mask_pattern = dataset_info['suit_interposed_mask_pattern']
    suit_fastigial_mask_pattern = dataset_info['suit_fastigial_mask_pattern']

    set_new_roi_mask = dataset_info['set_new_roi_mask']
    margin_crop_mask = dataset_info['margin_crop_mask']

    crop_trn_img_pattern = dataset_info['crop_trn_image_name_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    crop_trn_label_pattern = dataset_info['crop_trn_manual_corrected_pattern']
    crop_tst_label_pattern = dataset_info['crop_tst_manual_corrected_pattern']
    crop_init_mask_pattern = dataset_info['crop_initial_mask_pattern']

    crop_suit_dentate_mask_pattern = dataset_info['crop_suit_dentate_mask_pattern']
    crop_suit_interposed_mask_pattern = dataset_info['crop_suit_interposed_mask_pattern']
    crop_suit_fastigial_mask_pattern = dataset_info['crop_suit_fastigial_mask_pattern']

    train_roi_mask_pattern = dataset_info['train_roi_mask_pattern']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']

    modality_idx = []
    for m in modality:
        if m == 'B0':
            modality_idx.append(0)
        if m == 'T1':
            modality_idx.append(1)
        if m == 'LP':
            modality_idx.append(2)
        if m == 'FA':
            modality_idx.append(3)

    label_mapper = {0: 0, 10: 1, 150: 2}

    # load multi-modal images in train_patient_lst and setting roi
    train_img_suit_dentate_lst = []
    suit_interposed_fasigial_lst = []

    train_fname_lst = []
    dentate_label_fname_lst = []
    if train_patient_lst[0] is not None and num_epochs != 0 and mode != '2':
        for train_patient_id in train_patient_lst:

            file_patient_dir = file_output_dir + '/' + train_patient_id
            if not os.path.exists(file_patient_dir):
                os.makedirs(os.path.join(file_output_dir, train_patient_id))

            train_roi_mask_file = file_output_dir + train_roi_mask_pattern.format(train_patient_id)
            if os.path.exists(train_roi_mask_file) and set_new_roi_mask is True:
                os.remove(train_roi_mask_file)

            # load label images in train_patient_lst and setting roi

            suit_dentate_vol_lst = []
            suit_interposed_mask_vol_lst = []
            suit_fastigial_mask_vol_lst = []
            for side in ['left', 'right']:  # left + right
                suit_dentate_filepath = dataset_path + train_patient_id + suit_dentate_mask_pattern.format(train_patient_id, side)
                print(suit_dentate_filepath)
                suit_dentate_data = read_volume_data(suit_dentate_filepath)
                suit_dentate_vol = suit_dentate_data.get_data()
                if np.size(np.shape(suit_dentate_vol)) == 4:
                    suit_dentate_vol_lst.append(suit_dentate_vol[:, :, :, 0])
                else:
                    suit_dentate_vol_lst.append(suit_dentate_vol)

                suit_interposed_mask_filepath = dataset_path + train_patient_id + suit_interposed_mask_pattern.format(train_patient_id, side)
                print(suit_interposed_mask_filepath)
                suit_interposed_mask_data = read_volume_data(suit_interposed_mask_filepath)
                suit_interposed_mask_vol = suit_interposed_mask_data.get_data()
                if np.size(np.shape(suit_interposed_mask_vol)) == 4:
                    suit_interposed_mask_vol_lst.append(suit_interposed_mask_vol[:, :, :, 0])
                else:
                    suit_interposed_mask_vol_lst.append(suit_interposed_mask_vol)

                suit_fastigial_mask_filepath = dataset_path + train_patient_id + suit_fastigial_mask_pattern.format(train_patient_id, side)
                print(suit_fastigial_mask_filepath)
                suit_fastigial_mask_data = read_volume_data(suit_fastigial_mask_filepath)
                suit_fastigial_mask_vol = suit_fastigial_mask_data.get_data()
                if np.size(np.shape(suit_fastigial_mask_vol)) == 4:
                    suit_fastigial_mask_vol_lst.append(suit_fastigial_mask_vol[:, :, :, 0])
                else:
                    suit_fastigial_mask_vol_lst.append(suit_fastigial_mask_vol)

            suit_dentate_merge_vol = suit_dentate_vol_lst[0] + suit_dentate_vol_lst[1]
            suit_interposed_mask_merge_vol = suit_interposed_mask_vol_lst[0] + suit_interposed_mask_vol_lst[1]
            suit_fastigial_mask_merge_vol = suit_fastigial_mask_vol_lst[0] + suit_fastigial_mask_vol_lst[1]
            trn_merge_vol = suit_dentate_merge_vol + suit_interposed_mask_merge_vol + suit_fastigial_mask_merge_vol

            suit_interposed_mask_merge_vol[suit_interposed_mask_merge_vol == np.max(suit_interposed_mask_merge_vol)] = 10
            suit_fastigial_mask_merge_vol[suit_fastigial_mask_merge_vol == np.max(suit_fastigial_mask_merge_vol)] = 150
            suit_interposed_fastigial_mask_merge_vol = suit_interposed_mask_merge_vol + suit_fastigial_mask_merge_vol

            # crop the roi of the image
            train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(trn_merge_vol,
                                                                                       train_roi_mask_file,
                                                                                       suit_dentate_data, margin_crop_mask)
            suit_dentate_crop_vol = train_crop_mask.crop(suit_dentate_merge_vol)
            suit_interposed_fastigial_crop_vol = train_crop_mask.crop(suit_interposed_fastigial_mask_merge_vol)
            for key in label_mapper.keys():
                suit_interposed_fastigial_crop_vol[suit_interposed_fastigial_crop_vol == key] = label_mapper[key]

            suit_dentate_crop_array = np.zeros((1, 1) + suit_dentate_crop_vol.shape[0:3])
            suit_dentate_crop_array[0, 0] = suit_dentate_crop_vol

            suit_interposed_fastigial_crop_array = np.zeros((1, 1) + suit_interposed_fastigial_crop_vol.shape[0:3])
            suit_interposed_fastigial_crop_array[0, 0] = suit_interposed_fastigial_crop_vol
            suit_interposed_fasigial_lst.append(suit_interposed_fastigial_crop_array)

            for side in ['left', 'right']:
                # save the cropped suit dentate mask (train_roi)
                suit_dentate_filepath = dataset_path + train_patient_id + suit_dentate_mask_pattern.format(train_patient_id, side)
                cropped_suit_dentate_mask_file = file_output_dir + '/' + train_patient_id + crop_suit_dentate_mask_pattern.format(
                    train_patient_id, side)
                if (not os.path.exists(cropped_suit_dentate_mask_file)) or set_new_roi_mask is True:
                    crop_image(suit_dentate_filepath, train_crop_mask, cropped_suit_dentate_mask_file)

                # save the cropped suit interposed mask (test_roi)
                suit_interposed_mask_filepath = dataset_path + train_patient_id + suit_interposed_mask_pattern.format(
                    train_patient_id, side)
                cropped_suit_interposed_mask_file = file_output_dir + '/' + train_patient_id + \
                                                    crop_suit_interposed_mask_pattern.format(train_patient_id, side)
                if (not os.path.exists(cropped_suit_interposed_mask_file)) or set_new_roi_mask is True:
                    crop_image(suit_interposed_mask_filepath, train_crop_mask, cropped_suit_interposed_mask_file)

                # save the cropped suit fastigial mask (test_roi)
                suit_fastigial_mask_filepath = dataset_path + train_patient_id + suit_fastigial_mask_pattern.format(
                    train_patient_id, side)
                cropped_suit_fastigial_mask_file = file_output_dir + '/' + train_patient_id + \
                                                   crop_suit_fastigial_mask_pattern.format(train_patient_id, side)
                if (not os.path.exists(cropped_suit_fastigial_mask_file)) or set_new_roi_mask is True:
                    crop_image(suit_fastigial_mask_filepath, train_crop_mask, cropped_suit_fastigial_mask_file)

            # load training images in train_patient_lst and setting roi
            train_image_modal_lst= []
            train_fname_modality_lst = []
            for idx in modality_idx:
                training_img_path = dataset_path + train_patient_id + '/' + img_pattern[idx].format(train_patient_id)
                print(training_img_path)
                train_fname_modality_lst.append(training_img_path)
                training_vol = read_volume(training_img_path)
                if preprocess_trn == 2 or preprocess_trn == 3 or preprocess_trn == 5:
                    training_vol = normalize_image(training_vol, [0, 2 ** 8])

                if np.size(np.shape(training_vol)) == 4:
                    train_crop_vol = train_crop_mask.crop(training_vol[:, :, :, 0])
                else:
                    train_crop_vol = train_crop_mask.crop(training_vol)
                train_image_modal_lst.append(train_crop_vol)

                 # save the cropped image
                crop_train_img_path = file_output_dir + '/' + train_patient_id + '/' + \
                                           crop_trn_img_pattern[idx].format(train_patient_id)
                if (not os.path.exists(crop_train_img_path)) or set_new_roi_mask is True:
                    crop_image(training_img_path, train_crop_mask, crop_train_img_path)

            train_fname_lst.append(train_fname_modality_lst)

            # combine training image with suit_dentate_crop as training input
            #train_suit_dentate_crop_array = np.zeros((1, num_modality) + train_crop_vol.shape[0:3])
            train_suit_dentate_crop_array = np.zeros((1, 1) + train_crop_vol.shape[0:3])
            train_suit_dentate_crop_array[0, 0] = suit_dentate_crop_array
            # for idx in range(num_modality):
            #     train_suit_dentate_crop_array[0, idx] = train_image_modal_lst[idx]
            # #train_suit_dentate_crop_array[0, num_modality] = suit_dentate_crop_array
            train_img_suit_dentate_lst.append(train_suit_dentate_crop_array)

    # load multi-modal images in test_patient_lst and setting roi
    test_img_dentate_label_lst = []
    test_fname_lst = []
    if test_patient_lst[0] is not None:
        for test_patient_id in test_patient_lst:

            file_patient_dir = file_output_dir + '/' + test_patient_id
            if not os.path.exists(file_patient_dir):
                os.makedirs(os.path.join(file_output_dir, test_patient_id))

            test_roi_mask_file = file_output_dir + test_roi_mask_pattern.format(test_patient_id)
            if os.path.exists(test_roi_mask_file) and set_new_roi_mask is True:
                os.remove(test_roi_mask_file)

            # load initial mask (SUIT output or linearly registered from ref. training image) for setting roi
            init_mask_vol_lst = []
            dentate_label_vol_lst = []
            dentate_label_fname_side_lst = []
            for side in ['left', 'right']:  # left + right
                init_mask_filepath = dataset_path + test_patient_id + init_mask_pattern.format(test_patient_id, side)
                print(init_mask_filepath)
                init_mask_data = read_volume_data(init_mask_filepath)
                init_mask_vol = init_mask_data.get_data()
                if np.size(np.shape(init_mask_vol)) == 4:
                    init_mask_vol_lst.append(init_mask_vol[:, :, :, 0])
                else:
                    init_mask_vol_lst.append(init_mask_vol)

                dentate_label_filepath = dataset_path + test_patient_id + dentate_label_pattern.format(test_patient_id, side)
                print(dentate_label_filepath)
                dentate_label_fname_side_lst.append(dentate_label_filepath)
                dentate_label_data = read_volume_data(dentate_label_filepath)
                dentate_label_vol = dentate_label_data.get_data()
                if np.size(np.shape(dentate_label_vol)) == 4:
                    dentate_label_vol_lst.append(dentate_label_vol[:, :, :, 0])
                else:
                    dentate_label_vol_lst.append(dentate_label_vol)

            dentate_label_fname_lst.append(dentate_label_fname_side_lst)
            #init_mask_merge_vol = init_mask_vol_lst[0] + init_mask_vol_lst[1]
            dentate_label_merge_vol = dentate_label_vol_lst[0] + dentate_label_vol_lst[1]

            # crop the roi of the image
            test_crop_mask = find_crop_mask(test_roi_mask_file) or compute_crop_mask(dentate_label_merge_vol,
                                                                                     test_roi_mask_file,
                                                                                     init_mask_data,
                                                                                     margin_crop_mask)
            dentate_label_crop_vol = test_crop_mask.crop(dentate_label_merge_vol)
            dentate_label_crop_array = np.zeros((1, 1) + dentate_label_crop_vol.shape[0:3])
            dentate_label_crop_array[0, 0] = dentate_label_crop_vol

            for side in ['left', 'right']:
                # save the cropped initial masks (test_roi)
                init_mask_filepath = dataset_path + test_patient_id + init_mask_pattern.format(test_patient_id, side)
                cropped_init_mask_file = file_output_dir + '/' + test_patient_id + \
                                         crop_init_mask_pattern.format(test_patient_id, side)
                if (not os.path.exists(cropped_init_mask_file)) or set_new_roi_mask is True:
                    crop_image(init_mask_filepath, test_crop_mask, cropped_init_mask_file)

                # save the cropped suit labels (test_roi)
                dentate_label_filepath = dataset_path + test_patient_id + dentate_label_pattern.format(test_patient_id, side)
                cropped_tst_dentate_label_file = file_output_dir + '/'  + test_patient_id + \
                                         crop_tst_label_pattern.format(test_patient_id, side)
                if (not os.path.exists(cropped_tst_dentate_label_file)) or set_new_roi_mask is True:
                    crop_image(dentate_label_filepath, test_crop_mask, cropped_tst_dentate_label_file)

            # load test images in test_patient_lst and setting roi
            test_image_modal_lst = []
            test_fname_modality_lst = []
            for idx in modality_idx:
                test_img_path = dataset_path + test_patient_id + '/' + img_pattern[idx].format(test_patient_id)
                print(test_img_path)
                test_fname_modality_lst.append(test_img_path)
                test_vol = read_volume(test_img_path)
                if preprocess_tst == 2 or preprocess_tst == 3 or preprocess_tst == 5:
                    test_vol = normalize_image(test_vol, [0, 2 ** 8])
                if np.size(np.shape(test_vol)) == 4:
                    test_crop_vol = test_crop_mask.crop(test_vol[:, :, :, 0])
                else:
                    test_crop_vol = test_crop_mask.crop(test_vol)
                test_image_modal_lst.append(test_crop_vol)

                # save the cropped image
                test_img_path = dataset_path + test_patient_id + '/' + img_pattern[idx].format(test_patient_id)
                crop_test_img_path = file_output_dir + '/' + test_patient_id + '/' + \
                                     crop_tst_img_pattern[idx].format(test_patient_id)
                if (not os.path.exists(crop_test_img_path)) or set_new_roi_mask is True:
                    crop_image(test_img_path, test_crop_mask, crop_test_img_path)

            test_fname_lst.append(test_fname_modality_lst)

            # combine training image with suit_interposed_crop and suit_fastigial_crop, respectively, as test input
            #test_dentate_label_crop_array = np.zeros((1, num_modality) + test_crop_vol.shape[0:3])
            test_dentate_label_crop_array = np.zeros((1, 1) + test_crop_vol.shape[0:3])
            # test_dentate_label_crop_array[0, 0] = dentate_label_crop_array
            # for idx in range(num_modality):
            #     test_dentate_label_crop_array[0, idx] = test_image_modal_lst[idx]
            # #test_dentate_label_crop_array[0, num_modality] = dentate_label_crop_array
            test_img_dentate_label_lst.append(test_dentate_label_crop_array)

    return train_img_suit_dentate_lst, suit_interposed_fasigial_lst, test_img_dentate_label_lst, train_fname_lst, \
           dentate_label_fname_lst, test_fname_lst


def read_interposed_fastigial_dataset_rev(gen_conf, train_conf, train_patient_lst, test_patient_lst, preprocess_trn,
                                      preprocess_tst, file_output_dir):
    # suit dentate -> manual dentate learning => suit interposed/fastigial -> estimation of manualinterposed/fastigial

    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    img_pattern = dataset_info['image_name_pattern']
    dentate_label_pattern = dataset_info['manual_corrected_pattern']
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    init_mask_pattern = dataset_info['initial_mask_pattern']

    suit_dentate_mask_pattern = dataset_info['suit_dentate_mask_pattern']
    suit_interposed_mask_pattern = dataset_info['suit_interposed_mask_pattern']
    suit_fastigial_mask_pattern = dataset_info['suit_fastigial_mask_pattern']

    set_new_roi_mask = dataset_info['set_new_roi_mask']
    margin_crop_mask = dataset_info['margin_crop_mask']

    crop_trn_img_pattern = dataset_info['crop_trn_image_name_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    crop_trn_label_pattern = dataset_info['crop_trn_manual_corrected_pattern']
    #crop_tst_label_pattern = dataset_info['crop_tst_manual_corrected_pattern']
    #crop_init_mask_pattern = dataset_info['crop_initial_mask_pattern']

    crop_suit_dentate_mask_pattern = dataset_info['crop_suit_dentate_mask_pattern']
    crop_suit_interposed_mask_pattern = dataset_info['crop_suit_interposed_mask_pattern']
    crop_suit_fastigial_mask_pattern = dataset_info['crop_suit_fastigial_mask_pattern']

    train_roi_mask_pattern = dataset_info['train_roi_mask_pattern']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']

    modality_idx = []
    for m in modality:
        if m == 'B0':
            modality_idx.append(0)
        if m == 'T1':
            modality_idx.append(1)
        if m == 'LP':
            modality_idx.append(2)
        if m == 'FA':
            modality_idx.append(3)

    #label_mapper = {0: 0, 10: 1, 150: 2}

    # load multi-modal images in train_patient_lst and setting roi
    train_img_suit_dentate_lst = []
    dentate_label_crop_lst = []
    train_fname_lst = []
    dentate_label_fname_lst = []
    if train_patient_lst[0] is not None:
        for train_patient_id in train_patient_lst:

            file_patient_dir = file_output_dir + '/' + train_patient_id
            if not os.path.exists(file_patient_dir):
                os.makedirs(os.path.join(file_output_dir, train_patient_id))

            train_roi_mask_file = file_output_dir + train_roi_mask_pattern.format(train_patient_id)
            if os.path.exists(train_roi_mask_file) and set_new_roi_mask is True:
                os.remove(train_roi_mask_file)

            # load label images in train_patient_lst and setting roi
            init_mask_vol_lst = []
            suit_dentate_vol_lst = []
            dentate_label_vol_lst = []
            dentate_label_fname_side_lst = []
            for side in ['left', 'right']:  # left + right
                # init_mask_filepath = dataset_path + test_patient_id + init_mask_pattern.format(test_patient_id, side)
                # print(init_mask_filepath)
                # init_mask_data = read_volume_data(init_mask_filepath)
                # init_mask_vol = init_mask_data.get_data()
                # if np.size(np.shape(init_mask_vol)) == 4:
                #     init_mask_vol_lst.append(init_mask_vol[:, :, :, 0])
                # else:
                #     init_mask_vol_lst.append(init_mask_vol)

                suit_dentate_filepath = dataset_path + train_patient_id + suit_dentate_mask_pattern.format(train_patient_id, side)
                print(suit_dentate_filepath)
                suit_dentate_data = read_volume_data(suit_dentate_filepath)
                suit_dentate_vol = suit_dentate_data.get_data()

                if np.size(np.shape(suit_dentate_vol)) == 4:
                    suit_dentate_vol_lst.append(suit_dentate_vol[:, :, :, 0])
                else:
                    suit_dentate_vol_lst.append(suit_dentate_vol)

                dentate_label_filepath = dataset_path + train_patient_id + dentate_label_pattern.format(train_patient_id, side)
                print(dentate_label_filepath)
                dentate_label_fname_side_lst.append(dentate_label_filepath)
                dentate_label_data = read_volume_data(dentate_label_filepath)
                dentate_label_vol = dentate_label_data.get_data()

                if np.size(np.shape(dentate_label_vol)) == 4:
                    dentate_label_vol_lst.append(dentate_label_vol[:, :, :, 0])
                else:
                    dentate_label_vol_lst.append(dentate_label_vol)

            dentate_label_fname_lst.append(dentate_label_fname_side_lst)
            #init_mask_merge_vol = init_mask_vol_lst[0] + init_mask_vol_lst[1]
            suit_dentate_merge_vol = suit_dentate_vol_lst[0] + suit_dentate_vol_lst[1]
            dentate_label_merge_vol = dentate_label_vol_lst[0] + dentate_label_vol_lst[1]
            trn_merge_vol = suit_dentate_merge_vol + dentate_label_merge_vol

            # crop the roi of the image
            train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(trn_merge_vol,
                                                                                       train_roi_mask_file,
                                                                                       suit_dentate_data, margin_crop_mask)
            suit_dentate_crop_vol = train_crop_mask.crop(suit_dentate_merge_vol)
            # resample


            suit_dentate_crop_array = np.zeros((1, 1) + suit_dentate_crop_vol.shape[0:3])
            suit_dentate_crop_array[0, 0] = suit_dentate_crop_vol

            dentate_label_crop_vol = train_crop_mask.crop(dentate_label_merge_vol)
            # resample

            dentate_label_crop_array = np.zeros((1, 1) + dentate_label_crop_vol.shape[0:3])
            dentate_label_crop_array[0, 0] = dentate_label_crop_vol
            dentate_label_crop_lst.append(dentate_label_crop_array)

            for side in ['left', 'right']:
                # save the cropped initial masks (train_roi)
                # init_mask_filepath = dataset_path + train_patient_id + init_mask_pattern.format(train_patient_id, side)
                # cropped_init_mask_file = file_output_dir + '/' + train_patient_id + \
                #                          crop_init_mask_pattern.format(train_patient_id, side)
                # if (not os.path.exists(cropped_init_mask_file)) or set_new_roi_mask is True:
                #     crop_image(init_mask_filepath, train_crop_mask, cropped_init_mask_file)

                # save the cropped suit dentate mask (train_roi)
                suit_dentate_filepath = dataset_path + train_patient_id + suit_dentate_mask_pattern.format(train_patient_id, side)
                cropped_suit_dentate_mask_file = file_output_dir + '/' + train_patient_id + crop_suit_dentate_mask_pattern.format(
                    train_patient_id, side)
                if (not os.path.exists(cropped_suit_dentate_mask_file)) or set_new_roi_mask is True:
                    crop_image(suit_dentate_filepath, train_crop_mask, cropped_suit_dentate_mask_file)

                # save the cropped suit labels (train_roi)
                dentate_label_filepath = dataset_path + train_patient_id + dentate_label_pattern.format(train_patient_id, side)
                cropped_trn_dentate_label_file = file_output_dir + '/'  + train_patient_id + \
                                         crop_trn_label_pattern.format(train_patient_id, side)
                if (not os.path.exists(cropped_trn_dentate_label_file)) or set_new_roi_mask is True:
                    crop_image(dentate_label_filepath, train_crop_mask, cropped_trn_dentate_label_file)

            # load training images in train_patient_lst and setting roi
            train_image_modal_lst= []
            train_fname_modality_lst = []
            for idx in modality_idx:
                training_img_path = dataset_path + train_patient_id + '/' + img_pattern[idx].format(train_patient_id)
                print(training_img_path)
                train_fname_modality_lst.append(training_img_path)
                training_vol = read_volume(training_img_path)

                if preprocess_trn == 2 or preprocess_trn == 3 or preprocess_trn == 5:
                    training_vol = normalize_image(training_vol, [0, 2 ** 8])

                if np.size(np.shape(training_vol)) == 4:
                    train_crop_vol = train_crop_mask.crop(training_vol[:, :, :, 0])
                else:
                    train_crop_vol = train_crop_mask.crop(training_vol)

                #resample


                train_image_modal_lst.append(train_crop_vol)

                 # save the cropped image
                crop_train_img_path = file_output_dir + '/' + train_patient_id + '/' + \
                                           crop_trn_img_pattern[idx].format(train_patient_id)
                if (not os.path.exists(crop_train_img_path)) or set_new_roi_mask is True:
                    crop_image(training_img_path, train_crop_mask, crop_train_img_path)

            train_fname_lst.append(train_fname_modality_lst)

            # combine training image with suit_dentate_crop as training input
            #train_suit_dentate_crop_array = np.zeros((1, num_modality) + train_crop_vol.shape[0:3])
            #train_suit_dentate_crop_array = np.zeros((1, 1) + train_crop_vol.shape[0:3])
            #train_suit_dentate_crop_array[0, 0] = suit_dentate_crop_array
            # for idx in range(num_modality):
            #     train_suit_dentate_crop_array[0, idx] = train_image_modal_lst[idx]
            # #train_suit_dentate_crop_array[0, num_modality] = suit_dentate_crop_array
            train_img_suit_dentate_lst.append(suit_dentate_crop_array)

    # load multi-modal images in test_patient_lst and setting roi
    test_img_suit_interposed_fastigial_lst = []
    test_fname_lst = []
    if test_patient_lst[0] is not None:
        for test_patient_id in test_patient_lst:

            file_patient_dir = file_output_dir + '/' + test_patient_id
            if not os.path.exists(file_patient_dir):
                os.makedirs(os.path.join(file_output_dir, test_patient_id))

            test_roi_mask_file = file_output_dir + test_roi_mask_pattern.format(test_patient_id)
            if os.path.exists(test_roi_mask_file) and set_new_roi_mask is True:
                os.remove(test_roi_mask_file)

            # load initial mask (SUIT output or linearly registered from ref. training image) for setting roi
            suit_interposed_mask_vol_lst = []
            suit_fastigial_mask_vol_lst = []
            for side in ['left', 'right']:  # left + right

                suit_interposed_mask_filepath = dataset_path + test_patient_id + suit_interposed_mask_pattern.format(test_patient_id, side)
                print(suit_interposed_mask_filepath)
                suit_interposed_mask_data = read_volume_data(suit_interposed_mask_filepath)
                suit_interposed_mask_vol = suit_interposed_mask_data.get_data()
                if np.size(np.shape(suit_interposed_mask_vol)) == 4:
                    suit_interposed_mask_vol_lst.append(suit_interposed_mask_vol[:, :, :, 0])
                else:
                    suit_interposed_mask_vol_lst.append(suit_interposed_mask_vol)

                suit_fastigial_mask_filepath = dataset_path + test_patient_id + suit_fastigial_mask_pattern.format(test_patient_id, side)
                print(suit_fastigial_mask_filepath)
                suit_fastigial_mask_data = read_volume_data(suit_fastigial_mask_filepath)
                suit_fastigial_mask_vol = suit_fastigial_mask_data.get_data()
                if np.size(np.shape(suit_fastigial_mask_vol)) == 4:
                    suit_fastigial_mask_vol_lst.append(suit_fastigial_mask_vol[:, :, :, 0])
                else:
                    suit_fastigial_mask_vol_lst.append(suit_fastigial_mask_vol)

            suit_interposed_mask_merge_vol = suit_interposed_mask_vol_lst[0] + suit_interposed_mask_vol_lst[1]
            suit_fastigial_mask_merge_vol = suit_fastigial_mask_vol_lst[0] + suit_fastigial_mask_vol_lst[1]

            # suit_interposed_mask_merge_vol[
            #     suit_interposed_mask_merge_vol == np.max(suit_interposed_mask_merge_vol)] = 10
            # suit_fastigial_mask_merge_vol[suit_fastigial_mask_merge_vol == np.max(suit_fastigial_mask_merge_vol)] = 150
            suit_interposed_fastigial_mask_merge_vol = suit_interposed_mask_merge_vol + suit_fastigial_mask_merge_vol

            # crop the roi of the image
            test_crop_mask = find_crop_mask(test_roi_mask_file) or compute_crop_mask(suit_interposed_fastigial_mask_merge_vol,
                                                                                     test_roi_mask_file,
                                                                                     suit_interposed_mask_data,
                                                                                     margin_crop_mask)

            suit_interposed_fastigial_crop_vol = test_crop_mask.crop(suit_interposed_fastigial_mask_merge_vol)
            # for key in label_mapper.keys():
            #     suit_interposed_fastigial_crop_vol[suit_interposed_fastigial_crop_vol == key] = label_mapper[key]

            # resample


            suit_interposed_fastigial_crop_array = np.zeros((1, 1) + suit_interposed_fastigial_crop_vol.shape[0:3])
            suit_interposed_fastigial_crop_array[0, 0] = suit_interposed_fastigial_crop_vol

            for side in ['left', 'right']:
                # save the cropped suit interposed mask (test_roi)
                suit_interposed_mask_filepath = dataset_path + test_patient_id + suit_interposed_mask_pattern.format(
                    test_patient_id, side)
                cropped_suit_interposed_mask_file = file_output_dir + '/' + test_patient_id + \
                                                    crop_suit_interposed_mask_pattern.format(test_patient_id, side)
                if (not os.path.exists(cropped_suit_interposed_mask_file)) or set_new_roi_mask is True:
                    crop_image(suit_interposed_mask_filepath, test_crop_mask, cropped_suit_interposed_mask_file)

                # save the cropped suit fastigial mask (test_roi)
                suit_fastigial_mask_filepath = dataset_path + test_patient_id + suit_fastigial_mask_pattern.format(
                    test_patient_id, side)
                cropped_suit_fastigial_mask_file = file_output_dir + '/' + test_patient_id + \
                                                   crop_suit_fastigial_mask_pattern.format(test_patient_id, side)
                if (not os.path.exists(cropped_suit_fastigial_mask_file)) or set_new_roi_mask is True:
                    crop_image(suit_fastigial_mask_filepath, test_crop_mask, cropped_suit_fastigial_mask_file)

            # load test images in test_patient_lst and setting roi
            test_image_modal_lst = []
            test_fname_modality_lst = []
            for idx in modality_idx:
                test_img_path = dataset_path + test_patient_id + '/' + img_pattern[idx].format(test_patient_id)
                print(test_img_path)
                test_fname_modality_lst.append(test_img_path)
                test_vol = read_volume(test_img_path)
                if preprocess_tst == 2 or preprocess_tst == 3 or preprocess_tst == 5:
                    test_vol = normalize_image(test_vol, [0, 2 ** 8])
                if np.size(np.shape(test_vol)) == 4:
                    test_crop_vol = test_crop_mask.crop(test_vol[:, :, :, 0])
                else:
                    test_crop_vol = test_crop_mask.crop(test_vol)

                #resample

                test_image_modal_lst.append(test_crop_vol)

                # save the cropped image
                test_img_path = dataset_path + test_patient_id + '/' + img_pattern[idx].format(test_patient_id)
                crop_test_img_path = file_output_dir + '/' + test_patient_id + '/' + \
                                     crop_tst_img_pattern[idx].format(test_patient_id)
                if (not os.path.exists(crop_test_img_path)) or set_new_roi_mask is True:
                    crop_image(test_img_path, test_crop_mask, crop_test_img_path)

            test_fname_lst.append(test_fname_modality_lst)

            # combine training image with suit_interposed_crop and suit_fastigial_crop, respectively, as test input
            #test_suit_interposed_fastigial_crop_array = np.zeros((1, num_modality) + test_crop_vol.shape[0:3])
            #test_suit_interposed_fastigial_crop_array = np.zeros((1, 1) + test_crop_vol.shape[0:3])
            #test_suit_interposed_fastigial_crop_array[0, 0] = suit_interposed_fastigial_crop_array
            # for idx in range(num_modality):
            #     test_suit_interposed_fastigial_crop_array[0, idx] = test_image_modal_lst[idx]
            # #test_suit_interposed_fastigial_crop_array[0, num_modality] = suit_interposed_fastigial_crop_array
            test_img_suit_interposed_fastigial_lst.append(suit_interposed_fastigial_crop_array)

    return train_img_suit_dentate_lst, dentate_label_crop_lst, test_img_suit_interposed_fastigial_lst, train_fname_lst, \
           dentate_label_fname_lst, test_fname_lst


def build_sar_dataset_dict(dataset_path, dataset_fname, data_id, start_id, num_dataset_files, file_output_dir,
                           dataset_format, head_model, modality):

    # basic modality images (B1+ maps - real/imaginary)
    dataset_dict_org = dict()
    dataset_dict = dict()
    for df_idx in range(num_dataset_files):
        mat = loadmat(os.path.join(dataset_path, dataset_fname[df_idx]))
        dataset_dict_org.update(mat)

    # dataset_dict_duke = dict(list(mat1.items()) + list(mat2.items()))
    # dataset_dict.pop('__header__', None)
    # dataset_dict.pop('__version__', None)
    # dataset_dict.pop('__globals__', None)

    for id in range(data_id[0], data_id[1]+1):
        b1_src_name = 'B1_Ex%s' % str(id)
        sar_src_name = 'SAR_Ex%s' % str(id)

        b1_mag_name = 'B1_mag_Ex%s' % str(id+start_id-1)
        b1_real_name = 'B1_real_Ex%s' % str(id+start_id-1)
        b1_imag_name = 'B1_imag_Ex%s' % str(id+start_id-1)
        sar_name = 'SAR_Ex%s' % str(id+start_id-1)

        dataset_dict[b1_mag_name] = np.abs(dataset_dict_org[b1_src_name])
        dataset_dict[b1_real_name] = np.real(dataset_dict_org[b1_src_name])
        dataset_dict[b1_imag_name] = np.imag(dataset_dict_org[b1_src_name])
        dataset_dict[sar_name] = dataset_dict_org[sar_src_name]

    # save mask
    dataset_dict_tissue = loadmat(os.path.join(dataset_path, dataset_fname[-1]))
    dataset_dict['Mask'] = dataset_dict_tissue['Mask']

    # # remove background noise
    mask = remove_outliers(dataset_dict['Mask'])
    # mask = dataset_dict['Mask']

    roi_mask_file = os.path.join(file_output_dir, 'roi_mask_%s.nii.gz' % head_model)
    if os.path.exists(roi_mask_file):
        os.remove(roi_mask_file)
    __save_sar_volume(mask, roi_mask_file, dataset_format, 'float32', is_compressed=False)
    print('saved sar roi mask data of %s model as nii.gz' % head_model)
    crop_mask = find_crop_mask(roi_mask_file)

    # additional tissue property data (e.g., sigma)
    if 'Epsilon' in modality:
        dataset_dict['Epsilon'] = dataset_dict_tissue['Epsilon']
    if 'Rho' in modality:
        dataset_dict['Rho'] = dataset_dict_tissue['Rho']
    if 'Sigma' in modality:
        dataset_dict['Sigma'] = dataset_dict_tissue['Sigma']

    return dataset_dict, crop_mask


def build_sar_dataset_h5py(dataset_path, dataset_fname, data_id, start_id, num_dataset_files, file_output_dir,
                           dataset_format, head_model, modality):

    # basic modality images (B1+ maps - real/imaginary)
    dataset_dict_org = dict()
    dataset_dict = dict()
    for df_idx in range(num_dataset_files):
        f1 = h5py.File(os.path.join(dataset_path, dataset_fname[df_idx]), 'r')
        for key in f1.keys():
            dataset_dict_org.update({key: f1[key]})

    for id in range(data_id[0], data_id[1]+1):
        b1_src_name = 'B1_Ex%s' % str(id)
        sar_src_name = 'SAR_Ex%s' % str(id)

        b1_mag_name = 'B1_mag_Ex%s' % str(id+start_id-1)
        b1_real_name = 'B1_real_Ex%s' % str(id+start_id-1)
        b1_imag_name = 'B1_imag_Ex%s' % str(id+start_id-1)
        sar_name = 'SAR_Ex%s' % str(id+start_id-1)

        #dataset_dict[b1_mag_name] = np.abs(np.transpose(dataset_dict_org[b1_src_name], (1, 2, 0)))
        dataset_dict[b1_real_name] = np.transpose(dataset_dict_org[b1_src_name], (1, 2, 0))['real']
        dataset_dict[b1_imag_name] = np.transpose(dataset_dict_org[b1_src_name], (1, 2, 0))['imag']
        dataset_dict[b1_mag_name] = np.abs(dataset_dict[b1_real_name] + dataset_dict[b1_imag_name] * 1j)
        dataset_dict[sar_name] = np.transpose(dataset_dict_org[sar_src_name], (1, 2, 0))

    # save mask
    dataset_dict_tissue = h5py.File(os.path.join(dataset_path, dataset_fname[-1]), 'r')
    dataset_dict['Mask'] = np.transpose(dataset_dict_tissue['Mask'], (1, 2, 0))

    # # remove background noise
    mask = remove_outliers(dataset_dict['Mask'])

    roi_mask_file = os.path.join(file_output_dir, 'roi_mask_%s.nii.gz' % head_model)
    if os.path.exists(roi_mask_file):
        os.remove(roi_mask_file)
    __save_sar_volume(mask, roi_mask_file, dataset_format, 'float32', is_compressed=False)
    print('saved sar roi mask data of %s model as nii.gz' % head_model)
    crop_mask = find_crop_mask(roi_mask_file)

    # additional tissue property data (e.g., sigma)
    if 'Epsilon' in modality:
        dataset_dict['Epsilon'] = np.transpose(dataset_dict_tissue['Epsilon'], (1, 2, 0))
    if 'Rho' in modality:
        dataset_dict['Rho'] = np.transpose(dataset_dict_tissue['Rho'], (1, 2, 0))
    if 'Sigma' in modality:
        dataset_dict['Sigma'] = np.transpose(dataset_dict_tissue['Sigma'], (1, 2, 0))

    dataset_dict_tissue.close()

    return dataset_dict, crop_mask


def read_sar_dataset(gen_conf, train_conf, train_id_lst, test_id_lst, is_trn_inference):

    #from collections import Counter

    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_format = dataset_info['format']
    dataset_fname_duke = dataset_info['dataset_filename_duke']
    dataset_fname_ella = dataset_info['dataset_filename_ella']
    dataset_fname_louis = dataset_info['dataset_filename_louis']
    dataset_fname_austinman = dataset_info['dataset_filename_austinman']
    dataset_fname_austinwoman = dataset_info['dataset_filename_austinwoman']
    slice_crop_size_org = dataset_info['margin_crop_mask']
    modality = dataset_info['image_modality']
    augment_sar_data = dataset_info['augment_sar_data']
    data_augment = augment_sar_data[0]
    num_gen = augment_sar_data[1]
    normalized_range_min = dataset_info['normalized_range_min']
    normalized_range_max = dataset_info['normalized_range_max']
    normalized_range = [normalized_range_min, normalized_range_max]
    bg_value = normalized_range_min

    num_modality = len(modality)
    num_epochs = train_conf['num_epochs']
    mode = gen_conf['validation_mode']
    preprocess_trn = train_conf['preprocess']

    root_path = gen_conf['root_path']
    results_path = gen_conf['results_path']
    folder_names = dataset_info['folder_names']
    file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])
    if not os.path.exists(file_output_dir):
        os.makedirs(file_output_dir)

    dataset_dict_duke, crop_mask_duke = \
        build_sar_dataset_dict(dataset_path, dataset_fname_duke, [1, 40], 1, len(dataset_fname_duke)-1, file_output_dir,
                               dataset_format, 'duke', modality)

    dataset_dict_ella, crop_mask_ella = \
        build_sar_dataset_dict(dataset_path, dataset_fname_ella, [1, 40], 41, len(dataset_fname_ella), file_output_dir,
                               dataset_format, 'ella', modality)

    dataset_dict_louis, crop_mask_louis = \
        build_sar_dataset_dict(dataset_path, dataset_fname_louis, [1, 40], 81, len(dataset_fname_louis), file_output_dir,
                               dataset_format, 'louis', modality)

    dataset_dict_austinman, crop_mask_austinman = \
        build_sar_dataset_dict(dataset_path, dataset_fname_austinman, [1, 40], 121, len(dataset_fname_austinman), file_output_dir,
                               dataset_format, 'austinman', modality)

    dataset_dict_austinwoman, crop_mask_austinwoman = \
        build_sar_dataset_h5py(dataset_path, dataset_fname_austinwoman, [1, 40], 161, len(dataset_fname_austinwoman), file_output_dir,
                               dataset_format, 'austinwoman', modality)

    slice_crop_size_duke = np.zeros(6).astype(int)
    slice_crop_size_ella = np.zeros(6).astype(int)
    slice_crop_size_louis = np.zeros(6).astype(int)
    slice_crop_size_austinman = np.zeros(6).astype(int)
    slice_crop_size_austinwoman = np.zeros(6).astype(int)


    # for i in range(3):
    #     if crop_mask_duke.min_roi[i] >= crop_mask_ella.min_roi[i]:
    #         slice_crop_size_duke[2*i] = slice_crop_size_org[2*i] + crop_mask_duke.min_roi[i] - crop_mask_ella.min_roi[i]
    #         slice_crop_size_ella[2*i] = slice_crop_size_org[2*i]
    #     else:
    #         slice_crop_size_duke[2*i] = slice_crop_size_org[2*i]
    #         slice_crop_size_ella[2*i] = slice_crop_size_org[2*i] + crop_mask_ella.min_roi[i] - crop_mask_duke.min_roi[i]
    #
    #     if crop_mask_duke.max_roi[i] >= crop_mask_ella.max_roi[i]:
    #         slice_crop_size_duke[2*i+1] = slice_crop_size_org[2*i+1]
    #         slice_crop_size_ella[2*i+1] = slice_crop_size_org[2*i+1] + crop_mask_duke.max_roi[i] - crop_mask_ella.max_roi[i]
    #     else:
    #         slice_crop_size_duke[2*i+1] = slice_crop_size_org[2*i+1] + crop_mask_ella.max_roi[i] - crop_mask_duke.max_roi[i]
    #         slice_crop_size_ella[2*i+1] = slice_crop_size_org[2*i+1]

    # make the roi (with a designated margin) same size across head models
    for i in range(3):
        crop_mask_min = min([crop_mask_duke.min_roi[i], crop_mask_ella.min_roi[i], crop_mask_louis.min_roi[i],
                             crop_mask_austinman.min_roi[i], crop_mask_austinwoman.min_roi[i]])
        crop_mask_max = max([crop_mask_duke.max_roi[i], crop_mask_ella.max_roi[i], crop_mask_louis.max_roi[i],
                             crop_mask_austinman.max_roi[i], crop_mask_austinwoman.max_roi[i]])

        slice_crop_size_duke[2*i] = slice_crop_size_org[2*i] + crop_mask_duke.min_roi[i] - crop_mask_min
        slice_crop_size_ella[2*i] = slice_crop_size_org[2*i] + crop_mask_ella.min_roi[i] - crop_mask_min
        slice_crop_size_louis[2*i] = slice_crop_size_org[2*i] + crop_mask_louis.min_roi[i] - crop_mask_min
        slice_crop_size_austinman[2*i] = slice_crop_size_org[2*i] + crop_mask_austinman.min_roi[i] - crop_mask_min
        slice_crop_size_austinwoman[2*i] = slice_crop_size_org[2*i] + crop_mask_austinwoman.min_roi[i] - crop_mask_min

        slice_crop_size_duke[2*i+1] = slice_crop_size_org[2*i+1] + crop_mask_max - crop_mask_duke.max_roi[i]
        slice_crop_size_ella[2*i+1] = slice_crop_size_org[2*i+1] + crop_mask_max - crop_mask_ella.max_roi[i]
        slice_crop_size_louis[2*i+1] = slice_crop_size_org[2*i+1] + crop_mask_max - crop_mask_louis.max_roi[i]
        slice_crop_size_austinman[2*i+1] = slice_crop_size_org[2*i+1] + crop_mask_max - crop_mask_austinman.max_roi[i]
        slice_crop_size_austinwoman[2*i+1] = slice_crop_size_org[2*i+1] + crop_mask_max - crop_mask_austinwoman.max_roi[i]

    slice_crop_size_head_models = [slice_crop_size_duke, slice_crop_size_ella, slice_crop_size_louis,
                                   slice_crop_size_austinman, slice_crop_size_austinwoman]

    print(crop_mask_duke.min_roi)
    print(crop_mask_duke.max_roi)
    print(crop_mask_ella.min_roi)
    print(crop_mask_ella.max_roi)
    print(crop_mask_louis.min_roi)
    print(crop_mask_louis.max_roi)
    print(crop_mask_austinman.min_roi)
    print(crop_mask_austinman.max_roi)
    print(crop_mask_austinwoman.min_roi)
    print(crop_mask_austinwoman.max_roi)
    print(slice_crop_size_org)
    print(slice_crop_size_duke)
    print(slice_crop_size_ella)
    print(slice_crop_size_louis)
    print(slice_crop_size_austinman)
    print(slice_crop_size_austinwoman)

    # modality_idx = []
    # for m in modality:
    #     if m == 'B1_mag':
    #         modality_idx.append(0)
    #     if m == 'B1_real':
    #         modality_idx.append(1)
    #     if m == 'B1_imag':
    #         modality_idx.append(2)
    #     if m == 'Epsilon': # permittivity
    #         modality_idx.append(3)
    #     if m == 'Rho': # tissue denstity distribution
    #         modality_idx.append(4)
    #     if m == 'Sigma': # conductivity
    #         modality_idx.append(5)

    # load training images in train_patient_lst and setting roi
    # load multi-modal images in train_patient_lst and setting roi

    save_data = 0 # for debugging
    train_src_lst = []
    train_sar_lst = []
    train_sar_norm_lst = []

    #num_gen = 5 # number of generation in each mode

    if train_id_lst[0] is not None and (num_epochs > 0 or (num_epochs == 0 and is_trn_inference == 1)):
    #if train_id_lst[0] is not None and num_epochs != 0 and mode != '2':
    #if train_id_lst[0] is not None and mode != '2':
        for train_id in train_id_lst:
            train_src_modal_lst = []
            if int(train_id) <= 40:
                dataset_dict = dataset_dict_duke
                crop_mask = crop_mask_duke
                slice_crop_size = slice_crop_size_duke
            elif int(train_id) > 40 and int(train_id) <= 80:
                dataset_dict = dataset_dict_ella
                crop_mask = crop_mask_ella
                slice_crop_size = slice_crop_size_ella
            elif int(train_id) > 80 and int(train_id) <= 120:
                dataset_dict = dataset_dict_louis
                crop_mask = crop_mask_louis
                slice_crop_size = slice_crop_size_louis
            elif int(train_id) > 120 and int(train_id) <= 160:
                dataset_dict = dataset_dict_austinman
                crop_mask = crop_mask_austinman
                slice_crop_size = slice_crop_size_austinman
            elif int(train_id) > 160 and int(train_id) <= 200:
                dataset_dict = dataset_dict_austinwoman
                crop_mask = crop_mask_austinwoman
                slice_crop_size = slice_crop_size_austinwoman
            else:
                raise ValueError('training set id is out of range in given models')

            #mask_data_wo_bg_noise = remove_outliers(crop_mask.get_mask_data())

            for mod in modality:
                if mod in ['B1_mag', 'B1_real', 'B1_imag']:
                    train_src_name = '%s_Ex%s' % (mod, train_id)
                else:
                    train_src_name = '%s' % mod
                print('training data (source): %s ' % train_src_name)
                train_src_vol = dataset_dict[train_src_name]

                # # make background 0
                # bg_value = Counter(np.array(train_src_vol)).most_common()[0][0] # return most common value in voxels
                # train_src_vol[np.where(np.array(train_src_vol) == bg_value)[0]] = 0

                #train_src_vol_crop = crop_slice(train_src_vol, slice_crop_size)
                # crop using mask

                # apply bg-removed mask to input
                #train_src_vol = np.multiply(mask_data_wo_bg_noise, train_src_vol)
                train_src_vol = crop_mask.multiply(train_src_vol)

                train_src_vol_crop = crop_mask.crop_with_margin(train_src_vol, slice_crop_size)

                # normalization -> do this when building training samples
                #if preprocess_trn == 2 or preprocess_trn == 3 or preprocess_trn == 5:
                    #train_src_vol_crop = normalize_image(train_src_vol_crop, [0, 1])
                    #train_src_vol_crop = normalize_image(train_src_vol_crop, normalized_range) # tanh : -1~1
                    #train_src_vol_crop = np.divide(train_src_vol_crop, np.mean(train_src_vol_crop))

                if data_augment == 1:
                    #src_gen_lst = augment_data(train_src_vol_crop, num_gen, file_output_dir, train_src_name, save_data)
                    src_gen_lst = augment_data_v2(train_src_vol_crop, num_gen, file_output_dir, train_src_name, save_data)
                else:
                    src_gen_lst = [train_src_vol_crop]

                train_src_modal_lst.append(src_gen_lst)

            for i in range(len(src_gen_lst)):
                train_src_array = np.zeros((1, num_modality) + train_src_vol_crop.shape[0:3])
                for idx in range(num_modality):
                    train_src_array[0, idx] = train_src_modal_lst[idx][i]
                train_src_lst.append(train_src_array)

            train_sar_name = 'SAR_Ex%s' % (train_id)
            print('training data (target): %s ' % train_sar_name)
            train_sar_vol = dataset_dict[train_sar_name]

            # apply bg-removed mask to input
            #train_sar_vol = np.multiply(mask_data_wo_bg_noise, train_sar_vol)
            train_sar_vol = crop_mask.multiply(train_sar_vol)

            #train_sar_vol_crop = crop_slice(train_sar_vol, slice_crop_size)
            train_sar_vol_crop = crop_mask.crop_with_margin(train_sar_vol, slice_crop_size)

            # normalization -> do this when building training samples
            # if preprocess_trn == 2 or preprocess_trn == 3 or preprocess_trn == 5:
            #     train_sar_vol_crop = normalize_image(train_sar_vol_crop, normalized_range)  # tanh : -1~1
                #train_sar_vol_crop = np.divide(train_sar_vol_crop, np.mean(train_sar_vol_crop))

            if data_augment == 1:
                #sar_gen_lst = augment_data(train_sar_vol_crop, num_gen, file_output_dir, train_sar_name, save_data)
                sar_gen_lst = augment_data_v2(train_sar_vol_crop, num_gen, file_output_dir, train_sar_name, save_data)
            else:
                sar_gen_lst = [train_sar_vol_crop]

            for i in range(len(sar_gen_lst)):
                train_sar_array = np.zeros((1, 1) + train_sar_vol_crop.shape[0:3])
                train_sar_array[0, 0] = sar_gen_lst[i]
                train_sar_lst.append(train_sar_array)
            # train_sar_array[0, 0] = train_sar_vol_crop
            # train_sar_lst.append(train_sar_array)

    # load multi-modal images in test_patient_lst and setting roi
    test_src_lst = []
    test_sar_lst = []
    if test_id_lst[0] is not None:
        for test_id in test_id_lst:
            if int(test_id) <= 40:
                dataset_dict = dataset_dict_duke
                crop_mask = crop_mask_duke
                slice_crop_size = slice_crop_size_duke
            elif int(test_id) > 40 and int(test_id) <= 80:
                dataset_dict = dataset_dict_ella
                crop_mask = crop_mask_ella
                slice_crop_size = slice_crop_size_ella
            elif int(test_id) > 80 and int(test_id) <= 120:
                dataset_dict = dataset_dict_louis
                crop_mask = crop_mask_louis
                slice_crop_size = slice_crop_size_louis
            elif int(test_id) > 120 and int(test_id) <= 160:
                dataset_dict = dataset_dict_austinman
                crop_mask = crop_mask_austinman
                slice_crop_size = slice_crop_size_austinman
            elif int(train_id) > 160 and int(train_id) <= 200:
                dataset_dict = dataset_dict_austinwoman
                crop_mask = crop_mask_austinwoman
                slice_crop_size = slice_crop_size_austinwoman
            else:
                raise ValueError('test set id is out of range in given models')

            #mask_data_wo_bg_noise = remove_outliers(crop_mask.get_mask_data())

            # load test images in test_patient_lst and setting roi
            test_src_modal_lst = []
            for mod in modality:
                if mod in ['B1_mag', 'B1_real', 'B1_imag']:
                    test_src_name = '%s_Ex%s' % (mod, test_id)
                else:
                    test_src_name = '%s' % mod
                print('test data (source): %s ' % test_src_name)
                test_src_vol = dataset_dict[test_src_name]

                # apply bg-removed mask to input
                #test_src_vol = np.multiply(mask_data_wo_bg_noise, test_src_vol)
                test_src_vol = crop_mask.multiply(test_src_vol)

                #test_src_vol_crop = crop_slice(test_src_vol, slice_crop_size)
                test_src_vol_crop = crop_mask.crop_with_margin(test_src_vol, slice_crop_size)

                # normalization -> do this when building test samples
                #if preprocess_trn == 2 or preprocess_trn == 3 or preprocess_trn == 5:
                    #test_src_vol_crop = normalize_image(test_src_vol_crop, [0, 1])
                    #test_src_vol_crop = normalize_image(test_src_vol_crop, normalized_range) # tanh : -1~1
                    #test_src_vol_crop = np.divide(test_src_vol_crop, np.mean(test_src_vol_crop))
                test_src_modal_lst.append(test_src_vol_crop)

            test_src_array = np.zeros((1, num_modality) + test_src_vol_crop.shape[0:3])
            for idx in range(num_modality):
                test_src_array[0, idx] = test_src_modal_lst[idx]
            test_src_lst.append(test_src_array)

            test_tar_name = 'SAR_Ex%s' % (test_id)
            print('test data (target): %s ' % test_tar_name)
            test_sar_vol = dataset_dict[test_tar_name]

            # apply bg-removed mask to input
            #test_sar_vol = np.multiply(mask_data_wo_bg_noise, test_sar_vol)
            test_sar_vol = crop_mask.multiply(test_sar_vol)

            #test_sar_vol_crop = crop_slice(test_sar_vol, slice_crop_size)
            test_sar_vol_crop = crop_mask.crop_with_margin(test_sar_vol, slice_crop_size)

            # normalization -> do this when saving data or measuring performance
            #if preprocess_trn == 2 or preprocess_trn == 3 or preprocess_trn == 5:
                #test_sar_vol_crop = normalize_image(test_sar_vol_crop, normalized_range)  # tanh : -1~1
                #test_sar_vol_crop = np.divide(test_sar_vol_crop, np.mean(test_sar_vol_crop))

            test_sar_array = np.zeros((1, 1) + test_sar_vol_crop.shape[0:3])
            test_sar_array[0, 0] = test_sar_vol_crop
            test_sar_lst.append(test_sar_array)

    return train_src_lst, train_sar_lst, test_src_lst, test_sar_lst, slice_crop_size_head_models


# augment 3D volume
def augment_data(vol, num_gen, file_output_dir, fname, save_data):

    vol_w_shift_lst = data_generator(vol, num_gen, [0, 1, 2], file_output_dir, fname + '_w_shift', 'w_shift', save_data)
    vol_h_shift_lst = data_generator(vol, num_gen, [0, 1, 2], file_output_dir, fname + '_h_shift', 'h_shift', save_data)
    vol_z_shift_lst = data_generator(vol, num_gen, [0, 2, 1], file_output_dir, fname + '_z_shift', 'w_shift', save_data)

    vol_w_flip = data_generator(vol, num_gen, [0, 1, 2], file_output_dir, fname + '_w_flip', 'v_flip', save_data)
    vol_w_flip_w_shift_lst = data_generator(vol_w_flip[0], num_gen, [0, 1, 2], file_output_dir, fname +
                                            '_w_flip_w_shift', 'w_shift', save_data)
    vol_w_flip_h_shift_lst = data_generator(vol_w_flip[0], num_gen, [0, 1, 2], file_output_dir, fname +
                                            '_w_flip_h_shift', 'h_shift', save_data)
    vol_w_flip_z_shift_lst = data_generator(vol_w_flip[0], num_gen, [0, 2, 1], file_output_dir, fname +
                                            '_w_flip_z_shift', 'w_shift', save_data)

    vol_h_flip = data_generator(vol, num_gen, [0, 1, 2], file_output_dir, fname + '_h_flip', 'h_flip', save_data)
    vol_h_flip_w_shift_lst = data_generator(vol_h_flip[0], num_gen, [0, 1, 2], file_output_dir, fname +
                                            '_h_flip_w_shift', 'w_shift', save_data)
    vol_h_flip_h_shift_lst = data_generator(vol_h_flip[0], num_gen, [0, 1, 2], file_output_dir, fname +
                                            '_h_flip_h_shift', 'h_shift', save_data)
    vol_h_flip_z_shift_lst = data_generator(vol_h_flip[0], num_gen, [0, 2, 1], file_output_dir, fname +
                                            '_h_flip_z_shift', 'w_shift', save_data)

    vol_z_flip = data_generator(vol, num_gen, [0, 2, 1], file_output_dir, fname + '_z_flip', 'h_flip', save_data)
    vol_z_flip_w_shift_lst = data_generator(vol_z_flip[0], num_gen, [0, 1, 2], file_output_dir, fname +
                                            '_z_flip_w_shift', 'w_shift', save_data)
    vol_z_flip_h_shift_lst = data_generator(vol_z_flip[0], num_gen, [0, 1, 2], file_output_dir, fname +
                                            '_z_flip_h_shift', 'h_shift', save_data)
    vol_z_flip_z_shift_lst = data_generator(vol_z_flip[0], num_gen, [0, 2, 1], file_output_dir, fname +
                                            '_z_flip_z_shift', 'w_shift', save_data)

    vol_gen_lst = [vol] + vol_w_shift_lst + vol_h_shift_lst + vol_z_shift_lst + \
                  [vol_w_flip[0]] + vol_w_flip_w_shift_lst + vol_w_flip_h_shift_lst + vol_w_flip_z_shift_lst + \
                  [vol_h_flip[0]] + vol_h_flip_w_shift_lst + vol_h_flip_h_shift_lst + vol_h_flip_z_shift_lst + \
                  [vol_z_flip[0]] + vol_z_flip_w_shift_lst + vol_z_flip_h_shift_lst + vol_z_flip_z_shift_lst

    return vol_gen_lst


def augment_data_v2(vol, num_gen, file_output_dir, fname, save_data):

    vol_w_shift_lst = data_generator(vol, num_gen, [0, 1, 2], file_output_dir, fname + '_w_shift', 'w_shift', save_data)
    vol_h_shift_lst = data_generator(vol, num_gen, [0, 1, 2], file_output_dir, fname + '_h_shift', 'h_shift', save_data)
    vol_z_shift_lst = data_generator(vol, num_gen, [0, 2, 1], file_output_dir, fname + '_z_shift', 'w_shift', save_data)
    vol_scaling_lst = data_generator(vol, num_gen, [0, 1, 2], file_output_dir, fname + '_scaling', 'scaling', save_data)

    vol_w_flip = data_generator(vol, num_gen, [0, 1, 2], file_output_dir, fname + '_w_flip', 'v_flip', save_data)
    vol_w_flip_w_shift_lst = data_generator(vol_w_flip[0], num_gen, [0, 1, 2], file_output_dir, fname +
                                            '_w_flip_w_shift', 'w_shift', save_data)
    vol_w_flip_h_shift_lst = data_generator(vol_w_flip[0], num_gen, [0, 1, 2], file_output_dir, fname +
                                            '_w_flip_h_shift', 'h_shift', save_data)
    vol_w_flip_z_shift_lst = data_generator(vol_w_flip[0], num_gen, [0, 2, 1], file_output_dir, fname +
                                            '_w_flip_z_shift', 'w_shift', save_data)
    vol_w_flip_scaling_lst = data_generator(vol_w_flip[0], num_gen, [0, 1, 2], file_output_dir, fname +
                                            '_w_flip_scaling', 'scaling', save_data)

    vol_h_flip = data_generator(vol, num_gen, [0, 1, 2], file_output_dir, fname + '_h_flip', 'h_flip', save_data)
    vol_h_flip_w_shift_lst = data_generator(vol_h_flip[0], num_gen, [0, 1, 2], file_output_dir, fname +
                                            '_h_flip_w_shift', 'w_shift', save_data)
    vol_h_flip_h_shift_lst = data_generator(vol_h_flip[0], num_gen, [0, 1, 2], file_output_dir, fname +
                                            '_h_flip_h_shift', 'h_shift', save_data)
    vol_h_flip_z_shift_lst = data_generator(vol_h_flip[0], num_gen, [0, 2, 1], file_output_dir, fname +
                                            '_h_flip_z_shift', 'w_shift', save_data)
    vol_h_flip_scaling_lst = data_generator(vol_h_flip[0], num_gen, [0, 1, 2], file_output_dir, fname +
                                            '_h_flip_scaling', 'scaling', save_data)

    vol_z_flip = data_generator(vol, num_gen, [0, 2, 1], file_output_dir, fname + '_z_flip', 'h_flip', save_data)
    vol_z_flip_w_shift_lst = data_generator(vol_z_flip[0], num_gen, [0, 1, 2], file_output_dir, fname +
                                            '_z_flip_w_shift', 'w_shift', save_data)
    vol_z_flip_h_shift_lst = data_generator(vol_z_flip[0], num_gen, [0, 1, 2], file_output_dir, fname +
                                            '_z_flip_h_shift', 'h_shift', save_data)
    vol_z_flip_z_shift_lst = data_generator(vol_z_flip[0], num_gen, [0, 2, 1], file_output_dir, fname +
                                            '_z_flip_z_shift', 'w_shift', save_data)
    vol_z_flip_scaling_lst = data_generator(vol_z_flip[0], num_gen, [0, 1, 2], file_output_dir, fname +
                                            '_z_flip_scaling', 'scaling', save_data)

    vol_gen_lst = [vol] + vol_w_shift_lst + vol_h_shift_lst + vol_z_shift_lst + vol_scaling_lst + \
                  [vol_w_flip[0]] + vol_w_flip_w_shift_lst + vol_w_flip_h_shift_lst + vol_w_flip_z_shift_lst + vol_w_flip_scaling_lst + \
                  [vol_h_flip[0]] + vol_h_flip_w_shift_lst + vol_h_flip_h_shift_lst + vol_h_flip_z_shift_lst + vol_h_flip_scaling_lst + \
                  [vol_z_flip[0]] + vol_z_flip_w_shift_lst + vol_z_flip_h_shift_lst + vol_z_flip_z_shift_lst + vol_z_flip_scaling_lst

    return vol_gen_lst


def data_generator(data, num_gen, org_dim_idx, file_output_dir, file_name, mode, save_data):
    # data augmentation
    if mode == 'w_shift':
        datagen = ImageDataGenerator(width_shift_range=0.7, fill_mode='constant', cval=0)
    elif mode == 'h_shift':
        datagen = ImageDataGenerator(height_shift_range=0.7, fill_mode='constant', cval=0)
    elif mode == 'h_flip':
        datagen = ImageDataGenerator(horizontal_flip=True, fill_mode='constant', cval=0)
    elif mode == 'v_flip':
        datagen = ImageDataGenerator(vertical_flip=True, fill_mode='constant', cval=0)
    elif mode == 'scaling':
        datagen = ImageDataGenerator(zoom_range=[1.3, 0.5], fill_mode='constant', cval=0)
    else:
        raise NotImplementedError('Unknown data augmentation mode')

    data_trans = np.transpose(data, org_dim_idx)
    it = datagen.flow(expand_dims(data_trans, 0), seed=2, shuffle=False, batch_size=1)

    image_gen_lst = []
    for i in range(num_gen):
        batch = it.next()
        image_gen = batch[0]
        image_gen_trans = np.transpose(image_gen, org_dim_idx)
        image_gen_lst.append(image_gen_trans)
        if save_data == 1:
            src_filename = os.path.join(file_output_dir, '%s_%s.%s' % (file_name, i, 'nii.gz'))
            __save_sar_volume(image_gen_trans, src_filename, 'nii.gz', 'float32', is_compressed=False)

    return image_gen_lst


def unique_array(l):
    uniq = []
    dupl = []
    for i in l:
        if not i in uniq:
            uniq.append(i)
        else:
            dupl.append(i)

    return uniq, dupl


def crop_slice(vol, slice_crop_size):
    # center_pos = [round(vol.shape[0]/2), round(vol.shape[1]/2),
    #               round(vol.shape[2]/2)]
    idx = []
    dim = 0
    for vol_size in vol.shape:
        if slice_crop_size[2*dim] < vol_size - slice_crop_size[2*dim+1]:
            idx.append([slice_crop_size[2*dim], (vol_size - 1) - slice_crop_size[2*dim+1]])
        else:
            raise ValueError('start index should less than end index')
        dim += 1
    vol_crop = vol[idx[0][0]:idx[0][1]+1, idx[1][0]:idx[1][1]+1, idx[2][0]:idx[2][1]+1]

    return vol_crop


def save_volume_MICCAI2012(gen_conf, train_conf, volume, case_idx) :
    dataset = train_conf['dataset']
    approach = train_conf['approach']
    extraction_step = train_conf['extraction_step']
    root_path = gen_conf['root_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = root_path + gen_conf['dataset_path']
    results_path = root_path + gen_conf['results_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    folder_names = dataset_info['folder_names']

    data_filename = dataset_path + path + pattern[1].format(folder_names[3], case_idx)
    image_data = read_volume_data(data_filename)

    volume = np.multiply(volume, image_data.get_data() != 0)

    out_filename = results_path + path + pattern[2].format(folder_names[3], str(case_idx), approach +
                                                                       ' - ' + str(extraction_step))

    __save_volume(volume, image_data, out_filename, dataset_info['format'], is_compressed = True)


def save_volume_3T7T(gen_conf, train_conf, test_conf, volume, filename_ext, case_idx) :
    dataset = train_conf['dataset']
    approach = train_conf['approach']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    root_path = gen_conf['root_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = root_path + gen_conf['dataset_path']
    results_path = root_path + gen_conf['results_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    folder_names = dataset_info['folder_names']
    data_augment = train_conf['data_augment']
    preprocess_trn = train_conf['preprocess']
    preprocess_tst = test_conf['preprocess']

    print(volume.shape)
    volume_tmp = np.zeros(volume.shape + (1, ))
    volume_tmp[:, :, :, 0] = volume
    volume = volume_tmp
    print(volume.shape)

    # it should be 3T test image (not 7T label)
    #data_filename = dataset_path + path + pattern[0].format(folder_names[0], case_idx)
    file_dir = dataset_path + path + folder_names[0]
    data_filename = file_dir + '/' + filename_ext
    print(data_filename)

    file_output_dir = results_path + path + folder_names
    if not os.path.exists(file_output_dir):
        os.makedirs(os.path.join(results_path, path, folder_names))

    image_data = read_volume_data(data_filename)
    org_image_volume = image_data.get_data()
    print(org_image_volume.shape)
    min_vox_value = np.min(org_image_volume)
    print(min_vox_value)

    if np.size(volume.shape) != np.size(org_image_volume.shape):
        org_image_volume_tmp = np.zeros(org_image_volume.shape + (1,))
        org_image_volume_tmp[:, :, :, 0] = org_image_volume
        org_image_volume = org_image_volume_tmp

    volume = np.multiply(volume, org_image_volume != min_vox_value)
    volume[org_image_volume != min_vox_value] = volume[org_image_volume != min_vox_value] + 1

    if data_augment == 1:
        data_augment_label = '_mixup'
    elif data_augment == 2:
        data_augment_label = '_datagen'
    elif data_augment == 3:
        data_augment_label = '_mixup+datagen'
    else:
        data_augment_label = ''

    out_filename = results_path + path + pattern[1].format(folder_names, str(case_idx) + '_' + approach + '_' +
                                                           str(patch_shape) + '_' + str(extraction_step) +
                                                           data_augment_label + '_preproc_trn_opt_' +
                                                           str(preprocess_trn) + '_preproc_tst_opt_' +
                                                           str(preprocess_tst) +'_before_label_mapping')

    # out_filename = results_path + pattern[1].format(path.replace('/',''), str(case_idx) + '_' + approach + '_' +
    #                                                 str(extraction_step) + '_' + 'before_label_mapping')
    print(out_filename)
    __save_volume(volume, image_data, out_filename, dataset_info['format'], is_compressed = True)
    #label_mapper = {0 : 0, 1 : 10, 2 : 150, 3 : 250}
    label_mapper = {0: 0, 1: 0, 2: 10, 3: 150, 4: 250}
    for key in label_mapper.keys() :
        volume[volume == key] = label_mapper[key]

    out_filename = results_path + path + pattern[1].format(folder_names, str(case_idx) + '_' + approach + '_' +
                                                           str(patch_shape) + '_' + str(extraction_step) +
                                                           data_augment_label + '_preproc_trn_opt_' +
                                                           str(preprocess_trn) + '_preproc_tst_opt_' +
                                                           str(preprocess_tst))

    print(out_filename)
    __save_volume(volume, image_data, out_filename, dataset_info['format'], is_compressed = True)


def save_intermediate_volume(gen_conf, train_conf, volume, case_idx, filename_ext, label) :
    dataset = train_conf['dataset']
    root_path = gen_conf['root_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = root_path + gen_conf['dataset_path']
    results_path = root_path + gen_conf['results_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    folder_names = dataset_info['folder_names']

    print(volume.shape)
    volume_tmp = np.zeros(volume.shape + (1, ))
    volume_tmp[:, :, :, 0] = volume
    volume = volume_tmp
    print(volume.shape)

    # it should be 3T test image (not 7T label)
    if not filename_ext:
        data_filename = dataset_path + path + pattern[0].format(folder_names, case_idx)
    else:
        file_dir = dataset_path + path + folder_names
        file_path = file_dir + '/' + filename_ext
        data_filename = file_path

    print(data_filename)
    image_data = read_volume_data(data_filename)

    if not filename_ext:
        if not os.path.exists(results_path + path + folder_names):
            os.makedirs(os.path.join(results_path, path, folder_names))
        out_filename = results_path + path + pattern[1].format(folder_names, str(case_idx) + '_' + label)
    else:
        if not os.path.exists(results_path + path + folder_names[1]):
            os.makedirs(os.path.join(results_path, path, folder_names[1]))
        file_output_dir = results_path + path + folder_names[1]
        file_name, ext = os.path.splitext(filename_ext)
        out_filename = file_output_dir + '/' + file_name + '_' + label + ext

    print(out_filename)
    __save_volume(volume, image_data, out_filename, dataset_info['format'], is_compressed = False)


def save_volume_ADNI(gen_conf, train_conf, test_conf, volume, filename_ext) :
    approach = train_conf['approach']
    dataset = test_conf['dataset']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    root_path = gen_conf['root_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = root_path + gen_conf['dataset_path']
    results_path = root_path + gen_conf['results_path']
    path = dataset_info['path']
    folder_names = dataset_info['folder_names']
    data_augment = train_conf['data_augment']
    preprocess_trn = train_conf['preprocess']
    preprocess_tst = test_conf['preprocess']

    print(volume.shape)
    volume_tmp = np.zeros(volume.shape + (1, ))
    volume_tmp[:, :, :, 0] = volume
    volume = volume_tmp
    print(volume.shape)

    # it should be 3T test image (not 7T label)
    file_dir = dataset_path + path + folder_names[0]
    file_path = file_dir + '/' + filename_ext
    file_output_dir = results_path + path + folder_names[1]
    if not os.path.exists(file_output_dir):
        os.makedirs(os.path.join(results_path, path, folder_names[1]))

    file_name, ext = os.path.splitext(filename_ext)

    image_data = read_volume_data(file_path)
    min_vox_value = np.min(image_data.get_data())
    print(min_vox_value)
    volume = np.multiply(volume, image_data.get_data() != min_vox_value)
    volume[image_data.get_data() != min_vox_value] = volume[image_data.get_data() != min_vox_value] + 1

    if data_augment == 1:
        data_augment_label = '_mixup'
    elif data_augment == 2:
        data_augment_label = '_datagen'
    elif data_augment == 3:
        data_augment_label = '_mixup+datagen'
    else:
        data_augment_label = ''

    out_filename = file_output_dir + '/' + file_name + '_' + approach + '_' + str(patch_shape) + '_' + \
                   str(extraction_step) + data_augment_label + '_preproc_trn_opt_' + str(preprocess_trn) + \
                   '_preproc_tst_opt_' + str(preprocess_tst) + '_before_label_mapping' + ext
    print(out_filename)
    __save_volume(volume, image_data, out_filename, dataset_info['format'], is_compressed = True)

    label_mapper = {0: 0, 1: 0, 2: 10, 3: 150, 4: 250}
    for key in label_mapper.keys() :
        volume[volume == key] = label_mapper[key]

    out_filename = file_output_dir + '/' + file_name + '_' + approach + '_' + str(patch_shape) + '_' + \
                   str(extraction_step) + data_augment_label + '_preproc_trn_opt_' + str(preprocess_trn) + \
                   '_preproc_tst_opt_' + str(preprocess_tst) + ext
    print(out_filename)
    __save_volume(volume, image_data, out_filename, dataset_info['format'], is_compressed = True)


def save_volume_cbct(gen_conf, train_conf, test_conf, volume, filename_ext, case_idx):
    dataset = train_conf['dataset']
    approach = train_conf['approach']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    root_path = gen_conf['root_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = root_path + gen_conf['dataset_path']
    results_path = root_path + gen_conf['results_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    folder_names = dataset_info['folder_names']
    data_augment = train_conf['data_augment']
    preprocess_trn = train_conf['preprocess']
    preprocess_tst = test_conf['preprocess']

    print(volume.shape)
    volume_tmp = np.zeros(volume.shape + (1,))
    volume_tmp[:, :, :, 0] = volume
    volume = volume_tmp
    print(volume.shape)

    # it should be 3T test image (not 7T label)
    #data_filename = dataset_path + path + pattern[0].format(folder_names[0], case_idx)
    file_dir = dataset_path + path + folder_names[0]
    data_filename = file_dir + '/' + filename_ext
    print(data_filename)

    file_output_dir = results_path + path + folder_names[0]
    if not os.path.exists(file_output_dir):
        os.makedirs(os.path.join(results_path, path, folder_names[0]))

    image_data = read_volume_data(data_filename)
    org_image_volume = image_data.get_data()
    print(org_image_volume.shape)
    min_vox_value = np.min(org_image_volume)
    print(min_vox_value)

    if np.size(volume.shape) != np.size(org_image_volume.shape):
        org_image_volume_tmp = np.zeros(org_image_volume.shape + (1,))
        org_image_volume_tmp[:, :, :, 0] = org_image_volume
        org_image_volume = org_image_volume_tmp

    volume = np.multiply(volume, org_image_volume != min_vox_value)
    volume[org_image_volume != min_vox_value] = volume[org_image_volume != min_vox_value] + 1

    if data_augment == 1:
        data_augment_label = '_mixup'
    elif data_augment == 2:
        data_augment_label = '_datagen'
    elif data_augment == 3:
        data_augment_label = '_mixup+datagen'
    else:
        data_augment_label = ''

    out_filename = results_path + path + pattern[3].format(folder_names[0], str(case_idx) + '_' + approach + '_' +
                                                           str(patch_shape) + '_' + str(extraction_step) +
                                                           data_augment_label + '_preproc_trn_opt_' +
                                                           str(preprocess_trn) + '_preproc_tst_opt_' +
                                                           str(preprocess_tst) + '_before_label_mapping')

    print(out_filename)
    __save_volume(volume, image_data, out_filename, dataset_info['format'], is_compressed=True)
    label_mapper = {0: 0, 1: 0, 2: 150, 3: 250}
    for key in label_mapper.keys():
        volume[volume == key] = label_mapper[key]

    out_filename = results_path + path + pattern[3].format(folder_names[0], str(case_idx) + '_' + approach + '_' +
                                                           str(patch_shape) + '_' + str(extraction_step) +
                                                           data_augment_label + '_preproc_trn_opt_' +
                                                           str(preprocess_trn) + '_preproc_tst_opt_' +
                                                           str(preprocess_tst))

    print(out_filename)
    __save_volume(volume, image_data, out_filename, dataset_info['format'], is_compressed=True)


def save_volume_tha(gen_conf, train_conf, test_conf, volume, prob_volume, test_fname, test_patient_id, seg_label,
                    file_output_dir):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    num_classes = gen_conf['num_classes']
    path = dataset_info['path']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    threshold = test_conf['threshold']

    approach = train_conf['approach']
    loss = train_conf['loss']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    data_augment = train_conf['data_augment']
    preprocess_trn = train_conf['preprocess']
    preprocess_tst = test_conf['preprocess']
    dimension = test_conf['dimension']
    file_format = dataset_info['format']

    modality = dataset_info['image_modality']
    modality_idx = []
    for m in modality: # the order of modality images should be fixed due to the absolute path
        if m == 'T1':
            modality_idx.append(0)
        if m == 'B0':
            modality_idx.append(1)
        if m == 'FA':
            modality_idx.append(2)

    file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
    seg_output_dir = os.path.join(file_test_patient_dir, path)
    if not os.path.exists(seg_output_dir):
        os.makedirs(seg_output_dir)
    prob_map_output_dir = os.path.join(seg_output_dir, 'prob_map_crop')
    if not os.path.exists(prob_map_output_dir):
        os.makedirs(prob_map_output_dir)
    non_smoothed_output_dir = os.path.join(seg_output_dir, 'non_smoothed')
    if not os.path.exists(non_smoothed_output_dir):
        os.makedirs(non_smoothed_output_dir)
    non_smoothed_crop_output_dir = os.path.join(seg_output_dir, 'non_smoothed', 'crop')
    if not os.path.exists(non_smoothed_crop_output_dir):
        os.makedirs(non_smoothed_crop_output_dir)
    smoothed_output_dir = os.path.join(seg_output_dir, 'smoothed')
    if not os.path.exists(smoothed_output_dir):
        os.makedirs(smoothed_output_dir)
    smoothed_norm_output_dir = os.path.join(seg_output_dir, 'smoothed_norm')
    if not os.path.exists(smoothed_norm_output_dir):
        os.makedirs(smoothed_norm_output_dir)
    smoothed_norm_thres_output_dir = os.path.join(seg_output_dir, 'smoothed_norm_thres')
    if not os.path.exists(smoothed_norm_thres_output_dir):
        os.makedirs(smoothed_norm_thres_output_dir)
    smoothed_norm_thres_crop_output_dir = os.path.join(seg_output_dir, 'smoothed_norm_thres', 'crop')
    if not os.path.exists(smoothed_norm_thres_crop_output_dir):
        os.makedirs(smoothed_norm_thres_crop_output_dir)

    is_stl_out = False

    if is_stl_out:
        stl_output_dir = os.path.join(seg_output_dir, 'stl_out')
        if not os.path.exists(stl_output_dir):
            os.makedirs(os.path.join(seg_output_dir, 'stl_out'))

    crop_test_img_path = os.path.join(file_test_patient_dir, crop_tst_img_pattern[modality_idx[0]])
    crop_test_data = read_volume_data(crop_test_img_path)

    test_roi_mask_file = os.path.join(file_test_patient_dir, test_roi_mask_pattern)
    test_crop_mask = find_crop_mask(test_roi_mask_file)
    image_data = read_volume_data(test_fname[0])

    #threshold for overlaped patch images (prob. vol) within the segmented volume
    prob_vol_norm = np.zeros(prob_volume.shape)
    for i in range(num_classes):
        prob_vol_norm[:, :, :, i] = normalize_image(prob_volume[:, :, :, i], [0, 1])
    volume = np.multiply(volume, prob_vol_norm[:, :, :, num_classes-1])
    volume_thr = volume > threshold

    # save probability map for background and foreground
    idx = 0
    class_name_lst = ['bg', 'fg']
    for class_name in class_name_lst:
        prob_map_crop_filename = class_name + '_' + seg_label + '_prob_map_crop_' + approach + '_' + loss + '_.' + \
                                 file_format
        prob_map_crop_out_filepath = os.path.join(prob_map_output_dir, prob_map_crop_filename)
        print(prob_map_crop_out_filepath)
        __save_volume(prob_vol_norm[:, :, :, idx], crop_test_data, prob_map_crop_out_filepath, file_format,
                      is_compressed=False)
        idx += 1

    # split side (in LPI)
    left_mask, right_mask, volume_thr = compute_side_mask(volume_thr, crop_test_data, is_check_vol_diff=True)

    for mask, side in zip([left_mask, right_mask], ['left', 'right']):
        if volume_thr is None:
            failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
            with open(failed_cases_filepath, 'a') as f:
                f.write(test_patient_id + '_' + seg_label + '_' + side + '\n')
            continue

        # split the volume into left/right side
        vol = np.multiply(mask, volume_thr)

        # postprocessing
        vol_refined = postprocess(vol)
        if vol_refined is None:
            failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
            with open(failed_cases_filepath, 'a') as f:
                f.write(test_patient_id + '_' + seg_label + '_' + side + '\n')
            continue
        # save the cropped/refined result
        nii_crop_filename = side + '_' + seg_label + '_seg_crop_' + approach + '_' + loss + '.' + file_format
        nii_crop_out_filepath = os.path.join(non_smoothed_crop_output_dir, nii_crop_filename)
        print(nii_crop_out_filepath)
        __save_volume(vol_refined, crop_test_data, nii_crop_out_filepath, file_format, is_compressed=True)

        # uncrop segmentation only (left and right)
        vol_uncrop = test_crop_mask.uncrop(vol_refined)

        # save the uncropped result
        nii_uncrop_filename = side + '_' + seg_label + '_seg_' + approach + '_' + loss + '.' + file_format
        nii_uncrop_out_filepath = os.path.join(non_smoothed_output_dir, nii_uncrop_filename)
        print(nii_uncrop_out_filepath)
        __save_volume(vol_uncrop, image_data, nii_uncrop_out_filepath, file_format, is_compressed = True)

        # smoothing (before surface extraction)
        nii_smooth_filename = side + '_' + seg_label + '_seg_smooth_' + approach + '_' + loss + '.' + file_format
        nii_smooth_out_filepath = os.path.join(smoothed_output_dir, nii_smooth_filename)
        print(nii_smooth_out_filepath)
        __smooth_binary_img(nii_uncrop_out_filepath, nii_smooth_out_filepath, dim=dimension, maximumRMSError = 0.01,
                            numberOfIterations = 10, numberOfLayers = 3)

        # normalization
        nii_smooth_image = BrainImage(nii_smooth_out_filepath, None)
        vol_smooth_norm = nii_smooth_image.nii_data_normalized(bits=0)
        nii_smooth_norm_filename = side + '_' + seg_label + '_seg_smooth_norm_' + approach + '_' + loss + '.' + \
                                   file_format
        nii_smooth_norm_out_filepath = os.path.join(smoothed_norm_output_dir, nii_smooth_norm_filename)
        print(nii_smooth_norm_out_filepath)
        __save_volume(vol_smooth_norm, image_data, nii_smooth_norm_out_filepath, file_format, is_compressed=False)

        # threshold
        vol_smooth_norm_thres = vol_smooth_norm > 0.4
        nii_smooth_norm_thres_filename = side + '_' + seg_label + '_seg_smooth_norm_thres_' + approach + '_' + loss + '.' + \
                                         file_format
        nii_smooth_norm_thres_out_filepath = os.path.join(smoothed_norm_thres_output_dir, nii_smooth_norm_thres_filename)
        print(nii_smooth_norm_thres_out_filepath)
        __save_volume(vol_smooth_norm_thres, image_data, nii_smooth_norm_thres_out_filepath, file_format,
                      is_compressed=True)

        # crop threhold image for measure
        nii_smooth_norm_thres_crop_filename = side + '_' + seg_label + '_seg_smooth_norm_thres_crop_' + approach + '_' + \
                                              loss + '.' +  file_format
        nii_smooth_norm_thres_crop_out_filepath = os.path.join(smoothed_norm_thres_crop_output_dir,
                                                               nii_smooth_norm_thres_crop_filename)
        print(nii_smooth_norm_thres_crop_out_filepath)
        crop_image(nii_smooth_norm_thres_out_filepath, test_crop_mask, nii_smooth_norm_thres_crop_out_filepath)

        if is_stl_out:
            # save stl
            stl_filename = side + '_' + seg_label + '_seg' + '.stl'
            stl_out_filepath= os.path.join(stl_output_dir, stl_filename)
            print(stl_out_filepath)
            __create_stl(nii_uncrop_out_filepath, stl_out_filepath)

            # smooth stl
            stl_smooth_filename = side + '_' + seg_label + '_seg_smooth.stl'
            stl_smooth_out_filepath= os.path.join(stl_output_dir, stl_smooth_filename)
            print(stl_smooth_out_filepath)
            __smooth_stl(stl_out_filepath, stl_smooth_out_filepath)


def save_volume_tha_unseen(gen_conf, train_conf, test_conf, volume, prob_volume, test_fname, test_patient_id, seg_label,
                    file_output_dir, is_low_res):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    num_classes = gen_conf['num_classes']
    path = dataset_info['path']
    test_img_name_pattern = dataset_info['image_name_pattern']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    threshold = test_conf['threshold']
    approach = train_conf['approach']
    loss = train_conf['loss']
    data_augment = train_conf['data_augment']
    dimension = test_conf['dimension']
    file_format = dataset_info['format']

    modality = dataset_info['image_modality']
    modality_idx = []
    for m in modality: # the order of modality images should be fixed due to the absolute path
        if m == 'T1':
            modality_idx.append(0)
        if m == 'B0':
            modality_idx.append(1)
        if m == 'FA':
            modality_idx.append(2)

    file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
    seg_output_dir = os.path.join(file_test_patient_dir, path)
    if not os.path.exists(seg_output_dir):
        os.makedirs(seg_output_dir)
    prob_map_output_dir = os.path.join(seg_output_dir, 'prob_map_crop')
    if not os.path.exists(prob_map_output_dir):
        os.makedirs(prob_map_output_dir)
    non_smoothed_output_dir = os.path.join(seg_output_dir, 'non_smoothed')
    if not os.path.exists(non_smoothed_output_dir):
        os.makedirs(non_smoothed_output_dir)
    non_smoothed_crop_output_dir = os.path.join(seg_output_dir, 'non_smoothed', 'crop')
    if not os.path.exists(non_smoothed_crop_output_dir):
        os.makedirs(non_smoothed_crop_output_dir)
    smoothed_output_dir = os.path.join(seg_output_dir, 'smoothed')
    if not os.path.exists(smoothed_output_dir):
        os.makedirs(smoothed_output_dir)
    smoothed_norm_output_dir = os.path.join(seg_output_dir, 'smoothed_norm')
    if not os.path.exists(smoothed_norm_output_dir):
        os.makedirs(smoothed_norm_output_dir)
    smoothed_norm_thres_output_dir = os.path.join(seg_output_dir, 'smoothed_norm_thres')
    if not os.path.exists(smoothed_norm_thres_output_dir):
        os.makedirs(smoothed_norm_thres_output_dir)
    smoothed_norm_thres_crop_output_dir = os.path.join(seg_output_dir, 'smoothed_norm_thres', 'crop')
    if not os.path.exists(smoothed_norm_thres_crop_output_dir):
        os.makedirs(smoothed_norm_thres_crop_output_dir)

    is_stl_out = False

    if is_stl_out:
        stl_output_dir = os.path.join(seg_output_dir, 'stl_out')
        if not os.path.exists(stl_output_dir):
            os.makedirs(os.path.join(seg_output_dir, 'stl_out'))

    crop_test_img_path = os.path.join(file_test_patient_dir, crop_tst_img_pattern[modality_idx[0]])
    crop_test_data = read_volume_data(crop_test_img_path)
    test_roi_mask_file = os.path.join(file_test_patient_dir, test_roi_mask_pattern)
    test_crop_mask = find_crop_mask(test_roi_mask_file)
    image_data = read_volume_data(test_fname[0])

    if is_low_res is True:
        resampled_label = '_resampled.'
        test_img_path = os.path.join(dataset_path, test_patient_id, 'images', test_img_name_pattern[0])
        test_image = BrainImage(test_img_path, None)
        test_file = test_image.nii_file
        vox_size = test_file.header.get_zooms()
        image_vol = test_image.nii_data
        org_shape = np.shape(image_vol)
        thres = 0.7
    else:
        resampled_label = '.'

    #threshold for overlaped patch images (prob. vol) within the segmented volume
    prob_vol_norm = np.zeros(prob_volume.shape)
    for i in range(num_classes):
        prob_vol_norm[:, :, :, i] = normalize_image(prob_volume[:, :, :, i], [0, 1])
    volume = np.multiply(volume, prob_vol_norm[:, :, :, num_classes-1])
    volume_thr = volume > threshold

    # save probability map for background and foreground
    idx = 0
    class_name_lst = ['bg', 'fg']
    for class_name in class_name_lst:
        prob_map_crop_filename = class_name + '_' + seg_label + '_prob_map_crop_' + approach + '_' + loss + \
                                 resampled_label + file_format
        prob_map_crop_out_filepath = os.path.join(prob_map_output_dir, prob_map_crop_filename)
        print(prob_map_crop_out_filepath)
        __save_volume(prob_vol_norm[:, :, :, idx], crop_test_data, prob_map_crop_out_filepath, file_format,
                      is_compressed=False)
        idx += 1


    # split side (in LPI)
    left_mask, right_mask, volume_thr = compute_side_mask(volume_thr, crop_test_data, is_check_vol_diff=True)

    for mask, side in zip([left_mask, right_mask], ['left', 'right']):
        if volume_thr is None:
            failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
            with open(failed_cases_filepath, 'a') as f:
                f.write(test_patient_id + '_' + seg_label + '_' + side + '\n')
            continue

        # split the volume into left/right side
        vol = np.multiply(mask, volume_thr)

        # postprocessing
        vol_refined = postprocess(vol)
        if vol_refined is None:
            failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
            with open(failed_cases_filepath, 'a') as f:
                f.write(test_patient_id + '_' + seg_label + '_' + side + '\n')
            continue
        # save the cropped/refined result
        nii_crop_filename = side + '_' + seg_label + '_seg_crop_' + approach + '_' + loss + resampled_label + \
                            file_format
        nii_crop_out_filepath = os.path.join(non_smoothed_crop_output_dir, nii_crop_filename)
        print(nii_crop_out_filepath)
        __save_volume(vol_refined, crop_test_data, nii_crop_out_filepath, file_format, is_compressed=True)

        # uncrop segmentation only (left and right)
        vol_uncrop = test_crop_mask.uncrop(vol_refined)

        # save the uncropped result
        nii_uncrop_filename = side + '_' + seg_label + '_seg_' + approach + '_' + loss + resampled_label + file_format
        nii_uncrop_out_filepath = os.path.join(non_smoothed_output_dir, nii_uncrop_filename)
        print(nii_uncrop_out_filepath)
        __save_volume(vol_uncrop, image_data, nii_uncrop_out_filepath, file_format, is_compressed = True)

        if is_low_res is True:
            # resampling back to the original resolution
            res_label = '_org_res.'
            nii_uncrop_org_res_filename = side + '_' + seg_label + '_seg_' + approach + '_' + loss + res_label + \
                                          file_format
            nii_uncrop_org_res_out_filepath = os.path.join(non_smoothed_output_dir, nii_uncrop_org_res_filename)

            print('resampling %s back to %s' % (nii_uncrop_out_filepath, str(vox_size)))
            seg_resample_img = BrainImage(nii_uncrop_out_filepath, None)
            seg_resample_img.ResampleTo(nii_uncrop_org_res_out_filepath, new_affine=test_file.affine,
                                        new_shape=org_shape[:3], new_voxel_size=vox_size, is_ant_resample=False)
            seg_org_res_img_data = read_volume_data(nii_uncrop_org_res_out_filepath)
            seg_org_res_vol = seg_org_res_img_data.get_data()
            seg_org_res_vol_thres = normalize_image(seg_org_res_vol, [0, 1]) > thres
            seg_org_res_vol_thres_refined = postprocess(seg_org_res_vol_thres)
            __save_volume(seg_org_res_vol_thres_refined, seg_org_res_img_data, nii_uncrop_org_res_out_filepath,
                          file_format, is_compressed=True)
            nii_uncrop_out_filepath = nii_uncrop_org_res_out_filepath
        else:
            seg_org_res_img_data = image_data
            res_label = '.'

        # smoothing (before surface extraction)
        nii_smooth_filename = side + '_' + seg_label + '_seg_smooth_' + approach + '_' + loss + res_label + \
                              file_format
        nii_smooth_out_filepath = os.path.join(smoothed_output_dir, nii_smooth_filename)
        print(nii_smooth_out_filepath)
        __smooth_binary_img(nii_uncrop_out_filepath, nii_smooth_out_filepath, dim=dimension, maximumRMSError = 0.01,
                            numberOfIterations = 10, numberOfLayers = 3)

        # normalization
        nii_smooth_image = BrainImage(nii_smooth_out_filepath, None)
        vol_smooth_norm = nii_smooth_image.nii_data_normalized(bits=0)
        nii_smooth_norm_filename = side + '_' + seg_label + '_seg_smooth_norm_' + approach + '_' + loss + res_label + \
                                   file_format
        nii_smooth_norm_out_filepath = os.path.join(smoothed_norm_output_dir, nii_smooth_norm_filename)
        print(nii_smooth_norm_out_filepath)
        __save_volume(vol_smooth_norm, seg_org_res_img_data, nii_smooth_norm_out_filepath, file_format, is_compressed=False)

        # threshold
        vol_smooth_norm_thres = vol_smooth_norm > 0.4
        nii_smooth_norm_thres_filename = side + '_' + seg_label + '_seg_smooth_norm_thres_' + approach + '_' + loss + \
                                         res_label + file_format
        nii_smooth_norm_thres_out_filepath = os.path.join(smoothed_norm_thres_output_dir, nii_smooth_norm_thres_filename)
        print(nii_smooth_norm_thres_out_filepath)
        __save_volume(vol_smooth_norm_thres, seg_org_res_img_data, nii_smooth_norm_thres_out_filepath, file_format,
                      is_compressed = True)

        # crop threhold image for measure
        nii_smooth_norm_thres_crop_filename = side + '_' + seg_label + '_seg_smooth_norm_thres_crop_' + approach + \
                                              '_' + loss + res_label + file_format
        nii_smooth_norm_thres_crop_out_filepath = os.path.join(smoothed_norm_thres_crop_output_dir,
                                                               nii_smooth_norm_thres_crop_filename)
        print(nii_smooth_norm_thres_crop_out_filepath)
        crop_image(nii_smooth_norm_thres_out_filepath, test_crop_mask, nii_smooth_norm_thres_crop_out_filepath)

        if is_stl_out:
            # save stl
            stl_filename = side + '_' + seg_label + '_seg' + '.stl'
            stl_out_filepath= os.path.join(stl_output_dir, stl_filename)
            print(stl_out_filepath)
            __create_stl(nii_uncrop_out_filepath, stl_out_filepath)

            # smooth stl
            stl_smooth_filename = side + '_' + seg_label + '_seg_smooth.stl'
            stl_smooth_out_filepath= os.path.join(stl_output_dir, stl_smooth_filename)
            print(stl_smooth_out_filepath)
            __smooth_stl(stl_out_filepath, stl_smooth_out_filepath)


# processing smaller size
def save_volume_tha_v2(gen_conf, train_conf, test_conf, volume, prob_volume, test_fname, test_patient_id, seg_label,
                    file_output_dir):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    num_classes = gen_conf['num_classes']
    path = dataset_info['path']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    threshold = test_conf['threshold']
    approach = train_conf['approach']
    loss = train_conf['loss']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    data_augment = train_conf['data_augment']
    preprocess_trn = train_conf['preprocess']
    preprocess_tst = test_conf['preprocess']
    file_format = dataset_info['format']

    modality = dataset_info['image_modality']
    modality_idx = []
    for m in modality: # the order of modality images should be fixed due to the absolute path
        if m == 'T1':
            modality_idx.append(0)
        if m == 'B0':
            modality_idx.append(1)
        if m == 'FA':
            modality_idx.append(2)

    if data_augment == 1:
        data_augment_label = '_mixup'
    elif data_augment == 2:
        data_augment_label = '_datagen'
    elif data_augment == 3:
        data_augment_label = '_mixup+datagen'
    else:
        data_augment_label = ''

    file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
    seg_output_dir = os.path.join(file_test_patient_dir, path)
    if not os.path.exists(seg_output_dir):
        os.makedirs(os.path.join(file_test_patient_dir, path))

    crop_test_img_path = os.path.join(file_test_patient_dir, crop_tst_img_pattern[modality_idx[0]])
    crop_test_data = read_volume_data(crop_test_img_path)
    test_crop_vol = crop_test_data.get_data()
    source_size = test_crop_vol.shape
    target_size = volume.shape

    print('source_size: %s' % str(source_size))
    print('target_size: %s' % str(target_size))

    #upsampling of predicted volume and probability map
    scale_height, scale_width, scale_axis = source_size[0] / target_size[0], source_size[1] / target_size[1], \
                                            source_size[2] / target_size[2]
    volume = zoom(volume, (scale_height, scale_width, scale_axis))
    print(volume.shape)

    #threshold for overlaped patch images (prob. vol) within the segmented volume
    prob_vol_norm = np.zeros(source_size + (num_classes,))
    for i in range(num_classes):
        prob_volume_temp = zoom(prob_volume[:, :, :, i], (scale_height, scale_width, scale_axis))
        prob_vol_norm[:, :, :, i] = normalize_image(np.array(prob_volume_temp), [0, 1.0])
        print(prob_vol_norm[:, :, :, i].shape)

    volume = np.multiply(volume, prob_vol_norm[:, :, :, num_classes-1])
    volume_thr = volume > threshold

    # split side (in LPI)
    left_mask, right_mask, volume_thr = compute_side_mask(volume_thr, crop_test_data, is_check_vol_diff=True)

    if volume_thr is not None:
        # split the volume into left/right side
        left_vol = np.multiply(left_mask, volume_thr)
        right_vol = np.multiply(right_mask, volume_thr)

        # postprocessing
        left_vol_refined = postprocess(left_vol)
        right_vol_refined = postprocess(right_vol)

        # save the cropped/refined result
        out_filename_crop_left = seg_output_dir + '/' + 'left_' + seg_label + '_seg_crop_' + approach + '_' + loss + '_' + \
                                 str(patch_shape) + '_' + str(extraction_step) + data_augment_label + \
                                 '_preproc_trn_opt_' + str(preprocess_trn) + '_preproc_tst_opt_' + \
                                 str(preprocess_tst) + '.' + file_format

        out_filename_crop_right = seg_output_dir + '/' + 'right_' + seg_label + '_seg_crop_' + approach + '_' + loss + '_' + \
                                  str(patch_shape) + '_' + str(extraction_step) + data_augment_label + \
                                  '_preproc_trn_opt_' + str(preprocess_trn) + '_preproc_tst_opt_' + \
                                  str(preprocess_tst) + '.' + file_format

        print(out_filename_crop_left)
        print(out_filename_crop_right)

        __save_volume(left_vol_refined, crop_test_data, out_filename_crop_left, file_format,
                      is_compressed=True)
        __save_volume(right_vol_refined, crop_test_data, out_filename_crop_right, file_format,
                      is_compressed=True)

        # save probability map for background and foreground
        idx = 0
        class_name_lst = ['bg', 'fg']
        for class_name in class_name_lst:
            out_filename_crop_prob_map = seg_output_dir + '/' + class_name + '_' + seg_label + '_prob_map_crop_' + approach + '_' + \
                                         loss + '_' + str(patch_shape) + '_' + str(extraction_step) + \
                                         data_augment_label + '_preproc_trn_opt_' + str(preprocess_trn) + \
                                         '_preproc_tst_opt_' + str(preprocess_tst) + '.' + file_format
            print(out_filename_crop_prob_map)
            __save_volume(prob_vol_norm[:, :, :, idx], crop_test_data, out_filename_crop_prob_map,
                          file_format, is_compressed=False)
            idx += 1

        # uncrop segmentation only (left and right)
        test_roi_mask_file = os.path.join(file_test_patient_dir, test_roi_mask_pattern)
        test_crop_mask = find_crop_mask(test_roi_mask_file)
        left_vol_uncrop = test_crop_mask.uncrop(left_vol_refined)
        right_vol_uncrop = test_crop_mask.uncrop(right_vol_refined)

        image_data = read_volume_data(test_fname[0])

        # save the uncropped result
        out_filename_left = seg_output_dir + '/' + 'left_' + seg_label + '_seg_' + approach + '_' + loss + '_' + \
                            str(patch_shape) + '_' + str(extraction_step) + data_augment_label + \
                            '_preproc_trn_opt_' + str(preprocess_trn) + '_preproc_tst_opt_' + \
                            str(preprocess_tst) + '.' + file_format

        out_filename_right = seg_output_dir + '/' + 'right_' + seg_label + '_seg_' + approach + '_' + loss + '_' + \
                             str(patch_shape) + '_' + str(extraction_step) + data_augment_label + \
                             '_preproc_trn_opt_' + str(preprocess_trn) + '_preproc_tst_opt_' + \
                             str(preprocess_tst) + '.' + file_format

        print(out_filename_left)
        print(out_filename_right)

        __save_volume(left_vol_uncrop, image_data, out_filename_left, file_format, is_compressed = True)
        __save_volume(right_vol_uncrop, image_data, out_filename_right, file_format, is_compressed=True)


def save_volume_dentate(gen_conf, train_conf, test_conf, volume, prob_volume, test_fname, test_patient_id, seg_label,
                    file_output_dir):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    num_classes = gen_conf['num_classes']
    path = dataset_info['path']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    threshold = test_conf['threshold']

    approach = train_conf['approach']
    loss = train_conf['loss']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    data_augment = train_conf['data_augment']
    preprocess_trn = train_conf['preprocess']
    preprocess_tst = test_conf['preprocess']

    modality = dataset_info['image_modality']
    modality_idx = []
    for m in modality: # the order of modality images should be fixed due to the absolute path
        if m == 'B0':
            modality_idx.append(0)
        if m in ['T1', 'QSM']:
            modality_idx.append(1)
        if m == 'LP':
            modality_idx.append(2)
        if m == 'FA':
            modality_idx.append(2)

    if data_augment == 1:
        data_augment_label = '_mixup'
    elif data_augment == 2:
        data_augment_label = '_datagen'
    elif data_augment == 3:
        data_augment_label = '_mixup+datagen'
    else:
        data_augment_label = ''

    seg_output_dir = file_output_dir + '/' + test_patient_id + '/' + path + '/'
    if not os.path.exists(seg_output_dir):
        os.makedirs(os.path.join(file_output_dir, test_patient_id, path))

    crop_test_img_path = file_output_dir + '/' + test_patient_id + '/' \
                         + crop_tst_img_pattern[modality_idx[0]].format(test_patient_id)
    crop_test_data = read_volume_data(crop_test_img_path)

    #threshold for overlaped patch images (prob. vol) within the segmented volume
    #threshold = 0.5 # original 0
    print ('threshold: %s' % threshold)
    prob_vol_norm = np.zeros(prob_volume.shape)
    for i in range(num_classes):
        prob_vol_norm[:, :, :, i] = normalize_image(prob_volume[:, :, :, i], [0, 1])
    volume = np.multiply(volume, prob_vol_norm[:, :, :, num_classes-1])
    volume_thr = volume > threshold

    # split side (in LPI)
    left_mask, right_mask, volume_thr = compute_side_mask(volume_thr, crop_test_data, is_check_vol_diff=False)

    if volume_thr is not None:
        left_vol = np.multiply(left_mask, volume_thr)
        right_vol = np.multiply(right_mask, volume_thr)

        # postprocessing
        left_vol_refined = postprocess(left_vol)
        right_vol_refined = postprocess(right_vol)

        # save the cropped/refined result
        out_filename_crop_left = seg_output_dir + 'left_' + seg_label + '_seg_crop_' + approach + '_' + loss + '_' + \
                                 str(patch_shape) + '_' + str(extraction_step) + data_augment_label + \
                                 '_preproc_trn_opt_' + str(preprocess_trn) + '_preproc_tst_opt_' + \
                                 str(preprocess_tst)

        out_filename_crop_right = seg_output_dir + 'right_' + seg_label + '_seg_crop_' + approach + '_' + loss + '_' + \
                                  str(patch_shape) + '_' + str(extraction_step) + data_augment_label + \
                                  '_preproc_trn_opt_' + str(preprocess_trn) + '_preproc_tst_opt_' + \
                                  str(preprocess_tst)

        print(out_filename_crop_left)
        print(out_filename_crop_right)

        __save_volume(left_vol_refined, crop_test_data, out_filename_crop_left, dataset_info['format'],
                      is_compressed=True)
        __save_volume(right_vol_refined, crop_test_data, out_filename_crop_right, dataset_info['format'],
                      is_compressed=True)

        # save probability map for background and foreground
        idx = 0
        class_name_lst = ['bg', 'fg']
        for class_name in class_name_lst:
            out_filename_crop_prob_map = seg_output_dir + class_name + '_' + seg_label + '_prob_map_crop_' + approach + '_' + \
                                         loss + '_' + str(patch_shape) + '_' + str(extraction_step) + \
                                         data_augment_label + '_preproc_trn_opt_' + str(preprocess_trn) + \
                                         '_preproc_tst_opt_' + str(preprocess_tst)
            print(out_filename_crop_prob_map)
            __save_volume(prob_vol_norm[:, :, :, idx], crop_test_data, out_filename_crop_prob_map,
                          dataset_info['format'], is_compressed=False)
            idx += 1

        # uncrop segmentation only (left and right)
        test_roi_mask_file = file_output_dir + test_roi_mask_pattern.format(test_patient_id)
        test_crop_mask = find_crop_mask(test_roi_mask_file)
        left_vol_uncrop = test_crop_mask.uncrop(left_vol_refined)
        right_vol_uncrop = test_crop_mask.uncrop(right_vol_refined)

        image_data = read_volume_data(test_fname[0])

        # save the uncropped result
        out_filename_left = seg_output_dir + 'left_' + seg_label + '_seg_' + approach + '_' + loss + '_' + \
                            str(patch_shape) + '_' + str(extraction_step) + data_augment_label + \
                            '_preproc_trn_opt_' + str(preprocess_trn) + '_preproc_tst_opt_' + \
                            str(preprocess_tst)

        out_filename_right = seg_output_dir + 'right_' + seg_label + '_seg_' + approach + '_' + loss + '_' + \
                             str(patch_shape) + '_' + str(extraction_step) + data_augment_label + \
                             '_preproc_trn_opt_' + str(preprocess_trn) + '_preproc_tst_opt_' + \
                             str(preprocess_tst)

        print(out_filename_left)
        print(out_filename_right)

        __save_volume(left_vol_uncrop, image_data, out_filename_left, dataset_info['format'], is_compressed = True)
        __save_volume(right_vol_uncrop, image_data, out_filename_right, dataset_info['format'], is_compressed = True)


def save_volume_dentate_interposed(gen_conf, train_conf, test_conf, volume, prob_volume, test_fname, test_patient_id,
                                   file_output_dir, target):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    path = dataset_info['path']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    threshold = test_conf['threshold']

    multi_output = gen_conf['multi_output']
    num_classes = gen_conf['num_classes']
    approach = train_conf['approach']
    loss = train_conf['loss']
    data_augment = train_conf['data_augment']
    activation = train_conf['activation']
    exclusive_train = train_conf['exclusive_train']
    exclude_label_num = dataset_info['exclude_label_num']

    dimension = test_conf['dimension']
    file_format = dataset_info['format']

    modality = dataset_info['image_modality']
    modality_idx = []
    for m in modality: # the order of modality images should be fixed due to the absolute path
        if m == 'B0':
            modality_idx.append(0)
        if m in ['T1', 'QSM']:
            modality_idx.append(1)
        if m == 'LP':
            modality_idx.append(2)
        if m == 'FA':
            modality_idx.append(2)

    if len(target) == 2:
        target = 'both'

    file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
    seg_output_dir = os.path.join(file_test_patient_dir, path)
    if not os.path.exists(seg_output_dir):
        os.makedirs(seg_output_dir)
    prob_map_output_dir = os.path.join(seg_output_dir, 'prob_map_crop')
    if not os.path.exists(prob_map_output_dir):
        os.makedirs(prob_map_output_dir)
    non_smoothed_output_dir = os.path.join(seg_output_dir, 'non_smoothed')
    if not os.path.exists(non_smoothed_output_dir):
        os.makedirs(non_smoothed_output_dir)
    non_smoothed_crop_output_dir = os.path.join(seg_output_dir, 'non_smoothed', 'crop')
    if not os.path.exists(non_smoothed_crop_output_dir):
        os.makedirs(non_smoothed_crop_output_dir)
    smoothed_output_dir = os.path.join(seg_output_dir, 'smoothed')
    if not os.path.exists(smoothed_output_dir):
        os.makedirs(smoothed_output_dir)
    smoothed_norm_output_dir = os.path.join(seg_output_dir, 'smoothed_norm')
    if not os.path.exists(smoothed_norm_output_dir):
        os.makedirs(smoothed_norm_output_dir)
    smoothed_norm_thres_output_dir = os.path.join(seg_output_dir, 'smoothed_norm_thres')
    if not os.path.exists(smoothed_norm_thres_output_dir):
        os.makedirs(smoothed_norm_thres_output_dir)
    smoothed_norm_thres_crop_output_dir = os.path.join(seg_output_dir, 'smoothed_norm_thres', 'crop')
    if not os.path.exists(smoothed_norm_thres_crop_output_dir):
        os.makedirs(smoothed_norm_thres_crop_output_dir)

    is_stl_out = False

    if is_stl_out:
        stl_output_dir = os.path.join(seg_output_dir, 'stl_out')
        if not os.path.exists(stl_output_dir):
            os.makedirs(os.path.join(seg_output_dir, 'stl_out'))

    crop_test_img_path = os.path.join(file_test_patient_dir, crop_tst_img_pattern[modality_idx[0]].format(test_patient_id))
    crop_test_data = read_volume_data(crop_test_img_path)

    test_roi_mask_file = os.path.join(file_output_dir, test_roi_mask_pattern.format(test_patient_id))
    test_crop_mask = find_crop_mask(test_roi_mask_file)
    image_data = read_volume_data(test_fname[0])

    # save probability map for background and foreground
    if target == 'dentate':
        class_name_lst = ['bg', 'dentate']
        loss_ch = [loss, loss]
    elif target == 'interposed':
        class_name_lst = ['bg', 'interposed']
        loss_ch = [loss, loss]
    else:
        if multi_output == 1:
            class_name_lst = ['bg_dentate', 'bg_interposed', 'dentate', 'interposed']
            loss_ch = [loss[0], loss[1], loss[0], loss[1]]
            output_ch = [0, 1, 0, 1]
            idx_lst = [0, 0, num_classes[0]-1, num_classes[1]-1]
        else:
            class_name_lst = ['bg', 'dentate', 'interposed']
            loss_ch = [loss, loss, loss]

    if exclusive_train == 1:
        class_name_lst.remove(class_name_lst[exclude_label_num])

    if multi_output == 1:
        for class_name, l_ch, o_ch, idx in zip(class_name_lst, loss_ch, output_ch, idx_lst):
            prob_map_crop_filename = class_name + '_' + '_prob_map_crop_' + approach + '_' + l_ch + '.' + \
                                     file_format
            prob_map_crop_out_filepath = os.path.join(prob_map_output_dir, prob_map_crop_filename)
            print(prob_map_crop_out_filepath)
            __save_volume(prob_volume[o_ch][:, :, :, idx], crop_test_data, prob_map_crop_out_filepath, file_format,
                          is_compressed=False)
    else:
        idx = 0
        for class_name, l_ch in zip(class_name_lst, loss_ch):
            prob_map_crop_filename = class_name + '_' + '_prob_map_crop_' + approach + '_' + l_ch + '.' + \
                                     file_format
            prob_map_crop_out_filepath = os.path.join(prob_map_output_dir, prob_map_crop_filename)
            print(prob_map_crop_out_filepath)
            __save_volume(prob_volume[:, :, :, idx], crop_test_data, prob_map_crop_out_filepath, file_format,
                          is_compressed=False)
            idx += 1

    if target == 'dentate':
        key_lst = [1]
        seg_label_lst = ['dentate']
        if multi_output == 1:
            activation_ch = [activation[0]]
        else:
            activation_ch = [activation]

    elif target == 'interposed':
        key_lst = [1]
        seg_label_lst = ['interposed']
        if multi_output == 1:
            activation_ch = [activation[1]]
        else:
            activation_ch = [activation]
    else:
        seg_label_lst = ['dentate','interposed']
        if multi_output == 1:
            key_lst = [2, 3]
            activation_ch = [activation[0], activation[1]]
        else:
            key_lst = [1, 2]
            activation_ch = [activation, activation]

    if exclusive_train == 1:
        print('exclude_label_num: %s' % exclude_label_num)
        num_classes_org = gen_conf['num_classes']
        print('num_classes_org: %s' % num_classes_org)
        num_classes = num_classes_org - 1
        volume_org_shape = (volume.shape[0], volume.shape[1], volume.shape[2], num_classes_org)
        volume_org = np.zeros(volume_org_shape)
        label_idx = []
        for i in range(num_classes_org):
            label_idx.append(i)
        label_idx.remove(exclude_label_num)
        for i, j in zip(range(num_classes), label_idx):
            volume_org[:, :, :, j] = volume[:, :, :, i]
        volume_org[:, :, :, exclude_label_num] = (np.sum(volume_org, axis=3) == 0).astype(np.uint8)
        volume = volume_org

    #volume[:, :, :, 2] -= volume[:, :, :, 1]
    for a_ch, seg_label, key in zip(activation_ch, seg_label_lst, key_lst):
        if a_ch == 'softmax':
            if multi_output == 1:
                vol_temp = np.zeros(volume[output_ch[key]].shape)
                vol_temp[volume[output_ch[key]] == num_classes[output_ch[key]]-1] = 1
                print('threshold: %s' % threshold)
                prob_vol_norm = normalize_image(prob_volume[output_ch[key]][:, :, :, num_classes[output_ch[key]]-1],
                                                [0, 1])
                volume_prob = np.multiply(vol_temp, prob_vol_norm)
                volume_f = volume_prob > threshold
            else:
                vol_temp = np.zeros(volume.shape)
                vol_temp[volume == key] = 1
                print('threshold: %s' % threshold)
                prob_vol_norm = normalize_image(prob_volume[:, :, :, key], [0, 1])
                volume_prob = np.multiply(vol_temp, prob_vol_norm)
                volume_f = volume_prob > threshold
        elif a_ch == 'sigmoid':
            if multi_output == 1:
                volume_f = volume[output_ch[key]][:, :, :, num_classes[output_ch[key]]-1]
            else:
                volume_f = volume[:, :, :, key]

        # split side (in LPI)
        left_mask, right_mask, volume_f = compute_side_mask(volume_f, crop_test_data, is_check_vol_diff=False)

        for mask, side in zip([left_mask, right_mask], ['left', 'right']):
            if volume_f is None:
                failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
                with open(failed_cases_filepath, 'a') as f:
                    f.write(test_patient_id + '_' + seg_label + '_' + side + '\n')
                continue

            # split the volume into left/right side
            vol = np.multiply(mask, volume_f)

            # postprocessing
            vol_refined = postprocess(vol)
            if vol_refined is None:
                failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
                with open(failed_cases_filepath, 'a') as f:
                    f.write(test_patient_id + '_' + seg_label + '_' + side + '\n')
                continue
            # save the cropped/refined result
            nii_crop_filename = side + '_' + seg_label + '_seg_crop_' + approach + '_' + loss_ch[key] + '.' + file_format
            nii_crop_out_filepath = os.path.join(non_smoothed_crop_output_dir, nii_crop_filename)
            print(nii_crop_out_filepath)
            __save_volume(vol_refined, crop_test_data, nii_crop_out_filepath, file_format, is_compressed=True)

            # uncrop segmentation only (left and right)
            vol_uncrop = test_crop_mask.uncrop(vol_refined)

            # save the uncropped result
            nii_uncrop_filename = side + '_' + seg_label + '_seg_' + approach + '_' + loss_ch[key] + '.' + file_format
            nii_uncrop_out_filepath = os.path.join(non_smoothed_output_dir, nii_uncrop_filename)
            print(nii_uncrop_out_filepath)
            __save_volume(vol_uncrop, image_data, nii_uncrop_out_filepath, file_format, is_compressed = True)

            # smoothing (before surface extraction)
            nii_smooth_filename = side + '_' + seg_label + '_seg_smooth_' + approach + '_' + loss_ch[key] + '.' + file_format
            nii_smooth_out_filepath = os.path.join(smoothed_output_dir, nii_smooth_filename)
            print(nii_smooth_out_filepath)
            __smooth_binary_img(nii_uncrop_out_filepath, nii_smooth_out_filepath, dim=dimension, maximumRMSError = 0.01,
                                numberOfIterations = 10, numberOfLayers = 3)

            # normalization
            nii_smooth_image = BrainImage(nii_smooth_out_filepath, None)
            vol_smooth_norm = nii_smooth_image.nii_data_normalized(bits=0)
            nii_smooth_norm_filename = side + '_' + seg_label + '_seg_smooth_norm_' + approach + '_' + loss_ch[key] + '.' + \
                                       file_format
            nii_smooth_norm_out_filepath = os.path.join(smoothed_norm_output_dir, nii_smooth_norm_filename)
            print(nii_smooth_norm_out_filepath)
            __save_volume(vol_smooth_norm, image_data, nii_smooth_norm_out_filepath, file_format, is_compressed=False)

            # threshold
            vol_smooth_norm_thres = vol_smooth_norm > 0.6
            nii_smooth_norm_thres_filename = side + '_' + seg_label + '_seg_smooth_norm_thres_' + approach + '_' + loss_ch[key] + '.' + \
                                             file_format
            nii_smooth_norm_thres_out_filepath = os.path.join(smoothed_norm_thres_output_dir, nii_smooth_norm_thres_filename)
            print(nii_smooth_norm_thres_out_filepath)
            __save_volume(vol_smooth_norm_thres, image_data, nii_smooth_norm_thres_out_filepath, file_format,
                          is_compressed=True)

            # crop threhold image for measure
            nii_smooth_norm_thres_crop_filename = side + '_' + seg_label + '_seg_smooth_norm_thres_crop_' + approach + '_' + \
                                                  loss_ch[key] + '.' +  file_format
            nii_smooth_norm_thres_crop_out_filepath = os.path.join(smoothed_norm_thres_crop_output_dir,
                                                                   nii_smooth_norm_thres_crop_filename)
            print(nii_smooth_norm_thres_crop_out_filepath)
            crop_image(nii_smooth_norm_thres_out_filepath, test_crop_mask, nii_smooth_norm_thres_crop_out_filepath)

            if is_stl_out:
                # save stl
                stl_filename = side + '_' + seg_label + '_seg' + '.stl'
                stl_out_filepath= os.path.join(stl_output_dir, stl_filename)
                print(stl_out_filepath)
                __create_stl(nii_uncrop_out_filepath, stl_out_filepath)

                # smooth stl
                stl_smooth_filename = side + '_' + seg_label + '_seg_smooth.stl'
                stl_smooth_out_filepath= os.path.join(stl_output_dir, stl_smooth_filename)
                print(stl_smooth_out_filepath)
                __smooth_stl(stl_out_filepath, stl_smooth_out_filepath)


def save_volume_dentate_interposed_unseen(gen_conf, train_conf, test_conf, volume, prob_volume, test_fname, test_patient_id,
                                   file_output_dir, target, is_low_res):
    root_path = gen_conf['root_path']
    dataset_path = gen_conf['dataset_path']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    path = dataset_info['path']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    threshold = test_conf['threshold']

    multi_output = gen_conf['multi_output']
    num_classes = gen_conf['num_classes']
    approach = train_conf['approach']
    loss = train_conf['loss']
    data_augment = train_conf['data_augment']
    activation = train_conf['activation']
    file_format = dataset_info['format']
    dimension = test_conf['dimension']

    exclusive_train = train_conf['exclusive_train']
    exclude_label_num = dataset_info['exclude_label_num']

    modality = dataset_info['image_modality']
    modality_idx = []

    test_img_pattern = dataset_info['image_name_pattern']
    test_image_new_name_pattern = dataset_info['image_new_name_pattern']
    test_img_resampled_name_pattern = dataset_info['image_resampled_name_pattern']

    for m in modality: # the order of modality images should be fixed due to the absolute path
        if m == 'B0':
            modality_idx.append(0)
        if m in ['T1', 'QSM']:
            modality_idx.append(1)
        if m == 'LP':
            modality_idx.append(2)
        if m == 'FA':
            modality_idx.append(3)

    if len(target) == 2:
        target = 'both'

    file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
    seg_output_dir = os.path.join(file_test_patient_dir, path)
    if not os.path.exists(seg_output_dir):
        os.makedirs(seg_output_dir)
    prob_map_output_dir = os.path.join(seg_output_dir, 'prob_map_crop')
    if not os.path.exists(prob_map_output_dir):
        os.makedirs(prob_map_output_dir)
    non_smoothed_output_dir = os.path.join(seg_output_dir, 'non_smoothed')
    if not os.path.exists(non_smoothed_output_dir):
        os.makedirs(non_smoothed_output_dir)
    non_smoothed_crop_output_dir = os.path.join(seg_output_dir, 'non_smoothed', 'crop')
    if not os.path.exists(non_smoothed_crop_output_dir):
        os.makedirs(non_smoothed_crop_output_dir)
    smoothed_output_dir = os.path.join(seg_output_dir, 'smoothed')
    if not os.path.exists(smoothed_output_dir):
        os.makedirs(smoothed_output_dir)
    smoothed_norm_output_dir = os.path.join(seg_output_dir, 'smoothed_norm')
    if not os.path.exists(smoothed_norm_output_dir):
        os.makedirs(smoothed_norm_output_dir)
    smoothed_norm_thres_output_dir = os.path.join(seg_output_dir, 'smoothed_norm_thres')
    if not os.path.exists(smoothed_norm_thres_output_dir):
        os.makedirs(smoothed_norm_thres_output_dir)
    smoothed_norm_thres_crop_output_dir = os.path.join(seg_output_dir, 'smoothed_norm_thres', 'crop')
    if not os.path.exists(smoothed_norm_thres_crop_output_dir):
        os.makedirs(smoothed_norm_thres_crop_output_dir)

    is_stl_out = False

    if is_stl_out:
        stl_output_dir = os.path.join(seg_output_dir, 'stl_out')
        if not os.path.exists(stl_output_dir):
            os.makedirs(os.path.join(seg_output_dir, 'stl_out'))

    crop_test_img_path = os.path.join(file_test_patient_dir, crop_tst_img_pattern[modality_idx[0]].format(test_patient_id))
    crop_test_data = read_volume_data(crop_test_img_path)

    test_roi_mask_file = os.path.join(file_output_dir, test_roi_mask_pattern.format(test_patient_id))
    test_crop_mask = find_crop_mask(test_roi_mask_file)
    image_data = read_volume_data(test_fname[0])

    if is_low_res is True:
        resampled_label = '_resampled.'
        test_img_path = os.path.join(dataset_path, test_patient_id,
                                     test_img_pattern[modality_idx[0]].format(test_patient_id))
        if not os.path.exists(test_img_path):
            test_img_path = os.path.join(root_path, 'datasets', 'dcn', test_patient_id, 'image',
                                         test_image_new_name_pattern[modality_idx[0]].format(test_patient_id))
        test_image = BrainImage(test_img_path, None)
        test_file = test_image.nii_file
        vox_size = test_file.header.get_zooms()
        image_vol = test_image.nii_data
        org_shape = np.shape(image_vol)
        thres = 0.7
    else:
        resampled_label = '.'

    # save probability map for background and foreground
    if target == 'dentate':
        class_name_lst = ['bg', 'dentate']
        loss_ch = [loss, loss]
    elif target == 'interposed':
        class_name_lst = ['bg', 'interposed']
        loss_ch = [loss, loss]
    else:
        if multi_output == 1:
            class_name_lst = ['bg_dentate', 'bg_interposed', 'dentate', 'interposed']
            loss_ch = [loss[0], loss[1], loss[0], loss[1]]
            output_ch = [0, 1, 0, 1]
            idx_lst = [0, 0, num_classes[0]-1, num_classes[1]-1]
        else:
            class_name_lst = ['bg', 'dentate', 'interposed']
            loss_ch = [loss, loss, loss]

    if exclusive_train == 1:
        class_name_lst.remove(class_name_lst[exclude_label_num])

    if multi_output == 1:
        for class_name, l_ch, o_ch, idx in zip(class_name_lst, loss_ch, output_ch, idx_lst):
            prob_map_crop_filename = class_name + '_' + '_prob_map_crop_' + approach + '_' + l_ch + '.' + \
                                     file_format
            prob_map_crop_out_filepath = os.path.join(prob_map_output_dir, prob_map_crop_filename)
            print(prob_map_crop_out_filepath)
            __save_volume(prob_volume[o_ch][:, :, :, idx], crop_test_data, prob_map_crop_out_filepath, file_format,
                          is_compressed=False)
    else:
        idx = 0
        for class_name, l_ch in zip(class_name_lst, loss_ch):
            prob_map_crop_filename = class_name + '_' + '_prob_map_crop_' + approach + '_' + l_ch + '.' + \
                                     file_format
            prob_map_crop_out_filepath = os.path.join(prob_map_output_dir, prob_map_crop_filename)
            print(prob_map_crop_out_filepath)
            __save_volume(prob_volume[:, :, :, idx], crop_test_data, prob_map_crop_out_filepath, file_format,
                          is_compressed=False)
            idx += 1

    if target == 'dentate':
        key_lst = [1]
        seg_label_lst = ['dentate']
        if multi_output == 1:
            activation_ch = [activation[0]]
        else:
            activation_ch = [activation]

    elif target == 'interposed':
        key_lst = [1]
        seg_label_lst = ['interposed']
        if multi_output == 1:
            activation_ch = [activation[1]]
        else:
            activation_ch = [activation]
    else:
        seg_label_lst = ['dentate','interposed']
        if multi_output == 1:
            key_lst = [2, 3]
            activation_ch = [activation[0], activation[1]]
        else:
            key_lst = [1, 2]
            activation_ch = [activation, activation]

    if exclusive_train == 1:
        print('exclude_label_num: %s' % exclude_label_num)
        num_classes_org = gen_conf['num_classes']
        print('num_classes_org: %s' % num_classes_org)
        num_classes = num_classes_org - 1
        volume_org_shape = (volume.shape[0], volume.shape[1], volume.shape[2], num_classes_org)
        volume_org = np.zeros(volume_org_shape)
        label_idx = []
        for i in range(num_classes_org):
            label_idx.append(i)
        label_idx.remove(exclude_label_num)
        for i, j in zip(range(num_classes), label_idx):
            volume_org[:, :, :, j] = volume[:, :, :, i]
        volume_org[:, :, :, exclude_label_num] = (np.sum(volume_org, axis=3) == 0).astype(np.uint8)
        volume = volume_org

    for a_ch, seg_label, key in zip(activation_ch, seg_label_lst, key_lst):
        if a_ch == 'softmax':
            if multi_output == 1:
                vol_temp = np.zeros(volume[output_ch[key]].shape)
                vol_temp[volume[output_ch[key]] == num_classes[output_ch[key]]-1] = 1
                print('threshold: %s' % threshold)
                prob_vol_norm = normalize_image(prob_volume[output_ch[key]][:, :, :, num_classes[output_ch[key]]-1],
                                                [0, 1])
                volume_prob = np.multiply(vol_temp, prob_vol_norm)
                volume_f = volume_prob > threshold
            else:
                vol_temp = np.zeros(volume.shape)
                vol_temp[volume == key] = 1
                print('threshold: %s' % threshold)
                prob_vol_norm = normalize_image(prob_volume[:, :, :, key], [0, 1])
                volume_prob = np.multiply(vol_temp, prob_vol_norm)
                volume_f = volume_prob > threshold
        elif a_ch == 'sigmoid':
            if multi_output == 1:
                volume_f = volume[output_ch[key]][:, :, :, num_classes[output_ch[key]]-1]
            else:
                volume_f = volume[:, :, :, key]

        # split side (in LPI)
        left_mask, right_mask, volume_f = compute_side_mask(volume_f, crop_test_data, is_check_vol_diff=False)

        for mask, side in zip([left_mask, right_mask], ['left', 'right']):
            if volume_f is None:
                failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
                with open(failed_cases_filepath, 'a') as f:
                    f.write(test_patient_id + '_' + seg_label + '_' + side + '\n')
                continue

            # split the volume into left/right side
            vol = np.multiply(mask, volume_f)

            # postprocessing
            vol_refined = postprocess(vol)
            if vol_refined is None:
                failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
                with open(failed_cases_filepath, 'a') as f:
                    f.write(test_patient_id + '_' + seg_label + '_' + side + '\n')
                continue
            # save the cropped/refined result
            nii_crop_filename = side + '_' + seg_label + '_seg_crop_' + approach + '_' + loss_ch[key] + resampled_label + \
                                file_format
            nii_crop_out_filepath = os.path.join(non_smoothed_crop_output_dir, nii_crop_filename)
            print(nii_crop_out_filepath)
            __save_volume(vol_refined, crop_test_data, nii_crop_out_filepath, file_format, is_compressed=True)

            # uncrop segmentation only (left and right)
            vol_uncrop = test_crop_mask.uncrop(vol_refined)

            # save the uncropped result
            nii_uncrop_filename = side + '_' + seg_label + '_seg_' + approach + '_' + loss_ch[key] + resampled_label + \
                                  file_format
            nii_uncrop_out_filepath = os.path.join(non_smoothed_output_dir, nii_uncrop_filename)
            print(nii_uncrop_out_filepath)
            __save_volume(vol_uncrop, image_data, nii_uncrop_out_filepath, file_format, is_compressed = True)

            if is_low_res is True:
                # resampling back to the original resolution
                res_label = '_org_res.'
                nii_uncrop_org_res_filename = side + '_' + seg_label + '_seg_' + approach + '_' + loss_ch[key] + res_label + \
                                              file_format
                nii_uncrop_org_res_out_filepath = os.path.join(non_smoothed_output_dir, nii_uncrop_org_res_filename)

                print('resampling %s back to %s' % (nii_uncrop_out_filepath, str(vox_size)))
                seg_resample_img = BrainImage(nii_uncrop_out_filepath, None)
                seg_resample_img.ResampleTo(nii_uncrop_org_res_out_filepath, new_affine=test_file.affine,
                                            new_shape=org_shape[:3], new_voxel_size=vox_size, is_ant_resample=False)
                seg_org_res_img_data = read_volume_data(nii_uncrop_org_res_out_filepath)
                seg_org_res_vol = seg_org_res_img_data.get_data()
                seg_org_res_vol_thres = normalize_image(seg_org_res_vol, [0, 1]) > thres
                seg_org_res_vol_thres_refined = postprocess(seg_org_res_vol_thres)
                __save_volume(seg_org_res_vol_thres_refined, seg_org_res_img_data, nii_uncrop_org_res_out_filepath,
                              file_format, is_compressed=True)
                nii_uncrop_out_filepath = nii_uncrop_org_res_out_filepath
            else:
                seg_org_res_img_data = image_data
                res_label = '.'

            # smoothing (before surface extraction)
            nii_smooth_filename = side + '_' + seg_label + '_seg_smooth_' + approach + '_' + loss_ch[key] + res_label + \
                                  file_format
            nii_smooth_out_filepath = os.path.join(smoothed_output_dir, nii_smooth_filename)
            print(nii_smooth_out_filepath)
            __smooth_binary_img(nii_uncrop_out_filepath, nii_smooth_out_filepath, dim=dimension, maximumRMSError = 0.01,
                                numberOfIterations = 10, numberOfLayers = 3)

            # normalization
            nii_smooth_image = BrainImage(nii_smooth_out_filepath, None)
            vol_smooth_norm = nii_smooth_image.nii_data_normalized(bits=0)
            nii_smooth_norm_filename = side + '_' + seg_label + '_seg_smooth_norm_' + approach + '_' + loss_ch[key] + '.' + \
                                       file_format
            nii_smooth_norm_out_filepath = os.path.join(smoothed_norm_output_dir, nii_smooth_norm_filename)
            print(nii_smooth_norm_out_filepath)
            __save_volume(vol_smooth_norm, seg_org_res_img_data, nii_smooth_norm_out_filepath, file_format,
                          is_compressed=False)

            # threshold
            vol_smooth_norm_thres = vol_smooth_norm > 0.6
            nii_smooth_norm_thres_filename = side + '_' + seg_label + '_seg_smooth_norm_thres_' + approach + '_' + loss_ch[key] + '.' + \
                                             file_format
            nii_smooth_norm_thres_out_filepath = os.path.join(smoothed_norm_thres_output_dir, nii_smooth_norm_thres_filename)
            print(nii_smooth_norm_thres_out_filepath)
            __save_volume(vol_smooth_norm_thres, seg_org_res_img_data, nii_smooth_norm_thres_out_filepath, file_format,
                          is_compressed=True)

            # crop threhold image for measure
            nii_smooth_norm_thres_crop_filename = side + '_' + seg_label + '_seg_smooth_norm_thres_crop_' + approach + '_' + \
                                                  loss_ch[key] + '.' + file_format
            nii_smooth_norm_thres_crop_out_filepath = os.path.join(smoothed_norm_thres_crop_output_dir,
                                                                   nii_smooth_norm_thres_crop_filename)
            print(nii_smooth_norm_thres_crop_out_filepath)
            crop_image(nii_smooth_norm_thres_out_filepath, test_crop_mask, nii_smooth_norm_thres_crop_out_filepath)

            if is_stl_out:
                # save stl
                stl_filename = side + '_' + seg_label + '_seg' + '.stl'
                stl_out_filepath = os.path.join(stl_output_dir, stl_filename)
                print(stl_out_filepath)
                __create_stl(nii_uncrop_out_filepath, stl_out_filepath)

                # smooth stl
                stl_smooth_filename = side + '_' + seg_label + '_seg_smooth.stl'
                stl_smooth_out_filepath= os.path.join(stl_output_dir, stl_smooth_filename)
                print(stl_smooth_out_filepath)
                __smooth_stl(stl_out_filepath, stl_smooth_out_filepath)


def save_sar_volume(gen_conf, train_conf, test_vol, gt, sar_pred, sar_prob, ovr_pat, test_id, file_output_dir):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    file_format = dataset_info['format']
    modality = dataset_info['image_modality']
    num_modality = len(modality)

    sar_output_dir = os.path.join(file_output_dir, test_id)
    if not os.path.exists(sar_output_dir):
        os.makedirs(sar_output_dir)
    # sar_output_dir = os.path.join(file_test_patient_dir, path)
    # if not os.path.exists(sar_output_dir):
    #     os.makedirs(sar_output_dir)

    sar_gt = gt[0, 0, :, :, :]

    for m, idx in zip(modality, range(num_modality)):
        src_filename = os.path.join(sar_output_dir, 'source_%s_data.%s' % (m, file_format))
        __save_sar_volume(test_vol[0][idx, :, :, :], src_filename, dataset_info['format'], 'float32', is_compressed = False)
        print('saved source %s data as nii.gz' % m)

    sar_filename = os.path.join(sar_output_dir, 'sar_groundtruth_data.%s' % file_format)
    __save_sar_volume(sar_gt, sar_filename, dataset_info['format'], 'float32', is_compressed=False)
    print('saved sar groundtruth data as nii.gz')

    sar_filename = os.path.join(sar_output_dir, 'sar_prediction_data.%s' % file_format)
    __save_sar_volume(sar_pred, sar_filename, dataset_info['format'], 'float32', is_compressed = False)
    print('saved sar prediction data as nii.gz')

    sar_filename = os.path.join(sar_output_dir, 'sar_prob_data.%s' % file_format)
    __save_sar_volume(sar_prob, sar_filename, dataset_info['format'], 'float32', is_compressed = False)
    print('saved sar probability data as nii.gz')

    sar_err = np.abs(np.subtract(sar_gt, sar_pred))
    sar_err_filename = os.path.join(sar_output_dir, 'sar_err.%s' % file_format)
    __save_sar_volume(sar_err, sar_err_filename, dataset_info['format'], 'float32', is_compressed = True)
    print('saved sar error data as nii.gz')

    sar_filename = os.path.join(sar_output_dir, 'ovr_patches_data.%s' % file_format)
    __save_sar_volume(ovr_pat, sar_filename, dataset_info['format'], 'float32', is_compressed = False)
    print('saved overlap patches data as nii.gz')

    # Create a dictionary
    dict = {}
    dict['pred_sar'] = sar_pred
    dict['err_btw_pred_gt'] = sar_err
    sio.savemat(os.path.join(sar_output_dir, 'sar_prediction.mat'), dict)
    print('saved sar prediction and error data as mat')


def save_sar_volume_2_5D(gen_conf, train_conf, test_vol, gt, pred_vol_ensemble, pred_vol_total, ensemble_rescaled,
                         rescaled_out_total, test_id, trn_dim, file_output_dir, case_name):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    file_format = dataset_info['format']
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    [slice_crop_size_duke, slice_crop_size_ella, slice_crop_size_louis, slice_crop_size_austinman, slice_crop_size_austinwoman] \
        = dataset_info['margin_crop_mask']

    sar_output_dir = os.path.join(file_output_dir, '#' + str(case_name), test_id)
    if not os.path.exists(sar_output_dir):
        os.makedirs(sar_output_dir)
    # sar_output_dir = os.path.join(file_test_patient_dir, path)
    # if not os.path.exists(sar_output_dir):
    #     os.makedirs(sar_output_dir)

    if int(test_id) <= 40:
        head_model = 'duke'
        slice_crop_size = slice_crop_size_duke
    elif int(test_id) > 40 and int(test_id) <= 80:
        head_model = 'ella'
        slice_crop_size = slice_crop_size_ella
    elif int(test_id) > 80 and int(test_id) <= 120:
        head_model = 'louis'
        slice_crop_size = slice_crop_size_louis
    elif int(test_id) > 120 and int(test_id) <= 160:
        head_model = 'austinman'
        slice_crop_size = slice_crop_size_austinman
    elif int(test_id) > 160 and int(test_id) <= 200:
        head_model = 'austinwoman'
        slice_crop_size = slice_crop_size_austinwoman
    else:
        raise ValueError('test set id is out of range in given models')

    roi_mask_file = os.path.join(file_output_dir, 'roi_mask_%s.nii.gz' % head_model)
    crop_mask = find_crop_mask(roi_mask_file)

    sar_gt = gt[0, 0, :, :, :]

    for m, idx in zip(modality, range(num_modality)):
        src_filename = os.path.join(sar_output_dir, 'source_%s_data.%s' % (m, file_format))
        src_vol_uncrop = crop_mask.uncrop_with_margin(test_vol[0][idx, :, :, :], slice_crop_size)
        __save_sar_volume(src_vol_uncrop, src_filename, dataset_info['format'], 'float32', is_compressed = False)
        print('saved source %s data as nii.gz' % m)

    sar_filename = os.path.join(sar_output_dir, 'sar_ground_truth_data.%s' % file_format)
    sar_gt_uncrop = crop_mask.uncrop_with_margin(sar_gt, slice_crop_size)
    __save_sar_volume(sar_gt_uncrop, sar_filename, dataset_info['format'], 'float32', is_compressed=False)
    print('saved sar groundtruth data as nii.gz')

    for m, idx in zip(modality, range(num_modality)):
        src_filename = os.path.join(sar_output_dir, 'source_%s_data_normalized.%s' % (m, file_format))
        test_vol_normalized = normalize_image(test_vol[0][idx, :, :, :], [0, 1])
        src_vol_uncrop = crop_mask.uncrop_with_margin(test_vol_normalized, slice_crop_size)
        __save_sar_volume(src_vol_uncrop, src_filename, dataset_info['format'], 'float32', is_compressed = False)
        print('saved source %s data (normalized) as nii.gz' % m)

    sar_filename = os.path.join(sar_output_dir, 'sar_ground_truth_data_normalized.%s' % file_format)
    sar_gt_normalized = normalize_image(sar_gt, [0, 1])
    sar_gt_normalized_uncrop = crop_mask.uncrop_with_margin(sar_gt_normalized, slice_crop_size)
    __save_sar_volume(sar_gt_normalized_uncrop, sar_filename, dataset_info['format'], 'float32', is_compressed=False)
    print('saved sar groundtruth data (normalized) as nii.gz')

    sar_filename = os.path.join(sar_output_dir, 'sar_prediction_data_ensemble.%s' % file_format)
    pred_vol_ensemble_scaled = (pred_vol_ensemble + 1) / 2.0 # scale all pixels from [-1,1] to [0,1]
    pred_vol_ensemble_uncrop = crop_mask.uncrop_with_margin(pred_vol_ensemble_scaled, slice_crop_size)
    pred_vol_ensemble_uncrop = crop_mask.multiply(pred_vol_ensemble_uncrop)
    __save_sar_volume(pred_vol_ensemble_uncrop, sar_filename, dataset_info['format'], 'float32', is_compressed = False)
    print('saved sar prediction (ensemble) data (normalized) as nii.gz')

    for pred_vol, dim_label in zip(pred_vol_total, trn_dim):
        sar_filename = os.path.join(sar_output_dir, 'sar_prediction_data_%s.%s' % (dim_label, file_format))
        pred_vol_scaled = (pred_vol + 1) / 2.0  # scale all pixels from [-1,1] to [0,1]
        pred_vol_uncrop = crop_mask.uncrop_with_margin(pred_vol_scaled, slice_crop_size)
        __save_sar_volume(pred_vol_uncrop, sar_filename, dataset_info['format'], 'float32', is_compressed = False)
        print('saved sar data (normalized)(%s) as nii.gz' % dim_label)

    if np.sum(ensemble_rescaled.flatten()) != 0:
        sar_filename = os.path.join(sar_output_dir, 'sar_prediction_data_ensemble_slice_est_min_max.%s' % file_format)
        pred_vol_ensemble_slice_est_min_max_uncrop = crop_mask.uncrop_with_margin(ensemble_rescaled, slice_crop_size)
        pred_vol_ensemble_slice_est_min_max_uncrop = crop_mask.multiply(pred_vol_ensemble_slice_est_min_max_uncrop)
        __save_sar_volume(pred_vol_ensemble_slice_est_min_max_uncrop, sar_filename, dataset_info['format'], 'float32',
                          is_compressed=False)
        print('saved sar prediction (ensemble slice by slice rescaled to est min/max) data as nii.gz')

    if len(rescaled_out_total) != 1:
        for pred_vol_slice_est_min_max, dim_label in zip(rescaled_out_total, trn_dim):
            sar_filename = os.path.join(sar_output_dir, 'sar_prediction_data_slice_est_min_max_%s.%s' % (dim_label, file_format))
            pred_vol_slice_est_min_max_uncrop = crop_mask.uncrop_with_margin(pred_vol_slice_est_min_max, slice_crop_size)
            __save_sar_volume(pred_vol_slice_est_min_max_uncrop, sar_filename, dataset_info['format'], 'float32', is_compressed=False)
            print('saved sar data slice by slice rescaled to est min/max (%s) as nii.gz' % dim_label)

    if np.sum(ensemble_rescaled.flatten()) != 0:
        sar_filename = os.path.join(sar_output_dir, 'sar_prediction_data_ensemble_global_est_min_max.%s' % file_format)
        ensemble_global_rescaled = normalize_image(pred_vol_ensemble, [np.min(ensemble_rescaled.flatten()), np.max(ensemble_rescaled.flatten())])
        pred_vol_ensemble_global_est_min_max_uncrop = crop_mask.uncrop_with_margin(ensemble_global_rescaled, slice_crop_size)
        pred_vol_ensemble_global_est_min_max_uncrop = crop_mask.multiply(pred_vol_ensemble_global_est_min_max_uncrop)
        __save_sar_volume(pred_vol_ensemble_global_est_min_max_uncrop, sar_filename, dataset_info['format'], 'float32',
                          is_compressed=False)
        print('saved sar prediction (globally rescaled to est min/max) data as nii.gz')

    if len(rescaled_out_total) != 1:
        for pred_vol, pred_vol_slice_est_min_max, dim_label in zip(pred_vol_total, rescaled_out_total, trn_dim):
            sar_filename = os.path.join(sar_output_dir, 'sar_prediction_data_global_est_min_max_%s.%s' % (dim_label, file_format))

            pred_vol_global_est_min_max = normalize_image(pred_vol, [np.min(pred_vol_slice_est_min_max.flatten()),
                                                                           np.max(pred_vol_slice_est_min_max.flatten())])

            pred_vol_global_est_min_max_uncrop = crop_mask.uncrop_with_margin(pred_vol_global_est_min_max, slice_crop_size)
            __save_sar_volume(pred_vol_global_est_min_max_uncrop, sar_filename, dataset_info['format'], 'float32', is_compressed=False)
            print('saved sar data globally rescaled to est min/max (%s) as nii.gz' % dim_label)

    sar_err_normalized = np.abs(np.subtract(sar_gt_normalized, pred_vol_ensemble_scaled))
    sar_err_filename = os.path.join(sar_output_dir, 'sar_err.%s' % file_format)
    sar_err_normalized_uncrop = crop_mask.uncrop_with_margin(sar_err_normalized, slice_crop_size)
    sar_err_normalized_uncrop = crop_mask.multiply(sar_err_normalized_uncrop)
    __save_sar_volume(sar_err_normalized_uncrop, sar_err_filename, dataset_info['format'], 'float32', is_compressed = True)
    print('saved sar error data (normalized) as nii.gz')

    if np.sum(ensemble_rescaled.flatten()) != 0:
        sar_err_slice_est_min_max = np.abs(np.subtract(sar_gt, ensemble_rescaled))
        sar_err_filename = os.path.join(sar_output_dir, 'sar_err_slice_est_min_max.%s' % file_format)
        sar_err_slice_est_min_max_uncrop = crop_mask.uncrop_with_margin(sar_err_slice_est_min_max, slice_crop_size)
        sar_err_slice_est_min_max_uncrop = crop_mask.multiply(sar_err_slice_est_min_max_uncrop)
        __save_sar_volume(sar_err_slice_est_min_max_uncrop, sar_err_filename, dataset_info['format'], 'float32', is_compressed = True)
        print('saved sar error data (slice by slice est. min/max) as nii.gz')

        sar_err_global_est_min_max = np.abs(np.subtract(sar_gt, ensemble_global_rescaled))
        sar_err_filename = os.path.join(sar_output_dir, 'sar_err_global_est_min_max.%s' % file_format)
        sar_err_global_est_min_max_uncrop = crop_mask.uncrop_with_margin(sar_err_global_est_min_max, slice_crop_size)
        sar_err_global_est_min_max_uncrop = crop_mask.multiply(sar_err_global_est_min_max_uncrop)
        __save_sar_volume(sar_err_global_est_min_max_uncrop, sar_err_filename, dataset_info['format'], 'float32', is_compressed = True)
        print('saved sar error data (global est. min/max) as nii.gz')

    # Create a dictionary
    save_mat = False
    if save_mat:
        dict_norm = {}
        dict_norm['pred'] = pred_vol_ensemble_uncrop
        dict_norm['gt'] = sar_gt_normalized_uncrop
        dict_norm['err'] = sar_err_normalized_uncrop
        sio.savemat(os.path.join(sar_output_dir, 'sar_prediction_normalized.mat'), dict_norm)

        if np.sum(ensemble_rescaled.flatten()) != 0:
            dict_slice = {}
            dict_slice['pred'] = pred_vol_ensemble_slice_est_min_max_uncrop
            dict_slice['gt'] = sar_gt_uncrop
            dict_slice['err'] = sar_err_slice_est_min_max_uncrop
            sio.savemat(os.path.join(sar_output_dir, 'sar_prediction_slice_est_min_max.mat'), dict_slice)

            dict_global = {}
            dict_global['pred'] = pred_vol_ensemble_global_est_min_max_uncrop
            dict_global['gt'] = sar_gt_uncrop
            dict_global['err'] = sar_err_global_est_min_max_uncrop
            sio.savemat(os.path.join(sar_output_dir, 'sar_prediction_global_est_min_max.mat'), dict_global)
        print('saved sar prediction and error data as mat')


def save_volume_interposed_fastigial(gen_conf, train_conf, test_conf, volume, prob_volume, test_fname, test_patient_id, seg_label,
                    file_output_dir):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    path = dataset_info['path']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    target = dataset_info['target']

    approach = train_conf['approach']
    loss = train_conf['loss']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    data_augment = train_conf['data_augment']
    preprocess_trn = train_conf['preprocess']
    preprocess_tst = test_conf['preprocess']

    modality = dataset_info['image_modality']
    modality_idx = []
    for m in modality: # the order of modality images should be fixed due to the absolute path
        if m == 'B0':
            modality_idx.append(0)
        if m == 'T1':
            modality_idx.append(1)
        if m == 'LP':
            modality_idx.append(2)
        if m == 'FA':
            modality_idx.append(2)

    if data_augment == 1:
        data_augment_label = '_mixup'
    elif data_augment == 2:
        data_augment_label = '_datagen'
    elif data_augment == 3:
        data_augment_label = '_mixup+datagen'
    else:
        data_augment_label = ''

    seg_output_dir = file_output_dir + '/' + test_patient_id + '/' + path + '/'
    if not os.path.exists(seg_output_dir):
        os.makedirs(os.path.join(file_output_dir, test_patient_id, path))

    crop_test_img_path = file_output_dir + '/' + test_patient_id + '/' \
                         + crop_tst_img_pattern[modality_idx[0]].format(test_patient_id)
    crop_test_data = read_volume_data(crop_test_img_path)

    # split side (in LPI)
    # left_mask, right_mask = compute_side_mask(volume.shape, crop_test_data)
    # left_vol = np.multiply(left_mask, volume)
    # right_vol = np.multiply(right_mask, volume)

    # postprocessing
    if 'interposed' in target or 'fastigial' in target:
        min_voxels = 0
    else:
        min_voxels = 50

    # left_vol_refined = postprocess(left_vol,min_voxels)
    # right_vol_refined = postprocess(right_vol, min_voxels)

    # save the cropped/refined result
    out_filename_crop = seg_output_dir + seg_label + '_seg_crop_' + approach + '_' + loss + '_' + \
                             str(patch_shape) + '_' + str(extraction_step) + data_augment_label + \
                             '_preproc_trn_opt_' + str(preprocess_trn) + '_preproc_tst_opt_' + \
                             str(preprocess_tst)

    # out_filename_crop_right = seg_output_dir + 'right_' + seg_label + '_seg_crop_' + approach + '_' + loss + '_' + \
    #                           str(patch_shape) + '_' + str(extraction_step) + data_augment_label + \
    #                           '_preproc_trn_opt_' + str(preprocess_trn) + '_preproc_tst_opt_' + \
    #                           str(preprocess_tst)

    print(out_filename_crop)
    # print(out_filename_crop_right)

    __save_volume(volume, crop_test_data, out_filename_crop, dataset_info['format'],
                  is_compressed=True)
    # __save_volume(right_vol_refined, crop_test_data, out_filename_crop_right, dataset_info['format'],
    #               is_compressed=True)

    # save probability map for background and foreground
    idx = 0
    # if seg_label == 'interposed' or 'fastigial':
    if gen_conf['num_classes'] > 2:
        class_name_lst = ['bg', 'interposed', 'fastigial']
    else:
        class_name_lst = ['bg', 'fg']

    for class_name in class_name_lst:
        out_filename_crop_prob_map = seg_output_dir + class_name + '_' + seg_label + '_prob_map_crop_' + approach + '_' + \
                                     loss + '_' + str(patch_shape) + '_' + str(extraction_step) + \
                                     data_augment_label + '_preproc_trn_opt_' + str(preprocess_trn) + \
                                     '_preproc_tst_opt_' + str(preprocess_tst)
        print(out_filename_crop_prob_map)
        __save_volume(prob_volume[:, :, :, idx], crop_test_data, out_filename_crop_prob_map,
                      dataset_info['format'], is_compressed=True)
        idx += 1

    # uncrop segmentation only (left and right)
    test_roi_mask_file = file_output_dir + test_roi_mask_pattern.format(test_patient_id)
    test_crop_mask = find_crop_mask(test_roi_mask_file)
    vol_uncrop = test_crop_mask.uncrop(volume)
    # right_vol_uncrop = test_crop_mask.uncrop(right_vol_refined)

    image_data = read_volume_data(test_fname[0])

    # save the uncropped result
    out_filename = seg_output_dir + seg_label + '_seg_' + approach + '_' + loss + '_' + \
                        str(patch_shape) + '_' + str(extraction_step) + data_augment_label + \
                        '_preproc_trn_opt_' + str(preprocess_trn) + '_preproc_tst_opt_' + \
                        str(preprocess_tst)

    # out_filename_right = seg_output_dir + 'right_' + seg_label + '_seg_' + approach + '_' + loss + '_' + \
    #                      str(patch_shape) + '_' + str(extraction_step) + data_augment_label + \
    #                      '_preproc_trn_opt_' + str(preprocess_trn) + '_preproc_tst_opt_' + \
    #                      str(preprocess_tst)

    print(out_filename)
    #print(out_filename_right)

    __save_volume(vol_uncrop, image_data, out_filename, dataset_info['format'], is_compressed = True)
    #__save_volume(right_vol_uncrop, image_data, out_filename_right, dataset_info['format'], is_compressed=True)


def save_volume_4d(gen_conf, train_conf, test_conf, volume, filename_ext, case_idx) :
    dataset = train_conf['dataset']
    approach = train_conf['approach']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    root_path = gen_conf['root_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = root_path + gen_conf['dataset_path']
    results_path = root_path + gen_conf['results_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    folder_names = dataset_info['folder_names']
    data_augment = train_conf['data_augment']
    preprocess_trn = train_conf['preprocess']
    preprocess_tst = test_conf['preprocess']

    for t in range(len(volume)):
        print(volume[t].shape)
        volume_tmp = np.zeros(volume[t].shape + (1, ))
        volume_tmp[:, :, :, 0] = volume[t]
        volume[t] = volume_tmp
        print(volume[t].shape)

        # it should be 3T test image (not 7T label)
        #data_filename = dataset_path + path + pattern[0].format(folder_names[0], case_idx)
        file_dir = dataset_path + path + folder_names[0]
        data_filename = file_dir + '/' + filename_ext
        print(data_filename)

        file_output_dir = results_path + path + folder_names[0]
        if not os.path.exists(file_output_dir):
            os.makedirs(os.path.join(results_path, path, folder_names[0]))

        image_data = read_volume_data(data_filename)
        org_image_volume = image_data.get_data()
        print(org_image_volume.shape)
        min_vox_value = np.min(org_image_volume)
        print(min_vox_value)

        if np.size(volume[t].shape) != np.size(org_image_volume.shape):
            org_image_volume_tmp = np.zeros(org_image_volume.shape + (1,))
            org_image_volume_tmp[:, :, :, 0] = org_image_volume
            org_image_volume = org_image_volume_tmp

        volume[t] = np.multiply(volume[t], org_image_volume != min_vox_value)
        volume[t][org_image_volume != min_vox_value] = volume[t][org_image_volume != min_vox_value] + 1

        if data_augment == 1:
            data_augment_label = '_mixup'
        elif data_augment == 2:
            data_augment_label = '_datagen'
        elif data_augment == 3:
            data_augment_label = '_mixup+datagen'
        else:
            data_augment_label = ''

        out_filename = results_path + path + pattern[1].format(folder_names[0], str(case_idx) + '_' + approach + '_' +
                                                               str(patch_shape) + '_' + str(extraction_step) +
                                                               data_augment_label + '_preproc_trn_opt_' +
                                                               str(preprocess_trn) + '_preproc_tst_opt_' +
                                                               str(preprocess_tst) +'_before_label_mapping_' +
                                                               'time_point_' + str(t+1))

        # out_filename = results_path + pattern[1].format(path.replace('/',''), str(case_idx) + '_' + approach + '_' +
        #                                                 str(extraction_step) + '_' + 'before_label_mapping')
        print(out_filename)
        __save_volume(volume[t], image_data, out_filename, dataset_info['format'], is_compressed = True)
        #label_mapper = {0 : 0, 1 : 10, 2 : 150, 3 : 250}
        label_mapper = {0: 0, 1: 0, 2: 10, 3: 150, 4: 250}
        for key in label_mapper.keys() :
            volume[t][volume[t] == key] = label_mapper[key]

        out_filename = results_path + path + pattern[1].format(folder_names[0], str(case_idx) + '_' + approach + '_' +
                                                               str(patch_shape) + '_' + str(extraction_step) +
                                                               data_augment_label + '_preproc_trn_opt_' +
                                                               str(preprocess_trn) + '_preproc_tst_opt_' +
                                                               str(preprocess_tst) + 'time_point_' + str(t+1))

        print(out_filename)
        __save_volume(volume[t], image_data, out_filename, dataset_info['format'], is_compressed = True)


def save_volume(gen_conf, train_conf, volume, case_idx) :
    dataset = train_conf['dataset']
    approach = train_conf['approach']
    extraction_step = train_conf['extraction_step']
    root_path = gen_conf['root_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = root_path + gen_conf['dataset_path']
    results_path = root_path + gen_conf['results_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    inputs = dataset_info['inputs']

    if dataset == 'iSeg2017' or dataset == 'IBSR18':
        volume_tmp = np.zeros(volume.shape + (1, ))
        volume_tmp[:, :, :, 0] = volume
        volume = volume_tmp

    data_filename = dataset_path + path + pattern.format(case_idx, inputs[-1])
    image_data = read_volume_data(data_filename)

    volume = np.multiply(volume, image_data.get_data() != 0)

    if dataset == 'iSeg2017':
        volume[image_data.get_data() != 0] = volume[image_data.get_data() != 0] + 1

        label_mapper = {0 : 0, 1 : 10, 2 : 150, 3 : 250}
        for key in label_mapper.keys() :
            volume[volume == key] = label_mapper[key]

    file_output_dir = results_path + path
    if not os.path.exists(file_output_dir):
        os.makedirs(os.path.join(results_path, path))

    out_filename = file_output_dir + pattern.format(case_idx, approach + ' - ' + str(extraction_step))

    __save_volume(volume, image_data, out_filename, dataset_info['format'], is_compressed = True)


def __save_volume(volume, image_data, filename, format, is_compressed) :
    img = None
    #max_bit = np.ceil(np.log2(np.max(volume.flatten())))
    if format in ['nii', 'nii.gz'] :
        if is_compressed:
            img = nib.Nifti1Image(volume.astype('uint8'), image_data.affine)
        else:
            img = nib.Nifti1Image(volume, image_data.affine)
    elif format == 'analyze':
        if is_compressed:
            # labels were assigned between 0 and 255 (8bit)
            img = nib.analyze.AnalyzeImage(volume.astype('uint8'), image_data.affine)
        else:
            img = nib.analyze.AnalyzeImage(volume, image_data.affine)
        #img = nib.analyze.AnalyzeImage(volume.astype('uint' + max_bit), image_data.affine)
        #img.set_data_dtype('uint8')
    nib.save(img, filename)


def __save_sar_volume(volume, filename, format, data_type, is_compressed) :
    img = None
    if format in ['nii', 'nii.gz'] :
        if is_compressed:
            img = nib.Nifti1Image(volume.astype(data_type), np.eye(4))  #'uint8'
        else:
            img = nib.Nifti1Image(volume, np.eye(4))
    elif format == 'analyze':
        if is_compressed:
            # labels were assigned between 0 and 255 (8bit)
            img = nib.analyze.AnalyzeImage(volume.astype(data_type), np.eye(4)) #'uint8'
        else:
            img = nib.analyze.AnalyzeImage(volume, np.eye(4))
    nib.save(img, filename)


def __create_stl(str_file, str_file_out_path):
    nii_file = nib.load(str_file)
    structures_v, structures_f = generate_structures_surface(str_file, threshold=0.1)
    structures_v_tr = apply_image_orientation_to_stl(np.transpose(structures_v) + 1, nii_file)
    structures_v_tr = np.transpose(structures_v_tr)
    write_stl(str_file_out_path, structures_v_tr, structures_f)


def save_patches(gen_conf, train_conf, patch_data, case_name):

    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    folder_names = dataset_info['folder_names']
    root_path = gen_conf['root_path']
    patches_path = gen_conf['patches_path']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    num_classes = gen_conf['num_classes']
    multi_output = gen_conf['multi_output']
    approach = train_conf['approach']
    data_augment = train_conf['data_augment']
    mode = gen_conf['validation_mode']
    preprocess_trn = train_conf['preprocess']
    loss = train_conf['loss']
    dimension = train_conf['dimension']

    if data_augment == 1:
        data_augment_label = 'mixup'
    elif data_augment == 2:
        data_augment_label = 'datagen'
    elif data_augment == 3:
        data_augment_label = 'mixup+datagen'
    else:
        data_augment_label = ''

    data_patches_path = root_path + patches_path + '/' + dataset + '/' + folder_names[0]
    if not os.path.exists(data_patches_path):
        os.makedirs(os.path.join(root_path, patches_path, dataset, folder_names[0]))

    if multi_output == 1:
        loss = loss[0] + '_' + loss[1]
    patches_filename = generate_output_filename(root_path + patches_path, dataset + '/' + folder_names[0],
                                                'mode_' + mode, case_name, approach, loss, 'dim_' + str(dimension),
                                                'n_classes_' + str(num_classes), str(patch_shape), str(extraction_step),
                                                data_augment_label, 'preproc_trn_opt_' + str(preprocess_trn) +
                                                '_training_samples', 'hdf5')

    print('Saving training samples (patches)...')
    with h5py.File(patches_filename, 'w') as f:
        _ = f.create_dataset("x_train", data=patch_data[0])
        _ = f.create_dataset("y_train", data=patch_data[1])
        _ = f.create_dataset("x_val", data=patch_data[2])
        _ = f.create_dataset("y_val", data=patch_data[3])


def read_patches(gen_conf, train_conf, case_name):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    folder_names = dataset_info['folder_names']
    root_path = gen_conf['root_path']
    patches_path = gen_conf['patches_path']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    num_classes = gen_conf['num_classes']
    multi_output = gen_conf['multi_output']
    approach = train_conf['approach']
    data_augment = train_conf['data_augment']
    mode = gen_conf['validation_mode']
    preprocess_trn = train_conf['preprocess']
    loss = train_conf['loss']
    dimension = train_conf['dimension']

    if data_augment == 1:
        data_augment_label = 'mixup'
    elif data_augment == 2:
        data_augment_label = 'datagen'
    elif data_augment == 3:
        data_augment_label = 'mixup+datagen'
    else:
        data_augment_label = ''

    data_patches_path = root_path + patches_path + '/' + dataset + '/' + folder_names[0]
    if not os.path.exists(data_patches_path):
        os.makedirs(os.path.join(root_path, patches_path, dataset, folder_names[0]))

    if multi_output == 1:
        loss = loss[0] + '_' + loss[1]
    patches_filename = generate_output_filename(root_path + patches_path, dataset + '/' + folder_names[0], 'mode_'+
                                                mode, case_name, approach, loss, 'dim_' + str(dimension), 'n_classes_'
                                                + str(num_classes), str(patch_shape), str(extraction_step),
                                                data_augment_label, 'preproc_trn_opt_' + str(preprocess_trn) +
                                                '_training_samples', 'hdf5')

    patches = []
    if os.path.isfile(patches_filename):
        print('Training samples (patches) already exist!')
        print('Loading saved training samples (patches)...')
        with h5py.File(patches_filename, 'r') as f:
            f_x_train = f['x_train']
            f_y_train = f['y_train']
            f_x_val = f['x_val']
            f_y_val = f['y_val']
            patches = [f_x_train[:], f_y_train[:], f_x_val[:], f_y_val[:]]

    return patches


def read_volume(filename):
    return read_volume_data(filename).get_data()


def read_volume_data(filename):
    return nib.load(filename)


def localize_target(file_output_dir, modality_idx, train_patient_lst, train_fname_lst, test_patient_id,
                    test_fname_modality_lst, label_fname_lst, img_pattern, img_resampled_name_pattern,
                    init_reg_mask_pattern, target, is_res_diff, roi_pos,
                    is_scaling=False, is_reg=True):

    # composite registration step (from one of training T1 MRI to test T1 MRI)
    # for initial localization to set the roi
    # load T1 data from test set

    # is_reg: True for PD088, PD090, False for PD091

    test_ref_img_path = test_fname_modality_lst[0]
    test_ref_image = BrainImage(test_ref_img_path, None)
    test_vol = test_ref_image.nii_data_normalized(bits=8)
    if np.size(np.shape(test_vol)) == 4:
        test_vol = test_vol[:, :, :, 0]
    mean_tst_vol = np.mean(test_vol.flatten())
    test_vol_size = np.array(test_vol.shape)

    # load T1 data from training set
    mean_trn_vol = []
    train_vol_size_lst = []
    dist_size_lst = []
    trn_vox_size_lst = []
    for train_fname in train_fname_lst:
        training_img_path = train_fname[0]
        training_img = BrainImage(training_img_path, None)
        train_vol = training_img.nii_data_normalized(bits=8)
        train_vol_size = np.array(train_vol.shape)
        trn_vox_size = training_img.nii_file.header.get_zooms()
        mean_trn_vol.append(np.mean(train_vol.flatten()))
        #dist_size = np.linalg.norm(train_vol_size - test_vol_size)
        train_vol_size_lst.append(train_vol_size)
        #dist_size_lst.append(dist_size)
        trn_vox_size_lst.append(trn_vox_size)
    abs_dist_mean = abs(np.array(mean_trn_vol - mean_tst_vol))
    print(test_vol_size)
    print(train_vol_size_lst)
    #print('dist_size_lst: %s' % str(dist_size_lst))
    #trn_sel_ind = np.where(dist_size_lst == np.min(dist_size_lst))[0][0]
    trn_sel_ind = np.where(abs_dist_mean == np.min(abs_dist_mean))[0][0]
    # trn_sel_ind = 6

    # trn_sel_img_path = os.path.join(dataset_path, train_patient_lst[trn_sel_ind], 'images', img_pattern[0])
    trn_sel_img_path = train_fname_lst[trn_sel_ind][0]
    trn_sel_img = BrainImage(trn_sel_img_path, None)

    # scaling of trn_sel_img to the size of test data
    scaling_ratio = np.divide(np.array(test_vol_size), np.array(train_vol_size_lst[trn_sel_ind]))
    if ((np.linalg.norm(scaling_ratio) < 2 / 3 or np.linalg.norm(scaling_ratio) > 1.5)) and is_scaling:
        print('scaling_ratio_btw_trn_img_and test_img: %s' % scaling_ratio)
        print('scaling selected training data (%s) to the size of test data (%s) with scaling ratio (norm)(%s)' %
              (train_patient_lst[trn_sel_ind], test_patient_id, np.linalg.norm(scaling_ratio)))
        trn_sel_img_scaling_path = os.path.join(file_output_dir, test_patient_id, train_patient_lst[trn_sel_ind] +
                                                '_7T_T1_brain_scaled.nii.gz')
        trn_sel_img_scaled_vol = trn_sel_img.rescale(scaling_ratio)
        trn_sel_img_scaled_out = nib.Nifti1Image(trn_sel_img_scaled_vol, trn_sel_img.nii_file.affine,
                                                 trn_sel_img.nii_file.header)
        trn_sel_img_scaled_out.to_filename(trn_sel_img_scaling_path)
        trn_sel_img = BrainImage(trn_sel_img_scaling_path, None)

    trn_vox_size_avg = np.mean(trn_vox_size_lst, 0)
    tst_vox_size = test_ref_image.nii_file.header.get_zooms()
    if np.size(tst_vox_size) == 4:
        tst_vox_size = tst_vox_size[:3]
    print(trn_vox_size_avg)
    print(tst_vox_size)
    vox_resampled = []
    for trn_vox_avg, tst_vox in zip(trn_vox_size_avg, tst_vox_size):
        if tst_vox * 0.7 > trn_vox_avg or tst_vox * 2.0 < trn_vox_avg:
            vox_resampled.append(trn_vox_avg)
            is_res_diff = True
        else:
            vox_resampled.append(tst_vox)
    if is_res_diff is True:
        for idx in modality_idx:
            test_img_resampled_path = os.path.join(file_output_dir, test_patient_id,
                                                   img_resampled_name_pattern[idx].format(test_patient_id))
            if not os.path.exists(test_img_resampled_path):
                test_img_path = test_fname_modality_lst[idx]
                print('resampling %s (%s) to %s' % (test_img_path, str(tst_vox_size), str(vox_resampled)))
                test_image = BrainImage(test_img_path, None)
                test_image.ResampleTo(test_img_resampled_path, new_affine=None, new_shape=None,
                                      new_voxel_size=[vox_resampled[0], vox_resampled[1], vox_resampled[2]],
                                      is_ant_resample=False)

        test_ref_img_path = os.path.join(file_output_dir, test_patient_id,
                                         img_resampled_name_pattern[0].format(test_patient_id))
        test_ref_image = BrainImage(test_ref_img_path, None)

        sel_trn_reg2_tst_image_name = train_patient_lst[trn_sel_ind] + '_reg2_' + test_patient_id + '_' + \
                                      _remove_ending(img_resampled_name_pattern[0].format(test_patient_id), '.nii.gz')
        roi_pos_scaled = np.multiply(roi_pos, [tst_vox_size[0] / vox_resampled[0], tst_vox_size[1] / vox_resampled[1],
                                        tst_vox_size[2] / vox_resampled[2]])

        roi_pos = []
        for roi_pos_scaled_elem in roi_pos_scaled:
            roi_pos_scaled_int = [int(i) for i in roi_pos_scaled_elem]
            roi_pos.append(roi_pos_scaled_int)

    else:
        sel_trn_reg2_tst_image_name = train_patient_lst[trn_sel_ind] + '_reg2_' + test_patient_id + '_' + \
                                      _remove_ending(img_pattern, '.nii.gz')

    transform_name = sel_trn_reg2_tst_image_name + '_' + 'composite_rigid_affine.txt'

    if is_reg:
        init_reg_dir = os.path.join(file_output_dir, test_patient_id, 'init_reg')
        if not os.path.exists(init_reg_dir):
            os.makedirs(init_reg_dir)
        composite_transform_path = os.path.join(init_reg_dir, transform_name)
        if not os.path.exists(composite_transform_path):
            # registration
            print("composite registration of a reference image of %s to a test image of %s" %
                  (train_patient_lst[trn_sel_ind], test_patient_id))

            trn_sel_img.composite_rigid_affine_registration_v2(file_output_dir,
                                                               init_reg_dir,
                                                               img_pattern,
                                                               train_patient_lst[trn_sel_ind],
                                                               test_patient_id,
                                                               test_ref_image,
                                                               sel_trn_reg2_tst_image_name,
                                                               do_translation_first=True)

    # load label images in train_patient_lst and setting roi
    init_mask_vol_lst = []
    init_mask_filepath_lst = []
    for side, idx in zip(['left', 'right'], [0, 1]):  # left + right
        if is_reg:
            label_filepath = label_fname_lst[trn_sel_ind][idx]
            print(label_filepath)
             # scaling of training labels
            if (np.linalg.norm(scaling_ratio) < 2 / 3 or np.linalg.norm(scaling_ratio) > 1.5) and is_scaling:
                label_img = BrainImage(label_filepath, None)
                label_img_scaled_vol = label_img.rescale(scaling_ratio)
                label_filepath = os.path.join(file_output_dir, test_patient_id, 'training_labels',
                                              train_patient_lst[trn_sel_ind] + '_' + target + '_' + side +
                                              '_label_scaled.nii.gz')
                label_img_scaled_out = nib.Nifti1Image(label_img_scaled_vol, label_img.nii_file.affine,
                                                       label_img.nii_file.header)
                label_img_scaled_out.to_filename(label_filepath)

            init_mask_filepath = os.path.join(init_reg_dir, init_reg_mask_pattern.format(side, target))
            if not os.path.exists(init_mask_filepath):
                cmd = "Apply_Transform.sh %s %s %s %s" % (label_filepath, test_ref_img_path,
                                                          init_mask_filepath, composite_transform_path)
                try:
                    subprocess.check_call(['bash ./scripts/' + cmd], shell=True)
                except subprocess.CalledProcessError as e:
                    print(e.returncode)
                    print(e.output)

            if os.path.exists(init_mask_filepath):
                print(init_mask_filepath)
                init_mask_data = read_volume_data(init_mask_filepath)
                init_mask_vol = init_mask_data.get_data()
                if np.size(np.shape(init_mask_vol)) == 4:
                    init_mask_vol_lst.append(init_mask_vol[:, :, :, 0])
                else:
                    init_mask_vol_lst.append(init_mask_vol)
            init_mask_filepath_lst.append(init_mask_filepath)
        else:
            init_mask_filepath_lst.append('')

    if os.path.exists(init_mask_filepath_lst[0]) and os.path.exists(init_mask_filepath_lst[1]):
        init_mask_merge_vol = init_mask_vol_lst[0] + init_mask_vol_lst[1]
        init_mask_data = read_volume_data(init_mask_filepath_lst[0])
    else:
        init_mask_merge_vol = np.array([])
        init_mask_data = np.array([])

    return init_mask_merge_vol, init_mask_data, test_ref_img_path, is_res_diff, roi_pos


def create_log(filepath, content, is_debug):
    f = None
    if is_debug == 1:
        print(content)
        with open(filepath, 'a') as f:
            f.write('\n' + content)
    return f