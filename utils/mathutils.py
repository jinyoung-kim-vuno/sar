
import numpy as np
from skimage.transform import resize
from skimage import measure
from scipy  import ndimage
from stl import mesh
from scipy.linalg import sqrtm
from math import floor


def computeDice(autoSeg, groundTruth, label_mapper):
    """ Returns
    -------
    DiceArray : floats array

          Dice coefficient as a float on range [0,1].
          Maximum similarity = 1
          No similarity = 0 """

    #n_classes = int(np.max(groundTruth) + 1)

    DiceArray = []
    for key in label_mapper.keys():
        idx_Auto = np.where(autoSeg.flatten() == label_mapper[key])[0]
        idx_GT = np.where(groundTruth.flatten() == label_mapper[key])[0]

        autoArray = np.zeros(autoSeg.size, dtype=np.bool)
        autoArray[idx_Auto] = 1

        gtArray = np.zeros(autoSeg.size, dtype=np.bool)
        gtArray[idx_GT] = 1

        dsc = dice(autoArray, gtArray)

        # dice = np.sum(autoSeg[groundTruth==c_i])*2.0 / (np.sum(autoSeg) + np.sum(groundTruth))
        DiceArray.append(dsc)

    return DiceArray


def dice(im1, im2):
    """
    Computes the Dice coefficient
    ----------
    im1 : boolean array
    im2 : boolean array

    If they are not boolean, they will be converted.

    -------
    It returns the Dice coefficient as a float on the range [0,1].
        1: Perfect overlapping
        0: Not overlapping
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.size != im2.size:
        raise ValueError("Size mismatch between input arrays!!!")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return 1.0

    # Compute Dice
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def measure_cmd(struct1, struct2, pixdim):
    """
    :param struct1: ndarray of 3D image 1 (the result of nib.get_data())
    :param struct2: ndarray of 3D image 2 (the result of nib.get_data())
    :param pixdim: voxel spacing
    :return: average (Euclidan) distance between center of mass of the structures (in mm)
    """
    voxel_size = [pixdim[1], pixdim[0], pixdim[2]]

    struct1_cm = np.array(ndimage.measurements.center_of_mass(struct1))
    struct2_cm = np.array(ndimage.measurements.center_of_mass(struct2))

    cm_diff = (struct1_cm - struct2_cm) * voxel_size
    cm_dist = np.linalg.norm(cm_diff)
    return cm_dist, cm_diff


def measure_msd(struct1, struct2, pixdim):
    """
    :param struct1: ndarray of 3D image 1 (the result of nib.get_data())
    :param struct2: ndarray of 3D image 2 (the result of nib.get_data())
    :param pixdim:
    :return: averaged Euclidan distance between the structures surface points (in mm)
    """
    voxel_size = [pixdim[1], pixdim[0], pixdim[2]]

    verts1, faces1, normals1, values1 = measure.marching_cubes_lewiner(struct1, 0.5)
    verts2, faces2, normals2, values2 = measure.marching_cubes_lewiner(struct2, 0.5)
    min_s_dist_array = np.zeros(verts1.shape[0])
    for ind, surface_point in enumerate(verts1):
        min_s_dist_array[ind] = min(np.linalg.norm(((verts2 - surface_point) * voxel_size), axis=1))

    mean_surface_dist = np.mean(min_s_dist_array)
    return mean_surface_dist


def compute_statistics(input_data, num_modality):
    print("computing mean and std of training data...")
    mean = np.zeros((num_modality, ))
    std = np.zeros((num_modality, ))
    num_input = len(input_data)
    for modality in range(num_modality):
        modality_data = []
        for i in range(num_input):
            modality_data += np.array(input_data[i][0, modality]).flatten().tolist()
        mean[modality] = np.mean(np.array(modality_data))
        std[modality] = np.std(np.array(modality_data))

    print ('mean: ', mean, 'std: ', std)
    return mean, std


def measure_psnr(gt, pred):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""

    gt_array = np.array(gt, dtype=np.float32)
    pred_array = np.array(pred, dtype=np.float32)
    mse = np.mean((gt_array-pred_array) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(np.max(gt_array) / (np.sqrt(mse)))

# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(model, images, n_split=10, eps=1E-16):
    from keras.applications.inception_v3 import preprocess_input
    # enumerate splits of images/predictions
    scores = list()
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        # retrieve images
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset = images[ix_start:ix_end]
        # convert from uint8 to float32
        subset = subset.astype('float32')
        # scale images to the required size
        subset = scale_images(subset, (299, 299, 3))
        # pre-process images, scale to [-1,1]
        subset = preprocess_input(subset)
        # predict p(y|x)
        p_yx = model.predict(subset)
        # calculate p(y)
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = np.mean(sum_kl_d)
        # undo the log
        is_score = np.exp(avg_kl_d)
        # store
        scores.append(is_score)
    # average across images
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std


def measure_is_inception_v3(images):
    from keras.applications.inception_v3 import InceptionV3
    # load inception v3 model
    model = InceptionV3()

    #print('loaded', images.shape)
    # calculate inception score
    is_avg, _ = calculate_inception_score(model, images, n_split=10, eps=1E-16)
    #print('IS score', is_avg, is_std)
    return is_avg


# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# calculate frechet inception distance
def measure_fid_inception_v3(images1, images2):
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.inception_v3 import preprocess_input

    # prepare the inception v3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # convert integer to floating point values
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')
    # resize images
    images1 = scale_images(images1, (299, 299, 3))
    images2 = scale_images(images2, (299, 299, 3))
    #print('Scaled', images1.shape, images2.shape)

    # pre-process images
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)

    # calculate fid
    fid = calculate_fid(model, images1, images2)
    #print('FID: %.3f' % fid)

    return fid


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


def convert_arr_lst_to_elem(arr_lst):
    """
    This function extracts an element from array or combination of array and list
    :param arr_lst: array or combination of array and list
    :return: element
    """

    while type(arr_lst) is list:
        arr_lst = arr_lst[0]

    while type(arr_lst) is np.ndarray:
        arr_lst = arr_lst.tolist()
        while type(arr_lst) is list:
            arr_lst = arr_lst[0]

    return arr_lst


def transformPoint3d(x_array, y_array, z_array, trans):
    if np.array(x_array).size > 1:
        res = np.dot(np.transpose(np.array([x_array, y_array, z_array, np.ones(np.array(x_array).size)])),
                     np.transpose(trans))
    else:
        res = np.dot(np.array([x_array, y_array, z_array, 1]), np.transpose(trans))

    res = np.array(np.transpose(res))

    trans_array = [res[0], res[1], res[2]]

    return trans_array