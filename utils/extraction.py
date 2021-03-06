import numpy as np

from sklearn.feature_extraction.image import extract_patches as sk_extract_patches

def extract_patches(dimension, volume, patch_shape, extraction_step):
    actual_patch_shape = patch_shape
    actual_extraction_step = extraction_step

    if dimension == 2 :
        if len(actual_patch_shape) == 3 :
            actual_patch_shape = actual_patch_shape[:1] + (1, ) + actual_patch_shape[1:]
            actual_extraction_step = actual_extraction_step[:1] + (1, ) + actual_extraction_step[1:]
        else :
            actual_patch_shape = (1, ) + actual_patch_shape
            actual_extraction_step = (1, ) + actual_extraction_step

    patches = sk_extract_patches(
        volume,
        patch_shape=actual_patch_shape,
        extraction_step=actual_extraction_step)

    ndim = len(volume.shape)
    npatches = np.prod(patches.shape[:ndim])

    print(np.shape(patches))
    #print(np.shape(patches.reshape([-1] + list(patch_shape))))
    return patches.reshape((npatches, ) + patch_shape)


def extract_patches_rev(dimension, volume, patch_shape, extraction_step):
    'return original patch dimension'

    actual_patch_shape = patch_shape
    actual_extraction_step = extraction_step

    if dimension == 2:
        if len(actual_patch_shape) == 3:
            actual_patch_shape = actual_patch_shape[:1] + (1, ) + actual_patch_shape[1:]
            actual_extraction_step = actual_extraction_step[:1] + (1, ) + actual_extraction_step[1:]
        else :
            actual_patch_shape = (1, ) + actual_patch_shape
            actual_extraction_step = (1, ) + actual_extraction_step

    patches = sk_extract_patches(
        volume,
        patch_shape=actual_patch_shape,
        extraction_step=actual_extraction_step)

    print(np.shape(patches))

    return patches