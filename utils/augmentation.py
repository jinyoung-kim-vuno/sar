import numpy as np
from PIL import Image
from PIL import ImageOps
#from config import IMAGE_SIZE

######################################################################

#IMAGE_SIZE = 32

def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
        level: Level of the operation that will be in [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be
                scaled to level/PARAMETER_MAX.
    Returns:
        An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
        level: Level of the operation that will be in [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be
                scaled to level/PARAMETER_MAX.
    Returns:
        A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _, __):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _, __):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level, _):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level, _):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level, _):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level, image_size):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((image_size),
                            Image.AFFINE, (1, level, 0, 0, 1, 0),
                            resample=Image.BILINEAR)


def shear_y(pil_img, level, image_size):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((image_size),
                            Image.AFFINE, (1, 0, 0, level, 1, 0),
                            resample=Image.BILINEAR)


def translate_x(pil_img, level, image_size):
    level = int_parameter(sample_level(level), image_size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((image_size),
                            Image.AFFINE, (1, 0, level, 0, 1, 0),
                            resample=Image.BILINEAR)


def translate_y(pil_img, level, image_size):
    level = int_parameter(sample_level(level), image_size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((image_size),
                            Image.AFFINE, (1, 0, 0, 0, 1, level),
                            resample=Image.BILINEAR)


augmentations = [autocontrast,
                  equalize,
                  posterize,
                  rotate,
                  solarize,
                  shear_x,
                  shear_y,
                  translate_x,
                  translate_y]

