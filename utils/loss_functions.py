
from keras import backend as K
from keras.layers import Reshape, MaxPooling2D, MaxPooling3D
import numpy as np
from niftynet.layer.loss_segmentation import LossFunction
from utils.mathutils import measure_msd
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.distributions import percentile



def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy for keras (2.0.6). This lets you apply a weight to unbalanced classes.
    @url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
    @author: wassname

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


def dc_btw_dentate_interposed(dentate_prob_output, interposed_prob_output):

    def loss(y_true, y_pred):
        dentate_n_vox = K.flatten(dentate_prob_output)
        interposed_n_vox = K.flatten(interposed_prob_output)
        truepos = K.sum(dentate_n_vox * interposed_n_vox)
        fp_and_fn = K.sum(dentate_n_vox * (1 - interposed_n_vox)) + K.sum((1 - dentate_n_vox) * interposed_n_vox)
        dc = (2 * truepos) / (2 * truepos + fp_and_fn)
        return dc

    return loss


def local_sar_mean_loss(tf_g_sar_real, tf_gen_out):

    def loss(y_true, y_pred):

        mean_g_sar_real = K.mean(tf_g_sar_real, axis = (1,2), keepdims=False)
        mean_gen_out = K.mean(tf_gen_out, axis = (1,2), keepdims=False)

        return K.abs(K.mean((mean_g_sar_real-mean_gen_out)))

    return loss


def local_sar_peak_loss(tf_g_sar_real, tf_gen_out):

    def loss(y_true, y_pred):

        max_g_sar_real = K.max(tf_g_sar_real, axis = (1,2), keepdims=False)
        max_gen_out = K.max(tf_gen_out, axis = (1,2), keepdims=False)
        diff = max_g_sar_real-max_gen_out
        pos_idx = K.greater(diff, 0)
        peak_loss = K.mean(K.gather(K.flatten(diff), K.flatten(K.cast(pos_idx, dtype="int32"))))
        condition = K.greater(K.sum(K.cast(pos_idx, dtype="int32")), 0)

        return K.switch(condition, peak_loss, K.zeros_like(peak_loss))


    return loss


def local_sar_neg_loss(tf_gen_out):

    def loss(y_true, y_pred):

        min_gen_out = K.min(tf_gen_out, axis = (1,2), keepdims=False)
        neg_idx = K.greater(0, min_gen_out)
        neg_loss = K.abs(K.mean(K.gather(K.flatten(min_gen_out), K.flatten(K.cast(neg_idx, dtype="int32")))))
        condition = K.greater(K.sum(K.cast(neg_idx, dtype="int32")), 0)

        return K.switch(condition, neg_loss, K.zeros_like(neg_loss))

    return loss



def local_sar_min_loss(tf_g_sar_real, local_sar_min):

    def loss(y_true, y_pred):

        local_sar_min_float = K.cast(local_sar_min, dtype="float32")
        min_g_sar_real = K.min(tf_g_sar_real, axis = (1, 2), keepdims=False)
        diff_min = min_g_sar_real - local_sar_min_float[:,0]

        return K.mean(K.square(diff_min)) # average of difference between minimum intensities in a batch

    return loss


def local_sar_max_loss(tf_g_sar_real, local_sar_max):

    def loss(y_true, y_pred):

        local_sar_max_float = K.cast(local_sar_max, dtype="float32")
        max_g_sar_real = K.max(tf_g_sar_real, axis = (1, 2), keepdims=False)
        diff_max = max_g_sar_real - local_sar_max_float[:,0]

        return K.mean(K.square(diff_max)) # average of difference between maximum intensities in a batch

    return loss


def local_sar_min(tf_g_sar):

    def loss(y_true, y_pred):

        min_value = K.min(tf_g_sar, axis = (1, 2), keepdims=False)

        return K.mean(min_value) # average of minimum intensities in a batch

    return loss


def local_sar_max(tf_g_sar):

    def loss(y_true, y_pred):

        max_value = K.max(tf_g_sar, axis = (1, 2), keepdims=False)

        return K.mean(max_value) # average of maximum intensities in a batch

    return loss


def local_sar_est(local_sar_out):

    def loss(y_true, y_pred):

        local_sar_out_float = K.cast(local_sar_out, dtype="float32")

        return K.mean(local_sar_out_float[:,0]) # average of estimated intensities in a batch

    return loss


def mse(x, y):
    def loss(y_true, y_pred):
        return K.mean(K.square(x - y), axis=-1)
    return loss


def wasserstein_loss():

    def loss(y_true, y_pred):
        return K.mean(y_true * y_pred)

    return loss


def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


def psnr_loss():

    #import math

    def loss(y_true, y_pred):
        """
        PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
        It can be calculated as
        PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
        When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
        However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
        Thus we remove that component completely and only compute the remaining MSE component.
        """

        y_true = (K.flatten(y_true) + 1) / 2 # [-1~1] -> [0~1]
        y_pred = (K.flatten(y_pred) + 1) / 2 # [-1~1] -> [0~1]

        return -10. * np.log10(K.mean(K.square(y_pred - y_true)))


        # max_pixel = 1.0
        # y_true = (K.flatten(y_true) + 1) / 2 # [-1~1] -> [0~1]
        # y_pred = (K.flatten(y_pred) + 1) / 2 # [-1~1] -> [0~1]
        #
        # condition = tf.equal(y_true, y_pred)
        # psnr = 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
        #
        # return K.switch(condition, K.zeros_like(psnr), 1-psnr/100)

    return loss


def sar_tversky_focal_loss(tf_g_sar_real, tf_gen_out, thres=0.3, tv_alpha=0.3, tv_beta=0.7, f_gamma=2.0, f_alpha=0.25, w_focal=0.5):
    def loss(y_true, y_pred):

        sar_real_mask = K.cast(K.greater(tf_g_sar_real[:,:,0], thres), dtype="float32")
        sar_pred_mask = K.cast(K.greater(tf_gen_out[:,:,0], thres), dtype="float32")

        # tversky loss
        smooth = 1e-10
        y_true = K.flatten(sar_real_mask)
        y_pred = K.flatten(sar_pred_mask)
        tp = K.sum(y_true * y_pred)
        fp = y_pred * (1 - y_true)
        fn = (1 - y_pred) * y_true
        fp_and_fn = tv_alpha * K.sum(fp) + tv_beta * K.sum(fn)
        tversky = (tp + smooth) / ((tp + smooth) + fp_and_fn)
        tversky_loss = 1 - tversky

        # focal loss for binary label
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        focal_loss = -K.mean(f_alpha * K.pow(1. - pt_1, f_gamma) * K.log(pt_1)) \
                     - K.mean((1 - f_alpha) * K.pow(pt_0, f_gamma) * K.log(1. - pt_0))

        return (1 - w_focal) * tversky_loss + w_focal * focal_loss

    return loss


def tversky_loss(alpha = 0.3, beta = 0.7, smooth = 1e-10):
    # Ref: salehi et al., MICCAI MLMI 17, "tversky loss function for image segmentation using 3D FCDN"
    # -> the score is computed for each class separately and then summed
    # alpha=beta=0.5 : dice coefficient
    # alpha=beta=1   : tanimoto coefficient (also known as jaccard)
    # alpha+beta=1   : produces set of F*-scores
    # implemented by E. Moebel, 06/04/18

    def loss(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        truepos = K.sum(y_true * y_pred)
        fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
        tversky = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
        return 1-tversky

    return loss


def tversky_loss_multiclass(alpha = 0.3, beta = 0.7):
    # Ref: salehi et al., MICCAI MLMI 17, "tversky loss function for image segmentation using 3D FCDN"
    # -> the score is computed for each class separately and then summed
    # alpha=beta=0.5 : dice coefficient
    # alpha=beta=1   : tanimoto coefficient (also known as jaccard)
    # alpha+beta=1   : produces set of F*-scores
    # implemented by E. Moebel, 06/04/18
    def _tversky(y_true, y_pred, tv_alpha, tv_beta, classes_weight):

        # tversky loss for single class
        smooth = 1e-10

        ones = K.ones(K.shape(y_true))
        tp = tf.reduce_sum(y_true * y_pred, axis=[0, 1])
        fp = y_pred * (ones - y_true)
        fn = (ones - y_pred) * y_true
        fp_and_fn = tv_alpha * tf.reduce_sum(fp, axis=[0, 1]) + tv_beta * tf.reduce_sum(fn, axis=[0, 1])

        num = tp
        den = tp + fp_and_fn + smooth
        tversky = tf.reduce_sum(num / den * classes_weight)

        return 1 - tversky

    def loss(y_true, y_pred):

        # in build_training_set,
        # output_patch_shape = (np.prod(output_shape), num_classes)
        # y = np.vstack((y, np.zeros((N,) + output_patch_shape)))

        # get balanced weight alpha
        smooth = 1e-10

        w = K.sum(y_true, axis=(0, 1))  # for one-hot encoded input
        total_num = K.sum(w)
        classes_w_t1 = total_num / (w + smooth)
        sum_ = K.sum(classes_w_t1)
        classes_w_t2 = classes_w_t1 / (sum_ + smooth)
        tv_loss_multi = _tversky(y_true, y_pred, alpha, beta, classes_w_t2)

        return tv_loss_multi

    return loss


# def focal_loss(gamma = 2.0, alpha = 0.25):
#     def loss(y_true, y_pred):
#         # improve the stability of the focal loss and see issues 1 for more information
#
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#
#         return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - K.sum(
#             (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
#     return loss


def focal_loss_fixed(gamma=2., alpha=0.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def focal_loss(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss


# focal loss with multi label
def focal_loss_multiclass(gamma=2.):

    def _focal_loss(y_true, y_pred, gamma, classes_weight):

        classes_weight_grid = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
        classes_weight_grid += classes_weight

        # focal loss with no balanced weight which presented in paper function (4) https://arxiv.org/pdf/1708.02002.pdf
        zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
        one_minus_p = array_ops.where(tf.greater(y_true, zeros), y_true - y_pred, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(y_pred, 1e-8, 1.0))

        alpha = array_ops.where(tf.greater(y_true, zeros), classes_weight_grid, zeros)
        balanced_fl = tf.reduce_mean(tf.reduce_sum(alpha * FT, axis=2))

        return balanced_fl

    # classes_num contains sample number of each classes
    def loss(y_true, y_pred):
        # get balanced weight alpha
        smooth = 1e-10

        w = K.sum(y_true, axis=(0, 1))  # for one-hot encoded input
        total_num = K.sum(w)
        classes_w_t1 = total_num / (w + smooth)
        sum_ = K.sum(classes_w_t1)
        classes_w_t2 = classes_w_t1 / (sum_ + smooth)

        # get multi-class balanced focal loss
        focal_loss_multi = _focal_loss(y_true, y_pred, gamma, classes_w_t2)

        return focal_loss_multi

    return loss


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed


def tversky_focal_loss(tv_alpha=0.3, tv_beta=0.7, f_gamma=2.0, f_alpha=0.25, w_focal=0.5):
    def loss(y_true, y_pred):
        # tversky loss
        smooth = 1e-10
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        tp = K.sum(y_true * y_pred)
        fp = y_pred * (1 - y_true)
        fn = (1 - y_pred) * y_true
        fp_and_fn = tv_alpha * K.sum(fp) + tv_beta * K.sum(fn)
        tversky = (tp + smooth) / ((tp + smooth) + fp_and_fn)
        tversky_loss = 1 - tversky

        # focal loss for binary label
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        focal_loss = -K.mean(f_alpha * K.pow(1. - pt_1, f_gamma) * K.log(pt_1)) \
                     - K.mean((1 - f_alpha) * K.pow(pt_0, f_gamma) * K.log(1. - pt_0))

        return (1 - w_focal) * tversky_loss + w_focal * focal_loss

    return loss


def tversky_focal_loss_multiclass(tv_alpha=0.3, tv_beta=0.7, f_gamma=2.0, w_focal=0.5):

    def _tversky(y_true, y_pred, tv_alpha, tv_beta, classes_weight):

        # tversky loss
        smooth = 1e-10

        ones = K.ones(K.shape(y_true))
        #tp = K.sum(y_true * y_pred, (0, 1))
        tp = tf.reduce_sum(y_true * y_pred, axis=[0, 1])
        fp = y_pred * (ones - y_true)
        fn = (ones - y_pred) * y_true
        #fp_and_fn = tv_alpha * K.sum(fp, (0, 1)) + tv_beta * K.sum(fn, (0, 1))
        fp_and_fn = tv_alpha * tf.reduce_sum(fp, axis=[0, 1]) + tv_beta * tf.reduce_sum(fn, axis=[0, 1])

        num = tp
        den = tp + fp_and_fn + smooth
        #tversky = K.sum(num/den * classes_weight)
        tversky = tf.reduce_sum(num / den * classes_weight)

        return 1 - tversky

    def _focal_loss(y_true, y_pred, f_gamma, classes_weight):

        classes_weight_grid = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
        classes_weight_grid += classes_weight

        # focal loss with no balanced weight which presented in paper function (4) ICCV17, Focal Loss for Dense Object Detection
        zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
        one_minus_p = array_ops.where(tf.greater(y_true, zeros), y_true - y_pred, zeros)
        FT = -1 * (one_minus_p ** f_gamma) * tf.log(tf.clip_by_value(y_pred, 1e-8, 1.0))

        alpha = array_ops.where(tf.greater(y_true, zeros), classes_weight_grid, zeros)
        balanced_fl = tf.reduce_mean(tf.reduce_sum(alpha * FT, axis=2))
        #balanced_fl = K.mean(K.sum(alpha * FT, (2)))

        return balanced_fl

    def loss(y_true, y_pred):
        # multi-class generalized tversky loss

        # in build_training_set,
        # output_patch_shape = (np.prod(output_shape), num_classes)
        # y = np.vstack((y, np.zeros((N,) + output_patch_shape)))

        # get balanced weight alpha
        smooth = 1e-10

        w = K.sum(y_true, axis=(0, 1))  # for one-hot encoded input
        total_num = K.sum(w)
        classes_w_t1 = total_num / (w + smooth)
        sum_ = K.sum(classes_w_t1)
        classes_w_t2 = classes_w_t1 / (sum_ + smooth)

        # get multi-class balanced tversky loss
        tv_loss_multi = _tversky(y_true, y_pred, tv_alpha, tv_beta, classes_w_t2)

        # get multi-class balanced focal loss
        focal_loss_multi = _focal_loss(y_true, y_pred,  f_gamma, classes_w_t2)

        return (1 - w_focal) * tv_loss_multi + w_focal * focal_loss_multi

    return loss


# def select(n_class, opt):
#     if opt == 'categorical_crossentropy':
#         loss = 'categorical_crossentropy'
#     elif opt == 'weighted_categorical_crossentropy':
#         weights = np.array([1,8,8,8]) # 0: background, 1: CSF, 2: GM, 3: WM
#         loss = weighted_categorical_crossentropy(weights)
#     elif opt == 'dc':
#         loss = dc_loss(smooth=1)
#     elif opt == 'tversky':
#         loss = tversky_loss()
#     else:
#         loss = []
#         print('No found loss function')
#     return loss

def niftynet_builtin_loss(n_class, loss_func_opt, weight_map):
    def loss(y_true, y_pred):
        return LossFunction(n_class=n_class, loss_type=loss_func_opt).layer_op(y_pred, y_true, weight_map)
    return loss


def select(n_class, loss_func_opt, weight_map=None):

    if loss_func_opt == 'focal':
        return focal_loss_fixed(gamma=2.0, alpha=0.25)
    elif loss_func_opt == 'tversky':
        return tversky_loss(alpha=0.3, beta=0.7, smooth = 1e-10)
    elif loss_func_opt == 'dc_btw_dentate_interposed':
        return dc_btw_dentate_interposed()
    elif loss_func_opt == 'dc_loss':
        return tversky_loss(alpha=0.5, beta=0.5, smooth = 1e-10)
    elif loss_func_opt == 'focal_multiclass':
        return focal_loss_multiclass(gamma=2.0)
    elif loss_func_opt == 'tversky_multiclass':
        return tversky_loss_multiclass(alpha=0.3, beta=0.7)
    elif loss_func_opt == 'dc_multiclass':
        return tversky_loss_multiclass(alpha=0.5, beta=0.5)
    elif loss_func_opt == 'tversky_focal':
        return tversky_focal_loss(tv_alpha=0.3, tv_beta=0.7, f_gamma=2.0, f_alpha=0.25, w_focal=0.5)
    elif loss_func_opt == 'tversky_focal_multiclass':
        return tversky_focal_loss_multiclass(tv_alpha=0.3, tv_beta=0.7, f_gamma=2.0, w_focal=0.5)
    elif loss_func_opt == 'dice_focal':
        return tversky_focal_loss(tv_alpha=0.5, tv_beta=0.5, f_gamma=2.0, f_alpha=0.25, w_focal=0.5)
    elif loss_func_opt == 'dc_focal_multiclass':
        return tversky_focal_loss_multiclass(tv_alpha=0.5, tv_beta=0.5, f_gamma=2.0, w_focal=0.5)
    elif loss_func_opt == 'wasserstein':
        return wasserstein_loss()
    elif loss_func_opt == 'binary_crossentropy': # probability loss
        return loss_func_opt
    elif loss_func_opt == 'categorical_crossentropy': # probability loss
        return loss_func_opt
    elif loss_func_opt == 'mse': # probability loss
        return loss_func_opt
    elif loss_func_opt == 'mae': # probability loss
        return loss_func_opt
    elif loss_func_opt == 'msle': # probability loss
        return loss_func_opt
    else: # segmentation loss
        return niftynet_builtin_loss(n_class, loss_func_opt, weight_map)