#!/usr/bin/python

import os
import argparse

from config import general_config, training_config, test_config
from workflow.evaluate import run_evaluation_in_dataset

def main(args):
    #mode = sys.argv[1:][0]
    #mode = args.mode # simulation mode - 0: training + testing (leave one out or splitting data), 1: training only,
    # 2: testing only

    #To get reproducible results
    if args.is_set_random_seed == '1':
        import numpy as np
        np.random.seed(1) # NumPy
        import random
        random.seed(2) # Python
        from tensorflow import set_random_seed
        set_random_seed(3) # Tensorflow
        print('### numpy random seed fixed to 1 ###')

    run_evaluation_in_dataset(general_config, training_config, test_config, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run training and segmentation within FCN models')
    parser.add_argument('--mode', type=str, default='0', choices=['0','1','2'],
                        help='Set to 0, 1, or 2 to run k-fold (or loo), training only, or testing.')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='Enter "-2" for CPU only, "-1" for all GPUs available, '
                             'or a comma separated list of GPU id numbers ex: "0,1,4".')
    parser.add_argument('--num_of_gpu', type=int, default='-1',
                        help='Number of GPUs you have available for training. '
                             'If entering specific GPU ids under the --gpu_opt arg or if using CPU, '
                             'then this number will be inferred, else this argument must be included.')
    parser.add_argument('--num_classes', type=str, default='2',
                        help='Number of classes to be segmented ')
    parser.add_argument('--multi_output', type=str, default='0',
                        help='dual output option')
    parser.add_argument('--attention_loss', type=str, default='0',
                        help='add attention loss')
    parser.add_argument('--overlap_penalty_loss', type=str, default='0',
                        help='add overlap_penalty_loss to avoid overlap between dentate and interposed seg')
    parser.add_argument('--output_name', type=str, default='dentate_seg',
                        help='dual output name')
    parser.add_argument('--approach', type=str, default='unet',
                        help='FCN model used for training: cc_3d_fcn, unet, livianet, uresnet, deepmedic, '
                             'wnet, fc_capsnet, attention_unet, attention_se_fcn, pr_fb_net')
    parser.add_argument('--target', type=str, default='dentate',
                        help='target DCN region: dentate, interposed, fastigial')
    parser.add_argument('--activation', type=str, default='softmax',
                        help='last layer activation function - sigmoid: independent multi-label training, softmax: dependent single label training')
    parser.add_argument('--dataset', type=str, default='dcn_seg_dl',
                        help='Datasets for training: dcn_seg_dl, 3T7T, 3T7T_real, 3T7T_total, 3T+7T for '
                             'brain tissue segmentation, CBCT16, CBCT57, CT30 for cbct bone segmentation')
    parser.add_argument('--data_augment', type=str, default='0',
                        help='Data augmentation: 0: off, 1: mixup, 2.datagen, 3: mixup + datagen')
    parser.add_argument('--loss', type=str, default='Tversky',
                        help='loss function: Dice_Dense_NS (dice_dense_nosquare), Dice(dice), '
                             'GDSC (generalised_dice_loss), '
                             'WGDL (generalised_wasserstein_dice_loss; currently not working), '
                             'Tversky (tversky), CrossEntropy_Dense (cross_entropy_dense), '
                             'CrossEntropy (cross_entropy), SensSpec (sensitivity_specificity_loss)'),
    parser.add_argument('--exclusive_train', type=str, default='0',
                        help= 'if exclusive_train is 1, remove designated label during the training'),
    parser.add_argument('--exclude_label_num', type=str, default='1',
                        help= 'assigned label number(s) in a multi-label case'),
    parser.add_argument('--metric', type=str, default='acc',
                        help='metric for early stopping: acc for accuracy, acc_dc for dc, or loss'),
    parser.add_argument('--lamda', type=str, default='1.0,1.0,0.1,0.01',
                        help= 'weight for dentate, interposed, attnetion, and overlap penalty losses'),
    parser.add_argument('--preprocess_trn', type=str, default='2',
                        help='Preprocessing on training data: 0: no preprocessing, 1: standardization, 2: normalization, '
                             '3: normalization + standardization, 4: histogram matching (or normalization)' 
                             'to one training ref + standardization, 5: normalization + histogram matching)')
    parser.add_argument('--preprocess_tst', type=str, default='2',
                        help='Preprocessing on test data: 0: no preprocessing, 1: standardization, 2: normalization, '
                             '3: normalization + standardization, 4: histogram matching (or normalization)' 
                             'to one training ref + standardization, 5: normalization + histogram matching)')
    parser.add_argument('--num_k_fold', type=str, default='5', help='Number of k-fold cross validation')
    parser.add_argument('--batch_size', type=str, default='32', help='batch size in mini-batch mode in training/testing')
    parser.add_argument('--num_epochs', type=str, default='20', help='Number of epochs')
    parser.add_argument('--patience', type=str, default='10',
                        help='Number allowed to exit an epoch when val_loss decreases')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for learning')
    parser.add_argument('--initial_lr', type=str, default='0.001', help='initial learning rate of an optimizer')
    parser.add_argument('--is_set_random_seed', type=str, default='0',
                        help='set random seed to get reproducible results')
    parser.add_argument('--random_seed_num', type=str, default='1',
                        help='the number of random seed to set')
    parser.add_argument('--bg_discard_percentage', type=str, default='0',
                        help='remove in training samples if the number of label voxels is less than percentage')
    parser.add_argument('--normalized_range_min', type=str, default='-1',
                        help='normalized intensity range (min)')
    parser.add_argument('--normalized_range_max', type=str, default='1',
                        help='normalized intensity range (max)')
    parser.add_argument('--importance_spl', type=str, default='0',
                        help='computation to informative/important samples '
                             '(by sampling mini-batches from a distribution other than uniform) '
                             'thus accelerating the convergence')
    parser.add_argument('--oversampling', type=str, default='0',
                        help='oversampling samples of minor class to handle class imbalance')
    parser.add_argument('--trn_patch_size', type=str, default='32, 32, 32',
                        help='input patch size for training')
    parser.add_argument('--trn_output_size', type=str, default='32, 32, 32',
                        help='output patch size for training')
    parser.add_argument('--trn_step_size', type=str, default='9, 9, 9',
                        help='patch step size for training')
    parser.add_argument('--tst_patch_size', type=str, default='32, 32, 32',
                        help='input patch size for testing')
    parser.add_argument('--tst_output_size', type=str, default='32, 32, 32',
                        help='output patch size for testing')
    parser.add_argument('--tst_step_size', type=str, default='9, 9, 9',
                        help='patch step size for testing')
    parser.add_argument('--crop_margin', type=str, default='5, 5, 5',
                        help='margin to crop image')
    parser.add_argument('--threshold', type=str, default='0',
                        help='thresholding value for normalized probability map (votes from output patches) * foreground')
    parser.add_argument('--image_modality', type=str, default='T1,B0,FA',
                        help='used different contrast images or number of time-points for only 4d joint segmentation')
    parser.add_argument('--augment_sar_data', type=str, default='0,0', help='(option to augment sar prediction data, number of generation)')
    parser.add_argument('--trn_dim', type=str, default='axial,sagittal,coronal', help='data dimension to process')
    parser.add_argument('--ensemble_weight', type=str, default='0.6,0.2,0.2', help='weights for fusing prediction in each dimension')
    parser.add_argument('--continue_tr', type=str, default='0',
                        help='Continue training from checkpoints of a saved model')
    parser.add_argument('--is_measure', type=str, default='1',
                        help='measure CMD, MSD, DC and Volume if there exist ground truth labels')
    parser.add_argument('--is_trn_inference', type=str, default='0', help='inference option for training sets')
    parser.add_argument('--is_unseen_case', type=str, default='0',
                        help='test unseen cases with a pre-trained model after intial localization')
    parser.add_argument('--is_new_trn_label', type=str, default='0',
                        help='using new labels for pre-training')
    parser.add_argument('--new_label_path', type=str, default='/home/asaf/jinyoung/projects/results/dcn_seg_dl/'
                                                              'dentate_interposed_dilated_fc_densenet_dual_output_'
                                                              'only_dentate_only_interposed_softmax_attn_loss_0_1_'
                                                              'overlap_loss_updated_0_5_train_29_test_32',
                        help='a dataset folder')
    parser.add_argument('--folder_names', type=str, default='tha_baseline_test_t1_b0_fa',
                        help='a folder to save log, model, and results')
    parser.add_argument('--dataset_path', type=str, default='/home/asaf/jinyoung/projects/datasets/thalamus/',
                        help='a dataset folder')
    parser.add_argument('--is_fsl', type=str, default='1', help='few-shot learning for model style transfer to handle out of distribution')
    parser.add_argument('--num_model_samples', type=str, default='3',
                        help='the number of model samples for few-shot learning')
    parser.add_argument('--f_use', type=str, default='1', help='feedback network is used for training')
    parser.add_argument('--f_num_loop', type=str, default='3', help='the number of feedback loops')
    parser.add_argument('--f_spectral_norm', type=str, default='1', help='spectral normalization in feedback network')
    parser.add_argument('--g_network', type=str, default='fc_dense_contextnet', help='network for generator')
    parser.add_argument('--g_optimizer', type=str, default='Adam', help='optimizer for learning in generator')
    parser.add_argument('--g_initial_lr', type=str, default='0.001', help='initial learning rate of an optimizer in generator')
    parser.add_argument('--g_num_classes', type=str, default='2,2', help='number of classes in generator')
    parser.add_argument('--g_multi_output', type=str, default='1', help='dual output option in generator')
    parser.add_argument('--g_output_name', type=str, default='dentate_seg', help='dual output name in generator')
    parser.add_argument('--g_attention_loss', type=str, default='1', help='add attention loss in generator')
    parser.add_argument('--g_overlap_penalty_loss', type=str, default='',
                        help='add overlap_penalty_loss to avoid overlap between dentate and interposed seg in generator')
    parser.add_argument('--g_lamda', type=str, default='1.0,1.0,0.1,0.5',
                        help= 'weight for dentate, interposed, attnetion, and overlap penalty losses in generator'),
    parser.add_argument('--g_adv_loss_weight', type=str, default='0.05', help= 'weight for adversarial loss in generator'),
    parser.add_argument('--g_metric', type=str, default='loss_total',
                        help='metric for early stopping in generator: acc for accuracy, acc_dc for dc, or loss_total'),
    parser.add_argument('--g_loss', type=str, default='tversky_focal_multiclass,tversky_focal_multiclass',
                        help='loss function in generator: Dice_Dense_NS (dice_dense_nosquare), Dice(dice), '
                             'GDSC (generalised_dice_loss), '
                             'WGDL (generalised_wasserstein_dice_loss; currently not working), '
                             'Tversky (tversky), CrossEntropy_Dense (cross_entropy_dense), '
                             'CrossEntropy (cross_entropy), SensSpec (sensitivity_specificity_loss)'),
    parser.add_argument('--g_activation', type=str, default='softmax,softmax',
                        help='last layer activation function in generator - sigmoid: independent multi-label training, '
                             'softmax: dependent single label training')
    parser.add_argument('--g_spectral_norm', type=str, default='0', help='spectral normalization in generator')
    parser.add_argument('--g_trn_patch_size', type=str, default='32, 32, 32', help='input patch size for training in generator')
    parser.add_argument('--g_trn_output_size', type=str, default='32, 32, 32', help='output patch size for training in generator')
    parser.add_argument('--g_trn_step_size', type=str, default='5, 5, 5', help='patch step size for training in generator')
    parser.add_argument('--g_tst_patch_size', type=str, default='32, 32, 32', help='input patch size for testing in generator')
    parser.add_argument('--g_tst_output_size', type=str, default='32, 32, 32', help='output patch size for testing in generator')
    parser.add_argument('--g_tst_step_size', type=str, default='5, 5, 5', help='patch step size for testing in generator')

    parser.add_argument('--d_network', type=str, default='dilated_densenet', help='network for discriminator')
    parser.add_argument('--d_optimizer', type=str, default='Adam', help='optimizer for learning in discriminator')
    parser.add_argument('--d_initial_lr', type=str, default='0.001', help='initial learning rate of an optimizer in discriminator')
    parser.add_argument('--d_num_classes', type=str, default='2,2', help='number of classes in discriminator')
    parser.add_argument('--d_metric', type=str, default='loss_total',
                        help='metric for early stopping: acc for accuracy, acc_dc for dc, or loss_total in discriminator'),
    parser.add_argument('--d_loss', type=str, default='tversky_focal_multiclass,tversky_focal_multiclass',
                        help='loss function in discriminator: Dice_Dense_NS (dice_dense_nosquare), Dice(dice), '
                             'GDSC (generalised_dice_loss), '
                             'WGDL (generalised_wasserstein_dice_loss; currently not working), '
                             'Tversky (tversky), CrossEntropy_Dense (cross_entropy_dense), '
                             'CrossEntropy (cross_entropy), SensSpec (sensitivity_specificity_loss)'),
    parser.add_argument('--d_activation', type=str, default='softmax,softmax',
                        help='last layer activation function in discriminator - sigmoid: independent multi-label training, '
                             'softmax: dependent single label training')
    parser.add_argument('--d_spectral_norm', type=str, default='0', help='spectral normalization in discriminator')
    parser.add_argument('--d_trn_patch_size', type=str, default='32, 32, 32', help='input patch size for training in discriminator')
    parser.add_argument('--d_trn_output_size', type=str, default='4, 4, 4', help='output patch size for training in discriminator')
    parser.add_argument('--d_trn_step_size', type=str, default='5, 5, 5', help='patch step size for training in discriminator')
    parser.add_argument('--d_tst_patch_size', type=str, default='32, 32, 32', help='input patch size for testing in discriminator')
    parser.add_argument('--d_tst_output_size', type=str, default='4, 4, 4', help='output patch size for testing in discriminator')
    parser.add_argument('--d_tst_step_size', type=str, default='5, 5, 5', help='patch step size for testing in discriminator')

    args = parser.parse_args()

    # os.environ["CUDA_DEVICE_ORDER"]='1' # The GPU id to use, usually either "0" or "1"
    # os.environ["CUDA_VISIBLE_DEVICES"]='0'# use a single gpu (even if keras uses multiple gpus by default)

    if args.mode == '0':
        print('Running k-fold (or loo)')
    elif args.mode == '1':
        print('Running training only (or split training/testing)')
    else:
        print('Running testing only')

    if args.gpu_id == '-2':
        print('Using CPU only')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        num_of_gpu = int(args.num_of_gpu)
    elif args.gpu_id == '-1':
        print('Using all GPUs available')
        num_of_gpu = int(args.num_of_gpu)
        assert (num_of_gpu != -1), 'Use all GPUs option selected under --gpu_id, ' \
                                        'with this option the user MUST specify the number of GPUs ' \
                                        'available with the --num_of_gpu option.'
    else:
        print('Using specified GPU(s) - ID: ' + args.gpu_id)
        num_of_gpu = len(args.gpu_id.split(','))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if num_of_gpu > 1:
        batch_size = int(args.batch_size)
        assert batch_size >= num_of_gpu, 'Error: Must have at least as many items per batch as GPUs ' \
                                      'for multi-GPU training. For model parallelism instead of ' \
                                      'data parallelism, modifications must be made to the code.'

    args.num_of_gpu = num_of_gpu

    main(args)

# import sys, getopt
#
# def main(argv):
#    inputfile = ''
#    outputfile = ''
#    try:
#       opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
#    except getopt.GetoptError:
#       print('test.py -i <inputfile> -o <outputfile>')
#       sys.exit(2)
#    for opt, arg in opts:
#       if opt == '-h':
#          print('test.py -i <inputfile> -o <outputfile>')
#          sys.exit()
#       elif opt in ("-i", "--ifile"):
#          inputfile = arg
#       elif opt in ("-o", "--ofile"):
#          outputfile = arg
#    print('Input file is "', inputfile)
#    print('Output file is "', outputfile)
#
# if __name__ == "__main__":
#    main(sys.argv[1:])