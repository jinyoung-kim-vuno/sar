#!/bin/bash

##!/usr/bin/env bash
# for icv segmentation

mode='0' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
gpu_id='0' # specific gpu id ('0,1') or -1: all available GPUs, -2: cpu only
num_of_gpu='1' # only if gpu_id==-1
num_classes='2' # 2 # 3
multi_output='0'
output_name='icv_seg'
attention_loss='0'
overlap_penalty_loss='0'
loss='tversky_focal' #implemented: dc, tversky, tversky_focal, tversky_focal_multiclass, dice_focal, focal  #niftynet built-in: 'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input, CrossEntropy
approach='fc_densenet' #'multires_net' #unet #livianet # fc_densenet_dilated # deepmedic # densenet_dilated
dataset='icv_seg_dl' #'tha_seg_dl' #dcn_seg_dl
preprocess_trn='2'
preprocess_tst='2'
num_k_fold='8' #LOO for 8 manual datasets
batch_size='8'
num_epochs='50' #'20'
patience='10' #'2'
optimizer='Adam'
initial_lr='0.001'
is_set_random_seed='0'
random_seed_num='None'
metric='acc' #loss #acc #acc_dc
activation='softmax' # last layer activation - sigmoid: independent (probability) multi-label training, softmax: dependent single label training
target='icv' #'tha' #dentate # dentate,interposed # 'icv'
image_modality='T1' #'B0,T1,LP,FA' #'B0' #'B0,T1,LP,FA'
trn_patch_size='32,32,32' #'32,32,32'
trn_output_size='32,32,32'
trn_step_size='16,16,16' #
tst_patch_size='32,32,32' #'32,32,32'
tst_output_size='32,32,32'
tst_step_size='16,16,16'
crop_margin='5,5,5'
bg_discard_percentage='0.2'
normalized_range_min='0.0'
normalized_range_max='1.0'
bg_value='0'
threshold='0' #0.5
continue_tr='0'
is_unseen_case='0'
is_measure='1'
is_new_trn_label='0' # 0: 29 manual labels, 1: 31 segmentation using a proposed network, 2: suit labels, 3: 29 manual labels + 31 seg (self-training?)
new_label_path=''
folder_names='manual' # if is_unseen_case on, then data name for test set
dataset_path='/mnt/home/jinyoung/data/icv_seg/dataset2.hdf5'


## for GAN: dentate/interposed (dilated densenet wo decoder)
#mode='0' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
#gpu_id='0' # specific gpu id ('0,1') or -1: all available GPUs, -2: cpu only
#num_of_gpu='4' # only if gpu_id==-1
#dataset='dcn_seg_dl' #'tha_seg_dl' #dcn_seg_dl
#preprocess_trn='2'
#preprocess_tst='2'
#num_k_fold='5' #5
#batch_size='8'
#num_epochs='50' #'50' #'20'
#patience='20' #'20' #'2'
#optimizer='Adam'
#initial_lr='0.001'
#num_classes='2,2' # 2 # 3
#multi_output='1'
#output_name='dentate_seg,interposed_seg' # dentate_interposed_seg for single output
#attention_loss='1'
#overlap_penalty_loss='1'
#metric='loss_total' #loss #acc #acc_dc
#loss='tversky_focal_multiclass,tversky_focal_multiclass' #implemented: dc, tversky, tversky_focal, tversky_focal_multiclass, dice_focal, focal  #niftynet built-in: 'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input, CrossEntropy
#activation='softmax,softmax'
#is_set_random_seed='0' # must be off for sar prediction due to randomly chosen training/validation samples and update_image_pool
#random_seed_num='1'
#exclusive_train='0' #if exclusive_train is '1', during training, remove designated label
#exclude_label_num='2,1' #exclude_label_num: 0. bg, 1.dentate, 2. interposed for exclusive training or multi-output, -1: nothing
#target='dentate,interposed' #'tha' #dentate # dentate,interposed
#image_modality='B0' #'B0,T1,LP,FA' #'B0' #'B0,T1,LP,FA'
#bg_discard_percentage='0.0'
#bg_value='0'
#trn_patch_size='32,32,32' #for livianet 27,27,27 (32,32,32); for deepmedic 48,48,48; for CAFP 64,64,64; for others 32,32,32
#trn_output_size='32,32,32' #for livianet 9,9,9 (14,14,14); for deepmedic 16,16,16; for others 32,32,32
#trn_step_size='9,9,9' #for livianet and deepmedic 5,5,5 (9,9,9); for others 15,15,15
#tst_patch_size='32,32,32' #for livianet 27,27,27 (32,32,32); for deepmedic 48,48,48; for CAFP 64,64,64; for others 32,32,32
#tst_output_size='32,32,32' #for livianet 9,9,9 (14,14,14); for deepmedic 16,16,16; for others 32,32,32
#tst_step_size='9,9,9' #for livianet and deepmedic 5,5,5 (9,9,9); for others 15,15,15
#importance_spl='0' # not complete
#oversampling='0' # not complete
#threshold='0' #0.5
#continue_tr='0'
#is_unseen_case='0'
#is_measure='1'
#is_new_trn_label='0' # 0: 29 manual labels, 1: 31 segmentation using a proposed network, 2: suit labels, 3: 29 manual labels + 31 seg (self-training?)
##new_label_path='/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_dilated_fc_densenet_dual_output_only_dentate_only_interposed_softmax_attn_loss_0_1_overlap_loss_updated_0_5_train_29_test_32'
##new_label_path='/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_unet_20_patience_5_folds_32_32_for_tmi1_random_seed_fixed_train_29_test_31'
#new_label_path='/home/asaf/jinyoung/projects/results/dcn_seg_dl/dcn_net_train_29_test_31' #'/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_dilated_fc_densenet_dual_output_only_dentate_only_interposed_softmax_attn_loss_0_1_overlap_loss_updated_0_5_train_29_test_32'
#folder_names='cgan_dcn_seg_test_3rd' #'cgan_dcn_seg_test' #'dentate_interposed_fc_densenet_ag_rev_2_folds_for_bs' # 'fc_densenet_ag_rev_2_folds_for_bs' #fc_densenet_dilated_ag_rev_2_folds_for_bs #tha_baseline_test_t1_b0_fa #fc_densenet_cafp_2_folds_for_bs # fc_densenet_cafp_2_folds_for_bs
#dataset_path='/home/shira-raid1/DBS_data/Cerebellum_Tools/'
#crop_margin='5,5,5'
### gan
#approach='cgan' #'multires_net' #unet #livianet # fc_densenet_dilated # deepmedic # densenet_dilated # attention_unet #cenet
## generator
#g_num_classes='2,2' # 2 # 3
#g_multi_output='1'
#g_output_name='dentate_seg,interposed_seg' # dentate_interposed_seg for single output
#g_attention_loss='1'
#g_overlap_penalty_loss='1'
#g_lamda='1.0,1.0,0.1,0.5' # weight for segmentation losses, attention loss, and overlap loss  - dentate/interposed/attention/overlap penalty
#g_adv_loss_weight='0.01' #
#g_metric='loss_total' #loss #acc #acc_dc
#g_loss='tversky_focal_multiclass,tversky_focal_multiclass' #implemented: dc, tversky, tversky_focal, tversky_focal_multiclass, dice_focal, focal  #niftynet built-in: 'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input, CrossEntropy
#g_activation='softmax,softmax' # last layer activation - sigmoid: independent (probability) multi-label training, softmax: dependent single label training
#g_trn_patch_size='32,32,32' #for livianet 50,50,50 (32,32,32; 27,27,27); for deepmedic 64,64,64 (48,48,48);
#g_trn_output_size='32,32,32' #for livianet 32,32,32 (14,14,14; 9,9,9); for deepmedic 32,32,32 (16,16,16);
#g_trn_step_size='9,9,9'
#g_tst_patch_size='32,32,32' #for livianet 50,50,50 (32,32,32; 27,27,27); for deepmedic 64,64,64 (48,48,48);
#g_tst_output_size='32,32,32' #for livianet 32,32,32 (14,14,14; 9,9,9); for deepmedic 32,32,32 (16,16,16);
#g_tst_step_size='9,9,9'
## discriminator
#d_num_classes='1'
#d_metric='loss_total' #loss #acc #acc_dc
#d_loss='binary_crossentropy'
#d_activation='sigmoid'
#d_trn_patch_size='32,32,32' #for livianet 50,50,50 (32,32,32; 27,27,27); for deepmedic 64,64,64 (48,48,48);
#d_trn_output_size='32,32,32' #for livianet 32,32,32 (14,14,14; 9,9,9); for deepmedic 32,32,32 (16,16,16);
#d_trn_step_size='9,9,9'
#d_tst_patch_size='32,32,32' #for livianet 50,50,50 (32,32,32; 27,27,27); for deepmedic 64,64,64 (48,48,48);
#d_tst_output_size='32,32,32' #for livianet 32,32,32 (14,14,14; 9,9,9); for deepmedic 32,32,32 (16,16,16);
#d_tst_step_size='9,9,9'

#
## for conditional GAN: SAR prediction
#mode='1' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
#gpu_id='0,1' #'0,1' # specific gpu id ('0,1') or -1: all available GPUs, -2: cpu only
#num_of_gpu='2' # only if gpu_id==-1
#dataset='sar' #'tha_seg_dl' #dcn_seg_dl
#preprocess_trn='2'
#preprocess_tst='2'
#num_k_fold='5' #5
#batch_size='8'
#num_epochs='100' #'50' #'20'
#patience='50' #'20' #'2' # 0: save the model every epoch without considering the minimum g_loss
#num_classes='1' # 2 # 3
#multi_output='0'
#output_name='sar_pred' # dentate_interposed_seg for single output
#attention_loss='0'
#overlap_penalty_loss='0'
#metric='loss_total' #loss #acc #acc_dc
#loss='mae' #implemented: dc, tversky, tversky_focal, tversky_focal_multiclass, dice_focal, focal  #niftynet built-in: 'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input, CrossEntropy
#activation='tanh' #'tanh'
#is_set_random_seed='0' # must be off for sar prediction due to randomly chosen training/validation samples and update_image_pool
#random_seed_num='1'
#exclusive_train='0' #if exclusive_train is '1', during training, remove designated label
#exclude_label_num='2,1' #exclude_label_num: 0. bg, 1.dentate, 2. interposed for exclusive training or multi-output, -1: nothing
#target='sar' #'tha' #dentate # dentate,interposed
#image_modality='B1_mag,Epsilon,Rho,Sigma' #'B1_real,B1_imag,Sigma' #'B1_mag' #'B1_real,B1_imag' #'B1','Epsilon','Rho','Sigma'
#augment_sar_data='0,1' # (on or off, number of generation (min. :2))
#trn_dim='sagittal,axial' # '': 3d, 'axial,coronal,sagittal': 2.5D
#ensemble_weight='0.2,0.8' #'0.2,0.2,0.6' # 0.6,0.2,0.2 for axial,sagittal,coronal slices
#bg_discard_percentage='0'
#normalized_range_min='-1.0'
#normalized_range_max='1.0'
#trn_patch_size='160,160,80' # 32,32,32
#trn_output_size='160,160,80' # 32,32,32
#trn_step_size='0,0,0' # 16,16,16
#tst_patch_size='160,160,80' # 32,32,32
#tst_output_size='160,160,80' # 32,32,32
#tst_step_size='0,0,0' # 16,16,16
#importance_spl='0' # not complete
#oversampling='0' # not complete
#threshold='0' #0.5
#continue_tr='0'
#is_unseen_case='0'
#is_measure='1'
#is_trn_inference='0'
#is_new_trn_label='0'
#new_label_path='None'
#folder_names='trn#24_7_duke+louis+austinman+austinwoman_trn_ella_tst' #'trn#24_7_ella+louis+austinman+austinwoman_trn_duke_tst' #'trn#24_8_ella+louis+austinman_trn_duke_tst' #'trn#24_7_duke+ella+louis_trn_austinman_tst' #'trn#24_7_ella+louis_trn_duke_tst' #'trn#24_7_duke+ella_trn_louis_tst' #'trn#24_7_duke+ella+louis_trn_austinman_tst' #'trn#24_7_ella+louis+austinman_trn_duke_tst' #'trn#24_8_duke+louis_trn_ella_tst' #'trn#24_8_ella+louis_trn_duke_tst' #'trn#24_7_duke+louis+austinman_trn_ella_tst' #'trn#24_8_duke+ella_trn_louis_tst' #'trn#24_7_duke+ella_trn_louis_tst'  #'trn#24_7_ella_40_test_duke_40_train' #'trn#24_7_duke_40_test_ella_40_train' #'sar_prediction_baseline_test_trn#24_7' #'sar_prediction_baseline_test_b1+_real_imag_sigma_fc_dense_contextnet_with_perceptual_peak_negative_loss_data_2_5D_data_aug_2_fixed_debug' #'sar_prediction_baseline_test_b1+_real_imag_fc_dense_contextnet_with_perceptual_peak_negative_loss_data_2_5D_data_aug_3' # #
#dataset_path='/home/asaf/jinyoung/projects/datasets/sar/'
#crop_margin='2,3,3,4,-20,-12' # for duke, ella, louis, austinman, austinwoman  #'2,3,2,3,-20,-12' #'6,7,6,7,-20,-12' #'6,7,6,6,-20,-12' # for duke, ella, louis #'11,10,11,12,-21,-14' for duke and ella  #'3,3,3,3,21,20' # crop size in each side (e.g., 0,0,0,0,40,40: no crop in x and y slices, crop 40 slices in each side of z axis
### gan
#approach='cgan_sar_multi_task' #'cgan_sar_multi_task' #'cgan_sar' #'cyclegan_sar2' #'multires_net' #unet #livianet # fc_densenet_dilated # deepmedic # densenet_dilated # attention_unet #cenet
## feedback network
#f_use='1' # on or off
#f_num_loop='3' #the number of feedback loops, if this is more than 5, there may be gradient explosion ('nan' loss) during training)
#f_spectral_norm='1'
## few-shot model transfer
#is_fsl='0' # on: few-shot learning (sar distribution/sar peak transfer)
#num_model_samples='3'
## generator
#g_network='fc_dense_contextnet' #'u_net' #u_net #fc_dense_contextnet #dilated_densenet
#g_optimizer='Adam' #'Adam', 'AdamWithWeightnorm'
#g_initial_lr='0.0001' #'0.0001' # '0.00005' according to feedback adversarial network (cvpr19) # others: '0.001'
#g_num_classes='1' # 2 # 3
#g_multi_output='0'
#g_output_name='sar_pred' # dentate_interposed_seg for single output
#g_attention_loss='0'
#g_overlap_penalty_loss='0'
#g_lamda='1.0,0.01,0.01,0.1,0.1,0.1,0.1,0.1,0.01'  #'1.0,0.5,0.3,0.1,0.001,0.001' #1.0,1.0,0.5,0.3,0.1,0.001,0.001 # generator L1 loss, perceptual losses (for 1st, 2nd, and 3rd layers), peak value loss, and negative value loss
#g_adv_loss_weight='0.1' # weight for adversarial loss (cgan: 0.01)
#g_metric='loss_total' #loss #acc #acc_dc
#g_loss='mae' #implemented: dc, tversky, tversky_focal, tversky_focal_multiclass, dice_focal, focal  #niftynet built-in: 'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input, CrossEntropy
#g_activation='tanh' # 'linear 'tanh' # last layer activation - sigmoid: independent (probability) multi-label training, softmax: dependent single label training
#g_spectral_norm='0'
#g_trn_patch_size='160,160,80' #for livianet 50,50,50 (32,32,32; 27,27,27); for deepmedic 64,64,64 (48,48,48);
#g_trn_output_size='160,160,80' #for livianet 32,32,32 (14,14,14; 9,9,9); for deepmedic 32,32,32 (16,16,16);
#g_trn_step_size='0,0,0'
#g_tst_patch_size='160,160,80' #for livianet 50,50,50 (32,32,32; 27,27,27); for deepmedic 64,64,64 (48,48,48);
#g_tst_output_size='160,160,80' #for livianet 32,32,32 (14,14,14; 9,9,9); for deepmedic 32,32,32 (16,16,16);
#g_tst_step_size='0,0,0'
## discriminator
#d_network='u_net' #'u_net' #'patch_gan' #'dilated_densenet' #patch_gan #dilated_densenet #u_net
#d_optimizer='Adam' #'Adam'
#d_initial_lr='0.0001' # for cyclegan: '0.0001' according to feedback adversarial network (cvpr19) # others: '0.001'
#d_num_classes='1'
#d_metric='loss_total' #loss #acc #acc_dc
#d_loss='mse' #'mse' #'binary_crossentropy' 'mse'
#d_activation='linear'  #'linear' for mse loss #'sigmoid'
#d_spectral_norm='1'
#d_trn_patch_size='160,160,80' #for livianet 50,50,50 (32,32,32; 27,27,27); for deepmedic 64,64,64 (48,48,48);
#d_trn_output_size='160,160,80' #'20,20,10' #for livianet 32,32,32 (14,14,14; 9,9,9); for deepmedic 32,32,32 (16,16,16);
#d_trn_step_size='0,0,0'
#d_tst_patch_size='160,160,80' #for livianet 50,50,50 (32,32,32; 27,27,27); for deepmedic 64,64,64 (48,48,48);
#d_tst_output_size='160,160,80' #'20,20,10' #for livianet 32,32,32 (14,14,14; 9,9,9); for deepmedic 32,32,32 (16,16,16);
#d_tst_step_size='0,0,0'



## for cycleGAN: SAR prediction (dilated densenet wo decoder)
#mode='0' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
#gpu_id='0,1' # specific gpu id ('0,1') or -1: all available GPUs, -2: cpu only
#num_of_gpu='2' # only if gpu_id==-1
#dataset='sar' #'tha_seg_dl' #dcn_seg_dl
#preprocess_trn='2'
#preprocess_tst='2'
#num_k_fold='5' #5
#batch_size='8'
#num_epochs='50' #'50' #'20'
#patience='20' #'20' #'2' # 0: save the model every epoch without considering the minimum g_loss
#num_classes='1' # 2 # 3
#multi_output='0'
#output_name='sar_pred' # dentate_interposed_seg for single output
#attention_loss='0'
#overlap_penalty_loss='0'
#metric='loss_total' #loss #acc #acc_dc
#loss='mae' #implemented: dc, tversky, tversky_focal, tversky_focal_multiclass, dice_focal, focal  #niftynet built-in: 'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input, CrossEntropy
#activation='tanh' #'tanh'
#is_set_random_seed='0' # must be off for sar prediction due to randomly chosen training/validation samples and update_image_pool
#random_seed_num='1'
#exclusive_train='0' #if exclusive_train is '1', during training, remove designated label
#exclude_label_num='2,1' #exclude_label_num: 0. bg, 1.dentate, 2. interposed for exclusive training or multi-output, -1: nothing
#target='sar' #'tha' #dentate # dentate,interposed
#image_modality='B1_imag' #'B1_real,B1_imag,Epsilon,Rho,Sigma' #Epsilon,Rho,Sigma' #'B1_mag' #'B1_real,B1_imag,Sigma' #'B1_mag' #'B1_real,B1_imag' #'B1','Epsilon','Rho','Sigma'
#augment_sar_data='0,1' # (on or off, number of generation (min. :2))
#trn_dim='coronal,sagittal,axial' # '': 3d, 'coronal,sagittal,axial'': 2.5D
#ensemble_weight='0.2,0.2,0.6' # 0.6,0.2,0.2 for axial,sagittal,coronal slices
#bg_discard_percentage='0'
#normalized_range_min='-1.0'
#normalized_range_max='1.0'
#trn_patch_size='160,160,80' # 32,32,32
#trn_output_size='160,160,80' # 32,32,32
#trn_step_size='0,0,0' # 16,16,16
#tst_patch_size='160,160,80' # 32,32,32
#tst_output_size='160,160,80' # 32,32,32
#tst_step_size='0,0,0' # 16,16,16
#importance_spl='0' # not complete
#oversampling='0' # not complete
#threshold='0' #0.5
#continue_tr='0'
#is_unseen_case='0'
#is_measure='1'
#is_trn_inference='1'
#is_new_trn_label='0'
#new_label_path='None'
#folder_names='sar_prediction_baseline_test_trn#23_6' #'sar_prediction_baseline_test_b1+_real_imag_sigma_fc_dense_contextnet_with_perceptual_peak_negative_loss_data_2_5D_data_aug_2_fixed_debug' #'sar_prediction_baseline_test_b1+_real_imag_fc_dense_contextnet_with_perceptual_peak_negative_loss_data_2_5D_data_aug_3' # #
#dataset_path='/home/asaf/jinyoung/projects/datasets/sar/'
#crop_margin='11,10,11,12,-21,-14' #'3,3,3,3,21,20' # crop size in each side (e.g., 0,0,0,0,40,40: no crop in x and y slices, crop 40 slices in each side of z axis
### gan
#approach='composite_cyclegan_sar' #'multi_cyclegan_sar' #'single_cyclegan_sar' #'cgan_sar' #'cyclegan_sar2' #'multires_net' #unet #livianet # fc_densenet_dilated # deepmedic # densenet_dilated # attention_unet #cenet
#feedback='0,3'
## generator
#g_network='fc_dense_contextnet' #'u_net' #u_net #fc_dense_contextnet #dilated_densenet
#g_optimizer='Adam'
#g_initial_lr='0.0001' # for cyclegan: '0.0001' according to feedback adversarial network (cvpr19) # others: '0.001'
#g_num_classes='1' # 2 # 3
#g_multi_output='0'
#g_output_name='sar_pred' # dentate_interposed_seg for single output
#g_attention_loss='0'
#g_overlap_penalty_loss='0'
#g_lamda='1.0,1.0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.01'  #'0.5,0.001,1.0,0.5,0.3,0.1,0.001,0.001,0.001,0.0005,1.0,0.001,0.5,0.3,0.1,0.001,0.001,0.001' #'0.5,0.1,1.0,0.5,0.3,0.1,0.001,0.001,0.5,1.0,0.1,0.5,0.3,0.1,0.001,0.001' # composite cyclegan_sar '0.001,1.0,0.5,0.3,0.1,0.5,0.3,0.1,0.001,0.1,0.001' # single_cyclegan_sar (fwd/bwd) '0.5,0.1,1.0,0.5,0.3,0.1,0.001,0.001,0.5,1.0,0.1,0.5,0.3,0.1,0.001,0.001'   ; multi_cyclegan_sar (3 to 1) '1.0,0.001,0.001,0.001,0.5,0.3,0.1,0.001,0.001,0.001,0.001,1.0,0.001,0.5,0.3,0.1'  #cyclegan_sar: '1.0,0.01,0.01,1.0,0.5,0.3,0.1,0.001,0.001,0.01,1.0,1.0,0.01,0.5,0.3,0.1' #cyclegan_sar2: '1.0,0.01,0.01,1.0,0.5,0.3,0.1,0.5,0.3,0.1,0.001,0.001'   #'1.0,0.5,0.3,0.1,0.001,0.001' #1.0,1.0,0.5,0.3,0.1,0.001,0.001 # generator L1 loss, perceptual losses (for 1st, 2nd, and 3rd layers), peak value loss, and negative value loss
#g_adv_loss_weight='0.1' # weight for adversarial loss (cgan: 0.01)
#g_metric='loss_total' #loss #acc #acc_dc
#g_loss='mae' #implemented: dc, tversky, tversky_focal, tversky_focal_multiclass, dice_focal, focal  #niftynet built-in: 'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input, CrossEntropy
#g_activation='tanh' #'linear' #'tanh' # last layer activation - sigmoid: independent (probability) multi-label training, softmax: dependent single label training
#g_trn_patch_size='160,160,80' #for livianet 50,50,50 (32,32,32; 27,27,27); for deepmedic 64,64,64 (48,48,48);
#g_trn_output_size='160,160,80' #for livianet 32,32,32 (14,14,14; 9,9,9); for deepmedic 32,32,32 (16,16,16);
#g_trn_step_size='0,0,0'
#g_tst_patch_size='160,160,80' #for livianet 50,50,50 (32,32,32; 27,27,27); for deepmedic 64,64,64 (48,48,48);
#g_tst_output_size='160,160,80' #for livianet 32,32,32 (14,14,14; 9,9,9); for deepmedic 32,32,32 (16,16,16);
#g_tst_step_size='0,0,0'
## discriminator
#d_network='u_net' #'patch_gan' #'dilated_densenet' #'patch_gan' #'dilated_densenet' #patch_gan #dilated_densenet
#d_optimizer='Adam'
#d_initial_lr='0.0001' # for cyclegan: '0.0001' according to feedback adversarial network (cvpr19) # others: '0.001'
#d_num_classes='1'
#d_metric='loss_total' #loss #acc #acc_dc
#d_loss='mse' #'mse' #'binary_crossentropy' #'mse' #'binary_crossentropy'
#d_activation='linear' #'sigmoid' #'linear' #'sigmoid'
#d_spectral_norm='1'
#d_trn_patch_size='160,160,80' #for livianet 50,50,50 (32,32,32; 27,27,27); for deepmedic 64,64,64 (48,48,48);
#d_trn_output_size='160,160,80' #for livianet 32,32,32 (14,14,14; 9,9,9); for deepmedic 32,32,32 (16,16,16);
#d_trn_step_size='0,0,0'
#d_tst_patch_size='160,160,80' #for livianet 50,50,50 (32,32,32; 27,27,27); for deepmedic 64,64,64 (48,48,48);
#d_tst_output_size='160,160,80' #for livianet 32,32,32 (14,14,14; 9,9,9); for deepmedic 32,32,32 (16,16,16);
#d_tst_step_size='0,0,0'


## for thalamus (testing fc-dense context net using 5 folds cross-validation)
#mode='0' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
#gpu_id='0' # specific gpu id ('0,1') or -1: all available GPUs, -2: cpu only
##num_of_gpu='4' # only if gpu_id==-1
#num_classes='2'
#multi_output='0'
#output_name='tha_seg'
#attention_loss='0'
#overlap_penalty_loss='0'
#loss='tversky_focal_multiclass' #'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input
#approach='fc_densenet_dilated' #'unet' #'pr_fb_net' #unet #livianet, fc_densenet, fc_densenet_ms, fc_densenet_ms_attention
#dataset='tha_seg_dl' #'tha_seg_dl'
#preprocess_trn='2'
#preprocess_tst='2'
#num_k_fold='5'
#batch_size='8' #8, 16, 32
#num_epochs='50'
#patience='10'
#optimizer='Adam'
#initial_lr='0.001'
#is_set_random_seed='1'
#random_seed_num='1'
#metric='loss' #loss #acc #acc_dc
#lamda='0,0,0,0' # weight for losses - dentate/interposed/attention/overlap penalty
#exclusive_train='0' #if exclusive_train is '1', during training, remove designated label
#exclude_label_num='0' #exclude_label_num: 0. bg, 1. thalamus
#activation='softmax' # last layer activation - sigmoid: independent multi-label training, softmax: dependent single label training
#target='tha'
#image_modality='T1,B0,FA' #T1
#trn_patch_size='32,32,32' #for livianet 27,27,27 (32,32,32); for deepmedic 48,48,48; for CAFP 64,64,64; for others 32,32,32
#trn_output_size='32,32,32' #for livianet 9,9,9 (14,14,14); for deepmedic 16,16,16; for others 32,32,32
#trn_step_size='9,9,9' #for livianet and deepmedic 5,5,5 (9,9,9); for others 15,15,15
#tst_patch_size='32,32,32' #for livianet 27,27,27 (32,32,32); for deepmedic 48,48,48; for CAFP 64,64,64; for others 32,32,32
#tst_output_size='32,32,32' #for livianet 9,9,9 (14,14,14); for deepmedic 16,16,16; for others 32,32,32
#tst_step_size='9,9,9' #for livianet and deepmedic 5,5,5 (9,9,9); for others 15,15,15
#crop_margin='9,9,9'
#bg_discard_percentage='0.0'
#bg_value='0'
#importance_spl='0' # not complete
#oversampling='0' # not complete
#threshold='0.1'
#continue_tr='0'
#is_unseen_case='0'
#is_measure='1'
#is_new_trn_label='0' # 0: 29 manual labels, 1: 31 segmentation using a proposed network, 2: suit labels, 3: 29 manual labels + 31 seg (self-training?)
#new_label_path='/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_dilated_fc_densenet_dual_output_only_dentate_only_interposed_softmax_attn_loss_0_1_overlap_loss_updated_0_5_train_29_test_32'
##folder_names='tha_baseline_test_t1_b0_fa_miccai_spotlightnet_step_9_2nd' #'tha_baseline_test_t1_b0_fa_miccai_livianet_rev_in_32_out_14' #'tha_baseline_test_t1_b0_fa_miccai_only_glam_after_shortcut_dil_3' #'tha_baseline_test_t1_b0_fa_miccai_only_scSE' #tha_baseline_test_t1_b0_fa_miccai_only_glam, tha_baseline_test_t1_b0_fa_miccai_wo_cfp_glam
#folder_names='fc_dense_contextnet_glam_tha_test_t1_b0_fa_patch_32_32_32_step_9_random_seed_fixed_5_fold'
#dataset_path='/home/asaf/jinyoung/projects/datasets/thalamus/'


## for dentate/interposed (dilated densenet wo decoder)
#mode='0' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
#gpu_id='0' # specific gpu id ('0,1') or -1: all available GPUs, -2: cpu only
#num_of_gpu='4' # only if gpu_id==-1
#num_classes='3' # 2 # 3
#multi_output='0'
#output_name='dentate_interposed_seg' # dentate_interposed_seg for single output
#attention_loss='0'
#overlap_penalty_loss='0'
#loss='tversky_focal_multiclass' #implemented: dc, tversky, tversky_focal, tversky_focal_multiclass, dice_focal, focal  #niftynet built-in: 'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input, CrossEntropy
#approach='fc_densenet_dilated' #'multires_net' #unet #livianet # fc_densenet_dilated # deepmedic # densenet_dilated # attention_unet #cenet
#dataset='dcn_seg_dl' #'tha_seg_dl' #dcn_seg_dl
#preprocess_trn='2'
#preprocess_tst='2'
#num_k_fold='5' #5
#batch_size='8'
#num_epochs='50' #'20'
#patience='20' #'2'
#optimizer='Adam'
#initial_lr='0.001'
#is_set_random_seed='1'
#random_seed_num='1'
#metric='loss' #loss #acc #acc_dc
#lamda='1.0,1.0,0.1,0.5' # weight for losses - dentate/interposed/attention/overlap penalty
#exclusive_train='0' #if exclusive_train is '1', during training, remove designated label
#exclude_label_num='2,' #exclude_label_num: 0. bg, 1.dentate, 2. interposed for exclusive training or multi-output, -1: nothing
#activation='softmax' # last layer activation - sigmoid: independent (probability) multi-label training, softmax: dependent single label training
#target='dentate,interposed' #'tha' #dentate # dentate,interposed
#image_modality='B0' #'B0,T1,LP,FA' #'B0' #'B0,T1,LP,FA'
#trn_patch_size='32,32,32' #for livianet 50,50,50 (32,32,32; 27,27,27); for deepmedic 64,64,64 (48,48,48);
#trn_output_size='32,32,32' #for livianet 32,32,32 (14,14,14; 9,9,9); for deepmedic 32,32,32 (16,16,16);
#trn_step_size='5,5,5' #
#tst_patch_size='32,32,32' #for livianet 50,50,50 (32,32,32; 27,27,27); for deepmedic 64,64,64 (48,48,48);
#tst_output_size='32,32,32' #for livianet 32,32,32 (14,14,14; 9,9,9); for deepmedic 32,32,32 (16,16,16);
#tst_step_size='5,5,5'
#crop_margin='5,5,5'
#bg_discard_percentage='0.0'
#bg_value='0'
#importance_spl='0' # not complete
#oversampling='0' # not complete
#threshold='0' #0.5
#continue_tr='0'
#is_unseen_case='0'
#is_measure='1'
#is_new_trn_label='3' # 0: 29 manual labels, 1: 31 segmentation using a proposed network, 2: suit labels, 3: 29 manual labels + 31 seg (self-training?)
##new_label_path='/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_dilated_fc_densenet_dual_output_only_dentate_only_interposed_softmax_attn_loss_0_1_overlap_loss_updated_0_5_train_29_test_32'
##new_label_path='/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_unet_20_patience_5_folds_32_32_for_tmi1_random_seed_fixed_train_29_test_31'
#new_label_path='/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_fc_densenet_20_patience_5_folds_32_32_for_tmi1_random_seed_fixed_train_29_test_31'
#folder_names='dentate_interposed_fc_densenet_20_patience_5_folds_32_32_for_tmi1_random_seed_fixed_31_seg+29_manual_tst_29_manual' #'dentate_interposed_fc_densenet_ag_rev_2_folds_for_bs' # 'fc_densenet_ag_rev_2_folds_for_bs' #fc_densenet_dilated_ag_rev_2_folds_for_bs #tha_baseline_test_t1_b0_fa #fc_densenet_cafp_2_folds_for_bs # fc_densenet_cafp_2_folds_for_bs
#dataset_path='/home/shira-raid1/DBS_data/Cerebellum_Tools/'


## for dentate/interposed (single-output and attention loss)
#mode='0' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
#gpu_id='0' # specific gpu id ('0,1') or -1: all available GPUs, -2: cpu only
#num_of_gpu='4' # only if gpu_id==-1
#num_classes='3' # 2 # 3
#multi_output='0'
#output_name='dentate_interposed_seg' # dentate_interposed_seg for single output
#attention_loss='0'
#overlap_penalty_loss='0'
#loss='tversky_focal_multiclass' #implemented: dc, tversky, tversky_focal, tversky_focal_multiclass, dice_focal, focal  #niftynet built-in: 'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input, CrossEntropy
#approach='fc_densenet_dilated' #'multires_net' #unet #livianet # fc_densenet_dilated # deepmedic # densenet_dilated
#dataset='dcn_seg_dl' #'tha_seg_dl' #dcn_seg_dl
#preprocess_trn='2'
#preprocess_tst='2'
#num_k_fold='5' #5
#batch_size='8'
#num_epochs='50' #'20'
#patience='20' #'2'
#optimizer='Adam'
#initial_lr='0.001'
#is_set_random_seed='0'
#random_seed_num='None'
#metric='loss' #loss #acc #acc_dc
#lamda='1.0,1.0,0.1,0.5' # weight for losses - dentate/interposed/attention/overlap penalty
#exclusive_train='0' #if exclusive_train is '1', during training, remove designated label
#exclude_label_num='2,' #exclude_label_num: 0. bg, 1.dentate, 2. interposed for exclusive training or multi-output, -1: nothing
#activation='softmax' # last layer activation - sigmoid: independent (probability) multi-label training, softmax: dependent single label training
#target='dentate,interposed' #'tha' #dentate # dentate,interposed
#image_modality='B0' #'B0,T1,LP,FA' #'B0' #'B0,T1,LP,FA'
#trn_patch_size='32,32,32' #'32,32,32'
#trn_output_size='32,32,32'
#trn_step_size='5,5,5' #
#tst_patch_size='32,32,32' #'32,32,32'
#tst_output_size='32,32,32'
#tst_step_size='5,5,5'
#crop_margin='5,5,5'
#bg_discard_percentage='0.0'
#bg_value='0'
#importance_spl='0' # not complete
#oversampling='0' # not complete
#threshold='0' #0.5
#continue_tr='0'
#is_unseen_case='0'
#is_measure='1'
#is_new_trn_label='0' # 0: 29 manual labels, 1: 31 segmentation using a proposed network, 2: suit labels, 3: 29 manual labels + 31 seg (self-training?)
#new_label_path='/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_dilated_fc_densenet_dual_output_only_dentate_only_interposed_softmax_attn_loss_0_1_overlap_loss_updated_0_5_train_29_test_32'
#folder_names='final_model_single_output_wo_attn_loss_epoch_50_patience_20_5_folds' #'dentate_interposed_fc_densenet_ag_rev_2_folds_for_bs' # 'fc_densenet_ag_rev_2_folds_for_bs' #fc_densenet_dilated_ag_rev_2_folds_for_bs #tha_baseline_test_t1_b0_fa #fc_densenet_cafp_2_folds_for_bs # fc_densenet_cafp_2_folds_for_bs
#dataset_path='/home/shira-raid1/DBS_data/Cerebellum_Tools/'


## for dentate/interposed (multi-output) - only training 29 cases and testing 42 cases
#mode='1' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
#gpu_id='0' # specific gpu id ('0,1') or -1: all available GPUs, -2: cpu only
#num_of_gpu='4' # only if gpu_id==-1
#num_classes='2,3' # 2 # 3
#multi_output='1'
#output_name='dentate_seg,interposed_seg' # dentate_interposed_seg for single output
#attention_loss='1'
#overlap_penalty_loss='1'
#loss='tversky_focal_multiclass,tversky_focal_multiclass' #implemented: dc, tversky, tversky_focal, tversky_focal_multiclass, dice_focal, focal  #niftynet built-in: 'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input, CrossEntropy
#approach='fc_densenet_dilated' #'multires_net' #unet #livianet # fc_densenet_dilated # deepmedic # densenet_dilated
#dataset='dcn_seg_dl' #'tha_seg_dl' #dcn_seg_dl
#preprocess_trn='2'
#preprocess_tst='2'
#num_k_fold='2' #5
#batch_size='8'
#num_epochs='50' #'20'
#patience='10' #'2'
#optimizer='Adam'
#initial_lr='0.001'
#is_set_random_seed='0'
#random_seed_num='None'
#metric='loss_total' #loss #acc #acc_dc
#lamda='1.0,1.0,0.1,0.5' # weight for losses - dentate/interposed/attention/overlap penalty
#exclusive_train='0' #if exclusive_train is '1', during training, remove designated label
#exclude_label_num='2,' #exclude_label_num: 0. bg, 1.dentate, 2. interposed for exclusive training or multi-output, -1: nothing
#activation='softmax,softmax' # last layer activation - sigmoid: independent (probability) multi-label training, softmax: dependent single label training
#target='dentate,interposed' #'tha' #dentate # dentate,interposed
#image_modality='B0' #'B0,T1,LP,FA' #'B0' #'B0,T1,LP,FA'
#trn_patch_size='32,32,32' #'32,32,32'
#trn_output_size='32,32,32'
#trn_step_size='5,5,5' #
#tst_patch_size='32,32,32' #'32,32,32'
#tst_output_size='32,32,32'
#tst_step_size='5,5,5'
#crop_margin='5,5,5'
#bg_discard_percentage='0.0'
#bg_value='0'
#importance_spl='0' # not complete
#oversampling='0' # not complete
#threshold='0' #0.5
#continue_tr='0'
#is_unseen_case='0'
#is_measure='0'
#is_new_trn_label='1' # 0: 29 manual labels, 1: 31 segmentation using a proposed network, 2: suit labels, 3: 29 manual labels + 31 seg (self-training?)
#new_label_path='/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_dilated_fc_densenet_dual_output_only_dentate_only_interposed_softmax_attn_loss_0_1_overlap_loss_updated_0_5_train_29_test_32'
#folder_names='dentate_interposed_dilated_fc_densenet_dual_output_attn_loss_train_29_cases_test_42_cases_32_32_for_bs' #'dentate_interposed_fc_densenet_ag_rev_2_folds_for_bs' # 'fc_densenet_ag_rev_2_folds_for_bs' #fc_densenet_dilated_ag_rev_2_folds_for_bs #tha_baseline_test_t1_b0_fa #fc_densenet_cafp_2_folds_for_bs # fc_densenet_cafp_2_folds_for_bs
#dataset_path='/home/shira-raid1/DBS_data/Cerebellum_Tools/'


## for dentate/interposed (multi-output for only dentate / dentate + interposed)
#mode='0' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
#gpu_id='0' # specific gpu id ('0,1') or -1: all available GPUs, -2: cpu only
#num_of_gpu='4' # only if gpu_id==-1
#num_classes='2,2' # 2 # 3
#multi_output='1'
#output_name='dentate_seg,interposed_seg'
#attention_loss='1'
#overlap_penalty_loss='1'
#loss='tversky_focal_multiclass,tversky_focal_multiclass' #implemented: dc, tversky, tversky_focal, tversky_focal_multiclass, dice_focal, focal  #niftynet built-in: 'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input, CrossEntropy
#approach='fc_densenet_dilated' #'multires_net' #unet #livianet # fc_densenet_dilated # deepmedic # densenet_dilated
#dataset='dcn_seg_dl' #'tha_seg_dl' #dcn_seg_dl
#preprocess_trn='2'
#preprocess_tst='2'
#num_k_fold='5' #5
#batch_size='8'
#num_epochs='50' #'20'
#patience='20' #'2'
#optimizer='Adam'
#initial_lr='0.001'
#is_set_random_seed='1'
#random_seed_num='1'
#metric='loss_total' #loss #acc #acc_dc
#lamda='1.0,1.0,0.1,0.5' # weight for losses - dentate/interposed/attention/overlap penalty
#exclusive_train='0' #if exclusive_train is '1', during training, remove designated label
#exclude_label_num='2,1' #exclude_label_num: 0. bg, 1.dentate, 2. interposed for exclusive training or multi-output, -1: nothing
#activation='softmax,softmax' # last layer activation - sigmoid: independent (probability) multi-label training, softmax: dependent single label training
#target='dentate,interposed' #'tha' #dentate # dentate,interposed
#image_modality='B0' #'B0,T1,LP,FA' #'B0' #'B0,T1,LP,FA'
#trn_patch_size='40,40,40' #'32,32,32'
#trn_output_size='40,40,40'
#trn_step_size='5,5,5' #
#tst_patch_size='40,40,40' #'32,32,32'
#tst_output_size='40,40,40'
#tst_step_size='5,5,5'
#crop_margin='5,5,5'
#bg_discard_percentage='0.0'
#bg_value='0'
#importance_spl='0' # not complete
#oversampling='0' # not complete
#threshold='0' #0.5
#continue_tr='0'
#is_unseen_case='0'
#is_measure='1'
#is_new_trn_label='0' # 0: 29 manual labels, 1: 31 segmentation using a proposed network, 2: suit labels, 3: 29 manual labels + 31 seg (self-training #2)
#new_label_path='/home/asaf/jinyoung/projects/results/dcn_seg_dl/dcn_net_train_29_test_31' #'/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_dilated_fc_densenet_dual_output_only_dentate_only_interposed_softmax_attn_loss_0_1_overlap_loss_updated_0_5_train_29_test_32'
#folder_names='final_model_psize_40_attn_loss_0_1_overlap_loss_updated_0_5_wo_pre_train_fine_tune_train_50_epoch_20_patience_5_folds_random_seed_fix_2nd' #'dcn_net_pre-train_31_seg' #'dcn_net_train_29_test_31' #'final_model_attn_loss_0_1_overlap_loss_updated_0_5_wo_pre_train_fine_tune_train_50_epoch_20_patience_5_folds_train_only_29' #'dentate_interposed_fc_densenet_ag_rev_2_folds_for_bs' # 'fc_densenet_ag_rev_2_folds_for_bs' #fc_densenet_dilated_ag_rev_2_folds_for_bs #tha_baseline_test_t1_b0_fa #fc_densenet_cafp_2_folds_for_bs # fc_densenet_cafp_2_folds_for_bs
#dataset_path='/home/shira-raid1/DBS_data/Cerebellum_Tools/'


## training b0 images with qsm dentate labels
## for dentate/interposed (multi-output for only dentate / dentate + interposed)
#mode='0' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
#gpu_id='0' # specific gpu id ('0,1') or -1: all available GPUs, -2: cpu only
#num_of_gpu='4' # only if gpu_id==-1
#num_classes='2' # 2 # 3
#multi_output='0'
#output_name='dentate_seg'
#attention_loss='1'
#overlap_penalty_loss='0'
#loss='tversky_focal_multiclass' #implemented: dc, tversky, tversky_focal, tversky_focal_multiclass, dice_focal, focal  #niftynet built-in: 'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input, CrossEntropy
#approach='fc_densenet_dilated' #'multires_net' #unet #livianet # fc_densenet_dilated # deepmedic # densenet_dilated
#dataset='dcn_seg_dl' #'tha_seg_dl' #dcn_seg_dl
#preprocess_trn='2'
#preprocess_tst='2'
#num_k_fold='11' #5
#batch_size='8'
#num_epochs='50' #'20'
#patience='20' #'2'
#optimizer='Adam'
#initial_lr='0.001'
#is_set_random_seed='1'
#random_seed_num='1'
#metric='loss' #loss_total' #loss #acc #acc_dc
#lamda='1.0,1.0,0.1,0.5' # weight for losses - dentate/interposed/attention/overlap penalty
#exclusive_train='0' #if exclusive_train is '1', during training, remove designated label
#exclude_label_num='2,1' #exclude_label_num: 0. bg, 1.dentate, 2. interposed for exclusive training or multi-output, -1: nothing
#activation='softmax' # last layer activation - sigmoid: independent (probability) multi-label training, softmax: dependent single label training
#target='dentate' #'tha' #dentate # dentate,interposed
#image_modality='QSM' #'B0,T1,LP,FA' #'B0' #'B0,T1,LP,FA'
#trn_patch_size='32,32,32' #'32,32,32'
#trn_output_size='32,32,32'
#trn_step_size='5,5,5' #
#tst_patch_size='32,32,32' #'32,32,32'
#tst_output_size='32,32,32'
#tst_step_size='5,5,5'
#crop_margin='5,5,5'
#bg_discard_percentage='0.0'
#bg_value='0'
#importance_spl='0' # not complete
#oversampling='0' # not complete
#threshold='0' #0.5
#continue_tr='0'
#is_unseen_case='0'
#is_measure='0'
#is_new_trn_label='1' # 0: 29 manual labels, 1: 31 segmentation using a proposed network, 2: suit labels, 3: 29 manual labels + 31 seg (self-training #2)
#new_label_path='/home/asaf/jinyoung/projects/datasets/qsm_vs_b0' #'/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_dilated_fc_densenet_dual_output_only_dentate_only_interposed_softmax_attn_loss_0_1_overlap_loss_updated_0_5_train_29_test_32'
#folder_names='fc_dense_contextnet_w_attn_loss_qsm_guided_qsm_seg_step_5' #'dcn_net_pre-train_31_seg' #'dcn_net_train_29_test_31' #'final_model_attn_loss_0_1_overlap_loss_updated_0_5_wo_pre_train_fine_tune_train_50_epoch_20_patience_5_folds_train_only_29' #'dentate_interposed_fc_densenet_ag_rev_2_folds_for_bs' # 'fc_densenet_ag_rev_2_folds_for_bs' #fc_densenet_dilated_ag_rev_2_folds_for_bs #tha_baseline_test_t1_b0_fa #fc_densenet_cafp_2_folds_for_bs # fc_densenet_cafp_2_folds_for_bs
#dataset_path='/home/asaf/jinyoung/projects/datasets/qsm_vs_b0'


###### currently testing (self-training)
## for dentate/interposed (multi-output for only dentate / only interposed)
#mode='0' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
#gpu_id='0' # specific gpu id ('0,1') or -1: all available GPUs, -2: cpu only
#num_of_gpu='4' # only if gpu_id==-1
#num_classes='2,2' #'2,2' # 2 # 3
#multi_output='1'
#output_name='dentate_seg,interposed_seg'
#attention_loss='1'
#overlap_penalty_loss='1'
#loss='tversky_focal_multiclass,tversky_focal_multiclass' #implemented: dc, tversky, tversky_focal, tversky_focal_multiclass, dice_focal, focal  #niftynet built-in: 'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input, CrossEntropy
#approach='fc_densenet_dilated' #'multires_net' #unet #livianet # fc_densenet_dilated # deepmedic # densenet_dilated
#dataset='dcn_seg_dl' #'tha_seg_dl' #dcn_seg_dl
#preprocess_trn='2'
#preprocess_tst='2'
#num_k_fold='5' #5
#batch_size='8'
#num_epochs='50' #'20'
#patience='20' #'2'
#optimizer='Adam'
#initial_lr='0.001' # default 0.001 (later, apply 0.0001 for pre-training, 0.001 (default) for fine-training)
#is_set_random_seed='1'
#random_seed_num='1'
#metric='loss_total' #'loss_total' #loss #acc #acc_dc
#lamda='1.0,1.0,0.1,0.5' # default: 1.0,1.0,0.1,0.5; weight for losses - dentate/interposed/attention/overlap penalty -> later, re-do with 0.1,0.05,0.1,1.0
#exclusive_train='0' #if exclusive_train is '1', during training, remove designated label
#exclude_label_num='2,1' #exclude_label_num: 0. bg, 1. dentate, 2. interposed for exclusive training or multi-output, -1: nothing
#activation='softmax,softmax' # last layer activation - sigmoid: independent (probability) multi-label training, softmax: dependent single label training
#target='dentate,interposed' #'tha' #dentate # dentate,interposed
#image_modality='B0' #'B0,T1,LP,FA' #'B0' #'B0,T1,LP,FA'
#trn_patch_size='32,32,32' #'32,32,32'
#trn_output_size='32,32,32'
#trn_step_size='9,9,9' #
#tst_patch_size='32,32,32' #'32,32,32'
#tst_output_size='32,32,32'
#tst_step_size='9,9,9'
#crop_margin='5,5,5'
#bg_discard_percentage='0.0'
#bg_value='0'
#importance_spl='0' # not complete
#oversampling='0' # not complete
#threshold='0' #0.5
#continue_tr='1'
#is_unseen_case='0'
#is_measure='1'
#is_new_trn_label='0' # 0: 29 manual labels, 1: 31 segmentation using a proposed network, 2: suit labels, 3: 29 manual labels + 31 seg (self-training?)
#new_label_path='/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_dilated_fc_densenet_dual_output_only_dentate_only_interposed_softmax_attn_loss_0_1_overlap_loss_updated_0_5_train_29_test_32' #'/home/shira-raid1/DBS_data/Cerebellum_Tools/'
#folder_names='final_model_attn_loss_0_1_overlap_loss_updated_0_5_seg_fine-tune_29_test_5_folds_random_seed_fixed_step_9' #'final_model_attn_loss_0_1_overlap_loss_updated_0_5_suit_fine-tune_29_test' #'dentate_interposed_fc_densenet_ag_rev_2_folds_for_bs' # 'fc_densenet_ag_rev_2_folds_for_bs' #fc_densenet_dilated_ag_rev_2_folds_for_bs #tha_baseline_test_t1_b0_fa #fc_densenet_cafp_2_folds_for_bs # fc_densenet_cafp_2_folds_for_bs
#dataset_path='/home/shira-raid1/DBS_data/Cerebellum_Tools/'


## for dentate/interposed (single-output)
#mode='0' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
#gpu_id='0' # specific gpu id ('0,1') or -1: all available GPUs, -2: cpu only
#num_of_gpu='4' # only if gpu_id==-1
#num_classes='3' # 2 # 3
#multi_output='0'
#output_name='dentate_seg,interposed_seg'
#attention_loss='1'
#overlap_penalty_loss='0'
#loss='tversky_focal' #implemented: dc, tversky, tversky_focal, tversky_focal_multiclass, dice_focal, focal  #niftynet built-in: 'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input, CrossEntropy
#approach='fc_densenet_dilated' #'multires_net' #unet #livianet # fc_densenet_dilated # deepmedic # densenet_dilated
#dataset='dcn_seg_dl' #'tha_seg_dl' #dcn_seg_dl
#preprocess_trn='2'
#preprocess_tst='2'
#num_k_fold='2' #5
#batch_size='8'
#num_epochs='50' #'20'
#patience='10' #'2'
#optimizer='Adam'
#initial_lr='0.001'
#is_set_random_seed='0'
#random_seed_num='None'
#metric='acc' #loss #acc #acc_dc
#lamda='1.0,1.0,0.1,0.5' # weight for losses - dentate/interposed/attention/overlap penalty
#exclusive_train='0' #if exclusive_train is '1', during training, remove designated label
#exclude_label_num='2,' #exclude_label_num: 0. bg, 1.dentate, 2. interposed for exclusive training or multi-output, -1: nothing
#activation='softmax' # last layer activation - sigmoid: independent (probability) multi-label training, softmax: dependent single label training
#target='dentate,interposed' #'tha' #dentate # dentate,interposed
#image_modality='B0' #'B0,T1,LP,FA' #'B0' #'B0,T1,LP,FA'
#trn_patch_size='32,32,32' #'32,32,32'
#trn_output_size='32,32,32'
#trn_step_size='5,5,5' #
#tst_patch_size='32,32,32' #'32,32,32'
#tst_output_size='32,32,32'
#tst_step_size='5,5,5'
#crop_margin='5,5,5'
#bg_discard_percentage='0.0'
#bg_value='0'
#importance_spl='0' # not complete
#oversampling='0' # not complete
#threshold='0' #0.5
#continue_tr='0'
#is_unseen_case='0'
#is_measure='1'
#is_new_trn_label='1' # 0: 29 manual labels, 1: 31 segmentation using a proposed network, 2: suit labels, 3: 29 manual labels + 31 seg (self-training?)
#new_label_path='/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_dilated_fc_densenet_dual_output_only_dentate_only_interposed_softmax_attn_loss_0_1_overlap_loss_updated_0_5_train_29_test_32'
#folder_names='dentate_interposed_dilated_fc_densenet_ag_combined_dentate_interposed_train_2_folds_32_32_for_bs' #'dentate_interposed_fc_densenet_ag_rev_2_folds_for_bs' # 'fc_densenet_ag_rev_2_folds_for_bs' #fc_densenet_dilated_ag_rev_2_folds_for_bs #tha_baseline_test_t1_b0_fa #fc_densenet_cafp_2_folds_for_bs # fc_densenet_cafp_2_folds_for_bs
#dataset_path='/home/shira-raid1/DBS_data/Cerebellum_Tools/'

#
## for dentate and interposed nuclei (unseen case)
#mode='2' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
#gpu_id='-2' # specific gpu id ('0,1') or -1: all available GPUs, -2: cpu only
#num_of_gpu='4' # only if gpu_id==-1
#num_classes='2,2' # 2 # 3
#multi_output='1'
#output_name='dentate_seg,interposed_seg'
#attention_loss='1'
#overlap_penalty_loss='1'
#loss='tversky_focal_multiclass,tversky_focal_multiclass' #implemented: dc, tversky, tversky_focal, tversky_focal_multiclass, dice_focal, focal  #niftynet built-in: 'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input
#approach='fc_densenet_dilated' #'pr_fb_net' #unet #livianet # fc_densenet_dilated # deepmedic
#dataset='dcn_seg_dl' #'tha_seg_dl' #dcn_seg_dl
#preprocess_trn='2'
#preprocess_tst='2'
#num_k_fold='5' #
#batch_size='8'
#num_epochs='50' #'20'
#patience='20' #'2'
#optimizer='Adam'
#initial_lr='0.001'
#is_set_random_seed='0'
#random_seed_num='None'
#metric='loss_total' #loss #acc #acc_dc
#lamda='1.0,1.0,0.1,0.5' # weight for losses - dentate/interposed/attention/overlap penalty
#exclusive_train='0' #if exclusive_train is '1', during training, remove designated label
#exclude_label_num='2,1' #exclude_label_num: 0. bg, 1.dentate, 2. interposed
#activation='softmax,softmax' # last layer activation - sigmoid: independent multi-label training, softmax: dependent single label training
#target='dentate,interposed'  #'tha' #dentate # dentate,interpose
#image_modality='B0' #'B0,T1,LP,FA' #'B0' #'B0,T1,LP,FA'
#trn_patch_size='32,32,32' #'32,32,32'
#trn_output_size='32,32,32'
#trn_step_size='5,5,5' #
#tst_patch_size='32,32,32' #'32,32,32'
#tst_output_size='32,32,32'
#tst_step_size='5,5,5'
#crop_margin='5,5,5'
#bg_discard_percentage='0.0'
#bg_value='0'
#importance_spl='0' # not complete
#oversampling='0' # not complete
#threshold='0' #0.5
#continue_tr='0'
#is_unseen_case='1'
#is_measure='0'
#is_new_trn_label='0' # 0: 29 manual labels, 1: 31 segmentation using a proposed network, 2: suit labels, 3: 29 manual labels + 31 seg (self-training?)
#new_label_path='/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_dilated_fc_densenet_dual_output_only_dentate_only_interposed_softmax_attn_loss_0_1_overlap_loss_updated_0_5_train_29_test_32'
#folder_names='dentate_interposed_only_test_b0_080519_#5' #fc_densenet_dilated_ag_rev_2_folds_for_bs #tha_baseline_test_t1_b0_fa #fc_densenet_cafp_2_folds_for_bs # fc_densenet_cafp_2_folds_for_bs
#dataset_path='/home/shira-raid1/DBS_data/Cerebellum_Tools/'


## for dentate
#mode='0' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
#gpu_id='0' # specific gpu id ('0,1') or -1: all available GPUs, -2: cpu only
#num_of_gpu='4' # only if gpu_id==-1
#num_classes='2' # 2 # 3
#multi_output='0'
#output_name='dentate_seg,interposed_seg'
#attention_loss='1'
#overlap_penalty_loss='0'
#loss='tversky_focal_multiclass' #implemented: dc, tversky, tversky_focal, tversky_focal_multiclass, dice_focal, focal  #niftynet built-in: 'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input
#approach='deepmedic' #'pr_fb_net' #unet #livianet # fc_densenet_dilated # deepmedic
#dataset='dcn_seg_dl' #'tha_seg_dl' #dcn_seg_dl
#preprocess_trn='2'
#preprocess_tst='2'
#num_k_fold='2' # 5
#batch_size='16'
#num_epochs='50' #'20'
#patience='10' #'2'
#optimizer='Adam'
#initial_lr='0.001'
#is_set_random_seed='0'
#random_seed_num='None'
#metric='loss' #loss #acc #acc_dc
#lamda='1.0,1.0,0.1,0.5' # weight for losses - dentate/interposed/attention/overlap penalty
#exclusive_train='0' #if exclusive_train is '1', during training, remove designated label
#exclude_label_num='1' #exclude_label_num: 0. bg, 1.dentate, 2. interposed
#activation='softmax' # last layer activation - sigmoid: independent multi-label training, softmax: dependent single label training
#target='dentate'  #'tha' #dentate # dentate,interposed
#image_modality='B0' #'B0,T1,LP,FA' #'B0' #'B0,T1,LP,FA'
#trn_patch_size='48,48,48' #'32,32,32'
#trn_output_size='16,16,16'
#trn_step_size='5,5,5' #
#tst_patch_size='48,48,48' #'32,32,32'
#tst_output_size='16,16,16'
#tst_step_size='5,5,5'
#crop_margin='5,5,5'
#bg_discard_percentage='0.0'
#bg_value='0'
#importance_spl='0' # not complete
#oversampling='0' # not complete
#threshold='0' #0.5
#continue_tr='0'
#is_unseen_case='0'
#is_measure='1'
#is_new_trn_label='1' # 0: 29 manual labels, 1: 31 segmentation using a proposed network, 2: suit labels, 3: 29 manual labels + 31 seg (self-training?)
#new_label_path='/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_dilated_fc_densenet_dual_output_only_dentate_only_interposed_softmax_attn_loss_0_1_overlap_loss_updated_0_5_train_29_test_32'
#folder_names='deepmedic_2_folds_32_for_bs' #fc_densenet_dilated_ag_rev_2_folds_for_bs #tha_baseline_test_t1_b0_fa #fc_densenet_cafp_2_folds_for_bs # fc_densenet_cafp_2_folds_for_bs
#dataset_path='/home/shira-raid1/DBS_data/Cerebellum_Tools/'

#
## for thalamus (unseen case)
#mode='2' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
#gpu_id='-2' # specific gpu id ('0,1') or -1: all available GPUs, -2: cpu only
#num_of_gpu='4' # only if gpu_id==-1
#num_classes='2'
#multi_output='0'
#output_name='tha_seg'
#attention_loss='0'
#overlap_penalty_loss='0'
#loss='tversky_focal' #'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input
#approach='fc_densenet_ms' #'unet' #'pr_fb_net' #unet #livianet, fc_densenet, fc_densenet_ms, fc_densenet_ms_attention
#dataset='tha_seg_dl' #'tha_seg_dl'
#preprocess_trn='2'
#preprocess_tst='2'
#num_k_fold='5'
#batch_size='16' #16, 32
#num_epochs='50'
#patience='10'
#optimizer='Adam'
#initial_lr='0.001'
#is_set_random_seed='0'
#random_seed_num='None'
#metric='loss' #loss #acc #acc_dc
#lamda='1.0,1.0,0.1,0.5' # weight for losses - dentate/interposed/attention/overlap penalty
#exclusive_train='0' #if exclusive_train is '1', during training, remove designated label
#exclude_label_num='1' #exclude_label_num: 0. bg, 1. thalamus
#activation='softmax' # last layer activation - sigmoid: independent multi-label training, softmax: dependent single label training
#target='tha'
#image_modality='T1,B0,FA' #T1
#trn_patch_size='64,64,64' #for livianet 27,27,27 (32,32,32); for deepmedic 48,48,48; for CAFP 64,64,64; for others 32,32,32
#trn_output_size='32,32,32' #for livianet 9,9,9 (14,14,14); for deepmedic 16,16,16; for others 32,32,32
#trn_step_size='9,9,9' #for livianet and deepmedic 5,5,5 (9,9,9); for others 15,15,15
#tst_patch_size='64,64,64' #for livianet 27,27,27 (32,32,32); for deepmedic 48,48,48; for CAFP 64,64,64; for others 32,32,32
#tst_output_size='32,32,32' #for livianet 9,9,9 (14,14,14); for deepmedic 16,16,16; for others 32,32,32
#tst_step_size='9,9,9' #for livianet and deepmedic 5,5,5 (9,9,9); for others 15,15,15
#crop_margin='9,9,9'
#bg_discard_percentage='0.0'
#bg_value='0'
#importance_spl='0' # not complete
#oversampling='0' # not complete
#threshold='0.1'
#continue_tr='0'
#is_unseen_case='1'
#is_measure='0'
#is_new_trn_label='0' # 0: 29 manual labels, 1: 31 segmentation using a proposed network, 2: suit labels, 3: 29 manual labels + 31 seg (self-training?)
#new_label_path='/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_dilated_fc_densenet_dual_output_only_dentate_only_interposed_softmax_attn_loss_0_1_overlap_loss_updated_0_5_train_29_test_32'
##folder_names='tha_baseline_test_t1_b0_fa_miccai_spotlightnet_step_9_2nd' #'tha_baseline_test_t1_b0_fa_miccai_livianet_rev_in_32_out_14' #'tha_baseline_test_t1_b0_fa_miccai_only_glam_after_shortcut_dil_3' #'tha_baseline_test_t1_b0_fa_miccai_only_scSE' #tha_baseline_test_t1_b0_fa_miccai_only_glam, tha_baseline_test_t1_b0_fa_miccai_wo_cfp_glam
#folder_names='tha_only_test_t1_b0_fa_miccai_spotlight_080519'
#dataset_path='/home/asaf/jinyoung/projects/datasets/thalamus/'

#
## for thalamus (only test (for testing revised side mask) using 5 folds cross-validation)
#mode='0' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
#gpu_id='-2' # specific gpu id ('0,1') or -1: all available GPUs, -2: cpu only
#num_of_gpu='4' # only if gpu_id==-1
#num_classes='2'
#multi_output='0'
#output_name='tha_seg'
#attention_loss='0'
#overlap_penalty_loss='0'
#loss='tversky_focal' #'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input
#approach='fc_densenet_ms' #'unet' #'pr_fb_net' #unet #livianet, fc_densenet, fc_densenet_ms, fc_densenet_ms_attention
#dataset='tha_seg_dl' #'tha_seg_dl'
#preprocess_trn='2'
#preprocess_tst='2'
#num_k_fold='5'
#batch_size='16' #16, 32
#num_epochs='0'
#patience='10'
#optimizer='Adam'
#initial_lr='0.001'
#is_set_random_seed='0'
#random_seed_num='None'
#metric='loss' #loss #acc #acc_dc
#lamda='1.0,1.0,0.1,0.5' # weight for losses - dentate/interposed/attention/overlap penalty
#exclusive_train='0' #if exclusive_train is '1', during training, remove designated label
#exclude_label_num='1' #exclude_label_num: 0. bg, 1. thalamus
#activation='softmax' # last layer activation - sigmoid: independent multi-label training, softmax: dependent single label training
#target='tha'
#image_modality='T1,B0,FA' #T1
#trn_patch_size='64,64,64' #for livianet 27,27,27 (32,32,32); for deepmedic 48,48,48; for CAFP 64,64,64; for others 32,32,32
#trn_output_size='32,32,32' #for livianet 9,9,9 (14,14,14); for deepmedic 16,16,16; for others 32,32,32
#trn_step_size='9,9,9' #for livianet and deepmedic 5,5,5 (9,9,9); for others 15,15,15
#tst_patch_size='64,64,64' #for livianet 27,27,27 (32,32,32); for deepmedic 48,48,48; for CAFP 64,64,64; for others 32,32,32
#tst_output_size='32,32,32' #for livianet 9,9,9 (14,14,14); for deepmedic 16,16,16; for others 32,32,32
#tst_step_size='9,9,9' #for livianet and deepmedic 5,5,5 (9,9,9); for others 15,15,15
#crop_margin='9,9,9'
#bg_discard_percentage='0.0'
#bg_value='0'
#importance_spl='0' # not complete
#oversampling='0' # not complete
#threshold='0.1'
#continue_tr='0'
#is_unseen_case='0'
#is_measure='1'
#is_new_trn_label='0' # 0: 29 manual labels, 1: 31 segmentation using a proposed network, 2: suit labels, 3: 29 manual labels + 31 seg (self-training?)
#new_label_path='/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_dilated_fc_densenet_dual_output_only_dentate_only_interposed_softmax_attn_loss_0_1_overlap_loss_updated_0_5_train_29_test_32'
##folder_names='tha_baseline_test_t1_b0_fa_miccai_spotlightnet_step_9_2nd' #'tha_baseline_test_t1_b0_fa_miccai_livianet_rev_in_32_out_14' #'tha_baseline_test_t1_b0_fa_miccai_only_glam_after_shortcut_dil_3' #'tha_baseline_test_t1_b0_fa_miccai_only_scSE' #tha_baseline_test_t1_b0_fa_miccai_only_glam, tha_baseline_test_t1_b0_fa_miccai_wo_cfp_glam
#folder_names='tha_baseline_test_t1_b0_fa_miccai_spotlight_step_9_only_test_side_mask_revised_072019'
#dataset_path='/home/asaf/jinyoung/projects/datasets/thalamus/'


## for thalamus
#mode='0' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
#gpu_id='0' # specific gpu id ('0,1') or -1: all available GPUs, -2: cpu only
#num_of_gpu='4' # only if gpu_id==-1
#num_classes='2'
#multi_output='0'
#output_name='tha_seg'
#attention_loss='1'
#overlap_penalty_loss='0'
#loss='tversky_focal' #'Tversky' #Dice #Dice_Dense_NS
#approach='fc_densenet_ms' #'unet' #'pr_fb_net' #unet #livianet, fc_densenet, fc_densenet_ms, fc_densenet_ms_attention
#dataset='tha_seg_dl' #'tha_seg_dl'
#preprocess_trn='2'
#preprocess_tst='2'
#num_k_fold='5'
#batch_size='16' #16, 32
#num_epochs='50'
#patience='10'
#optimizer='Adam'
#initial_lr='0.001'
#is_set_random_seed='0'
#random_seed_num='None'
#metric='loss' #loss #acc #acc_dc
#lamda='1.0,1.0,0.1,0.5' # weight for losses - dentate/interposed/attention/overlap penalty
#exclusive_train='0' #if exclusive_train is '1', during training, remove designated label
#exclude_label_num='1' #exclude_label_num: 0. bg, 1. thalamus
#activation='softmax' # last layer activation - sigmoid: independent multi-label training, softmax: dependent single label training
#target='tha'
#image_modality='T1,B0,FA' #T1
#trn_patch_size='64,64,64' #
#trn_output_size='32,32,32' #
#trn_step_size='15,15,15' # previously, 9,9,9
#tst_patch_size='64,64,64' #
#tst_output_size='32,32,32' #
#tst_step_size='15,15,15' # previously, 9,9,9
#crop_margin='9,9,9'
#bg_discard_percentage='0.0'
#bg_value='0'
#importance_spl='0' # not complete
#oversampling='0' # not complete
#threshold='0.1'
#continue_tr='0'
#is_unseen_case='0'
#is_measure='1'
#is_new_trn_label='0' # 0: 29 manual labels, 1: 31 segmentation using a proposed network, 2: suit labels, 3: 29 manual labels + 31 seg (self-training?)
#new_label_path='/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_dilated_fc_densenet_dual_output_only_dentate_only_interposed_softmax_attn_loss_0_1_overlap_loss_updated_0_5_train_29_test_32'
##folder_names='tha_baseline_test_t1_b0_fa_43patients_mcp_glam_rev_032319'
#folder_names='tha_baseline_test_t1_b0_fa'
#dataset_path='/home/asaf/jinyoung/projects/datasets/thalamus/'
#

if [ $is_set_random_seed == '1' ]
then
    export PYTHONHASHSEED=0 # to make python 3 code reproducible by deterministic hashing
fi

echo python3 run.py --mode $mode --gpu_id $gpu_id --num_of_gpu $num_of_gpu --output_name $output_name \
--approach $approach --dataset $dataset --metric $metric --loss $loss \
--preprocess_trn $preprocess_trn --preprocess_tst $preprocess_tst --num_k_fold $num_k_fold --batch_size $batch_size \
--num_epochs $num_epochs --patience $patience --num_classes $num_classes \
--multi_output $multi_output --output_name $output_name --attention_loss $attention_loss --activation $activation \
--overlap_penalty_loss $overlap_penalty_loss --trn_patch_size $trn_patch_size --trn_output_size $trn_output_size \
--trn_step_size $trn_step_size --tst_patch_size $tst_patch_size --tst_output_size $tst_output_size --tst_step_size $tst_step_size \
--is_set_random_seed $is_set_random_seed --random_seed_num $random_seed_num --crop_margin $crop_margin \
--bg_discard_percentage $bg_discard_percentage \
--threshold $threshold --target $target --image_modality $image_modality --folder_names $folder_names  \
--dataset_path $dataset_path --continue_tr $continue_tr --is_unseen_case $is_unseen_case --is_measure $is_measure \

python3 run.py --mode $mode --gpu_id $gpu_id --num_of_gpu $num_of_gpu --output_name $output_name \
--approach $approach --dataset $dataset --metric $metric --loss $loss \
--preprocess_trn $preprocess_trn --preprocess_tst $preprocess_tst --num_k_fold $num_k_fold --batch_size $batch_size \
--num_epochs $num_epochs --patience $patience --num_classes $num_classes \
--multi_output $multi_output --output_name $output_name --attention_loss $attention_loss --activation $activation \
--overlap_penalty_loss $overlap_penalty_loss --trn_patch_size $trn_patch_size --trn_output_size $trn_output_size \
--trn_step_size $trn_step_size --tst_patch_size $tst_patch_size --tst_output_size $tst_output_size --tst_step_size $tst_step_size \
--is_set_random_seed $is_set_random_seed --random_seed_num $random_seed_num --crop_margin $crop_margin \
--bg_discard_percentage $bg_discard_percentage \
--threshold $threshold --target $target --image_modality $image_modality --folder_names $folder_names  \
--dataset_path $dataset_path --continue_tr $continue_tr --is_unseen_case $is_unseen_case --is_measure $is_measure \


#echo python run.py --mode $mode --gpu_id $gpu_id --num_classes $num_classes --multi_output $multi_output \
#--attention_loss $attention_loss --overlap_penalty_loss $overlap_penalty_loss --output_name $output_name --loss $loss \
#--exclusive_train $exclusive_train --exclude_label_num $exclude_label_num --approach $approach --dataset $dataset \
#--preprocess_trn $preprocess_trn --preprocess_tst $preprocess_tst --num_k_fold $num_k_fold --batch_size $batch_size \
#--num_epochs $num_epochs --patience $patience --optimizer $optimizer --initial_lr $initial_lr \
#--is_set_random_seed $is_set_random_seed --random_seed_num $random_seed_num --trn_patch_size $trn_patch_size \
#--trn_output_size $trn_output_size --trn_step_size $trn_step_size --tst_patch_size $tst_patch_size \
#--tst_output_size $tst_output_size --tst_step_size $tst_step_size --crop_margin $crop_margin \
#--bg_discard_percentage $bg_discard_percentage --importance_spl $importance_spl --oversampling $oversampling \
#--threshold $threshold --metric $metric --lamda $lamda --target $target --activation $activation \
#--image_modality $image_modality --folder_names $folder_names --dataset_path $dataset_path --continue_tr $continue_tr \
#--is_unseen_case $is_unseen_case --is_measure $is_measure --is_new_trn_label $is_new_trn_label \
#
#python run.py --mode $mode --gpu_id $gpu_id --num_classes $num_classes --multi_output $multi_output \
#--attention_loss $attention_loss --overlap_penalty_loss $overlap_penalty_loss --output_name $output_name --loss $loss \
#--exclusive_train $exclusive_train --exclude_label_num $exclude_label_num --approach $approach --dataset $dataset \
#--preprocess_trn $preprocess_trn --preprocess_tst $preprocess_tst --num_k_fold $num_k_fold --batch_size $batch_size \
#--num_epochs $num_epochs --patience $patience --optimizer $optimizer --initial_lr $initial_lr \
#--is_set_random_seed $is_set_random_seed --random_seed_num $random_seed_num --trn_patch_size $trn_patch_size \
#--trn_output_size $trn_output_size --trn_step_size $trn_step_size --tst_patch_size $tst_patch_size \
#--tst_output_size $tst_output_size --tst_step_size $tst_step_size --crop_margin $crop_margin \
#--bg_discard_percentage $bg_discard_percentage --importance_spl $importance_spl --oversampling $oversampling \
#--threshold $threshold --metric $metric --lamda $lamda --target $target --activation $activation \
#--image_modality $image_modality --folder_names $folder_names --dataset_path $dataset_path --continue_tr $continue_tr \
#--is_unseen_case $is_unseen_case --is_measure $is_measure --is_new_trn_label $is_new_trn_label \
#--new_label_path $new_label_path


#echo python3 run.py --mode $mode --gpu_id $gpu_id --num_of_gpu $num_of_gpu --output_name $output_name --exclusive_train $exclusive_train \
#--exclude_label_num $exclude_label_num --approach $approach --dataset $dataset --metric $metric --loss $loss \
#--preprocess_trn $preprocess_trn --preprocess_tst $preprocess_tst --num_k_fold $num_k_fold --batch_size $batch_size \
#--num_epochs $num_epochs --patience $patience --num_classes $num_classes \
#--multi_output $multi_output --output_name $output_name --attention_loss $attention_loss --activation $activation \
#--overlap_penalty_loss $overlap_penalty_loss --trn_patch_size $trn_patch_size --trn_output_size $trn_output_size \
#--trn_step_size $trn_step_size --tst_patch_size $tst_patch_size --tst_output_size $tst_output_size --tst_step_size $tst_step_size \
#--is_set_random_seed $is_set_random_seed --random_seed_num $random_seed_num --crop_margin $crop_margin \
#--bg_discard_percentage $bg_discard_percentage --importance_spl $importance_spl --oversampling $oversampling \
#--threshold $threshold --target $target --image_modality $image_modality --augment_sar_data $augment_sar_data \
#--trn_dim $trn_dim --ensemble_weight $ensemble_weight --folder_names $folder_names --normalized_range_min $normalized_range_min \
#--normalized_range_max $normalized_range_max --dataset_path $dataset_path --continue_tr $continue_tr --is_unseen_case $is_unseen_case --is_measure $is_measure \
#--is_trn_inference $is_trn_inference --is_new_trn_label $is_new_trn_label --new_label_path $new_label_path \
#--is_fsl $is_fsl --num_model_samples $num_model_samples --f_use $f_use --f_num_loop $f_num_loop --f_spectral_norm $f_spectral_norm \
#--g_spectral_norm $g_spectral_norm --g_network $g_network --g_optimizer $g_optimizer --g_initial_lr $g_initial_lr --g_num_classes $g_num_classes \
#--g_multi_output $g_multi_output --g_output_name $g_output_name --g_attention_loss $g_attention_loss \
#--g_overlap_penalty_loss $g_overlap_penalty_loss --g_lamda $g_lamda --g_adv_loss_weight $g_adv_loss_weight \
#--g_metric $g_metric --g_loss $g_loss --g_activation $g_activation --g_trn_patch_size $g_trn_patch_size \
#--g_trn_output_size $g_trn_output_size --g_trn_step_size $g_trn_step_size --g_tst_patch_size $g_tst_patch_size \
#--g_tst_output_size $g_tst_output_size --g_tst_step_size $g_tst_step_size \
#--d_network $d_network --d_optimizer $d_optimizer --d_initial_lr $d_initial_lr --d_num_classes $d_num_classes \
#--d_metric $d_metric --d_loss $d_loss --d_activation $d_activation --d_spectral_norm $d_spectral_norm --d_trn_patch_size $d_trn_patch_size \
#--d_trn_output_size $d_trn_output_size --d_trn_step_size $d_trn_step_size --d_tst_patch_size $d_tst_patch_size \
#--d_tst_output_size $d_tst_output_size --d_tst_step_size $d_tst_step_size
#
#
#python3 run.py --mode $mode --gpu_id $gpu_id --num_of_gpu $num_of_gpu --output_name $output_name --exclusive_train $exclusive_train \
#--exclude_label_num $exclude_label_num --approach $approach --dataset $dataset --metric $metric --loss $loss \
#--preprocess_trn $preprocess_trn --preprocess_tst $preprocess_tst --num_k_fold $num_k_fold --batch_size $batch_size \
#--num_epochs $num_epochs --patience $patience --num_classes $num_classes \
#--multi_output $multi_output --output_name $output_name --attention_loss $attention_loss --activation $activation \
#--overlap_penalty_loss $overlap_penalty_loss --trn_patch_size $trn_patch_size --trn_output_size $trn_output_size \
#--trn_step_size $trn_step_size --tst_patch_size $tst_patch_size --tst_output_size $tst_output_size --tst_step_size $tst_step_size \
#--is_set_random_seed $is_set_random_seed --random_seed_num $random_seed_num --crop_margin $crop_margin \
#--bg_discard_percentage $bg_discard_percentage --importance_spl $importance_spl --oversampling $oversampling \
#--threshold $threshold --target $target --image_modality $image_modality --augment_sar_data $augment_sar_data \
#--trn_dim $trn_dim --ensemble_weight $ensemble_weight --folder_names $folder_names --normalized_range_min $normalized_range_min \
#--normalized_range_max $normalized_range_max --dataset_path $dataset_path --continue_tr $continue_tr --is_unseen_case $is_unseen_case --is_measure $is_measure \
#--is_trn_inference $is_trn_inference --is_new_trn_label $is_new_trn_label --new_label_path $new_label_path \
#--is_fsl $is_fsl --num_model_samples $num_model_samples --f_use $f_use --f_num_loop $f_num_loop --f_spectral_norm $f_spectral_norm \
#--g_spectral_norm $g_spectral_norm --g_network $g_network --g_optimizer $g_optimizer --g_initial_lr $g_initial_lr --g_num_classes $g_num_classes \
#--g_multi_output $g_multi_output --g_output_name $g_output_name --g_attention_loss $g_attention_loss \
#--g_overlap_penalty_loss $g_overlap_penalty_loss --g_lamda $g_lamda --g_adv_loss_weight $g_adv_loss_weight \
#--g_metric $g_metric --g_loss $g_loss --g_activation $g_activation --g_trn_patch_size $g_trn_patch_size \
#--g_trn_output_size $g_trn_output_size --g_trn_step_size $g_trn_step_size --g_tst_patch_size $g_tst_patch_size \
#--g_tst_output_size $g_tst_output_size --g_tst_step_size $g_tst_step_size \
#--d_network $d_network --d_optimizer $d_optimizer --d_initial_lr $d_initial_lr --d_num_classes $d_num_classes \
#--d_metric $d_metric --d_loss $d_loss --d_activation $d_activation --d_spectral_norm $d_spectral_norm --d_trn_patch_size $d_trn_patch_size \
#--d_trn_output_size $d_trn_output_size --d_trn_step_size $d_trn_step_size --d_tst_patch_size $d_tst_patch_size \
#--d_tst_output_size $d_tst_output_size --d_tst_step_size $d_tst_step_size