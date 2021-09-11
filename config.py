general_config = {
    'args': None,
    'validation_mode': 0, # validation mode (k-fold or loo, training only, testing only)
    'multi_output' : 0,
    'output_name' : 'dentate_seg',
    'num_classes' : 2, # 4 for brain tissue, 3 for cbct,
    'root_path' : '/home/asaf/jinyoung/projects/', # udall
    #'root_path' : '/content/g_drive/', # colab root path
    #'dataset_path': '/home/shira-raid1/DBS_data/Cerebellum_Tools/',
    'dataset_path': '/home/asaf/jinyoung/projects/datasets/thalamus/',
    #'dataset_path' : 'datasets/',
    'log_path' : 'log',
    'model_path' : 'models',
    'results_path' : 'results/',
    'patches_path' : 'patches/',
    'dataset_info' : {
        # MR brain tissue segmentation
        'iSeg2017' : {
            'format' : 'analyze',
            'time_series': False,  # Y
            'size' : (144, 192, 256),
            'num_volumes' : 10,
            'image_modality' : ['T1', 'T2'],
            'general_pattern' : 'subject-{}-{}.hdr',
            'path' : 'iSeg2017/iSeg-2017-Training/',
            'inputs' : ['T1', 'T2', 'label']
        },
        'IBSR18' : {
            'format' : 'nii',
            'time_series': False,  # Y
            'size' : (256, 128, 256),
            'num_volumes' : 18,
            'image_modality' : ['T1'],
            'general_pattern' : 'IBSR_{0:02}/IBSR_{0:02}_{1}.nii.gz',
            'path' : 'IBSR18/',
            'inputs' : ['ana_strip', 'segTRI_ana']
        },
        'MICCAI2012' : {
            'format' : 'nii',
            'time_series': False,  # Y
            'size' : (256, 287, 256),
            'num_volumes' : [15, 20],
            'image_modality' : ['T1'],
            'general_pattern' : ['{}/{}_tmp.nii.gz', '{}/{}_3C_tmp.nii.gz', '{}/{}_{}.nii.gz'],
            'path' : 'MICCAI2012/',
            'folder_names' : ['training-images', 'training-labels', 'testing-images', 'testing-labels']
        },
        '3T7T': {
            'format' : 'analyze',
            'time_series': False,  # Y
            'size' : (191, 282, 300),
            'num_volumes' : 10,
            'image_modality' : ['T1'], # for joint 4d image segmentation, this can be used as the number of time points
            'general_pattern' : ['{}/NORMAL0{}_cbq-3T.hdr', '{}/NORMAL0{}-ls-corrected3.hdr'],
            'path' : '3T7T/',
            'folder_names' : ['3T_T1', '7T_T1', '7T_label']
        },
        '3T7T_real': {
            'format': 'analyze',
            'time_series': False,  # Y
            'size': (191, 282, 300),
            'num_volumes': 10,
            'image_modality': ['T1'],
            'general_pattern': ['{}/NORMAL0{}_cbq-3T-real.hdr', '{}/NORMAL0{}-ls-corrected3.hdr'],
            'path': '3T7T/',
            'folder_names': ['3T_T1_real', '7T_T1', '7T_label']
        },
        '3T7T_total': {
            'format': 'analyze',
            'time_series': False,  # Y
            'size': (191, 282, 300),
            'num_volumes': 20,
            'image_modality': ['T1'],
            'general_pattern': ['{}/NORMAL0{}_cbq-3T.hdr', '{}/NORMAL0{}-ls-corrected3.hdr'], #1~10: blurred 7T, 11~20: real 3T
            'path': '3T7T/',
            'folder_names': ['3T_T1_total', '7T_T1', '7T_label']
        },
        '3T+7T': {
            'format': 'analyze',
            'time_series': False,  # Y
            'size': (191, 282, 300),
            'num_volumes': 20,
            'image_modality': ['T1'],
            'general_pattern': ['{}/NORMAL0{}_cbq.hdr', '{}/NORMAL0{}-ls-corrected3.hdr'], # 1~10: 7T, 11~20: real 3T
            'path': '3T7T/',
            'folder_names': ['3T+7T', '7T_T1', '7T_label']
        },
        'ADNI': {
            'format': 'analyze',
            'time_series': False,  # Y
            'size': (0, 0, 0),
            'num_volumes': 10, # AD: 18, MCI: 10, NC: 10
            'image_modality': ['T1'],
            'general_pattern': [''],
            'path': 'ADNI/',
            #'folder_names': ['NC','NC/3T_T1']   # 'AD', 'MCI', 'NC'
            #'folder_names': ['AD','AD/3T_T1_real']   # 'AD', 'MCI', 'NC'
            #'folder_names': ['AD','AD/3T_T1_total']   # 'AD', 'MCI', 'NC'
            'folder_names': ['NC','NC/3T+7T']   # 'AD', 'MCI', 'NC'
        },

        # Dental CBCT segmentation
        'CBCT16': {
            'format': 'nii',
            'time_series': False,  # Y
            'size': (0, 0, 0),
            'num_volumes': 16,
            'image_modality': ['CBCT'],
            'general_pattern': ['{}/CBCT_0{}_origin.nii.gz', '{}/CBCT_0{}_mandible.nii.gz',
                                '{}/CBCT_0{}_midface.nii.gz', '{}/CBCT_0{}.nii.gz'],
            'path': 'dental_cbct_cmf/segmentation/',
            'folder_names': ['CBCT-16']
        },
        'CBCT57': {
            'format': 'nii',
            'time_series': False, # Y
            'size': (0, 0, 0),
            'num_volumes': 57,
            'image_modality': ['CBCT'],
            'general_pattern': ['{}/patient_0{}_origin.nii.gz', '{}/patient_0{}_mandible.nii.gz',
                                '{}/patient_0{}_midface.nii.gz', '{}/patient_0{}.nii.gz'],
            'path': 'dental_cbct_cmf/segmentation/',
            'folder_names': ['CBCT-57']
        },
        'CT30': {
            'format': 'nii',
            'time_series': False, # Y
            'size': (0, 0, 0),
            'num_volumes': 30,
            'image_modality': ['CT'],
            'general_pattern': ['{}/normal_CT_0{}_origin.nii.gz', '{}/normal_CT_0{}_mandible.nii.gz',
                                '{}/normal_CT_0{}_midface.nii.gz', '{}/normal_CT_0{}.nii.gz'],
            'path': 'dental_cbct_cmf/segmentation/',
            'folder_names': ['CT-30']
        },
        'dcn_seg_dl': {
            'format': 'nii.gz',
            'time_series': False, # Y
            'size': [],
            'target': ['dentate'], # ['dentate', 'interposed'], #['dentate'], # ['dentate', 'interposed'], ['interposed', 'fastigial']
            'exclude_label_num': (2),
            'image_modality': ['B0','QSM'], #['B0', 'T1', 'LP', 'FA'], #['B0'], #'['B0', 'T1', 'LP', 'FA'],

            #'image_name_pattern': ['biascor_meanb0.nii.gz', 'qsm_Regto_B0.nii.gz'],

            'image_name_pattern': ['{}_B0_LPI.nii', '{}_T1_LPI.nii',
                                   'monogenic_signal/{}_B0_LP_corrected.nii',
                                   'monogenic_signal/{}_B0_FA_corrected.nii'],
            'image_new_name_pattern': ['{}_B0_image.nii.gz', '{}_T1_image.nii.gz','{}_B0_LP_image.nii',
                                       '{}_B0_FA_image.nii'],

            'image_resampled_name_pattern': ['{}_B0_LPI_resampled.nii.gz', '{}_T1_LPI_resampled.nii.gz',
                                             '{}_B0_LP_corrected_resampled.nii.gz', '{}_B0_FA_corrected_resampled.nii.gz'],

            'manual_corrected_pattern': 'hc_{}_{}_dentate_corrected.nii.gz',
            'manual_corrected_dentate_v2_pattern': 'hc_{}_{}_dentate_corrected_v2.nii.gz',
            'manual_corrected_interposed_v2_pattern': 'hc_{}_{}_interposed_v2.nii.gz',
            'manual_corrected_dentate_interposed_v2_pattern': 'hc_{}_{}_dentate_interposed_merged_v2.nii.gz',

            # suit
            #'trn_new_label_dentate_pattern': 'DCN_masks/{}_{}_dentate_mask.nii',
            #'trn_new_label_interposed_pattern': 'DCN_masks/{}_{}_interposed_mask.nii',

            # dcn seg
            # 'trn_new_label_dentate_pattern': 'segmentation/{}_dentate_seg_fc_densenet_dilated_tversky_focal_multiclass.nii.gz',
            # 'trn_new_label_interposed_pattern': 'segmentation/{}_interposed_seg_fc_densenet_dilated_tversky_focal_multiclass.nii.gz',

            # in revised version
            # 'trn_new_label_dentate_pattern': 'segmentation/non_smoothed/{}_dentate_seg_{}_tversky_focal_multiclass.nii.gz',
            # 'trn_new_label_interposed_pattern': 'segmentation/non_smoothed/{}_interposed_seg_{}_tversky_focal_multiclass.nii.gz',

            'trn_new_label_dentate_pattern': '{}_dentate_Regto_B0_{}.nii.gz',
            'trn_new_label_interposed_pattern': '{}_interposed_Regto_B0_{}.nii.gz',

            'initial_mask_pattern': 'DCN_masks/thresh35/{}_{}_dentate_final.nii',
            'initial_interposed_mask_pattern_thres': 'DCN_masks/thresh35/{}_{}_interposed_thresh35.nii',
            'initial_interposed_mask_pattern_mask': 'DCN_masks/{}_{}_interposed_mask.nii',

            # 'initial_mask_pattern': 'DCN_masks/{}_{}_dentate_Regto_B0.nii.gz',
            # 'initial_interposed_mask_pattern_thres': '',
            # 'initial_interposed_mask_pattern_mask': '',

            'initial_reg_mask_pattern': 'ini_reg_{}_{}.nii.gz',

            'suit_dentate_mask_pattern': 'DCN_masks/{}_{}_dentate_mask.nii',
            'suit_interposed_mask_pattern': 'DCN_masks/{}_{}_interposed_mask.nii',
            'suit_fastigial_mask_pattern': 'DCN_masks/{}_{}_fastigial_mask.nii',

            'set_new_roi_mask': True, #True,#False,
            'margin_crop_mask': (5, 5, 5), # (10,10,5) for thalamus #(5, 5, 5) for dentate, interposed
            'crop_trn_image_name_pattern': ['{}_B0_LPI_trn_crop.nii', '{}_T1_LPI_trn_crop.nii',
                                                '{}_B0_LP_corrected_trn_crop.nii',
                                                '{}_B0_FA_corrected_trn_crop.nii'],
            'crop_tst_image_name_pattern': ['{}_B0_LPI_tst_crop.nii', '{}_T1_LPI_tst_crop.nii',
                                                '{}_B0_LP_corrected_tst_crop.nii',
                                                '{}_B0_FA_corrected_tst_crop.nii'],

            # 'crop_trn_image_name_pattern': ['biascor_meanb0_trn_crop.nii', 'qsm_Regto_B0_trn_crop.nii'],
            # 'crop_tst_image_name_pattern': ['biascor_meanb0_tst_crop.nii', 'qsm_Regto_B0_tst_crop.nii'],

            'crop_trn_manual_corrected_pattern': 'hc_{}_{}_dentate_corrected_trn_crop.nii.gz',
            'crop_tst_manual_corrected_pattern': 'hc_{}_{}_dentate_corrected_tst_crop.nii.gz',

            'crop_trn_manual_dentate_v2_corrected_pattern': 'hc_{}_{}_dentate_corrected_v2_trn_crop.nii.gz',
            'crop_tst_manual_dentate_v2_corrected_pattern': 'hc_{}_{}_dentate_corrected_v2_tst_crop.nii.gz',
            'crop_trn_manual_interposed_v2_pattern': 'hc_{}_{}_interposed_v2_trn_crop.nii.gz',
            'crop_tst_manual_interposed_v2_pattern': 'hc_{}_{}_interposed_v2_tst_crop.nii.gz',

            #'crop_trn_new_label_dentate_pattern': '{}_{}_dentate_mask_tst_crop.nii',
            #'crop_trn_new_label_interposed_pattern': '{}_{}_interposed_mask_tst_crop.nii',

            # 'crop_trn_new_label_dentate_pattern': '{}_dentate_seg_fc_densenet_dilated_tversky_focal_multiclass_crop.nii.gz',
            # 'crop_trn_new_label_interposed_pattern': '{}_interposed_seg_fc_densenet_dilated_tversky_focal_multiclass_crop.nii.gz',

            # in revised version
            # 'crop_trn_new_label_dentate_pattern': '{}_dentate_seg_{}_tversky_focal_multiclass_crop.nii.gz',
            # 'crop_trn_new_label_interposed_pattern': '{}_interposed_seg_{}_tversky_focal_multiclass_crop.nii.gz',

            'crop_trn_new_label_dentate_pattern': '{}_dentate_Regto_B0_{}_crop.nii.gz',
            'crop_trn_new_label_interposed_pattern': '{}_interposed_Regto_B0_{}_crop.nii.gz',

            'crop_suit_dentate_mask_pattern': '{}_{}_dentate_mask_trn_crop.nii',

            'crop_initial_mask_pattern': '{}_{}_dentate_final_tst_crop.nii',
            'crop_initial_interposed_mask_pattern': '{}_{}_interposed_final_tst_crop.nii',

            # 'crop_initial_mask_pattern': '{}_{}_dentate_Regto_B0_crop.nii.gz',
            # 'crop_initial_interposed_mask_pattern': '',

            'crop_initial_reg_mask_pattern': 'ini_reg_{}_{}_tst_crop.nii.gz',

            'crop_suit_interposed_mask_pattern': '{}_{}_interposed_mask_tst_crop.nii',
            'crop_suit_fastigial_mask_pattern': '{}_{}_fastigial_mask_tst_crop.nii',

            'train_roi_mask_pattern': '{}/train_roi_mask.nii.gz',
            'test_roi_mask_pattern': '{}/test_roi_mask.nii.gz',
            # total 71
            # 'patient_id': ['C050', 'C052', 'C053', 'C056', 'C057', 'C058', 'C059', 'C060', 'P030', 'P032', 'P035',
            #                'P038', 'P040', 'P042', 'P043', 'P045', 'PD053', 'PD055', 'PD060', 'PD061', 'PD062', 'PD063',
            #                'PD064', 'PD065', 'PD069', 'PD074', 'PD078', 'PD079', 'PD080', 'PD081', 'SLEEP101',
            #                'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105', 'SLEEP106', 'SLEEP107', 'SLEEP108',
            #                'SLEEP109', 'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116',
            #                'SLEEP117', 'SLEEP118', 'SLEEP119', 'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124',
            #                'SLEEP125', 'SLEEP126', 'SLEEP127', 'SLEEP128', 'SLEEP130', 'SLEEP131', 'SLEEP133',
            #                'SLEEP134', 'SLEEP135', 'SLEEP136', 'SLEEP137', 'SLEEP138', 'SLEEP139', 'SLEEP140',
            #                'SLEEP141', 'SLEEP142', 'SLEEP143', 'SLEEP144', 'SLEEP145'],
            #training data 42 (suit or training data without version2 of dentate and interposed ground truth)
            # 'patient_id': ['C050', 'C052', 'C053', 'C056', 'C057', 'C058', 'C059', 'C060', 'P030', 'P032', 'P035',
            #                'P038', 'P040', 'P042', 'P043', 'P045', 'PD053', 'PD055', 'PD060', 'PD061', 'PD062', 'PD063',
            #                'PD064', 'PD065', 'PD069', 'PD074', 'PD078', 'PD079', 'PD080', 'SLEEP106', 'SLEEP134',
            #                'SLEEP135', 'SLEEP136', 'SLEEP137', 'SLEEP138', 'SLEEP139', 'SLEEP140', 'SLEEP141',
            #                'SLEEP142', 'SLEEP143', 'SLEEP144', 'SLEEP145'],
            #training/test data 47 (with version1 of dentate ground truth)
            # 'patient_id': ['C052', 'C053', 'C056', 'PD062', 'PD063', 'PD078', 'PD079', 'PD080', 'PD081',
            #                'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105', 'SLEEP107', 'SLEEP108',
            #                'SLEEP109', 'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116',
            #                'SLEEP117', 'SLEEP118', 'SLEEP119', 'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124',
            #                'SLEEP125', 'SLEEP126', 'SLEEP127', 'SLEEP128', 'SLEEP130', 'SLEEP131', 'SLEEP133',
            #                'SLEEP134', 'SLEEP135', 'SLEEP136', 'SLEEP137', 'SLEEP138', 'SLEEP139', 'SLEEP140',
            #                'SLEEP142', 'SLEEP143', 'SLEEP144']

            #training/test data 29 (with version2 of dentate and interposed ground truth)
            'patient_id': ['PD081', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105', 'SLEEP107',
                           'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP115',
                           'SLEEP116','SLEEP117', 'SLEEP118', 'SLEEP119', 'SLEEP120', 'SLEEP122', 'SLEEP123',
                           'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127', 'SLEEP128', 'SLEEP130', 'SLEEP131',
                           'SLEEP133'],

            # QSM vs. B0 data
            # 'patient_id': ['FA005', 'FA010', 'FA013', 'FA014', 'FA015', 'FA016', 'FA017', 'FA021', 'FA022', 'FA024_1',
            #                'FA024_2'],

            'path': 'segmentation',
            'folder_names': ['fc_dense_ms'] #['pr_fb_net_test_dentate_2'] #['b0_dc_dentate_interposed'] #['b0_tversky_patch_step_5_onet']#['interposed_fastigial_seg_test_attention_onet']
            # #['interposed_fastigial_seg_test_patch_step_3'] #['b0_dc_patch_step_3'] #['interposed_fastigial_seg_test_only_b0']
            # #['loss_test'] #['b0+T1'] #['crop_margin_32_patch_size_48'] #['mixup'] #'['1_2mm_train'] #['b0']
        },
        'tha_seg_dl': {
            'format': 'nii.gz',
            'time_series': False,  # Y
            'size': [],
            'target': ['tha'],
            'exclude_label_num': (),
            'image_modality': ['T1','B0','FA'],
            'image_name_pattern': ['7T_T1_brain.nii.gz', 'registered_B0.nii.gz',
                                   'registered_FA.nii.gz'],
            'image_resampled_name_pattern': ['7T_T1_brain_resampled.nii.gz', 'registered_B0_resampled.nii.gz',
                                   'registered_FA_resampled.nii.gz'],

            'label_pattern': 'fused_{}_thalamus_final_GM_v{}.nii.gz',
            'initial_mask_pattern': 'fused_{}_thalamus_final_GM.nii.gz', # use initially corrected output temporally
            'initial_reg_mask_pattern': 'ini_reg_{}_{}.nii.gz',

            'staple_pattern': 'fused_{}_thalamus_inverted.nii.gz', # updated on 3/28/19 (previously fused_{}_thalamus_final_GM.nii.gz was used)
            'crop_staple_pattern': 'fused_{}_thalamus_inverted_tst_crop.nii.gz', # updated on 3/28/19 (previously fused_{}_thalamus_final_GM_tst_crop.nii.gz was used)

            'set_new_roi_mask': True,  # True,#False,
            'margin_crop_mask': (9, 9, 9), # for thalamus #(5, 5, 5) for dentate, interposed
            'crop_trn_image_name_pattern': ['7T_T1_brain_trn_crop.nii.gz', 'registered_B0_trn_crop.nii.gz',
                                   'registered_FA_trn_crop.nii.gz'],
            'crop_tst_image_name_pattern': ['7T_T1_brain_tst_crop.nii.gz', 'registered_B0_tst_crop.nii.gz',
                                   'registered_FA_tst_crop.nii.gz'],

            'crop_trn_image_downsampled_name_pattern': ['7T_T1_brain_trn_crop_downsampled.nii.gz',
                                                        'registered_B0_trn_crop_downsampled.nii.gz',
                                                        'registered_FA_trn_crop_downsampled.nii.gz'],
            'crop_tst_image_downsampled_name_pattern': ['7T_T1_brain_tst_crop_downsampled.nii.gz',
                                                        'registered_B0_tst_crop_downsampled.nii.gz',
                                                        'registered_FA_tst_crop_downsampled.nii.gz'],

            'crop_trn_label_pattern': 'fused_{}_thalamus_final_GM_v{}_trn_crop.nii.gz',
            'crop_tst_label_pattern': 'fused_{}_thalamus_final_GM_v{}_tst_crop.nii.gz',

            'crop_trn_label_downsampled_pattern': 'fused_thalamus_final_GM_v{}_trn_crop_downsampled.nii.gz',
            'crop_tst_label_downsampled_pattern': 'fused_thalamus_final_GM_v{}_tst_crop_downsampled.nii.gz',

            'crop_initial_mask_pattern': 'fused_{}_thalamus_final_GM_tst_crop.nii.gz',
            'crop_initial_reg_mask_pattern': 'ini_reg_{}_thalamus_tst_crop.nii.gz',

            'train_roi_mask_pattern': 'train_roi_mask.nii.gz',
            'test_roi_mask_pattern': 'test_roi_mask.nii.gz',
            # 'patient_id': ['ET018', 'ET028', 'ET030', 'MS001', 'P030', 'P040', 'PD050', 'PD061', 'PD074',
            #                'PD081', 'PD085', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105',
            #                'SLEEP106', 'SLEEP107', 'SLEEP108', 'SLEEP109'],
            'patient_id': ['ET018', 'ET019', 'ET020', 'ET021', 'ET028', 'ET030', 'MS001', 'P030', 'P040', 'PD050',
                           'PD061', 'PD074', 'PD081', 'PD085', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104',
                           'SLEEP105', 'SLEEP106', 'SLEEP107', 'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP112',
                           'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116', 'SLEEP117', 'SLEEP118', 'SLEEP119',
                           'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
                           'SLEEP128', 'SLEEP130', 'SLEEP131', 'SLEEP133'],
            'path': 'segmentation',
            'folder_names': ['tha_baseline_test_t1_b0_fa'] #['tha_baseline_test_t1_b0_fa']
        },
        'sar': {
            'dataset_filename_duke': ['Duke_Random_Excitations_set1', 'Duke_Random_Excitations_set2',
                                 'Duke_Random_Excitations_set3', 'Duke_Random_Excitations_set4',
                                 'Duke_data_Quadrature.mat'],
            'dataset_filename_ella': ['Ella_Random_Excitations_set1', 'Ella_Random_Excitations_set2'],
            'dataset_filename_louis': ['Louis_Random_Excitations_set1.mat', 'Louis_Random_Excitations_set2.mat'],
            'dataset_filename_austinman':['AustinMan_Random_Excitations_set1.mat', 'AustinMan_Random_Excitations_set2.mat'],
            'dataset_filename_austinwoman':['AustinWoman_Random_Excitations_set1.mat', 'AustinWoman_Random_Excitations_set2.mat'],
            'format': 'nii.gz',
            'image_modality': ['B1_mag','Epsilon','Rho','Sigma'], #['B1_mag','B1_real','B1_imag','Epsilon','Rho','Sigma'],
            'augment_sar_data': (0, 1),
            'normalized_range_min': -1.0,
            'normalized_range_max': 1.0,
            'trn_dim': ['axial','sagittal','coronal'], # ['axial','coronal','sagittal'] or ['']
            'ensemble_weight': (0.6,0.2,0.2),
            'path': '',
            'margin_crop_mask': (3, 3, 3, 3, 21 ,20), #(0, 0, 40),
            'folder_names': ['sar_prediction_baseline_test_b1+'],
            # 'case_id': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
            #             '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
            # 'case_id': ['1','2','3','4','5','6','7','8','9','10',
            #             '11','12','13','14','15','16','17','18','19','20',
            #             '21','22','23','24','25','26','27','28','29','30',
            #             '31','32','33','34','35','36','37','38','39','40']
            'case_id': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                        '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                        '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
                        '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                        '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
                        '61', '62', '63', '64', '65', '66', '67', '68', '69', '70',
                        '71', '72', '73', '74', '75', '76', '77', '78', '79', '80']
        }
    }
}

training_config = {
    'num_of_gpu' : 1,
    'exclusive_train' : True,
    'activation' : 'softmax', #sigmoid: independent multiple training, # softmax: dependent single label training
    'approach' : 'unet', #'approach' : 'cc_3d_fcn', unet, livianet, uresnet, fc_densenet, deepmedic, wnet, fc_capsnet, attention_unet, attention_se_fcn, fc_rna, pr_fb_net
    'dataset' : 'sar', #'dcn_seg_dl' #'3T7T', # 3T7T, 3T7T_real, 3T7T_total, 3T+7T, CBCT16, CBCT57, CT30 for training
    'data_augment': 0, # 0: offdcn_seg_dl, 1: mixup, 2. datagen, 3: mixup + datagen
    'dimension' : 2,
    'extraction_step' : (9, 9, 9), # for thalamus #(5, 5, 5) for dentate, #(2, 2, 2) for interposed
    'attention_loss' : 0,
    'overlap_penalty_loss' : 0,
    'loss' : 'Dice',  # Dice_Dense_NS (dice_dense_nosquare), Dice(dice), GDSC (generalised_dice_loss),
    # WGDL (generalised_wasserstein_dice_loss; currently not working), Tversky(tversky), CrossEntropy_Dense (cross_entropy_dense),
    # CrossEntropy (cross_entropy), SensSpec (sensitivity_specificity_loss), msd
                        # old version: weighted_categorical_crossentropy, categorical_crossentropy, dc, tversky
    'metric' : 'acc', # metric for early stopping: acc, acc_dc, loss
    'lamda' : (1.0, 1.0, 0.1, 0.01), #(1.0, 1.0, 0.1, 0.01),
    'batch_size' : 32,
    'num_epochs' : 20,
    'num_retrain' : 0,
    'optimizer' : 'Adam', #Adam, Adamax, Nadam, SGD
    'initial_lr' : '0.001',
    'is_set_random_seed' : '0',
    'random_seed_num' : '1',
    'output_shape' : (32, 32, 32), #for thalamus, dentate, interposed (48, 48, 48), #(32, 32, 32), #(32,32,32), #(9, 9, 9), #(32, 32, 32), #(9, 9, 9),
    'patch_shape' : (32, 32, 32), #for thalamus, dentate, interposed (48, 48, 48), #(32, 32, 32), #(9, 9, 9), #(32, 32, 32),
    'bg_discard_percentage' : 0, # 0 for dentate,interposed,thalamus segmentation, 0.2 for brain tissue segmentation
    'patience' : 2, #1
    'num_k_fold': 5,
    'validation_split' : 0.20,
    'shuffle' : True, # Default
    'verbose' : 1,
    'preprocess' : 2,   #0: no preprocessing, 1: standardization, 2: normalization,
    # 3: normalization + standardization, 4: histogram matching (or normalization) to one training ref + standardization,
    # 5: normalization + histogram matching
    'importance_sampling' : 0,
    'oversampling' : 0,
    'use_saved_patches' : False,
    'is_new_trn_label': 0,
    'new_label_path': '/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_dilated_fc_densenet_dual_output_only_dentate_only_interposed_softmax_attn_loss_0_1_overlap_loss_updated_0_5_train_29_test_32',
    'continue_tr' : 0,
    'is_fsl': 1,
    'GAN' :{
        'feedback' : {
            'use': 1,
            'num_loop': 3,
            'spectral_norm': 1,
        },
        'generator' : {
            'network' : 'fc_dense_context_net',
            'optimizer': 'Adam',  # Adam, Adamax, Nadam, SGD
            'initial_lr': '0.001',
            'num_classes' : [2,2],
            'multi_output': 1,
            'output_name': ['dentate_seg','interposed_seg'],
            'activation': ['softmax','softmax'],
             # sigmoid: independent multiple training, # softmax: dependent single label training
            'spectral_norm': 1,
            'attention_loss': 0,
            'overlap_penalty_loss': 0,
            'metric' : 'acc', # metric for early stopping: acc, acc_dc, loss
            'loss': ['tversky_focal_multiclass','tversky_focal_multiclass'],
            'lamda': (1.0,0.5,0.3,0.1,0.001,0.001),
            'adv_loss_weight': 0.05,
            'patch_shape': (32, 32, 32),
            # for thalamus, dentate, interposed (48, 48, 48), #(32, 32, 32), #(9, 9, 9), #(32, 32, 32),
            'output_shape': (32, 32, 32),
            # for thalamus, dentate, interposed (48, 48, 48), #(32, 32, 32), #(32,32,32), #(9, 9, 9), #(32, 32, 32), #(9, 9, 9),
            'extraction_step': (5, 5, 5),  # for thalamus #(5, 5, 5) for dentate, #(2, 2, 2) for interposed
            'model_name': []
        },
        'discriminator': {
            'network' : 'u_net', #'dilated_densenet',
            'optimizer': 'Adam',  # Adam, Adamax, Nadam, SGD
            'initial_lr': '0.001',
            'num_classes' : 1,
            'activation': 'sigmoid',
            # sigmoid: independent multiple training, # softmax: dependent single label training
            'spectral_norm': 1,
            'metric' : 'acc', # metric for early stopping: acc, acc_dc, loss
            'loss': 'binary_crossentropy',
            'patch_shape': (32, 32, 32),
            # for thalamus, dentate, interposed (48, 48, 48), #(32, 32, 32), #(9, 9, 9), #(32, 32, 32),
            'output_shape': (4, 4, 4),
            # for thalamus, dentate, interposed (48, 48, 48), #(32, 32, 32), #(32,32,32), #(9, 9, 9), #(32, 32, 32), #(9, 9, 9),
            'extraction_step': (5, 5, 5), # for thalamus #(5, 5, 5) for dentate, #(2, 2, 2) for interposed
            'model_name': []
        }
    }
}

test_config = {
    'dataset' : 'sar', #'ADNI', # dcn_seg_dl # tha_seg_dl
    'batch_size' : 32,
    'dimension' : 3,
    'extraction_step' : (9, 9, 9), # for thalamus #(5, 5, 5) for dentate, #(2, 2, 2) for interposed
    'output_shape' : (32, 32, 32), # for thalamus, dentate, interposed #(48, 48, 48), #(32, 32, 32), #(9, 9, 9),(32, 32, 32)
    'patch_shape' : (32, 32, 32), # for thalamus, dentate, interposed #(48, 48, 48), #(32, 32, 32),
    'threshold': 0, # 0.1 for thalamus, 0 for dentate
    'verbose' : 1,
    'preprocess' : 2,   #0: no preprocessing, 1: standardization, 2: normalization,
    # 3: normalization + standardization, 4: histogram matching (or normalization) to one training ref + standardization,
    # 5: normalization + histogram matching
    'is_measure': 1,
    'is_trn_inference': 1,
    'is_unseen_case' : 0,
    'GAN' :{
        'generator' : {
            'patch_shape': (32, 32, 32),  # for thalamus, dentate, interposed #(48, 48, 48), #(32, 32, 32),
            'output_shape': (32, 32, 32),
            'extraction_step': (5, 5, 5),
            'num_classes' : 5,
            'num_encoder_out_ch': 112, #528
        },
        'discriminator': {
            'patch_shape': (32, 32, 32),  # for thalamus, dentate, interposed #(48, 48, 48), #(32, 32, 32),
            'output_shape': (4, 4, 4),
            'extraction_step': (5, 5, 5),
            'num_classes' : 5
        }
    }
}
