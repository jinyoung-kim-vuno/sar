
def generate_model(gen_conf, train_conf):
    mode = gen_conf['validation_mode']
    approach = train_conf['approach']

    model = None
    if approach == 'deepmedic' :
        from .deepmedic import generate_deepmedic_model
        model =  generate_deepmedic_model(gen_conf, train_conf)
    if approach == 'livianet' :
        from .livianet import generate_livianet_model
        model = generate_livianet_model(gen_conf, train_conf)
    if approach == 'unet' :
        from .unet import generate_unet_model
        model = generate_unet_model(gen_conf, train_conf)
    if approach == 'uresnet' :
        from .uresnet import generate_uresnet_model
        model = generate_uresnet_model(gen_conf, train_conf)
    if approach == 'cenet' :
        from .cenet import generate_cenet_model
        model = generate_cenet_model(gen_conf, train_conf)
    if approach == 'cc_3d_fcn' :
        from .cc_3d_fcn import generate_cc_3d_fcn_model
        model = generate_cc_3d_fcn_model(gen_conf, train_conf)
    if approach == 'wnet' :
        from .wnet import generate_wunet_model
        model = generate_wunet_model(gen_conf, train_conf)
    if approach == 'fc_densenet':
        from .fc_densenet import generate_fc_densenet_model
        model = generate_fc_densenet_model(gen_conf, train_conf)
    if approach == 'fc_densenet_ms':
        from .fc_densenet_ms import generate_fc_densenet_ms
        model = generate_fc_densenet_ms(gen_conf, train_conf)
    if approach == 'fc_densenet_dilated':
        from .fc_densenet_dilated import generate_fc_densenet_dilated
        model = generate_fc_densenet_dilated(gen_conf, train_conf)
    if approach == 'densenet_dilated':
        from .densenet_dilated import generate_densenet_dilated
        model = generate_densenet_dilated(gen_conf, train_conf)
    if approach == 'fc_capsnet':
        from .fc_capsnet import generate_fc_capsnet_model
        model = generate_fc_capsnet_model(gen_conf, train_conf, mode)
    if approach == 'attention_unet':
        from .attention_unet import generate_attention_unet_model
        model = generate_attention_unet_model(gen_conf, train_conf)
    if approach == 'attention_se_fcn':
        from .attention_se_fcn import generate_attention_se_fcn_model
        model = generate_attention_se_fcn_model(gen_conf, train_conf)
    if approach == 'multires_net':
        from .multires_net import generate_multires_net
        model = generate_multires_net(gen_conf, train_conf)
    if approach == 'attention_onet':
        from .attention_onet import generate_attention_onet_model
        model = generate_attention_onet_model(gen_conf, train_conf)
    if approach == 'fc_rna':
        from .fc_rna import generate_fc_rna_model
        model = generate_fc_rna_model(gen_conf, train_conf)

    if approach == 'cgan':
        from .generator import build_fc_dense_contextnet, build_unet
        from .discriminator import build_dilated_densenet, build_patch_gan
        from .cgan import build_cgan
        d_model = build_dilated_densenet(gen_conf, train_conf) # optional
        g_model = build_fc_dense_contextnet(gen_conf, train_conf) # optional
        gan_model = build_cgan(gen_conf, train_conf, g_model, d_model)
        model = [g_model, d_model, gan_model]

    if approach == 'cgan_sar':
        from .generator import build_fc_dense_contextnet, build_fc_dense_contextnet_feedback, \
            build_g_dilated_densenet, build_g_unet
        from .discriminator import build_d_dilated_densenet, build_patch_gan, build_d_unet
        from .cgan import build_cgan_sar, build_cgan_feedback_sar, build_cgan_feedback_lse_sar
        from .feedback import build_feedback_net
        from .local_sar_estimator import build_local_sar_estimator

        g_network = train_conf['GAN']['generator']['network']
        d_network = train_conf['GAN']['discriminator']['network']
        is_feedback = train_conf['GAN']['feedback']['use']
        num_fb_loop = train_conf['GAN']['feedback']['num_loop']
        g_encoder_ch = []

        if g_network == 'fc_dense_contextnet':
            if is_feedback:
                g_model, g_encoder_ch = build_fc_dense_contextnet_feedback(gen_conf, train_conf)  # optional
            else:
                g_model = build_fc_dense_contextnet(gen_conf, train_conf)  # optional
        elif g_network == 'dilated_densenet':
            g_model = build_g_dilated_densenet(gen_conf, train_conf)
        elif g_network == 'u_net':
            g_model = build_g_unet(gen_conf, train_conf)  # optional
        else:
            raise NotImplementedError('choose fc_dense_contextnet or unet')

        if d_network == 'dilated_densenet':
            d_model, d_static_model = build_d_dilated_densenet(gen_conf, train_conf) # optional
        elif d_network == 'patch_gan':
            d_model, d_static_model = build_patch_gan(gen_conf, train_conf) # optional
        elif d_network == 'u_net':
            d_model, d_static_model = build_d_unet(gen_conf, train_conf)
        else:
            raise NotImplementedError('choose dilated_densenet or patch_gan')

        if is_feedback:
            fb_model = build_feedback_net(gen_conf, train_conf, g_encoder_ch)
            gan_model = build_cgan_feedback_sar(gen_conf, train_conf, g_model, d_static_model, fb_model, num_fb_loop,
                                                g_encoder_ch)
            model = [g_model, d_model, fb_model, gan_model, g_encoder_ch]
        else:
            gan_model = build_cgan_sar(gen_conf, train_conf, g_model, d_static_model)
            model = [g_model, d_model, gan_model]

    elif approach == 'cgan_sar_multi_task':
        from .generator import build_fc_dense_contextnet, build_fc_dense_contextnet_feedback, \
            build_g_dilated_densenet, build_g_unet
        from .discriminator import build_d_dilated_densenet, build_patch_gan, build_d_unet, build_d_unet_cr
        from .cgan import build_cgan_sar, build_cgan_feedback_sar, build_cgan_feedback_lse_sar, \
            build_cgan_fb_fsmt_lse_sar, build_cgan_sar_lse, build_cgan_sar_fsmt_lse
        from .feedback import build_feedback_net
        from .few_shot_model_transfer import build_few_shot_model_transfer
        from .local_sar_estimator import build_local_sar_estimator, build_fs_local_sar_estimator

        g_network = train_conf['GAN']['generator']['network']
        d_network = train_conf['GAN']['discriminator']['network']
        is_feedback = train_conf['GAN']['feedback']['use']
        num_fb_loop = train_conf['GAN']['feedback']['num_loop']
        is_fsl = train_conf['is_fsl']
        g_encoder_ch = []

        if g_network == 'fc_dense_contextnet':
            g_model, g_encoder_ch = build_fc_dense_contextnet_feedback(gen_conf, train_conf)  # optional
        elif g_network == 'dilated_densenet':
            g_model = build_g_dilated_densenet(gen_conf, train_conf)
        elif g_network == 'u_net':
            g_model = build_g_unet(gen_conf, train_conf)  # optional
        else:
            raise NotImplementedError('choose fc_dense_contextnet or unet')

        if d_network == 'dilated_densenet':
            d_model, d_static_model = build_d_dilated_densenet(gen_conf, train_conf) # optional
        elif d_network == 'patch_gan':
            d_model, d_static_model = build_patch_gan(gen_conf, train_conf) # optional
        elif d_network == 'u_net':
            d_model, d_static_model = build_d_unet(gen_conf, train_conf)
        else:
            raise NotImplementedError('choose dilated_densenet or patch_gan')

        if is_feedback:
            if is_fsl:
                d_cr_model = build_d_unet_cr(gen_conf, train_conf, d_model)
                fb_model = build_feedback_net(gen_conf, train_conf, g_encoder_ch)
                fsmt_model = build_few_shot_model_transfer(gen_conf, train_conf, g_encoder_ch)
                local_sar_model = build_fs_local_sar_estimator(gen_conf, train_conf, g_encoder_ch)
                gan_model = build_cgan_fb_fsmt_lse_sar(gen_conf, train_conf, g_model, d_static_model, fb_model, fsmt_model,
                                                        local_sar_model, num_fb_loop, g_encoder_ch)
                model = [g_model, d_model, d_cr_model, fb_model, fsmt_model, local_sar_model, gan_model, g_encoder_ch]
            else:
                d_cr_model = build_d_unet_cr(gen_conf, train_conf, d_model)
                fb_model = build_feedback_net(gen_conf, train_conf, g_encoder_ch)
                local_sar_model = build_local_sar_estimator(gen_conf, train_conf, g_encoder_ch)
                gan_model = build_cgan_feedback_lse_sar(gen_conf, train_conf, g_model, d_static_model, fb_model,
                                                        local_sar_model, num_fb_loop, g_encoder_ch)
                model = [g_model, d_model, d_cr_model, fb_model, local_sar_model, gan_model, g_encoder_ch]
        else:
            if is_fsl:
                d_cr_model = build_d_unet_cr(gen_conf, train_conf, d_model)
                fsmt_model = build_few_shot_model_transfer(gen_conf, train_conf, g_encoder_ch)
                local_sar_model = build_fs_local_sar_estimator(gen_conf, train_conf, g_encoder_ch)
                gan_model = build_cgan_sar_fsmt_lse(gen_conf, train_conf, g_model, d_static_model, fsmt_model,
                                                    local_sar_model, g_encoder_ch)
                model = [g_model, d_model, d_cr_model, fsmt_model, local_sar_model, gan_model, g_encoder_ch]
            else:
                d_cr_model = build_d_unet_cr(gen_conf, train_conf, d_model)
                local_sar_model = build_local_sar_estimator(gen_conf, train_conf, g_encoder_ch)
                gan_model = build_cgan_sar_lse(gen_conf, train_conf, g_model, d_static_model, local_sar_model, g_encoder_ch)
                model = [g_model, d_model, d_cr_model, local_sar_model, gan_model, g_encoder_ch]

    elif approach in ['single_cyclegan_sar', 'multi_cyclegan_sar', 'composite_cyclegan_sar']:
        from .generator import build_fc_dense_contextnet, build_g_dilated_densenet, build_g_unet
        from .discriminator import build_d_dilated_densenet, build_patch_gan, build_d_unet
        from .cyclegan import build_multi_cyclegan_sar_fwd, build_multi_cyclegan_sar_bwd, build_single_cyclegan_sar_fwd, \
            build_single_cyclegan_sar_bwd, build_cyclegan_sar

        g_network = train_conf['GAN']['generator']['network']
        d_network = train_conf['GAN']['discriminator']['network']
        dataset = train_conf['dataset']

        if g_network == 'fc_dense_contextnet':
            #gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_real', 'B1_imag']
            gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_mag'] #['B1_real','B1_imag','Epsilon','Rho','Sigma']  #['B1_mag'] #['B1_real','B1_imag','Sigma']
            train_conf['GAN']['generator']['num_classes'] = 1 #5 #1
            train_conf['GAN']['generator']['model_name'] = 'g_model_fc_dense_contextnet_XtoY'
            g_model_XtoY = build_fc_dense_contextnet(gen_conf, train_conf)  # generator: A -> B

            #gen_conf['dataset_info'][dataset]['image_modality'] = ['sar', 'sar']
            gen_conf['dataset_info'][dataset]['image_modality'] = ['sar'] #['sar','sar','sar','sar','sar'] #['sar']
            train_conf['GAN']['generator']['num_classes'] = 1 #5 #1 #3
            train_conf['GAN']['generator']['model_name'] = 'g_model_fc_dense_contextnet_YtoX'
            g_model_YtoX = build_fc_dense_contextnet(gen_conf, train_conf)  # generator: B -> A
        elif g_network == 'dilated_densenet':
            gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_real','B1_imag','Epsilon','Rho','Sigma']  #['B1_real','B1_imag','Sigma']
            train_conf['GAN']['generator']['num_classes'] = 5 #1
            train_conf['GAN']['generator']['model_name'] = 'g_model_dilated_densenet_XtoY'
            g_model_XtoY = build_g_dilated_densenet(gen_conf, train_conf)  # generator: A -> B

            gen_conf['dataset_info'][dataset]['image_modality'] = ['sar','sar','sar','sar','sar']
            train_conf['GAN']['generator']['num_classes'] = 5 #1 #3
            train_conf['GAN']['generator']['model_name'] = 'g_model_dilated_densenet_YtoX'
            g_model_YtoX = build_g_dilated_densenet(gen_conf, train_conf)  # generator: B -> A
        elif g_network == 'u_net':
            gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_real','B1_imag','Epsilon','Rho','Sigma']  #['B1_real','B1_imag','Sigma']
            train_conf['GAN']['generator']['num_classes'] = 5 #1
            train_conf['GAN']['generator']['model_name'] = 'g_model_u_net_XtoY'
            g_model_XtoY = build_g_unet(gen_conf, train_conf)  # generator: A -> B

            gen_conf['dataset_info'][dataset]['image_modality'] = ['sar','sar','sar','sar','sar']
            train_conf['GAN']['generator']['num_classes'] = 5 #1 #3
            train_conf['GAN']['generator']['model_name'] = 'g_model_u_net_YtoX'
            g_model_YtoX = build_g_unet(gen_conf, train_conf)  # generator: B -> A
        else:
            raise NotImplementedError('choose fc_dense_contextnet or unet')

        if d_network == 'dilated_densenet':
            #gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_real', 'B1_imag']
            gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_real','B1_imag','Epsilon','Rho','Sigma']  #['B1_mag'] #['B1_real','B1_imag','Sigma']
            train_conf['GAN']['generator']['num_classes'] = 5 #1
            train_conf['GAN']['discriminator']['model_name'] = 'd_model_dilated_densenet_Y'
            d_model_Y, d_model_Y_static = build_d_dilated_densenet(gen_conf, train_conf)  # discriminator: B -> [real/fake]

            #gen_conf['dataset_info'][dataset]['image_modality'] = ['sar', 'sar']
            gen_conf['dataset_info'][dataset]['image_modality'] = ['sar','sar','sar','sar','sar'] #['sar']
            train_conf['GAN']['generator']['num_classes'] = 5 #1 #3
            train_conf['GAN']['discriminator']['model_name'] = 'd_model_dilated_densenet_X'
            d_model_X, d_model_X_static= build_d_dilated_densenet(gen_conf, train_conf)  # discriminator: A -> [real/fake]
        elif d_network == 'patch_gan':
            gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_real','B1_imag','Epsilon','Rho','Sigma'] #['B1_mag'] #['B1_real','B1_imag','Sigma']
            train_conf['GAN']['generator']['num_classes'] = 5 #1
            train_conf['GAN']['discriminator']['model_name'] = 'd_model_patch_gan_Y'
            d_model_Y, d_model_Y_static= build_patch_gan(gen_conf, train_conf)  # discriminator: B -> [real/fake]

            gen_conf['dataset_info'][dataset]['image_modality'] = ['sar','sar','sar','sar','sar'] #['sar']
            train_conf['GAN']['generator']['num_classes'] = 5 #1 #3
            train_conf['GAN']['discriminator']['model_name'] = 'd_model_patch_gan_X'
            d_model_X, d_model_X_static = build_patch_gan(gen_conf, train_conf)  # discriminator: A -> [real/fake]
        elif d_network == 'u_net':
            gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_mag'] #['B1_real','B1_imag','Epsilon','Rho','Sigma'] #['B1_mag'] #['B1_real','B1_imag','Sigma']
            train_conf['GAN']['generator']['num_classes'] = 1 #5 #1
            train_conf['GAN']['discriminator']['model_name'] = 'd_model_patch_gan_Y'
            d_model_Y, d_model_Y_static= build_d_unet(gen_conf, train_conf)  # discriminator: B -> [real/fake]

            gen_conf['dataset_info'][dataset]['image_modality'] = ['sar'] #['sar','sar','sar','sar','sar'] #['sar']
            train_conf['GAN']['generator']['num_classes'] = 1 #5 #1 #3
            train_conf['GAN']['discriminator']['model_name'] = 'd_model_patch_gan_X'
            d_model_X, d_model_X_static = build_d_unet(gen_conf, train_conf)  # discriminator: A -> [real/fake]

        else:
            raise NotImplementedError('choose dilated_densenet or patch_gan')

        # composite: A -> B -> [real/fake, A]
        #gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_real', 'B1_imag']
        gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_mag'] #['B1_real','B1_imag','Epsilon','Rho','Sigma']  #['B1_mag'] #['B1_real','B1_imag','Sigma']
        train_conf['GAN']['generator']['num_classes'] = 1

        if approach == 'single_cyclegan_sar':
            c_model_XtoY = build_single_cyclegan_sar_fwd(gen_conf, train_conf, g_model_XtoY, d_model_Y_static, g_model_YtoX)
            # composite: B -> A -> [real/fake, B]
            c_model_YtoX = build_single_cyclegan_sar_bwd(gen_conf, train_conf, g_model_YtoX, d_model_X_static, g_model_XtoY)

            model = [g_model_XtoY, g_model_YtoX, d_model_Y, d_model_X, c_model_XtoY, c_model_YtoX]
        elif approach == 'multi_cyclegan_sar':
            c_model_XtoY = build_multi_cyclegan_sar_fwd(gen_conf, train_conf, g_model_XtoY, d_model_Y_static, g_model_YtoX)
            # composite: B -> A -> [real/fake, B]
            c_model_YtoX = build_multi_cyclegan_sar_bwd(gen_conf, train_conf, g_model_YtoX, d_model_X_static, g_model_XtoY)

            model = [g_model_XtoY, g_model_YtoX, d_model_Y, d_model_X, c_model_XtoY, c_model_YtoX]
        elif approach == 'composite_cyclegan_sar':
            c_model = build_cyclegan_sar(gen_conf, train_conf, g_model_XtoY, d_model_Y_static, g_model_YtoX, d_model_X_static)
            model = [g_model_XtoY, g_model_YtoX, d_model_Y, d_model_X, c_model]

    print(gen_conf['args'])
    if train_conf['num_retrain'] > 0:
        print('retraining...#' + str(train_conf['num_retrain']))

    return model


