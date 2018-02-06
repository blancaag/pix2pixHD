### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks_ as networks

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else 3

        ##### define networks
        # Generator network
        netG_input_nc = input_nc
        if not opt.no_instance: netG_input_nc += 1
        if self.use_features: netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance: netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder',
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)

        print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss() # adding L1 G loss
            self.criterionCos1 = networks.CosineLoss1() # adding cosine G loss
            self.criterionCos2 = networks.CosineLoss2() # adding cosine G loss            
            # self.criterionCEL = torch.nn.CosineEmbeddingLoss() # adding cosine G loss by torch.nn
           
            criterion = 'param'; KL = 'qp'
            if criterion == 'param':
                # print('Using parametric criterion KL_%s' % KL)
                # KL_minimizer = losses.KLN01Loss(direction=opt.KL, minimize=True)
                # KL_maximizer = losses.KLN01Loss(direction=opt.KL, minimize=False)
            
                self.criterionKL_min = networks.KLN01Loss(direction=KL, minimize=True)
                self.criterionKL_max = networks.KLN01Loss(direction=KL, minimize=False)
            
            if not opt.no_vgg_loss: self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = ['G_cos1_z', 'G_cos2_z', 'G_cos1', 'G_cos2', 
                               'E_KL_real', 'E_KL_fake', 'G_KL_fake', 'G_L1', 
                               'G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake'] # added G_cos1, G_cos2, G_L1, G_KL, E_KL (E_KL_real, E_KL_fake)
            
            self.loss_weights = [opt.lambda_G_cos1_z, opt.lambda_G_cos2_z, opt.lambda_G_cos1, opt.lambda_G_cos2, 
                                 opt.lambda_E_KL_real, opt.lambda_E_KL_fake, opt.lambda_G_KL_fake, opt.lambda_L1,
                                 1.0, opt.lambda_feat, opt.lambda_feat, 0.5, 0.5 ]
                                 
            print('===================== LOSSES =====================')
            [print ('%s: %.2f' %(i, j)) for i, j in zip(self.loss_names, self.loss_weights)]
            print('==================================================')
            
            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [{'params':[value],'lr':opt.lr}]
                    else:
                        params += [{'params':[value],'lr':0.0}]
            else:
                params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
         

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])

            if self.gpu_ids == '-1': #with CPU:
                input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_label = input_label.scatter_(1, label_map.data.long(), 1.0)
            else:
                input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            
        # get edges from instance map
        if not self.opt.no_instance:
            # inst_map = inst_map.data.cuda()
            if self.gpu_ids == '-1': inst_map = inst_map.data
            else: inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())

        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, inst, image, feat, infer=False):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat)
        # print(input_label.shape, inst_map.shape, real_image.shape, feat_map.shape)
        # torch.Size([1, 3, 256, 256]) torch.Size([1]) torch.Size([1, 3, 256, 256]) torch.Size([1])
        
        # Fake Generation
        if self.use_features: # def. false
            if not self.opt.load_features: # def. false
                feat_map = self.netE.forward(real_image, inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1) 
        else:
            input_concat = input_label 
        # print(input_concat.shape)
        fake_image = self.netG.forward(input_concat)
        fake_image_detached = self.netG.forward(input_concat).detach()

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss: loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat

        # adding L1
        # fake_image = self.netG.forward(input_concat)
        loss_G_L1 = self.criterionL1(fake_image, real_image) * self.opt.lambda_L1 
        
        # adding KL
        self.netGE = self.netG.model_down
                  
        loss_E_KL_real = self.criterionKL_min(self.netGE(real_image)) * self.opt.lambda_E_KL_real
        loss_E_KL_fake = self.criterionKL_max(self.netGE(fake_image.detach())) * self.opt.lambda_E_KL_fake
        loss_G_KL_fake = self.criterionKL_min(self.netGE(fake_image)) * self.opt.lambda_G_KL_fake;

        # adding Cosine loss
        loss_G_cos1 = self.criterionCos1(fake_image, real_image) * self.opt.lambda_G_cos1
        loss_G_cos2 = self.criterionCos2(fake_image, real_image) * self.opt.lambda_G_cos2          
        loss_G_cos1_z = self.criterionCos1(self.netGE(fake_image), self.netGE(real_image)) * self.opt.lambda_G_cos1_z
        loss_G_cos2_z = self.criterionCos2(self.netGE(fake_image), self.netGE(real_image)) * self.opt.lambda_G_cos2_z   

        # Only return the fake_B image if necessary to save BW
        return [[loss_G_cos1_z, loss_G_cos2_z, loss_G_cos1, loss_G_cos2, 
                 loss_E_KL_real, loss_E_KL_fake, loss_G_KL_fake, loss_G_L1, 
                 loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake], None if not infer else fake_image]

    def inference(self, label, inst):
        # Encode Inputs
        input_label, inst_map, _, _ = self.encode_input(Variable(label), Variable(inst), infer=True)

        # Fake Generation
        if self.use_features:
            # sample clusters from precomputed features
            feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else: input_concat = input_label
        fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst):
        # read precomputed feature clusters
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path).item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = torch.cuda.FloatTensor(1, self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])
                idx = (inst == i).nonzero()
                for k in range(self.opt.feat_num): feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc): feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == i).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        # edge = torch.cuda.ByteTensor(t.size()).zero_()
        if self.gpu_ids == '-1': edge = torch.ByteTensor(t.size()).zero_()
        else: edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features: self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd; print(lr, lrd)
        for param_group in self.optimizer_D.param_groups: param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups: param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr