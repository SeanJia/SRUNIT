import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from torch.autograd import grad
from torch import autograd


class SRUNITModel(BaseModel):
    """ This class implements SRUNIT model, described in the paper
    Semantically Robust Unpaired Image Translation for Data with Unmatched Semantics Statistics
    Zhiwei Jia, Bodi Yuan, Kangkang Wang, Hong Wu, David Clifford, Zhiqiang Yuan, Hao Su
    ICCV 2021

    The code is largely adapted from the PyTorch implementation of CUT
    https://github.com/taesungp/contrastive-unpaired-translation (models/cut_model.py)
    Namely, it equips CUT model with our proposed multi-scale semantic robustness loss.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model

        Configs w.r.t. semantic robustness loss are in options/base_options.py
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--nfdf', type=int, default=64, help='# of feature discrim filters in the first conv layer')
        parser.add_argument('--reg_layers', type=str, default='0,1,2,3,4', help='semantic robustness regularization on which layers')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.relu = torch.nn.ReLU()

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if self.opt.isTrain:
            self.reg = float(opt.reg)
            if self.reg > 0.0: 
                assert self.opt.netF == 'mlp_sample'
                assert self.opt.reg_type
                assert self.opt.reg_noise > 0.0
                self.loss_names += ['reg']

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
            opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(
            opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
            opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        # define loss functions
        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
        self.criterionNCE = []
        for nce_layer in self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

        if self.isTrain:
            self.netD = networks.define_D(
                opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                opt.init_gain, opt.no_antialias, self.gpu_ids, opt, noise=opt.D_noise)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
    def data_dependent_initialize(self, data, infer_mode=False):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward(infer_init=infer_mode)                     # compute fake images: G(A)

        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                  # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(
                    self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self, curr_epoch):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss() # G detached.
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netF, True)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()

        if self.reg > 0.0: # The Semantic Robustness Regularization in SRUNIT.
            if curr_epoch > self.opt.inact_epochs:
                self.optimizer_F.zero_grad() # self.opt.netF must be 'mlp_sample' here.

                if self.opt.reg_type == 'v1':
                    self.loss_G.backward(retain_graph=True)
                    self.set_requires_grad(self.netF, False)
                    self.loss_reg = self.compute_reg_loss() 
                    loss = self.reg * self.loss_reg 
                elif self.opt.reg_type == 'v2':
                    self.loss_reg = self.compute_reg_loss() 
                    loss = self.loss_G + self.reg * self.loss_reg 
                else:
                    assert False, self.opt.reg_type

                loss.backward()
                self.optimizer_G.step()
                self.optimizer_F.step()
            else:
                self.loss_reg = 0.0
        else:
            if self.opt.netF == 'mlp_sample':
                self.optimizer_F.zero_grad() # Clean the gradient before loss_G.backward()
            self.loss_G.backward()
            self.optimizer_G.step()
            if self.opt.netF == 'mlp_sample':
                self.optimizer_F.step()            

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self, infer_init=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) \
                if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.real_A = self.real[:self.real_A.size(0)]
        self.real_B = self.real[self.real_A.size(0):]
        self.fake_B, self.feats_real_A = self.netG(self.real_A, self.nce_layers)
        self.feats_fake_B = self.netG(self.fake_B, self.nce_layers, encode_only=True)

        if self.isTrain and self.reg > 0.0:

            # Only choose one layer each time to speed up SRUNIT.
            choices = [int(l) for l in self.opt.reg_layers.split(',')]
            self.choice = np.random.choice(choices, 1)[0] 
            self.feats_perturbed, self.noise_magnitude = self.netG(
                layers=[self.nce_layers[self.choice]],
                feats=[self.feats_real_A[self.choice]],
                noises=[self.opt.reg_noise])

        if self.opt.isTrain and self.opt.nce_idt:
            self.idt_B, self.feats_real_B = self.netG(self.real_B, self.nce_layers)

    def compute_reg_loss(self):

        # Distance function.
        def euc_dis(x, y, dim=-1):
            d = x - y
            d = d.pow(2).sum(dim) + 1e-12
            return d.pow(0.5)

        # To speed up SRUNIT, we only sample patches to compute the semantic robustness loss.
        feat_k_pool, sample_ids = self.netF(
            [self.feats_real_A[self.choice]], self.opt.num_patches, None, use_mlp=True, choice=self.choice)
        feat_q_pool, _ = self.netF(
            self.feats_perturbed, self.opt.num_patches, sample_ids, use_mlp=True, choice=self.choice)
        total_reg_loss = 0.0
        for f_q, f_k, noise_mag, samples in zip(feat_q_pool, feat_k_pool, self.noise_magnitude, sample_ids):
            noise_mag = (noise_mag.flatten(1, 3)[:, samples]).flatten(0, 1) # to support batch_size > 1 per gpu.
            loss = euc_dis(f_q, f_k) / noise_mag
            total_reg_loss += loss.mean()
        return total_reg_loss / len(self.nce_layers) # actually len(self.nce_layers) == 1 here.

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(self.pred_real, True).mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.feats_real_A, self.feats_fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.feats_idt_B = self.netG(self.idt_B, self.nce_layers, encode_only=True)
            self.loss_NCE_Y = self.calculate_NCE_loss(self.feats_real_B, self.feats_idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)

        assert not (self.opt.flip_equivariance and self.flipped_for_equivariance)
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k_pool, sample_ids = self.netF(src, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(tgt, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(
                feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers
