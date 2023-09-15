"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import jittor as jt
from jittor import nn
import models.networks as networks
import util.util as util
import warnings

warnings.filterwarnings("ignore")


def get_inst(t):
    edge = jt.init.zero([t.shape[0], 1, t.shape[2], t.shape[3]], int)
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float32()


def change_label(real_A):
    real_A = real_A[:, 0, :, :].unsqueeze(1)
    inst_A = get_inst(real_A)
    nc = 29
    input_label = jt.init.zero([real_A.shape[0], nc, real_A.shape[2], real_A.shape[3]])
    real_A = jt.round(real_A).int8()
    temp = jt.init.one([real_A.shape[0], nc, real_A.shape[2], real_A.shape[3]])
    input_label = input_label.scatter_(1, real_A, temp)
    input_label = jt.concat([input_label, inst_A], 1)
    return input_label, real_A[:, 0, :, :]


class Pix2PixModel(nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = jt.float32
        self.ByteTensor = jt.float32

        self.netG, self.netD, self.netE, self.netS = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.loss.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.loss.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.loss.KLDLoss()

    def execute(self, data, mode):
        input_semantics, real_image, ref_semantics, ref_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated, s_losses = self.compute_generator_loss(input_semantics, real_image, ref_semantics, ref_image)
            return g_loss, generated, s_losses

        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(input_semantics, real_image, ref_semantics, ref_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with jt.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, real_image, ref_semantics, ref_image)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        S_params = list(self.netS.parameters())  # Seg

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr, S_lr = opt.lr, opt.lr, opt.lr
        else:
            G_lr, D_lr, S_lr = opt.lr / 2, opt.lr * 2, opt.lr / 2

        optimizer_G = jt.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = jt.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        if not self.opt.no_seg_loss:
            optimizer_S = jt.optim.Adam(S_params, lr=S_lr, betas=(beta1, beta2))  # Seg
        else:
            optimizer_S = None
        return optimizer_G, optimizer_D, optimizer_S

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        util.save_network(self.netS, 'S', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netS = networks.define_S(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
                netS = util.load_network(netS, 'S', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        # return netG, netD, netE
        return netG, netD, netE, netS

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # change data types
        data['label'] = data['label'].long()
        # data['ref_label'] = data['ref_label'].long()

        # create one-hot label map
        label_map = data['label']
        # ref_label_map = data['ref_label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label else self.opt.label_nc
        input_label = jt.zeros((bs, nc, h, w), dtype=self.FloatTensor)  # [b,29,384,512]

        input_semantics = input_label.scatter_(1, label_map, jt.float32(1.0))
        # ref_semantics = input_label.scatter_(1, ref_label_map, jt.float32(1.0))

        # return input_semantics, data['image'], ref_semantics, data['ref_image']
        return input_semantics, data['image'], None, None

    def compute_generator_loss(self, input_semantics, real_image, ref_semantics, ref_image):
        G_losses = {}
        S_losses = {}

        fake_image, KLD_loss = self.generate_fake(input_semantics, real_image, ref_semantics, ref_image, compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(0.)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg

        if not self.opt.no_seg_loss:
            seg_label = self.netS(fake_image)
            real_A, target_label = change_label(input_semantics)
            S_losses['SEG'] = nn.cross_entropy_loss(seg_label, target_label, ignore_index=255)
            return G_losses, fake_image, S_losses
        else:
            return G_losses, fake_image, None

    def compute_discriminator_loss(self, input_semantics, real_image, ref_semantics, ref_image):
        D_losses = {}
        with jt.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, real_image, ref_semantics, ref_image)

        pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, ref_semantics, ref_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        # if self.opt.use_vae and self.opt.isTrain:
        z, mu, logvar = self.encode_z(real_image)
        if compute_kld_loss:
            KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image = self.netG(input_semantics, real_image, ref_semantics, ref_image, z=z)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = jt.concat([input_semantics, fake_image], dim=1)
        real_concat = jt.concat([input_semantics, real_image], dim=1)

        fake_and_real = jt.concat([fake_concat, real_concat], dim=0)
        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = jt.exp(0.5 * logvar)
        eps = jt.randn_like(std)
        return eps.multiply(std).add(mu)

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
