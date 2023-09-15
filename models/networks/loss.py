"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import jittor as jt
from jittor import nn
from models.networks.architecture import VGG19
from models.deeplab_jittor.deeplab import DeepLab


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=jt.float32, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(0.)
            # self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = nn.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return nn.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = jt.minimum(input - 1, self.get_zero_tensor(input))
                    loss = -jt.mean(minval)
                else:
                    minval = jt.minimum(-input - 1,
                                        self.get_zero_tensor(input))
                    loss = -jt.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -jt.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(
                    pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = jt.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def execute(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class SEGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(SEGLoss, self).__init__()
        self.seg = DeepLab(output_stride=16, num_classes=29)
        self.seg.load("./pretrained/Epoch_40.pkl")
        self.seg.eval()

    def execute(self, fake_image, input_semantics):
        real_A, target_label = change_label(input_semantics)
        fake_label = self.seg(fake_image)
        loss = 0
        for i in range(len(fake_label)):
            loss += nn.cross_entropy_loss(fake_label, target_label, ignore_index=255)
        return loss

class P2PLoss(nn.Module):
    def __init__(self, weight):
        super(P2PLoss, self).__init__()
        self.weight = weight

    def execute(self, output_pixel, embedding_pixel):
        n, c, h, w = embedding_pixel.shape
        embedding_pixel = embedding_pixel.view(n, c, h * w).permute(0, 2, 1)
        output_pixel = output_pixel.view(n, c, h * w).permute(0, 2, 1)
        z1, z2 = embedding_pixel.split(n // 2, dim=0)
        p1, p2 = output_pixel.split(n // 2, dim=0)

        z1 = jt.normalize(z1, p=2, dim=-1)  # [B HW C]
        z2 = jt.normalize(z2, p=2, dim=-1)
        p1 = jt.normalize(p1, p=2, dim=-1)
        p2 = jt.normalize(p2, p=2, dim=-1)
        loss = -(
                (p1 * z2.detach()).sum(dim=-1).mean() +
                (p2 * z1.detach()).sum(dim=-1).mean()
        ) * 0.5
        return loss *self.weight


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


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def execute(self, mu, logvar):
        return -0.5 * jt.sum(1 + logvar - mu.pow(2) - logvar.exp())
