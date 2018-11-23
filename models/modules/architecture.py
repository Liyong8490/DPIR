import math
import torch
import torchvision
import torch.nn as nn
from . import block as B
from models.models_DnCNN import DnCNN


class UNet(nn.Module):
    def __init__(self,
                 in_nc,
                 out_nc,
                 norm_type=None,
                 nf=64):
        super(UNet, self).__init__()
        self.EB_1 = B.sequential(
                B.conv_block(in_nc, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf * 2, kernel_size=3, pad_type='reflect'))
        self.EB_2 = B.sequential(
                B.conv_block(nf * 2, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf * 2, kernel_size=3, pad_type='reflect'),
                B.downsample_block(nf * 2, downs_factor=2, pool_type='max'))
        self.EB_3 = B.sequential(
                B.conv_block(nf * 2, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf * 2, kernel_size=3, pad_type='reflect'),
                B.downsample_block(nf * 2, downs_factor=2, pool_type='max'))
        self.EB_4 = B.sequential(
                B.conv_block(nf * 2, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf * 2, kernel_size=3, pad_type='reflect'),
                B.downsample_block(nf * 2, downs_factor=2, pool_type='max'))
        self.DB_4 = B.sequential(
                B.conv_block(nf * 2, nf * 2, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf * 2, nf * 2, kernel_size=3, pad_type='reflect'),
                B.pixelshuffle_block(nf * 2, nf * 2, upscale_factor=2, pad_type='reflect'))
        self.DB_3 = B.sequential(
                B.conv_block(nf * 4, nf * 2, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf * 2, nf * 2, kernel_size=3, pad_type='reflect'),
                B.pixelshuffle_block(nf * 2, nf * 2, upscale_factor=2, pad_type='reflect'))
        self.DB_2 = B.sequential(
                B.conv_block(nf * 4, nf * 2, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf * 2, nf * 2, kernel_size=3, pad_type='reflect'),
                B.pixelshuffle_block(nf * 2, nf * 2, upscale_factor=2, pad_type='reflect'))
        self.DB_1 = B.sequential(
                B.conv_block(nf * 4, nf * 2, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf * 2, out_nc, kernel_size=3, pad_type='reflect'))

    def forward(self, x):
        eb_1 = self.EB_1(x)
        eb_2 = self.EB_2(eb_1)
        eb_3 = self.EB_3(eb_2)
        eb_4 = self.EB_4(eb_3)
        db_4 = self.DB_4(eb_4)
        db_3 = self.DB_3(torch.cat((db_4, eb_3), dim=1))
        db_2 = self.DB_2(torch.cat((db_3, eb_2), dim=1))
        db_1 = self.DB_1(torch.cat((db_2, eb_1), dim=1))
        return db_1 + x


class DPIR(nn.Module):
    """
    Denoising Prior Driven Deep Neural Network for Image Restoration
    Weisheng Dong, Peiyao Wang, Wotao Yin, et. al. ArXiv: 1801.06756
    Image Restoration formulation:
                x,v = argmax_(x,v) 0.5*|y-Ax|^2 + lambda*J(v)
                        s.t. x=v
            apply half-quadratic splitting method:
                x,v = argmax_(x,v) 0.5*|y-Ax|^2 + eta*|x-v|^2 + lambda*J(v)
            following two sub-problems:
                1: x = argmax_x |y - Ax|^2 + eta*|x - v|^2
                2: v = argmax_v eta*|x - v|^2 + lambda*J(v)
            where sub-problem 2 is a typical denoising problem which can be replaced by a CNN denoiser.
    Using conjugate gradient algorithm to solve sub-problem 1:
            x   = x - delta*[A'(Ax - y) - eta*(x - v)]
                = A_t*x + delta*A'y + delta*v
            where A_t = [(1 - delta*eta)I - delta*A'A]
    In the problem of denoising:
            A is selected as a identify operator. Thus:
                A_t = (1 - delta*(eta + 1))I ,
                A' = A = I .
    """
    def __init__(self, input_chnl, K=5, nf=64, groups=1):
        super(DPIR, self).__init__()
        self.K = K
        self.denoiser = DnCNN(input_chnl, nf, groups)
        self.delta = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.eta = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.delta.data.uniform_(0.01, 0.1)
        self.eta.data.uniform_(0.01, 0.1)

    def forward(self, y):
        delta_AT_y = self.delta * y
        A_t = 1 - self.delta * (self.eta + 1)
        x = A_t * y + delta_AT_y
        for _ in range(self.K):
            v = self.denoiser(x)
            delta_v = self.delta * v
            x = A_t * x + delta_AT_y + delta_v
        return x
