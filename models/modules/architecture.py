import math
import torch
import torchvision
import torch.nn as nn
from . import block as B


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
                B.conv_block(nf, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf * 2, kernel_size=3, pad_type='reflect'))
        self.EB_2 = B.sequential(
                B.conv_block(nf * 2, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf * 2, kernel_size=3, pad_type='reflect'),
                B.downsample_block(nf * 2, downs_factor=2, pool_type='max'))
        self.EB_3 = B.sequential(
                B.conv_block(nf * 2, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf * 2, kernel_size=3, pad_type='reflect'),
                B.downsample_block(nf * 2, downs_factor=2, pool_type='max'))
        self.EB_4 = B.sequential(
                B.conv_block(nf * 2, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf, nf * 2, kernel_size=3, pad_type='reflect'),
                B.downsample_block(nf * 2, downs_factor=2, pool_type='max'))
        self.DB_4 = B.sequential(
                B.conv_block(nf * 2, nf * 2, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf * 2, nf * 2, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf * 2, nf * 2, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf * 2, nf * 2, kernel_size=3, pad_type='reflect'),
                B.pixelshuffle_block(nf * 2, nf * 2, upscale_factor=2, pad_type='reflect'))
        self.DB_3 = B.sequential(
                B.conv_block(nf * 4, nf * 2, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf * 2, nf * 2, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf * 2, nf * 2, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf * 2, nf * 2, kernel_size=3, pad_type='reflect'),
                B.pixelshuffle_block(nf * 2, nf * 2, upscale_factor=2, pad_type='reflect'))
        self.DB_2 = B.sequential(
                B.conv_block(nf * 4, nf * 2, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf * 2, nf * 2, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf * 2, nf * 2, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf * 2, nf * 2, kernel_size=3, pad_type='reflect'),
                B.pixelshuffle_block(nf * 2, nf * 2, upscale_factor=2, pad_type='reflect'))
        self.DB_1 = B.sequential(
                B.conv_block(nf * 4, nf * 2, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf * 2, nf * 2, kernel_size=3, pad_type='reflect'),
                B.conv_block(nf * 2, nf * 2, kernel_size=3, pad_type='reflect'),
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


