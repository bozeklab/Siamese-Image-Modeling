# Adapted from: https://github.com/applied-ai-lab/genesis.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import util.blocks as B


class Adios_mask(nn.Module):
    def __init__(self, num_blocks, mask_fbase=32, img_size=64, filter_start=32,
                 in_chnls=4, out_chnls=1, norm='in', N=4):
        super(Adios_mask, self).__init__()

        self.mask_encoder = UNet(
            num_blocks=int(np.log2(img_size)-1),
            img_size=img_size,
            filter_start=filter_start,
            in_chnls=in_chnls,
            out_chnls=out_chnls,
            norm=norm)

        # masking model
        self.mask_head = nn.Sequential(
            nn.Conv2d(mask_fbase, N, 1, 1, 0),
            nn.Softmax(dim=1)
        )


class UNet(nn.Module):

    def __init__(self, num_blocks, img_size=64,
                 filter_start=32, in_chnls=4, out_chnls=1,
                 norm='in'):
        super(UNet, self).__init__()
        # TODO(martin): make more general
        c = filter_start
        if norm == 'in':
            conv_block = B.ConvINReLU
        elif norm == 'gn':
            conv_block = B.ConvGNReLU
        else:
            conv_block = B.ConvReLU
        if num_blocks == 4:
            enc_in = [in_chnls, c, 2*c, 2*c]
            enc_out = [c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c]
            dec_out = [2*c, 2*c, c, c]
        elif num_blocks == 5:
            enc_in = [in_chnls, c, c, 2*c, 2*c]
            enc_out = [c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, c, c, c]
        elif num_blocks == 6:
            enc_in = [in_chnls, c, c, c, 2*c, 2*c]
            enc_out = [c, c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, c, c, c, c]
        self.down = []
        self.up = []
        # 3x3 kernels, stride 1, padding 1
        for i, o in zip(enc_in, enc_out):
            self.down.append(conv_block(i, o, 3, 1, 1))
        for i, o in zip(dec_in, dec_out):
            self.up.append(conv_block(i, o, 3, 1, 1))
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)
        self.featuremap_size = img_size // 2**(num_blocks-1)
        self.mlp = nn.Sequential(
            B.Flatten(),
            nn.Linear(2*c*self.featuremap_size**2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2*c*self.featuremap_size**2), nn.ReLU()
        )
        if out_chnls > 0:
            self.final_conv = nn.Conv2d(c, out_chnls, 1)
        else:
            self.final_conv = nn.Identity()
        self.out_chnls = out_chnls

    def forward(self, x):
        batch_size = x.size(0)
        x_down = [x]
        skip = []
        # Down
        for i, block in enumerate(self.down):
            act = block(x_down[-1])
            skip.append(act)
            if i < len(self.down)-1:
                act = F.interpolate(act, scale_factor=0.5, mode='nearest', recompute_scale_factor=True)
            x_down.append(act)
        # FC
        x_up = self.mlp(x_down[-1])
        x_up = x_up.view(batch_size, -1,
                         self.featuremap_size, self.featuremap_size)
        # Up
        for i, block in enumerate(self.up):
            features = torch.cat([x_up, skip[-1 - i]], dim=1)
            x_up = block(features)
            if i < len(self.up)-1:
                x_up = F.interpolate(x_up, scale_factor=2.0, mode='nearest', recompute_scale_factor=True)
        return self.final_conv(x_up)
