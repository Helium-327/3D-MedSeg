# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/25 11:06:02
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: MLP
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.mlp(x)