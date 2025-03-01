# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/28 14:18:13
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 强化版AttentionGate
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedAttnGate(nn.Module):
    def __init__(self, F_g, F_l, F_inter):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_inter, kernel_size=1),
            nn.InstanceNorm3d(F_inter)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_inter, kernel_size=1),
            nn.InstanceNorm3d(F_inter)
        )
        # 空间注意力分支
        self.spatial_att = nn.Sequential(
            nn.Conv3d(F_inter, 1, 3, padding=1), 
            nn.Softmax(dim=2)
        )
        # 通道注意力分支
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(F_inter, F_inter//4, 1),
            nn.Conv3d(F_inter//4, F_inter, 1),
            nn.Conv3d(F_inter, F_g, 1),
            nn.Sigmoid()
        )
        
    def forward(self, g, x):
        fused = F.relu(self.W_g(g) + self.W_x(x))
        
        # 空间注意力和通道注意力机制
        spatial_weight = self.spatial_att(fused)
        channel_weight = self.channel_att(fused)
        return x * (spatial_weight.expand_as(x) + channel_weight.expand_as(x))  # 双路加权