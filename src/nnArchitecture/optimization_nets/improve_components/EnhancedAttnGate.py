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
    
class MultiAxisDualAttnGate(nn.Module):
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
        self.spatial_d_att = nn.Sequential(
            nn.Conv3d(F_inter, 1, 3, padding=1), 
            nn.Softmax(dim=2)
        )
        self.spatial_h_att = nn.Sequential(
            nn.Conv3d(F_inter, 1, 3, padding=1),
            nn.Softmax(dim=3)
        )
        self.spatial_w_att = nn.Sequential(
            nn.Conv3d(F_inter, 1, 3, padding=1),
            nn.Softmax(dim=4)
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
        spatial_d_weight = self.spatial_d_att(fused)
        spatial_h_weight = self.spatial_h_att(fused)
        spatial_w_weight = self.spatial_w_att(fused)
        spatial_weight = spatial_d_weight * spatial_h_weight * spatial_w_weight
        
        channel_weight = self.channel_att(fused)
        return x * (spatial_weight.expand_as(x) + channel_weight.expand_as(x))  # 双路加权
    
    
class MultiAxisDualAttnGate_v2(nn.Module):
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
        self.spatial_d_att = nn.Conv3d(F_inter, 1, 3, padding=1), 
        self.spatial_h_att = nn.Conv3d(F_inter, 1, 3, padding=1),
        self.spatial_w_att = nn.Conv3d(F_inter, 1, 3, padding=1),

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
        spatial_d_feature = self.spatial_d_att(fused) 
        spatial_d_layerNorm = F.layer_norm(spatial_d_feature, x.size()[1:])
        spatial_d_weight = F.softmax(spatial_d_layerNorm, dim=2)
        
        spatial_h_feature = self.spatial_h_att(fused)
        spatial_h_layerNorm = F.layer_norm(spatial_h_feature, x.size()[1:])
        spatial_h_weight = F.softmax(spatial_h_layerNorm, dim=3)
        
        spatial_w_feature = self.spatial_w_att(fused)
        spatial_w_layerNorm = F.layer_norm(spatial_w_feature, x.size()[1:])
        spatial_w_weight = F.softmax(spatial_w_layerNorm, dim=4)
        
        spatial_weight = spatial_d_weight * spatial_h_weight * spatial_w_weight
        
        channel_weight = self.channel_att(fused)
        return x * (spatial_weight.expand_as(x) + channel_weight.expand_as(x))  # 双路加权
    
