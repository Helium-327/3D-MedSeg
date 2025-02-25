# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/16 15:44:14
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 
*      VERSION: v1.0
*      FEATURES: Attention UNet3D + ResConv + DenseASPP + SCGA()
=================================================
'''


import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.utils.test_unet import test_unet
from src.nnArchitecture.optimization_nets.improve_components.scga import SCGA

def init_weights_3d(m):
    """Initialize 3D卷积和BN层的权重"""
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

"""================================== 改进模块 =================================================="""
# 改进1：DoubleConv --> ResConv3D


class SE(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(SE, self).__init__()

        if in_channels // reduction_ratio <= 0:
                    raise ValueError(f"Reduction ratio {reduction_ratio} is too large for the number of input channels {in_channels}.")
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)
    
# class SCGA(nn.Module):
#     """Spatial Correlation Grouped Attention (空间相关性分组注意力机制)"""
#     def __init__(self, channels, factor=16):
#         super(SCGA, self).__init__()
#         self.group = factor
#         assert channels // self.group > 0
#         self.softmax = nn.Softmax(dim=-1)
#         self.averagePooling = nn.AdaptiveAvgPool3d((1, 1, 1))  # 3D 全局平均池化
#         self.maxPooling = nn.AdaptiveMaxPool3d((1, 1, 1))      # 3D 全局最大池化
#         self.Pool_h = nn.AdaptiveAvgPool3d((None, 1, 1))       # 高度方向池化
#         self.Pool_w = nn.AdaptiveAvgPool3d((1, None, 1))       # 宽度方向池化
#         self.Pool_d = nn.AdaptiveAvgPool3d((1, 1, None))       # 深度方向池化

#         self.groupNorm = nn.GroupNorm(channels // self.group, channels // self.group)
#         self.conv1x1x1 = nn.Sequential(
#             nn.Conv3d(channels // self.group, channels // self.group, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm3d(channels // self.group),
#             nn.ReLU(inplace=True)
#         )
#         self.conv3x3x3 = nn.Sequential(
#             nn.Conv3d(channels // self.group, channels // self.group, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm3d(channels // self.group),
#             nn.ReLU(inplace=True)
#         )
#         self.Channels_attention = SE(channels // self.group)

#     def forward(self, x):
#         b, c, d, h, w = x.size()
#         group_x = x.reshape(b * self.group, -1, d, h, w)  # 分组处理

#         # 高度、宽度、深度方向池化
#         x_h = self.Pool_h(group_x)  # [B*G, C/G, D, 1, 1]
#         x_w = self.Pool_w(group_x).permute(0, 1, 3, 2, 4)  # [B*G, C/G, 1, H, 1]
#         x_d = self.Pool_d(group_x).permute(0, 1, 4, 3, 2)  # [B*G, C/G, 1, 1, W]

#         # 拼接并卷积
#         hwd = self.conv1x1x1(torch.cat([x_h, x_w, x_d], dim=2))  # 拼接后卷积
#         x_h, x_w, x_d = torch.split(hwd, [d, h, w], dim=2)       # 拆分

#         # Apply sigmoid activation
#         x_h_sigmoid = F.sigmoid(x_h).view(b*self.group, c // self.group, d, 1, 1)
#         x_w_sigmoid = F.sigmoid(x_w).view(b*self.group, c // self.group, 1, h, 1)
#         x_d_sigmoid = F.sigmoid(x_d).view(b*self.group, c // self.group, 1, 1, w)
        
#         # Apply attention maps using broadcasting
#         x_attended = group_x * x_h_sigmoid * x_w_sigmoid * x_d_sigmoid
        
#         x1 = self.groupNorm(group_x * x_attended)  # 高度、宽度、深度注意力
#         x11 = self.softmax(self.averagePooling(x1).reshape(b * self.group, -1, 1).permute(0, 2, 1))  # 全局平均池化 + softmax
#         x12 = x1.reshape(b * self.group, c // self.group, -1)

#         # 3x3x3 路径
#         x2 = self.Channels_attention(self.conv3x3x3(group_x))  # 通过 3x3x3 卷积层
#         x21 = self.softmax(self.averagePooling(x2).reshape(b * self.group, -1, 1).permute(0, 2, 1))  # 全局平均池化 + softmax
#         x22 = x2.reshape(b * self.group, c // self.group, -1)

#         # 计算权重
#         weights = (torch.matmul(x11, x22) + torch.matmul(x21, x12)).reshape(b * self.group, -1, d, h, w)
#         return (group_x * weights.sigmoid()).reshape(b, c, d, h, w)

class SCGA(nn.Module):
    def __init__(self, channels, factor=8):
        super(SCGA, self).__init__()
        self.group = factor
        assert channels // self.group > 0
        self.softmax = nn.Softmax(dim=-1)
        self.averagePooling = nn.AdaptiveAvgPool3d((1, 1, 1))  # 3D 全局平均池化
        self.maxPooling = nn.AdaptiveMaxPool3d((1, 1, 1))      # 3D 全局最大池化
        self.Pool_h = nn.AdaptiveAvgPool3d((None, 1, 1))       # 高度方向池化
        self.Pool_w = nn.AdaptiveAvgPool3d((1, None, 1))       # 宽度方向池化
        self.Pool_d = nn.AdaptiveAvgPool3d((1, 1, None))       # 深度方向池化

        self.groupNorm = nn.GroupNorm(1, channels // self.group)
        self.conv1x1x1 = nn.Sequential(
            nn.Conv3d(channels // self.group, channels // self.group, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels // self.group, channels // self.group),
            nn.ReLU(inplace=True)
        )
        self.conv3x3x3 = nn.Sequential(
            nn.Conv3d(channels // self.group, channels // self.group, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(channels // self.group, channels // self.group),
            nn.ReLU(inplace=True)
        )
        self.channels_attention = SE(channels // self.group)

    def forward(self, x):
        b, c, d, h, w = x.size()
        group_x = x.reshape(b * self.group, -1, d, h, w)  # 分组处理

        # 高度、宽度、深度方向池化
        x_c = self.maxPooling(group_x)  # [B*G, C/G, 1, 1, 1]
        x_h = self.Pool_h(group_x)  # [B*G, C/G, D, 1, 1]
        x_w = self.Pool_w(group_x).permute(0, 1, 3, 2, 4)  # [B*G, C/G, 1, H, 1]
        x_d = self.Pool_d(group_x).permute(0, 1, 4, 3, 2)  # [B*G, C/G, 1, 1, W]

        # 拼接并卷积
        hwd = self.conv1x1x1(torch.cat([x_h, x_w, x_d], dim=2))  # 拼接后卷积
        x_h, x_w, x_d = torch.split(hwd, [d, h, w], dim=2)       # 拆分

        # Apply sigmoid activation
        x_h_sigmoid = x_h.sigmoid().view(b*self.group, c // self.group, d, 1, 1)
        x_w_sigmoid = x_w.sigmoid().view(b*self.group, c // self.group, 1, h, 1)
        x_d_sigmoid = x_d.sigmoid().view(b*self.group, c // self.group, 1, 1, w)
        
        # Apply attention maps using broadcasting
        x_attended = x_h_sigmoid * x_w_sigmoid * x_d_sigmoid
        
        x1 = self.groupNorm(group_x * x_attended)  # 高度、宽度、深度注意力
        x11 = self.softmax(self.averagePooling(x1).reshape(b * self.group, -1, 1).permute(0, 2, 1))  # 全局平均池化 + softmax
        x12 = x1.reshape(b * self.group, c // self.group, -1)

        # 3x3x3 路径
        x2 = self.conv3x3x3(group_x)  # 通过 3x3x3 卷积层
        x21 = self.softmax(self.averagePooling(x2).reshape(b * self.group, -1, 1).permute(0, 2, 1))  # 全局平均池化 + softmax
        x22 = x2.reshape(b * self.group, c // self.group, -1)

        # 计算权重
        weights = (torch.matmul(x11, x22) + torch.matmul(x21, x12)).reshape(b * self.group, -1, d, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, d, h, w)


class ResConv3D(nn.Module):
    """带残差连接的各向异性卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            # SCGA(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
        )
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.shortcut:
            residual = self.shortcut(residual)
        out += residual
        out = self.relu(out)
        return out

class DenseASPP3D(nn.Module):
    def __init__(self, in_channels, out_channels, reduce_rate=4, dilations=[1, 3, 5, 7]):
        super(DenseASPP3D, self).__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels//reduce_rate, kernel_size=3, padding=dilations[0], dilation=dilations[0]),
            nn.BatchNorm3d(out_channels//reduce_rate),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv3d(in_channels + out_channels//reduce_rate, out_channels//reduce_rate, kernel_size=3, padding=dilations[1], dilation=dilations[1]),
            nn.BatchNorm3d(out_channels//reduce_rate),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv3d(in_channels + 2*(out_channels//reduce_rate), out_channels//reduce_rate, kernel_size=3, padding=dilations[2], dilation=dilations[2]),
            nn.BatchNorm3d(out_channels//reduce_rate),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv3d(in_channels + 3*(out_channels//reduce_rate), out_channels//reduce_rate, kernel_size=3, padding=dilations[3], dilation=dilations[3]),
            nn.BatchNorm3d(out_channels//reduce_rate),
            nn.ReLU(inplace=True)
        )
        self.global_avg = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels//reduce_rate, kernel_size=1),
            nn.ReLU(inplace=True)
        ) 
        self.fusion = nn.Conv3d(5*(out_channels//reduce_rate), out_channels, 1)
        
        # self.conv1x1 = nn.Conv3d(in_channels, out_channels, 1)
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(torch.cat([x, x1], 1))
        x3 = self.aspp3(torch.cat([x, x1, x2], 1))
        x4 = self.aspp4(torch.cat([x, x1, x2, x3], 1))
        x5 = self.global_avg(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='trilinear', align_corners=True)
        fusion_x = torch.cat([x1, x2, x3, x4, x5], 1)
        x = self.fusion(fusion_x)
        return x

"""====================================== 原网络 ================================================= """

class AttentionBlock3D(nn.Module):
    """3D Attention Gate"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.attn = SCGA(F_int)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # psi = self.relu(g1 + x1)
        psi = self.attn(self.relu(g1 + x1))
        psi = self.psi(psi)
        return x * psi

class UpSample(nn.Module):
    """3D Up Convolution"""
    def __init__(self, in_channels, out_channels, trilinear=True):
        super(UpSample, self).__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # self.conv = ResConv3D(in_channels, out_channels)

    def forward(self, x):
        return self.up(x)

class ScgaResAtteUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, f_list=[32, 64, 128, 256], trilinear=True):
        super(ScgaResAtteUNet, self).__init__()
        
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.Conv1 = ResConv3D(in_channels, f_list[0])
        self.down_atte1 = SCGA(f_list[0])
        self.Conv2 = ResConv3D(f_list[0], f_list[1])
        self.down_atte2 = SCGA(f_list[1])
        self.Conv3 = ResConv3D(f_list[1], f_list[2])
        self.down_atte3 = SCGA(f_list[2])
        self.Conv4 = ResConv3D(f_list[2], f_list[3])
        self.down_atte4 = SCGA(f_list[3])
        
        self.bottleneck = DenseASPP3D(f_list[3], f_list[3])
        
        self.Up5 = UpSample(f_list[3], f_list[3], trilinear)
        self.Att5 = AttentionBlock3D(F_g=f_list[3], F_l=f_list[3], F_int=f_list[3]//2)
        self.UpConv5 = ResConv3D(f_list[3]*2, f_list[3]//2)
        self.up_atte5 = SCGA(f_list[3]//2)
        
        self.Up4 = UpSample(f_list[2], f_list[2], trilinear)
        self.Att4 = AttentionBlock3D(F_g=f_list[2], F_l=f_list[2], F_int=f_list[2]//2)
        self.UpConv4 = ResConv3D(f_list[2]*2, f_list[2]//2)
        self.up_atte4 = SCGA(f_list[2]//2)
        
        self.Up3 = UpSample(f_list[1], f_list[1], trilinear)
        self.Att3 = AttentionBlock3D(F_g=f_list[1], F_l=f_list[1], F_int=f_list[1]//2)
        self.UpConv3 = ResConv3D(f_list[1]*2, f_list[1]//2)
        self.up_atte3 = SCGA(f_list[1]//2)
        
        self.Up2 = UpSample(f_list[0], f_list[0], trilinear)
        self.Att2 = AttentionBlock3D(F_g=f_list[0], F_l=f_list[0], F_int=f_list[0]//2)
        self.UpConv2 = ResConv3D(f_list[0]*2, f_list[0])
        self.up_atte2 = SCGA(f_list[0])
        
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)

        self.apply(init_weights_3d)
        
    def forward(self, x):
        # Encoder
        x1 = self.Conv1(x)       # [B, 32, D, H, W]
        x1 = self.down_atte1(x1)
        
        x2 = self.MaxPool(x1)
        x2 = self.Conv2(x2)      # [B, 64, D/2, H/2, W/2]
        x2 = self.down_atte2(x2)
        
        x3 = self.MaxPool(x2)
        x3 = self.Conv3(x3)      # [B, 128, D/4, H/4, W/4]
        x3 = self.down_atte3(x3)
        
        x4 = self.MaxPool(x3)
        x4 = self.Conv4(x4)      # [B, 256, D/8, H/8, W/8]
        x4 = self.down_atte4(x4)
        
        x5 = self.MaxPool(x4)
        
        # Bottleneck
        # x5 = self.bottleneck(x5)      # [B, 256, D/16, H/16, W/16]
        
        # Decoder with Attention
        d5 = self.Up5(x5)        # [B, 256, D/8, H/8, W/8]
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpConv5(d5)    # [B, 128, D/8, H/8, W/8]
        d5 = self.up_atte5(d5)
        
        d4 = self.Up4(d5)        # [B, 128, D/4, H/4, W/4]
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpConv4(d4)    # [B, 64, D/4, H/4, W/4]
        # d4 = self.up_atte4(d4)
        
        d3 = self.Up3(d4)        # [B, 64, D/2, H/2, W/2]
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpConv3(d3)    # [B, 32, D/2, H/2, W/2]
        d3 = self.up_atte3(d3)
        
        d2 = self.Up2(d3)        # [B, 32, D, H, W]
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.UpConv2(d2)    # [B, 32, D, H, W]
        d2 = self.up_atte2(d2)
        
        out = self.outc(d2)  # [B, out_channels, D, H, W]
        return out
            
if __name__ == '__main__':

    test_unet(model_class=ScgaResAtteUNet)
    # dense_aspp = DenseASPP3D(in_channels=64, out_channels=64)
    # print(dense_aspp)