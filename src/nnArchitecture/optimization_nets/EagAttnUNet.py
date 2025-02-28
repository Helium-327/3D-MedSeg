# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/28 14:27:00
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: AttnUNet: AG ---> EAG
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''


import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.utils.test_unet import test_unet
# torch.autograd.set_detect_anomaly(True)

from src.nnArchitecture.optimization_nets.improve_components.EnhancedAttnGate import EnhancedAttnGate


def init_weights_3d(m):
    """Initialize 3D卷积和BN层的权重"""
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class CrossModalityFusionGate(nn.Module):
    """跨轴动态融合模块"""
    def __init__(self, mod_num=4):
        super().__init__()
        
        self.mod_num = mod_num
        self.scale = nn.Parameter(torch.ones(1))  # 可学习缩放参数
         
        self.gate = nn.Sequential(
            nn.Conv3d(mod_num, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(1, mod_num, 1),
            nn.Softmax(dim=1)
        )
        self.apply(init_weights_3d) 
        # self.axial_attns = AxialAttention3D(in_ch=mod_num, heads=4)

        # self.conv1x1 = nn.Conv3d(mod_num, out_ch, 1)
    def forward(self, x):
        # 计算权重
        gate_weights = self.gate(x)
        out = x * gate_weights + x # 权重加权
        
        # # 跨轴注意力 (暂时不使用)
        # attn_outputs = self.axial_attns(x)
        
        # # 残差
        # out = self.scale * attn_outputs + x
        
        out = F.relu(out)

        return out
    
class ResConv3D(nn.Module):
    
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.Dropout3d(p=dropout_rate)
            )
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        
    
    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.shortcut:
            residual = self.shortcut(residual)
        out += residual
        return self.relu(out)
    
class AttentionGate(nn.Module):
    """轴向注意力门控模块"""
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
        self.psi = nn.Sequential(
            nn.Conv3d(F_inter, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        # g: 上采样后的特征图
        # x: 跳跃连接的特征图
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # out = self.axial_attns(F.relu(g1 + x1))
        out = F.relu(g1 + x1)
        psi = self.psi(out)
        out = x * psi                   # [B, 256, 16, 16, 16]
        return out

class EnhancedAttentionGate(nn.Module):
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
            nn.Sigmoid()
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
        spatial_weight = self.spatial_att(fused)
        channel_weight = self.channel_att(fused)
        return x * (spatial_weight.expand_as(x) + channel_weight.expand_as(x))  # 双路加权

class EfficientDecoder(nn.Module):
    """高效解码器模块"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch)
        )
        
        self.conv_1x1 = nn.Conv3d(in_ch, out_ch, 1)
        
    def forward(self,x):
        # x = self.up(x)
        out = self.conv(x) + self.conv_1x1(x)
        out = F.relu(out)
        return out
    
    
class UpSample(nn.Module):
    """3D Up Convolution"""
    def __init__(self, in_channels, out_channels, trilinear=True):
        super(UpSample, self).__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x):
        return self.up(x)
    
class EagAttnUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, f_list=[32, 64, 128, 256], trilinear=True):
        super(EagAttnUNet, self).__init__()
        
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        # self.modality_fusion = CrossModalityFusionGate(mod_num=in_channels)
        self.Conv1 = ResConv3D(in_channels, f_list[0])
        self.Conv2 = ResConv3D(f_list[0], f_list[1])
        self.Conv3 = ResConv3D(f_list[1], f_list[2])
        self.Conv4 = ResConv3D(f_list[2], f_list[3])
        
        self.bottleneck = ResConv3D(f_list[3], f_list[3])
        
        self.Up5 = UpSample(f_list[3], f_list[3], trilinear)
        self.Att5 = EnhancedAttnGate(F_g=f_list[3], F_l=f_list[3], F_inter=f_list[3]//2)
        self.UpConv5 = ResConv3D(f_list[3]*2, f_list[3]//2)
        
        self.Up4 = UpSample(f_list[2], f_list[2], trilinear)
        self.Att4 = EnhancedAttnGate(F_g=f_list[2], F_l=f_list[2], F_inter=f_list[2]//2)
        self.UpConv4 = ResConv3D(f_list[2]*2, f_list[2]//2)
        
        self.Up3 = UpSample(f_list[1], f_list[1], trilinear)
        self.Att3 = EnhancedAttnGate(F_g=f_list[1], F_l=f_list[1], F_inter=f_list[1]//2)
        self.UpConv3 = ResConv3D(f_list[1]*2, f_list[1]//2)
        
        self.Up2 = UpSample(f_list[0], f_list[0], trilinear)
        self.Att2 = EnhancedAttnGate(F_g=f_list[0], F_l=f_list[0], F_inter=f_list[0]//2)
        self.UpConv2 = ResConv3D(f_list[0]*2, f_list[0])
        
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)
        
        self.apply(init_weights_3d)  # 初始化权重
        
    def forward(self, x):
        # Encoder
        # x_in = self.modality_fusion(x)       # [B, 4, D, H, W]
        x1 = self.Conv1(x)                # [B, 32, D, H, W]
        x2 = self.MaxPool(x1)
        x2 = self.Conv2(x2)      # [B, 64, D/2, H/2, W/2]
        x3 = self.MaxPool(x2)
        x3 = self.Conv3(x3)      # [B, 128, D/4, H/4, W/4]
        x4 = self.MaxPool(x3)
        x4 = self.Conv4(x4)      # [B, 256, D/8, H/8, W/8]
        x5 = self.MaxPool(x4)
        
        # Bottleneck
        x5 = self.bottleneck(x5)      # [B, 256, D/16, H/16, W/16]
        
        # Decoder with Attention
        d5 = self.Up5(x5)               # [B, 256, D/8, H/8, W/8]
        x4 = self.Att5(g=d5, x=x4)      # [B, 256, D/8, H/8, W/8]
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpConv5(d5)    # [B, 128, D/8, H/8, W/8]
        
        d4 = self.Up4(d5)        # [B, 128, D/4, H/4, W/4]
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpConv4(d4)    # [B, 64, D/4, H/4, W/4]
        
        d3 = self.Up3(d4)        # [B, 64, D/2, H/2, W/2]
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpConv3(d3)    # [B, 32, D/2, H/2, W/2]
        
        d2 = self.Up2(d3)        # [B, 32, D, H, W]
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.UpConv2(d2)    # [B, 32, D, H, W]
        
        out = self.outc(d2)  # [B, out_channels, D, H, W]
        return out
        
             
if __name__ == "__main__":
    test_unet(model_class=EagAttnUNet, batch_size=1)   