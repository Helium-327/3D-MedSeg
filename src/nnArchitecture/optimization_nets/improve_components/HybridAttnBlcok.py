# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/27 15:09:54
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 各项异性卷积
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridAttnBlock(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2, droupout_rate=0.2):
        super().__init__()
        assert in_ch // scale != 0, "in_ch should be divisible by scale"
        
        self.conv = AnisotropicConv3d(in_ch, out_ch)
        
        # 空间注意力分支（空洞卷积）
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(2, 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(2, 2, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.InstanceNorm3d(2),
            nn.Conv3d(2, 1, 1),
            nn.Sigmoid()
        )
        # 通道注意力分支
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_ch, out_ch//scale, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch//scale, out_ch, 1),
            nn.Sigmoid()
        )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        
        x_conv = self.conv(x)
        
        # 空间注意力
        avg_out = torch.mean(x_conv, dim=1, keepdim=True)

        max_out, _ = torch.max(x_conv, dim=1, keepdim=True)
        
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.spatial_conv(spatial_input)
        
        x_spatial_attented = x_conv * spatial_attn.expand_as(x_conv)
        x_norm = self.norm(x_spatial_attented)
        x_res = x_spatial_attented + self.beta * x_norm
        
        # 通道注意力
        channel_attn = self.channel_attn(x_res)
        
        # 融合注意力
        out = x_spatial_attented * channel_attn.expand_as(x_spatial_attented)
        
        return out

class AnisotropicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, directions=4):
        super().__init__()
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1, k ,k), padding=(0, k//2, k//2)),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(k, 1 ,k), padding=(k//2, 0,  k//2)),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(k, k,1), padding=(k//2, k//2, 0)),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(k, 1 ,1), padding=(k//2, 0, 0)),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True),
            ),
        ])
        
        self.attention_map = nn.Sequential(             # 添加非线性层提升表达能力
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, directions//2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(directions//2, directions, 1),
            nn.Softmax(dim=1)
        )
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        
        residual = x
        # 计算注意力权重
        att_weights  = self.attention_map(x)          # [B, 4, 1, 1, 1]
        
        # 计算各分支输出
        branch_outputs  = [conv(x) for conv in self.conv]  # 各个方向卷积的结果
        
        # 自适应权重
        weighted_outputs = [att_weights[:, i].unsqueeze(1).expand_as(out) * out for i, out in enumerate(branch_outputs)] # [b,c,4,d,h,w]
        
        out_main = sum(weighted_outputs)  # 加权融合
        
        out_shortcut = self.shortcut(x) if self.shortcut else x
        out = self.relu(out_main + out_shortcut)
        return out