# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/11 15:56:34
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: Inception Block
*      VERSION: v1.0
=================================================
'''
import torch

import torch.nn as nn
import torch.nn.functional as F

class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            # nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            # nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2), # 膨胀卷积，膨胀率为2
            nn.BatchNorm3d(out_channels),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.branch5 = nn.Sequential(
            nn.AvgPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.out_conv = nn.Conv3d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        out = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
            self.branch5(x),
        ], dim=1)
        out = F.relu(self.out_conv(out))
        return out

class D_Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=1, padding=1), # 膨胀卷积，膨胀率为2
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2), # 膨胀卷积，膨胀率为2
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=3, padding=3), # 膨胀卷积，膨胀率为2
            nn.BatchNorm3d(out_channels),
        )
        self.branch4 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
        )
        self.branch5 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.AvgPool3d(kernel_size=3, stride=1, padding=1),
        )
        self.fusion = nn.Conv3d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        out = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
            self.branch5(x),
        ], dim=1)
        out = F.relu(self.fusion(out))
        return out