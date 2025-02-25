# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/23 21:24:05
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 三维轴向注意力机制
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''

import torch

import torch.nn as nn
import torch.nn.functional as F


class SingleAxialAttention(nn.Module):
    def __init__(self, in_ch, heads=4, axis=0, eps=1e-6):
        """
        三维轴向注意力模块
        参数：
            in_ch: 输入通道维度
            heads: 注意力头数（默认4）
            axis: 注意力计算轴（0-深度轴，1-高度轴，2-宽度轴）
        """
        super(SingleAxialAttention, self).__init__()
        self.eps = eps
        self.heads = heads
        # 缩放因子，用于稳定softmax计算（类似Transformer的1/sqrt(d_k)）
        self.scale = (in_ch // heads) ** -0.5
        self.axis = axis  # 指定注意力作用的轴向
        
        # 将输入同时映射为Q,K,V（使用1x1x1卷积实现）
        self.to_qkv = nn.Conv3d(in_ch, in_ch * 3, 1, bias=False)
        # 输出投影层（融合多头注意力结果）
        self.proj = nn.Conv3d(in_ch, in_ch, 1)
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        # 生成Q,K,V三元组 [3×B, C, D, H, W]
        qkv = self.to_qkv(x).chunk(3, dim=1)
        
        # 轴向分解和维度重组（根据不同的注意力轴向进行维度变换）
        if self.axis == 0:  # 深度轴（D轴）注意力
            # 将D轴移至最后，合并H和W维度
            q, k, v = map(lambda t: t.permute(0, 1, 3, 4, 2).reshape(B, self.heads, C//self.heads, H*W, D), qkv)
        
        elif self.axis == 1:  # 高度轴（H轴）注意力
            # 将H轴移至最后，合并D和W维度
            q, k, v = map(lambda t: t.permute(0, 1, 2, 4, 3).reshape(B, self.heads, C//self.heads, D*W, H), qkv)
            
        else:  # 宽度轴（W轴）注意力
            # 保持W轴在最后，合并D和H维度
            q, k, v = map(lambda t: t.permute(0, 1, 2, 3, 4).reshape(B, self.heads, C//self.heads, D*H, W), qkv)
            
        # 注意力分数计算 ---------------------------------------------------
        # 计算QK^T / sqrt(d_k)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn= torch.einsum('bhdnk,bhdmk->bhdnm', q, k.transpose(-2, -1)) * self.scale
        # 归一化注意力权重
        attn = attn.softmax(dim=-1) + self.eps
        
        # 应用注意力到V并重组维度 ------------------------------------------
        # [注意] transpose(2,3)用于恢复被合并的空间维度
        out = (attn @ v).transpose(2, 3).reshape(B, C, D, H, W)
        # out = torch.einsum('bhdnm,bhdmk->bhdnk', attn, v).transpose(2, 3).reshape(B, C, D, H, W)
        
        # 最终投影输出
        return self.proj(out)

class AxialAttention3D(nn.Module):
    """三维轴向注意力模块"""
    def __init__(self, in_ch, heads=1):
        super().__init__()

        self.axial_attns = nn.ModuleList([
            SingleAxialAttention(in_ch, heads, axis=dim) for dim in range(3)
        ])
        
        self.fusion = nn.Conv3d(3*in_ch, in_ch, 1)
        
    def forward(self, x):
        
        # 跨轴注意力
        attn_outputs = [attn(x) for attn in self.axial_attns]
        out = torch.cat(attn_outputs, dim=1)

        # 融合
        out = self.fusion(out)

        # 残差
        out = out + x
        out = F.relu(out)
        return out
        
        
        