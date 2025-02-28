# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/24 09:10:26
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: AA-UNet (三维轴向注意力UNet)
*      VERSION: v2.0
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

def init_weights_3d(m):
    """Initialize 3D卷积和BN层的权重"""
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class SingleAxialAttention(nn.Module):
    def __init__(self, in_ch, heads=4, axis=0, eps=1e-6):
        """
        三维轴向注意力模块
        参数：
            in_ch: 输入通道维度
            heads: 注意力头数（默认4）
            axis: 注意力计算轴（0-深度轴，1-高度轴，2-宽度轴)
        """
        super(SingleAxialAttention, self).__init__()
        self.eps = eps
        self.heads = heads
        # 缩放因子，用于稳定softmax计算（类似Transformer的1/sqrt(d_k)
        self.scale = (in_ch // heads) ** -0.5
        self.axis = axis  # 指定注意力作用的轴向
        
        # 将输入同时映射为Q,K,V（使用1x1x1卷积实现）
        self.to_qkv = nn.Conv3d(in_ch, in_ch * 3, 1, bias=False)
        # 输出投影层（融合多头注意力结果）
        self.proj = nn.Conv3d(in_ch, in_ch, 1)
        self.apply(init_weights_3d) 
        
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
        self.apply(init_weights_3d) 
        
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
        

class CrossModalityFusionGate(nn.Module):
    """跨轴动态融合模块"""
    def __init__(self, mod_num=4, init_temp=0.3):
        super().__init__()
        
        self.mod_num = mod_num
        # self.scale = nn.Parameter(torch.ones(1))  # 可学习缩放参数
         
        self.gate = nn.Sequential(
            nn.Conv3d(mod_num, 16, 1),  # 扩展中间维度
            nn.GroupNorm(4, 16),        # 分组归一化
            nn.LeakyReLU(0.1),
            nn.Conv3d(16, mod_num, 1),
            nn.Tanh()  # 限制输出范围[-1,1]
        )
        self.temperature = nn.Parameter(torch.tensor([init_temp]))
        nn.init.constant_(self.temperature, init_temp)
        self.apply(init_weights_3d) 
        
        self.res_scale = nn.Parameter(torch.ones(1)*0.5)
        # self.axial_attns = AxialAttention3D(in_ch=mod_num, heads=4)

        # self.conv1x1 = nn.Conv3d(mod_num, out_ch, 1)
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        # 生成门控权重
        gate_logits = self.gate(x)  # [-1,1]
        
        # 温度缩放Softmax
        weights = F.softmax(gate_logits / (self.temperature.abs() + 1e-6), dim=1)
        
        # 稳定残差融合
        out = x * weights * self.res_scale + x * (1 - self.res_scale)
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

class ResConv3D_Anisotropic(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, directions=4):
        super().__init__()
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1, k ,k), padding=(0, k//2, k//2)),
                nn.InstanceNorm3d(out_channels),
            ),
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(k, 1 ,k), padding=(k//2, 0,  k//2)),
                nn.InstanceNorm3d(out_channels),
            ),
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(k, k,1), padding=(k//2, k//2, 0)),
                nn.InstanceNorm3d(out_channels),
            ),
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(k, 1 ,1), padding=(k//2, 0, 0)),
                nn.InstanceNorm3d(out_channels),
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
    
class HybridEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, scale=4, droupout_rate=0.2):
        super().__init__()
        assert in_ch // scale != 0, "in_ch should be divisible by scale"
        
        self.conv = ResConv3D_Anisotropic(in_ch, out_ch)
        
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
        
        
class AxialAttentionGate(nn.Module):
    """轴向注意力门控模块"""
    def __init__(self, F_g, F_l, F_inter):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_inter, kernel_size=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_inter)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_inter, kernel_size=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_inter)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_inter, 1, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        
        self.axial_attns = AxialAttention3D(F_inter)
        

    def forward(self, g, x):
        # g: 上采样后的特征图
        # x: 跳跃连接的特征图
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        out = self.axial_attns(F.relu(g1 + x1))
        # out = F.relu(g1 + x1)
        psi = self.psi(out)
        out = x * psi                   # [B, 256, 16, 16, 16]
        return out
    
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
        
        # 空间注意力和通道注意力机制
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
    
class AAUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, f_list=[32, 64, 128, 256], trilinear=True):
        super(AAUNet, self).__init__()
        
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.modality_fusion = CrossModalityFusionGate(mod_num=in_channels)
        self.Conv1 = HybridEncoderBlock(in_channels, f_list[0])
        self.Conv2 = HybridEncoderBlock(f_list[0], f_list[1])
        self.Conv3 = HybridEncoderBlock(f_list[1], f_list[2])
        self.Conv4 = HybridEncoderBlock(f_list[2], f_list[3])
        
        self.bottleneck = HybridEncoderBlock(f_list[3], f_list[3])
        
        self.Up5 = UpSample(f_list[3], f_list[3], trilinear)
        self.Att5 = EnhancedAttentionGate(F_g=f_list[3], F_l=f_list[3], F_inter=f_list[3]//2)
        self.UpConv5 = ResConv3D(f_list[3]*2, f_list[3]//2)
        
        self.Up4 = UpSample(f_list[2], f_list[2], trilinear)
        self.Att4 = EnhancedAttentionGate(F_g=f_list[2], F_l=f_list[2], F_inter=f_list[2]//2)
        self.UpConv4 = ResConv3D(f_list[2]*2, f_list[2]//2)
        
        self.Up3 = UpSample(f_list[1], f_list[1], trilinear)
        self.Att3 = EnhancedAttentionGate(F_g=f_list[1], F_l=f_list[1], F_inter=f_list[1]//2)
        self.UpConv3 = ResConv3D(f_list[1]*2, f_list[1]//2)
        
        self.Up2 = UpSample(f_list[0], f_list[0], trilinear)
        self.Att2 = EnhancedAttentionGate(F_g=f_list[0], F_l=f_list[0], F_inter=f_list[0]//2)
        self.UpConv2 = ResConv3D(f_list[0]*2, f_list[0])
        
        self.outc = nn.Conv3d(f_list[0], out_channels, kernel_size=1)
        
        self.apply(init_weights_3d)  # 初始化权重
        
    def forward(self, x):
        # Encoder
        x_in = self.modality_fusion(x)       # [B, 4, D, H, W]
        x1 = self.Conv1(x_in)                # [B, 32, D, H, W]
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
    test_unet(model_class=AAUNet, batch_size=2)   
        
             