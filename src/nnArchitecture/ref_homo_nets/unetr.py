# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/13 10:39:11
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: UNNETR 网络
*      VERSION: v1.0
=================================================
✅ 测试完成
'''
import sys
sys.path.append('/root/workspace/VoxelMedix/src/nnArchitecture')
# from __future__ import annotations


from collections.abc import Sequence
import torch
import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets.vit import ViT
from monai.utils import deprecated_arg, ensure_tuple_rep
# from _init_model import init_all_weights
torch.autograd.set_detect_anomaly(True) 
            
def init_weights_3d(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 适配GELU
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):  # 初始化Transformer线性层
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
class UNETR(nn.Module):
    """
    UNETR 模型，基于 Transformer 的 3D 医学图像分割网络。
    参考论文: "Hatamizadeh et al., UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: (Sequence[int] | int)=128,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
        predict_mode: bool = False
    ) -> None:
        """
        初始化 UNETR 模型。

        参数:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            img_size (Sequence[int] | int): 输入图像的尺寸。
            feature_size (int): 网络特征大小。默认: 16。
            hidden_size (int): 隐藏层大小。默认: 768。
            mlp_dim (int): 前馈网络层大小。默认: 3072。
            num_heads (int): 注意力头的数量。默认: 12。
            proj_type (str): 嵌入层类型。默认: "conv"。
            norm_name (tuple | str): 归一化类型。默认: "instance"。
            conv_block (bool): 是否使用卷积块。默认: True。
            res_block (bool): 是否使用残差块。默认: True。
            dropout_rate (float): Dropout 率。默认: 0.0。
            spatial_dims (int): 空间维度数。默认: 3。
            qkv_bias (bool): 是否在自注意力块中的 qkv 线性层使用偏置项。默认: False。
            save_attn (bool): 是否保存自注意力块的注意力权重。默认: False。
            predict_mode (bool): 是否处于预测模式。默认: False。
        """
        super().__init__()

        # 参数校验
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        # 初始化模型参数
        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self.predict_mode = predict_mode

        # 初始化 Vision Transformer (ViT)
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )

        # 初始化编码器部分
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        # 初始化解码器部分
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        # 输出层
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

        self.softmax = nn.Softmax(dim=1)
        # 投影相关参数
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]
        self.apply(init_weights_3d)

    def proj_feat(self, x):
        """将特征投影到指定维度。"""
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        """前向传播。"""
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        if self.predict_mode:
            return self.out(out)
        else:
            # return out, self.out(out)
            return self.out(out)


if __name__ == "__main__":
    # 设置模型参数
    in_channels = 4  # 输入通道数
    out_channels = 4  # 输出通道数
    img_size = (128, 128, 128)  # 输入图像尺寸
    feature_size = 16  # 特征大小
    hidden_size = 768  # 隐藏层大小
    num_heads = 12  # 注意力头数量
    spatial_dims = 3  # 空间维度数

    # 创建 UNETR 模型实例
    model = UNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        feature_size=feature_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        spatial_dims=spatial_dims,
        predict_mode=True  # 设置为预测模式
    )

    from ptflops import get_model_complexity_info
    # 使用Ptflops计算参数量和FLOPs
    macs, params = get_model_complexity_info(model, (4, 128, 128, 128), as_strings=True, print_per_layer_stat=True, verbose=True)

    print(f'Computational complexity: {macs}')
    print(f'Number of parameters: {params}')