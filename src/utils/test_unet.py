# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/19 14:59:32
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: UNet 通用测试脚本
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''

# test_unet.py
import torch
import torch.nn as nn
import time
from thop import profile, clever_format
from torchinfo import summary


def test_unet(model_class, batch_size=1, in_channels=4, spatial_size=128, num_classes=4, device_str=None):
    """
    测试 UNet 模型的前向传播、FLOPs 和参数量。

    参数:
        model_class (nn.Module): 模型类，必须是 nn.Module 的子类。
        batch_size (int): 批量大小。
        in_channels (int): 输入通道数。
        spatial_size (int): 输入的空间维度大小（假设为立方体）。
        num_classes (int): 输出类别数。
        device_str (str): 指定设备（'cuda' 或 'cpu'）。默认为自动检测。
    """
    # 检查模型类是否是 nn.Module 的子类
    if not issubclass(model_class, nn.Module):
        raise TypeError("model_class must be a subclass of nn.Module")

    # 设置设备
    if device_str is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    # 生成测试数据
    input_tensor = torch.randn(
        batch_size,
        in_channels,
        spatial_size,
        spatial_size,
        spatial_size
    ).to(device)
    
    # 初始化模型
    model = model_class(in_channels=in_channels, out_channels=num_classes).to(device)
    
    # 检查权重初始化是否正确
    for name, param in model.named_parameters():
        if 'weight' in name and 'conv' in name:
            print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
        elif 'bias' in name:
            print(f"{name}: value={param.data[:2]} (should be 0)")
        
    # 打印模型结构
    summary(model, input_size=(batch_size, in_channels, spatial_size, spatial_size, spatial_size), device=device.type)
    
    # 前向传播并测量时间
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    elapsed_time = time.time() - start_time
    print(f"前向传播时间: {elapsed_time:.6f}秒")
    
    # 验证输出尺寸
    assert output.shape == (batch_size, num_classes, spatial_size, spatial_size, spatial_size), \
        f"输出尺寸错误，期望: {(batch_size, num_classes, spatial_size, spatial_size, spatial_size)}, 实际: {output.shape}"
    
    # 计算FLOPs
    flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, 参数量: {params}")
    
    # 打印测试结果
    print("\n测试通过！")
    print("输入尺寸:", input_tensor.shape)
    print("输出尺寸:", output.shape)
    print("设备信息:", device)
    print("最后一层权重范数:", torch.norm(model.outc.weight).item())


if __name__ == "__main__":
    # 示例模型类
    class ScgaResAtteUNet(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv3d(32, out_channels, kernel_size=1)
            )
            self.outc = nn.Conv3d(out_channels, out_channels, kernel_size=1)

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return self.outc(x)

    # 示例调用
    test_unet(model_class=ScgaResAtteUNet, batch_size=1, in_channels=4, spatial_size=128, num_classes=4, device_str="cuda")