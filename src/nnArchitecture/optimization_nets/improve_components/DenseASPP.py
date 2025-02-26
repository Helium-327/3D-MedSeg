# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/23 21:17:37
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: DenseASPP (密集 空洞空间金字塔池化)
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''
#! 当前模块存在 验证指标不稳定的问题

import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class DenseASPP3D(nn.Module):
    def __init__(self, in_channels, out_channels, reduce_rate=4, dilations=[1, 2, 3, 5]):
        super(DenseASPP3D, self).__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels//reduce_rate, kernel_size=3, padding=dilations[0], dilation=dilations[0]),
            nn.GroupNorm(4, out_channels//reduce_rate),
            nn.Dropout3d(0.1)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv3d(in_channels + out_channels//reduce_rate, out_channels//reduce_rate, kernel_size=3, padding=dilations[1], dilation=dilations[1]),
            nn.GroupNorm(4, out_channels//reduce_rate),
            nn.Dropout3d(0.1)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv3d(in_channels + 2*(out_channels//reduce_rate), out_channels//reduce_rate, kernel_size=3, padding=dilations[2], dilation=dilations[2]),
            nn.GroupNorm(4, out_channels//reduce_rate),
            nn.Dropout3d(0.1)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv3d(in_channels + 3*(out_channels//reduce_rate), out_channels//reduce_rate, kernel_size=3, padding=dilations[3], dilation=dilations[3]),
            nn.GroupNorm(4, out_channels//reduce_rate),
            nn.Dropout3d(0.1)
        )
        self.global_avg = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels//reduce_rate, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels//reduce_rate, in_channels, kernel_size=1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        ) 
        self.fusion = nn.Sequential(
            nn.Conv3d(4*(out_channels//reduce_rate), out_channels, 1),
            nn.GroupNorm(8, out_channels),
        )
        self.act = nn.ReLU(inplace=True)
        # self.conv1x1 = nn.Conv3d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(torch.cat([x, x1], 1))
        x3 = self.aspp3(torch.cat([x, x1, x2], 1))
        x4 = self.aspp4(torch.cat([x, x1, x2, x3], 1))
        channel_attented = self.global_avg(x)
        # x5 = F.interpolate(x5, size=x4.size()[2:], mode='trilinear', align_corners=True)
        fusion_x = torch.cat([x1, x2, x3, x4], 1)
        out = self.fusion(fusion_x)
        out = out * channel_attented
        out = self.act(out)
        return out
    

def test_DenseASPP3D():
    # 配置测试参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    # 输入参数配置
    batch_size = 2
    in_channels = 64
    out_channels = 256
    spatial_size = (32, 32, 32)  # (Depth, Height, Width)

    # 初始化模型
    model = DenseASPP3D(in_channels, out_channels).to(device)
    model.eval()

    # 创建随机输入张量
    x = torch.randn(batch_size, in_channels, *spatial_size).to(device)

    # 参数数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Parameter Statistics]")
    print(f"Total parameters: {total_params:,}")
    print(f"≈ {total_params/1e6:.2f}M parameters")

    # FLOPs计算
    with torch.no_grad():
        flops, params = profile(model, inputs=(x,))
    print(f"\n[FLOPs Analysis]")
    print(f"Total FLOPs: {flops:,}")
    print(f"≈ {flops/1e9:.2f}G FLOPs")

    # 推理时间测试
    num_warmups = 10
    num_tests = 100
    print(f"\n[Performance Benchmark]")
    
    # CPU测试
    if device.type == 'cpu':
        with torch.no_grad():
            # 预热
            for _ in range(num_warmups):
                _ = model(x)
            
            # 正式测试
            start_time = time.time()
            for _ in range(num_tests):
                _ = model(x)
            elapsed_time = time.time() - start_time
        
        print(f"Average inference time: {elapsed_time/num_tests*1000:.2f}ms")

    # GPU测试
    else:
        # 创建CUDA事件
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = []

        with torch.no_grad():
            # 预热
            for _ in range(num_warmups):
                _ = model(x)
            
            # 基准测试
            for _ in range(num_tests):
                starter.record()
                _ = model(x)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))

        avg_time = sum(timings) / num_tests
        print(f"Average inference time: {avg_time:.2f}ms")

    # 显存占用分析（仅GPU）
    if device.type == 'cuda':
        print(f"\n[Memory Analysis]")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated()/1024**2:.2f}MB")
        print(f"Max memory cached: {torch.cuda.max_memory_reserved()/1024**2:.2f}MB")



def test_gradient_flow():
    # 配置测试参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Gradient Propagation Test]")
    print(f"Testing on device: {device}")

    # 输入参数配置
    batch_size = 2
    in_channels = 64
    out_channels = 256
    spatial_size = (32, 32, 32)

    # 初始化模型（必须处于训练模式）
    model = DenseASPP3D(in_channels, out_channels).to(device)
    model.train()  # 重要！确保处于训练模式

    # 创建随机输入和伪标签
    x = torch.randn(batch_size, in_channels, *spatial_size).to(device)
    target = torch.randn(batch_size, out_channels, *spatial_size).to(device)

    # 定义损失函数
    criterion = nn.MSELoss()

    # 前向传播
    output = model(x)
    
    # 计算损失
    loss = criterion(output, target)
    
    # 反向传播
    loss.backward()

    # 梯度检查函数
    def check_gradients(module, prefix=""):
        for name, param in module.named_parameters():
            full_name = f"{prefix}.{name}" if prefix else name
            if param.grad is None:
                print(f"✗ No gradient for {full_name}")
            elif torch.allclose(param.grad, torch.zeros_like(param.grad)):
                print(f"⚠ Zero gradient for {full_name}")
            else:
                grad_mean = param.grad.abs().mean().item()
                print(f"✓ Gradient OK for {full_name} (mean abs: {grad_mean:.3e})")

    # 检查各子模块梯度
    print("\nChecking gradients for main components:")
    check_gradients(model.aspp1, "aspp1")
    check_gradients(model.aspp2, "aspp2")
    check_gradients(model.aspp3, "aspp3")
    check_gradients(model.aspp4, "aspp4")
    check_gradients(model.global_avg, "global_avg")
    check_gradients(model.fusion, "fusion")

    # 检查最终输出梯度
    print("\nFinal output gradient statistics:")
    print(f"Output grad: {output.requires_grad}")  # 应为True
    if output.grad is not None:
        print(f"Output grad norm: {output.grad.norm().item():.3e}")
        
if __name__ == "__main__":
    test_DenseASPP3D()
    test_gradient_flow()  # 新增梯度测试