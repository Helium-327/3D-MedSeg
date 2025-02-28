# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/23 21:21:53
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: SCGA (Spatial Channel Group Attention)
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from thop import profile

#! 当前模块会出现梯度消失问题，需要进一步优化

class SCGAv2(nn.Module):
    def __init__(self, channels=32, factor=8): # factor不能太大
        super(SCGAv2, self).__init__()
        self.group = factor
        assert channels // self.group > 4, "factor too big, channels // self.group > 4"
        self.softmax = nn.Softmax(dim=-1)
        self.averagePooling = nn.AdaptiveAvgPool3d((1, 1, 1))  # 3D 全局平均池化
        self.maxPooling = nn.AdaptiveMaxPool3d((1, 1, 1))      # 3D 全局最大池化
        self.Pool_h = nn.AdaptiveAvgPool3d((None, 1, 1))       # 高度方向池化
        self.Pool_w = nn.AdaptiveAvgPool3d((1, None, 1))       # 宽度方向池化
        self.Pool_d = nn.AdaptiveAvgPool3d((1, 1, None))       # 深度方向池化

        self.groupNorm = nn.GroupNorm(channels // self.group, channels // self.group)
        self.conv1x1x1 = nn.Conv3d(channels // self.group, channels // self.group, kernel_size=1, stride=1, padding=0)
        self.conv3x3x3 = nn.Sequential(
            nn.Conv3d(channels // self.group, channels // self.group, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(channels // self.group, channels // self.group),
        )
            
            
        
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

class SCGA(nn.Module):
    def __init__(self, channels=32, factor=32): # factor不能太大
        super(SCGA, self).__init__()
        self.group = factor
        assert channels // self.group > 4, "factor too big, channels // self.group > 4"
        self.softmax = nn.Softmax(dim=-1)
        self.averagePooling = nn.AdaptiveAvgPool3d((1, 1, 1))  # 3D 全局平均池化
        self.maxPooling = nn.AdaptiveMaxPool3d((1, 1, 1))      # 3D 全局最大池化
        self.Pool_h = nn.AdaptiveAvgPool3d((None, 1, 1))       # 高度方向池化
        self.Pool_w = nn.AdaptiveAvgPool3d((1, None, 1))       # 宽度方向池化
        self.Pool_d = nn.AdaptiveAvgPool3d((1, 1, None))       # 深度方向池化

        self.groupNorm = nn.GroupNorm(channels // self.group, channels // self.group)
        self.conv1x1x1 = nn.Conv3d(channels // self.group, channels // self.group, kernel_size=1, stride=1, padding=0)
        self.conv3x3x3 = nn.Sequential(
            nn.Conv3d(channels // self.group, channels // self.group, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(channels // self.group, channels // self.group),
        )
            
            
        
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
    
    

"""========================================== 测试代码 =============================================="""
def test_SCGA():
    # 配置测试参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[SCGA Module Test]")
    print(f"Testing on device: {device}")

    # 参数配置
    batch_size = 2
    channels = 64
    spatial_size = (16, 16, 16)  # (Depth, Height, Width)
    factor = 8  # 分组因子

    # 初始化模块
    scga = SCGA(channels, factor=factor).to(device)
    
    # 测试1：输出形状验证
    def test_output_shape():
        print("\n[Test 1/4] Output Shape Verification")
        x = torch.randn(batch_size, channels, *spatial_size).to(device)
        try:
            output = scga(x)
            assert output.shape == x.shape
            print("✓ Output shape correct")
            return True
        except Exception as e:
            print(f"✗ Shape mismatch: {e}")
            return False

    # 测试2：梯度传播测试
    def test_gradient_flow():
        print("\n[Test 2/4] Gradient Propagation Check")
        scga.train()  # 确保训练模式
        x = torch.randn(batch_size, channels, *spatial_size, requires_grad=True).to(device)
        target = torch.randn_like(x)
        
        # 前向+反向
        output = scga(x)
        loss = F.mse_loss(output, target)
        loss.backward()

        # 梯度检查
        has_gradient = True
        for name, param in scga.named_parameters():
            if param.grad is None:
                print(f"✗ No gradient for {name}")
                has_gradient = False
            elif torch.allclose(param.grad, torch.zeros_like(param.grad)):
                print(f"⚠ Zero gradient for {name}")
            else:
                grad_mean = param.grad.abs().mean().item()
                if grad_mean < 1e-7:
                    print(f"⚠ Small gradient for {name} (mean={grad_mean:.2e})")
        print("✓ Gradient check completed" if has_gradient else "✗ Gradient issues found")
        return has_gradient

    # 测试3：数值稳定性测试
    def test_numerical_stability():
        print("\n[Test 3/4] Numerical Stability Check")
        scga.eval()
        
        # 测试极端输入
        tests = [
            ("Zero input", torch.zeros(batch_size, channels, *spatial_size)),
            ("Large values", torch.randn(batch_size, channels, *spatial_size) * 100),
            ("Small values", torch.randn(batch_size, channels, *spatial_size) * 0.01)
        ]
        
        passed = True
        for name, x in tests:
            x = x.to(device)
            try:
                output = scga(x)
                # 检查输出范围
                if output.min() < -1e3 or output.max() > 1e3:
                    print(f"⚠ {name}: Output range异常 ({output.min():.2f}, {output.max():.2f})")
                    passed = False
                # 检查NaN/Inf
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"✗ {name}: 包含NaN/Inf值")
                    passed = False
            except Exception as e:
                print(f"✗ {name}: 前向传播失败 - {str(e)}")
                passed = False
        print("✓ 数值稳定性测试通过" if passed else "✗ 数值稳定性问题")
        return passed

    # 测试4：性能分析
    def test_performance():
        print("\n[Test 4/4] Performance Analysis")
        scga.eval()
        x = torch.randn(batch_size, channels, *spatial_size).to(device)
        
        # 参数统计
        total_params = sum(p.numel() for p in scga.parameters())
        print(f"参数总量: {total_params/1e3:.1f}K")
        
        # FLOPs计算
        with torch.no_grad():
            flops, _ = profile(scga, inputs=(x,))
        print(f"FLOPs: {flops/1e9:.2f}G")
        
        # 推理时间测试
        num_runs = 100 if device.type == 'cuda' else 10
        timings = []
        with torch.no_grad():
            for _ in range(num_runs):
                if device.type == 'cuda':
                    starter = torch.cuda.Event(enable_timing=True)
                    ender = torch.cuda.Event(enable_timing=True)
                    starter.record()
                    _ = scga(x)
                    ender.record()
                    torch.cuda.synchronize()
                    timings.append(starter.elapsed_time(ender))
                else:
                    start = time.time()
                    _ = scga(x)
                    timings.append((time.time() - start)*1000)
        
        avg_time = sum(timings)/len(timings)
        print(f"平均推理时间: {avg_time:.2f}ms")
        return True

    # 执行所有测试
    test_output_shape()
    test_gradient_flow()
    test_numerical_stability()
    test_performance()

if __name__ == "__main__":
    test_SCGA()