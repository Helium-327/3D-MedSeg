# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/11 16:13:19
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: Soft Pooling
*      VERSION: v1.0
=================================================
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftPool3D(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(SoftPool3D, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x):
        
        x_exp = torch.exp(x)
        
        sum_exp = F.max_pool3d(x_exp, self.kernel_size, self.stride, self.padding)
        
        sum_x = F.max_pool3d(x, self.kernel_size, self.stride, self.padding)
        
        soft_pool = sum_x / (sum_exp + 1e-8)
        
        return soft_pool
    
    
if __name__ == '__main__':
    softPool = SoftPool3D()
    
    # 创建一个示例输入张量
    input_tensor = torch.randn(1, 4, 128, 128, 128)
    
    out = softPool(input_tensor)
    
    print(out.shape)
    
    
    
    