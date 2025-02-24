# -*- coding: UTF-8 -*-
'''

Describle:         数据增强探索

Created on         2024/07/31 16:10:16
Author:            @ Mr_Robot
Current State:     # TODO: 
                    1. 添加 monai 的数据增强
                    2. 添加 torchio 的数据增强
                    3. 添加 albumentation 的数据增强
'''

import torch
import numpy as np
import torchio.transforms as tio

seed = 42#seed必须是int，可以自行设置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
torch.cuda.manual_seed_all(seed)#多卡模式下，让所有显卡生成的随机数一致？这个待验证
np.random.seed(seed)#numpy产生的随机数一致


# class data_transform:
#     """
#     训练集数据增强
#     通过控制每个增强类来控制是否对vimage和vmask进行增强
#     """
#     def __init__(self, CropSize=(128, 128, 128),
#                  mean=(0.114, 0.090, 0.170, 0.096), 
#                  std=(0.199, 0.151, 0.282, 0.174),
#                  transform=None):
#         """
#         初始化函数
#         :param CropSize: 3D裁剪大小
#         :param mean: 均值
#         :param std: 标准差
#         :param vimageTrans: 图像增强
#         :param maskTrans: 标签增强
#         """
#         torch.manual_seed(seed)
        
#         self.transforms_list = [
#                 RandomCrop3D(size=CropSize),    # 随机裁剪
#                 # tioRandonCrop3d(size=CropSize),
#                 FrontGroundNormalize(mean=mean, std=std),   # 标准化
#                 # tioRandomAffine(),          # 随机旋转
                
#                 tioRandomFlip3d(),                 # 随机翻转
#                 tioRandomElasticDeformation3d(),
#                 tioZNormalization(),               # 归一化
#                 tioRandomNoise3d(),
#                 tioRandomGamma3d(),
#                 ]
        
#         self.transforms = Compose(self.transforms_list)
        
#         if transform:
#             self.transforms = transform
            
#     def __call__(self, vimage, vmask):
#         """
#         调用函数
#         :param vimage: 3D图像
#         :param vmask: 3D标签
#         :return: 3D图像和3D标签
#         """
#         return self.transforms(vimage, vmask)
    
#     def getNamesOfTrans(self):
#         """
#         返回数据增强名称
#         :return: 
#         """
#         return self.transforms.returnName()
    

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, vimage, vmask):
        for t in self.transforms:
            vimage, vmask = t(vimage, vmask)
        return vimage, vmask
    def returnName(self):
        return [t.__class__.__name__ for t in self.transforms]
    
class ToTensor(object):
    def __init__(self):
        pass
    
    def __call__(self, vimage, vmask):
        if isinstance(vimage, np.ndarray):
            vimage = torch.tensor(vimage).float()
        if isinstance(vmask, np.ndarray):
            vmask = torch.tensor(vmask).long()
        return vimage.float(), vmask.long()
    

""" 归一化处理 """
class FrontGroundNormalize(object):
    def __init__(self):
        pass
    
    def __call__(self, vimage, vmask):
        mask = torch.sum(vimage, dim=0) > 0 
        for k in range(4):
            x = vimage[k, ...]
            y = x[mask]
            if y.numel() > 0:
                x[mask] = (x[mask] - y.mean()) / (y.std() + 1e-6)
            vimage[k, ...] = x
        return vimage, vmask

""" 归一化处理 """
class tioZNormalization(object):
    def __init__(self):
        pass
    
    def __call__(self, vimage, vmask):
        vimage = tio.ZNormalization()(vimage)
        return vimage, vmask
    
"""随机裁剪"""
class RandomCrop3D(object):
    def __init__(self, size=(128, 128, 128)):
        self.size = size
        
    def __call__(self, vimage, vmask):
        img = np.array(vmask)
        x_start = np.random.randint(0, img.shape[-3] - self.size[0]) if img.shape[-3] > self.size[0] else 0
        y_start = np.random.randint(0, img.shape[-2] - self.size[1]) if img.shape[-2] > self.size[1] else 0
        z_start = np.random.randint(0, img.shape[-1] - self.size[2]) if img.shape[-1] > self.size[2] else 0
        
        vimage = vimage[:,x_start: x_start + self.size[0], y_start: y_start + self.size[1], z_start: z_start + self.size[2]]
        vmask = vmask[x_start: x_start + self.size[0], y_start: y_start + self.size[1], z_start: z_start + self.size[2]]
        return vimage, vmask    

class tioRandonCrop3d(object):
    def __init__(self, size=(128, 128, 128)):
        self.size = size
    
    def __call__(self, vimage, vmask):
        vimage = tio.RandomCrop(self.size)(vimage)
        vmask = tio.RandomCrop(self.size)(vmask)
        return vimage, vmask


"""随机旋转"""
class tioRandomAffine(object):
    def __init__(self, image_interpolation="bspline", scales=(0.9, 1.2), degrees=15):
        self.image_interpolation = image_interpolation
        self.scales = scales
        self.degrees = degrees
        self.RandomAffine = tio.RandomAffine(image_interpolation=self.image_interpolation, scales=self.scales, degrees=self.degrees)
    def __call__(self, vimage, vmask):
        vimage = self.RandomAffine(vimage)
        vmask = self.RandomAffine(vmask.unsqueeze(dim=0))
        vmask = vmask.squeeze(dim=0)
        return vimage, vmask
    
"""加噪"""

class tioRandomNoise3d(object):
    def __init__(self, mean=0.0, std=(0, 0.25)):
        self.mean = mean
        self.std = std
        self.RandomNoise3d = tio.RandomNoise(mean=self.mean, std=self.std)
    def __call__(self, vimage, vmask):
        vimage = self.RandomNoise3d(vimage)
            
        return vimage, vmask
    
class tioRandomGamma3d(object):
    def __init__(self, log_gamma=(-0.3, 0.3)):
        self.log_gamma = log_gamma
        self.RandomGamma3d = tio.RandomGamma(log_gamma=self.log_gamma)
    def __call__(self, vimage, vmask):
        vimage = self.RandomGamma3d(vimage)

        return vimage, vmask
    
class tioRandomFlip3d(object):
    def __init__(self, p=1, axes=('LR',)):
        self.p = p
        self.axes = axes
        self.RandomFlip3d = tio.RandomFlip(flip_probability = self.p, axes=self.axes)
        
    def __call__(self, vimage, vmask):
        if np.random.rand() < self.p:
            vimage = self.RandomFlip3d(vimage)
            vmask = self.RandomFlip3d(vmask.unsqueeze(dim=0))
            vmask = vmask.squeeze(dim=0)
        return vimage, vmask   
        
class tioRandomElasticDeformation3d(object):
    # 将随机位移分配给图像周围和内部的控制点粗网格。使用三次 B 样条函数从粗网格中插值每个体素的位移
    def __init__(self, 
                 num_control_points=(7, 7, 7), # or just 7
                 locked_borders=2,
    ):
        self.num_control_points = num_control_points
        self.locked_borders = locked_borders
        self.tioRandomElasticDeformation3d = tio.RandomElasticDeformation(num_control_points=self.num_control_points, locked_borders=self.locked_borders)
        
    def __call__(self, vimage, vmask):
        vimage = self.tioRandomElasticDeformation3d(vimage)
        
        return vimage, vmask
if __name__ == "__main__":
    
    CropSize = (128, 128, 128)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    trans = Compose([
        RandomCrop3D(size=CropSize),
        FrontGroundNormalize(mean=mean, std=std),
        tioRandomFlip3d(),
        tioRandomElasticDeformation3d(),
        tioZNormalization(),
        tioRandomNoise3d(),
        tioRandomGamma3d()
    ])
    
    data = torch.randn(4, 240, 240, 155)
    mask = torch.randint(0, 4, (240, 240, 155))
    data, mask = trans(data, mask)
    print(data.shape, mask.shape)
    
    print(trans.returnName())

    
    
