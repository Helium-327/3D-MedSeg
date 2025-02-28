# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/06 16:39:58
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 损失函数
*      VERSION: v1.0
=================================================
'''

from sklearn.model_selection import PredefinedSplit
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import torch

class BaseLoss:
    def __init__(self, sub_areas, labels, smooth=1e-5, w1=0.2, w2=0.2, w3=0.4):
        self.smooth = smooth
        self.sub_areas = sub_areas
        self.labels = labels
        self.num_classes = len(labels)
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def split_sub_areas(self, y_pred, y_mask):
        """
        分割出子区域
        :param y_pred: 预测值 [batch, 4, D, W, H]
        :param y_mask: 真实值 [batch, 4, D, W, H]
        """
        y_pred = F.softmax(y_pred, dim=1)
        et_pred = y_pred[:, 3, ...]
        tc_pred = y_pred[:, 1, ...] + y_pred[:, 3, ...]
        wt_pred = y_pred[:, 1:, ...].sum(dim=1) 
        
        et_mask = y_mask[:, 3, ...]
        tc_mask = y_mask[:, 1, ...] + y_mask[:, 3, ...]
        wt_mask = y_mask[:, 1:, ...].sum(dim=1)
        
        pred_list = [et_pred, tc_pred, wt_pred]
        mask_list = [et_mask, tc_mask, wt_mask]
        return pred_list, mask_list

class DiceLoss(BaseLoss):
    def __init__(self, smooth=1e-5, w1=0.2, w2=0.2, w3=0.4):
        super().__init__(['ET', 'TC', 'WT'], {'BG': 0, 'NCR': 1, 'ED': 2, 'ET': 3}, smooth, w1, w2, w3)

    def __call__(self, y_pred, y_mask):
        """
        DiceLoss
        :param y_pred: 预测值 [batch, 4, D, W, H]
        :param y_mask: 真实值 [batch, D, W, H]
        """
        y_mask = F.one_hot(y_mask, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        # y_mask = y_mask[:, 1:, ...]
        pred_list, mask_list = self.split_sub_areas(y_pred, y_mask)

        area_et_loss, area_tc_loss, area_wt_loss = self.get_every_sub_areas_loss(pred_list, mask_list)
        mean_loss = (area_et_loss + area_tc_loss + area_wt_loss) / 3
        
        # self._check_nan(area_et_loss, 'ET')
        # self._check_nan(area_tc_loss, 'TC')
        # self._check_nan(area_wt_loss, 'WT')
        # self._check_nan(mean_loss, "Mean")
        return mean_loss, area_et_loss, area_tc_loss, area_wt_loss

    def get_every_sub_areas_loss(self, pred_list, mask_list):
        loss_dict = {}
        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            intersection = (pred * mask).sum(dim=(-3, -2, -1))
            union = pred.sum(dim=(-3, -2, -1)) + mask.sum(dim=(-3, -2, -1))
            dice_c = (2. * intersection) / (union + self.smooth)
            loss_dict[sub_area] = 1. - dice_c.mean()

        area_et_loss = loss_dict['ET']
        area_tc_loss = loss_dict['TC']
        area_wt_loss = loss_dict['WT']
        
        # if not self._check_nan(area_et_loss, 'ET'):
        #     print(f"NaN detected in ET loss!")
        
        # if not self._check_nan(area_tc_loss, 'TC'):
        #     print(f"NaN detected in ET loss!")
        
        # if not self._check_nan(area_wt_loss, 'WT'):
        #     print(f"NaN detected in ET loss!")
            
        
        return area_et_loss, area_tc_loss, area_wt_loss

    def _check_nan(self, tensor, name):
        """NaN检查辅助函数[5,6](@ref)"""
        if torch.isnan(tensor).any():
            print(
                f"NaN detected in {name} loss! "
                "Possible causes:\n"
                "1. Invalid input data (check labels and input normalization)\n"
                "2. Learning rate too high (try reducing by 10x)\n"
                "3. Numerical instability in loss calculation"
            )
            return False
        else:
            return True
            
class FocalLoss(BaseLoss):
    def __init__(self, loss_type='mean', gamma=2, alpha=0.8, w1=0.3, w2=0.3, w3=0.3, w_bg=0.1):
        super().__init__(['ET', 'TC', 'WT'], {'BG': 0, 'NCR': 1, 'ED': 2, 'ET': 3}, 1e-5, w1, w2, w3)
        self.loss_type = loss_type
        self.gamma = gamma
        self.alpha = alpha
        self.w_bg = w_bg

    def __call__(self, y_pred, y_mask):
        """
        FocalLoss
        :param y_pred: 预测值 [batch, 4, D, W, H]
        :param y_mask: 真实值 [batch, D, W, H]
        """
        y_mask = F.one_hot(y_mask, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        pred_list, mask_list = self.split_sub_areas(y_pred, y_mask)

        loss_dict = {}
        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            loss = self.cal_focal_loss(pred, mask)
            loss_dict[sub_area] = loss

        et_loss = loss_dict['ET']
        tc_loss = loss_dict['TC']
        wt_loss = loss_dict['WT']
        mean_loss = (et_loss + tc_loss + wt_loss) / 3
        return mean_loss, et_loss, tc_loss, wt_loss

    def cal_focal_loss(self, y_pred, y_mask):
        cross_entropy = F.cross_entropy(y_pred, y_mask, reduction="mean")
        pt = torch.exp(-cross_entropy)
        focal_loss = (self.alpha * ((1 - pt) ** self.gamma) * cross_entropy)
        return focal_loss

class CELoss(BaseLoss):
    def __init__(self, loss_type='mean', smooth=1e-5, w1=0.3, w2=0.3, w3=0.3, w_bg=0.1):
        super().__init__(['ET', 'TC', 'WT'], {'BG': 0, 'NCR': 1, 'ED': 2, 'ET': 3}, smooth, w1, w2, w3)
        self.loss_type = loss_type
        self.w_bg = w_bg

    def __call__(self, y_pred, y_mask):
        """
        CrossEntropyLoss
        :param y_pred: 预测值 [batch, 4, D, W, H]
        :param y_mask: 真实值 [batch, D, W, H]
        """
        y_mask = F.one_hot(y_mask, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        pred_list, mask_list = self.split_sub_areas(y_pred, y_mask)

        loss_dict = {}
        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            ce_loss = F.cross_entropy(pred, mask, reduction="mean")
            loss_dict[sub_area] = ce_loss

        et_loss = loss_dict['ET']
        tc_loss = loss_dict['TC']
        wt_loss = loss_dict['WT']
        mean_loss = (et_loss + tc_loss + wt_loss) / 3
        return mean_loss, et_loss, tc_loss, wt_loss

# class H95DiceLoss(nn.Module):
#     def __init__(self, alpha=0.5, epsilon=1e-6):
#         super(H95DiceLoss, self).__init__()
#         self.alpha = alpha
#         self.epsilon = epsilon
        
#     def _compute_boundary(self, mask):
#         """计算二值掩膜的边界"""
#         kernel = torch.tensor([[[[-1, -1, -1],
#                                  [-1, 8, -1],
#                                  [-1, -1, -1]]]],
#                               dtype=torch.float32, device=mask.device)
        
#         boundaries = torch.abs(F.conv3d(mask.float(), kernel, padding=1))
        
#         return (boundaries > 0).float()
    
#     def _hausdorff_loss(self, pred, target):
#         """计算基于距离变化的Hausdorff近似损失"""
        
#         # 计算边界
#         pred_boundary = self._compute_boundary(pred)
#         target_boundary = self._compute_boundary(target)
        
#         # 计算距离变化
#         with torch.no_grad():
#             target_dist = self._distance_transform(target_boundary)
#             pred_dist = self._distance_transform(pred_boundary)
            
        # 计算双向距离损失
        
        

if __name__ == '__main__':
    y_pred = torch.randn(2, 4, 128, 128, 128)
    y_mask = torch.randint(0, 4, (2, 128, 128, 128))

    dice_loss_func = DiceLoss()
    ce_loss_func = CELoss()
    focal_loss_func = FocalLoss()

    print(f"DiceLoss: {dice_loss_func(y_pred, y_mask)}")
    print(f"CELoss: {ce_loss_func(y_pred, y_mask)}")
    print(f"FocalLoss: {focal_loss_func(y_pred, y_mask)}")