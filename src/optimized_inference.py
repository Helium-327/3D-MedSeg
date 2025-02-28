# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/24 15:49:32
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 优化之后的推理代码
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''

import os
import time
import shutil
from humanize import metric
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict, Any, Optional

from datasets.BraTS21 import BraTS21_3D
from datasets.transforms import Compose, FrontGroundNormalize, RandomCrop3D, ToTensor
from lossFunc import DiceLoss, CELoss
from metrics import *
from utils.logger_tools import custom_logger, get_current_date, get_current_time
from utils.ckpt_tools import load_checkpoint
from nnArchitecture.baselines.UNet3d import UNet3D
from nnArchitecture.baselines.AttentionUNet import AttentionUNet3D

from nnArchitecture.optimization_nets.DasppResAtteUNet import DasppResAtteUNet
from nnArchitecture.optimization_nets.ScgaResAtteUNet import ScgaResAtteUNet
from nnArchitecture.optimization_nets.AA_UNet import AAUNet


from nnArchitecture.ref_homo_nets.unetr import UNETR
from nnArchitecture.ref_homo_nets.unetrpp import UNETR_PP
from nnArchitecture.ref_homo_nets.segFormer3d import SegFormer3D

from nnArchitecture.ref_hetero_nets.Mamba3d import Mamba3d
from nnArchitecture.ref_hetero_nets.MogaNet import MogaNet

Tensor = torch.Tensor
Model = torch.nn.Module
DataLoader = torch.utils.data.DataLoader
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AdamW = torch.optim.AdamW
GradScaler = torch.amp.GradScaler
autocast = torch.amp.autocast('cuda')

def load_model(model_name):
    """加载模型"""
    if model_name == 'UNet3D':
        model = UNet3D(in_channels=4, out_channels=4)
    elif model_name == 'AttentionUNet3D':
        model = AttentionUNet3D(in_channels=4, out_channels=4)
    elif model_name == 'UNETR':
        model = UNETR(in_channels=4, out_channels=4)
    elif model_name == 'UNETR_PP':
        model = UNETR_PP(in_channels=4, out_channels=4)
    elif model_name == 'SegFormer3D':
        model = SegFormer3D(in_channels=4, out_channels=4)
    elif model_name == 'Mamba3d':
        model = Mamba3d(in_channels=4, out_channels=4)
    elif model_name == 'MogaNet':
        model = MogaNet(in_channels=4, out_channels=4)
    elif model_name == 'DasppResAtteUNet':
        model = DasppResAtteUNet(in_channels=4, out_channels=4)
    elif model_name == 'ScgaResAtteUNet':
        model = ScgaResAtteUNet(in_channels=4, out_channels=4)
    elif model_name == 'AAUNet':
        model = AAUNet(in_channels=4, out_channels=4)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model = model.to(Device)
    
    return model


def load_data(test_csv, local_train=True, test_length=10, batch_size=1, num_workers=4):
    """加载数据集"""
    TransMethods_test = Compose([
        ToTensor(),
        RandomCrop3D(size=(155, 240, 240)),
        FrontGroundNormalize(),
    ])

    test_dataset = BraTS21_3D(
        data_file=test_csv,
        transform=TransMethods_test,
        local_train=local_train,
        length=test_length,
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True  # 减少 worker 初始化时间
    )
    
    print(f"已加载测试数据: {len(test_loader)}")
    return test_loader


def inference(
    test_df: pd.DataFrame,
    test_loader: DataLoader,
    output_root: str,
    model: Model,
    metricer: EvaluationMetrics,
    scaler: torch.cuda.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    ckpt_path: str,
    affine: Optional[np.ndarray] = None,
    window_size: list[int, int, int] | int = 128,
    stride_ratio: float = 0.5,
    save_flag: bool = True,
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Dict[str, float]:
    """
    医学影像推理主函数
    
    参数说明：
    test_df: 包含病例元数据的DataFrame
    test_loader: 测试数据加载器
    output_root: 输出根目录
    model: 训练好的模型
    metricer: 指标计算器
    ...（其他参数说明）
    """
    # 初始化配置
    affine = affine or default_affine()
    start_time = time.time()
    
    # 模型加载优化
    model, optimizer, scaler, _, _ = load_checkpoint(model, optimizer, scaler, ckpt_path)
    model.to(device)
    
    # 创建输出目录
    output_path, new_ckpt_path = init_output_dir(output_root, model, ckpt_path)
    
    # 异步执行器
    executor = ThreadPoolExecutor(max_workers=4)
    
    # 指标容器
    metrics_accumulator = np.zeros((7, 4))
    
    for i, data in enumerate(tqdm(test_loader, desc="推理进度")):
        vimage, vmask = data[0], data[1]
        
        # 半精度推理优化[10](@ref)
        with torch.no_grad():
            pred_vimage = slide_window_pred(vimage.half(), model, window_size, stride_ratio)
        
        case_id = test_df.iloc[i]['patient_idx']
        
        # 异步保存结果
        if save_flag:
            future = executor.submit(
                async_save_results,
                test_df, pred_vimage, vmask, output_path, affine, case_id
            )
            future.add_done_callback(lambda x: print(f"案例 {case_id} 保存完成") if x.exception() is None else None)
        
        # 指标计算
        batch_metrics = metricer.update(y_pred=pred_vimage, y_mask=vmask)
        metrics_accumulator += batch_metrics
        
        # 内存优化[9](@ref)
        del pred_vimage, vmask
        torch.cuda.empty_cache()
    
    # 生成报告
    final_metrics = process_metrics(metrics_accumulator / len(test_loader))
    generate_report(
        model_name=model.__class__.__name__,
        ckpt_path=new_ckpt_path,
        inference_time=time.time() - start_time,
        metrics=final_metrics,
        output_path=output_path
    )
    
    return final_metrics

def slide_window_pred(inputs, model, roi_size=128, sw_batch_size=4, overlap=0.5, mode="gaussian"):
    """
    BraTS专用高效滑窗推理函数
    参数：
        inputs: 输入张量 (b, 4, 155, 240, 240)
        model: 训练好的分割模型
        roi_size: 窗口大小，默认128x128x128
        sw_batch_size: 滑动窗口批大小
        overlap: 窗口重叠率
        mode: 融合模式("gaussian"/"constant")
    """
    # 设备配置
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    
    # 计算滑动步长
    strides = [int(roi_size * (1 - overlap))] * 3
    strides[0] = roi_size  # 深度方向不重叠
    
    # 生成滑动窗口坐标
    dims = inputs.shape[2:]

    num_blocks = [int(np.ceil(d / s)) for d, s in zip(dims, strides)]
    
    # 初始化输出概率图和计数图
    output_map = torch.zeros((inputs.shape[0], 4, *dims), device=device)
    count_map = torch.zeros((1, 1, *dims), device=device)
    
    # 生成高斯权重窗口
    if mode == "gaussian":
        sigma = 0.125 * roi_size
        coords = torch.arange(roi_size, device=device).float()
        grid = torch.stack(torch.meshgrid(coords, coords, coords), dim=-1)
        center = roi_size // 2
        weights = torch.exp(-torch.sum((grid - center)**2, dim=-1) / (2 * sigma**2))
        weights = weights / weights.max()
    else:
        weights = torch.ones((roi_size, roi_size, roi_size), device=device)
    
    # 滑窗推理
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for d in tqdm(range(num_blocks[0])):
            for h in range(num_blocks[1]):
                for w in range(num_blocks[2]):
                    # 计算当前窗口坐标
                    d_start = min(d * strides[0], dims[0] - roi_size)
                    h_start = min(h * strides[1], dims[1] - roi_size)
                    w_start = min(w * strides[2], dims[2] - roi_size)
                    
                    # 提取窗口数据
                    window = inputs[
                        :, :,
                        d_start:d_start+roi_size,
                        h_start:h_start+roi_size,
                        w_start:w_start+roi_size
                    ]
                    
                    # 模型推理
                    pred = model(window)
                    
                    # 加权融合
                    output_map[
                        :, :,
                        d_start:d_start+roi_size,
                        h_start:h_start+roi_size,
                        w_start:w_start+roi_size
                    ] += pred * weights
                    
                    count_map[
                        :, :,
                        d_start:d_start+roi_size,
                        h_start:h_start+roi_size,
                        w_start:w_start+roi_size
                    ] += weights
                    
                    # 显存清理
                    del window, pred
                    if w % 4 == 0:
                        torch.cuda.empty_cache()
    
    # 归一化输出
    output_map /= count_map
    return output_map.cpu()


def init_output_dir(output_root: str, model: Model, ckpt_path: str) -> Tuple[str, str]:
    """初始化输出目录结构"""
    timestamp = f"{pd.Timestamp.now():%Y%m%d_%H:%M:%S}"
    output_path = os.path.join(output_root, f"{model.__class__.__name__}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    # 复制检查点文件
    new_ckpt_path = shutil.copy(ckpt_path, os.path.join(output_path, os.path.basename(ckpt_path)))
    return output_path, new_ckpt_path

def async_save_results(
    test_df: pd.DataFrame,
    pred_vimage: Tensor,
    vmask: Tensor,
    output_path: str,
    affine: np.ndarray,
    case_id: str
) -> None:
    """异步保存推理结果"""
    try:
        # 生成预测结果
        test_output_argmax = torch.argmax(pred_vimage, dim=1).to(torch.int64)
        
        # 保存NIfTI文件
        save_nii(
            mask=vmask[0].permute(1, 2, 0).cpu().numpy().astype(np.int8),
            pred=test_output_argmax[0].permute(1, 2, 0).cpu().numpy().astype(np.int8),
            output_dir=os.path.join(output_path, str(case_id)),
            affine=affine,
            case_id=case_id
        )
        
        # 复制原始数据
        case_dir = test_df.loc[test_df['patient_idx'] == case_id, 'patient_dir'].values[0]
        for fname in os.listdir(case_dir):
            shutil.copy(os.path.join(case_dir, fname), os.path.join(output_path, str(case_id), fname))
            
    except Exception as e:
        print(f"案例 {case_id} 保存失败: {str(e)}")
        raise

def save_nii(
    mask: np.ndarray,
    pred: np.ndarray,
    output_dir: str,
    affine: np.ndarray,
    case_id: str
) -> None:
    """保存NIfTI文件"""
    os.makedirs(output_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(mask, affine), os.path.join(output_dir, f'{case_id}_mask.nii.gz'))
    nib.save(nib.Nifti1Image(pred, affine), os.path.join(output_dir, f'{case_id}_pred.nii.gz'))

def process_metrics(raw_metrics: np.ndarray) -> Dict[str, Tuple[float, float, float, float]]:
    """处理指标数据"""
    return {
        'Dice': raw_metrics[0],
        'Jaccard': raw_metrics[1],
        'Accuracy': raw_metrics[2],
        'Precision': raw_metrics[3],
        'Recall': raw_metrics[4],
        'F1': raw_metrics[5],
        'H95': raw_metrics[6]
    }

def generate_report(
    model_name: str,
    ckpt_path: str,
    inference_time: float,
    metrics: Dict[str, Tuple[float, float, float, float]],
    output_path: str
) -> None:
    """生成评估报告"""
    # 构建表格数据
    table_data = [
        [metric, *[f"{v:.4f}" for v in values]]
        for metric, values in metrics.items()
    ]
    
    # 生成报告内容
    report = f"""
╒══════════════════════════════════╕
│       医学影像推理报告            │
╘══════════════════════════════════╛
    
[模型信息]
模型名称: {model_name}
检查点路径: {ckpt_path}
推理耗时: {inference_time:.2f}秒
    
[评估指标]
{tabulate(table_data, headers=["指标", "MEAN", "ET", "TC", "WT"], tablefmt="fancy_grid")}
    
[输出目录]
{output_path}
"""
    
    # 保存报告
    with open(os.path.join(output_path, "report.txt"), 'w') as f:
        f.write(report)
    print(report)

def default_affine() -> np.ndarray:
    """生成默认仿射矩阵"""
    return np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 239],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


if __name__ == '__main__':
    
    csv_file = '/root/workspace/VoxelMedix/data/raw/brats21_original/test.csv'
    
    out_dir = '/mnt/d/results'
    if not os.path.exists(out_dir):
        out_dir = '../../output'
        
    model_names = ['UNet3D', 'AttentionUNet3D']
    test_df = pd.read_csv(csv_file)
    test_loader = load_data(csv_file)

    # optimizer = AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0.00001)
    unet_model = load_model('UNet3D') 
    attention_unet_model = load_model('AttentionUNet3D')
    attention_unet_denseaspp_model = load_model('DasppResAtteUNet')
    attention_unet_scga_model = load_model('ScgaResAtteUNet')
    attention_unet_aa_model = load_model('AAUNet')
    
        
    # 初始化配置
    unet_config = {
        'test_df': test_df,
        'test_loader': test_loader,
        'output_root': out_dir,
        'model': unet_model,
        'metricer': EvaluationMetrics(),
        'scaler': GradScaler(),
        'optimizer': AdamW(unet_model.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0.00001),
        'ckpt_path': '/root/workspace/VoxelMedix/output/UNet3D_2025-02-24_15-15-01/best@e117_UNet3D__diceloss0.1605_dice0.8397_2025-02-16_18-07-54_19.pth'
    }
    attention_unet_config = {
        'test_df': test_df,
        'test_loader': test_loader,
        'output_root': out_dir,
        'model': attention_unet_model,
        'metricer': EvaluationMetrics(),
        'scaler': GradScaler(),
        'optimizer': AdamW(attention_unet_model.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0.00001),
        'ckpt_path': '/root/workspace/VoxelMedix/output/AttentionUNet3D_final_model.pth'
    }
    attention_unet_denseaspp_config = {
        'test_df': test_df,
        'test_loader': test_loader,
        'output_root': out_dir,
        'model': attention_unet_denseaspp_model,
        'metricer': EvaluationMetrics(),
        'scaler': GradScaler(),
        'optimizer': AdamW(attention_unet_denseaspp_model.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0.00001),
        'ckpt_path': '/root/workspace/BraTS_Solution/results/DasppResAtteUNet_2025-02-25_18-24-52/checkpoints/best@e50_DasppResAtteUNet__diceloss0.1436_dice0.8567_2025-02-25_18-24-52_11.pth'
    }
    attention_unet_scga_config = {
        'test_df': test_df,
        'test_loader': test_loader,
        'output_root': out_dir,
        'model': attention_unet_scga_model,
        'metricer': EvaluationMetrics(),
        'scaler': GradScaler(),
        'optimizer': AdamW(attention_unet_scga_model.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0.00001),
        'ckpt_path': '/root/workspace/BraTS_Solution/results/best@e53_ScgaResAtteUNet__diceloss0.1556_dice0.8447_2025-02-25_18-29-04_14.pth'
    }

    attention_aa_unet_cfg = {
        'test_df': test_df,
        'test_loader': test_loader,
        'output_root': out_dir,
        'model': attention_unet_aa_model,
        'metricer': EvaluationMetrics(),
        'scaler': GradScaler(),
        'optimizer': AdamW(attention_unet_aa_model.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0.00001),
        'ckpt_path': '/root/workspace/BraTS_Solution/results/best@e136_AAUNet__diceloss0.1404_dice0.8599_2025-02-25_19-55-50_9.pth'
    }
    # 执行推理
    # unet_results = inference(**unet_config)
    # attention_unet_results = inference(**attention_unet_config)
    # attention_unet_denseaspp_results = inference(**attention_unet_denseaspp_config)
    # attention_unet_scga_results = inference(**attention_unet_scga_config)
    attention_unet_aa_results = inference(**attention_aa_unet_cfg)