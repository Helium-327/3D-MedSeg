# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/02/28 15:12:56
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION:  加载模型
*      VERSION: v1.0
*      FEATURES: 
=================================================
'''

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.optim import Adam, SGD, RMSprop, AdamW
from metrics import EvaluationMetrics
from lossFunc import *
from metrics import *

# baseline网络
from nnArchitecture.baselines.UNet3d import UNet3D
from nnArchitecture.baselines.AttentionUNet import AttentionUNet3D

# 优化网络
from nnArchitecture.optimization_nets.DasppResAtteUNet import DasppResAtteUNet
from nnArchitecture.optimization_nets.ScgaResAtteUNet import ScgaResAtteUNet
from nnArchitecture.optimization_nets.ScgaDasppResAtteUNet import ScgaDasppResAtteUNet
from nnArchitecture.optimization_nets.AA_UNet import AAUNet
from nnArchitecture.optimization_nets.AnisotrpicUNet import AnisotrpicUNet
from nnArchitecture.optimization_nets.EagAttnUNet import EagAttnUNet

# 同构网络
from nnArchitecture.ref_homo_nets.unetr import UNETR
from nnArchitecture.ref_homo_nets.unetrpp import UNETR_PP
from nnArchitecture.ref_homo_nets.segFormer3d import SegFormer3D

# 异构网络
# from nnArchitecture.ref_hetero_nets.Mamba3d import Mamba3d
# from nnArchitecture.ref_hetero_nets.MogaNet import MogaNet


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def load_paths(args):
#     """加载路径"""
#     paths = {}
#     paths['root'] = args.root
#     paths['resume'] = args.resume if args.reume else None
#     paths['data'] = args.data_dir if args.data_dir else os.path.join(args.root, 'data')
#     paths['output'] = args.output_dir if args.output_dir else os.path.join(args.root, 'output')
#     paths['results'] = args.results_dir if args.results_dir else os.path.join(args.root, 'results')
#     paths['train_csv'] = args.train_csv_path if args.train_csv_path else os.path.join(paths['data'], 'brats21_original', 'train.csv')
#     paths['val_csv'] = args.val_csv_path if args.val_csv_path else os.path.join(paths['data'], 'brats21_original', 'val.csv')
#     paths['test_csv'] = args.test_csv_path if args.test_csv_path else os.path.join(paths['data'], 'brats21_original', 'test.csv')
#     return paths

def load_model(args):
    """加载模型"""
    if args.model_name == 'unet3d':
        model = UNet3D(in_channels=4, out_channels=4)
    elif args.model_name == 'attention_unet3d':
        model = AttentionUNet3D(in_channels=4, out_channels=4)
    elif args.model_name == 'unetr':
        model = UNETR(
        in_channels=4,
        out_channels=4,
        img_size=(128, 128, 128),
        feature_size=16,
        hidden_size=768,
        num_heads=12,
        spatial_dims=3,
        predict_mode=False  # 设置为预测模式
    )
    elif args.model_name == 'unetrpp':
        model  = UNETR_PP(
        in_channels=4,
        out_channels=4,  # 假设分割为2类
        feature_size=16,
        hidden_size=256,
        num_heads=8,
        pos_embed="perceptron",
        norm_name="instance",
        dropout_rate=0.1,
        depths=[3, 3, 3, 3],
        dims=[32, 64, 128, 256],
        conv_op=nn.Conv3d,
        do_ds=False,
    )
    elif args.model_name == 'segformer':
        model = SegFormer3D(in_channels=4, out_channels=4)
    elif args.model_name == 'mamba3d':
        model = Mamba3d(in_channels=4, out_channels=4)
    elif args.model_name == 'moganet':
        model = MogaNet(in_channels=4, out_channels=4)
    elif args.model_name == 'd_atte_unet':
        model = DasppResAtteUNet(in_channels=4, out_channels=4)
    elif args.model_name == 's_atte_unet':
        model = ScgaResAtteUNet(in_channels=4, out_channels=4)
    elif args.model_name == 'ds_atte_unet':
        model = ScgaDasppResAtteUNet(in_channels=4, out_channels=4)
    elif args.model_name == 'aa_unet':
        model = AAUNet(in_channels=4, out_channels=4)
    elif args.model_name == 'ani_attn_unet':
        model = AnisotrpicUNet(in_channels=4, out_channels=4)
    elif args.model_name == 'eag_attn_unet':
        model = EagAttnUNet(in_channels=4, out_channels=4)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    
    model = model.to(DEVICE)
    
    return model


def load_optimizer(args, model):
    """加载优化器"""
    optimizer = AdamW(model.parameters(), lr=float(args.lr), betas=(0.9, 0.99), weight_decay=float(args.wd))
    return optimizer

def load_scheduler(args, optimizer):
    """加载调度器"""
    scheduler = CosineAnnealingLR(optimizer, T_max=int(args.cosine_T_max), eta_min=float(args.cosine_eta_min))
    return scheduler

def load_loss(args):
    """加载损失函数"""
    if args.loss_type == 'diceloss':
        loss_function = DiceLoss()
    elif args.loss_type == 'bceloss':
        loss_function = FocalLoss()
    elif args.loss_type == 'celoss':
        loss_function = CELoss()
    else:
        raise ValueError(f"Loss function {args.loss_type} not supported")
    return loss_function