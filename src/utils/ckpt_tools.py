# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/30 15:11:38
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 模型保存和加载
*      VERSION: v1.0
=================================================
'''

import torch
import os

def load_checkpoint(model, optimizer, scaler, checkpoint_path):
    best_val_loss = float('inf')
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    # log_path = checkpoint['log_path']
    print(f"***Resuming training from epoch {start_epoch}...")
    return model, optimizer, scaler, start_epoch, best_val_loss

def save_checkpoint(model, optimizer, scaler, epoch, best_val_loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict()
        }
    torch.save(checkpoint, checkpoint_path)
    print(f"✨Saved {os.path.basename(checkpoint_path)} under {os.path.dirname(checkpoint_path)}")

    
    