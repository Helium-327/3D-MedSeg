# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/07/23 15:28:23
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 训练流程
=================================================
'''

import os
import time
import shutil
import torch
import pandas as pd
import logging
import readline # 解决input()无法使用Backspace的问题, ⚠️不能删掉

from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter 

from inference import inference
from train_and_val import train_one_epoch, val_one_epoch

from utils.ckpt_tools import *
from utils.logger_tools import *
from utils.shell_tools import *

from torchinfo import summary


# constant
TB_PORT = 6007
RANDOM_SEED = 42
scheduler_start_epoch = 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加速
torch.backends.cudnn.benchmark = True        #! 加速固定输入/网络结构的训练，但需避免动态变化场景，如数据增强
torch.backends.cudnn.deterministic = True     #! 确保结果可复现，但可能降低性能并引发兼容性问题

# 调试工具
# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
# torch.autograd.set_detect_anomaly(True)     #! 检测梯度异常，但会降低性能（谨慎使用，哥们）

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)                 #让显卡产生的随机数一致

logger = logging.getLogger(__name__)

# 定义日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train(model, Metrics, train_loader,  val_loader, test_loader, scaler, optimizer, scheduler, loss_function, 
          num_epochs, device, results_dir, logs_path, output_path, start_epoch, best_val_loss, test_csv,
          tb=False,  
          interval=10, 
          save_max=10, 
          early_stopping_patience=10,
          resume_tb_path=None,
          ):
    """
    模型训练流程
    :param model: 模型
    :param train_loader: 训练数据集
    :param val_loader: 验证数据集
    :param optimizer: 优化器
    :param loss_function: 又称 criterion 损失函数
    :param num_epochs: 训练轮数
    :param device: 设备
    """
    best_epoch = 0
    save_counter = 0
    early_stopping_counter = 0
    date_time_str = get_current_date() + '_' + get_current_time()
    end_epoch = start_epoch + num_epochs
    model_name = model.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    scheduler_name = scheduler.__class__.__name__
    loss_func_name = loss_function.__class__.__name__
    test_df = pd.read_csv(test_csv)

    if resume_tb_path:
        tb_dir = resume_tb_path
    else:
        tb_dir = os.path.join(results_dir, f'tensorBoard')
    ckpt_dir = os.path.join(results_dir, f'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    if scheduler:
        current_lr = scheduler.get_last_lr()[0]
    else:
        current_lr = optimizer.param_groups[0]["lr"]
    # 初始化日志
    logger.info(f'开始训练, 训练轮数:{num_epochs}, {model_name}模型写入tensorBoard, 使用 {optimizer_name} 优化器, 学习率: {current_lr}, 损失函数: {loss_func_name}')

    writer = SummaryWriter(tb_dir)
    
    # 添加模型结构到tensorboard
    writer.add_graph(model, input_to_model=torch.rand(1, 4, 128, 128, 128).to(DEVICE))
    print(f'{model_name}模型写入tensorBoard, 使用 {optimizer_name} 优化器, 学习率: {optimizer.param_groups[0]["lr"]}, 损失函数: {loss_func_name}')
    
    for epoch in range(start_epoch, end_epoch):
        epoch += 1
        if scheduler:
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]
        """-------------------------------------- 训练过程 --------------------------------------------------"""
        print(f"=== Training on [Epoch {epoch}/{end_epoch}] ===:")
        
        train_mean_loss = 0.0
        start_time = time.time()
        # 训练
        train_running_loss, train_et_loss, train_tc_loss, train_wt_loss = train_one_epoch(model, train_loader, scaler, optimizer, loss_function, device)
        
        # 计算平均loss
        train_mean_loss =  train_running_loss / len(train_loader) 
        mean_train_et_loss = train_et_loss / len(train_loader)
        mean_train_tc_loss = train_tc_loss / len(train_loader)
        mean_train_wt_loss = train_wt_loss / len(train_loader)
        
        if scheduler_name == 'CosineAnnealingLR' and epoch > scheduler_start_epoch: # 从第20个epoch开始，使用余弦退火学习率
            scheduler.step()                    # 每种调度器的step方法不同，传入的参数也不一样
        writer.add_scalars(f'{loss_func_name}/train',
                           {'Mean':train_mean_loss, 'ET': mean_train_et_loss, 'TC': mean_train_tc_loss, 'WT': mean_train_wt_loss}, epoch)
        end_time = time.time()
        train_cost_time = end_time - start_time
        # print info
        print(f"- Train mean loss: {train_mean_loss:.4f}\n"
              f"- ET loss: {mean_train_et_loss:.4f}\n"
              f"- TC loss: {mean_train_tc_loss:.4f}\n"
              f"- WT loss: {mean_train_wt_loss:.4f}\n"
              f"- Cost time: {train_cost_time/60:.2f}mins ⏱️\n")
        
        """-------------------------------------- 验证过程 --------------------------------------------------"""
        if (epoch) % interval == 0:
            print(f"=== Validating on [Epoch {epoch}/{end_epoch}] ===:")
            
            # 开始计时
            start_time = time.time()
            
            # 验证
            val_running_loss, val_et_loss, val_tc_loss, val_wt_loss, Metrics_list= val_one_epoch(model, Metrics, val_loader, loss_function, epoch, device)
            
            # 计算平均loss
            val_mean_loss = val_running_loss / len(val_loader)
            mean_val_et_loss = val_et_loss / len(val_loader)
            mean_val_tc_loss = val_tc_loss / len(val_loader)
            mean_val_wt_loss = val_wt_loss / len(val_loader)
            Metrics_list = Metrics_list / len(val_loader)
            
            # 记录验证结果
            val_scores = {}                 # FIXME:去掉_scores
            val_scores['epoch'] = epoch
            val_scores['Dice_scores'] = Metrics_list[0] 
            val_scores['Jaccard_scores'] = Metrics_list[1]
            val_scores['Accuracy_scores'] = Metrics_list[2]
            val_scores['Precision_scores'] = Metrics_list[3]
            val_scores['Recall_scores'] = Metrics_list[4]
            val_scores['F1_scores'] = Metrics_list[5]
            val_scores['H95_scores'] = Metrics_list[6]
            # val_metrics.append(val_scores)
            
            """-------------------------------------- TensorBoard 记录验证结果 --------------------------------------------------"""
            writer.add_scalars(f'{loss_func_name}/val', 
                               {'Mean':val_mean_loss, 'ET': mean_val_et_loss, 'TC': mean_val_tc_loss, 'WT': mean_val_wt_loss}, 
                               epoch)
            # 记录对比结果
            writer.add_scalars(f'{loss_func_name}/Mean', 
                               {'Train':train_mean_loss, 'Val':val_mean_loss}, 
                               epoch)
            writer.add_scalars(f'{loss_func_name}/ET',
                               {'Train':mean_train_et_loss, 'Val':mean_val_et_loss}, 
                               epoch)
            writer.add_scalars(f'{loss_func_name}/TC',
                               {'Train':mean_train_tc_loss, 'Val':mean_val_tc_loss}, 
                               epoch)
            writer.add_scalars(f'{loss_func_name}/WT',
                               {'Train':mean_train_wt_loss, 'Val':mean_val_wt_loss}, 
                               epoch)

            if epoch == start_epoch+1:
                # 后台启动tensorboards面板
                start_tensorboard(tb_dir, port=TB_PORT)

            if tb: 
                writer.add_scalars('metrics/Dice_coeff',
                                    {'Mean':val_scores['Dice_scores'][0], 'ET': val_scores['Dice_scores'][1], 'TC': val_scores['Dice_scores'][2], 'WT': val_scores['Dice_scores'][3]},
                                    epoch)

                writer.add_scalars('metrics/Jaccard_index', {'Mean':val_scores['Jaccard_scores'][0], 'ET': val_scores['Jaccard_scores'][1], 'TC': val_scores['Jaccard_scores'][2], 'WT': val_scores['Jaccard_scores'][3]},
                                    epoch)   

                writer.add_scalars('metrics/Accuracy',
                                    {'Mean':val_scores['Accuracy_scores'][0], 'ET': val_scores['Accuracy_scores'][1], 'TC': val_scores['Accuracy_scores'][2], 'WT': val_scores['Accuracy_scores'][3]},
                                    epoch)
                
                writer.add_scalars('metrics/Precision', 
                                    {'ET': val_scores['Precision_scores'][1], 'TC': val_scores['Precision_scores'][2], 'WT': val_scores['Precision_scores'][3]},
                                    epoch)
                
                writer.add_scalars('metrics/Recall', 
                                    {'Mean':val_scores['Recall_scores'][0], 'ET': val_scores['Recall_scores'][1], 'TC': val_scores['Recall_scores'][2], 'WT': val_scores['Recall_scores'][3]},
                                    epoch)
                writer.add_scalars('metrics/F1', 
                                    {'Mean':val_scores['F1_scores'][0], 'ET': val_scores['F1_scores'][1], 'TC': val_scores['F1_scores'][2], 'WT': val_scores['F1_scores'][3]},
                                    epoch) 
                writer.add_scalars('metrics/H95', 
                                    {'Mean':val_scores['H95_scores'][0], 'ET': val_scores['H95_scores'][1], 'TC': val_scores['H95_scores'][2], 'WT': val_scores['H95_scores'][3]},
                                    epoch)                               
            
            end_time = time.time()
            val_cost_time = end_time - start_time
            
            """-------------------------------------- 打印指标 --------------------------------------------------"""
            metric_table_header = ["Metric_Name", "MEAN", "ET", "TC", "WT"]
            metric_table_left = ["Dice", "Jaccard", "Accuracy", "Precision", "Recall", "F1", "H95"]


            val_info_str =  f"=== [Epoch {epoch}/{end_epoch}] ===\n"\
                            f"- Model:    {model_name}\n"\
                            f"- Optimizer:{optimizer_name}\n"\
                            f"- Scheduler:{scheduler_name}\n"\
                            f"- LossFunc: {loss_func_name}\n"\
                            f"- Lr:       {current_lr:.6f}\n"\
                            f"- val_cost_time:{val_cost_time:.4f}s ⏱️\n"

            # 优化点：直接通过映射获取指标名称，避免重复字符串格式化
            def format_value(value, decimals=4):
                # 返回一个格式化后的字符串，保留指定的小数位数
                return f"{value:.{decimals}f}"
            
            metric_scores_mapping = {metric: val_scores[f"{metric}_scores"] for metric in metric_table_left}
            metric_table = [[metric,
                            format_value(metric_scores_mapping[metric][0]),
                            format_value(metric_scores_mapping[metric][1]),
                            format_value(metric_scores_mapping[metric][2]),
                            format_value(metric_scores_mapping[metric][3])] for metric in metric_table_left]
            loss_str = f"Mean Loss: {val_mean_loss:.4f}, ET: {mean_val_et_loss:.4f}, TC: {mean_val_tc_loss:.4f}, WT: {mean_val_wt_loss:.4f}\n"
            table_str = tabulate(metric_table, headers=metric_table_header, tablefmt='grid')
            metrics_info = val_info_str + table_str + '\n' + loss_str  
            
            # 将指标表格写入日志文件
            custom_logger(metrics_info, logs_path)
            print(metrics_info)
            
            """------------------------------------- 保存权重文件 --------------------------------------------"""
            best_dice = val_scores['Dice_scores'][0]
            # last_ckpt_path = os.path.join(ckpt_dir, f'{model_name}_braTS21_{loss_func_name}_{get_current_date() + ' ' + get_current_time()}_{epoch}_{val_mean_loss:.4f}_{best_dice:.4f}.pth')

            if val_mean_loss < best_val_loss:
                early_stopping_counter = 0
                best_val_loss = val_mean_loss
                best_epoch = epoch
                with open(os.path.join(os.path.dirname(logs_path), "current_log.txt"), 'a') as f:
                    f.write(f"=== Best EPOCH {best_epoch} ===:\n"\
                            f"@ {get_current_date() + ' ' + get_current_time()}\n"\
                            f"current lr : {current_lr:.6f}\n"\
                            f"loss: Mean:{val_mean_loss:.4f}\t ET: {mean_val_et_loss:.4f}\t TC: {mean_val_tc_loss:.4f}\t WT: {mean_val_wt_loss:.4f}\n"
                            f"mean dice : {val_scores['Dice_scores'][0]:.4f}\t" \
                            f"ET : {val_scores['Dice_scores'][1]:.4f}\t"\
                            f"TC : {val_scores['Dice_scores'][2]:.4f}\t" \
                            f"WT : {val_scores['Dice_scores'][3]:.4f}\n\n")
                # 保存最佳模型
                save_counter += 1
                best_ckpt_path = os.path.join(ckpt_dir, f'best@e{best_epoch}_{model_name}__{loss_func_name.lower()}{best_val_loss:.4f}_dice{best_dice:.4f}_{date_time_str}_{save_counter}.pth')
                if save_counter > save_max:
                    removed_ckpt = [ckpt for ckpt in os.listdir(ckpt_dir) if (ckpt.endswith('.pth') and (int(ckpt.split('.')[-2].split('_')[-1]) == int(save_counter - save_max)))] # 获取要删除的文件名
                    os.remove(os.path.join(ckpt_dir, removed_ckpt[0]))
                    print(f"🗑️ Due to reach the max save amount, Removed {removed_ckpt[0]}")
                    save_checkpoint(model, optimizer, scaler, best_epoch, best_val_loss, best_ckpt_path)
                else:
                    save_checkpoint(model, optimizer, scaler, best_epoch, best_val_loss, best_ckpt_path)
            else:
                # 早停策略，如果连续patience个epoch没有改进，则停止训练
                if early_stopping_counter == 0 :
                    continue
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        print(f"🎃 Early stopping at epoch {epoch} due to no improvement in validation loss.")
                        break
            
    print(f"😃😃😃Train finished. Best val loss: 👉{best_val_loss:.4f} at epoch {best_epoch}")
    # 训练完成后关闭 SummaryWriter
    writer.close() 
    # 将最后一个保存的权重文件重命名为 final_model.pth

    # 获取检查点目录中所有.pth文件
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]

    if ckpt_files:
        # 按最后修改时间排序，获取最新的文件
        latest_ckpt = max(ckpt_files, key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)))
        
        # 构建完整路径
        latest_ckpt_path = os.path.join(ckpt_dir, latest_ckpt)
        final_model_path = os.path.join(ckpt_dir, f'{model_name}_final_model.pth')
        
        # 复制文件
        shutil.copy(latest_ckpt_path, final_model_path)
        print(f"✅ 最后一个权重文件已复制为 {final_model_path}")
    else:
        print(f"⚠️ 没有找到任何权重文件在 {ckpt_dir}")
    
    output_path = os.path.join(output_path, model_name, get_current_date()+'_'+get_current_time())
    # 自动推理
    inference(
        test_df=test_df,
        test_loader=test_loader, 
        output_path=output_path, 
        model=model,
        optimizer=optimizer, 
        Metricer=Metrics,
        scaler=scaler,
        ckpt_path=latest_ckpt_path,
        window_size=(128, 128, 128), 
        stride_ratio=0.5, 
        save_flag=True,
        device=DEVICE
        )
    print(f"🎉🎉🎉推理完成，结果保存在 {output_path}")
    
    

    