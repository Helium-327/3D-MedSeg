# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/30 15:20:24
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: tensorboard事件文件处理工具
*      VERSION: v1.0
=================================================
'''

import os
from tqdm import tqdm
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter


def read_tb_events(input_path):
    """读取现有的events文件
    :param input_path: 输入的events文件路径
    """
    # 读取现有的events文件
    ea = event_accumulator.EventAccumulator(input_path)
    ea.Reload()
    tags = ea.scalars.Keys()

    return ea, tags

def refix_one_tb_events(writer,ea, tags, cutoff_step):
    """修改tensorboard事件文件，使其能够在指定步数后追加
    :param writer: SummaryWriter对象
    :param ea: EventAccumulator对象
    :param tags: 标签列表
    :param cutoff_step: 截断步数，可以是resume的保存步数
    """
    # 遍历所有的标签，读取数据，并写入步数小于或等于cutoff_step的数据
    for tag in tags:
        scalar_list = ea.scalars.Items(tag)
        for scalar in scalar_list:
            if scalar.step <= cutoff_step:
                # 写入步数小于或等于cutoff_step的数据
                writer.add_scalar(tag, scalar.value, scalar.step, scalar.wall_time)
                
    # 关闭SummaryWriter
    writer.close()

def cutoff_tb_data(tb_path, cutoff_step):
    tb_events_file_name = [os.path.join(root, file) for root, dirs, files in os.walk(tb_path) for file in files if file.startswith('events.out.tfevents')]
    
    # 确保至少有一个文件被找到
    if not tb_events_file_name:
        raise FileNotFoundError(f"No events files found in path {tb_path}")
    else:
        for file in tqdm(tb_events_file_name):
            # 读取现有的events文件
            writer = SummaryWriter(os.path.dirname(file))
            ea, tags = read_tb_events(file)
            refix_one_tb_events(writer,ea, tags, cutoff_step)
            os.rename(file, file + '.bak')
            os.remove(file + '.bak')

    
if __name__ == "__main__":
    
    # 读取现有的events文件

    cutoff_step = 6
    # 读取现有的events文件
    tb_path = '/mnt/d/AI_Research/WS-HUB/WS-segBratsWorkflow/Helium-327-SegBrats/results/2024-09-25/20-34-16/tensorBoard/UNet3D_braTS21_2024-09-25_20-34-16'

    cutoff_tb_data(tb_path, cutoff_step)