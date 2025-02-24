# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/30 15:10:41
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 日志记录
*      VERSION: v1.0
=================================================
'''
import os

from datetime import datetime

# 获取当前日期
def get_current_date():
    return datetime.now().strftime('%Y-%m-%d')

# 获取当前时间
def get_current_time():
    return datetime.now().strftime('%H-%M-%S')

def custom_logger(content, file_pth, log_time=False):
    '''
    自定义日志写入函数
    :param content: 日志内容
    :param file_pth: 日志文件路径
    '''
    now = get_current_date() + " " + get_current_time()
    if not os.path.exists(file_pth):
        with open(file_pth, 'a') as f:
            if log_time:
                f.write(f"log Time: {now}\n")
            f.write(content + '\n')
    else:
        with open(file_pth, 'a') as f:
            if log_time:
                f.write(f"log Time: {now}\n")
            f.write(content + '\n')

def write_commit_file(file_path, content):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("# UNet3d brats实验记录\n\n")
            f.write("## 实验记录\n")
    with open(file_path, 'a') as f:
        f.write(f"- **Commit on:**`{get_current_date() +'-' + get_current_time()}`\n")
        f.write(f"  > {content}\n")
        
# 根据当前日期创建文件夹
def create_folder(folder_path):
    if os.path.exists(folder_path):
        folder_path = os.path.join(folder_path, get_current_date() + '_' + get_current_time())
        print(f"当前日期文件夹已存在，将创建时间文件夹：{folder_path} ")
        os.makedirs(folder_path)
    else: # 如果日期文件夹不存在，则先创建日期文件夹， 再创建时间文件夹
        os.makedirs(folder_path)
        print(f"当前日期文件夹 '{folder_path}' 已经创建。")
        folder_path = os.path.join(folder_path, get_current_date() + '_' + get_current_time())
        os.makedirs(folder_path)
        print(f"当前时间文件夹 '{folder_path}' 已经创建。")
    return folder_path