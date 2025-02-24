# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/12/30 15:12:35
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: shell工具
*      VERSION: v1.1
=================================================
'''
import subprocess
import sys
import time
import os
import argparse

def run_shell_command(command):
    try:
        start_time = time.time()
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elapsed_time = time.time() - start_time
        if result.returncode == 0:
            print(f"命令执行成功！耗时: {elapsed_time:.2f}秒\n输出: {result.stdout}")
        else:
            print(f"命令执行失败！耗时: {elapsed_time:.2f}秒\n错误信息: {result.stderr}", file=sys.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"执行命令时发生异常：{e}", file=sys.stderr)
        return False

def is_port_used(port):
    if not (0 < port <= 65535):
        print(f"端口号 {port} 不合法，请输入 1-65535 之间的端口号。", file=sys.stderr)
        return False
    command = f"lsof -i :{port}"
    return run_shell_command(command)

def kill_port_process(port):
    if not (0 < port <= 65535):
        print(f"端口号 {port} 不合法，请输入 1-65535 之间的端口号。", file=sys.stderr)
        return False
    if not is_port_used(port):
        print(f"端口 {port} 未被占用，无需清理。")
        return True
    command = f"lsof -t -i:{port} | xargs kill -9"
    print(f"正在清理端口 {port} 的占用...")
    return run_shell_command(command)

def start_tensorboard(log_path, port=6006, host='0.0.0.0'):
    if not os.path.exists(log_path):
        print(f"日志路径 {log_path} 不存在，请检查路径是否正确。", file=sys.stderr)
        return False
    
    if not (0 < port <= 65535):
        print(f"端口号 {port} 不合法，请输入 1-65535 之间的端口号。", file=sys.stderr)
        return False
    
    if not kill_port_process(port):
        print(f"清理端口 {port} 失败，启动 TensorBoard 失败。", file=sys.stderr)
        return False
    
    print(f"😃 正在启动 TensorBoard 面板...\n日志路径: {log_path}")
    command = f"nohup python3 -m tensorboard.main --logdir='{log_path}' --port={port} --host={host} > /dev/null 2>&1 &"
    if not run_shell_command(command):
        print(f"TensorBoard 启动失败！", file=sys.stderr)
        return False
    time.sleep(5)
    print(f"😃 TensorBoard 启动成功！\n请访问 http://{host}:{port} 查看 TensorBoard 面板。")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动 TensorBoard 面板")
    parser.add_argument("--log_path", type=str, default='/root/workspace/VoxelMedix/results/ScgaDasppResAtteUNet_2025-02-18_13-54-06/tensorBoard', help="TensorBoard 日志路径")
    parser.add_argument("--port", type=int, default=6008, help="TensorBoard 端口号")
    parser.add_argument("--host", type=str, default='0.0.0.0', help="TensorBoard 主机地址")
    args = parser.parse_args()
    
    start_tensorboard(args.log_path, args.port, args.host)
