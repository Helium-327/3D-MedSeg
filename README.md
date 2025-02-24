#  3D-MedSeg 🧠BraTS -- 3D Medical Segmentation for BraTS 


## 文件结构
```
./src
├── configs               # 配置文件
│   ├── 1_aaunet.yaml 
│   └── 1_unetr.yaml
├── datasets              # 数据集
│   ├── BraTS21.py
│   └── transforms.py
├── inference.py          # 推理脚本
├── lossFunc.py           # 损失函数
├── main.py               # 主函数
├── metrics.py            # 指标函数
├── nnArchitecture        # 网络结构
│   ├── baselines         # 基线网络
│   ├── optimization_nets # 优化网络
│   ├── ref_hetero_nets   # 异构网络
│   └── ref_homo_nets     # 同构网络
├── train.py              # 训练脚本
├── train_and_val.py      # 训练和验证脚本
└── utils                 # 工具函数
    ├── ckpt_tools.py     # 模型保存和加载
    ├── logger_tools.py   # 日志工具
    ├── shell_tools.py    # 命令行工具
    ├── tb_tools.py       # tensorboard工具
    └── test_unet.py      # 测试脚本
```

## 运行
```bash
python main.py --config configs/1_unetr.yaml
```


