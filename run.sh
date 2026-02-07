#!/bin/bash

# 安装依赖 (如果需要)
# pip install -r requirements.txt

# 运行训练
# 你可以通过修改 configs/config.yaml 来调整参数
python train_lora.py --config configs/config.yaml
