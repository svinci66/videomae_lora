#!/bin/bash

# 安装依赖 (如果需要)
# pip install -r requirements.txt

# 运行训练/评测
# 评测模式：提供 --test_checkpoint 参数
# 训练模式：不提供 --test_checkpoint 参数
python train_lora.py --config configs/config.yaml 
#--test_checkpoint /data/sj/videomae_hf/videomae_lora/ckp/best_model.pth
