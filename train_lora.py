import argparse
import os
import yaml
import torch
import torch.nn as nn
from transformers import VideoMAEImageProcessor, VideoMAEConfig

# Custom imports
from data.dataset import TIC
from models.model import TICModel
from utils.optimizer import get_optimizer_params
from utils.common import dict2namespace
from utils.engine import Trainer

os.environ['CUDA_VISIBLE_DEVICES'] = "7"

def get_args():
    parser = argparse.ArgumentParser(description='Training TIC Models with LoRA')
    parser.add_argument("--config", default='configs/config.yaml', type=str,
                        help="Path to the config file (yaml)")
    parser.add_argument("--test_checkpoint", type=str, default=None,
                        help="Path to checkpoint for evaluation (only used if is_train=False)")
    return parser.parse_args()


def main():
    args = get_args()
    
    # 1. 加载 Config
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)
    
    # 将 config 注入到 args 中，以便兼容 Dataset 的引用方式
    # Dataset 类中用到了 args.labels 和 args.sample_frames
    # 这里做一个简单的适配
    args.labels = config.data.labels
    args.sample_frames = config.train.sample_frames

    # Allow overriding is_train from command line via test_checkpoint presence
    # Only switch to eval mode if is_train was True and test_checkpoint is explicitly provided
    if args.test_checkpoint is not None and config.train.is_train:
        # Check if checkpoint file actually exists
        import os
        if os.path.exists(args.test_checkpoint):
            print(f"Test checkpoint provided: {args.test_checkpoint}. Switching to EVALUATION mode.")
            config.train.is_train = False
        else:
            print(f"Checkpoint file {args.test_checkpoint} not found. Starting training from scratch.")
            args.test_checkpoint = None


    # 2. 准备设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. 数据加载
    # 注意: Dataset 里还是使用了 tic.yml 结构的 config，这里需要确保传入的 config 结构匹配
    # 如果 config.yaml 结构变了，可能需要调整 Dataset 或在这里做适配
    # 假设 Dataset 需要 config.data.data_dir 等
    tic_data = TIC(config, args)
    train_loader, val_loader = tic_data.get_loaders()

    # 4. 模型初始化
    model_path = config.model.model_name
    print(f"Loading model config from {model_path}")
    model_config = VideoMAEConfig.from_pretrained(model_path)
    
    # 从 config 中覆盖一些模型配置 (如果需要)
    # model_config.num_labels = config.model.num_classes 
    
    print("Initializing TICModel...")
    model = TICModel(
        model_config, 
        num_classes=config.model.num_classes, 
        use_lora=config.model.use_lora
    )
    model = model.to(device)
    
    # 5. 图像处理器
    processor = VideoMAEImageProcessor.from_pretrained(model_path)

    # 6. 优化器 & 损失函数
    if config.train.is_train:
        optimizer_grouped_parameters = get_optimizer_params(
            model, 
            base_lr=float(config.train.optimizer.base_lr),
            weight_decay=float(config.train.optimizer.weight_decay),
            layer_decay=float(config.train.optimizer.layer_decay)
        )
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        loss_fn = nn.CrossEntropyLoss()
        
        # 7. 开始训练
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            processor=processor,
            config=config,
            args=args,
            device=device
        )
        
        print("Start Training...")
        trainer.run()
    else:
        # Evaluation mode
        if args.test_checkpoint is None:
            raise ValueError("Please provide --test_checkpoint when is_train=False in config")
            
        print(f"Loading checkpoint from {args.test_checkpoint}")
        # Load weights
        checkpoint = torch.load(args.test_checkpoint, map_location=device, weights_only=True)
        # Handle both full state dict and just model state dict if saved differently
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = Trainer(
            model=model,
            train_loader=None, # Not needed for eval
            val_loader=val_loader, # Use validation loader (or test loader if available)
            optimizer=None,
            loss_fn=loss_fn,
            processor=processor,
            config=config,
            args=args,
            device=device
        )
        
        trainer.evaluate()


if __name__ == "__main__":
    main()