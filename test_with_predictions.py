import argparse
import os
import yaml
import torch
import torch.nn as nn
from transformers import VideoMAEImageProcessor, VideoMAEConfig
from data.dataset import TIC
from models.model import TICModel
from utils.common import dict2namespace
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "7"

def get_args():
    parser = argparse.ArgumentParser(description='Test TIC Models with detailed predictions')
    parser.add_argument("--config", default='configs/config_body_binary.yaml', type=str,
                        help="Path to the config file (yaml)")
    parser.add_argument("--checkpoint", type=str, default='ckp/body_binary/best_model.pth',
                        help="Path to checkpoint for evaluation")
    return parser.parse_args()

def main():
    args = get_args()
    
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)
    
    args.labels = config.data.labels
    args.sample_frames = config.train.sample_frames
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tic_data = TIC(config, args)
    _, val_loader = tic_data.get_loaders()
    
    model_path = config.model.model_name
    print(f"Loading model config from {model_path}")
    model_config = VideoMAEConfig.from_pretrained(model_path)
    
    print("Initializing TICModel...")
    model = TICModel(
        model_config, 
        num_classes=config.model.num_classes, 
        use_lora=config.model.use_lora
    )
    model = model.to(device)
    
    processor = VideoMAEImageProcessor.from_pretrained(model_path)
    
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    checkpoint_classifier_weight_shape = None
    for key in state_dict.keys():
        if 'classifier.1.weight' in key:
            checkpoint_classifier_weight_shape = state_dict[key].shape
            break
    
    if checkpoint_classifier_weight_shape is not None:
        current_shape = model.classifier[1].weight.shape
        if checkpoint_classifier_weight_shape != current_shape:
            print(f"Checkpoint classifier shape: {checkpoint_classifier_weight_shape}")
            print(f"Current model classifier shape: {current_shape}")
            print("Adjusting model classifier to match checkpoint...")
            num_classes_from_checkpoint = checkpoint_classifier_weight_shape[0]
            hidden_size = model_config.hidden_size
            model.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, num_classes_from_checkpoint)
            ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print("\n" + "="*80)
    print("开始预测每个测试样本...")
    print("="*80)
    
    all_predictions = []
    all_labels = []
    sample_idx = 0
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):
            input_frame = data.to(device)
            label = label.to(device)
            
            B, T, C, H, W = input_frame.shape
            label_flat = torch.argmax(label[:, 0, :], dim=1)
            
            input_frame_np = input_frame.permute(0, 1, 3, 4, 2).cpu().numpy()
            video_list = [list(frames) for frames in input_frame_np]
            
            if len(video_list[0]) < args.sample_frames:
                video_list[0] += [video_list[0][-1]] * (args.sample_frames - len(video_list[0]))
            
            inputs = processor(video_list, return_tensors="pt", do_rescale=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            logits = model(inputs['pixel_values'])
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            for i in range(B):
                pred_class = preds[i].item()
                true_class = label_flat[i].item()
                pred_prob = probs[i].cpu().numpy()
                
                pred_label_name = args.labels[pred_class]
                true_label_name = args.labels[true_class]
                
                all_predictions.append({
                    'sample_idx': sample_idx,
                    'batch_idx': batch_idx,
                    'batch_sample_idx': i,
                    'predicted_class': pred_class,
                    'predicted_label': pred_label_name,
                    'true_class': true_class,
                    'true_label': true_label_name,
                    'probabilities': pred_prob,
                    'is_correct': pred_class == true_class
                })
                
                all_labels.append(true_class)
                
                print(f"\n样本 #{sample_idx} (Batch {batch_idx}, 样本 {i}):")
                print(f"  真实标签: {true_label_name} (类别 {true_class})")
                print(f"  预测标签: {pred_label_name} (类别 {pred_class})")
                print(f"  预测概率: {pred_prob}")
                print(f"  是否正确: {'✓' if pred_class == true_class else '✗'}")
                
                sample_idx += 1
    
    print("\n" + "="*80)
    print("总结统计")
    print("="*80)
    
    correct_count = sum(1 for p in all_predictions if p['is_correct'])
    total_count = len(all_predictions)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"\n总样本数: {total_count}")
    print(f"正确预测: {correct_count}")
    print(f"错误预测: {total_count - correct_count}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n各类别统计:")
    for class_idx, class_name in enumerate(args.labels):
        true_count = sum(1 for p in all_predictions if p['true_class'] == class_idx)
        pred_count = sum(1 for p in all_predictions if p['predicted_class'] == class_idx)
        correct_for_class = sum(1 for p in all_predictions 
                                if p['true_class'] == class_idx and p['predicted_class'] == class_idx)
        
        if true_count > 0:
            class_acc = correct_for_class / true_count
        else:
            class_acc = 0
        
        print(f"\n类别 '{class_name}':")
        print(f"  真实样本数: {true_count}")
        print(f"  预测样本数: {pred_count}")
        print(f"  正确预测数: {correct_for_class}")
        print(f"  类别准确率: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    print("\n" + "="*80)
    print("混淆矩阵")
    print("="*80)
    print(f"\n真实 \\ 预测", end="")
    for class_name in args.labels:
        print(f"\t{class_name}", end="")
    print()
    
    confusion_matrix = np.zeros((len(args.labels), len(args.labels)), dtype=int)
    for p in all_predictions:
        confusion_matrix[p['true_class']][p['predicted_class']] += 1
    
    for i, true_class_name in enumerate(args.labels):
        print(f"{true_class_name}", end="")
        for j in range(len(args.labels)):
            print(f"\t{confusion_matrix[i][j]}", end="")
        print()
    
    print("\n" + "="*80)
    print("详细预测列表")
    print("="*80)
    print("\n格式: 样本ID | 真实标签 | 预测标签 | 是否正确")
    print("-" * 60)
    for p in all_predictions:
        status = "✓" if p['is_correct'] else "✗"
        print(f"#{p['sample_idx']:04d} | {p['true_label']:15s} | {p['predicted_label']:15s} | {status}")

if __name__ == "__main__":
    main()