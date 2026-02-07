import argparse
import torch

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def video_masking(input_frame, mask_ratio=0.75, mask_type='random'):
    B, T, C, H, W = input_frame.shape
    
    if mask_type == 'random':
        # 随机遮掩像素点
        mask = torch.rand(B, T, H, W) > mask_ratio

    mask = mask.unsqueeze(2)  # [B, T, 1, H, W]
    mask = mask.to(input_frame.device)
    masked_video = input_frame * mask
    
    return masked_video, mask.squeeze(2)
