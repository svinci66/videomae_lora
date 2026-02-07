def get_optimizer_params(model, base_lr, weight_decay=0.05, layer_decay=0.75):
    """
    实现分层学习率衰减：越靠近输入的层，学习率越小。
    """
    param_groups = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "backbone" in name:
            # 简单逻辑：根据层号进行衰减
            # VideoMAE encoder 通常有 12 层
            layer_id = 12 
            if "encoder.layer." in name:
                try:
                    layer_id = int(name.split("encoder.layer.")[1].split(".")[0])
                except:
                    layer_id = 0
            elif "embeddings" in name:
                layer_id = 0
            
            # 计算该层的 scale: layer_decay^(12 - layer_id)
            scale = layer_decay ** (12 - layer_id)
            lr = base_lr * scale
        else:
            # 分类头使用原始学习率
            lr = base_lr
            
        group_name = f"lr_{lr:.2e}"
        if group_name not in param_groups:
            param_groups[group_name] = {"params": [], "lr": lr, "weight_decay": weight_decay}
        param_groups[group_name]["params"].append(param)
        
    return list(param_groups.values())
