import torch
import torch.nn as nn
from transformers import VideoMAEModel
from peft import get_peft_model, LoraConfig, TaskType

class TICModel(nn.Module):
    def __init__(self, config, num_classes=3, use_lora=True):
        super().__init__()
        # 1. 加载基础模型
        self.backbone = VideoMAEModel.from_pretrained(
            "/data/sj/videomae_hf/videomae-base-finetuned-kinetics",
            config=config
        )
        
        # 2. 应用 LoRA (如果启用)
        if use_lora:
            print(">>> 正在应用 LoRA 微调...")
            peft_config = LoraConfig(
                #task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False, 
                r=16, 
                lora_alpha=32, 
                lora_dropout=0.1,
                target_modules=["query", "value"] 
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            self.backbone.print_trainable_parameters()

        # ========== 关键修复在这里 ==========
        # 必须显式定义 hidden_size，才能在下面的 nn.Linear 中使用
        hidden_size = config.hidden_size 
        # ==================================

        # 3. Attention Pooling 层
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # 4. 最终分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        last_hidden_state = outputs.last_hidden_state  # [B, T, Hidden]
        
        attn_weights = self.attn(last_hidden_state)    # [B, T, 1]
        context = torch.sum(attn_weights * last_hidden_state, dim=1) # [B, Hidden]
        
        logits = self.classifier(context)
        return logits
