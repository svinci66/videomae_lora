import torch
import torch.nn as nn
import os
import json

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, processor, config, args, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.processor = processor
        self.config = config
        self.args = args 
        self.device = device
        self.best_val_acc = 0.0

        # Create checkpoint directory immediately
        if self.config.train.save_ckp:
            os.makedirs(self.config.train.save_ckp, exist_ok=True)


        # 从 config 中读取参数 (兼容旧 args 结构)
        self.num_epochs = config.data.training.n_epochs
        self.sample_frames = config.train.sample_frames
        self.save_ckp = config.train.save_ckp
        # snapshot_freq handling
        self.snapshot_freq = getattr(config.data.training, 'snapshot_freq', 10) 
    
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        num_batches = len(self.train_loader)
        
        for i, (data, label) in enumerate(self.train_loader):
            input_frame = data.to(self.device) # [B, T, C, H, W]
            label = label.to(self.device)
            B, T, C, H, W = input_frame.shape

            # 这里可以加入 masking 逻辑，如果需要
            # input_frame, mask = video_masking(input_frame)

            # 只用第0帧标签 (假设标签是视频级)
            label_flat = torch.argmax(label[:, 0, :], dim=1)  # [B]

            # 预处理: 转为 [B, T, H, W, C] 并转 list 供 HF processor 使用
            input_frame_np = input_frame.permute(0, 1, 3, 4, 2).cpu().numpy()
            video_list = [list(frames) for frames in input_frame_np]

            # 批量送入 processor
            inputs = self.processor(video_list, return_tensors="pt", do_rescale=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 前向传播
            logits = self.model(inputs['pixel_values'])

            # 计算损失
            loss = self.loss_fn(logits, label_flat)

            # 反向传播与优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            preds = torch.argmax(logits, dim=1)
            correct = (preds == label_flat).sum().item()
            total_correct += correct
            total_samples += B
            total_loss += loss.item() * B

            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                print(f"Epoch [{epoch+1}/{self.num_epochs}] Step [{i+1}/{num_batches}] "
                      f"Loss: {loss.item():.4f} Acc: {correct/B:.4f}")

        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples
        print(f"Epoch [{epoch+1}/{self.num_epochs}] Train Loss: {avg_loss:.4f} Train Acc: {acc:.4f}")
        return avg_loss, acc

    def validate(self, epoch):
        self.model.eval()
        val_total = 0
        val_correct = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for data, label in self.val_loader:
                input_frame = data.to(self.device)
                label = label.to(self.device)

                B, T, C, H, W = input_frame.shape
                label_flat = torch.argmax(label[:, 0, :], dim=1)

                input_frame_np = input_frame.permute(0, 1, 3, 4, 2).cpu().numpy()
                video_list = [list(frames) for frames in input_frame_np]
                
                # 补齐不足的帧数
                if len(video_list[0]) < self.sample_frames:
                    video_list[0] += [video_list[0][-1]] * (self.sample_frames - len(video_list[0]))
                
                inputs = self.processor(video_list, return_tensors="pt", do_rescale=False)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                logits = self.model(inputs['pixel_values'])
                loss = self.loss_fn(logits, label_flat)

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == label_flat).sum().item()
                val_total += B
                val_loss += loss.item() * B

        avg_loss = val_loss / val_total
        acc = val_correct / val_total
        print(f"Epoch [{epoch+1}/{self.num_epochs}] Val Loss: {avg_loss:.4f} Val Acc: {acc:.4f}")
        return avg_loss, acc

    def run(self):
        os.makedirs(self.save_ckp, exist_ok=True)
        
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            _, val_acc = self.validate(epoch)

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_path = os.path.join(self.save_ckp, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"【最佳模型已保存】Val Acc: {val_acc:.4f} @ {save_path}")

            # 定期保存 checkpoint (Reserve ckp handling)
            if (epoch + 1) % self.snapshot_freq == 0:
                ckpt_path = os.path.join(self.save_ckp, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"【Checkpoint Saved】Epoch {epoch+1} @ {ckpt_path}")


        # 保存最终模型
        final_path = os.path.join(self.save_ckp, "last_model.pth")
        torch.save(self.model.state_dict(), final_path)
        print(f"【最终模型已保存】{final_path}")
        print(f"Best Val Acc: {self.best_val_acc}")

    def evaluate(self):
        """仅进行评估"""
        print("Starting Evaluation...")
        loss, acc = self.validate(epoch=0)
        print(f"Final Evaluation Result - Loss: {loss:.4f}, Acc: {acc:.4f}")

