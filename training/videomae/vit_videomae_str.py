#!/usr/bin/env python3
"""
VideoMAE 微调脚本 - 短视频分类 (8类)
支持在线强数据增强（空间+时序），适用于模糊后的手部-物体交互视频。
数据组织：
    data_root/
        train/0/1/2/.../7/
        val/0/1/2/.../7/
        test/0/1/2/.../7/
每个子文件夹下存放对应类别的 MP4 文件。
"""

import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, AutoImageProcessor
import albumentations as A

# ==================== 配置参数 ====================
data_root = "data_str_blur4"            # 数据集根目录
train_path = os.path.join(data_root, "train")
val_path = os.path.join(data_root, "val")
test_path = os.path.join(data_root, "test")

class_names = ['0', '1', '2', '3', '4', '5', '6', '7']
num_classes = len(class_names)

# 视频处理参数（VideoMAE 固定使用 16 帧输入）
num_frames = 16                     # 最终输入帧数
num_frames_per_clip = 64            # 从原视频采样的连续片段长度（密集采样）
frame_height = 224
frame_width = 224

# 训练超参数（针对 8GB 显存优化）
batch_size = 8
learning_rate = 3e-5                # 配合强增强适当降低
num_epochs = 10                     # 增强后需要更多 epoch
weight_decay = 0.15
gradient_accumulation_steps = 1     # 有效 batch_size = 4
warmup_epochs = 5
max_grad_norm = 1.0
label_smoothing = 0.1               # 标签平滑正则化

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

save_model_path = 'videomae_best_str.pth'

# 模型选择：'base' 或 'small'
model_size = 'base'                 # 可选 'base' 或 'small'
if model_size == 'base':
    model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
else:
    model_name = "MCG-NJU/videomae-small-finetuned-kinetics"

# ==================== 在线数据增强定义 ====================
# 空间增强（逐帧应用，破坏环境一致性）
train_frame_transform = A.Compose([
    A.RandomResizedCrop(height=frame_height, width=frame_width, scale=(0.7, 1.0), p=0.8),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.ToGray(p=0.2),
    A.CoarseDropout(max_holes=1, max_height=32, max_width=32, fill_value=0, p=0.3),
])

# 验证/测试时仅中心裁剪（保持分辨率一致）
val_frame_transform = A.Compose([
    A.CenterCrop(height=frame_height, width=frame_width, p=1.0),
])

def temporal_shuffle(frames, shuffle_ratio=0.1):
    """随机交换少量相邻帧，增加时序鲁棒性"""
    if random.random() > 0.3:
        return frames
    num_swaps = max(1, int(len(frames) * shuffle_ratio))
    frames = frames.copy()
    for _ in range(num_swaps):
        i = random.randint(0, len(frames) - 2)
        frames[i], frames[i+1] = frames[i+1], frames[i]
    return frames

def temporal_speed_change(frames, target_len, speed_range=(0.9, 1.1)):
    """随机改变视频播放速度（通过帧插值/抽取）"""
    if random.random() < 0.3:
        speed = random.uniform(*speed_range)
        new_len = int(len(frames) / speed)
        if new_len < 2:
            return frames
        indices = np.linspace(0, len(frames)-1, new_len, dtype=int)
        frames = [frames[i] for i in indices]
    # 确保输出长度固定
    if len(frames) < target_len:
        frames = frames + [frames[-1]] * (target_len - len(frames))
    elif len(frames) > target_len:
        frames = frames[:target_len]
    return frames

def temporal_crop_resize(frames, target_len):
    """随机裁剪一段连续子序列，再线性插值回目标长度"""
    if len(frames) <= target_len:
        return frames
    start = random.randint(0, len(frames) - target_len)
    sub = frames[start:start+target_len]
    indices = np.linspace(0, len(sub)-1, target_len, dtype=int)
    return [sub[i] for i in indices]

# ==================== 数据集类 ====================
class VideoMAEDataset(Dataset):
    def __init__(self, root_dir, class_names, num_frames=16, num_frames_per_clip=64,
                 processor=None, training=True):
        self.num_frames = num_frames
        self.num_frames_per_clip = num_frames_per_clip
        self.processor = processor
        self.training = training
        self.samples = []

        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(class_dir, fname)
                    self.samples.append((video_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        # 使用 OpenCV 读取视频
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        if len(frames) == 0:
            print(f"警告: 无法读取视频 {video_path}，返回空张量")
            dummy_frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames
            inputs = self.processor(images=dummy_frames, return_tensors="pt")
            pixel_values = inputs.pixel_values.squeeze(0)
            return pixel_values, label

        total_frames = len(frames)
        # 训练时随机起始点，验证/测试时取中间片段
        if self.training:
            if total_frames < self.num_frames_per_clip:
                frames = frames + [frames[-1]] * (self.num_frames_per_clip - total_frames)
                total_frames = self.num_frames_per_clip
            start = random.randint(0, total_frames - self.num_frames_per_clip)
        else:
            if total_frames < self.num_frames_per_clip:
                frames = frames + [frames[-1]] * (self.num_frames_per_clip - total_frames)
                total_frames = self.num_frames_per_clip
            start = (total_frames - self.num_frames_per_clip) // 2

        clip = frames[start:start + self.num_frames_per_clip]

        # 应用数据增强（仅训练时）
        if self.training:
            # 时序增强
            clip = temporal_shuffle(clip, shuffle_ratio=0.1)
            clip = temporal_speed_change(clip, self.num_frames_per_clip, speed_range=(0.9, 1.1))
            clip = temporal_crop_resize(clip, self.num_frames_per_clip)
            # 空间增强（逐帧）
            clip = [train_frame_transform(image=frame)['image'] for frame in clip]
        else:
            # 验证/测试时中心裁剪
            clip = [val_frame_transform(image=frame)['image'] for frame in clip]

        # 降采样到 num_frames 帧
        indices = np.linspace(0, self.num_frames_per_clip - 1, self.num_frames, dtype=int)
        sampled_frames = [clip[i] for i in indices]

        # 使用 processor 转换为 tensor
        inputs = self.processor(images=sampled_frames, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)   # (num_frames, 3, H, W)
        return pixel_values, label

# ==================== 模型创建 ====================
def create_videomae_model(num_classes, model_name):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    # 可选的 dropout 增强（如果需要，可以在 config 中设置）
    # 这里保持默认，因为 VideoMAE 本身有较强的正则化
    return model, processor

# ==================== 训练与验证函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None,
                gradient_accumulation_steps=1, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for step, (inputs, labels) in enumerate(tqdm(dataloader, desc='训练')):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(pixel_values=inputs).logits
        loss = criterion(outputs, labels)

        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item() * inputs.size(0) * gradient_accumulation_steps
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='验证'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(pixel_values=inputs).logits
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# ==================== 主程序 ====================
def main():
    print("正在加载 VideoMAE 模型...")
    model, processor = create_videomae_model(num_classes, model_name)
    model = model.to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}, 可训练参数量: {trainable_params:,}")

    # 创建数据集
    print("正在创建数据集...")
    train_dataset = VideoMAEDataset(train_path, class_names,
                                    num_frames=num_frames,
                                    num_frames_per_clip=num_frames_per_clip,
                                    processor=processor,
                                    training=True)
    val_dataset = VideoMAEDataset(val_path, class_names,
                                  num_frames=num_frames,
                                  num_frames_per_clip=num_frames_per_clip,
                                  processor=processor,
                                  training=False)
    test_dataset = VideoMAEDataset(test_path, class_names,
                                   num_frames=num_frames,
                                   num_frames_per_clip=num_frames_per_clip,
                                   processor=processor,
                                   training=False)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # 损失函数（带标签平滑）
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 学习率调度：余弦退火 + 线性预热
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    warmup_steps = warmup_epochs * len(train_loader) // gradient_accumulation_steps
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-7, end_factor=1.0,
                                            total_iters=warmup_steps)
    # 注意：这里使用 LinearLR 进行预热，之后需要切换到 CosineAnnealingLR
    # 更简单的方式：使用 OneCycleLR，但为了清晰，我们保持简单
    # 实际上可以在 warmup 后手动更换 scheduler，或者直接用 CosineAnnealingWarmRestarts
    # 为简化，这里使用 CosineAnnealingLR 从头开始，并设置 eta_min
    # 但为了支持预热，建议使用自定义调度。这里提供一个简单有效的方案：
    # 直接使用 CosineAnnealingLR，并通过 warmup 阶段手动调整学习率过于复杂，
    # 我们改用 torch.optim.lr_scheduler.OneCycleLR，它内置了预热和余弦退火。
    # 为了方便，重新定义 scheduler:
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader) // gradient_accumulation_steps,
        pct_start=warmup_epochs / num_epochs,
        anneal_strategy='cos'
    )

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler,
            gradient_accumulation_steps, max_grad_norm
        )
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_model_path)
            print(f"保存最佳模型，验证准确率: {best_val_acc:.4f}")

    # 测试最佳模型
    print("\n加载最佳模型并在测试集上评估...")
    model.load_state_dict(torch.load(save_model_path, map_location=device))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")

    # 详细分类报告
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="测试集预测"):
            inputs = inputs.to(device)
            outputs = model(pixel_values=inputs).logits
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n测试集分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    main()
