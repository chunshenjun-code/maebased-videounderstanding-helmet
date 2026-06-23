#!/usr/bin/env python3
"""
VideoMAE 微调脚本 - 前景保留强数据增强
只对背景区域进行强增强，前景（手部+关键物品）保持不变
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
data_root = "data_str_blur4"
mask_root = "data_str_blur4_masks"  # 掩码根目录（由 generate_masks.py 生成）
train_path = os.path.join(data_root, "train")
val_path = os.path.join(data_root, "val")
test_path = os.path.join(data_root, "test")

class_names = ['0', '1', '2', '3', '4', '5', '6', '7']
num_classes = len(class_names)

# 视频参数
num_frames = 16
num_frames_per_clip = 64
frame_height = 224
frame_width = 224

# 训练参数
batch_size = 2
learning_rate = 3e-5
num_epochs = 50
weight_decay = 0.15
gradient_accumulation_steps = 2
warmup_epochs = 5
max_grad_norm = 1.0
label_smoothing = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

save_model_path = 'videomae_base_fg_preserving.pth'
model_name = "MCG-NJU/videomae-base-finetuned-kinetics"

# ==================== 背景强增强定义 ====================
# 注意：这些增强只会应用到背景区域（掩码中为0的区域）
background_aug = A.Compose([
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.9),
    A.GaussianBlur(blur_limit=(5, 15), p=0.6),
    A.GaussNoise(var_limit=(20.0, 80.0), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    A.CoarseDropout(max_holes=4, max_height=64, max_width=64, fill_value=0, p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
])

# 验证/测试时不做增强（可做中心裁剪）
val_transform = A.Compose([
    A.CenterCrop(height=frame_height, width=frame_width, p=1.0),
])


def apply_background_augmentation(frame, mask, aug_pipeline):
    """
    对背景区域应用增强，前景保持不变
    frame: (H,W,3) numpy array (RGB)
    mask: (H,W) numpy array, 前景=255, 背景=0
    """
    # 确保 mask 是二值且与 frame 同尺寸
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    # 先对全图做增强
    aug_frame = aug_pipeline(image=frame)['image']
    # 前景区域保留原图，背景区域使用增强图
    mask_3ch = np.stack([mask, mask, mask], axis=-1) // 255  # 转为 0/1
    result = np.where(mask_3ch, frame, aug_frame)
    return result


# ==================== 数据集类 ====================
class VideoMAEForegroundPreservingDataset(Dataset):
    def __init__(self, root_dir, mask_root, class_names, num_frames=16, num_frames_per_clip=64,
                 processor=None, training=True):
        self.num_frames = num_frames
        self.num_frames_per_clip = num_frames_per_clip
        self.processor = processor
        self.training = training
        self.samples = []  # (video_path, mask_path, label)

        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)
            mask_class_dir = os.path.join(mask_root, os.path.basename(root_dir), class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(class_dir, fname)
                    # 对应的掩码文件
                    mask_path = os.path.join(mask_class_dir, fname.replace('.mp4', '.npz').replace('.avi', '.npz'))
                    if not os.path.exists(mask_path):
                        print(f"警告: 掩码文件不存在 {mask_path}，将不使用背景增强")
                        mask_path = None
                    self.samples.append((video_path, mask_path, label))

    def __len__(self):
        return len(self.samples)

    def load_video_frames(self, video_path):
        """读取视频所有帧，返回 RGB 列表"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        return frames

    def load_masks(self, mask_path, total_frames):
        """加载掩码序列，如果掩码文件不存在则返回全零掩码（即全部视为背景）"""
        if mask_path is None or not os.path.exists(mask_path):
            return [np.zeros((frame_height, frame_width), dtype=np.uint8) for _ in range(total_frames)]
        data = np.load(mask_path)
        masks = data['masks']  # (T, H, W)
        # 如果掩码尺寸与目标尺寸不同，先不做 resize，在应用增强时再调整
        return [masks[i] for i in range(min(len(masks), total_frames))]

    def __getitem__(self, idx):
        video_path, mask_path, label = self.samples[idx]

        frames = self.load_video_frames(video_path)
        total_frames = len(frames)
        if total_frames == 0:
            dummy = [np.zeros((frame_height, frame_width, 3), dtype=np.uint8)] * self.num_frames
            inputs = self.processor(images=dummy, return_tensors="pt")
            return inputs.pixel_values.squeeze(0), label

        # 采样连续片段
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

        clip_frames = frames[start:start + self.num_frames_per_clip]

        # 加载对应的掩码（如果掩码长度不足，重复最后一个）
        masks = self.load_masks(mask_path, total_frames)
        if len(masks) < self.num_frames_per_clip:
            masks = masks + [masks[-1]] * (self.num_frames_per_clip - len(masks))
        clip_masks = masks[start:start + self.num_frames_per_clip]

        # 应用增强（仅训练时）
        if self.training:
            # 对每一帧，根据掩码对背景做增强
            augmented_frames = []
            for frame, mask in zip(clip_frames, clip_masks):
                # 确保掩码尺寸与帧一致
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                aug_frame = apply_background_augmentation(frame, mask, background_aug)
                augmented_frames.append(aug_frame)
            clip_frames = augmented_frames
        else:
            # 验证/测试时仅做中心裁剪
            clip_frames = [val_transform(image=frame)['image'] for frame in clip_frames]

        # 降采样到 num_frames 帧
        indices = np.linspace(0, self.num_frames_per_clip - 1, self.num_frames, dtype=int)
        sampled_frames = [clip_frames[i] for i in indices]

        # 转换为 tensor
        inputs = self.processor(images=sampled_frames, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)
        return pixel_values, label


# ==================== 模型创建、训练、验证函数（与之前相同） ====================
def create_videomae_model(num_classes, model_name):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model, processor


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
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    # 数据集（注意 mask_root 需要与 data_root 对应的划分一致）
    # 训练集掩码路径: mask_root/train/0/video.npz
    train_dataset = VideoMAEForegroundPreservingDataset(
        train_path, mask_root, class_names,
        num_frames=num_frames, num_frames_per_clip=num_frames_per_clip,
        processor=processor, training=True)
    val_dataset = VideoMAEForegroundPreservingDataset(
        val_path, mask_root, class_names,
        num_frames=num_frames, num_frames_per_clip=num_frames_per_clip,
        processor=processor, training=False)
    test_dataset = VideoMAEForegroundPreservingDataset(
        test_path, mask_root, class_names,
        num_frames=num_frames, num_frames_per_clip=num_frames_per_clip,
        processor=processor, training=False)

    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader) // gradient_accumulation_steps,
        pct_start=warmup_epochs / num_epochs,
        anneal_strategy='cos'
    )

    best_val_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler,
            gradient_accumulation_steps, max_grad_norm)
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_model_path)
            print(f"保存最佳模型，验证准确率: {best_val_acc:.4f}")

    # 测试
    print("\n加载最佳模型并在测试集上评估...")
    model.load_state_dict(torch.load(save_model_path, map_location=device))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="测试"):
            inputs = inputs.to(device)
            outputs = model(pixel_values=inputs).logits
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    print("\n测试集分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


if __name__ == "__main__":
    main()
