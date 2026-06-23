"""
2+1D CNN 微调脚本 - 短视频分类 (8类)
数据组织:
    data_root/
        train/0/1/2/ ... /7/
        val/0/1/2/ ... /7/
        test/0/1/2/ ... /7/
每个子文件夹下存放对应类别的MP4文件。
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import random

# ==================== 配置 ====================
data_root = "data_str_blur4"               # 数据集根目录
train_path = os.path.join(data_root, "train")
val_path = os.path.join(data_root, "val")
test_path = os.path.join(data_root, "test")

class_names = ['0', '1', '2', '3', '4', '5', '6', '7']
#class_names = ['0', '1', '2', '3','4','5']
#class_names = ['0', '1', '2', '3']
num_classes = len(class_names)

# 视频处理参数
num_frames = 16                   # 每个视频采样的帧数
frame_height = 224
frame_width = 224
sample_rate = 1                   # 采样间隔（目前使用均匀采样，此参数暂未用）

# 训练参数
batch_size = 8                    # 根据GPU内存调整
learning_rate = 0.0001
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

save_model_path = '3dcnn_data_str_blur4_16.pth'

# 数据增强与预处理
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],   # Kinetics-400 数据集均值
                         std=[0.22803, 0.22145, 0.216989])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989])
])

# ==================== 自定义数据集 ====================
class VideoDataset(Dataset):
    def __init__(self, root_dir, class_names, num_frames=16, transform=None):
        """
        root_dir: 包含类别子文件夹的路径 (如 train/ 或 val/)
        class_names: 类别名称列表，用于将文件夹名映射到标签
        num_frames: 每个视频采样的帧数
        transform: 图像预处理函数
        """
        self.num_frames = num_frames
        self.transform = transform
        self.samples = []  # 每个元素为 (video_path, label)

        # 遍历每个类别文件夹
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

        # 使用OpenCV读取视频
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        if len(frames) == 0:
            # 如果视频无法读取，返回一个全零的张量（或者跳过）
            print(f"警告: 无法读取视频 {video_path}，返回空张量")
            clip = torch.zeros((3, self.num_frames, 224, 224))
            return clip, label

        # 统一帧数：均匀采样 num_frames 帧
        total_frames = len(frames)
        if total_frames < self.num_frames:
            # 重复最后一帧补齐
            frames = frames + [frames[-1]] * (self.num_frames - total_frames)
            indices = list(range(self.num_frames))
        else:
            # 均匀采样
            indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
        sampled_frames = [frames[i] for i in indices]

        # 应用预处理到每一帧
        if self.transform:
            frames_tensor = torch.stack([self.transform(frame) for frame in sampled_frames], dim=1)  # (C, T, H, W)
        else:
            # 如果没有transform，至少转为tensor并归一化
            to_tensor = transforms.ToTensor()
            frames_tensor = torch.stack([to_tensor(frame) for frame in sampled_frames], dim=1)

        return frames_tensor, label

# ==================== 模型定义 ====================
def create_model(num_classes):
    # 加载预训练的r2plus1d_18，使用最新推荐方式
    weights = R2Plus1D_18_Weights.KINETICS400_V1
    model = r2plus1d_18(weights=weights)
    # 替换最后的全连接层
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# ==================== 训练函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc='训练'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
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
            outputs = model(inputs)
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
    # 1. 创建数据集和数据加载器
    print("正在创建数据集...")
    train_dataset = VideoDataset(train_path, class_names, num_frames=num_frames, transform=train_transform)
    val_dataset = VideoDataset(val_path, class_names, num_frames=num_frames, transform=val_transform)
    test_dataset = VideoDataset(test_path, class_names, num_frames=num_frames, transform=val_transform)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. 创建模型
    model = create_model(num_classes)
    model = model.to(device)

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 4. 训练循环
    best_val_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")

        scheduler.step(val_loss)

        # 保存最佳模型
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_model_path)
            print(f"保存最佳模型，验证准确率: {best_val_acc:.4f}")

    # 5. 在测试集上评估最佳模型
    print("\n加载最佳模型并在测试集上评估...")
    model.load_state_dict(torch.load(save_model_path, map_location=device))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")

    # 详细分类报告
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n测试集分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    main()
