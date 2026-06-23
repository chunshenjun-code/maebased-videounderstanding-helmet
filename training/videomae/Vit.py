"""
TimeSformer 微调脚本 - 短视频分类 (8类)
数据组织与原有代码完全一致：
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
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoImageProcessor, TimesformerForVideoClassification

# ==================== 配置 ====================
data_root = "data_str_blur4"
train_path = os.path.join(data_root, "train")
val_path = os.path.join(data_root, "val")
test_path = os.path.join(data_root, "test")

class_names = ['0', '1', '2', '3', '4', '5', '6', '7']
num_classes = len(class_names)

# 视频处理参数（TimeSformer 推荐 8 帧）
num_frames = 16
frame_height = 224
frame_width = 224

# 训练参数
batch_size = 4                       # 根据 GPU 内存调整
learning_rate = 5e-5
num_epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

save_model_path = 'timesformer_base_best.pth'

# ==================== 数据集 ====================
class TimeSformerDataset(Dataset):
    def __init__(self, root_dir, class_names, num_frames=8, processor=None):
        self.num_frames = num_frames
        self.processor = processor
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
            # 返回一个全零的列表，processor 将无法处理，在后续会报错
            # 更好的处理是在数据预处理阶段筛除这样的视频
            dummy = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames
            inputs = self.processor(images=dummy, return_tensors="pt")
            pixel_values = inputs.pixel_values.squeeze(0)
            return pixel_values, label

        total_frames = len(frames)
        if total_frames < self.num_frames:
            frames = frames + [frames[-1]] * (self.num_frames - total_frames)
            indices = list(range(self.num_frames))
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        sampled_frames = [frames[i] for i in indices]

        # 使用 processor 处理视频帧序列
        # processor 期望接收一个 images 列表，列表中每个元素是单帧的 RGB 图像 (H, W, C) 或 PIL Image
        # return_tensors="pt" 将返回 PyTorch 张量
        inputs = self.processor(images=sampled_frames, return_tensors="pt")
        # pixel_values 的形状为 (1, num_frames, 3, H, W)，去掉 batch 维度
        pixel_values = inputs.pixel_values.squeeze(0)

        return pixel_values, label


# ==================== 模型创建 ====================
def create_timesformer_model(num_classes):
    # 加载 TimeSformer 的图像处理器和预训练模型
    # 使用在 Kinetics-400 上预训练的 base 版本，这是一个很好的起点
    model_name = "facebook/timesformer-base-finetuned-k400"
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = TimesformerForVideoClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,   # 因为我们只有8类，与原始的400类不匹配
    )
    return model, processor


# ==================== 训练与验证 ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc='训练'):
        inputs, labels = inputs.to(device), labels.to(device)

        # TimeSformer 模型的前向传播输出 logits
        outputs = model(pixel_values=inputs).logits
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
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
    # 1. 创建模型和 processor
    print("正在加载 TimeSformer 模型...")
    model, processor = create_timesformer_model(num_classes)
    model = model.to(device)

    # 2. 创建数据集和数据加载器
    print("正在创建数据集...")
    train_dataset = TimeSformerDataset(train_path, class_names, num_frames=num_frames, processor=processor)
    val_dataset = TimeSformerDataset(val_path, class_names, num_frames=num_frames, processor=processor)
    test_dataset = TimeSformerDataset(test_path, class_names, num_frames=num_frames, processor=processor)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 使用 AdamW 优化器，并设置合适的权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    # 学习率调度器，当验证损失不再下降时，降低学习率
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
        if val_acc > best_val_acc:
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
            outputs = model(pixel_values=inputs).logits
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n测试集分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


if __name__ == "__main__":
    main()
