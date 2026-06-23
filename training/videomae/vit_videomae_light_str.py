import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, AutoImageProcessor

# ==================== 配置 ====================
data_root = "data2_str_blur4"
train_path = os.path.join(data_root, "train")
val_path = os.path.join(data_root, "val")
test_path = os.path.join(data_root, "test")

class_names = ['0', '1', '2', '3', '4', '5', '6', '7']
num_classes = len(class_names)

# VideoMAE 关键参数调整
num_clips = 1            # 每个视频采样1个片段
num_frames_per_clip = 64 # 每个片段64帧，再降采样
num_frames = 16          # 最终输入模型的帧数（VideoMAE 固定为16）
frame_height = 224
frame_width = 224

# 训练参数 (针对 8GB 显存优化)
batch_size = 8           # 从2开始尝试
learning_rate = 5e-5     # 常用初始学习率
num_epochs = 3
weight_decay = 0.1       # 增加权重衰减防止过拟合
gradient_accumulation_steps = 1  # 梯度累积步数，等效 batch_size=4
warmup_ratio = 0.1
max_grad_norm = 1.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

save_model_path = 'videomae_best_mv_blur4.pth'

# ==================== 数据集 ====================
class VideoMAEDataset(Dataset):
    def __init__(self, root_dir, class_names, num_frames=16, num_clips=1, num_frames_per_clip=64, processor=None, training=True):
        self.num_frames = num_frames
        self.num_clips = num_clips
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

    # ---------- 新增：光照增强函数 ----------
    def _lighting_augmentation(self, frames):
        """
        对 frames (list of numpy array, shape H,W,C, dtype=uint8) 进行随机的亮度/对比度调整
        返回增强后的相同结构的列表
        """
        # 随机决定是否做增强 (概率0.5)
        if np.random.rand() < 0.5:
            return frames

        # 亮度调整系数 beta: [-30, 30]  对比度调整系数 alpha: [0.7, 1.3]
        alpha = np.random.uniform(0.7, 1.3)   # 对比度
        beta = np.random.randint(-30, 30)     # 亮度

        aug_frames = []
        for frame in frames:
            # convertScaleAbs: dst = saturate(alpha * src + beta)
            augmented = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            aug_frames.append(augmented)
        return aug_frames
    # ---------------------------------------

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
            dummy = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames
            inputs = self.processor(images=dummy, return_tensors="pt")
            pixel_values = inputs.pixel_values.squeeze(0)
            return pixel_values, label

        total_frames = len(frames)
        if total_frames < self.num_frames_per_clip:
            frames = frames + [frames[-1]] * (self.num_frames_per_clip - total_frames)
            total_frames = self.num_frames_per_clip
        
        if self.training:
            start = np.random.randint(0, total_frames - self.num_frames_per_clip + 1)
        else:
            start = (total_frames - self.num_frames_per_clip) // 2
        
        clip = frames[start:start + self.num_frames_per_clip]
        indices = np.linspace(0, self.num_frames_per_clip - 1, self.num_frames, dtype=int)
        sampled_frames = [clip[i] for i in indices]

        # ----- 在线光照增强（仅在训练时）-----
        if self.training:
            sampled_frames = self._lighting_augmentation(sampled_frames)
        # -----------------------------------

        inputs = self.processor(images=sampled_frames, return_tensors="np")
        pixel_values = torch.from_numpy(inputs.pixel_values).squeeze(0)

        return pixel_values, label


# ==================== 模型创建 ====================
def create_videomae_model(num_classes):
    model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model, processor


# ==================== 训练与验证 ====================
def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None, gradient_accumulation_steps=1, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()  # 在epoch开始时清零梯度

    for step, (inputs, labels) in enumerate(tqdm(dataloader, desc='训练')):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(pixel_values=inputs).logits
        loss = criterion(outputs, labels)
        
        # 梯度累积：将损失除以累积步数
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item() * inputs.size(0) * gradient_accumulation_steps  # 还原损失
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
    print("正在加载 VideoMAE 模型...")
    model, processor = create_videomae_model(num_classes)
    model = model.to(device)

    # 2. 创建数据集和数据加载器
    print("正在创建数据集...")
    train_dataset = VideoMAEDataset(train_path, class_names, num_frames=num_frames, num_clips=num_clips, num_frames_per_clip=num_frames_per_clip, processor=processor, training=True)
    val_dataset = VideoMAEDataset(val_path, class_names, num_frames=num_frames, num_clips=num_clips, num_frames_per_clip=num_frames_per_clip, processor=processor, training=False)
    test_dataset = VideoMAEDataset(test_path, class_names, num_frames=num_frames, num_clips=num_clips, num_frames_per_clip=num_frames_per_clip, processor=processor, training=False)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 使用线性学习率调度和预热
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-7, end_factor=1.0, total_iters=warmup_steps)
    # 在优化器step后需要更新调度器，已在train_epoch中实现

    # 4. 训练循环
    best_val_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scheduler, gradient_accumulation_steps, max_grad_norm)
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")

        # 保存最佳模型
        #if val_acc > best_val_acc:
        #    best_val_acc = val_acc
        if epoch == num_epochs:
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
