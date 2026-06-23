"""
R(2+1)D 模型测试集混淆矩阵生成
使用训练好的 r2plus1d_18_best.pth 对 test 文件夹中的视频进行分类，
并输出混淆矩阵图。
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==================== 配置（与训练时保持一致） ====================
data_root = "data_str"               # 数据集根目录（与训练时相同）
#test_path = os.path.join(data_root, "test")
test_path = "new_test_valid_blur4"

#class_names = ['0', '1', '2', '3']
class_names = ['0', '1', '2', '3', '4', '5', '6', '7']
#class_names = ['0', '1', '2', '3', '4', '5']
num_classes = len(class_names)

# 视频处理参数（必须与训练时一致）
num_frames = 16
frame_height = 224
frame_width = 224

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 模型权重文件路径
model_weights_path = '3dcnn_data_str_blur4_16.pth'

# 验证预处理（与训练时的val_transform相同）
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989])
])

# ==================== 自定义数据集（与训练时一致） ====================
class VideoDataset(Dataset):
    def __init__(self, root_dir, class_names, num_frames=16, transform=None):
        """
        root_dir: 包含类别子文件夹的路径 (如 test/)
        class_names: 类别名称列表
        num_frames: 每个视频采样的帧数
        transform: 图像预处理函数
        """
        self.num_frames = num_frames
        self.transform = transform
        self.samples = []  # 每个元素为 (video_path, label)

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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        if len(frames) == 0:
            print(f"警告: 无法读取视频 {video_path}，返回空张量")
            clip = torch.zeros((3, self.num_frames, 224, 224))
            return clip, label

        # 均匀采样 num_frames 帧
        total_frames = len(frames)
        if total_frames < self.num_frames:
            frames = frames + [frames[-1]] * (self.num_frames - total_frames)
            indices = list(range(self.num_frames))
        else:
            indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
        sampled_frames = [frames[i] for i in indices]

        # 应用预处理
        if self.transform:
            frames_tensor = torch.stack([self.transform(frame) for frame in sampled_frames], dim=1)
        else:
            to_tensor = transforms.ToTensor()
            frames_tensor = torch.stack([to_tensor(frame) for frame in sampled_frames], dim=1)

        return frames_tensor, label

# ==================== 模型定义（与训练时一致） ====================
def create_model(num_classes):
    """创建与训练时结构相同的模型，并加载训练好的权重"""
    # 加载预训练结构（不使用权重，仅用结构）
    weights = R2Plus1D_18_Weights.KINETICS400_V1
    model = r2plus1d_18(weights=weights)
    # 替换最后的全连接层
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# ==================== 主程序 ====================
def main():
    # 1. 创建测试集数据加载器
    print("正在创建测试集...")
    test_dataset = VideoDataset(test_path, class_names, num_frames=num_frames, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    print(f"测试集样本数: {len(test_dataset)}")

    # 2. 加载模型
    print("加载模型...")
    model = create_model(num_classes)
    # 加载训练好的权重
    state_dict = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("模型加载完成")

    # 3. 预测并收集所有标签和预测结果
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="预测测试集"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. 计算评估指标
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n测试集准确率: {acc:.4f}")
    print("分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 5. 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix (R(2+1)D Model)\nTest Accuracy = {acc:.4f}')
    plt.tight_layout()
    plt.savefig('r2plus1d_confusion_matrix.png', dpi=300)
    plt.show()
    print("混淆矩阵已保存为 r2plus1d_confusion_matrix.png")

if __name__ == "__main__":
    main()
