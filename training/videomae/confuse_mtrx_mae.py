"""
VideoMAE 模型测试集混淆矩阵生成
使用训练好的 videomae_base_best.pth 对 test 文件夹中的视频进行分类，
并输出混淆矩阵图。
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, AutoImageProcessor

# ==================== 配置（必须与微调时保持一致） ====================
# 测试数据根目录（请根据你的实际路径修改）
test_path = "new_test_valid_blur4"          # 或者 "data_str_blur4/test"
#test_path = "data_str_blur4/test"
class_names = ['0', '1', '2', '3', '4', '5', '6', '7']
#class_names = ['0', '1', '2', '3', '4', '5']
#class_names = ['0', '1', '2', '3']
num_classes = len(class_names)

# VideoMAE 参数（必须与训练时一致）
num_frames = 16                     # 最终输入模型的帧数
num_frames_per_clip = 64            # 采样连续片段的帧数（训练时用的64）
frame_height = 224
frame_width = 224

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 模型权重文件路径（根据实际保存的路径修改）
#model_weights_path = 'videomae_sequential_finetuned.pth'
#model_weights_path = 'videomae_best_blur4.pth'
model_weights_path = 'videomae_best_blur4.pth'
# 预训练模型名称（必须与微调时使用的 checkpoint 一致）
model_name = "MCG-NJU/videomae-base-finetuned-kinetics"

# ==================== 自定义数据集（适配 VideoMAE 测试模式） ====================
class VideoMAETestDataset(Dataset):
    def __init__(self, root_dir, class_names, num_frames=16, num_frames_per_clip=64, processor=None):
        """
        root_dir: 包含类别子文件夹的路径 (如 test/)
        class_names: 类别名称列表
        num_frames: 最终输入模型的帧数（VideoMAE 固定为16）
        num_frames_per_clip: 从视频中采样连续片段的帧数（训练时用64，测试时保持一致）
        processor: VideoMAE 的图像处理器
        """
        self.num_frames = num_frames
        self.num_frames_per_clip = num_frames_per_clip
        self.processor = processor
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

        # 测试时：从视频中间取一段连续片段（与训练时验证/测试模式一致）
        if total_frames < self.num_frames_per_clip:
            # 如果视频长度不足，重复最后一帧补齐
            frames = frames + [frames[-1]] * (self.num_frames_per_clip - total_frames)
            total_frames = self.num_frames_per_clip

        start = (total_frames - self.num_frames_per_clip) // 2
        clip = frames[start:start + self.num_frames_per_clip]

        # 从连续片段中均匀降采样出 num_frames 帧
        indices = np.linspace(0, self.num_frames_per_clip - 1, self.num_frames, dtype=int)
        sampled_frames = [clip[i] for i in indices]

        # 使用 processor 处理视频帧序列（返回 PyTorch 张量）
        inputs = self.processor(images=sampled_frames, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)  # (num_frames, 3, H, W)

        return pixel_values, label


# ==================== 模型创建（与微调时结构一致） ====================
def create_videomae_model(num_classes, model_name):
    """创建与微调时结构相同的模型，并加载训练好的权重"""
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model, processor


# ==================== 主程序 ====================
def main():
    # 1. 创建模型和处理器
    print("正在加载模型结构...")
    model, processor = create_videomae_model(num_classes, model_name)

    # 2. 加载训练好的权重
    print(f"加载权重文件: {model_weights_path}")
    state_dict = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("模型加载完成")

    # 3. 创建测试集数据加载器
    print("正在创建测试集...")
    test_dataset = VideoMAETestDataset(
        test_path, class_names,
        num_frames=num_frames,
        num_frames_per_clip=num_frames_per_clip,
        processor=processor
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    print(f"测试集样本数: {len(test_dataset)}")

    # 4. 预测并收集所有标签和预测结果
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="预测测试集"):
            inputs = inputs.to(device)
            outputs = model(pixel_values=inputs).logits
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 5. 计算评估指标
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n测试集准确率: {acc:.4f}")
    print("分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 6. 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix (VideoMAE Model)\nTest Accuracy = {acc:.4f}')
    plt.tight_layout()
    plt.savefig('videomae_confusion_matrix.png', dpi=300)
    plt.show()
    print("混淆矩阵已保存为 videomae_confusion_matrix.png")


if __name__ == "__main__":
    main()
