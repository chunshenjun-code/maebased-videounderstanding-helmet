"""
视频分类推理脚本 - 使用微调的 VideoMAE 模型
输入：单个MP4视频文件
输出：预测类别和置信度（可选打印前k个概率）
API 与 video_3dcnn_classify.py 完全一致
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
import argparse

# ==================== 配置 ====================
# 类别名称（需与训练时一致）
class_names = ['0', '1', '2', '3', '4', '5', '6', '7']
model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
num_classes = len(class_names)

# 视频处理参数
num_frames = 16                   # 与训练时一致
frame_height = 224
frame_width = 224

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理（与 R(2+1)D 保持一致，也可使用 VideoMAEImageProcessor 但此处统一）
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],   # Kinetics-400 均值
                         std=[0.22803, 0.22145, 0.216989])
])

# ==================== 模型定义 ====================
def load_model(model_path, num_classes, device):
    """
    加载 HuggingFace 格式的 VideoMAE 模型权重。
    参考 confuse_mtrx_mae.py 中的成功加载方式。
    """
    # 1. 创建与训练时结构相同的模型骨架
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True   # 允许分类头尺寸不同
    )
    
    # 2. 加载微调后的权重（直接加载，无需键名映射）
    state_dict = torch.load(model_path, map_location=device)
    
    # 3. 处理可能的多卡训练保存的权重（去除 'module.' 前缀）
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    # 4. 严格加载（确保权重与模型结构完全匹配）
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model
# ==================== 视频预处理 ====================
def preprocess_video(video_path, num_frames=16, transform=None):
    """
    读取视频，采样固定帧数，进行预处理，返回模型输入张量 (1, C, T, H, W)
    """
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
        raise ValueError(f"无法读取视频: {video_path}")

    # 均匀采样 num_frames 帧
    total_frames = len(frames)
    if total_frames < num_frames:
        frames = frames + [frames[-1]] * (num_frames - total_frames)
        indices = list(range(num_frames))
    else:
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    sampled_frames = [frames[i] for i in indices]

    # 应用预处理
    if transform:
        frames_tensor = torch.stack([transform(frame) for frame in sampled_frames], dim=1)  # (C, T, H, W)
    else:
        to_tensor = transforms.ToTensor()
        frames_tensor = torch.stack([to_tensor(frame) for frame in sampled_frames], dim=1)

    # 增加batch维度
    input_tensor = frames_tensor.unsqueeze(0)  # (1, C, T, H, W)
    return input_tensor

# ==================== 推理 ====================
def predict_video(model, video_path, class_names, top_k=1):
    """
    对视频进行分类预测
    返回: 预测类别索引, 置信度, 以及前top_k个 (类别名, 概率)
    """
    input_tensor = preprocess_video(video_path, num_frames, transform)  # (1, C, T, H, W)
    # VideoMAE 期望输入形状为 (batch, T, C, H, W)，因此需要转换维度
    input_tensor = input_tensor.permute(0, 2, 1, 3, 4).contiguous()   # (1, T, C, H, W)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)  # 返回 VideoMAEOutput 对象
        logits = outputs.logits                     # (1, num_classes)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy().flatten()

    pred_idx = np.argmax(probabilities)
    confidence = probabilities[pred_idx]

    # 获取前 top_k 个
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    top_results = [(class_names[i], probabilities[i]) for i in top_indices]

    return pred_idx, confidence, top_results

# ==================== 主程序（与原始脚本兼容） ====================
def main():
    parser = argparse.ArgumentParser(description="使用微调的 VideoMAE 进行视频分类")
    parser.add_argument('video_path', type=str, help='输入视频文件路径')
    parser.add_argument('--model', type=str, default='videomae_best_blur.pt', help='训练好的模型权重文件路径')
    parser.add_argument('--top_k', type=int, default=3, help='输出前k个最可能的类别')
    parser.add_argument('--class_names', type=str, nargs='+', default=class_names, help='类别名称列表，默认使用0-7')
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        print(f"错误: 视频文件 {args.video_path} 不存在")
        return

    if not os.path.isfile(args.model):
        print(f"错误: 模型文件 {args.model} 不存在")
        return

    print(f"使用设备: {device}")
    print(f"加载模型: {args.model}")
    model = load_model(args.model, len(args.class_names), device)

    print(f"正在处理视频: {args.video_path}")
    try:
        pred_idx, confidence, top_results = predict_video(model, args.video_path, args.class_names, top_k=args.top_k)
    except Exception as e:
        print(f"推理失败: {e}")
        return

    print(f"\n预测结果:")
    print(f"  主类别: {args.class_names[pred_idx]} (置信度: {confidence:.4f})")
    if args.top_k > 1:
        print(f"  前{args.top_k}个可能类别:")
        for i, (cls_name, prob) in enumerate(top_results):
            print(f"    {i+1}. {cls_name}: {prob:.4f}")

if __name__ == "__main__":
    main()
