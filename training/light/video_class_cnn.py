"""
短视频分类 - 方案二：预训练CNN特征 + 时序池化
数据组织同方案一：
    data_root/
        train/0/1/2/ ...
        val/0/1/2/ ...
        test/0/1/2/ ...
"""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm  # 显示进度条

# ==================== 配置 ====================
data_root_train = "data_str_blur4"  # 修改为你的实际路径
data_root_test = "data_str_blur4"
train_path = os.path.join(data_root_train, "train")
val_path = os.path.join(data_root_train, "val")
test_path = os.path.join(data_root_test, "test")
#test_path = "new_test_valid_blur4"

class_names = ['0', '1', '2', '3', '4', '5', '6', '7']  # 类别名称
#class_names = ['0', '1', '2', '3']
#class_names = ['0', '1', '2', '3', '4', '5']
# 特征提取参数
target_frames = 30          # 统一帧数
batch_size = 32             # 特征提取时的batch大小（用于加速，但这里逐帧处理，可忽略）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 时序池化方式: 'mean' 或 'max'
pool_method = 'mean'

# 分类器选择: 'lr' (逻辑回归) 或 'svm'
classifier_type = 'lr'

# ==================== 加载预训练模型 ====================
def load_feature_extractor():
    """加载预训练ResNet50，去掉最后的全连接层，返回特征提取器"""
    model = models.resnet50(pretrained=True)
    # 移除全连接层 (avgpool + fc)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # 输出形状: (batch, 2048, 1, 1)
    model = model.to(device)
    model.eval()
    return model

# 图像预处理：与ImageNet训练一致
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== 单视频特征提取 ====================
def extract_cnn_features(video_path, model, target_frames=30):
    """
    提取视频每帧的2048维特征
    返回: numpy数组 shape (target_frames, 2048)
    """
    cap = cv2.VideoCapture(video_path)
    frames_rgb = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(frame_rgb)
    cap.release()

    if len(frames_rgb) == 0:
        print(f"警告: 无法读取视频 {video_path}")
        return None

    # 统一帧数
    n_frames = len(frames_rgb)
    if n_frames < target_frames:
        # 重复最后一帧补齐
        frames_rgb = frames_rgb + [frames_rgb[-1]] * (target_frames - n_frames)
    else:
        # 均匀采样target_frames帧
        indices = np.linspace(0, n_frames-1, target_frames, dtype=int)
        frames_rgb = [frames_rgb[i] for i in indices]

    # 提取特征
    feats = []
    with torch.no_grad():
        for frame in frames_rgb:
            # 预处理
            pil_img = Image.fromarray(frame)
            input_tensor = preprocess(pil_img).unsqueeze(0).to(device)  # (1,3,224,224)
            feat = model(input_tensor).cpu().numpy().flatten()  # (2048,)
            feats.append(feat)
    return np.array(feats)  # (target_frames, 2048)

# ==================== 时序池化 ====================
def temporal_pooling(frame_features, method='mean'):
    """
    将帧特征聚合成视频级特征
    frame_features: (T, D)
    method: 'mean' 或 'max'
    返回: (D,)
    """
    if method == 'mean':
        return np.mean(frame_features, axis=0)
    elif method == 'max':
        return np.max(frame_features, axis=0)
    else:
        raise ValueError("method must be 'mean' or 'max'")

# ==================== 数据集特征提取（带缓存） ====================
def extract_and_cache_features(data_path, model, cache_dir='cache', pool_method='mean'):
    """
    遍历data_path下的所有视频，提取特征并保存为.npy文件到cache_dir
    同时返回特征数组和标签
    """
    X, y = [], []
    os.makedirs(cache_dir, exist_ok=True)

    for label_name in class_names:
        class_dir = os.path.join(data_path, label_name)
        if not os.path.isdir(class_dir):
            continue
        # 创建对应类别的缓存子文件夹
        cache_class_dir = os.path.join(cache_dir, label_name)
        os.makedirs(cache_class_dir, exist_ok=True)

        video_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        for fname in tqdm(video_files, desc=f"处理 {label_name}"):
            video_path = os.path.join(class_dir, fname)
            cache_path = os.path.join(cache_class_dir, fname[:-4] + '.npy')

            # 如果缓存已存在，直接加载
            if os.path.exists(cache_path):
                vec = np.load(cache_path)
            else:
                # 提取帧特征并池化
                frame_feats = extract_cnn_features(video_path, model, target_frames)
                if frame_feats is None:
                    continue
                vec = temporal_pooling(frame_feats, method=pool_method)
                # 保存缓存
                np.save(cache_path, vec)

            X.append(vec)
            y.append(int(label_name))

    return np.array(X), np.array(y)

# ==================== 主程序 ====================
def main():
    # 加载特征提取器
    model = load_feature_extractor()

    # 提取特征（如果已缓存则直接加载）
    print("提取训练集特征...")
    X_train, y_train = extract_and_cache_features(train_path, model, cache_dir='cache_train', pool_method=pool_method)
    print(f"训练集样本数: {len(X_train)}")

    print("提取验证集特征...")
    X_val, y_val = extract_and_cache_features(val_path, model, cache_dir='cache_val', pool_method=pool_method)
    print(f"验证集样本数: {len(X_val)}")

    print("提取测试集特征...")
    X_test, y_test = extract_and_cache_features(test_path, model, cache_dir='cache_test', pool_method=pool_method)
    print(f"测试集样本数: {len(X_test)}")

    # 特征标准化（分类器通常需要）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 选择分类器
    if classifier_type == 'lr':
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif classifier_type == 'svm':
        clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    else:
        raise ValueError("classifier_type must be 'lr' or 'svm'")

    print(f"训练分类器: {classifier_type}")
    clf.fit(X_train, y_train)

    # 验证集评估
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"验证集准确率: {val_acc:.4f}")
    print("验证集分类报告:")
    print(classification_report(y_val, y_val_pred, target_names=class_names))

    # 测试集评估
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"\n测试集准确率: {test_acc:.4f}")
    print("测试集分类报告:")
    print(classification_report(y_test, y_test_pred, target_names=class_names))

    # 保存模型和标准化器
    joblib.dump(clf, f'cnn_{classifier_type}_model.pkl')
    joblib.dump(scaler, 'cnn_scaler.pkl')
    print(f"\n模型已保存为 cnn_{classifier_type}_model.pkl 和 cnn_scaler.pkl")

if __name__ == "__main__":
    main()
