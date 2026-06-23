import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from transformers import VideoMAEForVideoClassification, AutoImageProcessor

# 假设你的原始数据集类已经定义好（即您之前提供的 VideoMAEDataset）
# 如果这个文件与你的数据集类定义不在同一文件，请导入
from vit_videomae_light_str import VideoMAEDataset

# ==================== 配置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'videomae_best_mv_blur4.pth'   # 您训练好的模型权重
data_root = "data2_str_blur4"
val_path = os.path.join(data_root, "val")   # 也可以使用 train 或 test，建议用 val
class_names = ['0', '1', '2', '3', '4', '5', '6', '7']
num_classes = len(class_names)

# 特征提取参数
batch_size = 8
num_frames = 16
num_frames_per_clip = 64
num_clips = 1

# t-SNE 参数
n_samples_per_class = 50   # 每个类别最多取多少个样本（设为 None 表示全部使用）
random_state = 42
perplexity = 30
n_components = 2

# ==================== 根据文件名数字判断视角 ====================
def get_view_label(video_path):
    """
    根据文件名中的数字判断视角：
    数字 < 129 -> first_person
    数字 > 129 -> top_down
    """
    filename = os.path.basename(video_path)          # 例如 "92_aug0.mp4"
    basename = os.path.splitext(filename)[0]        # "92_aug0"
    num_str = basename.split('_')[0]                # "92"
    try:
        num = int(num_str)
    except ValueError:
        return 'unknown'
    if num < 129:
        return 'first_person'
    else:
        return 'top_down'

# ==================== 修改 Dataset 以返回视频路径 ====================
class FeatureDataset(VideoMAEDataset):
    """
    继承原来的 VideoMAEDataset，重写 __getitem__ 使其额外返回视频路径
    """
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        # 父类返回 (pixel_values, label)，但我们不需要父类的 label
        pixel_values, _ = super().__getitem__(idx)
        return pixel_values, label, video_path

# ==================== 加载模型（不注册 hook） ====================
def load_model(model_path, num_classes, device):
    model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, processor

# ==================== 主程序 ====================
def main():
    # 1. 创建数据集（验证集）
    print("Loading dataset...")
    _, processor = load_model(model_path, num_classes, device)  # 暂时只拿 processor
    full_dataset = FeatureDataset(
        root_dir=val_path,
        class_names=class_names,
        num_frames=num_frames,
        num_clips=num_clips,
        num_frames_per_clip=num_frames_per_clip,
        processor=processor,
        training=False      # 必须为 False，避免数据增强
    )

    # 可选：限制每个类别的样本数，避免 t-SNE 计算过慢
    if n_samples_per_class is not None:
        indices = []
        for c in range(num_classes):
            c_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label == c]
            if len(c_indices) > n_samples_per_class:
                c_indices = np.random.choice(c_indices, n_samples_per_class, replace=False)
            indices.extend(c_indices)
        dataset = Subset(full_dataset, indices)
        print(f"限制了每个类别最多 {n_samples_per_class} 个样本，总样本数: {len(dataset)}")
    else:
        dataset = full_dataset
        print(f"使用全部样本，总样本数: {len(dataset)}")

    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 2. 加载模型
    print("Loading model...")
    model, processor = load_model(model_path, num_classes, device)

    # 3. 提取特征（直接使用 model.videomae 获取 encoder 输出）
    print("Extracting features...")
    all_features = []   # 每个元素是 (768,) 的 numpy 数组
    all_labels = []
    all_views = []

    with torch.no_grad():
        for inputs, labels, video_paths in tqdm(val_loader, desc='Feature extraction'):
            inputs = inputs.to(device)
            # 调用 videomae 的 encoder 部分，获取 last_hidden_state
            outputs = model.videomae(pixel_values=inputs)
            # outputs 是 BaseModelOutput 类型，取 last_hidden_state
            if hasattr(outputs, 'last_hidden_state'):
                last_hidden = outputs.last_hidden_state
            else:
                last_hidden = outputs[0]   # 通常是 (batch, seq_len, hidden_dim)
            # 取 [CLS] token 的特征 (batch_size, hidden_dim)
            cls_features = last_hidden[:, 0, :].cpu().numpy()  # shape: (batch_size, 768)
            # 逐个样本添加到列表
            for i in range(cls_features.shape[0]):
                all_features.append(cls_features[i])
            all_labels.extend(labels.cpu().numpy())
            for vpath in video_paths:
                all_views.append(get_view_label(vpath))

    # 转换为 numpy 数组
    all_features = np.array(all_features)   # shape: (N, 768)
    all_labels = np.array(all_labels)
    all_views = np.array(all_views)

    print(f"提取到 {all_features.shape[0]} 个样本，特征维度 {all_features.shape[1]}")

    # 4. 标准化
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)

    # 5. t-SNE 降维
    print("Running t-SNE (this may take a while)...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, init='pca')
    features_2d = tsne.fit_transform(all_features_scaled)

    # 6. 可视化
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=all_labels, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter1, ticks=range(num_classes), label='Class')
    plt.title('t-SNE by Class')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    plt.subplot(1, 2, 2)
    unique_views = np.unique(all_views)
    colors = {'first_person': 'red', 'top_down': 'blue', 'unknown': 'gray'}
    for view in unique_views:
        mask = (all_views == view)
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    c=colors.get(view, 'black'), label=view, alpha=0.6, s=20)
    plt.legend()
    plt.title('t-SNE by View')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=300)
    plt.show()

    print("\n视角分布统计：")
    for view in unique_views:
        count = np.sum(all_views == view)
        print(f"  {view}: {count} samples")
        
    # 假设 features_2d, all_labels, all_views 已经存在
    for c in range(num_classes):
        mask_class = (all_labels == c)
        if np.sum(mask_class) < 5:
            continue
        plt.figure(figsize=(6,5))
        for view in ['first_person', 'top_down']:
            mask_view = (all_views == view)
            mask = mask_class & mask_view
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], label=view, alpha=0.6, s=20)
        plt.title(f'Class {class_names[c]}')
        plt.legend()
        plt.savefig(f'tsne_class_{c}.png')
        plt.close()

if __name__ == "__main__":
    main()
