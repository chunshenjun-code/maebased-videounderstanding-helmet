# vit_test_tsne_all.py
# 从四个视角的验证集提取 VideoMAE 特征，分别绘制两张独立的 t-SNE 图：
# - 图1：按 8 个动作类别着色（带图例）
# - 图2：按 4 个不同视角着色（带图例）

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Subset
from transformers import VideoMAEForVideoClassification, AutoImageProcessor
from vit_videomae_seft import VideoMAEDataset   # 复用原有的 Dataset 类

# ==================== 配置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'videomae_sequential_finetuned.pth'   # 请修改为实际路径

# 四个视角的数据根目录（与 vit_videomae_seft.py 中的 data_domains 保持一致）
data_domains = {
    "first": "data_str_blur4",       # 第一人称
    "down":  "data2_down_str_blur4", # 俯视
    "left":  "data2_left_str_blur4", # 左侧
    "right": "data2_right_str_blur4" # 右侧
}

val_subdir = "val"

class_names = ['0', '1', '2', '3', '4', '5', '6', '7']
num_classes = len(class_names)

num_frames = 16
num_frames_per_clip = 64
batch_size = 8

# t-SNE 参数
n_samples_per_class = None   # 若限制每个类别最多样本数，可设为 50；None 表示使用全部
perplexity = 30
n_components = 2
random_state = 42

# ==================== 自定义 Dataset ====================
class FourViewDataset(VideoMAEDataset):
    def __init__(self, root_dir, view_name, class_names, num_frames=16, num_clips=1,
                 num_frames_per_clip=64, processor=None, training=False):
        super().__init__(root_dir, class_names, num_frames, num_clips,
                         num_frames_per_clip, processor, training)
        self.view_name = view_name

    def __getitem__(self, idx):
        pixel_values, label = super().__getitem__(idx)
        return pixel_values, label, self.view_name

# ==================== 加载模型 ====================
def load_feature_extractor(model_path, num_classes, device):
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

def extract_features(model, dataloader, device):
    """提取每个样本的 [CLS] 特征"""
    all_features = []
    all_labels = []
    all_views = []
    with torch.no_grad():
        for inputs, labels, view_names in tqdm(dataloader, desc="特征提取"):
            inputs = inputs.to(device)
            outputs = model(pixel_values=inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            cls_features = last_hidden[:, 0, :]
            all_features.append(cls_features.cpu())
            all_labels.extend(labels.numpy())
            all_views.extend(view_names)
    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = np.array(all_labels)
    all_views = np.array(all_views)
    return all_features, all_labels, all_views

# ==================== 主程序 ====================
def main():
    # 1. 创建 processor
    processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

    # 2. 构建数据集
    all_datasets = []
    for view_name, domain_root in data_domains.items():
        val_dir = os.path.join(domain_root, val_subdir)
        if not os.path.isdir(val_dir):
            print(f"警告: {val_dir} 不存在，跳过视角 {view_name}")
            continue
        ds = FourViewDataset(
            root_dir=val_dir,
            view_name=view_name,
            class_names=class_names,
            num_frames=num_frames,
            num_frames_per_clip=num_frames_per_clip,
            processor=processor,
            training=False
        )
        all_datasets.append(ds)
        print(f"视角 {view_name}: {len(ds)} 个样本")

    if not all_datasets:
        raise RuntimeError("未找到任何有效视角的验证集！")

    full_dataset = ConcatDataset(all_datasets)
    print(f"总样本数: {len(full_dataset)}")

    # 可选采样
    if n_samples_per_class is not None:
        print("收集类别标签以进行采样...")
        all_labels_temp = []
        for i in range(len(full_dataset)):
            _, label, _ = full_dataset[i]
            all_labels_temp.append(label)
        indices_per_class = {c: [] for c in range(num_classes)}
        for idx, lbl in enumerate(all_labels_temp):
            indices_per_class[lbl].append(idx)
        selected_indices = []
        for c in range(num_classes):
            lst = indices_per_class[c]
            if len(lst) > n_samples_per_class:
                lst = np.random.choice(lst, n_samples_per_class, replace=False)
            selected_indices.extend(lst)
        dataset = Subset(full_dataset, selected_indices)
        print(f"采样后每个类别最多 {n_samples_per_class} 个样本，总样本数: {len(dataset)}")
    else:
        dataset = full_dataset

    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # 3. 加载模型
    print("加载模型...")
    model, processor = load_feature_extractor(model_path, num_classes, device)

    # 4. 提取特征
    print("开始提取特征...")
    all_features, all_labels, all_views = extract_features(model, val_loader, device)
    print(f"特征提取完成，样本数: {all_features.shape[0]}, 特征维度: {all_features.shape[1]}")

    # 5. 标准化 + t-SNE
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)

    print("运行 t-SNE (可能需要几分钟)...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=random_state, init='pca')
    features_2d = tsne.fit_transform(features_scaled)

    # ==================== 图1：按动作类别（带图例） ====================
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=all_labels, cmap='tab10', alpha=0.6, s=20)
    # 生成图例
    handles, labels = scatter.legend_elements()
    plt.legend(handles, labels, title="Class", loc='upper right', fontsize=9, title_fontsize=10)
    plt.title('t-SNE by Action Class', fontsize=14)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig('tsne_by_class.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("已保存图1: tsne_by_class.png (图例已添加)")

    # ==================== 图2：按四个视角（带图例） ====================
    plt.figure(figsize=(8, 6))
    unique_views = np.unique(all_views)
    view_colors = {'first': 'red', 'down': 'blue', 'left': 'green', 'right': 'orange'}
    for view in unique_views:
        mask = (all_views == view)
        color = view_colors.get(view, 'gray')
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    c=color, label=view, alpha=0.6, s=20)
    # 添加图例
    plt.legend(title="View", loc='upper right', fontsize=9, title_fontsize=10)
    plt.title('t-SNE by View (First/Down/Left/Right)', fontsize=14)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig('tsne_by_view.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("已保存图2: tsne_by_view.png (图例已添加)")

    # 统计信息
    print("\n视角分布统计：")
    for view in unique_views:
        count = np.sum(all_views == view)
        print(f"  {view}: {count} samples")

if __name__ == "__main__":
    main()
