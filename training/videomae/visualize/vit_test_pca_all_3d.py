# vit_pca_3d_visualization.py
# 使用 PCA 降维到 3D，可视化 VideoMAE 特征

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Subset
from transformers import VideoMAEForVideoClassification, AutoImageProcessor
from vit_videomae_seft import VideoMAEDataset

# ==================== 配置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'videomae_sequential_finetuned.pth'   # 请修改为实际路径

data_domains = {
    "first": "data_str_blur4",
    "down":  "data2_down_str_blur4",
    "left":  "data2_left_str_blur4",
    "right": "data2_right_str_blur4"
}
val_subdir = "val"

class_names = ['0', '1', '2', '3', '4', '5', '6', '7']
num_classes = len(class_names)

num_frames = 16
num_frames_per_clip = 64
batch_size = 8

# PCA 参数
n_components = 3
random_state = 42

n_samples_per_class = None   # 每类最多样本数

view_colors = {
    'first': 'red',
    'down':  'blue',
    'left':  'green',
    'right': 'orange'
}

# ==================== Dataset ====================
class FourViewDataset(VideoMAEDataset):
    def __init__(self, root_dir, view_name, class_names, num_frames=16, num_clips=1,
                 num_frames_per_clip=64, processor=None, training=False):
        super().__init__(root_dir, class_names, num_frames, num_clips,
                         num_frames_per_clip, processor, training)
        self.view_name = view_name

    def __getitem__(self, idx):
        pixel_values, label = super().__getitem__(idx)
        return pixel_values, label, self.view_name

# ==================== 模型与特征提取 ====================
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
    all_features = []
    all_labels = []
    all_views = []
    with torch.no_grad():
        for inputs, labels, view_names in tqdm(dataloader, desc="Extracting features"):
            inputs = inputs.to(device)
            outputs = model.videomae(pixel_values=inputs)
            if hasattr(outputs, 'last_hidden_state'):
                last_hidden = outputs.last_hidden_state
            else:
                last_hidden = outputs[0]
            cls_features = last_hidden[:, 0, :].cpu().numpy()
            all_features.append(cls_features)
            all_labels.extend(labels.numpy())
            all_views.extend(view_names)
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)
    all_views = np.array(all_views)
    return all_features, all_labels, all_views

# ==================== 绘图函数 ====================
def plot_3d_scatter(features_3d, labels_or_views, title, filename, is_class=True, view_color_dict=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    if is_class:
        unique = np.unique(labels_or_views)
        cmap = plt.cm.get_cmap('tab10', len(unique))
        for i, lbl in enumerate(unique):
            mask = (labels_or_views == lbl)
            ax.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                       c=[cmap(i)], label=str(lbl), alpha=0.6, s=15)
        ax.legend(title="Class", fontsize=8)
    else:
        unique = np.unique(labels_or_views)
        for v in unique:
            mask = (labels_or_views == v)
            color = view_color_dict.get(v, 'gray')
            ax.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                       c=color, label=v, alpha=0.6, s=15)
        ax.legend(title="View", fontsize=8)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

def plot_per_class_3d(features_3d, labels, views, class_names, view_color_dict, out_dir='./'):
    for c in range(len(class_names)):
        mask_class = (labels == c)
        if np.sum(mask_class) == 0:
            continue
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        unique_views = np.unique(views[mask_class])
        for v in unique_views:
            mask = mask_class & (views == v)
            color = view_color_dict.get(v, 'gray')
            ax.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                       c=color, label=v, alpha=0.6, s=15)
        ax.set_title(f'Class {class_names[c]} (PCA 3D)', fontsize=12)
        ax.legend(title="View", fontsize=8)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.tight_layout()
        save_path = os.path.join(out_dir, f'pca3d_class_{class_names[c]}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

# ==================== 主程序 ====================
def main():
    processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    
    # 数据集
    all_datasets = []
    for view_name, domain_root in data_domains.items():
        val_dir = os.path.join(domain_root, val_subdir)
        if not os.path.isdir(val_dir):
            print(f"Warning: {val_dir} not exists, skip {view_name}")
            continue
        ds = FourViewDataset(val_dir, view_name, class_names, num_frames,
                             num_frames_per_clip=num_frames_per_clip,
                             processor=processor, training=False)
        all_datasets.append(ds)
        print(f"{view_name}: {len(ds)} samples")
    if not all_datasets:
        raise RuntimeError("No valid view data!")
    full_dataset = ConcatDataset(all_datasets)
    print(f"Total samples: {len(full_dataset)}")
    
    # 采样
    if n_samples_per_class is not None:
        all_labels_temp = []
        for i in range(len(full_dataset)):
            _, label, _ = full_dataset[i]
            all_labels_temp.append(label)
        indices_per_class = {c: [] for c in range(num_classes)}
        for idx, lbl in enumerate(all_labels_temp):
            indices_per_class[lbl].append(idx)
        selected = []
        for c in range(num_classes):
            lst = indices_per_class[c]
            if len(lst) > n_samples_per_class:
                lst = np.random.choice(lst, n_samples_per_class, replace=False)
            selected.extend(lst)
        dataset = Subset(full_dataset, selected)
        print(f"Sampled to {len(dataset)} samples")
    else:
        dataset = full_dataset
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    
    # 模型
    print("Loading model...")
    model, processor = load_feature_extractor(model_path, num_classes, device)
    
    # 特征提取
    print("Extracting features...")
    features, labels, views = extract_features(model, loader, device)
    print(f"Feature shape: {features.shape}")
    
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # PCA
    print("Running PCA to 3D...")
    pca = PCA(n_components=3, random_state=random_state)
    features_3d = pca.fit_transform(features_scaled)
    print(f"Explained variance: {pca.explained_variance_ratio_}")
    print(f"Cumulative: {np.sum(pca.explained_variance_ratio_):.3f}")
    
    # 绘图
    plot_3d_scatter(features_3d, labels,
                    title='PCA 3D by Action Class',
                    filename='pca3d_by_class.png',
                    is_class=True)
    plot_3d_scatter(features_3d, views,
                    title='PCA 3D by View',
                    filename='pca3d_by_view.png',
                    is_class=False, view_color_dict=view_colors)
    plot_per_class_3d(features_3d, labels, views, class_names, view_colors)
    
    print("All PCA 3D visualizations done.")

if __name__ == "__main__":
    main()
