# vit_umap_3d_visualization.py
# 使用 UMAP 将特征降维到 3D，并用 matplotlib 绘制可交互旋转的 3D 散点图

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap
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

# UMAP 3D 参数
n_components = 3
n_neighbors = 50        # 可调整
min_dist = 0.1
metric = 'cosine'    #euclidean 或 'cosine'
random_state = 42

n_samples_per_class = None   # 每类最多样本数，None 表示全部

view_colors = {
    'first': 'red',
    'down': 'blue',
    'left': 'green',
    'right': 'orange'
}

# ==================== Dataset（同上） ====================
class FourViewDataset(VideoMAEDataset):
    def __init__(self, root_dir, view_name, class_names, num_frames=16, num_clips=1,
                 num_frames_per_clip=64, processor=None, training=False):
        super().__init__(root_dir, class_names, num_frames, num_clips,
                         num_frames_per_clip, processor, training)
        self.view_name = view_name

    def __getitem__(self, idx):
        pixel_values, label = super().__getitem__(idx)
        return pixel_values, label, self.view_name

# ==================== 模型加载与特征提取 ====================
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

# ==================== 3D 绘图函数 ====================
def plot_3d_scatter(features_3d, labels_or_views, title, filename, is_class=True, class_names=None, view_color_dict=None):
    """
    is_class: True 表示按类别着色（自动产生颜色映射）
               False 表示按视角着色（使用 view_color_dict）
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if is_class:
        # 按类别着色
        unique_labels = np.unique(labels_or_views)
        cmap = plt.cm.get_cmap('tab10', len(unique_labels))
        for i, lbl in enumerate(unique_labels):
            mask = (labels_or_views == lbl)
            color = cmap(i)
            ax.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                       c=[color], label=str(lbl), alpha=0.6, s=15)
        ax.legend(title="Class", fontsize=8, title_fontsize=10)
    else:
        # 按视角着色
        unique_views = np.unique(labels_or_views)
        for view in unique_views:
            mask = (labels_or_views == view)
            color = view_color_dict.get(view, 'gray')
            ax.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                       c=color, label=view, alpha=0.6, s=15)
        ax.legend(title="View", fontsize=8, title_fontsize=10)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_zlabel('UMAP3')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"已保存: {filename}")

def plot_per_class_3d(features_3d, labels, views, class_names, view_color_dict, out_dir='./'):
    """为每个类别单独绘制 3D 图，用不同视角颜色"""
    for c in range(len(class_names)):
        mask_class = (labels == c)
        if np.sum(mask_class) == 0:
            continue
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        unique_views_in_class = np.unique(views[mask_class])
        for v in unique_views_in_class:
            mask = mask_class & (views == v)
            if np.sum(mask) == 0:
                continue
            color = view_color_dict.get(v, 'gray')
            ax.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                       c=color, label=v, alpha=0.6, s=15)
        ax.set_title(f'Class {class_names[c]} (3D UMAP)', fontsize=12)
        ax.legend(title="View", fontsize=8, title_fontsize=9)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_zlabel('UMAP3')
        plt.tight_layout()
        save_path = os.path.join(out_dir, f'umap3d_class_{class_names[c]}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")

# ==================== 主程序 ====================
def main():
    processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    
    # 构建数据集
    all_datasets = []
    for view_name, domain_root in data_domains.items():
        val_dir = os.path.join(domain_root, val_subdir)
        if not os.path.isdir(val_dir):
            print(f"警告: {val_dir} 不存在，跳过 {view_name}")
            continue
        ds = FourViewDataset(val_dir, view_name, class_names, num_frames,
                             num_frames_per_clip=num_frames_per_clip,
                             processor=processor, training=False)
        all_datasets.append(ds)
        print(f"{view_name}: {len(ds)} 样本")
    
    if not all_datasets:
        raise RuntimeError("无有效视角数据！")
    
    full_dataset = ConcatDataset(all_datasets)
    print(f"总样本数: {len(full_dataset)}")
    
    # 可选采样
    if n_samples_per_class is not None:
        # 同之前采样逻辑...
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
        print(f"采样后样本数: {len(dataset)}")
    else:
        dataset = full_dataset
    
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    
    # 加载模型
    print("加载模型...")
    model, processor = load_feature_extractor(model_path, num_classes, device)
    
    # 提取特征
    print("提取特征...")
    features, labels, views = extract_features(model, val_loader, device)
    print(f"特征形状: {features.shape}")
    
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # UMAP 降维到 3D
    print("运行 UMAP (3D)...")
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors,
                        min_dist=min_dist, metric=metric,
                        random_state=random_state, verbose=True)
    features_3d = reducer.fit_transform(features_scaled)
    print("UMAP 完成")
    
    # 绘图1: 按类别
    plot_3d_scatter(features_3d, labels, 
                    title='UMAP 3D by Action Class',
                    filename='umap3d_by_class.png',
                    is_class=True)
    
    # 绘图2: 按视角
    plot_3d_scatter(features_3d, views,
                    title='UMAP 3D by View',
                    filename='umap3d_by_view.png',
                    is_class=False, view_color_dict=view_colors)
    
    # 绘图3: 每个类别单独图
    plot_per_class_3d(features_3d, labels, views, class_names, view_colors)
    
    print("所有三维可视化完成！")

if __name__ == "__main__":
    main()
