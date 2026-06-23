# vit_pca_visualization.py
# 使用 PCA 降维（2D）可视化 VideoMAE 特征
# - 图1：按 8 个动作类别着色（带图例）
# - 图2：按 4 个不同视角着色（带图例）
# - 额外：每个类别一张子图，区分四个视角

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Subset
from transformers import VideoMAEForVideoClassification, AutoImageProcessor
from vit_videomae_seft import VideoMAEDataset   # 复用原有的 Dataset 类

# ==================== 配置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model_path = 'videomae_sequential_finetuned.pth'   # 请修改为实际路径
#model_path = 'videomae_best_blur4.pth'  #分段微调前的模型
model_path = 'videomae_best_allv_blur4.pth' #整体训练的模型

# 四个视角的数据根目录
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
n_components = 2          # 降维到2维
n_samples_per_class = None  # 每类最大样本数（None=全部）
random_state = 42

# 视角颜色
view_colors = {
    'first': 'red',
    'down':  'blue',
    'left':  'green',
    'right': 'orange'
}

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

def extract_features_via_videomae(model, dataloader, device):
    """使用 model.videomae 提取 [CLS] 特征"""
    all_features = []
    all_labels = []
    all_views = []
    with torch.no_grad():
        for inputs, labels, view_names in tqdm(dataloader, desc="特征提取"):
            inputs = inputs.to(device)
            outputs = model.videomae(pixel_values=inputs)
            if hasattr(outputs, 'last_hidden_state'):
                last_hidden = outputs.last_hidden_state
            else:
                last_hidden = outputs[0]
            cls_features = last_hidden[:, 0, :]   # [CLS] token
            all_features.append(cls_features.cpu())
            all_labels.extend(labels.numpy())
            all_views.extend(view_names)
    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = np.array(all_labels)
    all_views = np.array(all_views)
    return all_features, all_labels, all_views

# ==================== 绘图函数 ====================
def plot_pca_by_class(features_2d, labels, save_path='pca_by_class.png'):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=labels, cmap='tab10', alpha=0.6, s=20)
    handles, labels_leg = scatter.legend_elements()
    plt.legend(handles, labels_leg, title="Class", loc='upper right', fontsize=9, title_fontsize=10)
    plt.title('PCA of [CLS] Features (by Action Class)', fontsize=14)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"已保存: {save_path}")

def plot_pca_by_view(features_2d, views, view_color_dict, save_path='pca_by_view.png'):
    plt.figure(figsize=(8, 6))
    unique_views = np.unique(views)
    for v in unique_views:
        mask = (views == v)
        color = view_color_dict.get(v, 'gray')
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    c=color, label=v, alpha=0.6, s=20)
    plt.legend(title="View", loc='upper right', fontsize=9, title_fontsize=10)
    plt.title('PCA of [CLS] Features (by View)', fontsize=14)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"已保存: {save_path}")

def plot_per_class_separate_pca(features_2d, labels, views, class_names, view_color_dict, out_dir='./'):
    for c in range(len(class_names)):
        mask_class = (labels == c)
        if np.sum(mask_class) == 0:
            continue
        plt.figure(figsize=(6, 5))
        unique_views_in_class = np.unique(views[mask_class])
        for v in unique_views_in_class:
            mask = mask_class & (views == v)
            if np.sum(mask) == 0:
                continue
            color = view_color_dict.get(v, 'gray')
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                        c=color, label=v, alpha=0.6, s=20)
        plt.title(f'Class {class_names[c]} (PCA)', fontsize=12)
        plt.legend(title="View", fontsize=8, title_fontsize=9)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.tight_layout()
        save_path = os.path.join(out_dir, f'pca_class_{class_names[c]}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")

# ==================== 主程序 ====================
def main():
    # 1. 创建 processor
    processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

    # 2. 构建四视角数据集
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

    # 可选：限制每类样本数
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
    all_features, all_labels, all_views = extract_features_via_videomae(model, val_loader, device)
    print(f"特征提取完成，样本数: {all_features.shape[0]}, 特征维度: {all_features.shape[1]}")

    # 5. 标准化（PCA 对尺度敏感，建议做）
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)

    # 6. PCA 降维
    print("运行 PCA...")
    pca = PCA(n_components=n_components, random_state=random_state)
    features_2d = pca.fit_transform(features_scaled)
    print(f"PCA 解释方差比: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
    print(f"累计解释方差: {pca.explained_variance_ratio_.sum():.3f}")

    # 7. 绘图
    plot_pca_by_class(features_2d, all_labels, save_path='pca_by_class.png')
    plot_pca_by_view(features_2d, all_views, view_colors, save_path='pca_by_view.png')
    plot_per_class_separate_pca(features_2d, all_labels, all_views, class_names,
                                view_colors, out_dir='./')

    # 统计信息
    print("\n视角分布统计：")
    for view in np.unique(all_views):
        count = np.sum(all_views == view)
        print(f"  {view}: {count} samples")

if __name__ == "__main__":
    main()
