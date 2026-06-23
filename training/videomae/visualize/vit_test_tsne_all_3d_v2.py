# vit_tsne_3d_interactive.py
# 使用 t-SNE 降维到 3D，并用 Plotly 生成交互式 HTML 图（带光晕特效）

import os
import torch
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
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

# t-SNE 参数
n_components = 3
perplexity = 30
random_state = 42
#init = 'pca'
init = 'random'
#max_iter = 1000
max_iter = 1500
verbose = 1

n_samples_per_class = None   # 如果样本太多，建议设置每类 50 个

# 视角颜色（用于按视角着色）
view_colors = {
    'first': '#FF4136',   # 红
    'down':  '#0074D9',   # 蓝
    'left':  '#2ECC40',   # 绿
    'right': '#FF851B'    # 橙
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

# ==================== 交互式 3D 绘图函数（带光晕特效） ====================
def plot_3d_interactive(features_3d, color_values, color_map, title, filename,
                        is_class=True, class_names=None, view_colors=None):
    """
    使用 Plotly 生成交互式 3D 散点图，实现「光晕特效」：
     - 每个点绘制两次：第一次大尺寸、低透明度（光晕层），第二次小尺寸、高透明度（核心层）
     - 或者直接增大点尺寸并启用 marker 轮廓线（模拟光晕）
    """
    if is_class:
        # 按类别着色：color_values 为标签整数
        unique_labels = np.unique(color_values)
        # 使用 plotly 的定性色板
        plotly_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                         '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
        colors = [plotly_colors[l % len(plotly_colors)] for l in color_values]
        legend_title = "Class"
        # 为图例创建自定义条目
        legend_names = {l: str(l) for l in unique_labels}
    else:
        # 按视角着色：color_values 为视角名称字符串
        colors = [view_colors.get(v, '#AAAAAA') for v in color_values]
        legend_title = "View"
        unique_views = np.unique(color_values)
        legend_names = {v: v for v in unique_views}

    # ---- 光晕层：大尺寸、低透明度、无轮廓 ----
    trace_glow = go.Scatter3d(
        x=features_3d[:, 0], y=features_3d[:, 1], z=features_3d[:, 2],
        mode='markers',
        marker=dict(
            size=12,                # 比核心层稍大
            color=colors,
            opacity=0.25,           # 半透明形成光晕
            symbol='circle',
            line=dict(width=0)      # 无边框
        ),
        name='glow',
        showlegend=False            # 不显示在图例中
    )

    # ---- 核心层：较小尺寸、高不透明度、白色边缘增强光晕感 ----
    trace_core = go.Scatter3d(
        x=features_3d[:, 0], y=features_3d[:, 1], z=features_3d[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=colors,
            opacity=0.9,
            symbol='circle',
            line=dict(width=1, color='white')   # 白边强化轮廓
        ),
        name='core',
        showlegend=False
    )

    # 组合两个 trace
    fig = go.Figure(data=[trace_glow, trace_core])

    # 配置图例（手动添加图例条目）
    if is_class:
        for lbl in unique_labels:
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=8, color=plotly_colors[lbl % len(plotly_colors)]),
                name=str(lbl),
                showlegend=True
            ))
    else:
        for view in unique_views:
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=8, color=view_colors.get(view, '#AAAAAA')),
                name=view,
                showlegend=True
            ))

    # 布局设置
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            zaxis_title='t-SNE 3',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))  # 初始视角
        ),
        legend=dict(title=legend_title, font=dict(size=12)),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 保存为 HTML 文件（支持交互旋转/缩放）
    fig.write_html(filename)
    print(f"Saved interactive 3D plot: {filename}")
    # 也可在笔记本中显示：fig.show()
    return fig

# ==================== 每个类别的单独交互图（按视角着色） ====================
def plot_per_class_interactive(features_3d, labels, views, class_names, view_colors, out_dir='./'):
    for c in range(len(class_names)):
        mask_class = (labels == c)
        if np.sum(mask_class) == 0:
            continue
        # 提取当前类别的特征、视角
        feats_c = features_3d[mask_class]
        views_c = views[mask_class]
        colors_c = [view_colors.get(v, '#AAAAAA') for v in views_c]

        # 光晕层 + 核心层
        trace_glow = go.Scatter3d(
            x=feats_c[:, 0], y=feats_c[:, 1], z=feats_c[:, 2],
            mode='markers',
            marker=dict(size=12, color=colors_c, opacity=0.25, line=dict(width=0)),
            showlegend=False
        )
        trace_core = go.Scatter3d(
            x=feats_c[:, 0], y=feats_c[:, 1], z=feats_c[:, 2],
            mode='markers',
            marker=dict(size=6, color=colors_c, opacity=0.9, line=dict(width=1, color='white')),
            showlegend=False
        )
        fig = go.Figure(data=[trace_glow, trace_core])

        # 手动添加图例
        unique_views = np.unique(views_c)
        for v in unique_views:
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=8, color=view_colors.get(v, '#AAAAAA')),
                name=v,
                showlegend=True
            ))

        fig.update_layout(
            title=f'Class {class_names[c]} (3D t-SNE, colored by view)',
            scene=dict(xaxis_title='t-SNE 1', yaxis_title='t-SNE 2', zaxis_title='t-SNE 3'),
            legend=dict(title="View"),
            width=900, height=700
        )
        save_path = os.path.join(out_dir, f'tsne3d_class_{class_names[c]}_interactive.html')
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

# ==================== 主程序 ====================
def main():
    processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    
    # 构建数据集
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
    
    # 采样（强烈建议控制样本数，避免 HTML 文件过大）
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
        print("Warning: using all samples, HTML file may be very large and slow.")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    
    # 模型加载
    print("Loading model...")
    model, processor = load_feature_extractor(model_path, num_classes, device)
    
    # 特征提取
    print("Extracting features...")
    features, labels, views = extract_features(model, loader, device)
    print(f"Feature shape: {features.shape}")
    
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # t-SNE 降维到 3D
    print("Running t-SNE to 3D...")
    tsne = TSNE(n_components=3, perplexity=perplexity,
                random_state=random_state, init=init, max_iter=max_iter,
                verbose=verbose)
    features_3d = tsne.fit_transform(features_scaled)
    print("t-SNE finished.")
    
    # 绘制交互式 3D 图（按类别）
    plot_3d_interactive(
        features_3d, labels, None, 
        title='t-SNE 3D by Action Class (with glow effect)',
        filename='tsne3d_by_class_interactive.html',
        is_class=True, class_names=class_names
    )
    
    # 绘制交互式 3D 图（按视角）
    plot_3d_interactive(
        features_3d, views, view_colors,
        title='t-SNE 3D by View (with glow effect)',
        filename='tsne3d_by_view_interactive.html',
        is_class=False, view_colors=view_colors
    )
    
    # 每个类别单独交互图（按视角着色）
    plot_per_class_interactive(features_3d, labels, views, class_names, view_colors)
    
    print("All interactive 3D plots generated as HTML files.")

if __name__ == "__main__":
    main()
