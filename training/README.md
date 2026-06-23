# 安全帽佩戴检测 - 视频动作分类训练

基于手部-物体交互视频的**安全帽佩戴动作分类**项目，包含从数据采集/预处理到三种深度学习方案的完整训练流程。

## 项目概述

针对安全帽佩戴场景的短视频进行分类（8 类动作），数据为不同视角下拍摄的手部-安全帽交互视频。项目包含：

- **数据准备**：基于 MediaPipe 的手部检测 + 背景模糊 / 手部裁剪居中 + 数据增强
- **三种训练方案**：从轻量级 CNN 到 3D CNN 再到 VideoMAE / TimeSformer 视频 Transformer
- **多视角支持**：第一人称、俯视、左侧、右侧四个视角的分段微调与评估
- **可视化分析**：t-SNE / PCA / UMAP 特征降维，混淆矩阵

## 目录结构

```
training/
├── data_prepare/          # 数据采集与预处理
│   ├── hand_v4.py         # 手部检测 + 裁剪居中（输出纯净视频）
│   ├── hand_v4_pure.py    # 精简版（仅裁剪，无标注）
│   ├── blur1.py           # 手部区域保留 + 背景高斯模糊（带羽化边缘）
│   ├── blur4.py           # 升级版：结合 YOLO 保留指定物体 + 背景渐变模糊
│   ├── batch_hand.py      # 批量手部裁剪
│   ├── batch_hand_blur4.py # 批量背景模糊（支持自定义 YOLO 物体 ID）
│   ├── data_strenghten.py # 随机旋转(-30°~30°) + 水平翻转，每视频生成5个增强版本
│   └── data_copy.py       # 跨文件夹视频拷贝工具
│
├── 3dcnn/                 # 方案一：R(2+1)D 3D CNN
│   ├── 3DCNN.py           # r2plus1d_18 预训练微调
│   └── confuse_mtrx_3dcnn.py  # 混淆矩阵生成
│
├── light/                 # 方案二：预训练CNN特征 + 时序池化（轻量级）
│   └── video_class_cnn.py # ResNet50 特征提取 + 时序池化 + LR/SVM
│
├── videomae/              # 方案三：视频 Transformer（VideoMAE / TimeSformer）
│   ├── Vit.py             # TimeSformer 微调
│   ├── vit_videomae.py    # VideoMAE 基础微调
│   ├── vit_videomae_str.py    # VideoMAE + 强数据增强（空间+时序）
│   ├── vit_videomae_fg.py     # VideoMAE + 前景保留增强（掩码保护手部）
│   ├── vit_videomae_light_str.py # VideoMAE + 光照增强
│   ├── vit_videomae_seft.py   # 多视角分段微调（经验回放防遗忘）
│   ├── confuse_mtrx_mae.py    # VideoMAE 混淆矩阵生成
│   └── visualize/             # 特征可视化
│       ├── vit_test_tsne.py / vit_test_tsne_all.py / t-SNE_3d.py ...
│       ├── vit_test_pca_all.py / vit_test_pca_all_3d.py ...
│       └── vit_test_umap_all.py / vit_test_umap_all_3d.py ...
│
├── requirements.txt       # Python 依赖
└── README.md              # 本文件
```

## 数据格式

所有模型共用统一的视频数据集结构：

```
data_root/
├── train/
│   ├── 0/    # 类别 0 的 MP4 视频
│   ├── 1/    # 类别 1 的 MP4 视频
│   ├── ...
│   └── 7/    # 类别 7 的 MP4 视频
├── val/
│   └── 0/1/.../7/
└── test/
    └── 0/1/.../7/
```

默认 8 类动作（`0` ~ `7`），也可调整为 4 类或 6 类（取消代码中对应 `class_names` 的注释即可）。

### 常见数据集命名

| 目录名 | 说明 |
|--------|------|
| `data_str` | 原始数据 |
| `data_str_blur4` | 背景模糊处理后的数据（手部保留清晰） |
| `data2_str_blur4` | 另一视角（如俯视）的模糊数据 |
| `data2_down_str_blur4` | 俯视视角 |
| `data2_left_str_blur4` | 左侧视角 |
| `data2_right_str_blur4` | 右侧视角 |
| `new_test_valid_blur4` | 独立测试集 |

## 数据准备流程

数据预处理包含两个主要阶段，可根据需求组合使用：

### 阶段一：数据增强

```bash
# 随机旋转 + 翻转，每个视频生成5个增强版本
python data_prepare/data_strenghten.py
# 编辑文件中 input_root / output_root 设置输入输出路径
# 随机角度范围 -30°~30°，随机水平翻转
```

### 阶段二：手部检测与背景处理

```bash
# 1a. 手部裁剪居中（可选，将手部区域裁剪并放置在画面中央）
python data_prepare/hand_v4.py --mode video --input input.mp4 --output_dir output/ --crop --margin 50 --preview

# 1b. 批量手部裁剪
python data_prepare/batch_hand.py
# 根据 hand_v4 的处理逻辑，读取 new_test_valid 输出到 new_test_valid_blur
# 如需修改输入输出路径，编辑 batch_hand.py 中的 input_root / output_root

# 1c. 批量背景模糊（推荐，效果比较好，手部保留清晰，背景模糊）
python data_prepare/batch_hand_blur4.py
# 需要 YOLO 权重文件 best_allv.pt
# 如需修改输入输出路径，编辑 batch_hand_blur4.py 中的 input_root / output_root
```

### 可选工具：数据拷贝

```bash
python data_prepare/data_copy.py -s 源文件夹B -d 目标文件夹A
python data_prepare/data_copy.py -s new_data -d data_str --no-overwrite
```

## 三种训练方案

### 方案一：R(2+1)D 3D CNN（`3dcnn/`）

使用在 Kinetics-400 上预训练的 `r2plus1d_18` 模型进行微调，输入 16 帧 x 224x224。

```bash
cd 3dcnn/
# 编辑 3DCNN.py 中的配置（data_root、num_epochs 等）
python 3DCNN.py

# 生成混淆矩阵
python confuse_mtrx_3dcnn.py
```

| 优势 | 劣势 |
|------|------|
| 训练速度快，显存要求低 | 表达能力强于时序池化但弱于 Transformer |
| 预训练模型成熟 | 输入帧数固定(16帧) |

### 方案二：预训练CNN特征 + 时序池化（`light/`）

使用预训练 ResNet50 提取每帧 2048 维特征，通过时序池化（均值/最大值）聚合为视频级特征，最后用 LR/SVM 分类。**最轻量级方案**。

```bash
cd light/
# 编辑 video_class_cnn.py 中的配置
python video_class_cnn.py
```

| 优势 | 劣势 |
|------|------|
| 训练极快（分钟级） | 时序信息有限（仅池化） |
| 支持缓存，重复运行快 | 精度上限较低 |
| 可解释性好 | 依赖特征质量 |

### 方案三：VideoMAE / TimeSformer Transformer（`videomae/`）

使用 HuggingFace 视频 Transformer 模型微调，推荐方案。

```bash
# 基础 VideoMAE 微调
cd videomae/
python vit_videomae.py

# TimeSformer 微调
python Vit.py

# 强数据增强版（空间+时序增强）
python vit_videomae_str.py

# 前景保留增强版（掩码保护手部区域）
python vit_videomae_fg.py

# 多视角分段微调（在第一人称预训练模型上，依次在俯视/左/右视角微调）
python vit_videomae_seft.py

# 生成混淆矩阵
python confuse_mtrx_mae.py
```

| 模型 | GPU 显存 | 特点 |
|------|----------|------|
| `vit_videomae.py` | ~8GB | VideoMAE base，基础版 | 推荐使用此代码训练基础模型，再使用`vit_videomae_seft.py`微调
| `Vit.py` | ~6GB | TimeSformer base，帧数灵活 |
| `vit_videomae_str.py` | ~8GB | 增加空间+时序增强 |
| `vit_videomae_fg.py` | ~8GB | 前景掩码保护，专为模糊数据设计 |
| `vit_videomae_seft.py` | ~8GB | 多视角顺序微调 + 经验回放 |

### 特征可视化

```bash
cd videomae/visualize/

# t-SNE 可视化（按类别着色）
python vit_test_tsne.py

# PCA 可视化（按类别 + 视角着色）
python vit_test_pca_all.py

# UMAP 可视化
python vit_test_umap_all.py
```

## 安装

```bash
# 1. 创建虚拟环境
建议使用conda管理环境，建议使训练设备环境与边缘设备环境具有相似性
# 或 venv\Scripts\activate  # Windows

# 2. 安装 PyTorch（根据你的 CUDA 版本从 pytorch.org 选择命令）
# CUDA 11.8 示例：
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
# CPU 版：
# pip install torch torchvision

# 3. 安装其他依赖
pip install -r requirements.txt

# 4. （可选）YOLO 物体检测支持
pip install ultralytics
```

> **注意**：首次运行手部检测脚本时会自动下载 MediaPipe 手部关键点模型（约 15MB）。

## 推荐工作流

1. **数据采集**：从多个视角拍摄安全帽佩戴视频
2. **数据增强**：使用 `data_strenghten.py` 扩充数据集
3. **数据预处理**：使用 `batch_hand_blur4.py` 对扩充后数据集进行背景模糊
4. **方案选择**：
   - 快速验证 → `light/video_class_cnn.py`
   - 精度优先 → `videomae/vit_videomae.py` 或 `vit_videomae_str.py`
   - 多视角泛化 → `videomae/vit_videomae_seft.py`
5. **评估**：生成混淆矩阵和 t-SNE / PCA 可视化

## 参考

- [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602)
- [TimeSformer: Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095)
- [R(2+1)D: A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248)
- [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
