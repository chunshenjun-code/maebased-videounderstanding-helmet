# 安全帽佩戴检测 - 实时推理系统

基于 VideoMAE 动作分类 + MediaPipe 手部检测 + 语音交互的**实时安全帽佩戴动作验证系统**，用于指导/评测用户按正确步骤佩戴安全帽。

## 功能概述

系统通过摄像头实时捕捉用户操作，检测手部动作并保持手部区域清晰（背景模糊保护隐私），使用 VideoMAE 模型对用户动作片段进行分类，同时支持语音指令交互与中文语音反馈。

### 核心工作流

```
摄像头实时画面 → 手部检测 → 背景模糊 → 保存动作片段 → VideoMAE 分类
                                                                 ↓
语音指令 → Vosk 语音识别 → 动作序列比对 → 中文朗读反馈
```

## 目录结构

```
inference/
├── blur4.py                      # 手部检测器 v4: 渐变背景模糊 + YOLO 物体保留(torch)
├── blur5.py                      # 手部检测器 v5: TensorRT YOLO 加速 + 渐变背景模糊
├── blur6.py                      # 手部检测器 v6: TensorRT YOLO 加速 + 统一背景模糊(更快)
│
├── video_videomae_classify.py    # VideoMAE 单视频推理模块
│
├── main_videomae_blur .py        # 实时系统 v1: blur1 + VideoMAE + 语音(基础版)
├── main_videomae_blur4.py        # 实时系统 v2: blur4 + YOLO 动态物体保留
├── main_videomae_blur4_lite.py   # 实时系统 v3: blur4 精简版
├── main_videomae_blur5.py        # 实时系统 v4: blur5 + TensorRT YOLO
│
├── requirements.txt              # Python 依赖
└── README.md                     # 本文件
```

## 各模块说明

### 手部检测与背景模糊 (blur4 / blur5 / blur6)

三个版本的手部检测器继承演进，基于 **MediaPipe Hand Landmarker** 检测手部关键点。

| 文件 | 手部检测 | 背景模糊方式 | 物体检测 | 推理加速 |
|------|----------|-------------|----------|----------|
| `blur4.py` | MediaPipe | 渐变模糊（距离变换） | YOLOv5 (torch.hub) | - |
| `blur5.py` | MediaPipe | 渐变模糊（距离变换） | YOLOv5 (TensorRT) | TensorRT + PyCUDA |
| `blur6.py` | MediaPipe | 统一模糊（无渐变，更快） | YOLOv5 (TensorRT) | TensorRT + PyCUDA |

功能特性：
- **手部裁剪居中**：将检测到手部区域裁剪并居中放置到白色背景
- **背景模糊**：手部区域保留清晰，背景高斯模糊（支持渐变羽化或统一模糊）
- **物体保留**：结合 YOLO 检测，可指定保留特定物体（如人、安全帽）与手部一同保持清晰
- **视频拆分**：按手部出现帧连续阈值将长视频拆分为多个片段
- **摄像头实时**：支持 USB 摄像头实时流处理

```bash
# 命令行使用示例
python blur4.py --mode video --input demo.mp4 --blur --output_dir output/
python blur5.py --mode camera --blur --yolo_weights best_fp16.engine --object_ids 0 --preview
```

### 视频分类推理 (video_videomae_classify.py)

单视频推理模块，加载微调后的 VideoMAE 模型对 MP4 文件进行分类。

```bash
# 单视频分类
python video_videomae_classify.py video.mp4 --model videomae_best_blur4.pth --top_k 3
```

输入参数：
- `video_path`: 输入视频文件路径（位置参数）
- `--model`: 模型权重文件（默认 `videomae_best_blur.pt`）
- `--top_k`: 输出前 k 个最可能类别（默认 3）
- `--class_names`: 类别名称列表（默认 `0 1 2 3 4 5 6 7`）

### 实时交互系统 (main_videomae_blur*.py)

多线程实时推理系统，架构如下：

```
┌─────────────────────┐     ┌─────────────────────┐
│   摄像头线程         │     │   音频线程           │
│   (手部检测+背景模糊 │     │   (Vosk 语音识别)    │
│    保存片段到磁盘)    │     │                     │
└─────────┬───────────┘     └──────────┬──────────┘
          │ segment_queue              │ command_queue
          ▼                            ▼
┌──────────────────────────────────────────────┐
│   分类线程                                     │
│   VideoMAE 推理 → 动作序列比对                  │
│   → 动态更新物体 ID → 语音反馈                  │
└──────────────────────────────────────────────┘
```

| 文件 | 手部检测器 | 物体保留 | 语音导入 | 特点 |
|------|-----------|---------|---------|------|
| `main_videomae_blur .py` | blur1 | ❌ | vosk + espeak | 基础版，仅背景模糊 |
| `main_videomae_blur4.py` | blur4 | ✅ 动态物体ID | vosk + espeak | YOLO 物体动态保留 |
| `main_videomae_blur4_lite.py` | blur4 | ✅ 动态物体ID | vosk + espeak | 精简版 |
| `main_videomae_blur5.py` | blur5(TensorRT) | ✅ 动态物体ID | vosk + espeak | TensorRT 加速 |

核心特性：
- **预设动作序列**：预设 5 步正确动作序列（"0→1→2→3→4"），实时比对用户操作
- **动态物体保留**：根据当前动作步骤自动切换需要保留清晰的物体 ID（如动作"0"时保留"人"，动作"1"时保留"安全帽"）
- **语音指令**：通过 Vosk 离线语音识别支持"重复"、"历史"、"下一步"等指令
- **中文语音反馈**：使用 espeak 朗读中文指导语（动作正确/错误提醒、进度播报）
- **纠错指导**：检测到错误动作时语音提示正确操作

```bash
# 启动实时交互系统
python main_videomae_blur4.py
```

## 安装

```bash
# 1. 创建虚拟环境（建议使用 conda）
conda create -n helmet-infer python=3.10
conda activate helmet-infer

# 2. 安装 PyTorch（根据 CUDA 版本选择）
# CUDA 11.8:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 3. 安装其他依赖
pip install -r requirements.txt

# 4. 安装 Vosk 中文语音模型
# 下载 vosk-model-small-cn-0.22 并解压到项目目录
# wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip

# 5. 安装 espeak（中文语音朗读）
# Linux:  sudo apt-get install espeak espeak-data
# macOS:  brew install espeak
# Windows: 下载安装 https://espeak.sourceforge.net/

# 6. （建议）YOLO 物体检测
git clone https://github.com/ultralytics/yolov5
cd yolov5 && pip install -r requirements.txt

# 7. （可选）TensorRT 推理加速
# 从 NVIDIA 官网下载 TensorRT 并安装对应 Python 包
pip install pycuda>=2022.1
```

### 模型文件清单

运行推理前需要准备以下模型文件：

| 文件 | 来源 | 用途 |
|------|------|------|
| `videomae_best_blur4.pth` | training 训练产出 | VideoMAE 视频分类 |
| `best_fp16.engine` | YOLO→TensorRT 转换 | TensorRT YOLO 物体检测 |
| `hand_landmarker.task` | MediaPipe（自动下载） | 手部关键点检测 |
| `vosk-model-small-cn-0.22/` | alphacephei.com | 中文语音识别 |

## 快速开始

```bash
# 1. 单视频分类测试
python video_videomae_classify.py test.mp4 --model videomae_best_blur4.pth

# 2. 摄像头实时分类（基础版）
python "main_videomae_blur .py"

# 3. 带 YOLO 物体保留 + 语音交互
python main_videomae_blur4.py
```

## 依赖关系说明

与 training 目录的区别：

| 模块 | training | inference |
|------|----------|-----------|
| PyTorch/Torchvision | ✅ | ✅ |
| Transformers | ✅ (训练) | ✅ (推理) |
| MediaPipe | ✅ | ✅ |
| Albumentations | ✅ | ❌ |
| scikit-learn / joblib | ✅ | ❌ |
| Vosk / pyaudio | ❌ | ✅ 语音识别 |
| TensorRT / PyCUDA | ❌ | ✅(可选，blur5/6) |
| espeak | ❌ | ✅(系统命令) 语音朗读 |

## 推荐使用流程

1. 使用 `training/` 训练 VideoMAE 模型 → 得到 `videomae_best_blur4.pth`
2. （可选）将 YOLO 模型转为 TensorRT 引擎 → 得到 `best_fp16.engine`
3. 使用 `inference/` 中的实时系统进行部署
4. 先用 `video_videomae_classify.py` 测试单视频分类效果
5. 启动 `main_videomae_blur4.py` 进行实时交互验证

## 参考

- [VideoMAE](https://arxiv.org/abs/2203.12602)
- [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [Vosk Speech Recognition](https://alphacephei.com/vosk/)
- [TensorRT](https://developer.nvidia.com/tensorrt)
- [espeak](https://espeak.sourceforge.net/)
