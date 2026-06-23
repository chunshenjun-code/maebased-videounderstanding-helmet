#!/usr/bin/env python3
"""
批量处理 data 文件夹下所有 MP4 文件，进行背景模糊处理（手部区域保留，背景渐变模糊），
并保持目录结构输出到 data_done。
依赖 blur4.py 中的 VideoHandDetector 类。
"""

import sys
from pathlib import Path
from typing import Union, List, Optional

# 导入 blur4 中的检测器类
from blur4 import VideoHandDetector

# ===================== 用户可修改的配置 =====================
# YOLO 模型路径（若需要保留物体清晰，请设置为实际权重文件路径）
YOLO_WEIGHTS = "best_allv.pt"          # 例如 "yolov5s.pt" 或 "best.pt"
YOLO_CONF = 0.5              # YOLO 置信度阈值
YOLO_IMGSZ = 640             # YOLO 推理图像尺寸

# 背景模糊参数
MARGIN = 50                  # 手部包围盒边距（像素）
BLUR_KERNEL = 51             # 模糊核大小（奇数）
BLUR_SIGMA = 30.0            # 模糊标准差

# 自定义 object_ids 映射函数
def get_object_ids_for_video(rel_path: Path) -> Optional[Union[int, List[int]]]:
    """
    根据视频文件的相对路径（相对于输入根目录）返回需要保留清晰的物体类别 ID。
    返回 None 表示仅保留手部区域清晰，不保留额外物体。
    返回整数或整数列表表示需要保留的物体类别。
    用户可根据实际需求自定义映射逻辑。
    """
    # 示例：若视频位于某个子文件夹下，则返回特定物体 ID
    if "0" in rel_path.parts or "3" in rel_path.parts:
        return 2        # 保留人（类别 0）
    elif "1" in rel_path.parts or "6" in rel_path.parts or "7" in rel_path.parts:
        return 1   # 保留车（类别 2）和摩托车（类别 3）
    else:
        return 0

    # 默认不保留额外物体，仅手部清晰
    return None
# ============================================================


def main():
    input_root = Path("data2_right_str")
    output_root = Path("data2_right_str_blur4")

    if not input_root.is_dir():
        print(f"错误：输入文件夹 {input_root} 不存在")
        sys.exit(1)

    # 查找所有 MP4 文件（递归）
    mp4_files = list(input_root.rglob("*.mp4"))
    if not mp4_files:
        print(f"在 {input_root} 中未找到任何 MP4 文件")
        return

    print(f"找到 {len(mp4_files)} 个 MP4 文件，开始处理...")

    for i, mp4_path in enumerate(mp4_files, 1):
        # 计算相对路径，构建输出路径
        rel_path = mp4_path.relative_to(input_root)
        out_path = output_root / rel_path

        # 确保输出文件夹存在
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i}/{len(mp4_files)}] 处理: {mp4_path}")
        print(f"输出至: {out_path}")

        # 根据相对路径获取 object_ids
        object_ids = get_object_ids_for_video(rel_path)

        # 创建检测器（可复用，但为了安全每个视频新建，并传入 YOLO 参数）
        detector = VideoHandDetector(
            confidence_threshold=0.5,
            yolo_weights=YOLO_WEIGHTS,
            yolo_conf=YOLO_CONF,
            yolo_imgsz=YOLO_IMGSZ
        )

        try:
            # 调用 process_video，启用背景模糊，不显示预览
            detector.process_video(
                input_video_path=str(mp4_path),
                output_video_path=str(out_path),
                show_preview=False,
                crop_hand=False,           # 不裁剪
                blur_background=True,      # 启用背景模糊
                crop_margin=MARGIN,
                blur_kernel_size=BLUR_KERNEL,
                blur_sigma=BLUR_SIGMA,
                object_ids=object_ids      # 传入自定义物体 ID
            )
        except Exception as e:
            print(f"处理 {mp4_path} 时出错: {e}")
            # 继续处理下一个文件
        finally:
            # process_video 内部已关闭 detector，此处无需额外操作
            pass

    print("\n所有视频处理完成！")


if __name__ == "__main__":
    main()
