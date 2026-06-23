#!/usr/bin/env python3
"""
批量处理 data 文件夹下所有 MP4 文件，进行手部裁剪居中，并保持目录结构输出到 data_done。
依赖 hand_v4.py 中的 VideoHandDetector 类。
"""

import sys
from pathlib import Path
from blur1 import VideoHandDetector  # 确保 hand_v4.py 在同一目录或 Python 路径中

def main():
    input_root = Path("new_test_valid")
    output_root = Path("new_test_valid_blur")

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

        # 为每个视频创建新的检测器（避免复用导致关闭问题）
        detector = VideoHandDetector(confidence_threshold=0.5)

        try:
            # 调用 process_video，启用裁剪居中，不显示预览
            detector.process_video(
                input_video_path=str(mp4_path),
                output_video_path=str(out_path),
                show_preview=False,
                crop_hand=False,
                crop_margin=50,   # 可根据需要调整边距
                blur_background=True
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
