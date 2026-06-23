#!/usr/bin/env python3
"""
对 data 文件夹下所有 MP4 视频进行随机旋转（-30°~30°）和随机左右翻转，
每个视频生成 5 个增强版本，保持目录结构保存到 data_done 文件夹。
"""

import os
import random
import cv2
import numpy as np
from pathlib import Path

# 设置随机种子（可选，保证可重复性）
random.seed(42)
np.random.seed(42)

def rotate_image(image, angle):
    """旋转图像（角度为度数），保持原尺寸，超出部分用黑色填充"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 计算旋转后图像的边界，但这里我们保持原尺寸，直接 warp
    rotated = cv2.warpAffine(image, rot_mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated

def flip_image(image, do_flip):
    """如果 do_flip 为 True，则水平翻转图像"""
    if do_flip:
        return cv2.flip(image, 1)  # 1 表示水平翻转
    return image

def process_video(input_path, output_path, angle, do_flip):
    """
    读取输入视频，对每一帧应用旋转和翻转，保存到输出视频。
    使用与输入相同的 FPS 和尺寸。
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"无法打开视频: {input_path}")
        return False

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 应用变换
        transformed = rotate_image(frame, angle)
        transformed = flip_image(transformed, do_flip)

        out.write(transformed)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"  已处理 {frame_count}/{total_frames} 帧")

    cap.release()
    out.release()
    print(f"  完成: {output_path} ({frame_count} 帧)")
    return True

def main():
    input_root = Path("data2_right")
    output_root = Path("data2_right_str")

    if not input_root.is_dir():
        print(f"错误：输入文件夹 {input_root} 不存在")
        return

    # 递归查找所有 MP4 文件
    mp4_files = list(input_root.rglob("*.mp4"))
    if not mp4_files:
        print(f"在 {input_root} 中未找到任何 MP4 文件")
        return

    print(f"找到 {len(mp4_files)} 个 MP4 文件，开始生成增强数据（每个视频生成 5 个版本）...\n")

    for idx, video_path in enumerate(mp4_files, 1):
        rel_path = video_path.relative_to(input_root)
        stem = rel_path.stem          # 不含扩展名的文件名
        parent = rel_path.parent

        print(f"[{idx}/{len(mp4_files)}] 处理: {video_path}")

        for aug_idx in range(5):
            # 随机生成增强参数
            angle = random.uniform(-30, 30)   # 随机角度
            do_flip = random.choice([True, False])  # 随机翻转
            #do_flip = False

            # 构建输出文件名：原文件名_aug{i}.mp4
            out_filename = f"{stem}_aug{aug_idx}.mp4"
            out_path = output_root / parent / out_filename

            # 确保输出目录存在
            out_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"  生成增强版本 {aug_idx}: 角度={angle:.2f}°, 翻转={do_flip}")
            process_video(video_path, out_path, angle, do_flip)

    print("\n所有视频增强完成！")

if __name__ == "__main__":
    main()
