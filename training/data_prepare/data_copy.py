#!/usr/bin/env python3
"""
脚本功能：将文件夹 B 中的视频文件拷贝到结构相同的文件夹 A 中。

使用方法：
    python copy_videos.py --source B --dest A
    python copy_videos.py -s B -d A --exts .mp4 .avi .mov --no-overwrite
"""

import os
import shutil
import argparse
from pathlib import Path

# 默认支持的视频文件扩展名（小写）
DEFAULT_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg', '.3gp', '.webm'}


def copy_videos(source_root: str, dest_root: str, video_exts: set, overwrite: bool = True):
    """
    遍历 source_root，将视频文件拷贝到 dest_root 的对应路径下。

    :param source_root: 源文件夹（B）
    :param dest_root:   目标文件夹（A）
    :param video_exts:  视频扩展名集合（如 {'.mp4', '.avi'}）
    :param overwrite:   是否覆盖已存在的文件
    """
    source_root = Path(source_root).resolve()
    dest_root = Path(dest_root).resolve()

    if not source_root.is_dir():
        print(f"错误：源文件夹不存在或不是目录 - {source_root}")
        return

    # 创建目标根目录（如果不存在）
    dest_root.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    skipped_count = 0

    for current_dir, subdirs, files in os.walk(source_root):
        # 计算当前目录相对于源根目录的路径
        rel_path = Path(current_dir).relative_to(source_root)
        target_dir = dest_root / rel_path

        for file in files:
            file_path = Path(current_dir) / file
            ext = file_path.suffix.lower()
            if ext not in video_exts:
                continue  # 不是视频文件，跳过

            # 目标文件完整路径
            target_file = target_dir / file

            # 如果目标文件已存在且不允许覆盖，则跳过
            if target_file.exists() and not overwrite:
                print(f"跳过已存在的文件：{target_file}")
                skipped_count += 1
                continue

            # 确保目标目录存在
            target_dir.mkdir(parents=True, exist_ok=True)

            # 拷贝文件（保留元数据）
            try:
                shutil.copy2(file_path, target_file)
                print(f"已拷贝：{file_path} -> {target_file}")
                copied_count += 1
            except Exception as e:
                print(f"拷贝失败 {file_path} -> {target_file}，错误：{e}")

    print(f"\n完成。共拷贝 {copied_count} 个文件，跳过 {skipped_count} 个文件。")


def main():
    parser = argparse.ArgumentParser(description="将文件夹 B 中的视频文件拷贝到结构相同的文件夹 A 中")
    parser.add_argument("-s", "--source", required=True, help="源文件夹（B）")
    parser.add_argument("-d", "--dest", required=True, help="目标文件夹（A）")
    parser.add_argument("--exts", nargs="+", default=None,
                        help="视频文件扩展名列表，如 .mp4 .avi（默认：{}）".format(' '.join(DEFAULT_VIDEO_EXTS)))
    parser.add_argument("--no-overwrite", action="store_true", help="不覆盖目标文件夹中已存在的文件")

    args = parser.parse_args()

    if args.exts:
        video_exts = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in args.exts}
    else:
        video_exts = DEFAULT_VIDEO_EXTS

    overwrite = not args.no_overwrite

    copy_videos(args.source, args.dest, video_exts, overwrite)


if __name__ == "__main__":
    main()
