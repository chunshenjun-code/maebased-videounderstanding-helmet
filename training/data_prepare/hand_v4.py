"""
视频手部检测与处理工具
- 支持摄像头实时流和现有视频文件
- 可检测手部并裁剪手部区域居中放置（周围填充白色）
- 可拆分视频片段（基于手部出现连续阈值）
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Union
import urllib.request
import time

# 新版 MediaPipe Tasks 相关导入
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class VideoHandDetector:
    """基于 MediaPipe Tasks API 的视频/摄像头人手检测器，支持裁剪手部区域居中"""

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_FILENAME = "hand_landmarker.task"

    # 手部关键点连接关系（21个关键点）
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
        (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
        (5, 9), (9, 13), (13, 17)  # 手掌连接
    ]

    def __init__(self, confidence_threshold: float = 0.5, model_path: Optional[str] = None):
        """初始化手部检测器"""
        if model_path is None:
            model_path = self.MODEL_FILENAME
            if not Path(model_path).exists():
                print(f"模型文件 {model_path} 不存在，正在下载...")
                self._download_model(model_path)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=confidence_threshold,
            min_hand_presence_confidence=confidence_threshold,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # 自定义绘制参数
        self.landmark_color = (0, 255, 0)      # 绿色关键点
        self.landmark_radius = 5
        self.connection_color = (255, 0, 0)    # 蓝色连线
        self.connection_thickness = 2

    def _download_model(self, save_path: str):
        """下载官方模型文件"""
        try:
            urllib.request.urlretrieve(self.MODEL_URL, save_path)
            print(f"模型下载完成：{save_path}")
        except Exception as e:
            raise RuntimeError(f"模型下载失败，请手动下载 {self.MODEL_URL} 并放置于当前目录") from e

    def _draw_landmarks(self, image: np.ndarray, landmarks, handedness=None):
        """手动绘制手部关键点和连接线（与之前相同）"""
        h, w, _ = image.shape
        points = []

        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))
            cv2.circle(image, (x, y), self.landmark_radius, self.landmark_color, -1)

        for connection in self.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(image, points[start_idx], points[end_idx],
                         self.connection_color, self.connection_thickness)

        if handedness and len(landmarks) > 0:
            wrist = points[0]
            label = handedness[0].category_name
            score = handedness[0].score
            color = (0, 0, 255) if label == "Left" else (0, 255, 0)
            label_text = f"{label} ({score:.2f})"
            cv2.putText(image, label_text, (wrist[0] - 30, wrist[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ---------- 新增：裁剪并居中手部区域 ----------
    def crop_and_center_hand(self, frame: np.ndarray, hand_landmarks_list: List, margin: int = 50) -> np.ndarray:
        """
        根据手部关键点，将手部区域裁剪并居中放置，周围填充白色。

        Args:
            frame: 原始 BGR 图像
            hand_landmarks_list: 检测到的所有手部的关键点列表（每个元素是 NormalizedLandmark 列表）
            margin: 包围框四周额外保留的像素数

        Returns:
            处理后的图像（与原图尺寸相同，手部区域居中，其余部分为白色）
        """
        h, w = frame.shape[:2]

        # 如果没有检测到手，直接返回原图（但在调用时应确保有手）
        if not hand_landmarks_list:
            return frame

        # 收集所有关键点的像素坐标
        all_x = []
        all_y = []
        for landmarks in hand_landmarks_list:
            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                all_x.append(x)
                all_y.append(y)

        # 计算最小外接矩形并添加边距
        x_min = max(0, min(all_x) - margin)
        x_max = min(w, max(all_x) + margin)
        # x_min = 0
        # x_max = w
        # y_min = max(0, min(all_y) - margin)
        # y_max = min(h, max(all_y) + margin)
        y_min = 0
        y_max = h

        # 裁剪区域
        cropped = frame[y_min:y_max, x_min:x_max]
        crop_h, crop_w = cropped.shape[:2]

        # 创建白色背景画布（与原图同尺寸）
        canvas = np.full_like(frame, 255)  # 白色背景

        # 计算将裁剪区域放置在画布中心的偏移量
        start_x = (w - crop_w) // 2
        start_y = (h - crop_h) // 2

        # 确保起始坐标不越界（若裁剪区域大于画布，理论上不会发生）
        if start_x < 0 or start_y < 0:
            # 如果裁剪区域意外大于画布（例如 margin 过大），则缩放到画布大小
            cropped = cv2.resize(cropped, (w, h))
            return cropped

        # 将裁剪区域复制到画布中心
        canvas[start_y:start_y + crop_h, start_x:start_x + crop_w] = cropped

        return canvas

    # ---------- 处理视频文件 ----------
    def process_video(self,
                      input_video_path: str,
                      output_video_path: Optional[str] = None,
                      show_preview: bool = False,
                      crop_hand: bool = False,       # 新增：是否裁剪手部居中
                      crop_margin: int = 50):        # 新增：裁剪边距
        """处理视频文件，检测人手并生成标注视频，可选择裁剪手部居中"""
        if not Path(input_video_path).exists():
            raise FileNotFoundError(f"视频文件不存在: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频信息: {width}x{height}, {fps}FPS, 共{total_frames}帧")

        out_writer = None
        if output_video_path:
            output_dir = Path(output_video_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            print(f"输出视频将保存至: {output_video_path}")

        stats = {
            'total_frames': total_frames,
            'frames_with_hands': 0,
            'max_hands_detected': 0,
            'hand_detection_percentage': 0.0
        }

        frame_count = 0
        has_hand_in_video = False

        print("开始处理视频...")
        print("按 'q' 键可提前退出处理")
        print("-" * 50)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                timestamp_ms = int(frame_count / fps * 1000)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)

                # 决定要写入/显示的帧（是否裁剪）
                if crop_hand and detection_result.hand_landmarks:
                    display_frame = self.crop_and_center_hand(frame, detection_result.hand_landmarks, margin=crop_margin)
                else:
                    display_frame = frame.copy()

                if detection_result.hand_landmarks:
                    stats['frames_with_hands'] += 1
                    has_hand_in_video = True
                    num_hands = len(detection_result.hand_landmarks)
                    stats['max_hands_detected'] = max(stats['max_hands_detected'], num_hands)

                    for hand_landmarks, handedness in zip(
                            detection_result.hand_landmarks,
                            detection_result.handedness):
                        self._draw_landmarks(display_frame, hand_landmarks, handedness)

                    status_text = f"Frame: {frame_count}/{total_frames} | Hands: {num_hands}"
                    cv2.putText(display_frame, status_text,
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(display_frame,
                                f"Frame: {frame_count}/{total_frames} | No hands detected",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                progress = frame_count / total_frames * 100
                progress_text = f"Progress: {progress:.1f}%"
                cv2.putText(display_frame, progress_text,
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if out_writer:
                    out_writer.write(display_frame)

                if show_preview:
                    cv2.imshow('Hand Detection - Preview', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("用户中断处理")
                        break

                if frame_count % 50 == 0:
                    print(f"已处理 {frame_count}/{total_frames} 帧 ({progress:.1f}%)")

        except KeyboardInterrupt:
            print("处理被中断")
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            cv2.destroyAllWindows()
            self.detector.close()

            stats['hand_detection_percentage'] = (
                stats['frames_with_hands'] / frame_count * 100) if frame_count > 0 else 0
            self._print_summary(stats, has_hand_in_video, frame_count, output_video_path)

        return has_hand_in_video, stats

    def split_video_by_hand(self,
                            input_video_path: str,
                            output_dir: str,
                            no_hand_threshold: int = 3,
                            show_preview: bool = False,
                            crop_hand: bool = False,       # 新增：是否裁剪手部居中
                            crop_margin: int = 50):        # 新增：裁剪边距
        """将视频按手部出现情况拆分为多个片段，可裁剪手部居中"""
        if not Path(input_video_path).exists():
            raise FileNotFoundError(f"视频文件不存在: {input_video_path}")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"视频信息: {width}x{height}, {fps}FPS, 共{total_frames}帧")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        segment_counter = 1
        in_segment = False
        no_hand_count = 0
        current_writer = None
        segment_files = []

        frame_count = 0
        print("开始拆分视频...")
        print(f"连续无手阈值: {no_hand_threshold} 帧")
        if crop_hand:
            print(f"手部裁剪已启用，边距: {crop_margin}")
        print("按 'q' 键可提前退出")
        print("-" * 50)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                timestamp_ms = int(frame_count / fps * 1000)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)

                has_hand = len(detection_result.hand_landmarks) > 0

                # 根据是否有手和裁剪选项，确定要写入/显示的帧
                if has_hand and crop_hand:
                    write_frame = self.crop_and_center_hand(frame, detection_result.hand_landmarks, margin=crop_margin)
                else:
                    write_frame = frame.copy()  # 注意复制，避免后续修改影响

                # 准备预览帧（如需绘制标注，则在 write_frame 上绘制）
                preview_frame = None
                if show_preview:
                    preview_frame = write_frame.copy()
                    if has_hand:
                        for hand_landmarks, handedness in zip(
                                detection_result.hand_landmarks,
                                detection_result.handedness):
                            self._draw_landmarks(preview_frame, hand_landmarks, handedness)
                        cv2.putText(preview_frame, f"Hands: {len(detection_result.hand_landmarks)}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(preview_frame, "No hand", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # 显示当前状态
                    status = "Recording" if in_segment else "Idle"
                    cv2.putText(preview_frame, f"Status: {status}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 片段逻辑
                if has_hand:
                    if not in_segment:
                        out_path = out_dir / f"{segment_counter}.mp4"
                        print(f"开始片段 {segment_counter} -> {out_path}")
                        current_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
                        if not current_writer.isOpened():
                            raise RuntimeError(f"无法创建输出视频: {out_path}")
                        segment_files.append(str(out_path))
                        in_segment = True
                        no_hand_count = 0
                        segment_counter += 1
                    else:
                        no_hand_count = 0

                    if current_writer and in_segment:
                        current_writer.write(write_frame)

                else:  # 无手
                    if in_segment:
                        no_hand_count += 1
                        if current_writer:
                            current_writer.write(write_frame)

                        if no_hand_count >= no_hand_threshold:
                            print(f"连续 {no_hand_threshold} 帧无手，结束片段 {segment_counter - 1}")
                            if current_writer:
                                current_writer.release()
                                current_writer = None
                            in_segment = False
                            no_hand_count = 0

                if show_preview and preview_frame is not None:
                    cv2.imshow('Split Preview', preview_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("用户中断")
                        break

                if frame_count % 100 == 0:
                    progress = frame_count / total_frames * 100
                    print(f"进度: {progress:.1f}% ({frame_count}/{total_frames})")

        except KeyboardInterrupt:
            print("处理被中断")
        finally:
            cap.release()
            if current_writer is not None:
                current_writer.release()
            cv2.destroyAllWindows()
            self.detector.close()

        print(f"\n拆分完成！共生成 {len(segment_files)} 个片段")
        return segment_files

    # ---------- 处理摄像头实时流 ----------
    def process_camera(self,
                       camera_id: int = 0,
                       output_video_path: Optional[str] = None,
                       show_preview: bool = True,
                       crop_hand: bool = False,
                       crop_margin: int = 50):
        """处理摄像头实时视频流，检测人手并实时显示标注，可裁剪手部居中"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头 {camera_id}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"摄像头信息: {width}x{height}, 目标FPS: {fps:.2f}")

        out_writer = None
        if output_video_path:
            output_dir = Path(output_video_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            print(f"实时流将保存至: {output_video_path}")

        print("开始处理摄像头实时流...")
        print("按 'q' 键退出")
        print("-" * 50)

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法获取摄像头帧")
                    break

                frame_count += 1
                timestamp_ms = int(time.time() * 1000)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)

                # 决定要显示的帧
                if crop_hand and detection_result.hand_landmarks:
                    display_frame = self.crop_and_center_hand(frame, detection_result.hand_landmarks, margin=crop_margin)
                else:
                    display_frame = frame.copy()

                # 绘制标注
                if detection_result.hand_landmarks:
                    num_hands = len(detection_result.hand_landmarks)
                    for hand_landmarks, handedness in zip(
                            detection_result.hand_landmarks,
                            detection_result.handedness):
                        self._draw_landmarks(display_frame, hand_landmarks, handedness)

                    status_text = f"Hands: {num_hands}"
                else:
                    status_text = "No hands"

                # 添加实时信息（帧率、帧计数）
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, status_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if detection_result.hand_landmarks else (0, 0, 255), 2)

                if out_writer:
                    out_writer.write(display_frame)

                if show_preview:
                    cv2.imshow('Camera Hand Detection', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("用户退出")
                        break

        except KeyboardInterrupt:
            print("处理被中断")
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            cv2.destroyAllWindows()
            self.detector.close()
            print(f"摄像头处理结束，共处理 {frame_count} 帧")

    def split_camera_stream(self,
                            camera_id: int = 0,
                            output_dir: str = "demo_train",
                            no_hand_threshold: int = 3,
                            show_preview: bool = True,
                            crop_hand: bool = False,
                            crop_margin: int = 50):
        """实时从摄像头捕获视频流，按手部出现情况拆分保存为片段，可裁剪手部居中"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头 {camera_id}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"摄像头信息: {width}x{height}, 目标FPS: {fps:.2f}")
        print(f"连续无手阈值: {no_hand_threshold} 帧")
        if crop_hand:
            print(f"手部裁剪已启用，边距: {crop_margin}")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        segment_counter = 1
        in_segment = False
        no_hand_count = 0
        current_writer = None
        segment_files = []

        frame_count = 0
        print("开始摄像头实时拆分...")
        print("按 'q' 键退出")
        print("-" * 50)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法获取摄像头帧")
                    break

                frame_count += 1
                timestamp_ms = int(time.time() * 1000)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)

                has_hand = len(detection_result.hand_landmarks) > 0

                # 根据是否有手和裁剪选项，确定要写入的帧
                if has_hand and crop_hand:
                    write_frame = self.crop_and_center_hand(frame, detection_result.hand_landmarks, margin=crop_margin)
                else:
                    write_frame = frame.copy()

                # 预览帧（可选绘制标注）
                preview_frame = None
                if show_preview:
                    preview_frame = write_frame.copy()
                    if has_hand:
                        for hand_landmarks, handedness in zip(
                                detection_result.hand_landmarks,
                                detection_result.handedness):
                            self._draw_landmarks(preview_frame, hand_landmarks, handedness)
                        cv2.putText(preview_frame, f"Hands: {len(detection_result.hand_landmarks)}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(preview_frame, "No hand", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    status = "Recording" if in_segment else "Idle"
                    cv2.putText(preview_frame, f"Status: {status}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 片段逻辑
                if has_hand:
                    if not in_segment:
                        out_path = out_dir / f"{segment_counter}.mp4"
                        print(f"开始片段 {segment_counter} -> {out_path}")
                        current_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
                        if not current_writer.isOpened():
                            raise RuntimeError(f"无法创建输出视频: {out_path}")
                        segment_files.append(str(out_path))
                        in_segment = True
                        no_hand_count = 0
                        segment_counter += 1
                    else:
                        no_hand_count = 0

                    if current_writer and in_segment:
                        current_writer.write(write_frame)

                else:  # 无手
                    if in_segment:
                        no_hand_count += 1
                        if current_writer:
                            current_writer.write(write_frame)

                        if no_hand_count >= no_hand_threshold:
                            print(f"连续 {no_hand_threshold} 帧无手，结束片段 {segment_counter - 1}")
                            if current_writer:
                                current_writer.release()
                                current_writer = None
                            in_segment = False
                            no_hand_count = 0

                if show_preview and preview_frame is not None:
                    cv2.imshow('Camera Split', preview_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("用户退出")
                        break

                if frame_count % 100 == 0:
                    print(f"已处理 {frame_count} 帧，当前片段: {segment_counter-1 if in_segment else '无'}")

        except KeyboardInterrupt:
            print("处理被中断")
        finally:
            cap.release()
            if current_writer is not None:
                current_writer.release()
            cv2.destroyAllWindows()
            self.detector.close()

        print(f"\n摄像头流拆分结束，共生成 {len(segment_files)} 个片段")
        return segment_files

    def _print_summary(self, stats: dict, has_hand: bool, processed_frames: int, output_path: Optional[str]):
        """打印处理摘要"""
        print("\n" + "=" * 50)
        print("视频处理完成!")
        print("=" * 50)
        print(f"处理总帧数: {processed_frames}")
        print(f"检测到手部的帧数: {stats['frames_with_hands']}")
        print(f"检测率: {stats['hand_detection_percentage']:.2f}%")
        print(f"最大同时检测手数: {stats['max_hands_detected']}")
        print(f"视频中是否检测到手: {'是' if has_hand else '否'}")
        if output_path:
            print(f"标注视频已保存至: {output_path}")
        print("=" * 50)


# ====== 主程序示例 ======
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="视频手部检测与裁剪工具")
    parser.add_argument("--mode", choices=["camera", "video"], default="camera",
                        help="运行模式：camera（摄像头实时）或 video（处理现有视频）")
    parser.add_argument("--input", type=str, default=None,
                        help="输入视频文件路径（mode=video时必需）")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="输出目录（默认 output）")
    parser.add_argument("--no_hand_threshold", type=int, default=30,
                        help="连续无手帧数阈值，用于结束片段（默认30，约1秒@30fps）")
    parser.add_argument("--crop", action="store_true",
                        help="启用手部区域裁剪居中")
    parser.add_argument("--margin", type=int, default=50,
                        help="裁剪边距（像素，默认50）")
    parser.add_argument("--preview", action="store_true",
                        help="显示实时预览窗口")
    args = parser.parse_args()

    detector = VideoHandDetector(confidence_threshold=0.5)

    if args.mode == "camera":
        detector.split_camera_stream(
            camera_id=0,
            output_dir=args.output_dir,
            no_hand_threshold=args.no_hand_threshold,
            show_preview=args.preview,
            crop_hand=args.crop,
            crop_margin=args.margin
        )
    else:  # video
        if not args.input:
            print("错误：处理视频文件时必须指定 --input")
            exit(1)
        detector.split_video_by_hand(
            input_video_path=args.input,
            output_dir=args.output_dir,
            no_hand_threshold=args.no_hand_threshold,
            show_preview=args.preview,
            crop_hand=args.crop,
            crop_margin=args.margin
        )