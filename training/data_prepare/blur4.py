import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Union
import urllib.request
import time
import torch

#from ultralytics import YOLO

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class VideoHandDetector:
    """基于 MediaPipe Tasks API 的视频/摄像头人手检测器，支持裁剪手部居中或背景模糊化，同时可结合 YOLO 保留指定物体清晰"""

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_FILENAME = "hand_landmarker.task"

    # 手部关键点连接关系（21个关键点，仅预览时使用）
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
        (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
        (5, 9), (9, 13), (13, 17)  # 手掌连接
    ]

    def __init__(self, confidence_threshold: float = 0.5, model_path: Optional[str] = None,
                 yolo_weights: Optional[str] = None, yolo_conf: float = 0.5, yolo_imgsz: int = 640):
        """初始化手部检测器，并可选择加载 YOLO 模型"""
        # ========== 原有手部模型初始化 ==========
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

        # 自定义绘制参数（仅预览用）
        self.landmark_color = (0, 255, 0)
        self.landmark_radius = 5
        self.connection_color = (255, 0, 0)
        self.connection_thickness = 2
        # ========== YOLO 模型初始化（新增） ==========
        self.yolo_model = None
        if yolo_weights is not None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(self.device)
            # self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights, force_reload=False)
            self.yolo_model = torch.hub.load('yolov5', 'custom', path=yolo_weights, source='local')
            #self.yolo_model = YOLO(yolo_weights)
            #self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights)
            self.yolo_model.conf = yolo_conf
            self.yolo_model.imgsz = yolo_imgsz
            self.yolo_model.to(self.device)
            print(f"YOLO 模型加载完成，使用设备: {self.device}")

    def _download_model(self, save_path: str):
        """下载官方模型文件"""
        try:
            urllib.request.urlretrieve(self.MODEL_URL, save_path)
            print(f"模型下载完成：{save_path}")
        except Exception as e:
            raise RuntimeError(f"模型下载失败，请手动下载 {self.MODEL_URL} 并放置于当前目录") from e

    def _draw_landmarks(self, image: np.ndarray, landmarks, handedness=None):
        """手动绘制手部关键点和连接线（仅预览使用）"""
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

    # ========== 手部掩码（原有） ==========
    def _get_hand_mask(self, frame: np.ndarray, hand_landmarks_list: List) -> np.ndarray:
        """根据手部关键点生成手部掩码（凸包内部为255，外部为0）"""
        h, w = frame.shape[:2]
        if not hand_landmarks_list:
            return np.zeros((h, w), dtype=np.uint8)

        points = []
        for landmarks in hand_landmarks_list:
            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append([x, y])

        points = np.array(points, dtype=np.int32)
        if len(points) < 3:
            return np.zeros((h, w), dtype=np.uint8)

        hull = cv2.convexHull(points)  # shape: (M, 1, 2)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        return mask

    # ========== YOLO 物体掩码（新增） ==========
    def _get_object_mask(self, frame: np.ndarray, target_ids: Union[int, List[int]]) -> np.ndarray:
        """根据 YOLO 检测结果生成物体掩码（矩形填充）"""
        if self.yolo_model is None:
            print("警告：YOLO 模型未加载，无法生成物体掩码")
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        if isinstance(target_ids, int):
            target_ids = [target_ids]

        results = self.yolo_model(frame)
        detections = results.xyxy[0]  # [x1,y1,x2,y2,conf,class]

        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for *box, conf, cls in detections:
            if int(cls) in target_ids:
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, min(x1, w))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h))
                y2 = max(0, min(y2, h))
                if x1 < x2 and y1 < y2:
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  # 填充矩形
        return mask

    def _get_combined_mask(self, frame: np.ndarray, hand_landmarks_list: List,
                           object_ids: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        """生成联合掩码：手部区域 + 指定物体区域"""
        hand_mask = self._get_hand_mask(frame, hand_landmarks_list)
        if object_ids is None:
            return hand_mask

        obj_mask = self._get_object_mask(frame, object_ids)
        combined = cv2.bitwise_or(hand_mask, obj_mask)
        return combined

    def _apply_blur_with_mask(self, frame: np.ndarray, mask: np.ndarray,
                              margin: int = 50, blur_kernel_size: int = 51,
                              blur_sigma: float = 30.0) -> np.ndarray:
        """根据掩码对图像应用渐变模糊（掩码区域清晰，越远越模糊）"""
        h, w = frame.shape[:2]
        if np.sum(mask) == 0:
            return frame

        dist_input = 255 - mask  # 掩码区域为0，背景为255
        dist = cv2.distanceTransform(dist_input, cv2.DIST_L2, 5)  # 距离变换，掩码内距离为0

        max_dist = dist.max()
        if max_dist <= margin:
            return frame

        d = np.clip((dist - margin) / (max_dist - margin), 0, 1).astype(np.float32)
        alpha = np.power(d, 0.1)  # 非线性增长

        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        blurred = cv2.GaussianBlur(frame, (blur_kernel_size, blur_kernel_size), blur_sigma)

        alpha_3c = np.stack([alpha, alpha, alpha], axis=-1)
        result = (frame * (1 - alpha_3c) + blurred * alpha_3c).astype(np.uint8)
        return result

    # ========== 修改后的背景模糊方法（支持物体ID） ==========
    def blur_background_hand(self, frame: np.ndarray, hand_landmarks_list: List,
                             margin: int = 50, blur_kernel_size: int = 51,
                             blur_sigma: float = 30.0,
                             object_ids: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        """
        根据手部和可选物体生成渐变模糊图像。
        :param object_ids: 需要保留清晰的物体类别 ID（整数或列表），若为 None 则仅保留手部。
        """
        combined_mask = self._get_combined_mask(frame, hand_landmarks_list, object_ids)
        return self._apply_blur_with_mask(frame, combined_mask, margin, blur_kernel_size, blur_sigma)

    # ========== 原有裁剪方法（未修改） ==========
    def crop_and_center_hand(self, frame: np.ndarray, hand_landmarks_list: List, margin: int = 50) -> np.ndarray:
        """将手部区域裁剪并居中放置，周围填充白色"""
        h, w = frame.shape[:2]
        if not hand_landmarks_list:
            return frame

        all_x = []
        all_y = []
        for landmarks in hand_landmarks_list:
            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                all_x.append(x)
                all_y.append(y)

        x_min = max(0, min(all_x) - margin)
        x_max = min(w, max(all_x) + margin)
        y_min = 0
        y_max = h

        cropped = frame[y_min:y_max, x_min:x_max]
        crop_h, crop_w = cropped.shape[:2]

        canvas = np.full_like(frame, 255)
        start_x = (w - crop_w) // 2
        start_y = (h - crop_h) // 2

        if start_x < 0 or start_y < 0:
            cropped = cv2.resize(cropped, (w, h))
            return cropped

        canvas[start_y:start_y + crop_h, start_x:start_x + crop_w] = cropped
        return canvas

    # ========== 修改后的处理函数（增加 object_ids 参数） ==========
    def process_video(self,
                      input_video_path: str,
                      output_video_path: Optional[str] = None,
                      show_preview: bool = False,
                      crop_hand: bool = False,
                      blur_background: bool = False,
                      crop_margin: int = 50,
                      blur_kernel_size: int = 51,
                      blur_sigma: float = 30.0,
                      object_ids: Optional[Union[int, List[int]]] = None):
        """
        处理视频文件，可进行裁剪或背景模糊。
        :param object_ids: 模糊背景时需保留清晰的物体 ID
        """
        if not Path(input_video_path).exists():
            raise FileNotFoundError(f"视频文件不存在: {input_video_path}")

        if crop_hand and blur_background:
            print("同时启用裁剪和模糊，将默认使用背景模糊。")
            crop_hand = False

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

                # 确定写入视频的帧
                if detection_result.hand_landmarks:
                    if blur_background:
                        write_frame = self.blur_background_hand(frame, detection_result.hand_landmarks,
                                                                 margin=crop_margin,
                                                                 blur_kernel_size=blur_kernel_size,
                                                                 blur_sigma=blur_sigma,
                                                                 object_ids=object_ids)
                    elif crop_hand:
                        write_frame = self.crop_and_center_hand(frame, detection_result.hand_landmarks,
                                                                 margin=crop_margin)
                    else:
                        write_frame = frame.copy()
                else:
                    write_frame = frame.copy()  # 无手时保持原图

                # 预览（如需）
                if show_preview:
                    preview_frame = write_frame.copy()
                    if detection_result.hand_landmarks:
                        num_hands = len(detection_result.hand_landmarks)
                        for hand_landmarks, handedness in zip(
                                detection_result.hand_landmarks,
                                detection_result.handedness):
                            self._draw_landmarks(preview_frame, hand_landmarks, handedness)
                        status_text = f"Frame: {frame_count}/{total_frames} | Hands: {num_hands}"
                        cv2.putText(preview_frame, status_text,
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    else:
                        cv2.putText(preview_frame,
                                    f"Frame: {frame_count}/{total_frames} | No hands detected",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    progress = frame_count / total_frames * 100
                    progress_text = f"Progress: {progress:.1f}%"
                    cv2.putText(preview_frame, progress_text,
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    cv2.imshow('Hand Detection - Preview', preview_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("用户中断处理")
                        break

                if out_writer:
                    out_writer.write(write_frame)

                if detection_result.hand_landmarks:
                    stats['frames_with_hands'] += 1
                    has_hand_in_video = True
                    num_hands = len(detection_result.hand_landmarks)
                    stats['max_hands_detected'] = max(stats['max_hands_detected'], num_hands)

                if frame_count % 50 == 0:
                    print(f"已处理 {frame_count}/{total_frames} 帧 ({frame_count/total_frames*100:.1f}%)")

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

    # ========== 拆分视频（增加 object_ids） ==========
    def split_video_by_hand(self,
                            input_video_path: str,
                            output_dir: str,
                            no_hand_threshold: int = 3,
                            show_preview: bool = False,
                            crop_hand: bool = False,
                            blur_background: bool = False,
                            crop_margin: int = 50,
                            blur_kernel_size: int = 51,
                            blur_sigma: float = 30.0,
                            object_ids: Optional[Union[int, List[int]]] = None):
        """将视频按手部出现情况拆分为多个片段，可进行裁剪或背景模糊"""
        if not Path(input_video_path).exists():
            raise FileNotFoundError(f"视频文件不存在: {input_video_path}")

        if crop_hand and blur_background:
            print("同时启用裁剪和模糊，将默认使用背景模糊。")
            crop_hand = False

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
        if blur_background:
            print(f"背景模糊已启用，边距: {crop_margin}, 模糊核: {blur_kernel_size}, sigma: {blur_sigma}")
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

                # 确定写入帧
                if has_hand:
                    if blur_background:
                        write_frame = self.blur_background_hand(frame, detection_result.hand_landmarks,
                                                                 margin=crop_margin,
                                                                 blur_kernel_size=blur_kernel_size,
                                                                 blur_sigma=blur_sigma,
                                                                 object_ids=object_ids)
                    elif crop_hand:
                        write_frame = self.crop_and_center_hand(frame, detection_result.hand_landmarks,
                                                                 margin=crop_margin)
                    else:
                        write_frame = frame.copy()
                else:
                    write_frame = frame.copy()

                # 预览（如需）
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
                    cv2.imshow('Split Preview', preview_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("用户中断")
                        break

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

    # ========== 摄像头实时流（增加 object_ids） ==========
    def process_camera(self,
                       camera_id: int = 0,
                       output_video_path: Optional[str] = None,
                       show_preview: bool = True,
                       crop_hand: bool = False,
                       blur_background: bool = False,
                       crop_margin: int = 50,
                       blur_kernel_size: int = 51,
                       blur_sigma: float = 30.0,
                       object_ids: Optional[Union[int, List[int]]] = None):
        """处理摄像头实时视频流，可裁剪或背景模糊"""
        if crop_hand and blur_background:
            print("同时启用裁剪和模糊，将默认使用背景模糊。")
            crop_hand = False

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

                # 确定写入帧
                if detection_result.hand_landmarks:
                    if blur_background:
                        write_frame = self.blur_background_hand(frame, detection_result.hand_landmarks,
                                                                 margin=crop_margin,
                                                                 blur_kernel_size=blur_kernel_size,
                                                                 blur_sigma=blur_sigma,
                                                                 object_ids=object_ids)
                    elif crop_hand:
                        write_frame = self.crop_and_center_hand(frame, detection_result.hand_landmarks,
                                                                 margin=crop_margin)
                    else:
                        write_frame = frame.copy()
                else:
                    write_frame = frame.copy()

                # 预览（如需）
                if show_preview:
                    preview_frame = write_frame.copy()
                    if detection_result.hand_landmarks:
                        num_hands = len(detection_result.hand_landmarks)
                        for hand_landmarks, handedness in zip(
                                detection_result.hand_landmarks,
                                detection_result.handedness):
                            self._draw_landmarks(preview_frame, hand_landmarks, handedness)
                        status_text = f"Hands: {num_hands}"
                    else:
                        status_text = "No hands"
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    cv2.putText(preview_frame, f"FPS: {current_fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(preview_frame, status_text, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if detection_result.hand_landmarks else (0, 0, 255), 2)
                    cv2.imshow('Camera Hand Detection', preview_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("用户退出")
                        break

                if out_writer:
                    out_writer.write(write_frame)

        except KeyboardInterrupt:
            print("处理被中断")
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            cv2.destroyAllWindows()
            self.detector.close()
            print(f"摄像头处理结束，共处理 {frame_count} 帧")

    # ========== 摄像头拆分（增加 object_ids） ==========
    def split_camera_stream(self,
                            camera_id: int = 0,
                            output_dir: str = "demo_train",
                            no_hand_threshold: int = 3,
                            show_preview: bool = True,
                            crop_hand: bool = False,
                            blur_background: bool = False,
                            crop_margin: int = 50,
                            blur_kernel_size: int = 51,
                            blur_sigma: float = 30.0,
                            object_ids: Optional[Union[int, List[int]]] = None):
        """实时从摄像头捕获视频流，按手部出现情况拆分保存，可裁剪或背景模糊"""
        if crop_hand and blur_background:
            print("同时启用裁剪和模糊，将默认使用背景模糊。")
            crop_hand = False

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
        if blur_background:
            print(f"背景模糊已启用，边距: {crop_margin}, 模糊核: {blur_kernel_size}, sigma: {blur_sigma}")

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

                # 确定写入帧
                if has_hand:
                    if blur_background:
                        write_frame = self.blur_background_hand(frame, detection_result.hand_landmarks,
                                                                 margin=crop_margin,
                                                                 blur_kernel_size=blur_kernel_size,
                                                                 blur_sigma=blur_sigma,
                                                                 object_ids=object_ids)
                    elif crop_hand:
                        write_frame = self.crop_and_center_hand(frame, detection_result.hand_landmarks,
                                                                 margin=crop_margin)
                    else:
                        write_frame = frame.copy()
                else:
                    write_frame = frame.copy()

                # 预览（如需）
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
                    cv2.imshow('Camera Split', preview_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("用户退出")
                        break

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
            print(f"处理后的视频已保存至: {output_path}")
        print("=" * 50)


# ====== 主程序示例（增加 --object_ids 参数） ======
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="视频手部检测与处理工具（裁剪或背景模糊）")
    parser.add_argument("--mode", choices=["camera", "video"], default="camera",
                        help="运行模式：camera（摄像头实时）或 video（处理现有视频）")
    parser.add_argument("--input", type=str, default=None,
                        help="输入视频文件路径（mode=video时必需）")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="输出目录（默认 output）")
    parser.add_argument("--no_hand_threshold", type=int, default=30,
                        help="连续无手帧数阈值，用于结束片段（默认30，约1秒@30fps）")
    parser.add_argument("--crop", action="store_true",
                        help="启用手部区域裁剪居中（与 --blur 互斥，若同时启用则默认使用 blur）")
    parser.add_argument("--blur", action="store_true",
                        help="启用手部区域保留、背景模糊（与 --crop 互斥，若同时启用则默认使用 blur）")
    parser.add_argument("--margin", type=int, default=50,
                        help="手部包围盒边距（像素，默认50）")
    parser.add_argument("--blur_kernel", type=int, default=51,
                        help="背景模糊核大小（奇数，默认51）")
    parser.add_argument("--blur_sigma", type=float, default=30.0,
                        help="背景模糊标准差（默认30.0）")
    parser.add_argument("--preview", action="store_true",
                        help="显示实时预览窗口（会绘制骨架，不影响输出视频）")
    # ========== 新增 YOLO 相关参数 ==========
    parser.add_argument("--yolo_weights", type=str, default=None,
                        help="YOLO 模型权重文件路径（如 best.pt），启用后可保留指定物体清晰")
    parser.add_argument("--object_ids", type=str, default=None,
                        help="需要保留清晰的物体类别 ID，逗号分隔（例如 '0,1' 表示人、自行车）")
    parser.add_argument("--yolo_conf", type=float, default=0.5,
                        help="YOLO 置信度阈值（默认0.5）")
    parser.add_argument("--yolo_imgsz", type=int, default=640,
                        help="YOLO 推理图像尺寸（默认640）")
    args = parser.parse_args()

    # 解析物体 ID 列表
    object_ids = None
    if args.object_ids is not None:
        object_ids = [int(id_str) for id_str in args.object_ids.split(',')]

    # 初始化检测器，传入 YOLO 参数
    detector = VideoHandDetector(
        confidence_threshold=0.5,
        yolo_weights=args.yolo_weights,
        yolo_conf=args.yolo_conf,
        yolo_imgsz=args.yolo_imgsz
    )

    # 处理互斥选项
    crop = args.crop
    blur = args.blur

    if args.mode == "camera":
        detector.split_camera_stream(
            camera_id=0,
            output_dir=args.output_dir,
            no_hand_threshold=args.no_hand_threshold,
            show_preview=args.preview,
            crop_hand=crop,
            blur_background=blur,
            crop_margin=args.margin,
            blur_kernel_size=args.blur_kernel,
            blur_sigma=args.blur_sigma,
            object_ids=object_ids
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
            crop_hand=crop,
            blur_background=blur,
            crop_margin=args.margin,
            blur_kernel_size=args.blur_kernel,
            blur_sigma=args.blur_sigma,
            object_ids=object_ids
        )
