import threading
import queue
import time
import collections
import os
import json
from vosk import Model, KaldiRecognizer
import pyaudio
import numpy as np
import cv2
from pathlib import Path
import mediapipe as mp
import subprocess

# 使用 blur4 中的手部检测器（支持保留多个物体）
from blur4 import VideoHandDetector
import video_videomae_classify as vc   # VideoMAE 分类模块

# ==================== 正确序列与物体映射 ====================
CORRECT_SEQUENCE = ["0", "1", "2", "3", "4"]          # 预设的正确动作序列（与 class_names 中的标签对应）
ACTION_TO_OBJECT_ID = {
    "0": [2],   # 动作 "0" 对应保留物体类别 2（如 person）
    "1": [1],   # 动作 "1" 对应保留物体类别 1（如 bicycle）
    "2": [0],
    "3": [2],
    "4": [2]
}
DEFAULT_OBJECT_IDS = []   # 无额外物体，仅保留手部

def compare_sequence(history, correct):
    """比较历史动作与正确序列，返回 (匹配数, 错误索引, 错误动作, 正确动作)"""
    if not correct:
        return 0, None, None, None
    match_len = 0
    for i, (h, c) in enumerate(zip(history, correct)):
        if h == c:
            match_len += 1
        else:
            return match_len, i, h, c
    if len(history) >= len(correct):
        return len(correct), None, None, None
    else:
        return len(history), len(history), None, correct[len(history)]

def get_expected_object_ids(history_actions):
    """根据历史动作序列，返回当前应保留的物体 ID 列表"""
    match_len, err_idx, err_action, expected = compare_sequence(history_actions, CORRECT_SEQUENCE)
    if err_idx is not None:
        # 若存在错误，期望动作是错误位置对应的正确动作（用于纠错指导）
        expected_action = expected
    else:
        if match_len < len(CORRECT_SEQUENCE):
            expected_action = CORRECT_SEQUENCE[match_len]
        else:
            expected_action = None
    if expected_action is not None:
        obj_ids = ACTION_TO_OBJECT_ID.get(expected_action, [])
        if not isinstance(obj_ids, list):
            obj_ids = [obj_ids]
        return obj_ids
    else:
        return DEFAULT_OBJECT_IDS

def get_expected_next_action(history_actions):
    """根据历史动作（仅包含正确的动作）返回期望的下一个正确动作，若已全部完成则返回 None"""
    match_len, err_idx, err_action, expected = compare_sequence(history_actions, CORRECT_SEQUENCE)
    if err_idx is not None:
        # 理论上 history 中不会有错误动作，此分支不会触发，但保留防御
        return expected
    if match_len < len(CORRECT_SEQUENCE):
        return CORRECT_SEQUENCE[match_len]
    else:
        return None

# ==================== 语音朗读 ====================
def speak_chinese(text):
    """使用 espeak 朗读中文"""
    subprocess.run(['espeak', '-v', 'zh', text])

# ==================== 自定义背景模糊函数（复用 blur4 的实现） ====================
def apply_blur_with_mask(frame: np.ndarray, mask: np.ndarray,
                         margin: int = 50, blur_kernel_size: int = 51,
                         blur_sigma: float = 30.0) -> np.ndarray:
    """
    根据掩码对图像应用渐变模糊（掩码区域清晰，越远越模糊）
    """
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

# ==================== 摄像头线程（支持动态物体保留，仅每5帧执行一次完整检测） ====================
def camera_thread(segment_queue, stop_event, expected_object_ids_shared, lock,
                  camera_id=0, out_dir='segments', no_hand_threshold=15,
                  enable_blur=True, blur_margin=50, blur_kernel=51, blur_sigma=30.0,
                  detection_interval=5):
    """
    实时检测手部并保存视频片段，支持背景模糊（保留手部 + 当前期望物体）。
    每隔 detection_interval 帧执行一次完整检测（手部关键点 + YOLO物体检测），
    其余帧复用上一次的联合掩码进行模糊，以降低计算负载。
    """
    detector = VideoHandDetector()   # blur4 中的检测器
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_id}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    in_segment = False
    no_hand_count = 0
    segment_idx = 1
    writer = None
    seg_path = None

    # 用于缓存上一次检测的结果（联合掩码、手部存在标志）
    last_combined_mask = None
    last_has_hand = False
    frame_counter = 0

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            timestamp_ms = int(time.time() * 1000)

            # 每 detection_interval 帧执行一次完整检测
            if frame_counter % detection_interval == 0:
                # 完整检测：手部关键点 + YOLO（如果需要）
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                try:
                    detection_result = detector.detector.detect_for_video(mp_image, timestamp_ms)
                except Exception:
                    detection_result = type('R', (), {'hand_landmarks': []})()

                has_hand = len(detection_result.hand_landmarks) > 0
                last_has_hand = has_hand

                if enable_blur and has_hand:
                    # 获取当前期望的物体 ID（线程安全）
                    with lock:
                        obj_ids = expected_object_ids_shared[0]

                    # 生成联合掩码（手部 + 期望物体）
                    combined_mask = detector._get_combined_mask(frame, detection_result.hand_landmarks, obj_ids)
                    last_combined_mask = combined_mask

                    # 应用模糊
                    processed_frame = apply_blur_with_mask(frame, combined_mask,
                                                           margin=blur_margin,
                                                           blur_kernel_size=blur_kernel,
                                                           blur_sigma=blur_sigma)
                else:
                    # 不启用模糊或无手：直接使用原始帧，并清除缓存的掩码
                    processed_frame = frame
                    last_combined_mask = None
            else:
                # 非检测帧：使用上一次的检测结果
                has_hand = last_has_hand   # 复用上一次的手部存在标志
                if enable_blur and has_hand and last_combined_mask is not None:
                    # 使用上一次的联合掩码对当前帧进行模糊（区域可能略有偏移，但计算成本极低）
                    processed_frame = apply_blur_with_mask(frame, last_combined_mask,
                                                           margin=blur_margin,
                                                           blur_kernel_size=blur_kernel,
                                                           blur_sigma=blur_sigma)
                else:
                    processed_frame = frame

            # 片段录制逻辑（基于当前的 has_hand，非检测帧为复用值）
            if has_hand:
                if not in_segment:
                    seg_path = os.path.join(out_dir, f"segment_{int(time.time())}_{segment_idx}.mp4")
                    writer = cv2.VideoWriter(seg_path, fourcc, fps, (width, height))
                    if not writer.isOpened():
                        print(f"无法创建分段输出: {seg_path}")
                        writer = None
                    else:
                        in_segment = True
                        no_hand_count = 0
                        segment_idx += 1
                        print(f"开始新片段: {seg_path}")

                if writer is not None:
                    writer.write(processed_frame)
                no_hand_count = 0
            else:
                if in_segment:
                    no_hand_count += 1
                    if writer is not None:
                        writer.write(processed_frame)
                    if no_hand_count >= no_hand_threshold:
                        if writer is not None:
                            writer.release()
                        in_segment = False
                        no_hand_count = 0
                        print(f"结束片段, 放入队列进行分类: {seg_path}")
                        segment_queue.put(seg_path)

            # 预览窗口（显示原始帧，便于观察）
            cv2.imshow('Hand Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    except KeyboardInterrupt:
        stop_event.set()
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()

# ==================== 音频线程（保持不变） ====================
def audio_thread(command_queue, stop_event, device_index=32, energy_threshold=200,
                 silence_duration=0.5, model_path="vosk-model-small-cn-0.22"):
    """连续监听麦克风，使用Vosk将语音识别为文字，并将完整语句放入command_queue"""
    try:
        model = Model(model_path)
    except Exception as e:
        print(f"加载Vosk模型失败: {e}")
        return

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000
    CHUNK = 1024
    chunk_duration = CHUNK / RATE
    max_silent_chunks = int(silence_duration / chunk_duration) if silence_duration > 0 else 0

    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"无法打开音频设备 {device_index}: {e}")
        p.terminate()
        return

    recognizer = KaldiRecognizer(model, RATE)
    is_speaking = False
    silent_chunks = 0
    speech_chunks = 0

    print(f"音频线程启动，设备索引 {device_index}，能量阈值 {energy_threshold}，静音阈值 {silence_duration}秒")
    try:
        while not stop_event.is_set():
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except Exception as e:
                print(f"音频读取错误: {e}")
                break

            audio_data = np.frombuffer(data, dtype=np.int16)
            energy = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))

            if energy > energy_threshold:
                if not is_speaking:
                    is_speaking = True
                    speech_chunks = 0
                    recognizer.Reset()
                    print("\n[语音开始]")
                recognizer.AcceptWaveform(data)
                speech_chunks += 1
                silent_chunks = 0
            else:
                if is_speaking:
                    silent_chunks += 1
                    recognizer.AcceptWaveform(data)
                    if silent_chunks > max_silent_chunks and speech_chunks > 5:
                        result = json.loads(recognizer.FinalResult())
                        text = result.get('text', '').strip()
                        if text:
                            command_queue.put((text, time.time()))
                            print(f"\n[语句结束] 识别结果: {text}")
                        else:
                            print("\n[语句结束] 未识别出文字")
                        is_speaking = False
                        speech_chunks = 0
                        silent_chunks = 0

            if is_speaking and speech_chunks % 10 == 0:
                partial = json.loads(recognizer.PartialResult())
                partial_text = partial.get('partial', '').strip()
                if partial_text:
                    import sys
                    sys.stdout.write(f"\r[实时] {partial_text}")
                    sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        if is_speaking:
            result = json.loads(recognizer.FinalResult())
            text = result.get('text', '').strip()
            if text:
                command_queue.put((text, time.time()))
                print(f"\n[最终] 识别结果: {text}")
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("音频线程已停止")

# ==================== 分类线程（VideoMAE） ====================
def classifier_thread(segment_queue, action_queue, stop_event, model):
    """使用 VideoMAE 模型对视频片段进行分类"""
    if model is None:
        print('VideoMAE模型未加载，分类线程退出')
        return

    while not stop_event.is_set():
        try:
            seg_path = segment_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if not os.path.exists(seg_path):
            print(f"片段不存在: {seg_path}")
            continue

        print(f"开始分类: {seg_path}")
        try:
            pred_idx, confidence, _ = vc.predict_video(model, seg_path, vc.class_names, top_k=1)
            label = vc.class_names[pred_idx] if pred_idx < len(vc.class_names) else str(pred_idx)
            ts = time.time()
            action_queue.put((label, ts))
            print(f"{seg_path} -> {label} 置信度: {confidence:.4f}")
        except Exception as e:
            print(f"分类出错: {e}")

# ==================== 增强的反馈生成 ====================
def generate_feedback(command, action_sequence, cmd_ts):
    """
    根据语音指令和动作序列生成反馈，支持纠错、指导、查询历史等。
    action_sequence 是 collections.deque，元素为 (action, timestamp)，仅包含正确的动作。
    """
    history_actions = [act for act, _ in action_sequence]
    match_len, err_idx, err_action, expected = compare_sequence(history_actions, CORRECT_SEQUENCE)

    if "重复" in command:
        if history_actions:
            last_action = history_actions[-1]
            msg = f"正在重复动作: {last_action}"
            speak_chinese(msg)
            return msg
        else:
            msg = "还没有任何动作记录，请先做出动作。"
            speak_chinese(msg)
            return msg

    elif "历史" in command:
        if history_actions:
            msg = f"最近的动作有: {' -> '.join(history_actions)}"
            speak_chinese(msg)
            return msg
        else:
            msg = "暂无历史动作。"
            speak_chinese(msg)
            return msg

    elif "下一步" in command:
        if match_len >= len(CORRECT_SEQUENCE):
            msg = "恭喜！您已完成所有正确动作。"
            speak_chinese(msg)
            return msg
        if err_idx is not None:
            msg = f"检测到动作错误：第{err_idx+1}步应为'{expected}'，您做了'{err_action}'。请纠正后再继续。"
            speak_chinese(msg)
            return msg
        next_action = CORRECT_SEQUENCE[match_len]
        msg = f"下一步动作是：{next_action}"
        speak_chinese(msg)
        return msg

    else:
        # 通用反馈：若有错误则提醒，否则显示进度
        if err_idx is not None:
            msg = f"提醒：第{err_idx+1}步动作'{err_action}'不正确，应为'{expected}'。"
        else:
            msg = f"收到指令: {command}，当前已完成 {match_len}/{len(CORRECT_SEQUENCE)} 个正确动作。"
        speak_chinese(msg)
        return msg

# ==================== 主线程 ====================
def main():
    # ========== 配置参数 ==========
    MODEL_PATH = "videomae_best_blur4.pth"   # VideoMAE 模型文件
    ENABLE_BLUR = True                       # 是否启用背景模糊
    BLUR_MARGIN = 50                         # 手部包围盒边距
    BLUR_KERNEL = 51                         # 模糊核大小（奇数）
    BLUR_SIGMA = 30.0                        # 模糊标准差
    NO_HAND_THRESHOLD = 15                   # 连续无手帧数阈值（结束片段）
    DETECTION_INTERVAL = 5                   # 每隔多少帧执行一次完整检测（手部+YOLO）
    # =============================

    # 队列与事件
    segment_queue = queue.Queue()
    command_queue = queue.Queue()
    action_queue = queue.Queue()
    stop_event = threading.Event()

    # 共享变量：当前期望保留的物体 ID 列表（用锁保护）
    expected_object_ids = [DEFAULT_OBJECT_IDS]
    lock = threading.Lock()

    # 加载 VideoMAE 模型
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_PATH)
    model = None
    try:
        print('加载 VideoMAE 模型...')
        model = vc.load_model(model_path, len(vc.class_names), vc.device)
        print('模型加载成功')
    except Exception as e:
        print('加载模型失败:', e)
        # 模型加载失败时分类线程不会真正工作，其他线程继续

    # 启动摄像头线程（传入共享变量和锁，增加检测间隔参数）
    v_thread = threading.Thread(
        target=camera_thread,
        args=(segment_queue, stop_event, expected_object_ids, lock),
        kwargs={
            'camera_id': 0,
            'out_dir': 'segments',
            'no_hand_threshold': NO_HAND_THRESHOLD,
            'enable_blur': ENABLE_BLUR,
            'blur_margin': BLUR_MARGIN,
            'blur_kernel': BLUR_KERNEL,
            'blur_sigma': BLUR_SIGMA,
            'detection_interval': DETECTION_INTERVAL
        }
    )

    # 启动音频线程
    #a_thread = threading.Thread(target=audio_thread, args=(command_queue, stop_event))

    # 启动分类线程（仅当模型加载成功时才真正工作）
    c_thread = threading.Thread(target=classifier_thread, args=(segment_queue, action_queue, stop_event, model))

    v_thread.start()
    #a_thread.start()
    c_thread.start()

    # 动作序列：仅存储正确的动作 (action, timestamp)
    action_sequence = collections.deque(maxlen=100)

    try:
        while not stop_event.is_set():
            # 处理分类结果
            try:
                action, ts = action_queue.get_nowait()
                print(f"分类到动作: {action} @ {ts}")

                # 获取当前期望的正确动作
                history_actions = [act for act, _ in action_sequence]
                expected = get_expected_next_action(history_actions)

                if expected is None:
                    # 已完成全部正确动作，理论上应已清空序列，此处防御
                    print("所有动作已完成并重置，请重新开始。")
                    action_sequence.clear()
                    expected = CORRECT_SEQUENCE[0]

                if action == expected:
                    # 动作正确，加入历史
                    action_sequence.append((action, ts))
                    print(f"✅ 动作正确: {action}")

                    # 检查是否完成整个正确序列
                    if len(action_sequence) == len(CORRECT_SEQUENCE):
                        print("🎉 恭喜！已完成全部正确动作序列！序列已清空，可重新开始。")
                        speak_chinese("恭喜！已完成全部正确动作序列！")
                        action_sequence.clear()
                else:
                    # 动作错误，不加入历史，并给出反馈
                    error_msg = f"❌ 动作错误：应该做 '{expected}'，实际做了 '{action}'"
                    print(error_msg)
                    speak_chinese(f"动作错误，应该做 {expected}")

                # 无论正确与否，都基于当前正确的历史序列更新期望物体 ID
                current_history = [act for act, _ in action_sequence]
                new_obj_ids = get_expected_object_ids(current_history)
                with lock:
                    expected_object_ids[0] = new_obj_ids
                    print(f"当前期望物体 ID: {new_obj_ids}")

            except queue.Empty:
                pass

            # 处理语音指令
            #try:
            #    command, cmd_ts = command_queue.get_nowait()
            #    feedback = generate_feedback(command, action_sequence, cmd_ts)
            #    print(f"反馈: {feedback}")
            #except queue.Empty:
            #    pass

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("程序终止中...")
        stop_event.set()
        v_thread.join()
        #a_thread.join()
        c_thread.join()

if __name__ == "__main__":
    main()
