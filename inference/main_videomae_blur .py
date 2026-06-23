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

# 使用 blur1 中的手部检测器（包含背景模糊方法）
from blur1 import VideoHandDetector
import video_videomae_classify as vc   # 原始的3DCNN分类模块

# ---------- 摄像头线程（支持背景模糊）----------

def speak_chinese(text):
    # 使用 espeak 朗读中文，-v zh 指定中文语音
    subprocess.run(['espeak', '-v', 'zh', text])

def camera_thread(segment_queue, stop_event, camera_id=0, out_dir='segments',
                  no_hand_threshold=15, enable_blur=False, blur_margin=50,
                  blur_kernel=51, blur_sigma=30.0):
    """
    实时检测手部并保存视频片段，可选择对帧进行背景模糊处理。
    当 enable_blur=True 且检测到手时，调用 blur_background_hand 处理当前帧。
    """
    detector = VideoHandDetector()
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

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(time.time() * 1000)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            try:
                detection_result = detector.detector.detect_for_video(mp_image, timestamp_ms)
            except Exception:
                detection_result = type('R', (), {'hand_landmarks': []})()

            has_hand = len(detection_result.hand_landmarks) > 0

            # ---------- 背景模糊处理 ----------
            if enable_blur and has_hand:
                processed_frame = detector.blur_background_hand(
                    frame,
                    detection_result.hand_landmarks,
                    margin=blur_margin,
                    blur_kernel_size=blur_kernel,
                    blur_sigma=blur_sigma
                )
            else:
                processed_frame = frame  # 不模糊时保持原帧
            # ---------------------------------

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

            # 预览窗口（可选）
            cv2.imshow('Hand Stream', frame)  # 预览原图（不影响写入）
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

# ---------- 音频线程（不变）----------
# ---------- 音频线程（基于audio_module_vosk_stream.py改进）----------
def audio_thread(command_queue, stop_event, device_index=32, energy_threshold=200,
                 silence_duration=0.5, model_path="vosk-model-small-cn-0.22"):
    """
    连续监听麦克风，使用Vosk将语音识别为文字，并将完整语句放入command_queue。
    基于能量阈值检测人声，静音超时后输出识别结果。
    """
    try:
        model = Model(model_path)
    except Exception as e:
        print(f"加载Vosk模型失败: {e}")
        return

    # 音频参数
    FORMAT = pyaudio.paInt16
    CHANNELS = 1          # 单声道
    RATE = 48000          # Vosk模型通常使用16kHz
    CHUNK = 1024
    chunk_duration = CHUNK / RATE
    max_silent_chunks = int(silence_duration / chunk_duration) if silence_duration > 0 else 0

    # 初始化PyAudio
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

            # 计算RMS能量
            audio_data = np.frombuffer(data, dtype=np.int16)
            energy = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))

            if energy > energy_threshold:
                # 检测到声音
                if not is_speaking:
                    # 新的一句话开始
                    is_speaking = True
                    speech_chunks = 0
                    recognizer.Reset()
                    print("\n[语音开始]")
                recognizer.AcceptWaveform(data)
                speech_chunks += 1
                silent_chunks = 0
            else:
                # 静音
                if is_speaking:
                    # 当前在语音中，累积静音
                    silent_chunks += 1
                    recognizer.AcceptWaveform(data)   # 保留句尾停顿
                    if silent_chunks > max_silent_chunks and speech_chunks > 5:
                        # 静音超时，语句结束
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

            # 可选：打印部分识别结果（每10个语音块）
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
        # 处理最后未结束的语音
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
# ---------- 分类线程（适配原始3DCNN_classify）----------
def classifier_thread(segment_queue, action_queue, stop_event, model):
    """
    使用原始的videomae模型对视频片段进行分类。
    参数 model 是由 vc.load_model() 加载的模型对象。
    """
    if model is None:
        print('videomae模型未加载，分类线程退出')
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

# ---------- 主线程 ----------
def main():
    # ========== 配置参数 ==========
    MODEL_PATH = "videomae_best_blur4.pth"   # videomae 模型文件
    ENABLE_BLUR = True                   # 是否启用背景模糊
    BLUR_MARGIN = 50                      # 手部包围盒边距
    BLUR_KERNEL = 51                      # 模糊核大小（奇数）
    BLUR_SIGMA = 30.0                     # 模糊标准差
    NO_HAND_THRESHOLD = 15                # 连续无手帧数阈值（结束片段）
    # =============================

    # 队列
    segment_queue = queue.Queue()
    command_queue = queue.Queue()
    action_queue = queue.Queue()
    stop_event = threading.Event()

    # 加载3D CNN模型
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_PATH)
    model = None
    try:
        print('加载videomae模型...')
        model = vc.load_model(model_path, len(vc.class_names), vc.device)
        print('模型加载成功')
    except Exception as e:
        print('加载模型失败:', e)
        # 模型加载失败时分类线程不会启动，其他线程继续

    # 启动摄像头线程（传入模糊参数）
    v_thread = threading.Thread(
        target=camera_thread,
        args=(segment_queue, stop_event),
        kwargs={
            'camera_id': 0,
            'out_dir': 'segments',
            'no_hand_threshold': NO_HAND_THRESHOLD,
            'enable_blur': ENABLE_BLUR,
            'blur_margin': BLUR_MARGIN,
            'blur_kernel': BLUR_KERNEL,
            'blur_sigma': BLUR_SIGMA
        }
    )

    # 启动音频线程
    a_thread = threading.Thread(target=audio_thread, args=(command_queue, stop_event))

    # 启动分类线程（仅当模型加载成功时才真正工作）
    c_thread = threading.Thread(target=classifier_thread, args=(segment_queue, action_queue, stop_event, model))

    v_thread.start()
    a_thread.start()
    c_thread.start()

    # 动作序列（滑动窗口）
    action_sequence = collections.deque(maxlen=100)

    try:
        while not stop_event.is_set():
            # 获取分类结果
            try:
                action, ts = action_queue.get_nowait()
                action_sequence.append((action, ts))
                print(f"分类到动作: {action} @ {ts}")
            except queue.Empty:
                pass

            # 处理语音指令
            try:
                command, cmd_ts = command_queue.get_nowait()
                feedback = generate_feedback(command, action_sequence, cmd_ts)
                print(f"反馈: {feedback}")
            except queue.Empty:
                pass

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("程序终止中...")
        stop_event.set()
        v_thread.join()
        a_thread.join()
        c_thread.join()

def generate_feedback(command, action_sequence, cmd_ts):
    """根据指令和动作序列生成反馈（示例逻辑）"""
    if "重复" in command and action_sequence:
        last_action = action_sequence[-1][0]
        speak_chinese(f"正在重复动作: {last_action}")
        return f"正在重复动作: {last_action}"
    elif "历史" in command:
        actions = [a for a, _ in action_sequence]
        speak_chinese(f"最近的动作有: {', '.join(actions)}")
        return f"最近的动作有: {', '.join(actions)}"
    else:
        speak_chinese(f"收到指令: {command}，当前动作序列长度: {len(action_sequence)}")
        return f"收到指令: {command}，当前动作序列长度: {len(action_sequence)}"

if __name__ == "__main__":
    main()
