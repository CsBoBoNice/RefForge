"""音频抽取与读写（ffmpeg 调用的轻封装）。

本模块只依赖 numpy / soundfile，不导入 torch 等重依赖，保证主流程启动开销低。
"""

import os
import subprocess
import wave

import numpy as np

from io_utils import ensure_dir, ffmpeg_path, ffprobe_path


def extract_audio(video_path, wav_path, sample_rate=16000, start_sec=0.0,
                  duration_sec=None):
    """用 ffmpeg 抽取单声道 16-bit PCM WAV；成功返回 True。"""
    exe = ffmpeg_path()
    if not os.path.isfile(exe):
        return False
    ensure_dir(os.path.dirname(wav_path))
    cmd = [exe, "-y", "-v", "error"]
    if start_sec and start_sec > 0:
        cmd += ["-ss", "%.3f" % float(start_sec)]
    cmd += ["-i", video_path]
    if duration_sec and duration_sec > 0:
        cmd += ["-t", "%.3f" % float(duration_sec)]
    cmd += ["-vn", "-ac", "1", "-ar", str(int(sample_rate)),
            "-c:a", "pcm_s16le", "-f", "wav", wav_path]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    except OSError:
        return False
    return (result.returncode == 0 and os.path.isfile(wav_path)
            and os.path.getsize(wav_path) > 44)


def has_audio_stream(video_path):
    """判断视频是否含音频流；无法判断时返回 None。"""
    exe = ffprobe_path()
    if not os.path.isfile(exe):
        return None
    cmd = [exe, "-v", "error", "-select_streams", "a",
           "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return bool(result.stdout.strip())


def read_wav_mono(path):
    """读取 WAV 为 float32 单声道数组，返回 (data, sample_rate)。

    使用标准库 wave，兼容无 soundfile 时的降级；16-bit PCM 归一化到 [-1, 1]。
    """
    with wave.open(path, "rb") as fh:
        channels = fh.getnchannels()
        rate = fh.getframerate()
        width = fh.getsampwidth()
        raw = fh.readframes(fh.getnframes())
    if width != 2:
        raise ValueError("unsupported sample width: %d" % width)
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), int(rate)


def write_wav(path, data, sample_rate):
    """写出 16-bit PCM 单声道 WAV。"""
    ensure_dir(os.path.dirname(path))
    clipped = np.clip(np.asarray(data, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm.tobytes())
    return path


def cut_clip(audio, sample_rate, start_sec, end_sec):
    """按秒裁剪单声道数组；越界自动收敛，无有效样本时返回空数组。"""
    start = max(0, int(round(float(start_sec) * sample_rate)))
    end = min(len(audio), int(round(float(end_sec) * sample_rate)))
    if end <= start:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(audio[start:end], dtype=np.float32)


def wav_levels(audio):
    """返回 (mean_dbfs, peak_dbfs)；空音频返回 (None, None)。"""
    data = np.asarray(audio, dtype=np.float64)
    if data.size == 0:
        return None, None
    floor = 1.0 / 32768.0
    rms = float(np.sqrt(np.mean(data * data)))
    peak = float(np.max(np.abs(data)))
    if rms <= 0.0 and peak <= 0.0:
        return None, None
    mean_dbfs = 20.0 * np.log10(max(rms, floor))
    peak_dbfs = 20.0 * np.log10(max(peak, floor))
    return float(mean_dbfs), float(peak_dbfs)
