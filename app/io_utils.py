"""路径解析、目录管理、JSON / 图片写出等通用工具。

所有路径均基于包根目录（app/ 的上一级）解析，不写死开发机绝对路径。
"""

import json
import os
import re

import cv2
import numpy as np


VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")

SHOT_DETAIL_DIR = "detailed"


def shot_detail_dir(shot_dir):
    """shot_XXXX 下存放详细信息文件的子目录（asr / prompt 元数据 / 三轨音频等）。"""
    return os.path.join(shot_dir, SHOT_DETAIL_DIR)


def package_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def app_dir():
    return os.path.join(package_root(), "app")


def bin_dir():
    return os.path.join(package_root(), "bin")


def ffmpeg_path():
    return os.path.join(bin_dir(), "ffmpeg.exe")


def ffprobe_path():
    return os.path.join(bin_dir(), "ffprobe.exe")


def models_dir():
    return os.path.join(package_root(), "models")


def asr_model_dir(cfg=None):
    """语音模型根目录；cfg.asr.models 可用相对包根或绝对路径覆盖。"""
    value = ""
    if isinstance(cfg, dict):
        value = str((cfg.get("asr") or {}).get("models") or "")
    value = value.strip()
    if not value:
        return os.path.join(models_dir(), "asr")
    if os.path.isabs(value):
        return os.path.normpath(value)
    return os.path.normpath(os.path.join(package_root(), value))


def _resolve_pkg_path(value, default_relative):
    value = str(value or "").strip()
    if not value:
        return os.path.normpath(os.path.join(package_root(), default_relative))
    if os.path.isabs(value):
        return os.path.normpath(value)
    return os.path.normpath(os.path.join(package_root(), value))


def llama_bin_dir(cfg=None):
    """llama.cpp 可执行文件目录；cfg.describe.bin 可用相对包根或绝对路径覆盖。"""
    value = ""
    if isinstance(cfg, dict):
        value = (cfg.get("describe") or {}).get("bin") or ""
    return _resolve_pkg_path(value, os.path.join("llama_cpp", "llama_bin"))


def llama_model_dir(cfg=None):
    """llama.cpp 模型目录；cfg.describe.models 可用相对包根或绝对路径覆盖。"""
    value = ""
    if isinstance(cfg, dict):
        value = (cfg.get("describe") or {}).get("models") or ""
    return _resolve_pkg_path(value, os.path.join("models", "llm"))


def llama_server_path(cfg=None):
    name = "llama-server.exe"
    if isinstance(cfg, dict):
        name = str((cfg.get("describe") or {}).get("server") or name)
    return os.path.join(llama_bin_dir(cfg), name)


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def natural_sort_key(name):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", os.path.basename(name))]


def safe_stem(path):
    """由视频路径得到可作为目录名的源文件名（去扩展名、去除非法字符）。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r'[\\/:*?"<>|]', "_", stem).strip()
    return stem or "video"


def list_videos(directory):
    if not os.path.isdir(directory):
        return []
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if f.lower().endswith(VIDEO_EXTENSIONS)]
    files.sort(key=natural_sort_key)
    return files


def write_json(path, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def save_bgr_jpeg(path, image_bgr, quality=92):
    ensure_dir(os.path.dirname(path))
    ok = cv2.imwrite(path, image_bgr,
                     [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise IOError("failed to write image: %s" % path)
    return path


def scale_to_long_side(image, max_side):
    if image is None or max_side is None:
        return image
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= max_side:
        return image
    scale = float(max_side) / float(long_side)
    new_size = (max(1, int(round(width * scale))),
                max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def diag_of_shape(shape):
    height, width = shape[:2]
    return float(np.hypot(width, height))
