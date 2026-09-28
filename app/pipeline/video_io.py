"""视频元信息读取、灰度抽帧与彩色帧回读。

优先使用 OpenCV，元信息读取失败时回落到包内 ffprobe。
"""

import json
import os
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np

from io_utils import ffprobe_path


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    frame_count: int
    duration_sec: float


def _probe_with_ffprobe(path):
    exe = ffprobe_path()
    if not os.path.isfile(exe):
        return None
    cmd = [
        exe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of", "json", path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode("utf-8", "ignore"))
        stream = (data.get("streams") or [{}])[0]
    except (OSError, ValueError, subprocess.SubprocessError):
        return None

    fps = _parse_rate(stream.get("avg_frame_rate"))
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    frame_count = int(stream.get("nb_frames") or 0)
    duration = float(stream.get("duration") or 0.0)
    if fps <= 0:
        fps = 25.0
    if duration <= 0 and frame_count > 0:
        duration = frame_count / fps
    if frame_count <= 0 and duration > 0:
        frame_count = int(round(duration * fps))
    if width <= 0 or height <= 0 or frame_count <= 0:
        return None
    return VideoMeta(fps, width, height, frame_count, duration)


def _parse_rate(text):
    if not text:
        return 0.0
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            num, den = float(num), float(den)
        except ValueError:
            return 0.0
        return num / den if den else 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def read_metadata(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        meta = _probe_with_ffprobe(path)
        if meta is None:
            raise IOError("cannot open video: %s" % path)
        return meta

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if fps <= 0:
        fps = 25.0
    if width <= 0 or height <= 0 or frame_count <= 0:
        meta = _probe_with_ffprobe(path)
        if meta is None:
            raise IOError("invalid video metadata: %s" % path)
        return meta

    duration = frame_count / fps if fps > 0 else 0.0
    return VideoMeta(fps, width, height, frame_count, duration)


def decode_gray_frames(path, long_side=960, start_frame=0, end_frame=None):
    """解码 [start_frame, end_frame) 为灰度图，长边缩放到 long_side 以内。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("cannot open video: %s" % path)

    start_frame = max(0, int(start_frame or 0))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    limit = None if end_frame is None else max(0, int(end_frame))

    frames = []
    target = None
    index = start_frame
    while True:
        if limit is not None and index >= limit:
            break
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        if target is None:
            current_long = max(height, width)
            if long_side and current_long > long_side:
                scale = float(long_side) / float(current_long)
                target = (max(1, int(round(width * scale))),
                          max(1, int(round(height * scale))))
            else:
                target = (width, height)
        if (gray.shape[1], gray.shape[0]) != target:
            gray = cv2.resize(gray, target, interpolation=cv2.INTER_AREA)
        frames.append(gray)
        index += 1
    cap.release()

    if not frames:
        raise IOError("no frames decoded: %s" % path)
    return frames


def read_color_frame(path, frame_index, max_side=None):
    """回读指定帧的彩色图。

    注意：H.264 等编码的随机 seek 可能不准，故这里**顺序解码**到目标帧。
    批量读取请用 make_frame_reader（保持同一个 VideoCapture 顺序前进）。
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("cannot open video: %s" % path)

    target = max(0, int(frame_index))
    frame = None
    position = -1
    while position < target:
        ok, frame = cap.read()
        if not ok:
            frame = None
            break
        position += 1
    cap.release()

    if frame is None:
        raise IOError("cannot read frame %d: %s" % (frame_index, path))
    if max_side:
        frame = _scale_long_side(frame, max_side)
    return frame


def _scale_long_side(image, max_side):
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= max_side:
        return image
    scale = float(max_side) / float(long_side)
    new_size = (max(1, int(round(width * scale))),
                max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _canonical_size(frame_size, max_side):
    """把声明分辨率换算为统一输出尺寸；无效时返回 None。"""
    if not frame_size:
        return None
    width, height = int(frame_size[0]), int(frame_size[1])
    if width <= 0 or height <= 0:
        return None
    if max_side:
        long_side = max(width, height)
        if long_side > max_side:
            scale = float(max_side) / float(long_side)
            width = max(1, int(round(width * scale)))
            height = max(1, int(round(height * scale)))
    return (width, height)


def make_frame_reader(path, max_side=None, frame_offset=0, frame_size=None):
    """返回带缓存的彩色帧读取函数。

    保持一个 VideoCapture 顺序解码（避免 seek 不准导致取错帧），
    支持 frame_offset（内存兜底时的全局帧号偏移）。

    传入 frame_size（源声明分辨率）时，所有帧统一重采样到该尺寸，避免同
    一视频内分辨率切换导致首尾帧/参考图尺寸不一致。
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("cannot open video: %s" % path)
    offset = int(frame_offset or 0)
    cache = {}
    position = [-1]
    target_size = _canonical_size(frame_size, max_side)

    def _grab(target):
        if position[0] < 0 or target < position[0]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            position[0] = -1
        frame = None
        while position[0] < target:
            ok, frame = cap.read()
            if not ok:
                raise IOError("cannot read frame %d: %s" % (target, path))
            position[0] += 1
        return frame

    def reader(frame_index):
        target = int(frame_index) + offset
        if target in cache:
            return cache[target]
        frame = _grab(target)
        if frame is None:
            raise IOError("cannot read frame %d: %s" % (target, path))
        if target_size is not None:
            if (frame.shape[1], frame.shape[0]) != target_size:
                frame = cv2.resize(frame, target_size,
                                   interpolation=cv2.INTER_AREA)
        elif max_side:
            frame = _scale_long_side(frame, max_side)
        cache[target] = frame
        return frame

    return reader
