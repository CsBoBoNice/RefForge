"""镜头分割：PySceneDetect 硬切检测 + 长短镜头迭代修正。

保证每个片段时长落在 [min_segment_sec, max_segment_sec]，
除非输入总时长本身不足 min_segment_sec。
"""

import math
from dataclasses import dataclass

import cv2

try:
    from scenedetect import (AdaptiveDetector, FrameTimecode, SceneManager,
                             detect, open_video)
    _SCENEDETECT_ERROR = None
except Exception as exc:  # pragma: no cover - 依赖缺失时降级
    AdaptiveDetector = None
    FrameTimecode = None
    SceneManager = None
    detect = None
    open_video = None
    _SCENEDETECT_ERROR = exc


DEFAULT_THRESHOLDS = {"low": 4.5, "medium": 3.0, "high": 2.0}


@dataclass
class Segment:
    index: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration_sec: float
    method: str
    below_min: bool = False
    shot_id: str = ""
    file: str = ""
    shot_dir: str = ""


def scenedetect_available():
    return detect is not None


def scenedetect_install_hint():
    return "python\\python.exe -m pip install \"scenedetect>=0.6.4,<0.8\""


class _NormalizingStream(object):
    """包装 scenedetect 视频流，把尺寸不一致的帧重采样回声明分辨率。

    真实素材（如多段录屏拼接）可能在同一文件内切换分辨率：OpenCV 会按各段
    实际尺寸返回帧，而 PySceneDetect 会把尺寸变化的帧判定为损坏并跳过（见其
    ``MAX_FRAME_SIZE_ERRORS``）。这里统一重采样到 ``frame_size``，保证检测
    覆盖全部帧、且不产生误报。
    """

    def __init__(self, inner):
        self._inner = inner
        width, height = inner.frame_size
        self._size = (int(width), int(height))

    def read(self, decode=True, advance=True):
        frame = self._inner.read(decode=decode, advance=advance)
        if decode and hasattr(frame, "shape") and min(self._size) > 0:
            height, width = frame.shape[:2]
            if (width, height) != self._size:
                frame = cv2.resize(frame, self._size,
                                   interpolation=cv2.INTER_AREA)
        return frame

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _open_stream(path):
    """打开视频流；可用时返回尺寸归一化包装，失败时回退原始流。"""
    video = open_video(path)
    try:
        width, height = video.frame_size
    except Exception:
        return video
    if not (width and height):
        return video
    return _NormalizingStream(video)


def _close_stream(video):
    target = getattr(video, "_inner", video)
    capture = getattr(target, "capture", None)
    if capture is not None:
        try:
            capture.release()
            return
        except Exception:
            pass
    release = getattr(target, "release", None)
    if release is not None:
        try:
            release()
        except Exception:
            pass


def sensitivity_threshold(cfg, sensitivity):
    thresholds = cfg["segmentation"].get("adaptive_thresholds", DEFAULT_THRESHOLDS)
    key = str(sensitivity or "medium").strip().lower()
    if key not in thresholds:
        key = "medium"
    return float(thresholds.get(key, DEFAULT_THRESHOLDS["medium"]))


def detect_shot_cuts(path, fps, sensitivity, min_scene_len_sec, cfg):
    """用 PySceneDetect 检测硬切。

    返回 (cuts_frames, meta)；检测不可用或异常时返回 (None, meta)。
    """
    key = str(sensitivity or "medium").strip().lower()
    threshold = sensitivity_threshold(cfg, sensitivity)
    meta = {"method": "none", "threshold": threshold, "sensitivity": key,
            "scene_count": 0}
    if not scenedetect_available():
        meta["reason"] = "scenedetect_unavailable"
        return None, meta

    min_len = max(1, int(round(float(min_scene_len_sec) * float(fps or 25.0))))
    video = None
    try:
        video = _open_stream(path)
        manager = SceneManager()
        manager.add_detector(AdaptiveDetector(
            adaptive_threshold=threshold, min_scene_len=min_len))
        manager.detect_scenes(video=video, show_progress=False)
        scenes = manager.get_scene_list(start_in_scene=True)
    except Exception as exc:
        meta["reason"] = str(exc)
        return None, meta
    finally:
        if video is not None:
            _close_stream(video)

    if not scenes:
        meta["method"] = "pyscenedetect_adaptive"
        return [0], meta

    cuts = [int(scenes[0][0].get_frames())]
    for start_tc, _end_tc in scenes[1:]:
        cuts.append(int(start_tc.get_frames()))
    end_frame = int(scenes[-1][1].get_frames())
    if end_frame not in cuts:
        cuts.append(end_frame)
    meta.update({"method": "pyscenedetect_adaptive", "scene_count": len(scenes)})
    return sorted(set(cuts)), meta


def make_subdetector(path, sensitivity, min_scene_len_sec, cfg):
    """返回在 [start_frame, end_frame) 内做二次检测的函数；不可用时返回 None。"""
    if not scenedetect_available() or open_video is None:
        return None
    threshold = sensitivity_threshold(cfg, sensitivity)
    min_len_sec = float(min_scene_len_sec)

    def subdetect(start_frame, end_frame):
        try:
            video = _open_stream(path)
        except Exception:
            return []
        try:
            fps = float(video.frame_rate or 25.0)
            start_tc = FrameTimecode(timecode=int(start_frame) / fps, fps=fps)
            end_tc = FrameTimecode(timecode=int(end_frame) / fps, fps=fps)
            video.seek(start_tc)
            manager = SceneManager()
            manager.add_detector(AdaptiveDetector(
                adaptive_threshold=threshold,
                min_scene_len=max(1, int(round(min_len_sec * fps)))))
            manager.detect_scenes(video, end_time=end_tc, show_progress=False)
            scenes = manager.get_scene_list(start_in_scene=True)
        except Exception:
            return []
        finally:
            _close_stream(video)
        cuts = []
        for start_scene, _end_scene in scenes:
            frame = int(start_scene.get_frames())
            if int(start_frame) < frame < int(end_frame):
                cuts.append(frame)
        return cuts

    return subdetect


def _normalize_bounds(cuts, total):
    bounds = {0, int(total)}
    for cut in cuts or []:
        value = int(cut)
        if 0 < value < int(total):
            bounds.add(value)
    return sorted(bounds)


def _to_intervals(bounds):
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _split_over_max(intervals, max_f):
    result = []
    for start, end in intervals:
        length = end - start
        if length > max_f:
            count = int(math.ceil(length / float(max_f)))
            for k in range(count):
                a = start + int(round(k * length / float(count)))
                b = start + int(round((k + 1) * length / float(count)))
                if b > a:
                    result.append((a, b))
        else:
            result.append((start, end))
    return result


def _merge_under_min(intervals, min_f, max_f):
    intervals = list(intervals)
    changed = True
    while changed and len(intervals) > 1:
        changed = False
        for i, (start, end) in enumerate(intervals):
            if end - start >= min_f:
                continue
            options = []
            if i > 0:
                left_start = intervals[i - 1][0]
                options.append((end - left_start, i - 1))
            if i < len(intervals) - 1:
                right_end = intervals[i + 1][1]
                options.append((right_end - start, i + 1))
            if not options:
                break
            fitting = [opt for opt in options if opt[0] <= max_f]
            merged_len, j = min(fitting or options, key=lambda opt: opt[0])
            a = min(start, intervals[j][0])
            b = max(end, intervals[j][1])
            lo, hi = min(i, j), max(i, j)
            intervals[lo:hi + 1] = [(a, b)]
            changed = True
            break
    return intervals


def _compliant(intervals, min_f, max_f):
    return all(min_f <= (end - start) <= max_f for start, end in intervals)


def _force_fix(intervals, min_f, max_f):
    intervals = _split_over_max(intervals, max_f)
    for _ in range(100):
        if len(intervals) <= 1:
            break
        index = next((i for i, (s, e) in enumerate(intervals) if e - s < min_f), None)
        if index is None:
            break
        if index == 0:
            j = 1
        elif index == len(intervals) - 1:
            j = index - 1
        else:
            left_len = intervals[index - 1][1] - intervals[index - 1][0]
            right_len = intervals[index + 1][1] - intervals[index + 1][0]
            j = index - 1 if left_len <= right_len else index + 1
        a = min(intervals[index][0], intervals[j][0])
        b = max(intervals[index][1], intervals[j][1])
        lo, hi = min(index, j), max(index, j)
        intervals[lo:hi + 1] = [(a, b)]
        intervals = _split_over_max(intervals, max_f)
    return intervals


def _method_for(start, end, original_cuts, detected):
    if not detected:
        return "equal_split"
    inner = any(start < cut < end for cut in original_cuts)
    if inner:
        return "merge"
    if start in original_cuts and end in original_cuts:
        return "pyscenedetect_adaptive"
    return "equal_split"


def build_segments(total_frames, fps, cuts, cfg, subdetect_fn=None):
    """把检测到的切点整理为合规片段列表。

    返回 (segments, meta)。meta 含 detected_shot_count / segment_count /
    split_iterations / fallback。
    """
    seg_cfg = cfg["segmentation"]
    min_sec = float(seg_cfg["min_segment_sec"])
    max_sec = float(seg_cfg["max_segment_sec"])
    max_iter = max(1, int(seg_cfg["max_split_iterations"]))
    fps = float(fps or 25.0)
    min_f = max(1, int(round(min_sec * fps)))
    max_f = max(min_f, int(round(max_sec * fps)))
    total = max(0, int(total_frames))

    meta = {"detected_shot_count": 0, "segment_count": 0,
            "split_iterations": 0, "fallback": False}

    if total <= 0:
        meta["fallback"] = True
        return [], meta

    detected = bool(cuts) and len(cuts) >= 2
    original_cuts = set(int(c) for c in (cuts or []) if 0 <= int(c) <= total)
    original_cuts.add(0)
    original_cuts.add(total)

    if total < min_f:
        segment = Segment(1, 0, total, 0.0, total / fps, total / fps,
                          "below_min", True)
        meta.update({"detected_shot_count": 1 if detected else 0,
                     "segment_count": 1, "split_iterations": 1,
                     "fallback": True})
        return [segment], meta

    bounds = _normalize_bounds(cuts, total) if detected else [0, total]
    intervals = _to_intervals(bounds)

    if subdetect_fn is not None:
        extra = []
        for start, end in intervals:
            if end - start > max_f:
                extra.extend(subdetect_fn(start, end))
        if extra:
            bounds = _normalize_bounds(list(original_cuts) + extra, total)
            intervals = _to_intervals(bounds)

    detected_shot_count = len(intervals)

    iterations = 0
    while True:
        intervals = _split_over_max(intervals, max_f)
        intervals = _merge_under_min(intervals, min_f, max_f)
        iterations += 1
        if _compliant(intervals, min_f, max_f) or iterations >= max_iter:
            break

    if not _compliant(intervals, min_f, max_f):
        intervals = _force_fix(intervals, min_f, max_f)
        meta["fallback"] = True

    segments = []
    for position, (start, end) in enumerate(intervals, start=1):
        below_min = (end - start) < min_f
        segments.append(Segment(
            index=position,
            start_frame=int(start),
            end_frame=int(end),
            start_sec=float(start) / fps,
            end_sec=float(end) / fps,
            duration_sec=float(end - start) / fps,
            method=_method_for(start, end, original_cuts, detected),
            below_min=below_min,
            shot_id="shot_%04d" % position,
            shot_dir="shot_%04d" % position,
        ))

    meta.update({"detected_shot_count": detected_shot_count,
                 "segment_count": len(segments),
                 "split_iterations": iterations,
                 "fallback": meta["fallback"]})
    return segments, meta
