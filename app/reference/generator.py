"""故事图生成：所有片段统一为 `grid_rows × grid_cols`（默认 3x3 九宫格）。

选帧规则：在片段有效区间内按**均匀时间间隔**确定 count 个目标时刻，
再在每个目标时刻附近（偏差 ≤ `reference.select_tolerance_ms`，默认 50ms）
挑选**最清晰**的一帧，选出的帧最终**严格按时间（帧号）升序**排列，
保证九宫格 1..9 为时间先后，避免故事图无法完整描述视频经过。
"""

import math

import numpy as np

from reference.storyboard import make_grid


def _clip(value, low, high):
    return max(low, min(high, value))


def _clearest_index(frame_quality, target, radius, low, high):
    """在目标时刻附近（时间偏差 ≤ radius 帧）取最清晰的一帧。

    优先未判为坏帧（黑 / 白 / 低信息 / 模糊）的候选；若候选全为坏帧
    或清晰度数据不可用，则退回离目标最近的帧。
    """
    if frame_quality is not None and radius > 0:
        start = _clip(int(math.ceil(target - radius)), low, high)
        end = _clip(int(math.floor(target + radius)), low, high)
        candidates = list(range(start, end + 1))
        sharpness = np.asarray(frame_quality.sharpness)
        if candidates and 0 <= start < len(sharpness):
            bad = np.asarray(frame_quality.bad)
            if len(bad) == len(sharpness):
                clear = [i for i in candidates
                         if 0 <= i < len(sharpness) and not bool(bad[i])]
            else:
                clear = [i for i in candidates if 0 <= i < len(sharpness)]
            pool = clear if clear else [i for i in candidates
                                        if 0 <= i < len(sharpness)]
            if pool:
                return max(pool, key=lambda i: float(sharpness[i]))
    return _clip(int(round(target)), low, high)


def reference_indices(frame_quality, valid_start, valid_end, fps, cfg):
    """参考图将取样的帧号（时间升序，共 `grid_rows*grid_cols` 个）。

    以帧号为单位的均匀时间间隔确定目标时刻，逐点就近取清晰帧；
    片段过短导致目标重叠时保证结果非递减，供调用方顺序预读去重。
    """
    ref = cfg.get("reference", {})
    rows = int(ref.get("grid_rows", 3))
    cols = int(ref.get("grid_cols", 3))
    count = max(1, rows * cols)
    tolerance_ms = float(ref.get("select_tolerance_ms", 50.0))

    low = int(min(valid_start, valid_end))
    high = int(max(valid_start, valid_end))
    if high <= low or count <= 1:
        return [low]

    radius = max(0.0, tolerance_ms) / 1000.0 * float(fps)
    targets = np.linspace(float(low), float(high), count)

    result = []
    for target in targets:
        index = _clearest_index(frame_quality, target, radius, low, high)
        if result:
            index = max(index, result[-1])
        result.append(index)
    return result


def _read_all(frame_reader, indices):
    images = []
    for index in indices:
        try:
            images.append(frame_reader(index))
        except Exception:
            return None
    return images


def generate(frame_quality, frame_reader, fallback_frames, valid_start,
             valid_end, fps, cfg):
    """按均匀时间间隔选 count 帧生成 `grid_rows × grid_cols` 宫格故事图。

    `fallback_frames` 仅在参考帧全部读取失败时使用（退化为起点/终点帧补齐）。
    """
    ref = cfg.get("reference", {})
    rows = int(ref.get("grid_rows", 3))
    cols = int(ref.get("grid_cols", 3))
    gap = int(ref.get("grid_gap_px", 10))
    margin = int(ref.get("grid_margin_px", 10))
    count = max(1, rows * cols)
    max_side = int(cfg["output"]["max_side_reference"])

    indices = reference_indices(frame_quality, valid_start, valid_end, fps, cfg)
    images = _read_all(frame_reader, indices) if indices else None
    if not images and fallback_frames:
        images = _read_all(frame_reader, list(fallback_frames))
    if not images:
        return None
    while len(images) < count:
        images.append(images[-1])
    return make_grid(images, cols, rows, max_side=max_side,
                     gap=gap, margin=margin)
