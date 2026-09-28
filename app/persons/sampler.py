"""智能抽帧：以每 1 秒为窗口，窗口内取最清晰的一帧作为代表帧。

- 窗口内按时间均匀取最多 `candidates_per_window` 个候选帧；原生帧率低于候选数时取全部；
- 以灰度 Laplacian 方差最大者为代表帧；全为黑/白/低信息帧则跳过该窗口；
- 返回 `[(frame_index, time_sec), ...]`，帧号为源视频全局帧号。
"""

import cv2
import numpy as np


def _windows(fps, fps_target, total_frames, candidates):
    window_frames = max(1, int(round(fps / max(fps_target, 1e-6))))
    if window_frames >= candidates:
        offsets = sorted(set(
            int(round(value)) for value in
            np.linspace(0, window_frames - 1, candidates)))
    else:
        offsets = list(range(window_frames))
    return window_frames, set(offsets)


def select_representative_frames(path, cfg, logger=None):
    pc = cfg.get("persons") or {}
    fps_target = float(pc.get("fps_target", 1.0) or 1.0)
    candidates = max(1, int(pc.get("candidates_per_window", 5) or 5))
    fq = cfg.get("frame_quality") or {}
    black_mean = float(fq.get("black_mean", 15))
    black_std = float(fq.get("black_std", 10))
    white_mean = float(fq.get("white_mean", 240))
    white_std = float(fq.get("white_std", 10))
    low_info_std = float(fq.get("low_info_std", 8))

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise IOError("cannot open video: %s" % path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 1e-6:
        fps = 25.0
    window_frames, offsets = _windows(fps, fps_target, total, candidates)

    results = []
    best = None  # (laplacian_var, frame_index)
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        local = index % window_frames
        if local in offsets:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean = float(gray.mean())
            std = float(gray.std())
            laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            bad = ((mean < black_mean and std < black_std)
                   or (mean > white_mean and std < white_std)
                   or std < low_info_std)
            if not bad and (best is None or laplacian > best[0]):
                best = (laplacian, index)
        if local == window_frames - 1 or (total > 0 and index == total - 1):
            if best is not None:
                results.append((best[1], best[1] / fps))
            best = None
        index += 1
    if best is not None:
        results.append((best[1], best[1] / fps))
    capture.release()
    if logger is not None:
        logger.debug("  persons sampler: %d representative frames", len(results))
    return results
