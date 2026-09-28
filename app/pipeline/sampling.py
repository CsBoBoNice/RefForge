"""第二层：多帧采样。

在有效区间内按帧号等间隔均匀采样，K 由时长决定。
"""

import numpy as np


def choose_k(duration_sec, cfg):
    sampling = cfg["sampling"]
    if duration_sec < 1.0:
        return int(sampling["k_short"])
    if duration_sec <= 3.0:
        return int(sampling["k_normal"])
    return int(sampling["k_long"])


def sample_frames(valid_start, valid_end, duration_sec, cfg):
    span = int(valid_end) - int(valid_start)
    if span <= 0:
        return [int(valid_start)]

    k = choose_k(duration_sec, cfg)
    effective = span + 1
    k = min(k, effective)
    k = max(k, 2)

    raw = np.linspace(int(valid_start), int(valid_end), k)
    indices = sorted({int(round(value)) for value in raw})
    if len(indices) < 2:
        indices = [int(valid_start), int(valid_end)]
    return indices
