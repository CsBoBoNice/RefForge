"""第一层：有效区间修剪。

从两端跳过连续坏帧（黑场 / 白场 / 严重模糊 / 低信息量），
得到 valid_start / valid_end，均为好帧。
"""

import numpy as np


def _first_stable_good(bad, tolerance, min_valid):
    n = len(bad)
    for start in range(n):
        if bad[start]:
            continue
        window = bad[start:start + tolerance + min_valid + 1]
        if int((~window).sum()) >= min_valid + 1:
            return start
    good = np.where(~bad)[0]
    return int(good[0]) if good.size else 0


def _last_stable_good(bad, tolerance, min_valid):
    n = len(bad)
    for end in range(n - 1, -1, -1):
        if bad[end]:
            continue
        window = bad[max(0, end - tolerance - min_valid):end + 1]
        if int((~window).sum()) >= min_valid + 1:
            return end
    good = np.where(~bad)[0]
    return int(good[-1]) if good.size else n - 1


def find_valid_range(quality, cfg):
    trim = cfg["trim"]
    tolerance = int(trim["trim_tolerance_frames"])
    min_valid = int(trim["min_valid_frames"])
    invalid_ratio = float(trim["invalid_black_ratio"])

    n = len(quality.bad)
    if n == 0:
        return 0, 0, "invalid"

    if float(quality.bad.mean()) > invalid_ratio:
        return 0, n - 1, "invalid"

    valid_start = _first_stable_good(quality.bad, tolerance, min_valid)
    valid_end = _last_stable_good(quality.bad, tolerance, min_valid)

    if valid_end < valid_start:
        valid_start, valid_end = valid_end, valid_start

    if valid_end - valid_start < min_valid:
        best = int(np.argmax(quality.quality_score))
        return best, best, "ok"

    return valid_start, valid_end, "ok"


def invalid_reason(quality):
    if len(quality.mean) == 0:
        return "no_frames"
    if bool(quality.is_black.all()):
        return "all_frames_black"
    if bool(quality.is_white.all()):
        return "all_frames_white"
    if bool(quality.is_low_info.all()):
        return "all_frames_low_info"
    return "invalid_quality"
