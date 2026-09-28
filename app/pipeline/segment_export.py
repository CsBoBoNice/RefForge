"""片段导出：ffmpeg copy / reencode + ffprobe 时长校验。"""

import json
import os
import subprocess

from io_utils import ensure_dir, ffmpeg_path, ffprobe_path


def probe_duration(path):
    exe = ffprobe_path()
    if not os.path.isfile(exe):
        return 0.0
    cmd = [exe, "-v", "error", "-show_entries", "format=duration",
           "-of", "json", path]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode("utf-8", "ignore"))
        return float(data.get("format", {}).get("duration") or 0.0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def _run(cmd):
    try:
        return subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).returncode
    except OSError:
        return -1


def export_segment(src_path, start_sec, end_sec, dest_path, encode_mode, cfg):
    """把源视频的 [start_sec, end_sec) 导出为 dest_path；失败返回 None。"""
    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec, float(end_sec))
    duration = end_sec - start_sec
    if duration <= 0:
        return None
    ensure_dir(os.path.dirname(dest_path))
    exe = ffmpeg_path()
    if not os.path.isfile(exe):
        return None

    mode = str(encode_mode or "auto").strip().lower()
    tolerance = float(cfg["segmentation"]["copy_duration_tolerance_sec"])

    if mode in ("copy", "auto"):
        cmd = [exe, "-y", "-ss", "%.3f" % start_sec, "-i", src_path,
               "-t", "%.3f" % duration, "-c", "copy", "-avoid_negative_ts",
               "make_zero", dest_path]
        if _run(cmd) == 0 and os.path.isfile(dest_path):
            if mode == "copy":
                return dest_path
            actual = probe_duration(dest_path)
            if abs(actual - duration) <= tolerance:
                return dest_path
            try:
                os.remove(dest_path)
            except OSError:
                pass

    cmd = [exe, "-y", "-ss", "%.3f" % start_sec, "-i", src_path,
           "-t", "%.3f" % duration, "-c:v", "libx264", "-crf", "18",
           "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac",
           "-movflags", "+faststart", dest_path]
    if _run(cmd) == 0 and os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path
    return None
