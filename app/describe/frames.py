"""按时间均匀采样视频帧：ffmpeg 2fps -> PNG（1080P 画框内，每格取清晰帧）。

仅依赖包内 ffmpeg / ffprobe（`bin/`），产物写入片段目录下独立的
`frames/` 子目录，避免 shot 目录过于杂乱。
"""

import glob
import os
import subprocess

from io_utils import ensure_dir, ffmpeg_path, ffprobe_path

DEFAULT_FPS = 3.0
DEFAULT_MAX_SIDE = 1920
DEFAULT_FORMAT = "png"


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


def probe_video(path):
    """返回 (width, height, duration_sec)；读取失败返回 None。"""
    exe = ffprobe_path()
    if not os.path.isfile(exe):
        return None
    cmd = [
        exe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,duration",
        "-of", "json", path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        import json
        data = json.loads(out.decode("utf-8", "ignore"))
        stream = (data.get("streams") or [{}])[0]
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    fps = _parse_rate(stream.get("avg_frame_rate"))
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    duration = float(stream.get("duration") or 0.0)
    if duration <= 0 and fps > 0:
        duration = 0.0
    if width <= 0 or height <= 0:
        return None
    return width, height, duration


def fit_long_side(width, height, max_side):
    """限制在「长边 ≤ max_side、短边 ≤ max_side*9/16」（即 1080P 画框，不放大）。

    横屏按 1920x1080、竖屏按 1080x1920 约束，返回偶数尺寸 (w, h)。
    """
    width = max(2, int(width))
    height = max(2, int(height))
    if max_side and int(max_side) > 0:
        long_side = int(max_side)
        short_side = int(round(long_side * 9.0 / 16.0))
        if width >= height:
            max_w, max_h = long_side, short_side
        else:
            max_w, max_h = short_side, long_side
        scale = min(1.0, float(max_w) / width, float(max_h) / height)
        if scale < 1.0:
            width = int(round(width * scale))
            height = int(round(height * scale))
    width -= width % 2
    height -= height % 2
    return max(2, width), max(2, height)


def frame_paths(out_dir, image_format=DEFAULT_FORMAT):
    pattern = os.path.join(out_dir, "frame_*.%s" % image_format)
    return sorted(glob.glob(pattern))


def _run_ffmpeg(exe, video_path, pattern, fps, out_w, out_h, start,
                duration_sec, max_frames):
    cmd = [exe, "-y", "-v", "error"]
    if start > 0:
        cmd += ["-ss", "%.3f" % start]
    cmd += ["-i", video_path]
    if duration_sec > 0:
        cmd += ["-t", "%.3f" % duration_sec]
    cmd += ["-vf", "fps=%s,scale=%d:%d" % (_fmt_number(fps), out_w, out_h),
            "-an", "-sn", "-pix_fmt", "rgb24", "-compression_level", "3"]
    if max_frames and int(max_frames) > 0:
        cmd += ["-frames:v", str(int(max_frames))]
    cmd += [pattern]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       check=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", "ignore").strip()
        raise RuntimeError("ffmpeg frame extraction failed: %s" % detail)


def _frame_sharpness(path):
    """用拉普拉斯方差衡量清晰度；读取失败返回 -1。"""
    try:
        import cv2
    except ImportError:
        return 0.0
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return -1.0
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def _select_sharp_frames(out_dir, image_format, fps, candidates):
    """返回候选帧临时目录与文件名模板。"""
    temp_dir = os.path.join(out_dir, "_candidates")
    pattern = os.path.join(temp_dir, "cand_%05d." + image_format)
    return temp_dir, pattern


def extract_frames(video_path, out_dir, fps=DEFAULT_FPS, max_side=DEFAULT_MAX_SIDE,
                   start_sec=0.0, duration_sec=None, image_format=DEFAULT_FORMAT,
                   max_frames=0, select_keyframes=True, keyframe_candidates=3,
                   logger=None):
    """按 fps 均匀抽帧为 PNG，写入 out_dir；返回抽帧结果 dict。

    select_keyframes 时：以 fps*keyframe_candidates 抽出候选帧，再在每个时间格
    内选取最清晰的一帧（近似关键帧），避免把运动模糊帧喂给多模态模型。

    返回：{frames:[{name,path,time_sec}], width, height, fps, count, selected}
    """
    import shutil
    exe = ffmpeg_path()
    if not os.path.isfile(exe):
        raise RuntimeError("ffmpeg not found: %s" % exe)
    if not os.path.isfile(video_path):
        raise IOError("video not found: %s" % video_path)

    probe = probe_video(video_path)
    if probe is None:
        raise IOError("cannot probe video: %s" % video_path)
    src_w, src_h, src_duration = probe
    out_w, out_h = fit_long_side(src_w, src_h, max_side)
    if duration_sec is None or duration_sec <= 0:
        duration_sec = max(0.0, src_duration - max(0.0, float(start_sec or 0.0)))
    duration_sec = float(duration_sec)

    ensure_dir(out_dir)
    for old in frame_paths(out_dir, image_format):
        try:
            os.remove(old)
        except OSError:
            pass

    start = max(0.0, float(start_sec or 0.0))
    candidates = max(1, int(keyframe_candidates or 1))
    selected = False
    if select_keyframes and candidates > 1:
        temp_dir, cand_pattern = _select_sharp_frames(
            out_dir, image_format, fps, candidates)
        shutil.rmtree(temp_dir, ignore_errors=True)
        ensure_dir(temp_dir)
        cand_fps = float(fps) * candidates
        limit = int(max_frames) * candidates if max_frames else 0
        _run_ffmpeg(exe, video_path, cand_pattern, cand_fps, out_w, out_h,
                    start, duration_sec, limit)
        cand_paths = sorted(glob.glob(
            os.path.join(temp_dir, "cand_*." + image_format)))
        if cand_paths:
            groups = [cand_paths[i:i + candidates]
                      for i in range(0, len(cand_paths), candidates)]
            for index, group in enumerate(groups, start=1):
                best = max(group, key=_frame_sharpness)
                shutil.copyfile(best, os.path.join(
                    out_dir, "frame_%05d.%s" % (index, image_format)))
            selected = True
        shutil.rmtree(temp_dir, ignore_errors=True)

    if not selected:
        pattern = os.path.join(out_dir, "frame_%05d." + image_format)
        _run_ffmpeg(exe, video_path, pattern, fps, out_w, out_h, start,
                    duration_sec, max_frames)

    paths = frame_paths(out_dir, image_format)
    frames = []
    for index, path in enumerate(paths):
        frames.append({
            "name": os.path.basename(path),
            "path": path,
            "time_sec": round(index / float(fps), 3),
        })
    if not frames:
        raise RuntimeError("no frames extracted: %s" % video_path)

    if logger:
        logger.info(
            "  frames: %d @ %.1ffps %dx%d (source %dx%d, %.2fs, select=%s)"
            " -> %s",
            len(frames), fps, out_w, out_h, src_w, src_h, duration_sec,
            selected, os.path.basename(out_dir))
    return {
        "frames": frames,
        "width": out_w,
        "height": out_h,
        "source_width": src_w,
        "source_height": src_h,
        "fps": float(fps),
        "duration_sec": duration_sec,
        "count": len(frames),
        "selected": selected,
    }


def _fmt_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == int(number):
        return str(int(number))
    return ("%.3f" % number).rstrip("0")
