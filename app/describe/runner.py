"""视频描述阶段编排：准备输入图像 -> 组装 prompt -> 多模态推理 -> 生成中文 -> 落盘。

输入图像由 `describe.input_mode` 决定：
- `storyboard`（默认）：把片段的 3x3 宫格故事图 `scene_reference.jpg` 按最大 1080P
  画框缓存到 `detailed/` 后再推理（原图已 ≤1080P 时直接用原图），避免图片过大导致推理过慢；
- `frames`：用 ffmpeg 按时间均匀采样的多帧图像。

对每个片段产物：
- `shot_XXXX/frames/frame_XXXXX.png`：`frames` 模式下的采样帧（独立子目录）；
- `shot_XXXX/detailed/storyboard_1080p.jpg`：`storyboard` 模式下缩放后的故事图缓存（仅当需要缩放时）；
- `shot_XXXX/video_prompt_zh.txt`：中文 H3 full-reference prompt；
- `shot_XXXX/detailed/video_prompt.json`：结构化记录（输入模式 / 图像清单 / 字幕 / prompt 文案 / 元数据）。
"""

import json
import os
import shutil
import time

import cv2

from io_utils import ensure_dir, shot_detail_dir, write_json

from . import client as client_mod
from . import frames as frames_mod
from . import prompt as prompt_mod

SCHEMA_VERSION = "1.0"

PROMPT_ZH = "video_prompt_zh.txt"
PROMPT_JSON = "video_prompt.json"
STORYBOARD_CACHE = "storyboard_1080p.jpg"


def available(cfg):
    return client_mod.available(cfg)


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _shot_duration(shot):
    duration = shot.get("duration_sec")
    if duration:
        return float(duration)
    try:
        return max(0.0, float(shot.get("end_sec", 0.0))
                   - float(shot.get("start_sec", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _read_subtitles(shot_dir, duration_sec):
    """读取 detailed/asr.json，转换为相对片段起点的字幕条目；返回 (entries, speakers)。"""
    payload = _read_json(os.path.join(shot_detail_dir(shot_dir), "asr.json"))
    if not payload:
        return [], []
    start = float(payload.get("shot_start_sec", 0.0) or 0.0)
    entries = []
    speakers = set()
    for seg in payload.get("segments", []) or []:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        rel_start = max(0.0, float(seg.get("start_sec", 0.0)) - start)
        rel_end = min(duration_sec, float(seg.get("end_sec", 0.0)) - start)
        if rel_end <= rel_start:
            continue
        speaker = str(seg.get("speaker") or "").strip()
        entries.append({
            "start_sec": round(rel_start, 3),
            "end_sec": round(rel_end, 3),
            "speaker": speaker,
            "text": text,
        })
        if speaker:
            speakers.add(speaker)
    return entries, sorted(speakers)


def _clean_frames(frames_dir, image_format):
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir, ignore_errors=True)


def _write_outputs(shot_dir, prompt_text, payload):
    with open(os.path.join(shot_dir, PROMPT_ZH), "w", encoding="utf-8") as fh:
        fh.write(prompt_text + "\n")
    write_json(os.path.join(shot_detail_dir(shot_dir), PROMPT_JSON), payload)


def _input_mode(desc_cfg):
    return prompt_mod.normalize_mode(desc_cfg.get("input_mode"))


def _reference_grid(cfg):
    ref = cfg.get("reference") or {}
    return (int(ref.get("grid_rows", 3) or 3),
            int(ref.get("grid_cols", 3) or 3))


def _cache_storyboard(src_path, shot_dir, max_side, jpeg_quality, logger=None):
    """把 3x3 故事图按最大 1080P 画框缓存到 `detailed/`，返回用于推理的图像路径。

    原图长边 / 短边已在画框内（≤ max_side）时直接用原图，不放大、不重复编码；
    超出时缩放到画框内并写 `detailed/storyboard_1080p.jpg`，避免图片过大拖慢推理。
    """
    image = cv2.imread(src_path)
    if image is None:
        raise IOError("cannot read storyboard: %s" % src_path)
    height, width = image.shape[:2]
    out_w, out_h = frames_mod.fit_long_side(width, height, max_side)
    if (out_w, out_h) == (width, height):
        return src_path
    resized = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    cache_path = os.path.join(shot_detail_dir(shot_dir), STORYBOARD_CACHE)
    ensure_dir(os.path.dirname(cache_path))
    ok = cv2.imwrite(cache_path, resized,
                     [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise IOError("failed to write storyboard cache: %s" % cache_path)
    if logger:
        logger.info("  storyboard cache: %dx%d -> %dx%d %s",
                    width, height, out_w, out_h, os.path.basename(cache_path))
    return cache_path


def process_shot(server, job, shot, cfg, logger):
    desc_cfg = cfg.get("describe") or {}
    shot_dir = shot["shot_dir"]
    duration = _shot_duration(shot)
    if duration <= 0:
        raise RuntimeError("invalid shot duration")

    segment_file = shot.get("file") or ""
    segment_path = os.path.join(shot_dir, segment_file) if segment_file else ""
    if segment_path and os.path.isfile(segment_path):
        input_path = segment_path
        start_sec = 0.0
    else:
        input_path = job["video"]
        start_sec = float(shot.get("start_sec", 0.0) or 0.0)

    subtitles, speakers = _read_subtitles(shot_dir, duration)

    mode = _input_mode(desc_cfg)
    storyboard_path = os.path.join(shot_dir, "scene_reference.jpg")
    use_storyboard = (mode == prompt_mod.INPUT_MODE_STORYBOARD
                      and os.path.isfile(storyboard_path))
    if mode == prompt_mod.INPUT_MODE_STORYBOARD and not use_storyboard \
            and logger:
        logger.warning("  %s storyboard missing; falling back to frames",
                       shot.get("shot_id"))
    effective_mode = (prompt_mod.INPUT_MODE_STORYBOARD if use_storyboard
                      else prompt_mod.INPUT_MODE_FRAMES)

    frames_dir = os.path.join(
        shot_dir, str(desc_cfg.get("frame_dir") or "frames"))
    started = time.monotonic()
    info = None
    storyboard_image = None
    if use_storyboard:
        storyboard_image = _cache_storyboard(
            storyboard_path, shot_dir,
            int(desc_cfg.get("storyboard_max_side", 1920) or 1920),
            int((cfg.get("output") or {}).get("jpeg_quality", 92) or 92),
            logger=logger)
        image_paths = [storyboard_image]
        user_text = prompt_mod.build_user_prompt(
            effective_mode, duration, 0.0, [], subtitles, speakers,
            grid=_reference_grid(cfg))
    else:
        info = frames_mod.extract_frames(
            input_path, frames_dir,
            fps=float(desc_cfg.get("frame_fps", 2.0) or 2.0),
            max_side=int(desc_cfg.get("frame_max_side", 1920) or 1920),
            start_sec=start_sec,
            duration_sec=duration if start_sec > 0 else None,
            image_format=str(desc_cfg.get("frame_format") or "png"),
            max_frames=int(desc_cfg.get("max_frames", 0) or 0),
            select_keyframes=bool(desc_cfg.get("frame_select", True)),
            keyframe_candidates=int(desc_cfg.get("frame_candidates", 3) or 1),
            logger=logger)
        image_paths = [frame["path"] for frame in info["frames"]]
        user_text = prompt_mod.build_user_prompt(
            effective_mode, duration, info["fps"], info["frames"],
            subtitles, speakers, grid=_reference_grid(cfg))

    chinese, _ = server.chat(
        prompt_mod.system_prompt(effective_mode), user_text,
        image_paths=image_paths)
    chinese = prompt_mod.clean_output(chinese)
    if not chinese:
        raise RuntimeError("empty prompt")

    missing = prompt_mod.missing_sections(chinese)
    elapsed = time.monotonic() - started
    grid_rows, grid_cols = _reference_grid(cfg)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": job.get("source_file") or os.path.basename(job["video"]),
        "shot_id": shot.get("shot_id"),
        "duration_sec": round(duration, 3),
        "input_mode": effective_mode,
        "storyboard": (os.path.relpath(storyboard_image, shot_dir)
                       .replace("\\", "/") if storyboard_image else None),
        "storyboard_source": ("scene_reference.jpg" if use_storyboard else None),
        "storyboard_cached": bool(use_storyboard
                                  and storyboard_image != storyboard_path),
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "frame_fps": info["fps"] if info else 0.0,
        "frame_count": info["count"] if info else 0,
        "frame_select": bool(info.get("selected", False)) if info else False,
        "frame_dir": (os.path.relpath(frames_dir, shot_dir).replace("\\", "/")
                      if info else ""),
        "frames": ([{"name": frame["name"], "time_sec": frame["time_sec"]}
                    for frame in info["frames"]] if info else []),
        "speakers": speakers,
        "subtitles": subtitles,
        "language": "zh-CN",
        "prompt": chinese,
        "engine": str(desc_cfg.get("engine") or "llama.cpp"),
        "model": os.path.basename(client_mod.model_path(cfg)),
        "missing_sections": missing,
        "elapsed_sec": round(elapsed, 2),
    }
    _write_outputs(shot_dir, chinese, payload)

    if info is not None and not desc_cfg.get("keep_frames", True):
        _clean_frames(frames_dir, str(desc_cfg.get("frame_format") or "png"))

    if logger:
        logger.info(
            "  describe %s (%s): frames=%d subs=%d chars=%d elapsed=%.1fs%s",
            shot.get("shot_id"), effective_mode,
            info["count"] if info else 0, len(subtitles), len(chinese),
            elapsed, (" missing=%s" % ",".join(missing)) if missing else "")
    return "ok"


def run_phase(jobs, output_root, cfg, logger, debug=False):
    """加载模型一次，顺序处理所有片段的描述生成，结束后释放显存。"""
    stats = {"describe_ok": 0, "describe_error": 0, "describe_shots": 0,
             "describe_skipped": 0}
    if not jobs:
        return stats

    ready, missing = client_mod.available(cfg)
    if not ready:
        if logger:
            logger.warning("describe disabled; missing: %s", ", ".join(missing))
        return stats

    desc_cfg = cfg.get("describe") or {}
    limit = int(desc_cfg.get("limit", 0) or 0)
    log_path = os.path.join(output_root, "llama_server.log")
    server = client_mod.LlamaServer(cfg, logger=logger, log_path=log_path)
    processed = 0
    try:
        server.start()
        for job in jobs:
            for shot in job.get("shots", []) or []:
                if limit > 0 and processed >= limit:
                    break
                processed += 1
                stats["describe_shots"] += 1
                try:
                    result = process_shot(server, job, shot, cfg, logger)
                except Exception as exc:  # noqa: BLE001 - 单片段失败不影响整体
                    stats["describe_error"] += 1
                    if logger:
                        logger.error("  describe %s failed: %s",
                                     shot.get("shot_id"), exc)
                    continue
                if result == "skipped":
                    stats["describe_skipped"] += 1
                else:
                    stats["describe_ok"] += 1
            if limit > 0 and processed >= limit:
                break
    except Exception as exc:  # noqa: BLE001 - 服务启动失败
        if logger:
            logger.error("describe phase failed: %s", exc)
        stats["describe_error"] += 1
    finally:
        server.close()
    return stats
