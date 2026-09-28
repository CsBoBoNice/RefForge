"""CLI 入口与流程编排。

流程：源视频 -> 镜头分割（智能分割）-> 逐片段生成 3x3 宫格故事图
-> 人声分离 -> 语音识别 -> 多模态视频描述。

说明：本工具不再做运镜识别，也不再输出 `motion_meta.json`；`scene_reference.jpg`
（3x3 宫格故事图）按均匀时间间隔选帧（每点就近取清晰帧）。

用法：
    python main.py [--input DIR] [--output DIR] [--config FILE] [--single FILE]
                   [--debug] [--overwrite]
                   [--no-split] [--split-sensitivity low|medium|high]
                   [--min-seg-sec SEC] [--max-seg-sec SEC]
                   [--segment-encode auto|copy|reencode] [--clean-segments]
                   [--interactive]
"""

import argparse
import contextlib
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from config import load_config
from io_utils import (ensure_dir, list_videos, package_root, safe_stem,
                      save_bgr_jpeg, shot_detail_dir, write_json)
from logging_utils import setup_logger
from pipeline import motion, sampling, features, quality as quality_mod, trim
from pipeline import segmentation, segment_export, video_io
from pipeline.segmentation import Segment
from pipeline.video_io import VideoMeta
from reference import generator
from timing import StageTimer
import describe
import interactive
import persons
import speech


SEGMENTS_SCHEMA_VERSION = "1.0"


def _resolve(path):
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(package_root(), path))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="镜头分割 + 故事图生成 + 语音识别 + 视频描述")
    parser.add_argument("--input", default="./input", help="批处理输入目录")
    parser.add_argument("--output", default="./output", help="输出根目录")
    parser.add_argument("--config", default="./config.json", help="阈值配置")
    parser.add_argument("--single", default=None, help="只处理单个视频文件")
    parser.add_argument("--debug", action="store_true", help="输出中间产物")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的输出；默认跳过已处理源文件")
    parser.add_argument("--no-split", action="store_true",
                        help="关闭镜头分割，输入整体按单镜头处理")
    parser.add_argument("--split-sensitivity", default=None,
                        choices=["low", "medium", "high"],
                        help="镜头分割灵敏度")
    parser.add_argument("--min-seg-sec", type=float, default=None,
                        help="片段最短时长（秒）")
    parser.add_argument("--max-seg-sec", type=float, default=None,
                        help="片段最长时长（秒）")
    parser.add_argument("--segment-encode", default=None,
                        choices=["auto", "copy", "reencode"],
                        help="片段导出编码模式")
    parser.add_argument("--clean-segments", action="store_true",
                        help="处理完成后删除片段视频，仅保留清单与产物")
    parser.add_argument("--no-asr", action="store_true",
                        help="关闭逐片段语音识别（默认开启）")
    parser.add_argument("--no-describe", action="store_true",
                        help="关闭逐片段视频描述生成（默认开启）")
    parser.add_argument("--describe-limit", type=int, default=None,
                        help="仅处理前 N 个片段的描述（调试用；0/未指定=全部）")
    parser.add_argument("--no-persons", action="store_true",
                        help="关闭人物提取与归类（默认开启）")
    parser.add_argument("--persons-device", default=None,
                        choices=["gpu", "cpu"],
                        help="人物提取推理设备（默认 gpu）")
    parser.add_argument("--persons-domain", default=None,
                        choices=["real", "anime"],
                        help="人物提取模型领域（real=真人 / anime=动漫，默认 anime）")
    parser.add_argument("--persons-max", type=int, default=None,
                        help="仅处理前 N 个源视频的人物提取（调试用；0/未指定=全部）")
    parser.add_argument("--no-separate", action="store_true",
                        help="关闭人声/背景音乐分离（默认开启）")
    parser.add_argument("--separate-model", default=None,
                        choices=["fast", "balanced", "best", "dereverb"],
                        help="人声分离模型档位（默认 fast）")
    parser.add_argument("--separate-device", default=None,
                        choices=["gpu", "cpu"],
                        help="人声分离推理设备（默认 gpu）")
    parser.add_argument("--workers", type=int, default=None,
                        help="逐片段处理的并行线程数；0/未指定=自动")
    parser.add_argument("--opencv-threads", type=int, default=None,
                        help="OpenCV 内部线程数；0/未指定=保持默认")
    parser.add_argument("--interactive", action="store_true",
                        help="交互式选择分割时长 / 是否分割 / 分离强度 / 人物领域 /"
                             " 输入图像 / 描述片段（双击启动默认启用）")
    return parser.parse_args(argv)


def _resolve_workers(cfg, override=None):
    if override is not None:
        return max(1, int(override))
    value = int((cfg.get("performance") or {}).get("workers", 0) or 0)
    if value > 0:
        return value
    return max(1, min(8, os.cpu_count() or 1))


def _apply_overrides(cfg, args):
    seg = cfg["segmentation"]
    if args.no_split:
        seg["enabled"] = False
    if args.split_sensitivity:
        seg["sensitivity"] = args.split_sensitivity
    if args.min_seg_sec is not None:
        seg["min_segment_sec"] = float(args.min_seg_sec)
    if args.max_seg_sec is not None:
        seg["max_segment_sec"] = float(args.max_seg_sec)
    if args.segment_encode:
        seg["segment_encode"] = args.segment_encode
    if args.clean_segments:
        seg["keep_segments"] = False
    if args.no_asr:
        cfg["asr"]["enabled"] = False
    if args.no_describe:
        cfg["describe"]["enabled"] = False
    if args.describe_limit is not None:
        cfg["describe"]["limit"] = int(args.describe_limit)
    if args.no_persons:
        cfg["persons"]["enabled"] = False
    if args.persons_device:
        cfg["persons"]["device"] = args.persons_device
    if args.persons_domain:
        cfg["persons"]["domain"] = args.persons_domain
    if args.persons_max is not None:
        cfg["persons"]["limit"] = int(args.persons_max)
    if args.no_separate:
        cfg["separation"]["enabled"] = False
    if args.separate_model:
        cfg["separation"]["model"] = args.separate_model
    if args.separate_device:
        cfg["separation"]["device"] = args.separate_device
    if args.opencv_threads is not None:
        cfg["performance"]["opencv_threads"] = int(args.opencv_threads)


def _timed(timer, name):
    if timer is None:
        return contextlib.nullcontext()
    return timer.stage(name)


class _SegmentJob(object):
    """单个片段的处理上下文。"""

    def __init__(self, input_path, segment_meta, shot_id, out_dir,
                 frame_offset):
        self.input_path = input_path
        self.segment_meta = segment_meta
        self.shot_id = shot_id
        self.out_dir = out_dir
        self.frame_offset = max(0, int(frame_offset or 0))
        self.status = "pending"
        self.reason = ""
        self.gray_frames = None
        self.frame_quality = None
        self.valid_start = 0
        self.valid_end = 0
        self.sample_idx = []
        self.pairs = []
        self.agg = None
        self.fallback = False


def _dump_debug(debug_dir, frame_quality, sample_idx, pairs, frame_reader):
    ensure_dir(debug_dir)
    sampled_dir = os.path.join(debug_dir, "sampled_frames")
    ensure_dir(sampled_dir)
    for index in sample_idx:
        try:
            save_bgr_jpeg(os.path.join(sampled_dir, "frame_%06d.jpg" % index),
                          frame_reader(index), 90)
        except Exception:
            pass

    pairs_payload = []
    for pair in pairs:
        pairs_payload.append({
            "index_i": pair.index_i, "index_j": pair.index_j,
            "match_count": pair.match_count, "inlier_count": pair.inlier_count,
            "inlier_ratio": pair.inlier_ratio, "scale_step": pair.scale_step,
            "rotation_step_deg": pair.rotation_step_deg,
            "translation_step": list(pair.translation_step),
            "perspective_error": pair.perspective_error,
            "flow_magnitude": pair.flow_magnitude,
            "center_inlier_ratio": pair.center_inlier_ratio,
            "success": pair.success,
        })
    write_json(os.path.join(debug_dir, "pairs.json"), pairs_payload)

    lines = ["frame,mean,std,sharpness,quality_score,bad"]
    for index in range(len(frame_quality.mean)):
        lines.append("%d,%.4f,%.4f,%.4f,%.6f,%d" % (
            index, frame_quality.mean[index], frame_quality.std[index],
            frame_quality.sharpness[index], frame_quality.quality_score[index],
            int(frame_quality.bad[index])))
    with open(os.path.join(debug_dir, "quality.csv"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _prepare_segment(job, cfg, timer=None):
    """解码、质量、修剪、采样、特征匹配与尺度轨迹累积。"""
    try:
        frame_offset = job.frame_offset
        end_frame = None
        if frame_offset > 0:
            end_frame = frame_offset + int(job.segment_meta.frame_count)

        with _timed(timer, "prepare.decode"):
            gray_frames = video_io.decode_gray_frames(
                job.input_path, long_side=960, start_frame=frame_offset,
                end_frame=end_frame)
        with _timed(timer, "prepare.quality"):
            frame_quality = quality_mod.assess_frames(gray_frames, cfg)
        with _timed(timer, "prepare.trim_sample"):
            valid_start, valid_end, status = trim.find_valid_range(
                frame_quality, cfg)

        if status == "invalid":
            job.status = "invalid"
            job.reason = trim.invalid_reason(frame_quality)
            return job

        valid_duration = (valid_end - valid_start + 1) / float(
            job.segment_meta.fps)
        sample_idx = sampling.sample_frames(valid_start, valid_end,
                                            valid_duration, cfg)

        with _timed(timer, "prepare.match"):
            feature_list = [features.extract_features(gray_frames[i], cfg)
                            for i in sample_idx]
            pairs = []
            for position in range(len(sample_idx) - 1):
                idx_i, idx_j = sample_idx[position], sample_idx[position + 1]
                matched = features.match_pair(feature_list[position],
                                              feature_list[position + 1], cfg)
                if matched is None:
                    pairs.append(motion.failed_pair(
                        idx_i, idx_j, frame_quality.sharpness[idx_i]))
                else:
                    evidence = motion.estimate_pair(
                        matched[0], matched[1], gray_frames[idx_i].shape,
                        frame_quality.sharpness[idx_i], cfg)
                    evidence.index_i, evidence.index_j = idx_i, idx_j
                    pairs.append(evidence)
        with _timed(timer, "prepare.accumulate"):
            agg = motion.accumulate_base(
                pairs, sample_idx, gray_frames[0].shape, cfg,
                frame_quality.sharpness_median,
                frame_quality.low_sharpness_ratio,
                job.segment_meta.duration_sec)

        job.gray_frames = gray_frames
        job.frame_quality = frame_quality
        job.valid_start = valid_start
        job.valid_end = valid_end
        job.sample_idx = sample_idx
        job.pairs = pairs
        job.agg = agg
        job.status = "prepared"
    except Exception as exc:
        job.status = "error"
        job.reason = str(exc)
    return job


def _finalize_segment(job, cfg, logger, debug=False, timer=None):
    """按均匀时间间隔选帧，生成 3x3 宫格故事图 `scene_reference.jpg`。"""
    ensure_dir(shot_detail_dir(job.out_dir))
    if job.status == "invalid":
        logger.info("  %s invalid (%s)", job.shot_id, job.reason)
        return "invalid"
    if job.status == "error":
        logger.error("  %s failed: %s", job.shot_id, job.reason)
        return "error"

    segment_meta = job.segment_meta
    reader = None
    with _timed(timer, "finalize.reference"):
        reader = video_io.make_frame_reader(
            job.input_path, max_side=int(cfg["output"]["max_side_frame"]),
            frame_offset=job.frame_offset,
            frame_size=(segment_meta.width, segment_meta.height))
        # 按帧号升序一次性预读故事图所需帧，避免 reader 多次回退重复顺序解码。
        for target in sorted(set(generator.reference_indices(
                job.frame_quality, job.valid_start, job.valid_end,
                segment_meta.fps, cfg))):
            try:
                reader(target)
            except Exception:
                pass
        reference = generator.generate(
            job.frame_quality, reader, (job.valid_start, job.valid_end),
            job.valid_start, job.valid_end, segment_meta.fps, cfg)
        if reference is None:
            job.fallback = True
            logger.warning("  %s scene_reference generation failed",
                           job.shot_id)
        else:
            save_bgr_jpeg(os.path.join(job.out_dir, "scene_reference.jpg"),
                          reference, int(cfg["output"]["jpeg_quality"]))

    if debug:
        try:
            _dump_debug(os.path.join(shot_detail_dir(job.out_dir), "debug"),
                        job.frame_quality, job.sample_idx, job.pairs, reader)
        except Exception as exc:
            logger.warning("  debug dump failed: %s", exc)

    logger.info(
        "  %s valid_range=[%d,%d] scene_reference=%s",
        job.shot_id, job.valid_start, job.valid_end,
        "failed" if job.fallback else "ok")
    return "ok"


def _process_job(job, cfg, logger, debug, timer):
    _prepare_segment(job, cfg, timer)
    try:
        result = _finalize_segment(job, cfg, logger, debug, timer)
    except Exception as exc:
        logger.error("  %s failed: %s", job.shot_id, exc)
        logger.debug(traceback.format_exc())
        result = "error"
    return result


def _build_segments(path, meta, cfg, logger):
    seg_cfg = cfg["segmentation"]
    if not seg_cfg.get("enabled", True):
        segment = Segment(1, 0, meta.frame_count, 0.0, meta.duration_sec,
                          meta.duration_sec, "none",
                          shot_id="shot_0001", shot_dir="shot_0001")
        return [segment], {"method": "none", "detected_shot_count": 1,
                           "segment_count": 1, "split_iterations": 0,
                           "fallback": False}

    cuts, detect_meta = segmentation.detect_shot_cuts(
        path, meta.fps, seg_cfg["sensitivity"], seg_cfg["min_scene_len_sec"],
        cfg)
    if cuts is None:
        logger.warning(
            "  shot detect unavailable/failed (%s); using equal-length split",
            detect_meta.get("reason", "unknown"))

    subdetect_fn = None
    if seg_cfg.get("long_shot_subdetect", True):
        subdetect_fn = segmentation.make_subdetector(
            path, seg_cfg.get("long_shot_subdetect_sensitivity", "high"),
            seg_cfg["min_scene_len_sec"], cfg)

    segments, build_meta = segmentation.build_segments(
        meta.frame_count, meta.fps, cuts, cfg, subdetect_fn)
    build_meta["method"] = detect_meta.get("method", "none")
    return segments, build_meta


def _enumerate_source(path, cfg, logger):
    """交互模式下预先枚举片段，供第四步按编号选择描述片段。

    只做镜头分割（不导出片段、不生成故事图），结果缓存在 `choices.precomputed`
    中，`process_source` 直接复用，不会重复分割。
    """
    meta = video_io.read_metadata(path)
    segments, build_meta = _build_segments(path, meta, cfg, logger)
    return {"meta": meta, "segments": segments, "build_meta": build_meta}


def _enumerator(output_root, overwrite, cfg, logger):
    """构造交互枚举回调；已被处理且不覆盖的源视频返回 None（不列入选择）。"""
    def _enumerate(path):
        source_dir = os.path.join(output_root, "segments", safe_stem(path))
        manifest = os.path.join(source_dir, "segments.json")
        if not overwrite and os.path.isfile(manifest):
            return None
        return _enumerate_source(path, cfg, logger)
    return _enumerate


def process_source(path, output_root, cfg, logger, args, stats,
                   speech_jobs, cleanup, separation_jobs=None, pool=None,
                   timer=None, describe_jobs=None, persons_jobs=None,
                   precomputed=None):
    cached = (precomputed or {}).get(path)
    if cached is not None:
        meta = cached["meta"]
    else:
        meta = video_io.read_metadata(path)
    source_file = os.path.basename(path)
    stem = safe_stem(path)
    source_dir = os.path.join(output_root, "segments", stem)
    segments_json = os.path.join(source_dir, "segments.json")

    if os.path.isdir(source_dir) and os.path.isfile(segments_json) and \
            not args.overwrite:
        logger.info("skip (exists): %s", source_file)
        stats["skipped"] += 1
        return
    if args.overwrite and os.path.isdir(source_dir):
        shutil.rmtree(source_dir, ignore_errors=True)
    ensure_dir(source_dir)

    if cfg["persons"].get("enabled", True) and persons_jobs is not None:
        persons_jobs.append({"video": path, "source_dir": source_dir})

    logger.info("processing: %s (%.2fs, %d frames)", source_file,
                meta.duration_sec, meta.frame_count)

    if cached is not None:
        segments = cached["segments"]
        build_meta = cached["build_meta"]
    else:
        with _timed(timer, "source.segmentation"):
            segments, build_meta = _build_segments(path, meta, cfg, logger)
    if not segments:
        logger.warning("  no segments produced for %s", source_file)
        stats["error"] += 1
        return

    seg_cfg = cfg["segmentation"]
    encode_mode = seg_cfg.get("segment_encode", "auto")

    def _export_one(segment):
        shot_dir = os.path.join(source_dir, segment.shot_dir)
        ensure_dir(shot_dir)
        ensure_dir(shot_detail_dir(shot_dir))
        if segment.method == "none":
            segment.file = ""
            return
        seg_name = "seg_%04d.mp4" % segment.index
        seg_path = os.path.join(shot_dir, seg_name)
        exported = segment_export.export_segment(
            path, segment.start_sec, segment.end_sec, seg_path,
            encode_mode, cfg)
        if exported:
            segment.file = seg_name
        else:
            segment.file = ""
            logger.warning("  segment %d export failed; processing in memory",
                           segment.index)

    with _timed(timer, "source.export"):
        if pool is not None and len(segments) > 1:
            list(pool.map(_export_one, segments))
        else:
            for segment in segments:
                _export_one(segment)

    durations = [seg.duration_sec for seg in segments]
    logger.info("  split: shots=%d segments=%d iterations=%d fallback=%s "
                "duration=[%.2f, %.2f]s",
                build_meta.get("detected_shot_count", 0), len(segments),
                build_meta.get("split_iterations", 0),
                build_meta.get("fallback", False), min(durations),
                max(durations))

    segments_payload = {
        "schema_version": SEGMENTS_SCHEMA_VERSION,
        "source": {
            "file": source_file,
            "fps": round(float(meta.fps), 3),
            "frame_count": int(meta.frame_count),
            "duration_sec": round(float(meta.duration_sec), 3),
        },
        "sensitivity": seg_cfg["sensitivity"],
        "min_segment_sec": float(seg_cfg["min_segment_sec"]),
        "max_segment_sec": float(seg_cfg["max_segment_sec"]),
        "detected_shot_count": int(build_meta.get("detected_shot_count", 0)),
        "segment_count": len(segments),
        "split_iterations": int(build_meta.get("split_iterations", 0)),
        "fallback": bool(build_meta.get("fallback", False)),
        "segments": [
            {
                "index": int(seg.index),
                "shot_id": seg.shot_id,
                "start_frame": int(seg.start_frame),
                "end_frame": int(seg.end_frame),
                "start_sec": round(float(seg.start_sec), 3),
                "end_sec": round(float(seg.end_sec), 3),
                "duration_sec": round(float(seg.duration_sec), 3),
                "method": seg.method,
                "below_min": bool(seg.below_min),
                "file": seg.file or None,
                "shot_dir": seg.shot_dir,
            }
            for seg in segments
        ],
    }
    write_json(segments_json, segments_payload)

    jobs = []
    for segment in segments:
        shot_dir = os.path.join(source_dir, segment.shot_dir)
        if segment.file:
            input_path = os.path.join(shot_dir, segment.file)
            try:
                segment_meta = video_io.read_metadata(input_path)
            except Exception:
                segment_meta = VideoMeta(meta.fps, meta.width, meta.height,
                                         segment.end_frame - segment.start_frame,
                                         segment.duration_sec)
            frame_offset = 0
        else:
            input_path = path
            segment_meta = VideoMeta(
                meta.fps, meta.width, meta.height,
                segment.end_frame - segment.start_frame, segment.duration_sec)
            frame_offset = segment.start_frame
        jobs.append(_SegmentJob(input_path, segment_meta, segment.shot_id,
                                shot_dir, frame_offset))

    with _timed(timer, "source.scene_reference"):
        if pool is not None and len(jobs) > 1:
            results = list(pool.map(
                lambda item: _process_job(item, cfg, logger, args.debug,
                                          timer), jobs))
        else:
            results = [_process_job(item, cfg, logger, args.debug, timer)
                       for item in jobs]
    for item, result in zip(jobs, results):
        stats[result] = stats.get(result, 0) + 1
        if result == "ok" and item.fallback:
            stats["fallback"] += 1

    shots = [{
        "shot_id": segment.shot_id,
        "shot_dir": os.path.join(source_dir, segment.shot_dir),
        "file": segment.file or "",
        "start_sec": float(segment.start_sec),
        "end_sec": float(segment.end_sec),
    } for segment in segments]
    describe_shots = [shot for shot, item in zip(shots, jobs)
                      if item.status != "invalid"]

    if cfg["separation"].get("enabled", True) and separation_jobs is not None:
        separation_jobs.append({"video": path, "source_dir": source_dir,
                                "shots": shots})

    if cfg["asr"].get("enabled", True):
        speech_jobs.append({"video": path, "source_dir": source_dir,
                            "shots": shots})

    if cfg["describe"].get("enabled", True) and describe_jobs is not None:
        describe_jobs.append({"video": path, "source_file": source_file,
                              "source_dir": source_dir,
                              "shots": describe_shots or shots})

    if not seg_cfg.get("keep_segments", True):
        for segment in segments:
            if segment.file:
                cleanup.append(os.path.join(source_dir, segment.shot_dir,
                                            segment.file))


def main(argv=None):
    args = parse_args(argv)
    timer = StageTimer()
    start_wall = time.perf_counter()
    input_dir = _resolve(args.input)
    output_dir = _resolve(args.output)
    config_path = _resolve(args.config)
    cfg = load_config(config_path)
    _apply_overrides(cfg, args)

    opencv_threads = int((cfg.get("performance") or {}).get(
        "opencv_threads", 0) or 0)
    if opencv_threads > 0:
        import cv2
        cv2.setNumThreads(opencv_threads)

    ensure_dir(output_dir)
    logger = setup_logger(os.path.join(output_dir, "run.log"))

    if args.single:
        videos = [_resolve(args.single)]
    else:
        videos = list_videos(input_dir)

    logger.info("input=%s output=%s videos=%d split=%s",
                input_dir, output_dir, len(videos),
                cfg["segmentation"].get("enabled", True))

    choices = None
    precomputed = None
    if args.interactive:
        choices = interactive.collect_choices(
            videos, cfg, logger,
            enumerate_fn=_enumerator(output_dir, bool(args.overwrite), cfg,
                                     logger))
        precomputed = choices.precomputed

    with _timed(timer, "check.engines"):
        if cfg["asr"].get("enabled", True):
            speech_ready, missing = speech.available(cfg)
            if speech_ready:
                logger.info(
                    "asr=%s device=%s models=%s language=%s center=%s",
                    cfg["asr"].get("engine"), cfg["asr"].get("device"),
                    cfg["asr"].get("models"), cfg["asr"].get("language"),
                    cfg["asr"].get("voiceprint_center") or "<output>/voiceprint_center.json")
            else:
                logger.warning("asr disabled; missing: %s", ", ".join(missing))
                cfg["asr"]["enabled"] = False

        if cfg["separation"].get("enabled", True):
            sep_ready, sep_missing = speech.separation_available(cfg)
            if sep_ready:
                logger.info(
                    "separation=%s device=%s models=%s",
                    cfg["separation"].get("model"),
                    cfg["separation"].get("device"),
                    cfg["separation"].get("models"))
            else:
                logger.warning("separation disabled; missing: %s",
                               ", ".join(sep_missing))
                cfg["separation"]["enabled"] = False

        if cfg["describe"].get("enabled", True):
            desc_ready, desc_missing = describe.available(cfg)
            if desc_ready:
                logger.info(
                    "describe=%s model=%s input=%s frames=%sfps max_side=%s",
                    cfg["describe"].get("engine"),
                    cfg["describe"].get("model"),
                    cfg["describe"].get("input_mode"),
                    cfg["describe"].get("frame_fps"),
                    cfg["describe"].get("frame_max_side"))
            else:
                logger.warning("describe disabled; missing: %s",
                               ", ".join(desc_missing))
                cfg["describe"]["enabled"] = False

        if cfg["persons"].get("enabled", True):
            persons_ready, persons_missing = persons.available(cfg)
            if persons_ready:
                logger.info(
                    "persons domain=%s models=%s device=%s pose=%s face=%s",
                    cfg["persons"].get("domain"),
                    cfg["persons"].get("models"),
                    cfg["persons"].get("device"),
                    cfg["persons"].get("pose_model"),
                    cfg["persons"].get("face_pack"))
            else:
                logger.warning("persons disabled; missing: %s",
                               ", ".join(persons_missing))
                cfg["persons"]["enabled"] = False

    if not videos:
        logger.warning("no input videos found")
        return 0

    stats = {"ok": 0, "invalid": 0, "error": 0, "skipped": 0, "fallback": 0,
             "asr_ok": 0, "asr_error": 0, "asr_segments": 0,
             "asr_speakers": 0, "separation_ok": 0, "separation_skip": 0,
             "separation_error": 0, "describe_ok": 0, "describe_error": 0,
             "describe_shots": 0, "describe_skipped": 0,
             "persons_ok": 0, "persons_error": 0, "persons_skipped": 0,
             "persons_count": 0, "persons_unknown": 0}
    speech_jobs = []
    separation_jobs = []
    describe_jobs = []
    persons_jobs = []
    cleanup = []
    workers = _resolve_workers(cfg, args.workers)
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    logger.info("workers=%d", workers)
    with _timed(timer, "phase.scene_reference"):
        try:
            for path in videos:
                try:
                    process_source(path, output_dir, cfg, logger, args, stats,
                                   speech_jobs, cleanup,
                                   separation_jobs=separation_jobs, pool=pool,
                                   timer=timer, describe_jobs=describe_jobs,
                                   persons_jobs=persons_jobs,
                                   precomputed=precomputed)
                except Exception as exc:
                    logger.error("source failed: %s: %s",
                                 os.path.basename(path), exc)
                    logger.debug(traceback.format_exc())
                    stats["error"] += 1
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

    # 交互选择：按编号过滤需要生成视频描述的片段（不选描述则关闭该阶段）。
    if choices is not None:
        if choices.describe_none:
            cfg["describe"]["enabled"] = False
            describe_jobs = []
        elif not choices.describe_all:
            for job in describe_jobs:
                allowed = choices.describe_shots.get(job["video"], set())
                job["shots"] = [shot for shot in job["shots"]
                                if shot.get("shot_id") in allowed]
            describe_jobs = [job for job in describe_jobs if job["shots"]]

    # 人声分离阶段：模型只加载一次；输出原始/人声/伴奏三轨，人声轨供后续 ASR 使用。
    separation_result = {}
    if cfg["separation"].get("enabled", True) and separation_jobs:
        with _timed(timer, "phase.separation"):
            separation_result = speech.run_separation_phase(
                separation_jobs, cfg, logger, stats=stats)
        for job in speech_jobs:
            info = separation_result.get(job["video"])
            if info and info.get("vocals"):
                job["audio_source"] = info["vocals"]

    # 语音识别阶段：模型只加载一次；全部视频转写完成后立即停止并卸载模型（释放显存）
    if cfg["asr"].get("enabled", True) and speech_jobs:
        logger.info("asr phase: %d source(s)", len(speech_jobs))
        with _timed(timer, "phase.asr"):
            result = speech.run_phase(speech_jobs, output_dir, cfg, logger,
                                      debug=args.debug)
        for key, value in result.items():
            stats[key] = stats.get(key, 0) + value

    # 人物提取与归类阶段：在统一流程（分割→故事图→分离→ASR）之后，整片处理；
    # 模型只加载一次，结束卸载。
    persons_limit = int((cfg.get("persons") or {}).get("limit", 0) or 0)
    if cfg["persons"].get("enabled", True) and persons_jobs:
        if persons_limit > 0:
            persons_jobs = persons_jobs[:persons_limit]
        logger.info("persons phase: %d source(s)", len(persons_jobs))
        with _timed(timer, "phase.persons"):
            result = persons.run_phase(persons_jobs, output_dir, cfg, logger,
                                       stats=stats, debug=args.debug)
        for key, value in result.items():
            stats[key] = stats.get(key, 0) + value

    # 视频描述阶段：ASR 之后逐片段生成中文 prompt；模型只加载一次，结束卸载
    if cfg["describe"].get("enabled", True) and describe_jobs:
        logger.info("describe phase: %d source(s)", len(describe_jobs))
        with _timed(timer, "phase.describe"):
            result = describe.run_phase(describe_jobs, output_dir, cfg, logger,
                                        debug=args.debug)
        for key, value in result.items():
            stats[key] = stats.get(key, 0) + value

    # --clean-segments：待 ASR / 描述阶段读取完毕后再删除片段视频
    with _timed(timer, "phase.cleanup"):
        for seg_file in cleanup:
            try:
                os.remove(seg_file)
            except OSError:
                pass

    logger.info(
        "done: ok=%d invalid=%d error=%d skipped=%d fallback=%d "
        "separation_ok=%d separation_skip=%d separation_error=%d "
        "asr_ok=%d asr_error=%d asr_segments=%d "
        "describe_ok=%d describe_error=%d describe_skipped=%d "
        "persons_ok=%d persons_error=%d persons_count=%d persons_unknown=%d",
        stats["ok"], stats["invalid"], stats["error"], stats["skipped"],
        stats["fallback"], stats["separation_ok"], stats["separation_skip"],
        stats["separation_error"], stats["asr_ok"], stats["asr_error"],
        stats["asr_segments"], stats["describe_ok"], stats["describe_error"],
        stats["describe_skipped"], stats["persons_ok"], stats["persons_error"],
        stats["persons_count"], stats["persons_unknown"])
    timer.add("total", time.perf_counter() - start_wall)
    timer.report(logger)
    return 0


if __name__ == "__main__":
    sys.exit(main())
