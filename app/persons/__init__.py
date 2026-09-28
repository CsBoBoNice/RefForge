"""人物提取与归类子系统。

从单个源视频整片提取所有人物，以人脸相似度为主键归类，每人一个文件夹，
内含多张原分辨率矩形截图；全程无脸 track 入 `unknown/`。

对外接口：
    available(cfg) -> (ok, missing)                     # 依赖/模型是否就绪
    run_phase(jobs, output_root, cfg, logger, stats, debug) -> dict
"""

import bisect
import math
import os
import shutil
import traceback

import cv2
import numpy as np

from io_utils import ensure_dir
from pipeline import segmentation, video_io

from . import cluster, crop as crop_mod, export, sampler
from .associate import associate_track_id, owner_index
from .models import available as models_available
from .models import build_models, domain_of


def available(cfg):
    return models_available(cfg)


def _crop_from_capture(capture, frame_index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    return frame if ok else None


def _scene_cuts(video, meta, cfg, logger):
    """用 PySceneDetect 取硬切帧号；track 在切点处强制断开，避免跨人物被串成一条 track。"""
    seg_cfg = cfg.get("segmentation") or {}
    try:
        cuts, _ = segmentation.detect_shot_cuts(
            video, meta.fps, seg_cfg.get("sensitivity", "medium"),
            seg_cfg.get("min_scene_len_sec", 0.5), cfg)
    except Exception as exc:
        logger.warning("  persons: scene cut detection failed (%s)", exc)
        return []
    if not cuts:
        return []
    return sorted(int(value) for value in cuts)


def process_video(job, cfg, models, logger, debug=False):
    video = job["video"]
    source_dir = job["source_dir"]
    persons_dir = os.path.join(source_dir, "persons")
    if os.path.isdir(persons_dir):
        shutil.rmtree(persons_dir, ignore_errors=True)
    ensure_dir(persons_dir)

    pc = models.persons_params(cfg)
    embed_mode = getattr(models, "embed_mode", "face")
    kp_conf = float(pc.get("kp_conf", 0.3))
    margin = float(pc.get("crop_margin_ratio", 0.05))
    min_face_px = float(pc.get("min_face_px", 48))
    face_det_score = float(pc.get("face_det_score", 0.55))
    face_embed_score = float(pc.get("face_embed_score", 0.1))
    dedup_threshold = float(pc.get("dedup_threshold", 0.92))
    max_per_person = int(pc.get("max_per_person", 300))
    min_crop_width = int(pc.get("min_crop_width", 32))
    min_crop_height = int(pc.get("min_crop_height", 64))

    meta = video_io.read_metadata(video)
    frames = sampler.select_representative_frames(video, cfg, logger)
    tracker = models.make_tracker(cfg)
    face_engine = models.face()

    cut_list = _scene_cuts(video, meta, cfg, logger)
    track_map = {}  # (base_track_id, scene_index) -> effective track_id

    records = []
    recent_tracks = {}  # track_id -> (frame_index, bbox)
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise IOError("cannot open video: %s" % video)
    try:
        for frame_index, time_sec in frames:
            frame = _crop_from_capture(capture, frame_index)
            if frame is None:
                continue
            height, width = frame.shape[:2]
            diagonal = math.hypot(width, height) or 1.0
            whole_faces = face_engine.detect(
                frame, int(min_face_px), threshold=face_embed_score)
            detections = tracker.update(frame, whole_faces=whole_faces)
            if not detections:
                continue
            det_boxes = [d["bbox"] for d in detections]
            for order, det in enumerate(detections):
                base_id = associate_track_id(
                    det, order, frame_index, recent_tracks, diagonal)
                det["track_id"] = base_id
                scene = bisect.bisect_right(cut_list, frame_index)
                track_id = track_map.get((base_id, scene))
                if track_id is None:
                    track_id = len(track_map) + 1
                    track_map[(base_id, scene)] = track_id
                rect = crop_mod.crop_rect(
                    det["bbox"], det["kps"], det["kps_conf"], kp_conf, margin,
                    width, height)
                if rect is None:
                    continue
                x1, y1, x2, y2 = rect
                if (x2 - x1) < min_crop_width or (y2 - y1) < min_crop_height:
                    continue
                crop_image = frame[y1:y2, x1:x2]
                if crop_image.size == 0:
                    continue
                gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
                sharpness = crop_mod.sharpness_of(gray)
                thumb = cv2.resize(gray, (32, 32)).astype(np.float32)
                level = det.get("level") or crop_mod.classify_level(
                    det.get("kps_conf"), kp_conf)

                candidates = []
                for face_box, score, keypoints in face_engine.detect(
                        crop_image, int(min_face_px),
                        threshold=face_embed_score):
                    candidates.append((face_box, score, keypoints, crop_image,
                                       x1, y1))
                for face_box, score, keypoints in whole_faces:
                    candidates.append((face_box, score, keypoints, frame, 0, 0))

                chosen = None  # (source, keypoints, score, short_side, box)
                for face_box, score, keypoints, source, ox, oy in candidates:
                    if score < face_embed_score:
                        continue
                    face_short = min(face_box[2] - face_box[0],
                                     face_box[3] - face_box[1])
                    if face_short < min_face_px:
                        continue
                    center_x = (face_box[0] + face_box[2]) / 2.0 + ox
                    center_y = (face_box[1] + face_box[3]) / 2.0 + oy
                    if owner_index(det_boxes, (center_x, center_y)) != order:
                        continue
                    if chosen is None or score > chosen[2]:
                        chosen = (source, keypoints, score, face_short, face_box)

                embedding = None
                face_score = None
                face_size = 0.0
                face_low = False
                if embed_mode == "person":
                    # 动漫：始终对人物裁剪图做 CLIP 嵌入，背影/侧脸帧也能参与聚类
                    embedding = face_engine.embed(crop_image, None)
                    if chosen is not None:
                        face_score = chosen[2]
                        face_size = float(chosen[3])
                        face_low = chosen[2] < face_det_score
                elif chosen is not None:
                    embedding = face_engine.embed(chosen[0], chosen[1], chosen[4])
                    if embedding is not None:
                        face_score = chosen[2]
                        face_size = float(chosen[3])
                        face_low = chosen[2] < face_det_score

                records.append({
                    "frame_index": int(frame_index),
                    "time_sec": float(time_sec),
                    "track_id": int(track_id),
                    "conf": float(det["conf"]),
                    "level": level,
                    "sharpness": float(sharpness),
                    "thumb": thumb,
                    "embedding": embedding,
                    "face_score": face_score,
                    "face_size": face_size,
                    "face_low": face_low,
                    "crop": rect,
                })
            for det in detections:
                recent_tracks[det["track_id"]] = (frame_index, det["bbox"])
            for recent_id in list(recent_tracks):
                if frame_index - recent_tracks[recent_id][0] > 3:
                    del recent_tracks[recent_id]
    finally:
        capture.release()

    domain = domain_of(cfg)
    if not records:
        export.write_outputs(persons_dir, [], [], meta, video, pc, logger, domain)
        return {"person_count": 0, "unknown_count": 0}

    features, best_faces = cluster.build_tracks(records)
    conflicts = cluster.conflict_pairs(records)
    mapping = cluster.cluster_tracks(
        features, best_faces, conflicts,
        float(pc.get("cluster_threshold", 0.45)),
        float(pc.get("cluster_ambiguous_low", 0.35)))

    grouped = {}
    unknown_by_track = {}
    for record in records:
        track_id = record["track_id"]
        if track_id in mapping:
            grouped.setdefault(mapping[track_id], []).append(record)
        else:
            unknown_by_track.setdefault(track_id, []).append(record)

    person_groups = []
    for group in grouped.values():
        person_groups.append(_finalize_group(group, dedup_threshold,
                                             max_per_person))
    person_groups = [group for group in person_groups if group]
    person_groups.sort(key=lambda group: (
        -len(group), min(record["frame_index"] for record in group)))

    unknown_groups = []
    for track_id, group in sorted(
            unknown_by_track.items(),
            key=lambda item: min(r["frame_index"] for r in item[1])):
        unknown_groups.append(_finalize_group(group, dedup_threshold,
                                              max_per_person))
    unknown_groups = [group for group in unknown_groups if group]

    metadata = export.write_outputs(persons_dir, person_groups, unknown_groups,
                                    meta, video, pc, logger, domain)
    logger.info("  persons[%s]: %d person(s), %d unknown track(s)", domain,
                metadata["person_count"], metadata["unknown_track_count"])
    return {"person_count": metadata["person_count"],
            "unknown_count": metadata["unknown_track_count"]}


def _finalize_group(group, dedup_threshold, max_per_person):
    max_sharpness = max(record["sharpness"] for record in group)
    for record in group:
        record["score"] = crop_mod.score_of(record, max_sharpness)
    group.sort(key=lambda record: record["score"], reverse=True)
    kept = cluster.dedup_records(group, dedup_threshold)
    return kept[:max_per_person]


def run_phase(jobs, output_root, cfg, logger, stats=None, debug=False):
    result = {"persons_ok": 0, "persons_error": 0, "persons_skipped": 0,
              "persons_count": 0, "persons_unknown": 0}
    if not jobs:
        return result
    models = build_models(cfg)
    logger.info("persons phase: %d source(s) domain=%s device=%s", len(jobs),
                models.domain, models.device)
    try:
        for job in jobs:
            source = os.path.basename(job["video"])
            try:
                info = process_video(job, cfg, models, logger, debug=debug)
                result["persons_ok"] += 1
                result["persons_count"] += info["person_count"]
                result["persons_unknown"] += info["unknown_count"]
                logger.info("  persons done: %s (%d person, %d unknown)",
                            source, info["person_count"], info["unknown_count"])
            except Exception as exc:
                result["persons_error"] += 1
                logger.error("  persons failed: %s: %s", source, exc)
                logger.debug(traceback.format_exc())
    finally:
        models.close()
    return result


__all__ = ["available", "run_phase"]
