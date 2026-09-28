"""配置加载与默认值合并。

所有阈值集中在此处的 DEFAULT_CONFIG，与包根目录 config.json 保持一致；
用户配置文件缺失或字段缺失时使用默认值。
"""

import copy
import json
import os


DEFAULT_CONFIG = {
    "frame_quality": {
        "black_mean": 15,
        "black_std": 10,
        "white_mean": 240,
        "white_std": 10,
        "low_info_std": 8,
        "blur_ratio_of_median": 0.4,
        "exposure_ideal_min": 80,
        "exposure_ideal_max": 200,
    },
    "trim": {
        "trim_tolerance_frames": 2,
        "min_valid_frames": 2,
        "invalid_black_ratio": 0.9,
    },
    "sampling": {
        "k_short": 7,
        "k_normal": 10,
        "k_long": 14,
    },
    "performance": {
        "workers": 0,
        "opencv_threads": 0,
    },
    "matching": {
        "max_features": 2000,
        "ratio_test": 0.75,
        "ransac_reproj_threshold": 3.0,
        "min_match_count": 8,
        "min_inlier_count": 8,
        "low_inlier_ratio": 0.3,
    },
    "output": {
        "jpeg_quality": 92,
        "max_side_frame": 1920,
        "max_side_reference": 4096,
    },
    "reference": {
        "grid_rows": 3,
        "grid_cols": 3,
        "grid_gap_px": 10,
        "grid_margin_px": 10,
        "select_tolerance_ms": 50,
    },
    "segmentation": {
        "enabled": True,
        "sensitivity": "medium",
        "adaptive_thresholds": {"low": 4.5, "medium": 3.0, "high": 2.0},
        "min_scene_len_sec": 0.5,
        "min_segment_sec": 1.0,
        "max_segment_sec": 15.0,
        "max_split_iterations": 5,
        "long_shot_subdetect": True,
        "long_shot_subdetect_sensitivity": "high",
        "segment_encode": "auto",
        "copy_duration_tolerance_sec": 0.1,
        "keep_segments": True,
    },
    "separation": {
        "enabled": True,
        "model": "fast",
        "models": "models/separator",
        "device": "gpu",
        "sample_rate": 44100,
    },
    "asr": {
        "enabled": True,
        "engine": "qwen3-asr",
        "device": "gpu",
        "dtype": "bfloat16",
        "models": "models/asr",
        "asr_model": "Qwen3-ASR-1.7B",
        "forced_aligner": "Qwen3-ForcedAligner-0.6B",
        "vad_model": "fsmn-vad",
        "speaker_model": "campplus",
        "verify_model": "eres2netv2",
        "language": "auto",
        "max_new_tokens": 2048,
        "max_inference_batch_size": 4,
        "vad_max_segment_sec": 60,
        "min_speech_sec": 0.6,
        "min_archive_sec": 1.0,
        "max_archive_sec": 30.0,
        "local_speaker_threshold": 0.55,
        "diarize_window_sec": 1.2,
        "diarize_hop_sec": 0.4,
        "diarize_smooth_radius": 1,
        "sentence_merge_gap_sec": 0.8,
        "voiceprint_sim_threshold": 0.30,
        "verify_sim_threshold": 0.55,
        "voiceprint_center": "",
        "sample_rate": 16000,
        "debug": False,
    },
    "persons": {
        "enabled": True,
        "domain": "anime",
        "models": "models/person",
        "pose_model": "yolo11x-pose.pt",
        "face_pack": "buffalo_l",
        "device": "gpu",
        "fps_target": 1.0,
        "candidates_per_window": 5,
        "det_conf": 0.45,
        "kp_conf": 0.3,
        "min_face_px": 48,
        "face_det_score": 0.55,
        "face_embed_score": 0.10,
        "cluster_threshold": 0.45,
        "cluster_ambiguous_low": 0.35,
        "dedup_threshold": 0.92,
        "max_per_person": 300,
        "crop_margin_ratio": 0.05,
        "min_crop_width": 32,
        "min_crop_height": 64,
        "jpeg_quality": 92,
        "keep_unknown": True,
        "debug": False,
        "anime": {
            "models": "models/person/anime",
            "person_onnx": "person/person_detect_v1.1_m/model.onnx",
            "face_onnx": "face/face_detect_v1.4_s/model.onnx",
            "clip_model": "clip/clip-vit-large-patch14",
            "input_size": 640,
            "nms_iou": 0.5,
            "head_body_ratio": 7.0,
            "det_conf": 0.30,
            "kp_conf": 0.3,
            "min_face_px": 32,
            "face_det_score": 0.50,
            "face_embed_score": 0.30,
            "cluster_threshold": 0.80,
            "cluster_ambiguous_low": 0.72,
            "dedup_threshold": 0.88,
            "max_per_person": 300,
            "crop_margin_ratio": 0.10,
            "min_crop_width": 32,
            "min_crop_height": 64,
            "clip_batch_size": 8,
        },
    },
    "describe": {
        "enabled": True,
        "engine": "llama.cpp",
        "bin": "llama_cpp/llama_bin",
        "server": "llama-server.exe",
        "models": "models/llm",
        "model": "llm_model.gguf",
        "mmproj": "mmproj_model.gguf",
        "host": "127.0.0.1",
        "port": 8080,
        "context": 50000,
        "gpu_layers": 999,
        "flash_attn": True,
        "jinja": True,
        "reasoning": "off",
        "enable_thinking": False,
        "threads": 0,
        "input_mode": "storyboard",
        "storyboard_max_side": 1920,
        "frame_fps": 2.0,
        "frame_max_side": 1920,
        "frame_dir": "frames",
        "frame_format": "png",
        "frame_select": True,
        "frame_candidates": 3,
        "max_frames": 0,
        "keep_frames": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 4096,
        "repeat_penalty": 1.1,
        "load_timeout_sec": 900,
        "request_timeout_sec": 1800,
        "limit": 0,
    },
}


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path=None):
    """加载配置；缺失文件或字段时回落到内置默认值。"""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                user_cfg = json.load(fh)
            if isinstance(user_cfg, dict):
                cfg = _deep_merge(cfg, user_cfg)
        except (OSError, ValueError):
            cfg = copy.deepcopy(DEFAULT_CONFIG)
    return cfg


def get_cfg(cfg, dotted_key, default=None):
    """按 'a.b.c' 读取嵌套配置。"""
    current = cfg
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
