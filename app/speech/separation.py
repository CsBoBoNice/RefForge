"""人声与背景音乐分离（双轨）。

对每个源视频：
1. 用 ffmpeg 抽取**原始音轨** -> `output/segments/<源名>/audio/original.wav`；
2. 用 audio-separator 分离出**人声**与**伴奏** -> `vocals.wav` / `instrumental.wav`；
3. 写 `audio/separation.json` 记录模型档位 / 设备 / 耗时。

后续 ASR 与按人归档均改用分离后的 `vocals.wav`，以提高识别质量与说话人资料纯度。

设备：`cfg.separation.device = "gpu"`（默认 cuda）/ `"cpu"`；GPU 不可用时自动回退 CPU。
模型：全部本地（`separation.models`），运行期禁止联网；模型缺失时整段跳过并告警。
"""

import gc
import logging
import os
import shutil
import subprocess
import tempfile
import time
import traceback

from io_utils import (bin_dir, ensure_dir, ffmpeg_path, package_root,
                      shot_detail_dir, write_json)

TIERS = {
    "fast": "UVR-MDX-NET-Voc_FT.onnx",
    "balanced": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "best": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
    "dereverb": "Reverb_HQ_By_FoxJoy.onnx",
}
DEFAULT_TIER = "fast"
CUSTOM_NAMES = {
    "Vocals": "vocals",
    "Instrumental": "instrumental",
    "No Vocals": "instrumental",
    "Inst": "instrumental",
    # 去混响档（Reverb_HQ_By_FoxJoy）：No Reverb 为干净人声，Reverb 为剔除的混响。
    "No Reverb": "vocals",
    "Reverb": "instrumental",
}


def config(cfg):
    return cfg.get("separation") or {}


def enabled(cfg):
    return bool(config(cfg).get("enabled", True))


def resolve_model_dir(cfg):
    value = str(config(cfg).get("models") or "models/separator").strip()
    if os.path.isabs(value):
        return os.path.normpath(value)
    return os.path.normpath(os.path.join(package_root(), value))


def tier_of(cfg):
    tier = str(config(cfg).get("model", DEFAULT_TIER) or DEFAULT_TIER)
    return tier if tier in TIERS else DEFAULT_TIER


def model_filename(cfg, tier=None):
    return TIERS.get(tier or tier_of(cfg), TIERS[DEFAULT_TIER])


def model_path(cfg, tier=None):
    return os.path.join(resolve_model_dir(cfg), model_filename(cfg, tier))


def available(cfg):
    """返回 (ok, missing)；只检查模块与本地模型文件是否存在。"""
    import importlib.util
    missing = []
    if importlib.util.find_spec("audio_separator") is None:
        missing.append("python-package:audio-separator")
    path = model_path(cfg)
    if not os.path.isfile(path):
        missing.append(path)
    return (not missing), missing


def resolve_device(cfg):
    requested = str(config(cfg).get("device", "gpu") or "gpu").lower()
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    return "gpu" if torch.cuda.is_available() else "cpu"


def _ensure_ffmpeg_on_path():
    folder = bin_dir()
    if not os.path.isdir(folder):
        return
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if folder not in parts:
        os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")


def extract_original_audio(video_path, out_path):
    """抽取原始音轨（保留原始采样率 / 声道，PCM 16-bit WAV）。"""
    exe = ffmpeg_path()
    if not os.path.isfile(exe):
        return False
    ensure_dir(os.path.dirname(out_path))
    cmd = [exe, "-y", "-v", "error", "-i", video_path, "-vn",
           "-c:a", "pcm_s16le", out_path]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    except OSError:
        return False
    return (result.returncode == 0 and os.path.isfile(out_path)
            and os.path.getsize(out_path) > 44)


def cut_wav(src_path, out_path, start_sec, end_sec):
    """按秒区间切出 WAV 片段（保留原始声道/采样率，PCM 16-bit）。"""
    duration = float(end_sec) - float(start_sec)
    if duration <= 0:
        return False
    exe = ffmpeg_path()
    if not os.path.isfile(exe):
        return False
    ensure_dir(os.path.dirname(out_path))
    cmd = [exe, "-y", "-v", "error", "-ss", "%.3f" % float(start_sec),
           "-i", src_path, "-t", "%.3f" % duration,
           "-c:a", "pcm_s16le", out_path]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    except OSError:
        return False
    return (result.returncode == 0 and os.path.isfile(out_path)
            and os.path.getsize(out_path) > 44)


def _pick_stems(files, audio_dir):
    """把分离输出归类为 (vocals, instrumental)，统一返回 audio_dir 下全路径。"""
    candidates = [os.path.join(audio_dir, os.path.basename(path))
                  for path in (files or [])]
    if os.path.isdir(audio_dir):
        for name in os.listdir(audio_dir):
            path = os.path.join(audio_dir, name)
            if path not in candidates:
                candidates.append(path)
    vocals = None
    instrumental = None
    for full in candidates:
        if not os.path.isfile(full):
            continue
        name = os.path.basename(full).lower()
        if "no_reverb" in name or "no reverb" in name:
            vocals = full
        elif ("vocal" in name and "no" not in name):
            vocals = full
        elif ("instrument" in name or "no_vocal" in name
              or "no vocal" in name or name.startswith("inst")
              or "reverb" in name):
            instrumental = full
    return vocals, instrumental


class SeparationEngine:
    """封装 audio-separator，模型只加载一次并在全部视频间复用。"""

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.device = resolve_device(cfg)
        self.model_file = model_path(cfg)
        self._separator = None
        self._temp_output = tempfile.mkdtemp(prefix="separation_")

    def load(self):
        _ensure_ffmpeg_on_path()
        from audio_separator.separator import Separator
        model_dir = resolve_model_dir(self.cfg)
        ensure_dir(model_dir)
        self._separator = Separator(
            model_file_dir=model_dir,
            output_dir=self._temp_output,
            output_format="WAV",
            sample_rate=44100,
            log_level=logging.WARNING,
        )
        if self.device == "cpu":
            import torch
            self._separator.torch_device = torch.device("cpu")
            self._separator.torch_device_cpu = torch.device("cpu")
            self._separator.onnx_execution_provider = ["CPUExecutionProvider"]
        self._separator.load_model(
            model_filename=os.path.basename(self.model_file))

    def separate(self, original_wav, audio_dir):
        self._separator.output_dir = audio_dir
        instance = getattr(self._separator, "model_instance", None)
        if instance is not None:
            instance.output_dir = audio_dir
        return self._separator.separate(
            original_wav, custom_output_names=CUSTOM_NAMES)

    def unload(self):
        self._separator = None
        shutil.rmtree(self._temp_output, ignore_errors=True)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def process_video(job, engine, cfg, logger, stats):
    """处理单个源视频；成功返回音频信息字典，跳过返回 None。"""
    from .audio import has_audio_stream

    video_path = job["video"]
    source_dir = job["source_dir"]
    source_file = os.path.basename(video_path)
    audio_dir = os.path.join(source_dir, "audio")

    has_audio = has_audio_stream(video_path)
    if has_audio is False:
        logger.info("  separation %s: no audio stream, skipped", source_file)
        stats["separation_skip"] = stats.get("separation_skip", 0) + 1
        return None

    ensure_dir(audio_dir)
    original = os.path.join(audio_dir, "original.wav")
    if not extract_original_audio(video_path, original):
        logger.warning("  separation %s: original audio extraction failed",
                       source_file)
        stats["separation_error"] = stats.get("separation_error", 0) + 1
        return None

    started = time.monotonic()
    try:
        files = engine.separate(original, audio_dir)
    except Exception as exc:
        logger.error("  separation %s failed: %s", source_file, exc)
        logger.debug(traceback.format_exc())
        stats["separation_error"] = stats.get("separation_error", 0) + 1
        return None

    vocals, instrumental = _pick_stems(files, audio_dir)
    if vocals is None:
        logger.warning("  separation %s: vocals stem missing", source_file)
        stats["separation_error"] = stats.get("separation_error", 0) + 1
        return None

    elapsed = time.monotonic() - started
    payload = {
        "schema_version": "1.0",
        "source": source_file,
        "tier": tier_of(cfg),
        "model_file": model_filename(cfg),
        "device": engine.device,
        "sample_rate": 44100,
        "elapsed_sec": round(elapsed, 3),
        "tracks": {
            "original": "original.wav",
            "vocals": os.path.basename(vocals),
            "instrumental": (os.path.basename(instrumental)
                             if instrumental else None),
        },
    }
    write_json(os.path.join(audio_dir, "separation.json"), payload)

    # 每个分割片段：原始音轨放 shot 根目录，人声/伴奏放 shot/detailed/。
    for shot in job.get("shots") or []:
        shot_dir = shot.get("shot_dir")
        if not shot_dir:
            continue
        detail_dir = shot_detail_dir(shot_dir)
        shot_start = float(shot.get("start_sec", 0.0))
        shot_end = float(shot.get("end_sec", 0.0))
        cut_wav(original, os.path.join(shot_dir, "original.wav"),
                shot_start, shot_end)
        cut_wav(vocals, os.path.join(detail_dir, "vocals.wav"),
                shot_start, shot_end)
        if instrumental:
            cut_wav(instrumental, os.path.join(detail_dir, "instrumental.wav"),
                    shot_start, shot_end)

    stats["separation_ok"] = stats.get("separation_ok", 0) + 1
    logger.info("  separation %s: tier=%s device=%s elapsed=%.1fs",
                source_file, tier_of(cfg), engine.device, elapsed)
    return {"audio_dir": audio_dir, "original": original, "vocals": vocals,
            "instrumental": instrumental, "device": engine.device}


def run_phase(jobs, cfg, logger, stats=None):
    """分离阶段：加载模型一次，逐视频处理后卸载。返回 {video: info}。"""
    result = {}
    local = {"separation_ok": 0, "separation_skip": 0, "separation_error": 0}
    if not jobs or not enabled(cfg):
        return result

    ready, missing = available(cfg)
    if not ready:
        logger.warning("separation disabled; missing: %s", ", ".join(missing))
        return result

    from .audio import has_audio_stream
    audible = [job for job in jobs
               if has_audio_stream(job["video"]) is not False]
    if not audible:
        logger.info("separation phase: no source with audio; skipped")
        return result
    jobs = audible

    engine = SeparationEngine(cfg, logger)
    started = time.monotonic()
    try:
        engine.load()
        logger.info("separation phase: %d source(s), tier=%s model=%s device=%s",
                    len(jobs), tier_of(cfg), model_filename(cfg), engine.device)
        for job in jobs:
            try:
                info = process_video(job, engine, cfg, logger, local)
            except Exception as exc:
                local["separation_error"] += 1
                logger.error("  separation %s failed: %s",
                             os.path.basename(job.get("video", "?")), exc)
                logger.debug(traceback.format_exc())
                info = None
            if info:
                result[job["video"]] = info
    finally:
        engine.unload()
        logger.info(
            "separation done: ok=%d skip=%d error=%d elapsed=%.1fs",
            local["separation_ok"], local["separation_skip"],
            local["separation_error"], time.monotonic() - started)
    if stats is not None:
        for key, value in local.items():
            stats[key] = stats.get(key, 0) + value
    return result
