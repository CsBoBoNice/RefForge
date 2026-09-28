"""语音模型加载与设备选择（惰性导入，避免主流程被 torch 拖慢）。

统一入口：
    models = get_models(cfg)     # 首次调用时才导入 torch / funasr / qwen_asr
    ...
    shutdown()                   # 识别结束后卸载并释放显存

设备：cfg.asr.device = "gpu"（默认，cuda:0）或 "cpu"；GPU 不可用时自动回退 CPU。
"""

import importlib.util
import os
import threading

from io_utils import asr_model_dir

os.environ.setdefault("MODELSCOPE_NO_NETWORK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_LANGUAGE_MAP = {
    "auto": None,
    "chinese": "Chinese",
    "zh": "Chinese",
    "english": "English",
    "en": "English",
    "japanese": "Japanese",
    "ja": "Japanese",
}


def configure_language(cfg):
    """把 cfg.asr.language 映射为 Qwen3 语言参数（None = 自动）。"""
    value = str((cfg.get("asr") or {}).get("language", "auto") or "auto").strip()
    if value.lower() in _LANGUAGE_MAP:
        return _LANGUAGE_MAP[value.lower()]
    if value.lower() in ("none", "null", ""):
        return None
    return value[:1].upper() + value[1:].lower()


def resolve_model_dir(cfg, key, default):
    """解析 cfg.asr.<key> 指向的模型目录（相对 cfg.asr.models / 包根）。"""
    asr = cfg.get("asr") or {}
    name = str(asr.get(key) or default).strip()
    if os.path.isabs(name):
        return os.path.normpath(name)
    return os.path.normpath(os.path.join(asr_model_dir(cfg), name))


def _model_specs(cfg):
    return [
        ("asr_model", "Qwen3-ASR-1.7B"),
        ("forced_aligner", "Qwen3-ForcedAligner-0.6B"),
        ("vad_model", "fsmn-vad"),
        ("speaker_model", "campplus"),
        ("verify_model", "eres2netv2"),
    ]


def available(cfg):
    """快速可用性检查（只查文件与模块是否存在，不触发重导入）。

    返回 (ok, missing)，missing 为缺失的目录/模块描述。
    """
    asr = cfg.get("asr") or {}
    missing = []
    for key, default in _model_specs(cfg):
        path = resolve_model_dir(cfg, key, default)
        if not os.path.isdir(path):
            missing.append(path)
    for module in ("torch", "funasr", "qwen_asr"):
        if importlib.util.find_spec(module) is None:
            missing.append("python-package:%s" % module)
    return (not missing), missing


def resolve_device(cfg):
    """返回 (device_str, use_cuda)。"""
    requested = str((cfg.get("asr") or {}).get("device", "gpu") or "gpu").lower()
    if requested == "cpu":
        return "cpu", False
    import torch
    if torch.cuda.is_available():
        return "cuda:0", True
    return "cpu", False


def torch_dtype(cfg, use_cuda):
    import torch
    if not use_cuda:
        return torch.float32
    value = str((cfg.get("asr") or {}).get("dtype", "bfloat16") or "bfloat16")
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    return torch.bfloat16


class SpeechModels:
    """按需加载并复用语音模型；一次识别阶段内只加载一次。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device, self.use_cuda = resolve_device(cfg)
        self.dtype = torch_dtype(cfg, self.use_cuda)
        self._vad = None
        self._speaker = None
        self._verify = None
        self._asr = None

    def model_path(self, key, default):
        return resolve_model_dir(self.cfg, key, default)

    def vad(self):
        if self._vad is None:
            from funasr import AutoModel
            asr = self.cfg.get("asr") or {}
            self._vad = AutoModel(
                model=self.model_path("vad_model", "fsmn-vad"),
                device=self.device,
                max_single_segment_time=int(
                    asr.get("vad_max_segment_sec", 60)) * 1000,
                disable_update=True, disable_pbar=True, disable_log=True,
            )
        return self._vad

    def speaker_embedder(self):
        if self._speaker is None:
            from funasr import AutoModel
            self._speaker = AutoModel(
                model=self.model_path("speaker_model", "campplus"),
                device=self.device,
                disable_update=True, disable_pbar=True, disable_log=True,
            )
        return self._speaker

    def verify_embedder(self):
        """ERes2NetV2：本地目录无 config.yaml，显式给出类别与前端配置。"""
        if self._verify is None:
            from funasr import AutoModel
            self._verify = AutoModel(
                model="ERes2NetV2",
                model_conf={},
                model_path=self.model_path("verify_model", "eres2netv2"),
                frontend="WavFrontend",
                frontend_conf={"fs": 16000},
                device=self.device,
                disable_update=True, disable_pbar=True, disable_log=True,
            )
        return self._verify

    def asr(self):
        if self._asr is None:
            from qwen_asr import Qwen3ASRModel
            asr_cfg = self.cfg.get("asr") or {}
            self._asr = Qwen3ASRModel.from_pretrained(
                self.model_path("asr_model", "Qwen3-ASR-1.7B"),
                dtype=self.dtype,
                device_map=self.device,
                max_inference_batch_size=int(
                    asr_cfg.get("max_inference_batch_size", 4)),
                max_new_tokens=int(asr_cfg.get("max_new_tokens", 2048)),
                forced_aligner=self.model_path(
                    "forced_aligner", "Qwen3-ForcedAligner-0.6B"),
                forced_aligner_kwargs=dict(dtype=self.dtype,
                                           device_map=self.device),
            )
        return self._asr

    def unload(self):
        self._vad = None
        self._speaker = None
        self._verify = None
        self._asr = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


_models = None
_lock = threading.Lock()


def get_models(cfg):
    global _models
    with _lock:
        if _models is None:
            _models = SpeechModels(cfg)
        return _models


def shutdown():
    global _models
    with _lock:
        if _models is not None:
            _models.unload()
            _models = None
