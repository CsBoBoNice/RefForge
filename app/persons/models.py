"""人物提取模型容器（惰性加载、用后释放）。

两种领域模型：

- `real`（真人）：YOLO11-pose + FaceEngine（buffalo_l 的 SCRFD / ArcFace ONNX），
  模型路径在 `models/person/` 下（`detect/<pose_model>` 与 `face/<face_pack>/`）；
- `anime`（动漫，含 2D / 3D）：YOLO11-pose（低阈值取人体框）+ 人脸锚点扩展、
  OpenCV 动漫人脸级联 + CLIP 图像嵌入，模型路径在 `models/person/anime/` 下。

阶段内只加载一次、阶段结束调用 `close()` 释放显存。
"""

import os

from io_utils import package_root


def persons_cfg(cfg):
    return cfg.get("persons") or {}


def domain_of(cfg):
    value = str(persons_cfg(cfg).get("domain") or "anime").strip().lower()
    return "anime" if value == "anime" else "real"


def anime_cfg(cfg):
    return persons_cfg(cfg).get("anime") or {}


def effective_persons_cfg(cfg):
    """按领域返回生效的参数：动漫模式下用 `persons.anime` 覆盖顶层同名键。"""
    pc = dict(persons_cfg(cfg))
    if domain_of(cfg) != "anime":
        return pc
    merged = {key: value for key, value in pc.items() if key != "anime"}
    merged.update(pc.get("anime") or {})
    return merged


def models_root(cfg):
    value = str(persons_cfg(cfg).get("models") or "models/person").strip()
    if os.path.isabs(value):
        return os.path.normpath(value)
    return os.path.normpath(os.path.join(package_root(), value))


def pose_model_path(cfg):
    root = models_root(cfg)
    name = str(persons_cfg(cfg).get("pose_model") or "yolo11x-pose.pt")
    return os.path.normpath(os.path.join(root, "detect", name))


def face_dir(cfg):
    root = models_root(cfg)
    pack = str(persons_cfg(cfg).get("face_pack") or "buffalo_l")
    return os.path.normpath(os.path.join(root, "face", pack))


def anime_root(cfg):
    value = str(anime_cfg(cfg).get("models") or "models/person/anime").strip()
    if os.path.isabs(value):
        return os.path.normpath(value)
    return os.path.normpath(os.path.join(package_root(), value))


def anime_person_model_path(cfg):
    root = anime_root(cfg)
    name = str(anime_cfg(cfg).get("person_onnx")
               or "person/person_detect_v1.1_m/model.onnx")
    return os.path.normpath(os.path.join(root, name))


def anime_face_model_path(cfg):
    root = anime_root(cfg)
    name = str(anime_cfg(cfg).get("face_onnx")
               or "face/face_detect_v1.4_s/model.onnx")
    return os.path.normpath(os.path.join(root, name))


def anime_clip_dir(cfg):
    root = anime_root(cfg)
    name = str(anime_cfg(cfg).get("clip_model")
               or "clip/clip-vit-large-patch14")
    return os.path.normpath(os.path.join(root, name))


def resolve_device(cfg):
    requested = str(persons_cfg(cfg).get("device") or "gpu").lower()
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        return "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_yolo(cfg, device):
    from ultralytics import YOLO
    model = YOLO(pose_model_path(cfg))
    if device == "gpu":
        model.to("cuda:0")
    return model


class PersonModels(object):
    """真人领域：YOLO pose 检测器 + 人脸引擎，共享同一次加载。"""

    domain = "real"
    embed_mode = "face"

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = resolve_device(cfg)
        self._yolo = None
        self._face = None

    def persons_params(self, cfg):
        return effective_persons_cfg(cfg)

    def yolo(self):
        if self._yolo is None:
            self._yolo = _load_yolo(self.cfg, self.device)
        return self._yolo

    def face(self):
        if self._face is None:
            from .face import FaceEngine
            self._face = FaceEngine(face_dir(self.cfg), self.cfg)
        return self._face

    def make_tracker(self, cfg):
        from .detect import PersonTracker
        return PersonTracker(self, cfg, tracker_file="botsort.yaml")

    def close(self):
        self._yolo = None
        if self._face is not None:
            self._face.close()
            self._face = None
        _empty_cache()


class AnimeModels(object):
    """动漫领域：动漫人物/人脸 ONNX 检测器 + CLIP 嵌入，共享同一次加载。"""

    domain = "anime"
    embed_mode = "person"

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = resolve_device(cfg)
        self._person = None
        self._face = None

    def persons_params(self, cfg):
        return effective_persons_cfg(cfg)

    def person_detector(self):
        if self._person is None:
            from .anime.onnx_detector import OnnxYoloDetector
            ac = anime_cfg(self.cfg)
            self._person = OnnxYoloDetector(
                anime_person_model_path(self.cfg), self.cfg,
                input_size=int(ac.get("input_size", 640)),
                iou=float(ac.get("nms_iou", 0.5)))
        return self._person

    def face(self):
        if self._face is None:
            from .anime.face import AnimeFaceEngine
            self._face = AnimeFaceEngine(anime_face_model_path(self.cfg),
                                         self.cfg, self.device)
        return self._face

    def make_tracker(self, cfg):
        from .anime.detect import AnimePersonTracker
        return AnimePersonTracker(self, cfg)

    def close(self):
        if self._person is not None:
            self._person.close()
            self._person = None
        if self._face is not None:
            self._face.close()
            self._face = None
        _empty_cache()


def _empty_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def build_models(cfg):
    if domain_of(cfg) == "anime":
        return AnimeModels(cfg)
    return PersonModels(cfg)


def _clip_ready(clip_dir):
    if not os.path.isfile(os.path.join(clip_dir, "config.json")):
        return False
    for name in ("model.safetensors", "pytorch_model.bin"):
        if os.path.isfile(os.path.join(clip_dir, name)):
            return True
    return False


def _real_available(cfg):
    missing = []
    try:
        import ultralytics  # noqa: F401
    except Exception:
        missing.append("ultralytics")
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        missing.append("onnxruntime")
    if not os.path.isfile(pose_model_path(cfg)):
        missing.append("pose_model(%s)" % os.path.basename(pose_model_path(cfg)))
    face = face_dir(cfg)
    for name in ("det_10g.onnx", "w600k_r50.onnx"):
        if not os.path.isfile(os.path.join(face, name)):
            missing.append("face/%s" % name)
    return (not missing), missing


def _anime_available(cfg):
    missing = []
    for module in ("torch", "transformers", "onnxruntime"):
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    person = anime_person_model_path(cfg)
    if not os.path.isfile(person):
        missing.append("anime_person(%s)" % os.path.basename(person))
    face = anime_face_model_path(cfg)
    if not os.path.isfile(face):
        missing.append("anime_face(%s)" % os.path.basename(face))
    clip_dir = anime_clip_dir(cfg)
    if not _clip_ready(clip_dir):
        missing.append("anime_clip(%s)" % os.path.basename(clip_dir))
    return (not missing), missing


def available(cfg):
    if domain_of(cfg) == "anime":
        return _anime_available(cfg)
    return _real_available(cfg)
