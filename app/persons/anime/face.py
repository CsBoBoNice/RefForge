"""动漫人脸检测（deepghs/anime_face_detection，YOLO ONNX）与 CLIP 嵌入。

对外接口与真人侧的 `FaceEngine` 保持一致：

- `detect(image, min_face_px, threshold)` -> `[(box, score, keypoints), ...]`，
  动漫检测器不提供关键点，`keypoints` 恒为 `None`；
- `embed(image, keypoints, face_box)` -> L2 归一化 CLIP 图像特征；有 `face_box`
  时只嵌入人脸区域（身份信号），否则嵌入整张人物裁剪图（背影/侧脸兜底）。

用动漫专用检测器（F1≈0.95）替代早期 OpenCV LBP 级联，避免在插画/道具上产生
大量假阳性，并显著提升近正面/侧脸召回；仍只检较正面的脸，背影帧靠人物框兜底。
"""

import os

from .embed import ClipEmbedder
from .onnx_detector import OnnxYoloDetector


class AnimeFaceEngine(object):
    def __init__(self, face_model_path, cfg, device="gpu"):
        if not os.path.isfile(face_model_path):
            raise IOError("anime face model not found: %s" % face_model_path)
        ac = (cfg.get("persons") or {}).get("anime") or {}
        self.input_size = int(ac.get("input_size", 640))
        self.iou = float(ac.get("nms_iou", 0.5))
        self.conf = float(ac.get("face_embed_score", 0.30))
        self.detector = OnnxYoloDetector(
            face_model_path, cfg, input_size=self.input_size, iou=self.iou)
        self.clip = ClipEmbedder(_clip_dir(cfg), cfg)

    def detect(self, image, min_face_px=0, threshold=None):
        conf = self.conf if threshold is None else float(threshold)
        results = []
        for box, score in self.detector.detect(image, conf):
            if min_face_px and min(box[2] - box[0], box[3] - box[1]) < min_face_px:
                continue
            results.append((box, score, None))
        return results

    def embed(self, image, keypoints, face_box=None):
        if image is None or image.size == 0:
            return None
        if face_box is not None:
            height, width = image.shape[:2]
            x1 = int(max(0, min(width - 1, face_box[0])))
            y1 = int(max(0, min(height - 1, face_box[1])))
            x2 = int(max(0, min(width, face_box[2])))
            y2 = int(max(0, min(height, face_box[3])))
            if x2 - x1 < 2 or y2 - y1 < 2:
                return None
            image = image[y1:y2, x1:x2]
        return self.clip.embed(image)

    def close(self):
        if self.detector is not None:
            self.detector.close()
            self.detector = None
        if self.clip is not None:
            self.clip.close()
            self.clip = None


def _clip_dir(cfg):
    from ..models import anime_clip_dir
    return anime_clip_dir(cfg)
