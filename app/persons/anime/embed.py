"""动漫身份嵌入：CLIP 图像嵌入（路线 B）。

用项目已装的 `transformers` 加载离线 CLIP（默认 `openai/clip-vit-large-patch14`），
对人物/人脸裁剪图提投影后的图像特征并 L2 归一化。CLIP 训练集含大量动漫插画，
对画风鲁棒、2D/3D 通吃；局限是"语义相似"而非"身份相同"，因此聚类阈值需独立校准
（见 `persons.anime.cluster_threshold`）。

不引入 `open_clip`：`transformers` 为既有依赖，避免新增重量级包。
"""

import os

import cv2
import numpy as np


class ClipEmbedder(object):
    def __init__(self, clip_dir, cfg):
        self.clip_dir = clip_dir
        pc = cfg.get("persons") or {}
        ac = pc.get("anime") or {}
        self.device = "cpu"
        try:
            import torch
            requested = str(pc.get("device") or "gpu").lower()
            self.device = "cuda:0" if (requested != "cpu"
                                       and torch.cuda.is_available()) else "cpu"
        except Exception:
            pass
        self._processor = None
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import CLIPImageProcessor, CLIPModel
        self._processor = CLIPImageProcessor.from_pretrained(self.clip_dir)
        model = CLIPModel.from_pretrained(self.clip_dir)
        model = model.to(self.device).eval()
        self._model = model

    def embed(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            return None
        height, width = image_bgr.shape[:2]
        if height < 2 or width < 2:
            return None
        self._ensure()
        import torch
        from PIL import Image
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._processor(images=Image.fromarray(rgb),
                                 return_tensors="pt")
        tensor = {key: value.to(self.device) for key, value in tensor.items()}
        with torch.no_grad():
            features = self._model.get_image_features(**tensor)
        vector = features[0].to(torch.float32).cpu().numpy().reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-9:
            return None
        return (vector / norm).astype(np.float32)

    def close(self):
        self._model = None
        self._processor = None


def clip_ready(clip_dir):
    if not os.path.isfile(os.path.join(clip_dir, "config.json")):
        return False
    for name in ("model.safetensors", "pytorch_model.bin"):
        if os.path.isfile(os.path.join(clip_dir, name)):
            return True
    return False
