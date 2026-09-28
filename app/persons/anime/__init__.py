"""动漫人物提取后端（2D / 3D 通用）。

参考 `new_requirements.md` 的修正后方案：

- 人物框：`deepghs/anime_person_detection` 的 YOLO ONNX（`onnxruntime` 直驱）+ 人脸锚点扩展（补救漏检）；
- 人脸：`deepghs/anime_face_detection` 的 YOLO ONNX（`onnxruntime` 直驱）；
- 身份嵌入：CLIP 人物裁剪图嵌入（`transformers`，对画风鲁棒，背影/侧脸帧也能聚类）；
- 跟踪：`associate_track_id` 中心/尺寸关联（硬切处断开）。

聚类 / 去重沿用真人侧的框架，仅使用 `persons.anime` 的独立阈值。
"""

from .detect import AnimePersonTracker, expand_face_to_person
from .face import AnimeFaceEngine

__all__ = ["AnimeFaceEngine", "AnimePersonTracker", "expand_face_to_person"]
