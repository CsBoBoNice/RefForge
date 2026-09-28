"""动漫人物框与跟踪：动漫人物检测（deepghs YOLO ONNX）+ 帧间关联。

1. 用**动漫专用**人物检测器（`person_detect_v1.1_m`，F1≈0.87，onnxruntime 直驱）
   取全身/半身框，替代在 2D 赛璐璐上大量漏检的 COCO YOLO；
2. 对整帧动漫人脸框按头身比扩展出近似人物框（路线 A），仅在未被任何人物框
   覆盖时追加，补救遮挡/远景漏检；
3. track_id 置空，由 `process_video` 的 `associate_track_id` 按中心距离 + 尺寸比
   做帧间关联（与真人路径在跟踪器缺 id 时的兜底逻辑一致），硬切处再断开。

不依赖 `ultralytics` 跟踪：其 ONNX 路径会误判 `onnx` 依赖并尝试联网安装。
"""

from ..associate import owner_index
from .onnx_detector import OnnxYoloDetector

MIN_PERSON_SIDE = 8


def expand_face_to_person(face_box, head_body_ratio, width, height):
    """由人脸框按头身比推算人物框；越界截断，退化时返回 None。"""
    x1, y1, x2, y2 = [float(v) for v in face_box]
    face_w = max(1.0, x2 - x1)
    face_h = max(1.0, y2 - y1)
    person_h = face_h / 0.8 * max(1.0, float(head_body_ratio))
    center_x = (x1 + x2) / 2.0
    person_w = max(face_w * 1.6, person_h * 0.35)
    top = y1 - face_h * 0.25
    left = center_x - person_w / 2.0
    right = center_x + person_w / 2.0
    bottom = top + person_h
    px1 = int(max(0, min(width, round(left))))
    py1 = int(max(0, min(height, round(top))))
    px2 = int(max(0, min(width, round(right))))
    py2 = int(max(0, min(height, round(bottom))))
    if (px2 - px1) < MIN_PERSON_SIDE or (py2 - py1) < MIN_PERSON_SIDE:
        return None
    return [px1, py1, px2, py2]


class AnimePersonTracker(object):
    def __init__(self, models, cfg, tracker_file=None):
        pc = models.persons_params(cfg)
        self.detector = models.person_detector()
        self.conf = float(pc.get("det_conf", 0.30))
        self.head_body_ratio = float(pc.get("head_body_ratio", 7.0))

    def update(self, frame_bgr, whole_faces=None):
        detections = []
        for box, score in self.detector.detect(frame_bgr, self.conf):
            detections.append({
                "bbox": box,
                "conf": float(score),
                "track_id": None,
                "kps": None,
                "kps_conf": None,
            })
        if not whole_faces:
            return detections
        height, width = frame_bgr.shape[:2]
        boxes = [det["bbox"] for det in detections]
        for face_box, score, keypoints in whole_faces:
            center = ((face_box[0] + face_box[2]) / 2.0,
                      (face_box[1] + face_box[3]) / 2.0)
            if owner_index(boxes, center) is not None:
                continue
            person = expand_face_to_person(
                face_box, self.head_body_ratio, width, height)
            if person is None:
                continue
            boxes.append(person)
            detections.append({
                "bbox": person,
                "conf": 0.5,
                "track_id": None,
                "kps": None,
                "kps_conf": None,
                "level": "head",
            })
        return detections
