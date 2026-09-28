"""裁剪框计算、姿态层级标注与质量评分。

- 裁剪矩形 = union(人体检测框, 可见关键点外接框) 外扩 margin，越界截断；
- 层级 = 身体链上可见的最低关节（任一侧可见即算）；
- 评分 = 簇内归一化清晰度 × 检测置信度 × 人脸因子。
"""

import cv2

_ANKLES = (15, 16)
_KNEES = (13, 14)
_HIPS = (11, 12)
_SHOULDERS = (5, 6)
_FACE = (0, 1, 2, 3, 4)

LEVELS = ("full_body", "upper_body", "torso", "chest", "head", "unknown")


def classify_level(kps_conf, kp_conf):
    if kps_conf is None or len(kps_conf) == 0:
        return "unknown"
    visible = set(i for i, value in enumerate(kps_conf) if value >= kp_conf)
    if visible & set(_ANKLES):
        return "full_body"
    if visible & set(_KNEES):
        return "upper_body"
    if visible & set(_HIPS):
        return "torso"
    if visible & set(_SHOULDERS):
        return "chest"
    if visible & set(_FACE):
        return "head"
    return "unknown"


def crop_rect(bbox, kps, kps_conf, kp_conf, margin, width, height):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if kps is not None and kps_conf is not None and len(kps_conf):
        visible = kps_conf >= kp_conf
        if visible.any():
            xs = kps[visible, 0]
            ys = kps[visible, 1]
            x1 = min(x1, float(xs.min()))
            y1 = min(y1, float(ys.min()))
            x2 = max(x2, float(xs.max()))
            y2 = max(y2, float(ys.max()))
    margin_x = (x2 - x1) * margin
    margin_y = (y2 - y1) * margin
    cx1 = max(0, int(round(x1 - margin_x)))
    cy1 = max(0, int(round(y1 - margin_y)))
    cx2 = min(width, int(round(x2 + margin_x)))
    cy2 = min(height, int(round(y2 + margin_y)))
    if cx2 - cx1 < 1 or cy2 - cy1 < 1:
        return None
    return [cx1, cy1, cx2, cy2]


def center_inside(rect, point):
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def rect_area(rect):
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def sharpness_of(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def score_of(record, max_sharpness):
    if max_sharpness > 0:
        normalized = max(0.0, record["sharpness"]) / max_sharpness
    else:
        normalized = 0.0
    face_factor = record.get("face_score") or 1.0
    return normalized * float(record["conf"]) * float(face_factor)
