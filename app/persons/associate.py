"""检测归属辅助：track 断链兜底关联与人脸-人体框归属。

- `associate_track_id`：BoT-SORT 未给 id 时，按中心距离就近继承最近 track；
- `owner_index`：人脸中心落在多个重叠人体框时，返回包含中心的面积最小框。
"""

import math

from .crop import center_inside, rect_area


def _box_size(bbox):
    return max(1.0, (bbox[2] - bbox[0])), max(1.0, (bbox[3] - bbox[1]))


def associate_track_id(det, order, frame_index, recent_tracks, diagonal,
                       max_frames=8, max_distance=0.25,
                       min_size_ratio=0.5, max_size_ratio=2.0):
    track_id = det.get("track_id")
    if track_id is not None:
        return int(track_id)
    x1, y1, x2, y2 = det["bbox"]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    det_w, det_h = _box_size(det["bbox"])
    best = None
    best_distance = None
    for recent_id, (seen_frame, bbox) in recent_tracks.items():
        if frame_index - seen_frame > max_frames:
            continue
        box_w, box_h = _box_size(bbox)
        ratio = (det_h / box_h) if det_h >= box_h else (box_h / det_h)
        if ratio < min_size_ratio or ratio > max_size_ratio:
            continue
        box_x = (bbox[0] + bbox[2]) / 2.0
        box_y = (bbox[1] + bbox[3]) / 2.0
        distance = math.hypot(center_x - box_x, center_y - box_y) / diagonal
        if distance < max_distance and (best_distance is None
                                        or distance < best_distance):
            best = recent_id
            best_distance = distance
    if best is not None:
        return int(best)
    return -(frame_index + 1) * 10000 - order


def owner_index(det_boxes, point):
    """返回包含 point 的面积最小的框下标；都不包含时返回 None。"""
    owner = None
    owner_area = None
    for index, box in enumerate(det_boxes):
        if not center_inside(box, point):
            continue
        area = rect_area(box)
        if owner_area is None or area < owner_area:
            owner_area = area
            owner = index
    return owner
