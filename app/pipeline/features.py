"""第二层：特征提取与匹配。

第一阶段落地方案：SIFT（不可用时回落 ORB）+ BFMatcher + Lowe ratio test。
"""

import threading
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Features:
    keypoints: list
    descriptors: np.ndarray


_DETECTOR_LOCAL = threading.local()


def _make_detector(max_features):
    try:
        return cv2.SIFT_create(nfeatures=int(max_features)), cv2.NORM_L2
    except Exception:
        return cv2.ORB_create(nfeatures=int(max_features)), cv2.NORM_HAMMING


def _get_detector(max_features):
    key = int(max_features)
    cache = getattr(_DETECTOR_LOCAL, "cache", None)
    if cache is None:
        cache = {}
        _DETECTOR_LOCAL.cache = cache
    if key not in cache:
        cache[key] = _make_detector(key)
    return cache[key]


def extract_features(gray, cfg):
    detector, _ = _get_detector(cfg["matching"]["max_features"])
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None:
        descriptors = np.empty((0, 0), dtype=np.float32)
    return Features(keypoints=keypoints or [], descriptors=descriptors)


def match_pair(feat_i, feat_j, cfg):
    """返回 (pts_i, pts_j, match_count)，失败返回 None。"""
    matching = cfg["matching"]
    des_i = feat_i.descriptors
    des_j = feat_j.descriptors
    if des_i is None or des_j is None:
        return None
    if len(des_i) < 2 or len(des_j) < 2:
        return None

    _, norm_type = _get_detector(matching["max_features"])
    matcher = cv2.BFMatcher(norm_type)
    try:
        knn = matcher.knnMatch(des_i, des_j, k=2)
    except cv2.error:
        return None

    ratio = float(matching["ratio_test"])
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            good.append(best)

    if not good:
        return None

    kp_i = feat_i.keypoints
    kp_j = feat_j.keypoints
    pts_i = np.float32([kp_i[m.queryIdx].pt for m in good])
    pts_j = np.float32([kp_j[m.trainIdx].pt for m in good])
    return pts_i, pts_j, len(good)
