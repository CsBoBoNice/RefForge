"""多帧运动估计与矩阵累积（尺度轨迹与路径统计）。

对每对相邻采样帧估计相似变换（平移/旋转/缩放），逐步累积得到
累积尺度轨迹 `cum_scale` 与路径统计量，供 `--debug` 诊断输出。
不再做运镜分类 / 主体 / 光流等识别。

注意：故事图选帧已改为「均匀时间间隔 + 就近取清晰帧」，不再依赖本模块
的 `cum_scale`；本模块保留以维持 `--debug` 诊断与后续扩展。
"""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


EPS = 1e-8


@dataclass
class PairEvidence:
    index_i: int
    index_j: int
    match_count: int
    inlier_count: int
    inlier_ratio: float
    scale_step: float
    rotation_step_deg: float
    translation_step: tuple
    perspective_error: float
    flow_magnitude: float
    sharpness: float
    center_inlier_ratio: float
    success: bool
    h_inlier_ratio: float = 0.0
    parallax_cv: float = 0.0
    affine: object = None
    homography: object = None


@dataclass
class AggregateEvidence:
    sample_indices: list
    scale_total: float = 1.0
    zoom_path: float = 0.0
    rotation_deg_total: float = 0.0
    rotation_path_deg: float = 0.0
    translation_total: tuple = (0.0, 0.0)
    translation_norm: float = 0.0
    translation_path: float = 0.0
    translation_path_ratio: float = 0.0
    direction_change_deg: float = 0.0
    perspective_avg: float = 0.0
    flow_magnitude_avg: float = 0.0
    avg_match_count: float = 0.0
    avg_inlier_count: float = 0.0
    avg_inlier_ratio: float = 0.0
    match_fail_rate: float = 1.0
    low_inlier_rate: float = 1.0
    sharpness_median: float = 0.0
    low_sharpness_ratio: float = 0.0
    center_inlier_ratio_mean: float = 0.0
    center_inlier_ratio_std: float = 0.0
    score_zoom: float = 0.0
    score_rot: float = 0.0
    score_trans: float = 0.0
    max_translation_step_norm: float = 0.0
    max_flow_magnitude: float = 0.0
    diag: float = 1.0
    proc_long_side: float = 0.0
    h_inlier_ratio_avg: float = 0.0
    h_inlier_ratio_min: float = 0.0
    parallax_cv_avg: float = 0.0
    cum_scale: list = field(default_factory=lambda: [1.0])
    cum_rotation_deg: list = field(default_factory=lambda: [0.0])
    cum_translation: list = field(default_factory=lambda: [(0.0, 0.0)])
    cum_homography: list = field(default_factory=list)


def _to_homography(matrix):
    if matrix is None:
        return None
    if matrix.shape == (3, 3):
        return matrix.astype(np.float64)
    hom = np.eye(3, dtype=np.float64)
    hom[:2, :] = matrix.astype(np.float64)
    return hom


def _scale_of(matrix):
    return float(math.hypot(matrix[0, 0], matrix[1, 0]))


def _rotation_deg_of(matrix):
    return float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))


def _rms_residual(matrix, src, dst):
    if src.shape[0] == 0:
        return 0.0
    hom = _to_homography(matrix)
    src_h = np.hstack([src, np.ones((src.shape[0], 1), dtype=np.float64)])
    proj = (hom @ src_h.T).T
    proj = proj[:, :2] / np.clip(proj[:, 2:3], EPS, None)
    diff = proj - dst
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _center_mask(points, shape):
    height, width = shape[:2]
    x0, x1 = 0.25 * width, 0.75 * width
    y0, y1 = 0.25 * height, 0.75 * height
    return ((points[:, 0] >= x0) & (points[:, 0] <= x1) &
            (points[:, 1] >= y0) & (points[:, 1] <= y1))


def center_displacement(matrix, shape):
    """画面中心点在该变换下的内容位移。

    相似变换的平移项包含"绕图像中心缩放/旋转"引入的偏置，
    直接使用会高估平移量；按 REQUIREMENTS 2.2 的定义，
    平移应取画面内容的位移，故在画面中心处测量。
    """
    height, width = shape[:2]
    center = np.array([width / 2.0, height / 2.0, 1.0])
    hom = _to_homography(matrix)
    projected = hom @ center
    denom = projected[2]
    if abs(denom) < EPS:
        denom = EPS
    projected = projected[:2] / denom
    return (float(projected[0] - center[0]), float(projected[1] - center[1]))


def failed_pair(index_i, index_j, sharpness):
    return PairEvidence(
        index_i=index_i, index_j=index_j,
        match_count=0, inlier_count=0, inlier_ratio=0.0,
        scale_step=1.0, rotation_step_deg=0.0, translation_step=(0.0, 0.0),
        perspective_error=0.0, flow_magnitude=0.0,
        sharpness=float(sharpness), center_inlier_ratio=0.0, success=False)


def estimate_pair(pts_i, pts_j, image_shape, sharpness, cfg):
    matching = cfg["matching"]
    threshold = float(matching["ransac_reproj_threshold"])
    min_match = int(matching["min_match_count"])
    index_i, index_j = 0, 1

    count = 0 if pts_i is None else int(len(pts_i))
    if pts_i is None or pts_j is None or count < min_match:
        return failed_pair(index_i, index_j, sharpness)

    pts_i = np.asarray(pts_i, dtype=np.float64)
    pts_j = np.asarray(pts_j, dtype=np.float64)

    homography, mask_h = cv2.findHomography(
        pts_i, pts_j, cv2.RANSAC, threshold)
    affine, mask_a = cv2.estimateAffinePartial2D(
        pts_i, pts_j, method=cv2.RANSAC, ransacReprojThreshold=threshold)

    if affine is None or mask_a is None:
        return failed_pair(index_i, index_j, sharpness)

    inlier_mask = mask_a.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / float(count) if count else 0.0

    scale_step = _scale_of(affine)
    rotation_step_deg = _rotation_deg_of(affine)

    if inlier_count < int(matching["min_inlier_count"]):
        return failed_pair(index_i, index_j, sharpness)
    if not math.isfinite(scale_step) or scale_step < 0.5 or scale_step > 2.0:
        return failed_pair(index_i, index_j, sharpness)

    translation_step = center_displacement(affine, image_shape)

    diag = float(np.hypot(image_shape[1], image_shape[0]))
    err_affine = _rms_residual(affine, pts_i[inlier_mask], pts_j[inlier_mask])
    if homography is not None:
        err_homo = _rms_residual(homography, pts_i[inlier_mask], pts_j[inlier_mask])
        perspective_error = float(np.clip(
            (err_affine - err_homo) / (diag + EPS), 0.0, 1.0))
    else:
        perspective_error = 0.0

    flow = np.linalg.norm(pts_j - pts_i, axis=1)
    flow_magnitude = float(np.median(flow) / (diag + EPS)) if len(flow) else 0.0

    if mask_h is not None and len(mask_h):
        h_inlier_ratio = float(np.asarray(mask_h).ravel().mean())
    else:
        h_inlier_ratio = 0.0
    if len(flow) and float(np.mean(flow)) > EPS:
        parallax_cv = float(np.std(flow) / (np.mean(flow) + EPS))
    else:
        parallax_cv = 0.0

    center = _center_mask(pts_i, image_shape)
    center_count = int(center.sum())
    if center_count > 0:
        center_inlier_ratio = float(
            np.logical_and(center, inlier_mask).sum()) / center_count
    else:
        center_inlier_ratio = inlier_ratio

    return PairEvidence(
        index_i=index_i, index_j=index_j,
        match_count=count, inlier_count=inlier_count,
        inlier_ratio=float(inlier_ratio),
        scale_step=scale_step, rotation_step_deg=rotation_step_deg,
        translation_step=translation_step,
        perspective_error=perspective_error, flow_magnitude=flow_magnitude,
        sharpness=float(sharpness), center_inlier_ratio=center_inlier_ratio,
        success=True, h_inlier_ratio=h_inlier_ratio, parallax_cv=parallax_cv,
        affine=affine.astype(np.float64),
        homography=None if homography is None else homography.astype(np.float64))


def _angle_between(v1, v2):
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < EPS or n2 < EPS:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))


def accumulate_base(pairs, sample_indices, image_shape, cfg,
                    sharpness_median, low_sharpness_ratio, duration_sec):
    """矩阵累积与路径统计；产出 `cum_scale` 等诊断量（故事图已不再依赖）。"""
    agg = AggregateEvidence(sample_indices=list(sample_indices))
    agg.sharpness_median = float(sharpness_median)
    agg.low_sharpness_ratio = float(low_sharpness_ratio)
    agg.diag = float(np.hypot(image_shape[1], image_shape[0]))
    agg.proc_long_side = float(max(image_shape[0], image_shape[1]))

    total_pairs = len(pairs)
    if total_pairs == 0:
        return agg

    successful = [p for p in pairs if p.success]
    failed = total_pairs - len(successful)
    low_inlier_threshold = float(cfg["matching"]["low_inlier_ratio"])

    agg.avg_match_count = float(np.mean([p.match_count for p in pairs]))
    if successful:
        agg.avg_inlier_count = float(np.mean([p.inlier_count for p in successful]))
        agg.avg_inlier_ratio = float(np.mean([p.inlier_ratio for p in successful]))
        agg.low_inlier_rate = float(np.mean(
            [p.inlier_ratio < low_inlier_threshold for p in successful]))
        agg.perspective_avg = float(np.mean(
            [p.perspective_error for p in successful]))
        agg.flow_magnitude_avg = float(np.mean(
            [p.flow_magnitude for p in successful]))
        agg.h_inlier_ratio_avg = float(np.mean(
            [p.h_inlier_ratio for p in successful]))
        agg.h_inlier_ratio_min = float(np.min(
            [p.h_inlier_ratio for p in successful]))
        agg.parallax_cv_avg = float(np.mean(
            [p.parallax_cv for p in successful]))
    agg.match_fail_rate = float(failed) / float(total_pairs)

    diag = float(np.hypot(image_shape[1], image_shape[0]))
    agg.diag = diag
    agg.proc_long_side = float(max(image_shape[0], image_shape[1]))

    # 鲁棒累积：剔除低内点率对与尺度离群对，避免逐对噪声累积成漂移
    usable = [p for p in successful if p.inlier_ratio >= low_inlier_threshold]
    if len(usable) < 2:
        usable = successful
    if len(usable) >= 3:
        median_scale = float(np.median([max(p.scale_step, EPS) for p in usable]))
        kept = [p for p in usable
                if abs(math.log(max(p.scale_step, EPS) / median_scale)) < 0.25]
        if len(kept) >= 2:
            usable = kept
    usable_ids = set(id(p) for p in usable)

    agg.max_translation_step_norm = 0.0
    agg.max_flow_magnitude = 0.0
    for p in successful:
        agg.max_translation_step_norm = max(
            agg.max_translation_step_norm,
            float(np.hypot(*p.translation_step)) / (diag + EPS))
        agg.max_flow_magnitude = max(agg.max_flow_magnitude, p.flow_magnitude)

    transform = np.eye(3, dtype=np.float64)
    homography_acc = np.eye(3, dtype=np.float64)
    zoom_path = 0.0
    rotation_path = 0.0
    translation_path = 0.0
    direction_change = 0.0
    prev_dir = None
    min_step = 0.02 * diag

    agg.cum_homography = [homography_acc.copy()]
    for pair in pairs:
        if pair.success and id(pair) in usable_ids:
            affine_h = _to_homography(pair.affine)
            transform = affine_h @ transform
            if pair.homography is not None:
                homography_acc = _to_homography(pair.homography) @ homography_acc
            zoom_path += abs(math.log(max(pair.scale_step, EPS)))
            rotation_path += abs(pair.rotation_step_deg)
            step_dir = np.asarray(pair.translation_step, dtype=np.float64)
            step_norm = float(np.linalg.norm(step_dir))
            translation_path += step_norm
            if step_norm > min_step:
                if prev_dir is not None:
                    direction_change += _angle_between(prev_dir, step_dir)
                prev_dir = step_dir
        agg.cum_scale.append(_scale_of(transform))
        agg.cum_rotation_deg.append(_rotation_deg_of(transform))
        agg.cum_translation.append(center_displacement(transform, image_shape))
        agg.cum_homography.append(homography_acc.copy())

    agg.scale_total = _scale_of(transform)
    agg.rotation_deg_total = _rotation_deg_of(transform)
    agg.translation_total = center_displacement(transform, image_shape)
    agg.zoom_path = zoom_path
    agg.rotation_path_deg = rotation_path
    agg.translation_path = translation_path
    agg.translation_norm = float(
        np.hypot(*agg.translation_total) / (diag + EPS))
    agg.translation_path_ratio = float(
        translation_path / (np.hypot(*agg.translation_total) + EPS))
    agg.direction_change_deg = direction_change

    scale_delta = abs(math.log(max(agg.scale_total, EPS)))
    rot_deg = abs(agg.rotation_deg_total)
    agg.score_zoom = scale_delta / 0.10
    agg.score_rot = rot_deg / 3.0
    agg.score_trans = agg.translation_norm / 0.03

    _center_inlier_stats(agg, successful)
    return agg


def _center_inlier_stats(agg, successful):
    ratios = [p.center_inlier_ratio for p in successful]
    if len(ratios) >= 2:
        agg.center_inlier_ratio_mean = float(np.mean(ratios))
        agg.center_inlier_ratio_std = float(np.std(ratios))
    elif ratios:
        agg.center_inlier_ratio_mean = float(ratios[0])
        agg.center_inlier_ratio_std = 0.0
