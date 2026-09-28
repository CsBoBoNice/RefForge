"""第一层：帧质量评估。

quality_score = sharpness_score * exposure_score * contrast_score
任一维度过差都会显著拉低总分。
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FrameQuality:
    sharpness: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    quality_score: np.ndarray
    is_black: np.ndarray
    is_white: np.ndarray
    is_blur: np.ndarray
    is_low_info: np.ndarray
    bad: np.ndarray
    sharpness_median: float
    low_sharpness_ratio: float


def assess_frames(gray_frames, cfg):
    fq = cfg["frame_quality"]
    black_mean = float(fq["black_mean"])
    black_std = float(fq["black_std"])
    white_mean = float(fq["white_mean"])
    white_std = float(fq["white_std"])
    low_info_std = float(fq["low_info_std"])
    blur_ratio = float(fq["blur_ratio_of_median"])

    n = len(gray_frames)
    sharpness = np.zeros(n, dtype=np.float64)
    mean = np.zeros(n, dtype=np.float64)
    std = np.zeros(n, dtype=np.float64)
    for i, gray in enumerate(gray_frames):
        mean[i] = float(gray.mean())
        std[i] = float(gray.std())
        sharpness[i] = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    median_sharpness = float(np.median(sharpness)) if n else 0.0
    blur_threshold = blur_ratio * median_sharpness

    is_black = (mean < black_mean) & (std < black_std)
    is_white = (mean > white_mean) & (std < white_std)
    is_low_info = std < low_info_std
    is_blur = sharpness < blur_threshold

    if median_sharpness > 0:
        sharpness_score = np.clip(sharpness / (1.5 * median_sharpness), 0.0, 1.0)
    else:
        sharpness_score = np.zeros(n, dtype=np.float64)
    exposure_score = np.clip(1.0 - np.abs(mean - 128.0) / 128.0, 0.0, 1.0)
    contrast_score = np.clip(std / 60.0, 0.0, 1.0)
    quality_score = sharpness_score * exposure_score * contrast_score

    bad = is_black | is_white | is_low_info | is_blur
    low_sharpness_ratio = float(is_blur.mean()) if n else 0.0

    return FrameQuality(
        sharpness=sharpness,
        mean=mean,
        std=std,
        quality_score=quality_score,
        is_black=is_black,
        is_white=is_white,
        is_blur=is_blur,
        is_low_info=is_low_info,
        bad=bad,
        sharpness_median=median_sharpness,
        low_sharpness_ratio=low_sharpness_ratio,
    )
