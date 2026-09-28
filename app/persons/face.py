"""人脸检测（SCRFD）与人脸嵌入（ArcFace），基于 onnxruntime 直驱 buffalo_l。

不引入 `insightface` 包：直接加载 buffalo_l 的 `det_10g.onnx`（SCRFD，3 尺度）
与 `w600k_r50.onnx`（ArcFace，512 维），自行实现检测解码、NMS 与 5 点对齐。

输入/输出约定：
- 检测输入为 BGR uint8 图像（调用方负责裁剪到人体区域）。
- 检测返回人脸框（原图坐标）、检测分、5 关键点（原图坐标）。
- 嵌入返回 L2 归一化后的 512 维 np.float32 向量。
"""

import os

import cv2
import numpy as np
import onnxruntime as ort

ARCFACE_SIZE = (112, 112)
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

_FEAT_STRIDE = (8, 16, 32)
_NUM_ANCHORS = 2


def _distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points, distance):
    values = []
    for i in range(0, distance.shape[1], 2):
        values.append(points[:, 0] + distance[:, i])
        values.append(points[:, 1] + distance[:, i + 1])
    return np.stack(values, axis=-1)


def _nms(boxes, scores, threshold):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = order[1:][iou <= threshold]
    return keep


def _add_torch_dll_dir():
    """把包内 torch 的 lib 目录加入 DLL 搜索路径，让 onnxruntime 找到 cuDNN/cuBLAS。"""
    try:
        import torch
        lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib_dir) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(lib_dir)
    except Exception:
        pass


def _best_providers(cfg):
    device = str((cfg.get("persons") or {}).get("device") or "gpu").lower()
    available = ort.get_available_providers()
    if device == "cpu" or "CUDAExecutionProvider" not in available:
        return ["CPUExecutionProvider"], "cpu"
    _add_torch_dll_dir()
    return ["CUDAExecutionProvider", "CPUExecutionProvider"], "gpu"


class FaceEngine(object):
    def __init__(self, face_dir, cfg):
        self.face_dir = face_dir
        self.det_size = (640, 640)
        self.det_score = float((cfg.get("persons") or {}).get(
            "face_det_score", 0.55))
        self.nms_threshold = 0.4
        providers, self.device = _best_providers(cfg)

        options = ort.SessionOptions()
        options.graph_optimization_level = \
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.det_session = ort.InferenceSession(
            os.path.join(face_dir, "det_10g.onnx"), sess_options=options,
            providers=providers)
        self.rec_session = ort.InferenceSession(
            os.path.join(face_dir, "w600k_r50.onnx"), sess_options=options,
            providers=providers)
        self.det_input = self.det_session.get_inputs()[0].name
        self.rec_input = self.rec_session.get_inputs()[0].name

    def _detect_raw(self, image, threshold=None):
        threshold = self.det_score if threshold is None else float(threshold)
        height, width = image.shape[:2]
        input_w, input_h = self.det_size
        image_ratio = float(height) / float(width)
        model_ratio = float(input_h) / float(input_w)
        if image_ratio > model_ratio:
            new_height = input_h
            new_width = int(round(new_height / image_ratio))
        else:
            new_width = input_w
            new_height = int(round(new_width * image_ratio))
        new_width = max(1, min(input_w, new_width))
        new_height = max(1, min(input_h, new_height))
        det_scale = float(new_height) / float(height)
        resized = cv2.resize(image, (new_width, new_height))
        canvas = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        canvas[:new_height, :new_width, :] = resized
        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 128.0, (input_w, input_h), (127.5, 127.5, 127.5),
            swapRB=True)
        outputs = self.det_session.run(None, {self.det_input: blob})

        score_list, box_list, kps_list = [], [], []
        for index, stride in enumerate(_FEAT_STRIDE):
            scores = outputs[index]
            bbox_preds = outputs[index + 3] * stride
            kps_preds = outputs[index + 6] * stride
            feat_h = input_h // stride
            feat_w = input_w // stride
            centers = np.stack(np.mgrid[:feat_h, :feat_w][::-1],
                               axis=-1).astype(np.float32)
            centers = (centers * stride).reshape((-1, 2))
            centers = np.repeat(centers, _NUM_ANCHORS, axis=0)
            pos = np.where(scores[:, 0] >= threshold)[0]
            if len(pos) == 0:
                continue
            boxes = _distance2bbox(centers, bbox_preds)[pos]
            kps = _distance2kps(centers, kps_preds)[pos]
            score_list.append(scores[pos, 0])
            box_list.append(boxes)
            kps_list.append(kps)
        if not score_list:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), \
                np.zeros((0, 5, 2), np.float32)
        boxes = np.vstack(box_list) / det_scale
        kps = np.vstack(kps_list) / det_scale
        scores = np.concatenate(score_list)
        keep = _nms(boxes, scores, self.nms_threshold)
        boxes = boxes[keep]
        kps = kps[keep]
        scores = scores[keep]
        return boxes, scores, kps

    def detect(self, image, min_face_px=0, threshold=None):
        boxes, scores, kps = self._detect_raw(image, threshold)
        height, width = image.shape[:2]
        results = []
        for box, score, keypoints in zip(boxes, scores, kps):
            x1 = float(np.clip(box[0], 0, width - 1))
            y1 = float(np.clip(box[1], 0, height - 1))
            x2 = float(np.clip(box[2], 0, width - 1))
            y2 = float(np.clip(box[3], 0, height - 1))
            if x2 - x1 < min_face_px or y2 - y1 < min_face_px:
                continue
            results.append((
                [x1, y1, x2, y2], float(score),
                np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)))
        results.sort(key=lambda item: item[1], reverse=True)
        return results

    def embed(self, image, keypoints, face_box=None):
        matrix, _ = cv2.estimateAffinePartial2D(
            np.asarray(keypoints, dtype=np.float32), ARCFACE_TEMPLATE)
        if matrix is None:
            return None
        aligned = cv2.warpAffine(image, matrix, ARCFACE_SIZE,
                                 borderValue=0.0)
        blob = cv2.dnn.blobFromImage(
            aligned, 1.0 / 127.5, ARCFACE_SIZE, (127.5, 127.5, 127.5),
            swapRB=True)
        feat = self.rec_session.run(None, {self.rec_input: blob})[0]
        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(feat))
        if norm <= 1e-6:
            return None
        return feat / norm

    def close(self):
        self.det_session = None
        self.rec_session = None


def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)
