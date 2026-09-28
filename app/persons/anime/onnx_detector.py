"""通用 YOLO ONNX 检测器（onnxruntime 直驱）。

用于动漫人脸检测等需要逐图/逐裁剪调用的场景：letterbox 预处理、YOLOv8 输出
解码（`(N, 4+nc)` 或 `(4+nc, N)`）、单类别取最大分类分、NMS，返回原图坐标框。
`device=cpu` 或 CUDA EP 不可用时回退 CPU，不引入新依赖。
"""

import cv2
import numpy as np
import onnxruntime as ort

from ..face import _best_providers


def _letterbox(image, size):
    height, width = image.shape[:2]
    ratio = min(float(size) / float(height), float(size) / float(width))
    new_w = max(1, int(round(width * ratio)))
    new_h = max(1, int(round(height * ratio)))
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, ratio, pad_x, pad_y


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


class OnnxYoloDetector(object):
    def __init__(self, model_path, cfg, input_size=640, iou=0.5):
        providers, device = _best_providers(cfg)
        options = ort.SessionOptions()
        options.graph_optimization_level = \
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = int(input_size)
        self.iou = float(iou)
        self.device = device

    def detect(self, image_bgr, conf):
        if image_bgr is None or image_bgr.size == 0:
            return []
        height, width = image_bgr.shape[:2]
        canvas, ratio, pad_x, pad_y = _letterbox(image_bgr, self.input_size)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
        blob /= 255.0
        output = self.session.run(
            None, {self.input_name: np.ascontiguousarray(blob)})[0][0]
        if output.shape[0] < output.shape[1]:
            output = output.T
        if output.shape[1] < 5:
            return []
        scores = output[:, 4:].max(axis=1)
        keep = scores >= float(conf)
        output = output[keep]
        scores = scores[keep]
        if len(output) == 0:
            return []
        cx, cy, bw, bh = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
        boxes = np.stack([
            (cx - bw / 2.0 - pad_x) / ratio,
            (cy - bh / 2.0 - pad_y) / ratio,
            (cx + bw / 2.0 - pad_x) / ratio,
            (cy + bh / 2.0 - pad_y) / ratio,
        ], axis=1)
        index = _nms(boxes, scores, self.iou)
        results = []
        for i in index:
            x1 = float(np.clip(boxes[i, 0], 0, width - 1))
            y1 = float(np.clip(boxes[i, 1], 0, height - 1))
            x2 = float(np.clip(boxes[i, 2], 0, width - 1))
            y2 = float(np.clip(boxes[i, 3], 0, height - 1))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            results.append(([x1, y1, x2, y2], float(scores[i])))
        results.sort(key=lambda item: item[1], reverse=True)
        return results

    def close(self):
        self.session = None
