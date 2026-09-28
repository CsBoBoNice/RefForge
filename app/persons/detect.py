"""人物检测与跟踪：YOLO11-pose + BoT-SORT / ByteTrack。

对代表帧序列按时间顺序逐帧调用 `YOLO.track`，输出每个人体的框、track_id、
置信度与 COCO 17 关键点（含可见性）；track 仅用于帧间锚定。

真人领域用 `botsort.yaml`（带 ReID）；动漫领域用 `bytetrack.yaml`（纯运动关联，
对低帧率/重复帧更稳），并由子类按人脸锚点补充人物框。
"""


class PersonTracker(object):
    def __init__(self, models, cfg, tracker_file="botsort.yaml"):
        pc = models.persons_params(cfg)
        self.model = models.yolo()
        self.conf = float(pc.get("det_conf", 0.45))
        self.tracker_file = tracker_file
        self.device = "cuda:0" if models.device == "gpu" else "cpu"
        self._reset_tracker()

    def _reset_tracker(self):
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            return
        for tracker in (getattr(predictor, "trackers", None) or []):
            try:
                tracker.reset()
            except Exception:
                pass

    def update(self, frame_bgr, whole_faces=None):
        results = self.model.track(
            source=frame_bgr, persist=True, classes=[0], conf=self.conf,
            tracker=self.tracker_file, verbose=False, device=self.device)
        result = results[0]
        detections = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        ids = None
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
        kps = None
        kps_conf = None
        if result.keypoints is not None:
            kps = result.keypoints.xy.cpu().numpy()
            if getattr(result.keypoints, "conf", None) is not None:
                kps_conf = result.keypoints.conf.cpu().numpy()
        for i in range(len(boxes)):
            detections.append({
                "bbox": [float(v) for v in boxes[i]],
                "conf": float(confs[i]),
                "track_id": int(ids[i]) if ids is not None else None,
                "kps": kps[i] if kps is not None else None,
                "kps_conf": kps_conf[i] if kps_conf is not None else None,
            })
        return detections
