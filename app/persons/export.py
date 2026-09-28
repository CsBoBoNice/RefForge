"""人物截图的落盘与 metadata.json 写出。"""

import os

import cv2

from io_utils import ensure_dir, save_bgr_jpeg, write_json


class FrameReader(object):
    """按源帧号回读原分辨率彩色帧（顺序或跳转）。"""

    def __init__(self, path):
        self.capture = cv2.VideoCapture(path)
        self.position = -1

    def get(self, frame_index):
        if self.position != frame_index - 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = self.capture.read()
        self.position = frame_index if ok else -1
        return frame if ok else None

    def close(self):
        self.capture.release()


def _representative(records):
    def key(record):
        return (record.get("face_size") or 0.0,
                (record["crop"][2] - record["crop"][0])
                * (record["crop"][3] - record["crop"][1]))
    return max(records, key=key)


def _save_records(reader, records, out_dir, prefix, quality):
    ensure_dir(out_dir)
    saved = []
    for record in records:
        frame = reader.get(record["frame_index"])
        if frame is None:
            continue
        x1, y1, x2, y2 = record["crop"]
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            continue
        name = "%s_f%07d.jpg" % (prefix, record["frame_index"])
        path = os.path.join(out_dir, name)
        save_bgr_jpeg(path, crop, quality)
        record["file"] = name
        saved.append(record)
    return saved


def _person_metadata(records):
    times = [r["time_sec"] for r in records]
    frames = [r["frame_index"] for r in records]
    levels = {"full_body": 0, "upper_body": 0, "torso": 0, "chest": 0,
              "head": 0, "unknown": 0}
    for record in records:
        levels[record.get("level", "unknown")] = \
            levels.get(record.get("level", "unknown"), 0) + 1
    tracks = sorted(set(r["track_id"] for r in records))
    representative = _representative(records)
    return {
        "num_images": len(records),
        "tracks_merged": tracks,
        "time_range": [round(min(times), 3), round(max(times), 3)],
        "frame_range": [int(min(frames)), int(max(frames))],
        "representative": representative.get("file"),
        "levels": levels,
    }


def write_outputs(persons_dir, person_groups, unknown_groups, meta, video,
                  pc, logger, domain="real"):
    quality = int(pc.get("jpeg_quality", 92))
    keep_unknown = bool(pc.get("keep_unknown", True))
    ensure_dir(persons_dir)
    reader = FrameReader(video)
    metadata = {
        "schema_version": "1.0",
        "domain": str(domain or "real"),
        "source": {
            "file": os.path.basename(video),
            "fps": round(float(meta.fps), 3),
            "frame_count": int(meta.frame_count),
            "duration_sec": round(float(meta.duration_sec), 3),
        },
        "cluster_threshold": float(pc.get("cluster_threshold", 0.45)),
        "ambig_low": float(pc.get("cluster_ambiguous_low", 0.35)),
        "dedup_threshold": float(pc.get("dedup_threshold", 0.92)),
        "max_per_person": int(pc.get("max_per_person", 300)),
        "person_count": len(person_groups),
        "unknown_track_count": len(unknown_groups) if keep_unknown else 0,
        "persons": {},
        "unknown": {},
    }
    try:
        for index, records in enumerate(person_groups, start=1):
            person_id = "person_%03d" % index
            saved = _save_records(reader, records,
                                  os.path.join(persons_dir, person_id),
                                  person_id, quality)
            if not saved:
                continue
            metadata["persons"][person_id] = _person_metadata(saved)
            logger.info("  %s: %d images (tracks=%d)", person_id, len(saved),
                        len(metadata["persons"][person_id]["tracks_merged"]))
        if keep_unknown:
            for index, records in enumerate(unknown_groups, start=1):
                track_dir = "track_%04d" % index
                saved = _save_records(reader, records,
                                      os.path.join(persons_dir, "unknown",
                                                   track_dir),
                                      "unknown", quality)
                if not saved:
                    continue
                times = [r["time_sec"] for r in saved]
                metadata["unknown"][track_dir] = {
                    "source_track_id": saved[0]["track_id"],
                    "num_images": len(saved),
                    "time_range": [round(min(times), 3), round(max(times), 3)],
                }
    finally:
        reader.close()
    write_json(os.path.join(persons_dir, "metadata.json"), metadata)
    return metadata
