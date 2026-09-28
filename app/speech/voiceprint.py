"""全局说话人声纹中心库（JSON 文件存储，无数据库）。

结构：
{
  "schema_version": "1.0",
  "engine": "campplus+eres2netv2",
  "thresholds": {"campplus": 0.30, "eres2netv2": 0.55},
  "next_id": 3,
  "speakers": {
    "speaker_1": {
      "campplus_center": [192 维],
      "eres2netv2_center": [192 维],
      "samples": 7,
      "videos": ["a.mp4", "b.mp4"]
    }
  }
}

同一说话人在全局（完整视频 JSON / 镜头 JSON / 按人归档）中 ID 一致。
ERes2NetV2 相似度作为主匹配依据，CAM++ 作为一致性校验（两者都过才合并），
从而在跨视频合并时尽量降低误合并。
"""

import json
import os

import numpy as np

from io_utils import package_root


def _normalize(vector):
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr
    return arr / norm


def _cosine(a, b):
    va = _normalize(a)
    vb = _normalize(b)
    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return -1.0
    return float(np.dot(va, vb))


class VoiceprintCenter:
    def __init__(self, cfg, output_root):
        asr = cfg.get("asr") or {}
        value = str(asr.get("voiceprint_center") or "").strip()
        if not value:
            self.path = os.path.join(output_root, "voiceprint_center.json")
        elif os.path.isabs(value):
            self.path = os.path.normpath(value)
        else:
            self.path = os.path.normpath(os.path.join(package_root(), value))
        self.cam_threshold = float(asr.get("voiceprint_sim_threshold", 0.62))
        self.eres_threshold = float(asr.get("verify_sim_threshold", 0.66))
        self.db = self._load()

    def _load(self):
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and isinstance(
                        data.get("speakers"), dict):
                    data.setdefault("next_id", 1)
                    return data
            except (OSError, ValueError):
                pass
        return {
            "schema_version": "1.0",
            "engine": "campplus+eres2netv2",
            "thresholds": {"campplus": self.cam_threshold,
                           "eres2netv2": self.eres_threshold},
            "next_id": 1,
            "speakers": {},
        }

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.db, fh, ensure_ascii=False, indent=2)
        return self.path

    def _update_center(self, center, sample, count):
        old = np.asarray(center, dtype=np.float32)
        new = np.asarray(sample, dtype=np.float32)
        if old.size == 0:
            return _normalize(new).tolist()
        merged = (old * float(count) + new) / (float(count) + 1.0)
        return _normalize(merged).tolist()

    def match_or_create(self, cam_embedding, eres_embedding, video_name=""):
        """按声纹匹配已有全局说话人，未达阈值则新建；返回全局唯一 ID。

        短窗声纹下 ERes2NetV2 比 CAM++ 更稳定，故以 ERes 相似度为主判据，
        CAM++ 相似度作为一致性校验（两者都达标才合并）。
        """
        cam = _normalize(cam_embedding).tolist()
        eres = (_normalize(eres_embedding).tolist()
                if eres_embedding is not None else None)

        best_id = None
        best_score = -1.0
        best_cam = -1.0
        best_eres = -1.0
        for spk_id, info in self.db["speakers"].items():
            cam_center = info.get("campplus_center")
            eres_center = info.get("eres2netv2_center")
            cam_sim = _cosine(cam, cam_center) if cam_center else -1.0
            eres_sim = (_cosine(eres, eres_center)
                        if (eres is not None and eres_center) else -1.0)
            score = eres_sim if eres_sim >= 0 else cam_sim
            if score > best_score:
                best_score, best_id = score, spk_id
                best_cam, best_eres = cam_sim, eres_sim

        if best_id is not None:
            if best_eres >= 0:
                accept = (best_eres >= self.eres_threshold
                          and best_cam >= self.cam_threshold)
            else:
                accept = best_cam >= self.cam_threshold
            if accept:
                info = self.db["speakers"][best_id]
                count = int(info.get("samples", 1))
                info["campplus_center"] = self._update_center(
                    info.get("campplus_center"), cam, count)
                if eres is not None:
                    info["eres2netv2_center"] = self._update_center(
                        info.get("eres2netv2_center"), eres, count)
                info["samples"] = count + 1
                if video_name and video_name not in info.setdefault("videos", []):
                    info["videos"].append(video_name)
                self.save()
                return best_id

        spk_id = "speaker_%d" % int(self.db.get("next_id", 1))
        self.db["next_id"] = int(self.db.get("next_id", 1)) + 1
        info = {
            "campplus_center": cam,
            "samples": 1,
            "videos": [video_name] if video_name else [],
        }
        if eres is not None:
            info["eres2netv2_center"] = eres
        self.db["speakers"][spk_id] = info
        self.save()
        return spk_id
