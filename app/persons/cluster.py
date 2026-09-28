"""跨 track 身份聚类（约束感知的凝聚式合并）与簇内去重。

- 身份特征 = 该 track 所有合格人脸嵌入的均值；
- 相似度 ≥ 聚类阈值直接合并；歧义区间用各自质量最高的单张人脸二次复核；
- 同一帧同时出现的两个 track 视为"互斥、禁止合并"，参与合并判定以避免传递性错误。
"""

import numpy as np


def cosine(vector_a, vector_b):
    if vector_a is None or vector_b is None:
        return 0.0
    vector_a = np.asarray(vector_a, dtype=np.float32)
    vector_b = np.asarray(vector_b, dtype=np.float32)
    denom = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denom)


def mean_vector(vectors):
    vectors = [v for v in vectors if v is not None]
    if not vectors:
        return None
    stacked = np.vstack([np.asarray(v, dtype=np.float32) for v in vectors])
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-9:
        return None
    return mean / norm


def build_tracks(records):
    """返回 {track_id: 均值人脸嵌入} 与 {track_id: 质量最高的单张人脸嵌入}。"""
    grouped = {}
    for record in records:
        grouped.setdefault(record["track_id"], []).append(record)
    features = {}
    best_faces = {}
    for track_id, group in grouped.items():
        high = [r for r in group if r.get("embedding") is not None
                and not r.get("face_low")]
        low = [r for r in group if r.get("embedding") is not None
               and r.get("face_low")]
        primary = high or low
        mean = mean_vector([r["embedding"] for r in primary])
        if mean is not None:
            features[track_id] = mean
        best = None
        best_key = None
        for record in primary:
            key = (record.get("face_score") or 0.0) * record.get("face_size", 0.0)
            if best is None or key > best_key:
                best = record["embedding"]
                best_key = key
        if best is not None:
            best_faces[track_id] = best
    return features, best_faces


def conflict_pairs(records):
    by_frame = {}
    for record in records:
        by_frame.setdefault(record["frame_index"], set()).add(record["track_id"])
    conflicts = set()
    for track_ids in by_frame.values():
        ids = sorted(track_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                conflicts.add(frozenset((ids[i], ids[j])))
    return conflicts


def cluster_tracks(features, best_faces, conflicts, high=0.45, low=0.35):
    """约束感知凝聚式合并；返回 track_id -> cluster_index 的映射。"""
    high = float(high)
    low = float(low)

    track_ids = list(features.keys())
    pairs = []
    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            a, b = track_ids[i], track_ids[j]
            if frozenset((a, b)) in conflicts:
                continue
            similarity = cosine(features[a], features[b])
            if similarity >= low:
                pairs.append((similarity, a, b))
    pairs.sort(key=lambda item: item[0], reverse=True)

    clusters = {track_id: [track_id] for track_id in track_ids}
    member = {track_id: track_id for track_id in track_ids}

    def cluster_conflict(root_a, root_b):
        for x in clusters[root_a]:
            for y in clusters[root_b]:
                if x != y and frozenset((x, y)) in conflicts:
                    return True
        return False

    for similarity, a, b in pairs:
        root_a, root_b = member[a], member[b]
        if root_a == root_b:
            continue
        if similarity >= high:
            merge = True
        else:
            face_a = best_faces.get(a)
            face_b = best_faces.get(b)
            merge = (face_a is not None and face_b is not None
                     and cosine(face_a, face_b) >= high)
        if not merge or cluster_conflict(root_a, root_b):
            continue
        clusters[root_a].extend(clusters[root_b])
        for track_id in clusters[root_b]:
            member[track_id] = root_a
        del clusters[root_b]

    mapping = {}
    for index, members in enumerate(clusters.values()):
        for track_id in members:
            mapping[track_id] = index
    return mapping


def _pixel_similarity(a, b):
    return 1.0 - float(np.mean(np.abs(np.asarray(a, np.float32)
                                       - np.asarray(b, np.float32)))) / 255.0


def dedup_records(records, threshold, pixel_threshold=0.995):
    """簇内贪心去重（输入需按 score 降序）。

    - 两者都有本人脸嵌入：余弦 > threshold 视为近重复；
    - 两者都没有嵌入（背影/侧脸/无脸）：用 32x32 缩略图做近像素重复抑制；
    - 一有一无：不做嵌入比较（避免用 track 均值把不同姿态误判为重复）。
    """
    kept = []
    for record in records:
        embedding = record.get("embedding")
        duplicate = False
        for kept_record in kept:
            kept_embedding = kept_record.get("embedding")
            if embedding is not None and kept_embedding is not None:
                if cosine(embedding, kept_embedding) > threshold:
                    duplicate = True
                    break
            elif embedding is None and kept_embedding is None:
                if _pixel_similarity(record["thumb"],
                                     kept_record["thumb"]) > pixel_threshold:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(record)
    return kept
