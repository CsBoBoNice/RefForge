"""语音活动检测（fsmn-vad）、声纹嵌入（CAM++ / ERes2NetV2）与本地聚类。

说明：
- VAD 片段即“说话轮次”候选；对每个轮次提取 CAM++ 与 ERes2NetV2 声纹向量。
- 本地聚类只用 CAM++（发音人区分），跨视频合并由全局声纹中心负责（CAM++ 匹配 +
  ERes2NetV2 验证）。
"""

import numpy as np

_SR = 16000


def _normalize(vector):
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr
    return arr / norm


def detect_speech(vad_model, wav_path):
    """返回语音区间列表 [(start_sec, end_sec), ...]（按时间升序，已剔除无效项）。"""
    result = vad_model.generate(input=wav_path)
    if not result:
        return []
    value = result[0].get("value") if isinstance(result[0], dict) else None
    regions = []
    for item in (value or []):
        try:
            start_ms, end_ms = int(item[0]), int(item[1])
        except (TypeError, IndexError, ValueError):
            continue
        if start_ms < 0 or end_ms <= start_ms:
            continue
        regions.append((start_ms / 1000.0, end_ms / 1000.0))
    regions.sort(key=lambda pair: pair[0])
    return regions


def extract_embedding(embedder, clip, sample_rate=_SR):
    """提取单段音频的声纹向量，返回一维 np.float32 数组；失败返回 None。"""
    if clip is None or len(clip) == 0:
        return None
    try:
        result = embedder.generate(input=np.asarray(clip, dtype=np.float32),
                                   fs=int(sample_rate))
    except Exception:
        return None
    if not result:
        return None
    emb = result[0].get("spk_embedding") if isinstance(result[0], dict) else None
    if emb is None:
        return None
    if hasattr(emb, "detach"):
        emb = emb.detach().cpu().numpy()
    arr = np.asarray(emb, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    return arr


def cluster_embeddings(embeddings, threshold):
    """贪心余弦聚类，返回每个向量所属的本地说话人编号列表。

    先到先得：与已有簇心相似度达到阈值则并入，否则新建簇。
    """
    labels = [0] * len(embeddings)
    centers = []
    members = []
    for index, embedding in enumerate(embeddings):
        vector = _normalize(embedding)
        best_index, best_sim = -1, -1.0
        for cluster_index, center in enumerate(centers):
            sim = float(np.dot(vector, center))
            if sim > best_sim:
                best_sim, best_index = sim, cluster_index
        if best_index >= 0 and best_sim >= float(threshold):
            members[best_index].append(index)
            group = np.stack([_normalize(embeddings[i])
                              for i in members[best_index]])
            centers[best_index] = _normalize(group.mean(axis=0))
            labels[index] = best_index
        else:
            centers.append(vector)
            members.append([index])
            labels[index] = len(centers) - 1
    return labels


def make_windows(regions, window_sec, hop_sec):
    """把语音区间切成重叠的短窗（用于细粒度说话人区分）。

    - 区间不长于 window_sec 时整段作为一个窗；
    - 否则按 hop_sec 步进滑窗，并在末尾补一个覆盖结尾的窗，保证全覆盖。
    返回按时间升序的 [(start_sec, end_sec), ...]。
    """
    window = max(0.2, float(window_sec))
    hop = max(0.1, float(hop_sec))
    windows = []
    for start, end in regions:
        length = float(end) - float(start)
        if length <= window:
            windows.append((float(start), float(end)))
            continue
        position = float(start)
        while position + window <= float(end) + 1e-6:
            windows.append((position, position + window))
            position += hop
        if windows and windows[-1][1] < float(end) - 1e-6:
            windows.append((float(end) - window, float(end)))
    windows.sort(key=lambda pair: pair[0])
    return windows


def smooth_labels(labels, radius=1):
    """对按时间排序的窗口标签做多数滤波，抑制边界处偶发误判。"""
    if radius <= 0 or len(labels) <= 2:
        return list(labels)
    out = list(labels)
    for index in range(len(labels)):
        lo = max(0, index - radius)
        hi = min(len(labels), index + radius + 1)
        neighbourhood = labels[lo:hi]
        out[index] = max(set(neighbourhood),
                         key=lambda label: neighbourhood.count(label))
    return out
