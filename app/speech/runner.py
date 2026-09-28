"""语音识别阶段编排：抽音 -> VAD -> 声纹聚类 -> 全局说话人 -> 转写 -> 导出。

对每个完整视频产出：
- `<输入视频同目录>/<源文件名>.srt`：带说话人前缀的字幕；
- `<output>/segments/<源文件名>/transcript.json`：整视频结构化转录（谁/何时/什么）；
- `<output>/segments/<源文件名>/speakers/<speaker>/`：1~30s 语音归档
  （`seg_XXXX.wav` + `seg_XXXX.json` + `<speaker>_global.json`）；
- 每个 `shot_XXXX/asr.json`：按镜头切分的转录结果；
- `<output>/voiceprint_center.json`：全局说话人声纹中心库（跨视频合并）。
"""

import os
import shutil
import tempfile
import time
import traceback

import numpy as np

from io_utils import ensure_dir, safe_stem, shot_detail_dir, write_json

from . import audio as audio_mod
from . import diarize as diarize_mod
from . import export as export_mod
from . import models as models_mod
from . import transcribe as transcribe_mod
from .voiceprint import VoiceprintCenter

UNKNOWN_SPEAKER = "unknown"


def _assign_labels(cam_embeddings, threshold):
    """对 CAM++ 窗口向量贪心余弦聚类；嵌入缺失的窗口归入时间上最近的簇。"""
    count = len(cam_embeddings)
    labels = [None] * count
    valid = [i for i, emb in enumerate(cam_embeddings)
             if emb is not None and getattr(emb, "size", 0) > 0]
    if not valid:
        return [0] * count
    assigned = diarize_mod.cluster_embeddings(
        [cam_embeddings[i] for i in valid], threshold)
    for position, index in enumerate(valid):
        labels[index] = assigned[position]
    for index in range(count):
        if labels[index] is None:
            nearest = min(valid, key=lambda other: abs(other - index))
            labels[index] = labels[nearest]
    return labels


def _write_empty(video_path, source_dir, shots, logger, reason):
    stem = safe_stem(video_path)
    source_file = os.path.basename(video_path)
    write_json(os.path.join(source_dir, "transcript.json"), {
        "schema_version": "1.0",
        "source": {"file": source_file},
        "speaker_count": 0,
        "segments": [],
        "reason": reason,
    })
    export_mod.write_srt(
        [], os.path.join(os.path.dirname(video_path), stem + ".srt"))
    for shot in shots:
        write_json(os.path.join(shot_detail_dir(shot["shot_dir"]), "asr.json"), {
            "schema_version": "1.0",
            "source": source_file,
            "shot_id": shot.get("shot_id"),
            "shot_start_sec": round(float(shot.get("start_sec", 0.0)), 3),
            "shot_end_sec": round(float(shot.get("end_sec", 0.0)), 3),
            "text": "",
            "segments": [],
            "reason": reason,
        })
        export_mod.write_srt(
            [], os.path.join(shot["shot_dir"], _shot_srt_name(shot)))
    if logger:
        logger.info("  asr %s: no speech (%s)", source_file, reason)


def _write_transcript(video_path, source_dir, segments, duration_sec):
    source_file = os.path.basename(video_path)
    speakers = sorted({seg["speaker"] for seg in segments if seg.get("speaker")})
    write_json(os.path.join(source_dir, "transcript.json"), {
        "schema_version": "1.0",
        "source": {"file": source_file,
                   "duration_sec": round(float(duration_sec), 3)},
        "speaker_count": len(speakers),
        "speakers": speakers,
        "segments": segments,
    })
    export_mod.write_srt(
        segments,
        os.path.join(os.path.dirname(video_path),
                     safe_stem(video_path) + ".srt"))


def _assign_segments_to_shots(shots, segments):
    """把每个语音段分配给与其时间重叠最多的镜头（跨切点的连续说话按多数归属）。

    只保留该镜头内确实能听到的内容：同一语音段只会出现在一个镜头里，
    不再因“跨切点”同时出现在相邻两个镜头。
    """
    buckets = [[] for _ in shots]
    for seg in segments:
        seg_start = float(seg["start_sec"])
        seg_end = float(seg["end_sec"])
        best_index = -1
        best_overlap = 0.0
        for index, shot in enumerate(shots):
            low = max(seg_start, float(shot.get("start_sec", 0.0)))
            high = min(seg_end, float(shot.get("end_sec", 0.0)))
            overlap = high - low
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
        if best_index >= 0:
            buckets[best_index].append(seg)
    return buckets


def _shot_srt_name(shot):
    filename = str(shot.get("file") or "")
    if filename.lower().endswith(".mp4"):
        return filename[:-4] + ".srt"
    return "%s.srt" % (shot.get("shot_id") or "shot")


def _write_shot_jsons(source_dir, shots, segments, video_path):
    source_file = os.path.basename(video_path)
    buckets = _assign_segments_to_shots(shots, segments)
    for shot, subset in zip(shots, buckets):
        start = float(shot.get("start_sec", 0.0))
        end = float(shot.get("end_sec", 0.0))
        clean = [{key: value for key, value in seg.items() if key != "index"}
                 for seg in subset]
        shot_dir = shot["shot_dir"]
        write_json(os.path.join(shot_detail_dir(shot_dir), "asr.json"), {
            "schema_version": "1.0",
            "source": source_file,
            "shot_id": shot.get("shot_id"),
            "shot_start_sec": round(start, 3),
            "shot_end_sec": round(end, 3),
            "text": "\n".join(seg["text"] for seg in clean if seg["text"]),
            "segments": clean,
        })
        # 每段字幕：内容与 asr.json 一致，时间改为相对该片段起点的秒数，
        # 便于直接配合 `seg_XXXX.mp4` 播放。
        srt_segments = []
        for seg in clean:
            rel_start = max(0.0, float(seg["start_sec"]) - start)
            rel_end = min(end, float(seg["end_sec"])) - start
            if rel_end <= rel_start:
                continue
            srt_segments.append({
                "speaker": seg.get("speaker"),
                "start_sec": rel_start,
                "end_sec": rel_end,
                "text": seg.get("text", ""),
            })
        export_mod.write_srt(
            srt_segments, os.path.join(shot_dir, _shot_srt_name(shot)))


def _archive_speakers(source_dir, video_path, segments, audio, sample_rate,
                      min_sec, max_sec, logger):
    source_file = os.path.basename(video_path)
    speakers_root = os.path.join(source_dir, "speakers")
    ensure_dir(speakers_root)
    archived = {}
    for index, seg in enumerate(segments):
        duration = float(seg["end_sec"]) - float(seg["start_sec"])
        speaker = seg.get("speaker")
        if not speaker or speaker == UNKNOWN_SPEAKER:
            continue
        if duration < float(min_sec) or duration > float(max_sec):
            continue
        speaker_dir = os.path.join(
            speakers_root, export_mod.safe_speaker_name(speaker))
        ensure_dir(speaker_dir)
        existing = [name for name in os.listdir(speaker_dir)
                    if name.startswith("seg_") and name.endswith(".wav")]
        sequence = len(existing) + 1
        wav_name = "seg_%04d.wav" % sequence
        clip = audio_mod.cut_clip(audio, sample_rate, seg["start_sec"],
                                  seg["end_sec"])
        audio_mod.write_wav(os.path.join(speaker_dir, wav_name),
                            clip, sample_rate)
        payload = {
            "source": source_file,
            "speaker": speaker,
            "start_sec": round(float(seg["start_sec"]), 3),
            "end_sec": round(float(seg["end_sec"]), 3),
            "duration": round(duration, 3),
            "text": seg.get("text", ""),
            "language": seg.get("language", ""),
            "words": seg.get("words", []),
        }
        write_json(os.path.join(speaker_dir, "seg_%04d.json" % sequence),
                   payload)
        archived.setdefault(speaker, 0)
        archived[speaker] += 1

    for name in os.listdir(speakers_root):
        speaker_dir = os.path.join(speakers_root, name)
        if not os.path.isdir(speaker_dir):
            continue
        json_files = sorted(f for f in os.listdir(speaker_dir)
                            if f.startswith("seg_") and f.endswith(".json"))
        items = []
        for json_file in json_files:
            with open(os.path.join(speaker_dir, json_file),
                      "r", encoding="utf-8") as fh:
                import json
                items.append(json.load(fh))
        write_json(os.path.join(speaker_dir, "%s_global.json" % name), {
            "source": source_file,
            "speaker": name,
            "count": len(items),
            "total_duration": round(
                sum(float(item.get("duration", 0.0)) for item in items), 3),
            "segments": items,
        })
    if logger and archived:
        logger.info("  asr archive: %s",
                    ", ".join("%s=%d" % (key, value)
                              for key, value in sorted(archived.items())))


def _cluster_representatives_by_audio(embedder, clips, labels, sample_rate):
    """用同一本地簇全部窗口的拼接音频提取声纹，得到更稳定的簇代表向量。"""
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    reps = {}
    for label, indices in groups.items():
        parts = [clips[i] for i in indices
                 if clips[i] is not None and len(clips[i]) > 0]
        if not parts:
            continue
        concatenated = np.concatenate(parts)
        embedding = diarize_mod.extract_embedding(embedder, concatenated,
                                                  sample_rate)
        if embedding is not None:
            reps[label] = embedding
    return reps


def _speaker_for_span(window_times, window_speakers, start, end):
    """按与句子时间区间重叠的窗口投票决定说话人；无重叠则取最近窗口。"""
    votes = {}
    for (win_start, win_end), speaker in zip(window_times, window_speakers):
        if win_end > start and win_start < end:
            votes[speaker] = votes.get(speaker, 0) + 1
    if votes:
        return max(votes, key=lambda key: (votes[key], key))
    if not window_times:
        return UNKNOWN_SPEAKER
    middle = (start + end) / 2.0
    nearest = min(range(len(window_times)),
                  key=lambda i: abs(
                      (window_times[i][0] + window_times[i][1]) / 2.0 - middle))
    return window_speakers[nearest]


def _build_segments(keep, keep_regions, asr_outputs, window_times,
                    window_labels, language, merge_gap):
    """把每个 VAD 区间的转写按句切分，逐句映射到本地说话人并合并相邻同人句。

    返回 (grouped, used_clusters)：grouped 每项含 `_cluster`（本地簇编号）。
    """
    grouped = []
    for order, source_index in enumerate(keep):
        region_start, region_end = keep_regions[order]
        output = asr_outputs[source_index]
        words = [{
            "text": str(word.get("text", "")),
            "start": round(float(word.get("start", 0.0)) + region_start, 3),
            "end": round(float(word.get("end", 0.0)) + region_start, 3),
        } for word in (output.get("words", []) or [])]
        sentences = transcribe_mod.split_sentences_with_words(
            output.get("text", ""), words)
        if not sentences:
            continue
        lang = output.get("language") or (language or "")
        for sentence_text, first, last in sentences:
            sentence_words = words[first:last + 1]
            valid = [w for w in sentence_words if w["end"] >= w["start"]]
            if valid:
                span_start = valid[0]["start"]
                span_end = valid[-1]["end"]
            else:
                span_start, span_end = region_start, region_end
            cluster = _speaker_for_span(
                window_times, window_labels, span_start, span_end)
            previous = grouped[-1] if grouped else None
            if (previous is not None and previous["region"] == region_start
                    and previous["_cluster"] == cluster
                    and span_start - previous["end_sec"] <= merge_gap):
                previous["end_sec"] = round(span_end, 3)
                previous["text"] = transcribe_mod.join_text(
                    previous["text"], sentence_text)
                previous["words"].extend(sentence_words)
            else:
                grouped.append({
                    "_cluster": cluster,
                    "region": region_start,
                    "start_sec": round(span_start, 3),
                    "end_sec": round(span_end, 3),
                    "text": sentence_text,
                    "language": lang,
                    "words": list(sentence_words),
                })
    used = {seg["_cluster"] for seg in grouped}
    return grouped, used


def process_video(job, cfg, model, center, logger, stats, debug=False):
    video_path = job["video"]
    source_dir = job["source_dir"]
    shots = job.get("shots") or []
    asr_cfg = cfg.get("asr") or {}
    language = models_mod.configure_language(cfg)
    sample_rate = int(asr_cfg.get("sample_rate", 16000))
    min_sec = float(asr_cfg.get("min_archive_sec", 1.0))
    max_sec = float(asr_cfg.get("max_archive_sec", 30.0))
    min_speech = float(asr_cfg.get("min_speech_sec", 0.6))
    threshold = float(asr_cfg.get("local_speaker_threshold", 0.55))
    window_sec = float(asr_cfg.get("diarize_window_sec", 1.2))
    hop_sec = float(asr_cfg.get("diarize_hop_sec", 0.4))
    smooth_radius = int(asr_cfg.get("diarize_smooth_radius", 1))
    merge_gap = float(asr_cfg.get("sentence_merge_gap_sec", 0.8))

    ensure_dir(source_dir)
    temp_dir = tempfile.mkdtemp(prefix="speech_")
    wav_path = os.path.join(temp_dir, "audio.wav")
    vad_path = os.path.join(temp_dir, "vad.wav")
    started = time.monotonic()
    # 默认对分离后的人声轨做识别；无人声分离结果时回退到源视频原始音轨。
    audio_source = job.get("audio_source") or video_path
    try:
        if not audio_mod.extract_audio(audio_source, wav_path, sample_rate):
            if audio_mod.has_audio_stream(audio_source) is False:
                _write_empty(video_path, source_dir, shots, logger, "no_audio")
                stats["asr_ok"] = stats.get("asr_ok", 0) + 1
                return
            raise RuntimeError("audio_extract_failed")

        data, sr = audio_mod.read_wav_mono(wav_path)
        duration = len(data) / float(sr) if sr else 0.0

        # VAD 分句始终基于**原始音轨**：分离后的人声轨常残留背景音乐，
        # 会把整段并成一句（无静音边界），导致 ForcedAligner 词级时间戳漂移。
        # 识别 / 声纹 / 归档仍使用分离后的人声轨（data）。
        vad_input = wav_path
        if audio_source != video_path and audio_mod.extract_audio(
                video_path, vad_path, sample_rate):
            vad_input = vad_path

        regions = diarize_mod.detect_speech(model.vad(), vad_input)
        if min_speech > 0:
            regions = [region for region in regions
                       if region[1] - region[0] >= min_speech]
        if not regions:
            _write_empty(video_path, source_dir, shots, logger, "no_speech")
            stats["asr_ok"] = stats.get("asr_ok", 0) + 1
            return

        clips = [audio_mod.cut_clip(data, sr, start, end)
                 for start, end in regions]

        # 先转写（拿到词级时间戳），再只对有实际语音文本的片段做声纹分析，
        # 避免纯音效/静音片段污染声纹中心库。
        asr_outputs = transcribe_mod.transcribe_clips(
            model.asr(), clips, language=language, sample_rate=sr)
        keep = [index for index, output in enumerate(asr_outputs)
                if str(output.get("text", "")).strip()]
        if not keep:
            _write_empty(video_path, source_dir, shots, logger,
                         "empty_transcript")
            stats["asr_ok"] = stats.get("asr_ok", 0) + 1
            return

        # 细粒度说话人区分：在语音区间上滑窗做声纹聚类（而非整个 VAD 段一个说话人），
        # 再按标点分句把句子映射到说话人，解决同一 VAD 段内多说话人。
        keep_regions = [regions[index] for index in keep]
        window_times = diarize_mod.make_windows(
            keep_regions, window_sec, hop_sec)
        window_clips = [audio_mod.cut_clip(data, sr, start, end)
                        for start, end in window_times]
        speaker = model.speaker_embedder()
        verify = model.verify_embedder()
        cam_embeddings = [diarize_mod.extract_embedding(speaker, clip, sr)
                          for clip in window_clips]
        eres_embeddings = [diarize_mod.extract_embedding(verify, clip, sr)
                           for clip in window_clips]

        labels = diarize_mod.smooth_labels(
            _assign_labels(cam_embeddings, threshold), smooth_radius)
        rep_cam = _cluster_representatives_by_audio(
            speaker, window_clips, labels, sr)
        rep_eres = _cluster_representatives_by_audio(
            verify, window_clips, labels, sr)

        grouped, used = _build_segments(
            keep, keep_regions, asr_outputs, window_times, labels,
            language, merge_gap)
        if not grouped:
            _write_empty(video_path, source_dir, shots, logger,
                         "empty_transcript")
            stats["asr_ok"] = stats.get("asr_ok", 0) + 1
            return

        # 只为真正被句子用到的本地簇登记全局说话人，避免边界簇污染中心库。
        source_file = os.path.basename(video_path)
        cluster_to_speaker = {}
        for cluster_id in used:
            cluster_to_speaker[cluster_id] = center.match_or_create(
                rep_cam.get(cluster_id), rep_eres.get(cluster_id), source_file)
        segments = [export_mod.segment_payload(
            cluster_to_speaker.get(seg["_cluster"], UNKNOWN_SPEAKER),
            seg["start_sec"], seg["end_sec"], seg["text"], seg["language"],
            seg["words"], index=index)
            for index, seg in enumerate(grouped)]

        _write_transcript(video_path, source_dir, segments, duration)
        _write_shot_jsons(source_dir, shots, segments, video_path)
        _archive_speakers(source_dir, video_path, segments, data, sr,
                          min_sec, max_sec, logger)

        stats["asr_ok"] = stats.get("asr_ok", 0) + 1
        stats["asr_segments"] = stats.get("asr_segments", 0) + len(segments)
        stats["asr_speakers"] = len({seg["speaker"] for seg in segments})
        if logger:
            logger.info(
                "  asr %s: regions=%d speakers=%s source=%s elapsed=%.1fs",
                os.path.basename(video_path), len(regions),
                sorted({seg["speaker"] for seg in segments}),
                "vocals" if job.get("audio_source") else "original",
                time.monotonic() - started)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_phase(jobs, output_root, cfg, logger, debug=False):
    """加载模型一次，顺序处理所有视频，结束后卸载。返回统计。"""
    stats = {"asr_ok": 0, "asr_error": 0, "asr_segments": 0, "asr_speakers": 0}
    if not jobs:
        return stats

    ready, missing = models_mod.available(cfg)
    if not ready:
        if logger:
            logger.warning("asr disabled; missing: %s", ", ".join(missing))
        return stats

    model = models_mod.get_models(cfg)
    center = VoiceprintCenter(cfg, output_root)
    if logger:
        logger.info("asr phase: %d source(s), device=%s, center=%s",
                    len(jobs), model.device, center.path)
    try:
        for job in jobs:
            try:
                process_video(job, cfg, model, center, logger, stats,
                              debug=debug)
            except Exception as exc:
                stats["asr_error"] = stats.get("asr_error", 0) + 1
                if logger:
                    logger.error("  asr %s failed: %s",
                                 os.path.basename(job.get("video", "?")), exc)
                    logger.debug(traceback.format_exc())
    finally:
        center.save()
        models_mod.shutdown()
    return stats
