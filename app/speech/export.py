"""字幕与结构化转录的导出工具。"""

import os

from io_utils import ensure_dir


def _format_srt_time(seconds):
    total_ms = max(0, int(round(float(seconds) * 1000.0)))
    hours, rem = divmod(total_ms, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, millis = divmod(rem, 1000)
    return "%02d:%02d:%02d,%03d" % (hours, minutes, secs, millis)


def write_srt(segments, out_path):
    """写出整视频 SRT；每段带说话人前缀（全局唯一 ID）。"""
    ensure_dir(os.path.dirname(out_path))
    blocks = []
    for index, segment in enumerate(segments, 1):
        speaker = segment.get("speaker") or "unknown"
        text = str(segment.get("text") or "").strip()
        blocks.append(
            "%d\n%s --> %s\n%s: %s\n" % (
                index,
                _format_srt_time(segment.get("start_sec", 0.0)),
                _format_srt_time(segment.get("end_sec", 0.0)),
                speaker,
                text,
            ))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(blocks))
    return out_path


def safe_speaker_name(speaker_id):
    name = str(speaker_id or "unknown").strip() or "unknown"
    for ch in '\\/:*?"<>|':
        name = name.replace(ch, "_")
    return name


def segment_payload(speaker, start_sec, end_sec, text, language, words,
                    index=None):
    payload = {
        "speaker": speaker,
        "start_sec": round(float(start_sec), 3),
        "end_sec": round(float(end_sec), 3),
        "text": text,
        "language": language,
        "words": words,
    }
    if index is not None:
        payload["index"] = int(index)
    return payload
