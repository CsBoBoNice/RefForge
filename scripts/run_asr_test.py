"""语音识别验收：对 `input/` 下每个真实视频核对 SRT / 结构化 JSON / 按人归档 / 声纹中心。

用法：dev.bat scripts\\run_asr_test.py

素材取自 `input/`（有什么视频就用什么视频），复制到系统临时目录后运行，
不污染 `input/` / `output/`（整视频 SRT 写在临时输入目录）。需要 `models/asr/`
下模型齐全，首次运行会加载 Qwen3-ASR（GPU 约 8s）。
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import acceptance_common as common  # noqa: E402

from config import load_config  # noqa: E402
import speech  # noqa: E402


def _check_transcript(errors, name, transcript):
    speakers = set()
    segments = transcript.get("segments", [])
    for seg in segments:
        speaker = seg.get("speaker")
        speakers.add(speaker)
        if not speaker or not speaker.startswith("speaker_"):
            errors.append("%s bad speaker id %r" % (name, speaker))
        if not str(seg.get("text", "")).strip():
            errors.append("%s empty text segment" % name)
        if not seg.get("language"):
            errors.append("%s missing language" % name)
    return segments, speakers


def _check_shots(errors, name, source_dir, manifest, audio_present):
    for seg in manifest.get("segments", []):
        shot_dir = os.path.join(source_dir, seg["shot_dir"])
        detail_dir = os.path.join(shot_dir, "detailed")
        if not os.path.isfile(os.path.join(detail_dir, "asr.json")):
            errors.append("%s %s missing detailed/asr.json" % (
                name, seg["shot_dir"]))
        seg_file = seg.get("file") or ""
        srt_name = (os.path.splitext(seg_file)[0] + ".srt"
                    if seg_file.lower().endswith(".mp4")
                    else "%s.srt" % seg["shot_id"])
        if not os.path.isfile(os.path.join(shot_dir, srt_name)):
            errors.append("%s %s missing %s" % (
                name, seg["shot_dir"], srt_name))
        if audio_present:
            if not os.path.isfile(os.path.join(shot_dir, "original.wav")):
                errors.append("%s %s missing original.wav" % (
                    name, seg["shot_dir"]))
            for wav in ("vocals.wav", "instrumental.wav"):
                if not os.path.isfile(os.path.join(detail_dir, wav)):
                    errors.append("%s %s missing detailed/%s" % (
                        name, seg["shot_dir"], wav))


def _check_archive(errors, name, source_dir, segments, min_sec, max_sec):
    archived = [seg for seg in segments
                if min_sec <= seg["end_sec"] - seg["start_sec"] <= max_sec]
    for speaker in {seg["speaker"] for seg in archived}:
        speaker_dir = os.path.join(source_dir, "speakers", speaker)
        wavs = [f for f in os.listdir(speaker_dir)
                if f.endswith(".wav")] if os.path.isdir(speaker_dir) else []
        if not wavs:
            errors.append("%s no archived wav for %s" % (name, speaker))
        global_json = os.path.join(speaker_dir, "%s_global.json" % speaker)
        if not os.path.isfile(global_json):
            errors.append("%s missing global json for %s" % (name, speaker))


def main():
    videos = common.input_videos()
    if not videos:
        print("no input videos found in %s"
              % os.path.join(common.ROOT, "input"))
        return 1

    cfg = load_config(os.path.join(common.ROOT, "config.json"))
    ready, missing = speech.available(cfg)
    if not ready:
        print("speech models unavailable: %s" % ", ".join(missing))
        return 1
    separation_ready, sep_missing = speech.separation_available(cfg)
    if not separation_ready:
        print("separation unavailable (audio tracks will be skipped): %s"
              % ", ".join(sep_missing))

    work_dir = os.path.join(tempfile.gettempdir(), "ref_forge_asr")
    input_dir, names = common.stage_input(work_dir, videos)
    output_dir = os.path.join(work_dir, "output")
    print("work dir: %s" % work_dir)
    print("videos: %s" % ", ".join(names))

    common.run_pipeline(input_dir, output_dir, ["--no-describe", "--no-persons"])

    min_sec = float(cfg["asr"].get("min_archive_sec", 1.0))
    max_sec = float(cfg["asr"].get("max_archive_sec", 30.0))

    errors = []
    center_path = os.path.join(output_dir, "voiceprint_center.json")
    center = common.read_json(center_path) if os.path.isfile(center_path) else {}
    if not os.path.isfile(center_path):
        errors.append("missing voiceprint_center.json")

    video_speakers = {}
    separation_seen = False
    for name in names:
        stem = os.path.splitext(name)[0]
        source_dir = os.path.join(output_dir, "segments", stem)
        srt_path = os.path.join(input_dir, stem + ".srt")
        transcript_path = os.path.join(source_dir, "transcript.json")

        if not os.path.isfile(srt_path):
            errors.append("%s missing SRT" % name)
        if not os.path.isfile(transcript_path):
            errors.append("%s missing transcript.json" % name)
            continue

        audio_dir = os.path.join(source_dir, "audio")
        audio_present = os.path.isdir(audio_dir)
        if audio_present:
            separation_seen = True
            for fname in ("original.wav", "vocals.wav", "instrumental.wav",
                          "separation.json"):
                if not os.path.isfile(os.path.join(audio_dir, fname)):
                    errors.append("%s missing audio/%s" % (name, fname))

        transcript = common.read_json(transcript_path)
        segments, speakers = _check_transcript(errors, name, transcript)
        video_speakers[name] = speakers

        manifest_path = os.path.join(source_dir, "segments.json")
        if not os.path.isfile(manifest_path):
            errors.append("%s missing segments.json" % name)
            continue
        manifest = common.read_json(manifest_path)
        _check_shots(errors, name, source_dir, manifest, audio_present)
        _check_archive(errors, name, source_dir, segments, min_sec, max_sec)

    audio_videos = [name for name in names
                    if common.has_audio(os.path.join(input_dir, name))]
    if separation_ready and audio_videos and not separation_seen:
        errors.append("no source produced separated audio tracks")

    print("\n%-38s %-10s %s" % ("video", "speakers", "shots"))
    for name in names:
        stem = os.path.splitext(name)[0]
        manifest_path = os.path.join(output_dir, "segments", stem,
                                     "segments.json")
        shot_count = len(common.read_json(manifest_path).get("segments", [])) \
            if os.path.isfile(manifest_path) else 0
        print("%-38s %-10s %d" % (
            stem[:38], ",".join(sorted(video_speakers.get(name, {"-"}))),
            shot_count))

    print("\nvoiceprint speakers: %s" % ", ".join(sorted(
        center.get("speakers", {}).keys())))
    if errors:
        print("\nASR test FAILED (%d):" % len(errors))
        for item in errors:
            print("  - %s" % item)
        return 1
    print("\nASR test passed")
    print("work dir: %s" % work_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
