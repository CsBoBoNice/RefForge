"""人声分离验收：对 `input/` 下每个含音轨的视频核对原始 / 人声 / 伴奏三轨与元数据。

用法：
    dev.bat scripts\\run_separation_test.py                 # 默认最佳档 + GPU
    dev.bat scripts\\run_separation_test.py --tier fast --device cpu

素材取自 `input/`（有什么视频就用什么视频），仅对含音轨的视频验收；视频复制到
系统临时目录后跑主程序（关闭 ASR / 描述），校验 `output/segments/<源名>/audio/`
下三轨齐全、时长与元数据一致。
"""

import argparse
import os
import sys
import tempfile
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import acceptance_common as common  # noqa: E402

from config import load_config  # noqa: E402
import speech  # noqa: E402


def _wav_seconds(path):
    with wave.open(path, "rb") as fh:
        return fh.getnframes() / float(fh.getframerate())


def main():
    parser = argparse.ArgumentParser(description="人声分离验收")
    parser.add_argument("--tier", default="best",
                        choices=["fast", "balanced", "best", "dereverb"])
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = parser.parse_args()

    videos = [path for path in common.input_videos()
              if common.has_audio(path) is not False]
    if not videos:
        print("no input video with an audio stream in %s"
              % os.path.join(common.ROOT, "input"))
        return 1

    cfg = load_config(os.path.join(common.ROOT, "config.json"))
    ready, missing = speech.separation_available(cfg)
    if not ready:
        print("separation models unavailable: %s" % ", ".join(missing))
        return 1

    work_dir = os.path.join(tempfile.gettempdir(),
                            "ref_forge_separation")
    input_dir, names = common.stage_input(work_dir, videos)
    output_dir = os.path.join(work_dir, "output")
    print("work dir: %s" % work_dir)
    print("videos: %s" % ", ".join(names))

    common.run_pipeline(input_dir, output_dir, [
        "--no-asr", "--no-describe", "--no-persons",
        "--separate-model", args.tier, "--separate-device", args.device])

    errors = []
    print("\n%-34s %-8s %-8s %-10s %s" % (
        "video", "orig(s)", "voc(s)", "inst?", "device"))
    for name in names:
        stem = os.path.splitext(name)[0]
        audio_dir = os.path.join(output_dir, "segments", stem, "audio")
        original = os.path.join(audio_dir, "original.wav")
        vocals = os.path.join(audio_dir, "vocals.wav")
        instrumental = os.path.join(audio_dir, "instrumental.wav")
        meta_path = os.path.join(audio_dir, "separation.json")
        for path in (original, vocals, instrumental, meta_path):
            if not os.path.isfile(path):
                errors.append("%s missing %s" % (name, path))
        if not os.path.isfile(vocals):
            print("%-34s %s" % (stem[:34], "FAILED"))
            continue
        meta = common.read_json(meta_path)
        if meta.get("tier") != args.tier:
            errors.append("%s tier mismatch %s" % (name, meta.get("tier")))
        if meta.get("device") != args.device:
            errors.append("%s device mismatch %s" % (name, meta.get("device")))
        for key in ("original", "vocals", "instrumental"):
            if not meta.get("tracks", {}).get(key):
                errors.append("%s meta missing track %s" % (name, key))
        orig_sec = _wav_seconds(original)
        voc_sec = _wav_seconds(vocals)
        if abs(orig_sec - voc_sec) > 0.5:
            errors.append("%s duration mismatch %.2f vs %.2f" % (
                name, orig_sec, voc_sec))
        print("%-34s %-8.2f %-8.2f %-10s %s" % (
            stem[:34], orig_sec, voc_sec,
            "Y" if os.path.isfile(instrumental) else "N",
            meta.get("device")))

    if errors:
        print("\nseparation test FAILED (%d):" % len(errors))
        for item in errors:
            print("  - %s" % item)
        return 1
    print("\nseparation test passed (tier=%s device=%s)" % (
        args.tier, args.device))
    print("work dir: %s" % work_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
