"""镜头分割验收：对 `input/` 下每个真实视频跑分割 + 3x3 故事图，校验时长约束与结构。

用法：dev.bat scripts\\run_segmentation_test.py

素材取自 `input/`（有什么视频就用什么视频），复制到系统临时目录后运行，
不污染 `input/` / `output/`。校验每段时长落在 [min_segment_sec, max_segment_sec]
（低于下限的 below_min 段除外）、`scene_reference.jpg` / `detailed/` 结构完整。
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import acceptance_common as common  # noqa: E402

import cv2  # noqa: E402
from config import load_config  # noqa: E402


def _check_storyboard(errors, name, path):
    image = cv2.imread(path)
    if image is None:
        errors.append("%s invalid scene_reference.jpg" % name)
        return None
    return image.shape[1], image.shape[0]


def main():
    videos = common.input_videos()
    if not videos:
        print("no input videos found in %s"
              % os.path.join(common.ROOT, "input"))
        return 1

    work_dir = os.path.join(tempfile.gettempdir(), "ref_forge_segtest")
    input_dir, names = common.stage_input(work_dir, videos)
    output_dir = os.path.join(work_dir, "output")
    print("work dir: %s" % work_dir)
    print("videos: %s" % ", ".join(names))

    common.run_pipeline(input_dir, output_dir,
                        ["--no-asr", "--no-separate", "--no-describe",
                         "--no-persons"])

    cfg = load_config(os.path.join(common.ROOT, "config.json"))
    ref = cfg.get("reference", {})
    grid_cols = int(ref.get("grid_cols", 3))
    grid_rows = int(ref.get("grid_rows", 3))

    errors = []
    print("\n%-30s %-7s %-8s %-16s %-13s %s" % (
        "source", "shots", "segments", "duration range", "storyboard", "ok"))
    for name in names:
        stem = os.path.splitext(name)[0]
        manifest = os.path.join(output_dir, "segments", stem, "segments.json")
        if not os.path.isfile(manifest):
            errors.append("%s missing segments.json" % name)
            continue
        data = common.read_json(manifest)
        segs = data.get("segments", [])
        durations = [s["duration_sec"] for s in segs]
        min_sec = data["min_segment_sec"]
        max_sec = data["max_segment_sec"]
        ok = True
        board_desc = "-"
        for seg in segs:
            dur = seg["duration_sec"]
            if not seg.get("below_min") and (
                    dur < min_sec - 1e-6 or dur > max_sec + 1e-6):
                errors.append("%s segment %d duration %.2fs out of [%s,%s]" % (
                    name, seg["index"], dur, min_sec, max_sec))
                ok = False
            shot_dir = os.path.join(output_dir, "segments", stem,
                                    seg["shot_dir"])
            if not os.path.isdir(shot_dir):
                errors.append("%s missing shot dir %s" % (
                    name, seg["shot_dir"]))
                ok = False
                continue
            board = os.path.join(shot_dir, "scene_reference.jpg")
            if not os.path.isfile(board):
                errors.append("%s %s missing scene_reference.jpg" % (
                    name, seg["shot_dir"]))
                ok = False
            else:
                size = _check_storyboard(errors, name, board)
                if size and board_desc == "-":
                    board_desc = "%dx%d" % size
            if not os.path.isdir(os.path.join(shot_dir, "detailed")):
                errors.append("%s %s missing detailed/" % (
                    name, seg["shot_dir"]))
                ok = False
            if seg.get("file") and not os.path.isfile(
                    os.path.join(shot_dir, seg["file"])):
                errors.append("%s %s missing segment video" % (
                    name, seg["file"]))
                ok = False

        rng = ("%.2f~%.2f" % (min(durations), max(durations))
               if durations else "-")
        print("%-30s %-7d %-8d %-16s %-13s %s" % (
            name[:30], data.get("detected_shot_count", 0), len(segs), rng,
            board_desc, "Y" if ok else "N"))

    if errors:
        print("\nerrors:")
        for message in errors:
            print("  -", message)
        return 1
    print("\nsegmentation test passed (grid=%dx%d)"
          % (grid_cols, grid_rows))
    print("work dir: %s" % work_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
