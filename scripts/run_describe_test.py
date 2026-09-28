"""视频描述验收：对 `input/` 下每个真实视频核对抽帧与中文 prompt 组装。

用法：
    dev.bat scripts\\run_describe_test.py          # 只做抽帧 + prompt 组装（快，不加载大模型）
    dev.bat scripts\\run_describe_test.py --llm    # 额外对第一个视频跑一次端到端（加载 llm_model.gguf）

素材取自 `input/`（有什么视频就用什么视频）。默认阶段不联网、不加载模型；
`--llm` 时启动 llama-server 对第一个视频的一个片段做完整中文描述。
输出写入系统临时目录，不污染 `input/` / `output/`。
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import acceptance_common as common  # noqa: E402

from config import load_config  # noqa: E402
from describe import frames as frames_mod  # noqa: E402
from describe import prompt as prompt_mod  # noqa: E402
import describe  # noqa: E402

CJK = re.compile("[\u4e00-\u9fff]")
BAD_LABEL = re.compile(r"<(?:Subject|Video) \d")


def _check_frames(errors, cfg, video, work_dir):
    desc = cfg["describe"]
    out_dir = os.path.join(work_dir, "frames")
    started = time.monotonic()
    info = frames_mod.extract_frames(
        video, out_dir,
        fps=float(desc["frame_fps"]),
        max_side=int(desc["frame_max_side"]),
        image_format=str(desc["frame_format"]),
        select_keyframes=bool(desc.get("frame_select", True)),
        keyframe_candidates=int(desc.get("frame_candidates", 3)))
    probe = frames_mod.probe_video(video)
    if probe is None:
        errors.append("%s ffprobe failed" % os.path.basename(video))
        return None
    src_w, src_h, src_dur = probe
    expected = src_dur * float(desc["frame_fps"])
    if abs(info["count"] - expected) > 2:
        errors.append("%s frame count %d not close to %.1f" % (
            os.path.basename(video), info["count"], expected))
    for frame in info["frames"]:
        if not frame["name"].endswith(".png"):
            errors.append("%s non-png frame %s" % (
                os.path.basename(video), frame["name"]))
        path = os.path.join(out_dir, frame["name"])
        if not os.path.isfile(path):
            errors.append("%s missing frame file %s" % (
                os.path.basename(video), path))
    if desc.get("frame_select", True) and not info.get("selected"):
        errors.append("%s keyframe selection did not run"
                      % os.path.basename(video))
    if os.path.isdir(os.path.join(out_dir, "_candidates")):
        errors.append("%s candidate temp dir not cleaned"
                      % os.path.basename(video))
    long_side = max(info["width"], info["height"])
    short_side = min(info["width"], info["height"])
    if long_side > 1920 or short_side > 1080:
        errors.append("%s frames exceed 1080P: %dx%d" % (
            os.path.basename(video), info["width"], info["height"]))
    print("frames [%s]: %d, %dx%d (source %dx%d, %.2fs, select=%s) in %.1fs"
          % (os.path.basename(video), info["count"], info["width"],
             info["height"], src_w, src_h, src_dur, info.get("selected"),
             time.monotonic() - started))
    return info


def _check_prompt(errors, cfg, video, info):
    references = prompt_mod.reference_mapping_text()
    for token in ("<Picture 1>", "<Picture 2>", "<Audio 1>", "故事图"):
        if token not in references:
            errors.append("reference mapping missing %s" % token)
    if BAD_LABEL.search(references):
        errors.append("reference mapping must not use <Subject N> / <Video N>")
    subtitles = [{"start_sec": 0.2, "end_sec": 1.4, "speaker": "speaker_1",
                  "text": "hello"}]
    for mode in ("storyboard", "frames"):
        system = prompt_mod.system_prompt(mode)
        for heading in prompt_mod.SECTION_HEADINGS:
            if heading not in system:
                errors.append("[%s] system prompt missing %s" % (mode, heading))
        if BAD_LABEL.search(system):
            errors.append("[%s] system prompt must not use <Subject N> / <Video N>"
                          % mode)
        if not CJK.search(system):
            errors.append("[%s] system prompt is not in Chinese" % mode)
        if prompt_mod.missing_sections(system):
            errors.append("[%s] system prompt headings not detected" % mode)
        user = prompt_mod.build_user_prompt(
            mode, info["duration_sec"], info["fps"], info["frames"], subtitles,
            ["speaker_1"])
        for token in ("片段时长", "speaker_1: hello", "<Picture 1>"):
            if token not in user:
                errors.append("[%s] user prompt missing %r" % (mode, token))
        if mode == "frames":
            if info["frames"][0]["name"] not in user:
                errors.append("[frames] user prompt missing frame timeline")
            if "%s fps" % int(info["fps"]) not in user:
                errors.append("[frames] user prompt missing fps")
        if mode == "storyboard" and "故事图" not in user:
            errors.append("[storyboard] user prompt missing storyboard note")
        if BAD_LABEL.search(user):
            errors.append("[%s] user prompt must not use <Subject N> / <Video N>"
                          % mode)
        if not CJK.search(user):
            errors.append("[%s] user prompt is not in Chinese" % mode)
        print("prompt [%s][%s]: system=%d chars, user=%d chars"
              % (os.path.basename(video), mode, len(system), len(user)))
    cleaned = prompt_mod.clean_output(
        "```\nsubject_definitions:\n<Picture 1> x\n```")
    if cleaned.startswith("```") or not cleaned.startswith("subject_definitions:"):
        errors.append("clean_output did not strip fences")


def _check_llm(errors, cfg, video, work_dir):
    ready, missing = describe.available(cfg)
    if not ready:
        print("llm stage skipped; missing: %s" % ", ".join(missing))
        return
    input_dir = os.path.join(work_dir, "llm_input")
    output_dir = os.path.join(work_dir, "llm_output")
    os.makedirs(input_dir, exist_ok=True)
    local = os.path.join(input_dir, os.path.basename(video))
    shutil.copy2(video, local)
    subprocess.run(
        [sys.executable, os.path.join(common.APP_DIR, "main.py"),
         "--single", local, "--output", output_dir, "--overwrite",
         "--no-asr", "--no-separate", "--no-persons",
         "--describe-limit", "1"],
        check=True)
    stem = os.path.splitext(os.path.basename(video))[0]
    manifest = os.path.join(output_dir, "segments", stem, "segments.json")
    if not os.path.isfile(manifest):
        errors.append("llm stage missing segments.json")
        return
    with open(manifest, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not data.get("segments"):
        errors.append("llm stage no segments")
        return
    shot_dir = os.path.join(output_dir, "segments", stem,
                            data["segments"][0]["shot_dir"])
    detail_dir = os.path.join(shot_dir, "detailed")
    mode = str(cfg["describe"].get("input_mode") or "storyboard").lower()
    if mode not in ("storyboard", "frames"):
        mode = "storyboard"
    if os.path.isfile(os.path.join(shot_dir, "video_prompt_en.txt")):
        errors.append("llm stage should not write video_prompt_en.txt")
    if not os.path.isfile(os.path.join(shot_dir, "video_prompt_zh.txt")):
        errors.append("llm stage missing video_prompt_zh.txt")
    payload_path = os.path.join(detail_dir, "video_prompt.json")
    payload = {}
    if not os.path.isfile(payload_path):
        errors.append("llm stage missing detailed/video_prompt.json")
    else:
        with open(payload_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("input_mode") != mode:
            errors.append("llm stage input_mode %r != %r"
                          % (payload.get("input_mode"), mode))
    if mode == "storyboard":
        if not os.path.isfile(os.path.join(shot_dir, "scene_reference.jpg")):
            errors.append("llm stage missing scene_reference.jpg")
        sb_rel = payload.get("storyboard")
        if sb_rel and not os.path.isfile(os.path.join(shot_dir, sb_rel)):
            errors.append("llm stage missing storyboard image %s" % sb_rel)
        if payload.get("storyboard_cached") and \
                not os.path.isfile(os.path.join(
                    detail_dir, "storyboard_1080p.jpg")):
            errors.append("llm stage missing detailed/storyboard_1080p.jpg")
    elif not os.path.isfile(os.path.join(shot_dir, "frames",
                                         "frame_00001.png")):
        errors.append("llm stage missing frames/frame_00001.png")
    zh_path = os.path.join(shot_dir, "video_prompt_zh.txt")
    if os.path.isfile(zh_path):
        with open(zh_path, "r", encoding="utf-8") as fh:
            zh_text = fh.read()
        missing_sections = prompt_mod.missing_sections(zh_text)
        if missing_sections:
            errors.append("Chinese prompt missing %s"
                          % ",".join(missing_sections))
        if not CJK.search(zh_text):
            errors.append("Chinese prompt has no Chinese content")
        print("llm prompt [%s]: %d chars; output=%s"
              % (mode, len(zh_text), shot_dir))


def main():
    parser = argparse.ArgumentParser(description="视频描述验收")
    parser.add_argument("--llm", action="store_true",
                        help="额外对第一个视频运行一次端到端大模型描述（较慢）")
    args = parser.parse_args()

    videos = common.input_videos()
    if not videos:
        print("no input videos found in %s"
              % os.path.join(common.ROOT, "input"))
        return 1

    work_dir = os.path.join(tempfile.gettempdir(), "ref_forge_describe")
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)
    cfg = load_config(os.path.join(common.ROOT, "config.json"))

    errors = []
    print("videos: %s" % ", ".join(os.path.basename(v) for v in videos))
    for index, video in enumerate(videos):
        sub_dir = os.path.join(work_dir, "video_%02d" % index)
        os.makedirs(sub_dir, exist_ok=True)
        info = _check_frames(errors, cfg, video, sub_dir)
        if info:
            _check_prompt(errors, cfg, video, info)
    if args.llm:
        _check_llm(errors, cfg, videos[0], work_dir)

    if errors:
        print("\ndescribe test FAILED (%d):" % len(errors))
        for item in errors:
            print("  - %s" % item)
        return 1
    print("\ndescribe test passed (%d video(s))" % len(videos))
    print("work dir: %s" % work_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
