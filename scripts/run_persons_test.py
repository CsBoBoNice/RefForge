"""人物提取与归类验收：对 `input/` 下每个真实视频核对人物产物结构与元信息。

用法：dev.bat scripts\\run_persons_test.py [--domain real|anime]

默认 `--domain anime`（与程序默认一致），需先执行 `scripts\\download_anime_person_models.py`
备好动漫模型；动漫模型缺失时人物阶段会被引擎检查关闭，测试将报缺少 `metadata.json`。
`--domain real` 用真人模型，无需动漫模型。

素材取自 `input/`（有什么视频就用什么视频），复制到系统临时目录后运行，不污染
`input/` / `output/`。只跑分割 + 故事图 + 人物提取（关闭 ASR / 分离 / 描述）。
校验：`persons/metadata.json` 结构、`person_XXX/` 与 `unknown/` 目录、图片命名与
帧号可溯源、图片为原帧矩形（原分辨率内）、去重与 `max_per_person` 截断、环境未变。
"""

import argparse
import importlib.metadata as metadata
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import acceptance_common as common  # noqa: E402

import cv2  # noqa: E402
from config import load_config  # noqa: E402

PERSON_RE = re.compile(r"^person_(\d{3})_f(\d{7})\.jpg$")
UNKNOWN_RE = re.compile(r"^unknown_f(\d{7})\.jpg$")
EXPECTED_TORCH = "2.9.1+cu128"
EXPECTED_NUMPY = "2.2.6"


def _check_environment(errors):
    try:
        version = metadata.version("opencv-python-headless")
    except metadata.PackageNotFoundError:
        errors.append("opencv-python-headless not installed")
        version = "?"
    try:
        metadata.version("opencv-python")
        errors.append("opencv-python must not be installed (conflicts with headless)")
    except metadata.PackageNotFoundError:
        pass
    if cv2.__version__ != "4.10.0":
        errors.append("unexpected cv2 version %s" % cv2.__version__)
    try:
        import torch
        if torch.__version__ != EXPECTED_TORCH:
            errors.append("torch changed: %s" % torch.__version__)
        import numpy
        if numpy.__version__ != EXPECTED_NUMPY:
            errors.append("numpy changed: %s" % numpy.__version__)
    except Exception as exc:
        errors.append("env import failed: %s" % exc)
    print("env: cv2-headless=%s torch=%s numpy=%s" % (
        version, EXPECTED_TORCH, EXPECTED_NUMPY))


def _check_person_images(errors, label, folder, files, meta_person, bounds,
                         min_width, min_height, max_per_person):
    if len(files) != int(meta_person.get("num_images", -1)):
        errors.append("%s num_images mismatch: %d vs %d" % (
            label, int(meta_person.get("num_images", -1)), len(files)))
    if not files:
        errors.append("%s has no images" % label)
    if len(files) > max_per_person:
        errors.append("%s exceeds max_per_person (%d > %d)" % (
            label, len(files), max_per_person))
    frame_limit = bounds[0]
    seen = set()
    for name in files:
        match = PERSON_RE.match(name)
        if not match:
            errors.append("%s bad filename %s" % (label, name))
            continue
        frame_index = int(match.group(2))
        if frame_index in seen:
            errors.append("%s duplicate frame %d" % (label, frame_index))
        seen.add(frame_index)
        if frame_limit and frame_index >= frame_limit:
            errors.append("%s frame %d out of range" % (label, frame_index))
        image = cv2.imread(os.path.join(folder, name))
        if image is None:
            errors.append("%s unreadable %s" % (label, name))
            continue
        height, width = image.shape[:2]
        if width < min_width or height < min_height:
            errors.append("%s crop too small %s (%dx%d)" % (
                label, name, width, height))
        if width > bounds[1] or height > bounds[2]:
            errors.append("%s crop exceeds source %s (%dx%d)" % (
                label, name, width, height))
    levels = meta_person.get("levels") or {}
    for key in ("full_body", "upper_body", "torso", "chest", "head", "unknown"):
        if key not in levels:
            errors.append("%s levels missing %s" % (label, key))
    representative = meta_person.get("representative")
    if representative not in files:
        errors.append("%s representative not in folder: %s" % (
            label, representative))


def _check_unknown(errors, persons_dir, unknown_meta, min_width, min_height):
    for track_dir, info in unknown_meta.items():
        folder = os.path.join(persons_dir, "unknown", track_dir)
        if not os.path.isdir(folder):
            errors.append("unknown %s folder missing" % track_dir)
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
        if len(files) != int(info.get("num_images", -1)):
            errors.append("unknown %s num_images mismatch" % track_dir)
        for name in files:
            if not UNKNOWN_RE.match(name):
                errors.append("unknown %s bad filename %s" % (track_dir, name))
                continue
            image = cv2.imread(os.path.join(folder, name))
            if image is None:
                errors.append("unknown %s unreadable %s" % (track_dir, name))
                continue
            height, width = image.shape[:2]
            if width < min_width or height < min_height:
                errors.append("unknown crop too small %s" % name)


def main(argv=None):
    parser = argparse.ArgumentParser(description="人物提取与归类验收")
    parser.add_argument("--domain", default="anime", choices=["real", "anime"],
                        help="人物模型领域（默认 anime，与程序默认一致）")
    args = parser.parse_args(argv)

    videos = common.input_videos()
    if not videos:
        print("no input videos found in %s" % os.path.join(common.ROOT, "input"))
        return 1

    work_dir = os.path.join(tempfile.gettempdir(),
                            "ref_forge_persons_%s" % args.domain)
    input_dir, names = common.stage_input(work_dir, videos)
    output_dir = os.path.join(work_dir, "output")
    print("work dir: %s" % work_dir)
    print("domain: %s" % args.domain)
    print("videos: %s" % ", ".join(names))

    common.run_pipeline(input_dir, output_dir,
                        ["--no-asr", "--no-separate", "--no-describe",
                         "--persons-domain", args.domain])

    cfg = load_config(os.path.join(common.ROOT, "config.json"))
    pc = cfg["persons"]
    max_per_person = int(pc["max_per_person"])
    min_width = int(pc["min_crop_width"])
    min_height = int(pc["min_crop_height"])

    errors = []
    print("\n%-30s %-8s %-9s %-8s %s" % (
        "source", "persons", "images", "unknown", "ok"))
    for name in names:
        stem = os.path.splitext(name)[0]
        source_dir = os.path.join(output_dir, "segments", stem)
        persons_dir = os.path.join(source_dir, "persons")
        manifest = os.path.join(persons_dir, "metadata.json")
        if not os.path.isfile(manifest):
            errors.append("%s missing persons/metadata.json" % name)
            print("%-30s %-8s %-9s %-8s %s" % (name[:30], "-", "-", "-", "N"))
            continue

        meta = common.read_json(manifest)
        if meta.get("schema_version") != "1.0":
            errors.append("%s metadata schema_version" % name)
        if meta.get("domain") != args.domain:
            errors.append("%s metadata domain mismatch: %s" % (
                name, meta.get("domain")))
        if (meta.get("source") or {}).get("file") != name:
            errors.append("%s metadata source.file mismatch" % name)

        source_path = os.path.join(input_dir, name)
        capture = cv2.VideoCapture(source_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int((meta.get("source") or {}).get("frame_count", 0))
        capture.release()
        bounds = (frame_count, width, height)

        persons = meta.get("persons") or {}
        total_images = 0
        for person_id, info in persons.items():
            folder = os.path.join(persons_dir, person_id)
            files = [f for f in os.listdir(folder)
                     if f.lower().endswith(".jpg")] if os.path.isdir(folder) else []
            total_images += len(files)
            _check_person_images(errors, "%s/%s" % (name, person_id), folder,
                                 files, info, bounds, min_width, min_height,
                                 max_per_person)

        unknown_meta = meta.get("unknown") or {}
        if meta.get("unknown_track_count") != len(unknown_meta):
            errors.append("%s unknown_track_count mismatch" % name)
        if unknown_meta:
            _check_unknown(errors, persons_dir, unknown_meta, min_width,
                           min_height)

        ok = not any(e.startswith(name) for e in errors)
        print("%-30s %-8d %-9d %-8d %s" % (
            name[:30], len(persons), total_images, len(unknown_meta),
            "Y" if ok else "N"))

    _check_environment(errors)

    if errors:
        print("\nerrors:")
        for message in errors:
            print("  -", message)
        return 1
    print("\npersons test passed")
    print("work dir: %s" % work_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
