"""下载动漫人物提取所需的模型到 models/person/anime/。

- 动漫人物检测：`deepghs/anime_person_detection` 的 `person_detect_v1.1_m/model.onnx`
  （YOLO ONNX，F1≈0.87，~99MB）；用 `onnxruntime` 直驱；
- 动漫人脸检测：`deepghs/anime_face_detection` 的 `face_detect_v1.4_s/model.onnx`
  （YOLO ONNX，F1≈0.95，~43MB）；用 `onnxruntime` 直驱；
- CLIP 图像嵌入：`openai/clip-vit-large-patch14`（~1.6GB），用项目已装的
  `transformers` 加载。

全部经 HuggingFace 镜像 hf-mirror.com 下载，运行期完全离线；不新增第三方依赖。

用法：
    dev.bat scripts\\download_anime_person_models.py            # 下载缺失模型
    dev.bat scripts\\download_anime_person_models.py --force     # 强制重新下载
    dev.bat scripts\\download_anime_person_models.py --clip-only # 仅下载 CLIP
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_utils import pick_fastest  # noqa: E402

ANIME_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "person", "anime")
PERSON_DIR = os.path.join(ANIME_DIR, "person")
FACE_DIR = os.path.join(ANIME_DIR, "face")
CLIP_DIR = os.path.join(ANIME_DIR, "clip", "clip-vit-large-patch14")

ENDPOINT = "https://hf-mirror.com"
# 候选 HuggingFace 端点：先按实测网速挑选最快者，再整体从该端点下载。
ENDPOINTS = ["https://hf-mirror.com", "https://huggingface.co"]
_PROBE_REPO = "openai/clip-vit-large-patch14"
_PROBE_FILE = "model.safetensors"


def _pick_endpoint():
    probes = ["%s/%s/resolve/main/%s" % (endpoint, _PROBE_REPO, _PROBE_FILE)
              for endpoint in ENDPOINTS]
    best = pick_fastest(probes, group="hf")
    for endpoint in ENDPOINTS:
        if best == "%s/%s/resolve/main/%s" % (endpoint, _PROBE_REPO, _PROBE_FILE):
            return endpoint
    return ENDPOINT


PERSON_REPO = "deepghs/anime_person_detection"
PERSON_SUBDIR = "person_detect_v1.1_m"
FACE_REPO = "deepghs/anime_face_detection"
FACE_SUBDIR = "face_detect_v1.4_s"

CLIP_REPO = "openai/clip-vit-large-patch14"
CLIP_ALLOW = ["*.json", "*.txt", "*.safetensors"]

MODEL_FILES = ("model.onnx", "labels.json", "threshold.json")


def _present(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


_ENDPOINT_CACHE = None


def _snapshot(repo, local_dir, patterns):
    global _ENDPOINT_CACHE
    if _ENDPOINT_CACHE is None:
        _ENDPOINT_CACHE = _pick_endpoint()
    endpoint = _ENDPOINT_CACHE
    os.environ["HF_ENDPOINT"] = endpoint
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        print("[fail] huggingface_hub unavailable: %s" % exc)
        return False
    print("[try ] %s @ %s" % (repo, endpoint))
    try:
        snapshot_download(repo_id=repo, local_dir=local_dir,
                          allow_patterns=patterns, endpoint=endpoint)
    except Exception as exc:
        print("[fail] %s: %s" % (repo, exc))
        return False
    return True


def download_person(force=False):
    sub = os.path.join(PERSON_DIR, PERSON_SUBDIR)
    if not force and all(_present(os.path.join(sub, name))
                         for name in MODEL_FILES):
        print("[skip] %s" % PERSON_SUBDIR)
        return True
    patterns = ["%s/%s" % (PERSON_SUBDIR, name) for name in
                MODEL_FILES + ("model_artifacts.json",)]
    if not _snapshot(PERSON_REPO, PERSON_DIR, patterns):
        return False
    if not _present(os.path.join(sub, "model.onnx")):
        print("[fail] %s missing after download" % PERSON_SUBDIR)
        return False
    return True


def download_face(force=False):
    sub = os.path.join(FACE_DIR, FACE_SUBDIR)
    if not force and all(_present(os.path.join(sub, name))
                         for name in MODEL_FILES):
        print("[skip] %s" % FACE_SUBDIR)
        return True
    patterns = ["%s/%s" % (FACE_SUBDIR, name) for name in
                MODEL_FILES + ("model_artifacts.json",)]
    if not _snapshot(FACE_REPO, FACE_DIR, patterns):
        return False
    if not _present(os.path.join(sub, "model.onnx")):
        print("[fail] %s missing after download" % FACE_SUBDIR)
        return False
    return True


def _clip_present():
    if not os.path.isfile(os.path.join(CLIP_DIR, "config.json")):
        return False
    for name in ("model.safetensors", "pytorch_model.bin"):
        if os.path.isfile(os.path.join(CLIP_DIR, name)):
            return True
    return False


def download_clip(force=False):
    if not force and _clip_present():
        print("[skip] %s" % CLIP_REPO)
        return True
    if not _snapshot(CLIP_REPO, CLIP_DIR, CLIP_ALLOW):
        return False
    if not _clip_present():
        print("[fail] %s missing weights after download" % CLIP_REPO)
        return False
    print("[ok  ] %s -> %s" % (CLIP_REPO, CLIP_DIR))
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="下载动漫人物提取模型")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument("--clip-only", action="store_true", help="仅下载 CLIP")
    parser.add_argument("--detector-only", action="store_true",
                        help="仅下载动漫人物/人脸检测模型")
    args = parser.parse_args(argv)

    failures = []
    if not args.clip_only:
        if not download_person(force=args.force):
            failures.append(PERSON_SUBDIR)
        if not download_face(force=args.force):
            failures.append(FACE_SUBDIR)
    if not args.detector_only and not download_clip(force=args.force):
        failures.append(CLIP_REPO)
    if failures:
        print("failed: %s" % ", ".join(failures))
        return 1
    print("all anime person models ready under %s" % ANIME_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
