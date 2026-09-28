"""下载人物提取与归类所需的模型到 models/person/。

- YOLO11x-pose：`detect/yolo11x-pose.pt`（约 113MB）；
- buffalo_l：解压其中的 `det_10g.onnx`（SCRFD）与 `w600k_r50.onnx`（ArcFace）
  到 `face/buffalo_l/`（约 190MB）。

来源优先级：GitHub Release（ultralytics / insightface）→ 中国 HuggingFace 镜像
（hf-mirror）兜底；下载后校验文件存在与大小，运行期完全离线。

用法：
    dev.bat scripts\\download_person_models.py            # 下载缺失模型
    dev.bat scripts\\download_person_models.py --force     # 强制重新下载
"""

import argparse
import os
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_utils import download_smart  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSON_DIR = os.path.join(ROOT, "models", "person")
DETECT_DIR = os.path.join(PERSON_DIR, "detect")
FACE_DIR = os.path.join(PERSON_DIR, "face", "buffalo_l")

YOLO_NAME = "yolo11x-pose.pt"
YOLO_SIZE = 118_481_010
YOLO_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
    "yolo11x-pose.pt",
    "https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/yolo11x-pose.pt",
]

BUFFALO_ZIP_SIZE = 288_621_354
BUFFALO_URLS = [
    "https://github.com/deepinsight/insightface/releases/download/v0.7/"
    "buffalo_l.zip",
    "https://hf-mirror.com/public-data/insightface/resolve/main/models/"
    "buffalo_l.zip",
]
BUFFALO_FILES = ("det_10g.onnx", "w600k_r50.onnx")

FACE_FILES = {"det_10g.onnx": 16_923_827, "w600k_r50.onnx": 174_383_860}


def _present(path, expected):
    if not os.path.isfile(path):
        return False
    size = os.path.getsize(path)
    if expected >= 1_000_000:
        return size >= int(expected * 0.98)
    return size > 0


def _download_first(urls, dest, expected):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    return download_smart(urls, dest, expected, group="person")


def download_yolo(force=False):
    dest = os.path.join(DETECT_DIR, YOLO_NAME)
    if not force and _present(dest, YOLO_SIZE):
        print("[skip] %s" % YOLO_NAME)
        return True
    return _download_first(YOLO_URLS, dest, YOLO_SIZE)


def download_buffalo(force=False):
    if not force and all(
            _present(os.path.join(FACE_DIR, name), FACE_FILES[name])
            for name in BUFFALO_FILES):
        print("[skip] buffalo_l")
        return True
    os.makedirs(FACE_DIR, exist_ok=True)
    archive = os.path.join(tempfile.gettempdir(), "buffalo_l.zip")
    if force or not _present(archive, BUFFALO_ZIP_SIZE):
        if not _download_first(BUFFALO_URLS, archive, BUFFALO_ZIP_SIZE):
            return False
    print("[unzip] buffalo_l.zip")
    try:
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            for name in BUFFALO_FILES:
                member = next(
                    (n for n in names if os.path.basename(n) == name), None)
                if member is None:
                    print("[fail] %s not in buffalo_l.zip" % name)
                    return False
                with zf.open(member) as source, \
                        open(os.path.join(FACE_DIR, name), "wb") as target:
                    shutil.copyfileobj(source, target)
    except (zipfile.BadZipFile, OSError) as exc:
        print("[fail] unzip: %s" % exc)
        return False
    os.remove(archive)
    print("[ok  ] buffalo_l -> %s" % FACE_DIR)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="下载人物提取模型")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    args = parser.parse_args(argv)
    failures = []
    if not download_yolo(force=args.force):
        failures.append(YOLO_NAME)
    if not download_buffalo(force=args.force):
        failures.append("buffalo_l")
    if failures:
        print("failed: %s" % ", ".join(failures))
        return 1
    print("all person models ready under %s" % PERSON_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
