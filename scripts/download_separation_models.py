"""下载人声分离（双轨）所需的全部模型到 models/separator/。

开发机联网时执行一次；脚本会比对多个下载源（国内 GitHub 代理优先），
逐个模型按可用源尝试下载，成功后后续运行完全离线。

用法：
    dev.bat scripts\\download_separation_models.py              # 下载缺失模型
    dev.bat scripts\\download_separation_models.py --force       # 强制重新下载
    dev.bat scripts\\download_separation_models.py --only best   # 只下载某档

档位与模型（与 config.json 的 separation 保持一致）：
    fast      UVR-MDX-NET-Voc_FT.onnx                        (MDX-Net)
    balanced  model_bs_roformer_ep_317_sdr_12.9755.ckpt      (BS-Roformer)
    best      model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt (Mel-Band Roformer)
    dereverb  Reverb_HQ_By_FoxJoy.onnx                       (MDX-Net 去混响)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_utils import download_smart  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models", "separator")

UVR_BASE = ("https://github.com/TRvlvr/model_repo/releases/download/"
            "all_public_uvr_models/")
CFG_BASE = ("https://github.com/nomadkaraoke/python-audio-separator/releases/"
            "download/model-configs/")

# 国内 GitHub 加速源（按实测网速从快到慢）；"" 表示直连 GitHub。
MIRRORS = [
    "https://ghproxy.net/",
    "https://gh-proxy.com/",
    "https://mirror.ghproxy.com/",
    "",
]

# name, 期望字节数, 所在仓库基址, 档位
CATALOG = [
    ("UVR-MDX-NET-Voc_FT.onnx", 66_800_000, UVR_BASE, "fast"),
    ("model_bs_roformer_ep_317_sdr_12.9755.ckpt", 639_300_000, UVR_BASE,
     "balanced"),
    ("model_bs_roformer_ep_317_sdr_12.9755.yaml", 2_273, CFG_BASE,
     "balanced"),
    ("model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt", 1_007_800_000,
     UVR_BASE, "best"),
    ("model_mel_band_roformer_ep_3005_sdr_11.4360.yaml", 1_794, CFG_BASE,
     "best"),
    ("Reverb_HQ_By_FoxJoy.onnx", 66_800_000, UVR_BASE, "dereverb"),
]

# audio-separator 运行期还会读取的 UVR 元数据文件；预置后可完全离线加载。
APP_DATA = "https://raw.githubusercontent.com/TRvlvr/application_data/main/"
AUX_FILES = [
    ("download_checks.json", APP_DATA + "filelists/download_checks.json"),
    ("vr_model_data.json", APP_DATA + "vr_model_data/model_data_new.json"),
    ("mdx_model_data.json", APP_DATA + "mdx_model_data/model_data_new.json"),
]


def _present(path, expected):
    if not os.path.isfile(path):
        return False
    size = os.path.getsize(path)
    if expected >= 1_000_000:
        return size >= int(expected * 0.98)
    return size > 0


def download_one(name, expected, base, force=False):
    dest = os.path.join(MODELS_DIR, name)
    if not force and _present(dest, expected):
        print("[skip] %s" % name)
        return True
    os.makedirs(MODELS_DIR, exist_ok=True)
    urls = [prefix + base + name for prefix in MIRRORS]
    return download_smart(urls, dest, expected, group="github")


def download_aux(force=False):
    print("== UVR 元数据（离线加载所需） ==")
    ok = True
    for name, url in AUX_FILES:
        dest = os.path.join(MODELS_DIR, name)
        if not force and os.path.isfile(dest) and os.path.getsize(dest) > 0:
            print("[skip] %s" % name)
            continue
        os.makedirs(MODELS_DIR, exist_ok=True)
        urls = [prefix + url for prefix in MIRRORS]
        if not download_smart(urls, dest, 0, group="github"):
            ok = False
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description="下载人声分离模型")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument("--only", default=None,
                        help="只下载指定档位：fast/balanced/best/dereverb")
    args = parser.parse_args(argv)

    selected = CATALOG
    if args.only:
        selected = [item for item in CATALOG if item[3] == args.only]
        if not selected:
            print("unknown tier: %s" % args.only)
            return 2

    failures = []
    for name, expected, base, _tier in selected:
        if not download_one(name, expected, base, force=args.force):
            failures.append(name)
    if not download_aux(force=args.force):
        failures.append("uvr-metadata")
    if failures:
        print("failed: %s" % ", ".join(failures))
        return 1
    print("all separation models ready under %s" % MODELS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
