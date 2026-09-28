"""下载运行期资产（ffmpeg / llama.cpp / 多模态 GGUF）到包内目录。

由 `一键环境搭建.bat`（经 `scripts/setup_env.py`）调用，也可单独运行：

    dev.bat scripts\\download_runtime.py                 # 下载缺失资产
    dev.bat scripts\\download_runtime.py --force         # 强制重新下载
    dev.bat scripts\\download_runtime.py --only ffmpeg   # 只下 ffmpeg
    dev.bat scripts\\download_runtime.py --only llama --only llm

目标目录：
    bin/                    ffmpeg.exe / ffprobe.exe / ffplay.exe（静态 GPL，含 libx264）
                            BtbN zip / gyan.dev zip / npmmirror tar.xz 三源选最快
    llama_cpp/llama_bin/    llama-server.exe + DLL（b10991，CPU + CUDA 12.4 合并）
    models/llm/             llm_model.gguf / mmproj_model.gguf
                            （ModelScope `unsloth/Qwen3.5-4B-GGUF`，可自动切换 HF 镜像）

多源下载：所有资产先按实测网速挑选最快源，再下载；失败自动切换下一个源。
仅用标准库，不新增依赖。
"""

import argparse
import json
import os
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_utils import download_smart, size_ok  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_DIR = os.path.join(ROOT, "bin")
LLAMA_BIN_DIR = os.path.join(ROOT, "llama_cpp", "llama_bin")
LLM_DIR = os.path.join(ROOT, "models", "llm")

TMP_DIR = os.path.join(os.environ.get("TEMP") or ROOT, "ref_forge_runtime")

FFMPEG_EXES = ("ffmpeg.exe", "ffprobe.exe", "ffplay.exe")
# 官方 zip 候选（BtbN 的 `releases/latest/download/` 与 `releases/download/latest/`
# 等价，均 302 到 latest 资产的 zip）。
FFMPEG_URLS = [
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-master-latest-win64-gpl.zip",
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
]
# npmmirror 镜像同类静态 Windows 构建：国内高速（实测 >15MB/s），但资产为 tar.xz
# 且**无 `latest/` 别名**，需先从索引 JSON 解析最新版本目录再拼 win32-x64-gpl 资产名。
FFMPEG_NPM_INDEX = "https://registry.npmmirror.com/-/binary/ffmpeg-builds/"

LLAMA_BUILD = "b10991"
# llama-server.exe 只是极小的启动器，真正的实现与后端在同目录 DLL 中。
LLAMA_MARKERS = ("llama-server.exe", "llama-server-impl.dll", "ggml-base.dll")
_LLAMA_BASE = "https://github.com/ggml-org/llama.cpp/releases/download/%s/" % LLAMA_BUILD
# 资产名 -> (文件名, 期望字节数)；CPU 版提供各档 ggml-cpu-*.dll，
# CUDA 版提供 ggml-cuda.dll（含静态 CUDA 运行时）。
LLAMA_ASSETS = {
    "cpu": ("llama-%s-bin-win-cpu-x64.zip" % LLAMA_BUILD, 18_428_726),
    "cuda": ("llama-%s-bin-win-cuda-12.4-x64.zip" % LLAMA_BUILD, 254_197_034),
}

# 国内 GitHub 加速源（"" 表示直连 GitHub）。
GITHUB_MIRRORS = [
    "https://ghproxy.net/",
    "https://gh-proxy.com/",
    "https://mirror.ghproxy.com/",
    "",
]

LLM_REPO = "unsloth/Qwen3.5-4B-GGUF"
LLM_FILES = [
    ("Qwen3.5-4B-UD-Q4_K_XL.gguf", "llm_model.gguf", 2_912_109_728),
    ("mmproj-F16.gguf", "mmproj_model.gguf", 672_423_616),
]


def _llm_urls(remote):
    """同一文件的候选源：ModelScope（resolve / API）+ HF 镜像 + HF 官方。"""
    return [
        "https://modelscope.cn/models/%s/resolve/master/%s" % (LLM_REPO, remote),
        "https://www.modelscope.cn/api/v1/models/%s/repo?Revision=master"
        "&FilePath=%s" % (LLM_REPO, remote),
        "https://hf-mirror.com/%s/resolve/main/%s" % (LLM_REPO, remote),
        "https://huggingface.co/%s/resolve/main/%s" % (LLM_REPO, remote),
    ]


def _extract_zip_flat(archive_path, dest_dir, wanted):
    hits = set()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = os.path.basename(info.filename)
            if not name or (wanted is not None and name.lower() not in wanted):
                continue
            with archive.open(info) as source, \
                    open(os.path.join(dest_dir, name), "wb") as target:
                shutil.copyfileobj(source, target)
            hits.add(name.lower())
    return hits


def _extract_tar_flat(archive_path, dest_dir, wanted):
    hits = set()
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            if not name or (wanted is not None and name.lower() not in wanted):
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, open(os.path.join(dest_dir, name), "wb") as target:
                shutil.copyfileobj(source, target)
            hits.add(name.lower())
    return hits


def _extract_flat(archive_path, dest_dir, want=None):
    """把 zip / tar.xz 内文件平铺解压到 dest_dir（去掉子目录）；want=None 表示全部。"""
    os.makedirs(dest_dir, exist_ok=True)
    wanted = {name.lower() for name in want} if want else None
    try:
        is_tar = tarfile.is_tarfile(archive_path)
    except OSError:
        is_tar = False
    if is_tar:
        return _extract_tar_flat(archive_path, dest_dir, wanted)
    return _extract_zip_flat(archive_path, dest_dir, wanted)


def _npmmirror_ffmpeg_url():
    """解析 npmmirror 上最新的 Windows x64 GPL 静态 ffmpeg tar.xz；失败返回 None。"""
    def fetch(url):
        request = urllib.request.Request(url, headers={"User-Agent": "RefForge-setup"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        entries = fetch(FFMPEG_NPM_INDEX)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    versions = []
    for item in entries:
        match = re.match(r"^v(\d+)\.(\d+)(?:\.(\d+))?/$", item.get("name", ""))
        if match:
            versions.append((tuple(int(x or 0) for x in match.groups()),
                             item["name"].rstrip("/")))
    if not versions:
        return None
    tag = max(versions)[1]
    asset = "ffmpeg-%s-win32-x64-gpl.tar.xz" % tag.lstrip("v")
    try:
        names = {item.get("name") for item in fetch(FFMPEG_NPM_INDEX + tag + "/")}
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if asset not in names:
        return None
    return FFMPEG_NPM_INDEX + tag + "/" + asset


def _ffmpeg_urls():
    """ffmpeg 候选源：npmmirror（若可解析）在前，其后 BtbN 与 gyan.dev。"""
    urls = []
    npmmirror = _npmmirror_ffmpeg_url()
    if npmmirror:
        urls.append(npmmirror)
    urls.extend(FFMPEG_URLS)
    return urls


def ensure_ffmpeg(force=False):
    print("== ffmpeg ==")
    if not force and all(size_ok(os.path.join(BIN_DIR, name), 1)
                         for name in FFMPEG_EXES):
        print("[skip] ffmpeg (%s)" % ", ".join(FFMPEG_EXES))
        return True
    os.makedirs(TMP_DIR, exist_ok=True)
    archive = os.path.join(TMP_DIR, "ffmpeg.zip")
    if force or not size_ok(archive, 1):
        if not download_smart(_ffmpeg_urls(), archive, group="ffmpeg"):
            return False
    print("[unzip] ffmpeg -> %s" % BIN_DIR)
    hits = _extract_flat(archive, BIN_DIR, FFMPEG_EXES)
    os.remove(archive)
    missing = [name for name in FFMPEG_EXES if name.lower() not in hits]
    if missing:
        print("[fail] ffmpeg missing after extract: %s" % ", ".join(missing))
        return False
    return True


def _llama_ready():
    return all(os.path.isfile(os.path.join(LLAMA_BIN_DIR, name))
               for name in LLAMA_MARKERS)


def _ensure_llama_zip(asset, archive, force):
    name, expected = LLAMA_ASSETS[asset]
    if force or not size_ok(archive, expected):
        urls = [prefix + _LLAMA_BASE + name for prefix in GITHUB_MIRRORS]
        return download_smart(urls, archive, expected, group="github")
    return True


def ensure_llama(force=False):
    print("== llama.cpp (%s) ==" % LLAMA_BUILD)
    if not force and _llama_ready():
        print("[skip] llama.cpp (%s)" % LLAMA_BIN_DIR)
        return True
    os.makedirs(TMP_DIR, exist_ok=True)
    cpu_zip = os.path.join(TMP_DIR, "llama-cpu.zip")
    cuda_zip = os.path.join(TMP_DIR, "llama-cuda.zip")

    if not _ensure_llama_zip("cpu", cpu_zip, force):
        print("[fail] llama.cpp CPU 包下载失败")
        return False
    print("[unzip] llama.cpp CPU")
    _extract_flat(cpu_zip, LLAMA_BIN_DIR)
    os.remove(cpu_zip)

    # CUDA 版后解压，覆盖 llama-server.exe 等公共文件；失败不阻塞 CPU 回退。
    if _ensure_llama_zip("cuda", cuda_zip, force):
        print("[unzip] llama.cpp CUDA（覆盖公共文件）")
        _extract_flat(cuda_zip, LLAMA_BIN_DIR)
        os.remove(cuda_zip)
    else:
        print("[warn] llama.cpp CUDA 包下载失败，仅使用 CPU 后端")

    if not _llama_ready():
        print("[fail] llama.cpp 未就绪（缺少 %s）" % ", ".join(LLAMA_MARKERS))
        return False
    return True


def ensure_llm(force=False):
    print("== 多模态 GGUF（%s） ==" % LLM_REPO)
    ok = True
    for remote, local, expected in LLM_FILES:
        dest = os.path.join(LLM_DIR, local)
        if not force and size_ok(dest, expected):
            print("[skip] %s" % local)
            continue
        if not download_smart(_llm_urls(remote), dest, expected, group="llm"):
            ok = False
    return ok


def ensure_all(force=False, only=None):
    selected = set(only or ["ffmpeg", "llama", "llm"])
    results = {}
    if "ffmpeg" in selected:
        results["ffmpeg"] = ensure_ffmpeg(force=force)
    if "llama" in selected:
        results["llama"] = ensure_llama(force=force)
    if "llm" in selected:
        results["llm"] = ensure_llm(force=force)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="下载运行期资产（ffmpeg/llama.cpp/GGUF）")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument("--only", action="append", choices=["ffmpeg", "llama", "llm"],
                        help="只下载指定资产（可重复）")
    args = parser.parse_args(argv)

    results = ensure_all(force=args.force, only=args.only)
    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print("failed: %s" % ", ".join(failed))
        return 1
    print("runtime assets ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
