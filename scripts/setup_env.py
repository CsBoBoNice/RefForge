"""一键环境搭建（在包内 Python 中执行）：安装依赖 + 下载运行期资产与全部模型。

由根目录 `一键环境搭建.bat` 在引导好包内 Python 后调用，也可用 `dev.bat` 手动运行：

    dev.bat scripts\\setup_env.py                # 依赖 + 运行期资产 + 模型（全流程）
    dev.bat scripts\\setup_env.py --skip-pip     # 仅补运行期资产与模型
    dev.bat scripts\\setup_env.py --skip-models  # 仅依赖 + 运行期资产
    dev.bat scripts\\setup_env.py --force        # 强制重新下载资产/模型

安装阶段与 `requirements.txt` / `scripts/setup_runtime.bat` 保持一致；
模型下载复用既有 `download_*_models.py`，运行期资产复用 `download_runtime.py`。
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
PYEXE = os.path.join(ROOT, "python", "python.exe")

sys.path.insert(0, SCRIPTS)

from download_utils import pick_fastest  # noqa: E402

TSINGHUA = "https://pypi.tuna.tsinghua.edu.cn/simple"
HUAWEI = "https://repo.huaweicloud.com/repository/pypi/simple"
ALIYUN = "https://mirrors.aliyun.com/pypi/simple"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
ALIYUN_TORCH = "https://mirrors.aliyun.com/pytorch-wheels/cu128"

# PyPI 候选镜像：实测网速后选最快者作为 `-i`。
PYPI_MIRRORS = [TSINGHUA, HUAWEI, ALIYUN, "https://pypi.org/simple"]


def pick_index():
    """按实测网速返回最快的 PyPI 镜像（以 pip 索引页为探测样本）。"""
    probes = [mirror.rstrip("/") + "/pip/" for mirror in PYPI_MIRRORS]
    best = pick_fastest(probes, group="pypi")
    for mirror, probe in zip(PYPI_MIRRORS, probes):
        if best == probe:
            return mirror
    return TSINGHUA

BASE_PACKAGES = [
    "opencv-python-headless==4.10.0.84",
    "numpy==2.2.6",
    "Pillow==12.3.0",
    "scenedetect==0.6.4",
    "click==8.5.0",
    "platformdirs==4.11.9",
    "requests",
]
SPEECH_PACKAGES = [
    "funasr==1.4.15",
    "qwen-asr==0.0.6",
    "modelscope==1.40.1",
    "transformers==4.57.6",
    "librosa==0.11.0",
    "soundfile==0.14.0",
]
SEPARATION_PACKAGES = [
    "beartype==0.18.5",
    "diffq-fixed==0.2.4",
    "einops==0.8.2",
    "julius==0.2.8",
    "ml_collections==1.1.0",
    "resampy==0.4.3",
    "rotary-embedding-torch==0.6.5",
    "samplerate==0.1.0",
    "absl-py==2.5.0",
    "ml_dtypes==0.6.0",
    "onnx-weekly==1.24.0.dev20260914",
    "onnx2torch-py313==1.6.0",
    "onnxruntime-gpu==1.23.2",
]
PERSON_PACKAGES = ["lap==0.5.13"]

MODEL_SCRIPTS = [
    ("语音识别 / 声纹 / VAD", "download_models.py"),
    ("人声分离四档", "download_separation_models.py"),
    ("真人人物提取", "download_person_models.py"),
    ("动漫人物提取", "download_anime_person_models.py"),
]


def _env():
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUTF8"] = "1"
    prefix = os.pathsep.join([
        os.path.join(ROOT, "python", "Scripts"),
        os.path.join(ROOT, "python"),
        os.path.join(ROOT, "bin"),
    ])
    env["PATH"] = prefix + os.pathsep + env.get("PATH", "")
    return env


def _run(cmd, check=True):
    print("$ %s" % " ".join(cmd))
    result = subprocess.run(cmd, env=_env())
    if check and result.returncode != 0:
        raise RuntimeError("command failed (code=%s): %s"
                           % (result.returncode, " ".join(cmd)))
    return result.returncode


def _pip(args):
    _run([sys.executable, "-m", "pip", "install",
          "--no-warn-script-location", "--disable-pip-version-check"] + args)


def install_pip_deps():
    print("\n== [1/3] 安装 Python 依赖 ==")
    index = pick_index()
    print("[best ] PyPI 镜像：%s" % index)
    _pip(["--upgrade", "pip", "setuptools", "wheel", "-i", index])

    print("\n-- 基础依赖 --")
    _pip(BASE_PACKAGES + ["-i", index])

    print("\n-- PyTorch cu128（覆盖 RTX 30/40/50 系）--")
    _pip(["torch==2.9.1+cu128", "torchaudio==2.9.1+cu128",
          "--index-url", TORCH_INDEX, "--extra-index-url", index])

    print("\n-- 语音识别依赖 --")
    _pip(SPEECH_PACKAGES + ["-i", index])

    print("\n-- 人声分离依赖 --")
    _pip(["torchvision==0.24.1+cu128", "--no-deps", "-f", ALIYUN_TORCH])
    _pip(["audio-separator==0.47.0", "--no-deps", "-i", index])
    _pip(SEPARATION_PACKAGES + ["-i", index])

    print("\n-- 人物提取依赖 --")
    _pip(["ultralytics==8.3.253", "--no-deps", "-i", index])
    _pip(PERSON_PACKAGES + ["-i", index])
    _run([sys.executable, os.path.join(SCRIPTS, "verify_person_env.py")])


def download_runtime_assets(force=False):
    print("\n== [2/3] 下载运行期资产（ffmpeg / llama.cpp / GGUF）==")
    sys.path.insert(0, SCRIPTS)
    import download_runtime
    results = download_runtime.ensure_all(force=force)
    failed = [name for name, ok in results.items() if not ok]
    if failed:
        raise RuntimeError("runtime assets failed: %s" % ", ".join(failed))


def download_models(force=False):
    print("\n== [3/3] 下载模型 ==")
    for label, script in MODEL_SCRIPTS:
        print("\n-- %s --" % label)
        cmd = [sys.executable, os.path.join(SCRIPTS, script)]
        if force:
            cmd.append("--force")
        _run(cmd)


def main(argv=None):
    parser = argparse.ArgumentParser(description="RefForge 一键环境搭建")
    parser.add_argument("--skip-pip", action="store_true", help="跳过依赖安装")
    parser.add_argument("--skip-runtime", action="store_true",
                        help="跳过 ffmpeg / llama.cpp / GGUF 下载")
    parser.add_argument("--skip-models", action="store_true", help="跳过模型下载")
    parser.add_argument("--force", action="store_true", help="强制重新下载资产/模型")
    args = parser.parse_args(argv)

    expected = os.path.normcase(os.path.abspath(PYEXE))
    actual = os.path.normcase(os.path.abspath(sys.executable))
    if expected != actual:
        print("[错误] 请使用包内 Python 运行本脚本：")
        print("        一键环境搭建.bat   或   dev.bat scripts\\setup_env.py")
        print("        当前解释器：%s" % sys.executable)
        return 2

    print("RefForge 环境搭建：Python %s" % sys.version.split()[0])
    if not args.skip_pip:
        install_pip_deps()
    if not args.skip_runtime:
        download_runtime_assets(force=args.force)
    if not args.skip_models:
        download_models(force=args.force)

    print("\n============================================")
    print(" 环境搭建完成。运行：启动.bat")
    print("============================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
