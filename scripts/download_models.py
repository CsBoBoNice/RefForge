"""下载全部语音模型到 models/asr/（开发机联网时执行一次）。

用法：
    dev.bat scripts\\download_models.py             # 下载缺失模型
    dev.bat scripts\\download_models.py --force     # 强制重新下载

模型（ModelScope）：
    Qwen/Qwen3-ASR-1.7B                           -> models/asr/Qwen3-ASR-1.7B
    Qwen/Qwen3-ForcedAligner-0.6B                 -> models/asr/Qwen3-ForcedAligner-0.6B
    iic/speech_fsmn_vad_zh-cn-16k-common-pytorch  -> models/asr/fsmn-vad
    iic/speech_campplus_sv_zh-cn_16k-common       -> models/asr/campplus
    iic/speech_eres2netv2_sv_zh-cn_16k-common     -> models/asr/eres2netv2

网速慢时可先 `pip install -U modelscope` 后重试；ModelScope 默认走国内源。
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models", "asr")

MODELS = [
    ("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR-1.7B"),
    ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner-0.6B"),
    ("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", "fsmn-vad"),
    ("iic/speech_campplus_sv_zh-cn_16k-common", "campplus"),
    ("iic/speech_eres2netv2_sv_zh-cn_16k-common", "eres2netv2"),
]


def _is_present(path):
    if not os.path.isdir(path):
        return False
    for root, _dirs, files in os.walk(path):
        if any(name.endswith((".safetensors", ".ckpt", ".bin", ".pt"))
               for name in files):
            return True
    return False


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in argv
    os.makedirs(MODELS_DIR, exist_ok=True)
    failures = []
    for repo_id, name in MODELS:
        dest = os.path.join(MODELS_DIR, name)
        if not force and _is_present(dest):
            print("[skip] %s (already present)" % name)
            continue
        print("[download] %s -> %s" % (repo_id, dest))
        cmd = [sys.executable, "-m", "modelscope.cli.cli", "download",
               "--model", repo_id, "--local_dir", dest]
        if force:
            cmd.append("--force")
        try:
            subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            print("[error] %s: %s" % (repo_id, exc))
            failures.append(repo_id)
    if failures:
        print("failed: %s" % ", ".join(failures))
        return 1
    print("all models ready under models/asr/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
