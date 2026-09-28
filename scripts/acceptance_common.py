"""验收脚本公共工具：以 `input/` 中实际存在的视频为验收素材。

验收脚本不再生成或依赖固定命名的合成视频；`input/` 下有什么视频就用什么
视频验收。由于整视频 SRT 会写在源视频同目录，脚本统一把视频复制到系统
临时目录后再跑主程序，避免污染 `input/`。
"""

import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(ROOT, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def input_videos(input_dir=None):
    """返回 `input/` 下所有视频的绝对路径（自然序，无视频时为空列表）。"""
    from io_utils import list_videos
    directory = input_dir or os.path.join(ROOT, "input")
    return list_videos(directory)


def has_audio(video_path):
    """视频是否含音频流；ffprobe 不可用时返回 None。"""
    from speech.audio import has_audio_stream
    return has_audio_stream(video_path)


def stage_input(work_dir, video_paths=None):
    """清空 work_dir，把视频复制到 work_dir/input，返回 (input_dir, [文件名])。"""
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)
    stage_dir = os.path.join(work_dir, "input")
    os.makedirs(stage_dir, exist_ok=True)
    paths = video_paths if video_paths is not None else input_videos()
    names = []
    for path in paths:
        name = os.path.basename(path)
        shutil.copy2(path, os.path.join(stage_dir, name))
        names.append(name)
    return stage_dir, names


def run_pipeline(input_dir, output_dir, extra_args=()):
    """在临时输入目录上跑主程序（--overwrite），额外的阶段开关由 extra_args 传入。"""
    cmd = [sys.executable, os.path.join(APP_DIR, "main.py"),
           "--input", input_dir, "--output", output_dir, "--overwrite"]
    cmd.extend(extra_args)
    subprocess.run(cmd, check=True)
