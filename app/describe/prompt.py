"""H3 full-reference（Ref2VA）描述 prompt 的组装与清理。

系统 / 用户模板以 UTF-8 资源文件形式放在 `prompts/` 下，便于直接阅读和修改；
本模块只负责按输入模式加载模板、填充多模态输入信息、以及清理模型输出。

输入模式（`describe.input_mode`）：
- `storyboard`（默认）：直接把片段的 3x3 宫格故事图（`scene_reference.jpg`）作为推理图像；
- `frames`：使用 ffmpeg 按时间均匀采样的多帧图像进行推理。

两种模式使用不同的模板文件（`storyboard_*` / `frames_*`），因为随附图像的含义不同。
"""

import os
import re

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

INPUT_MODE_STORYBOARD = "storyboard"
INPUT_MODE_FRAMES = "frames"

SYSTEM_FILES = {
    INPUT_MODE_STORYBOARD: "storyboard_system.txt",
    INPUT_MODE_FRAMES: "frames_system.txt",
}
USER_FILES = {
    INPUT_MODE_STORYBOARD: "storyboard_user.txt",
    INPUT_MODE_FRAMES: "frames_user.txt",
}

MAX_CHARACTER_REFS = 5

_DEFAULT_USER = (
    "片段时长：{{DURATION}} 秒\n\n"
    "帧时间线：\n{{FRAME_LINES}}\n\n"
    "字幕转写：\n{{SUBTITLES}}\n\n"
    "任务：现在为本片段写出六段式全参考改写，只返回六段。\n"
)


def normalize_mode(mode):
    value = str(mode or "").strip().lower()
    if value in SYSTEM_FILES:
        return value
    return INPUT_MODE_STORYBOARD


def _load(name, fallback=""):
    path = os.path.join(PROMPT_DIR, name)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return fallback


def system_prompt(mode=INPUT_MODE_STORYBOARD):
    return _load(SYSTEM_FILES[normalize_mode(mode)]).strip()


def reference_mapping_text(grid_rows=3, grid_cols=3):
    """列出后续会喂给视频生成模型的参考资产及其标签映射。

    仅使用 `<Picture N>` 与 `<Audio N>`：`<Picture 1>` 为 3x3 宫格故事图，
    第 n 个主要人物为 `<Picture n+1>`（外貌来自第 n 张人物参考图），
    对应声音参考为 `<Audio n>`。
    """
    rows = int(grid_rows or 3)
    cols = int(grid_cols or 3)
    count = max(1, rows * cols)
    lines = [
        "- <Picture 1>: 本片段的 %dx%d 宫格故事图（storyboard）；%d 个画面按从左到右、"
        "从上到下的故事顺序排列，用于定义镜头顺序、视角、主体位置与关键情节节点。"
        "这 %d 格只是片段内部按时间先后的关键画面，不代表多个镜头；它不是场景参考图。"
        % (rows, cols, count, count),
    ]
    for index in range(1, MAX_CHARACTER_REFS + 1):
        lines.append(
            "- <Picture %d> / <Audio %d>: 第 %d 个主要人物的参考图 / 声音音色参考（若存在）。"
            % (index + 1, index, index))
    return "\n".join(lines)


def format_frames(frames):
    lines = []
    for frame in frames:
        lines.append("%s @ %.3fs" % (frame.get("name", "?"),
                                     float(frame.get("time_sec", 0.0))))
    return "\n".join(lines) if lines else "(no frames)"


def format_subtitles(subtitles):
    if not subtitles:
        return "No speech detected in this segment."
    lines = []
    for item in subtitles:
        speaker = str(item.get("speaker") or "").strip()
        prefix = ("%s: " % speaker) if speaker else ""
        lines.append("[%.2fs - %.2fs] %s%s" % (
            float(item.get("start_sec", 0.0)),
            float(item.get("end_sec", 0.0)),
            prefix, str(item.get("text", "")).strip()))
    return "\n".join(lines)


def format_speakers(speakers):
    return ", ".join(speakers) if speakers else "none detected"


def build_user_prompt(mode, duration_sec, fps, frames, subtitles, speakers,
                      grid=None):
    mode = normalize_mode(mode)
    template = _load(USER_FILES[mode], _DEFAULT_USER)
    rows, cols = (grid or (3, 3))[:2]
    rows = int(rows or 3)
    cols = int(cols or 3)
    replacements = {
        "{{DURATION}}": "%.2f" % float(duration_sec),
        "{{FPS}}": _fmt_number(fps),
        "{{FRAME_COUNT}}": str(len(frames or [])),
        "{{REFERENCES}}": reference_mapping_text(rows, cols),
        "{{SPEAKERS}}": format_speakers(speakers),
        "{{FRAME_LINES}}": format_frames(frames or []),
        "{{SUBTITLES}}": format_subtitles(subtitles),
        "{{GRID_ROWS}}": str(rows),
        "{{GRID_COLS}}": str(cols),
        "{{GRID_COUNT}}": str(rows * cols),
    }
    text = template
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text.strip()


def _fmt_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == int(number):
        return str(int(number))
    return ("%.3f" % number).rstrip("0")


_THINK_BLOCK = re.compile(r"(?is)<(?:think|thinking|reasoning)\b[^>]*>.*?</(?:think|thinking|reasoning)>")
_THINK_OPEN = re.compile(r"(?is)<(?:think|thinking|reasoning)\b[^>]*>.*")
_CODE_FENCE = re.compile(r"(?s)^\s*```[a-zA-Z0-9_-]*\s*(.*?)\s*```\s*$")
_SPECIAL_TOKEN = re.compile(r"(?m)<\|(?:im_start|im_end|endoftext|eot_id|start_header_id|end_header_id)\|>")


def clean_output(text):
    """清理模型输出：去掉思考块、代码围栏与特殊 token。"""
    if not text:
        return ""
    cleaned = str(text).replace("\r\n", "\n")
    cleaned = _THINK_BLOCK.sub("", cleaned)
    cleaned = _SPECIAL_TOKEN.sub("", cleaned)
    fence = _CODE_FENCE.match(cleaned)
    if fence:
        cleaned = fence.group(1)
    # 思考块未闭合时，去掉从起始标签到结尾的内容（通常思考在正文之前）。
    if re.search(r"(?is)<(?:think|thinking|reasoning)\b", cleaned):
        cleaned = _THINK_OPEN.sub("", cleaned)
    return cleaned.strip()


SECTION_HEADINGS = (
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)


def missing_sections(text):
    lowered = (text or "").lower()
    return [name for name in SECTION_HEADINGS if name not in lowered]
