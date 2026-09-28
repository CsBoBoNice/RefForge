"""交互式配置：双击 `启动.bat` 时一次性收集运行参数。

流程（全部选择完成后才开始处理，避免每执行一个阶段问一次，共 8 项）：

1. 运行设备：GPU（默认）或 CPU；
2. 分割时长：片段最短 / 最长时长（默认 1s / 15s）；
3. 是否执行「镜头分割 -> 故事图 -> 人声分离 -> 语音识别」；不执行时原视频整体
   视为单个片段，仍继续生成故事图 / 人声分离 / 语音识别（等价于 `--no-split`）；
4. 人声分离强度：低（`fast`）/ 中（`balanced`）/ 高（`best`），写回
   `separation.model`；
5. 人物提取与归类：动漫（默认）/ 真人 / 不执行；
6. 是否生成视频描述；不生成时跳过后续两步；
7. 视频描述输入图像：故事图识别（`storyboard`）或帧识别（`frames`）；
8. 需要生成视频描述的片段：默认全部，可按编号 / 范围选择，或选择不生成。

本模块只负责交互收集与校验，并把结果写回 `cfg`；片段枚举通过外部传入的
``enumerate_fn(path)`` 完成，避免与 `main.py` 的编排逻辑相互导入。
"""

import os
import re
import sys

BANNER = "=" * 52
DEFAULT_MIN_SEC = 1.0
DEFAULT_MAX_SEC = 15.0


class Choices(object):
    """交互选择结果。"""

    def __init__(self):
        self.min_sec = DEFAULT_MIN_SEC
        self.max_sec = DEFAULT_MAX_SEC
        self.split = True
        self.separate_tier = "fast"
        self.persons = True
        self.persons_domain = "anime"
        self.describe = True
        self.device = "gpu"
        self.input_mode = "storyboard"
        self.describe_all = True
        self.describe_none = False
        self.describe_shots = {}   # source path -> set(shot_id)
        self.precomputed = {}      # source path -> {"meta", "segments", "build_meta"}


def _fmt(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == int(number):
        return str(int(number))
    return ("%.3f" % number).rstrip("0").rstrip(".")


def _prompt_raw(text, hint=""):
    """打印提示并读取一行；空行返回 ""，EOF 返回 None。"""
    suffix = " [%s]" % hint if hint else ""
    sys.stdout.write("%s%s: " % (text, suffix))
    sys.stdout.flush()
    try:
        line = input()
    except EOFError:
        print()
        return None
    return line.strip()


def _prompt_float(text, default, lo=None, hi=None):
    for _ in range(10):
        raw = _prompt_raw(text, _fmt(default))
        if raw is None or raw == "":
            return default
        try:
            value = float(raw)
        except ValueError:
            print("  请输入数字。")
            continue
        if lo is not None and value < lo:
            print("  需大于等于 %s。" % _fmt(lo))
            continue
        if hi is not None and value > hi:
            print("  需小于等于 %s。" % _fmt(hi))
            continue
        return value
    return default


def _prompt_yes_no(text, default):
    hint = "Y/n" if default else "y/N"
    for _ in range(10):
        raw = _prompt_raw(text, hint)
        if raw is None or raw == "":
            return default
        if raw.lower() in ("y", "yes", "是", "1"):
            return True
        if raw.lower() in ("n", "no", "否", "0"):
            return False
        print("  请输入 y 或 n。")
    return default


def _prompt_choice(text, mapping, default):
    hint = default
    for _ in range(10):
        raw = _prompt_raw(text, hint)
        if raw is None or raw == "":
            return mapping[default]
        if raw in mapping:
            return mapping[raw]
        print("  输入无效，请输入 %s。" % " / ".join(sorted(mapping)))
    return mapping[default]


def _parse_selection(raw, count):
    """解析 '1,3,5-8' 形式的编号表达式；返回 (sorted list, ok)。"""
    result = set()
    for chunk in re.split(r"[,\s]+", raw):
        if not chunk:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", chunk)
        if not match:
            return None, False
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if start > end:
            start, end = end, start
        if start < 1 or end > count:
            return None, False
        result.update(range(start, end + 1))
    if not result:
        return None, False
    return sorted(result), True


def _prompt_selection(count):
    """返回 None（全部）/ []（不生成）/ 编号列表。"""
    for _ in range(10):
        raw = _prompt_raw("  选择", "")
        if raw is None or raw == "" or raw.lower() == "all":
            return None
        raw = raw.lower()
        if raw in ("none", "n", "no", "0", "-", "不生成"):
            return []
        parsed, ok = _parse_selection(raw, count)
        if ok:
            return parsed
        print("  输入无效，请使用编号或范围（如 1,3,5-8）；回车=全部，"
              "none=不生成描述。")
    return None


def _print_entries(entries):
    print("  可描述的片段（共 %d 个）：" % len(entries))
    for order, entry in enumerate(entries, start=1):
        print("   %3d) [%s] %s  %.2f-%.2fs  (%.2fs)" % (
            order, entry["name"], entry["shot_id"], entry["start_sec"],
            entry["end_sec"], entry["duration_sec"]))
    print("  输入编号或范围（如 1,3,5-8）；回车=全部，none=不生成描述。")


def _entry_for(seg, path):
    return {
        "path": path,
        "name": os.path.basename(path),
        "shot_id": seg.shot_id or ("shot_%04d" % seg.index),
        "start_sec": float(seg.start_sec),
        "end_sec": float(seg.end_sec),
        "duration_sec": float(seg.duration_sec),
    }


def collect_choices(videos, cfg, logger, enumerate_fn):
    """交互收集 8 项选择，写回 cfg，并返回 `Choices`。"""
    choices = Choices()
    if not videos:
        return choices

    print(BANNER)
    print("交互配置：请依次完成 8 项选择，全部完成后自动执行后续流程。")
    print(BANNER)

    # 第一步：运行设备
    print("[1/8] 运行设备（GPU 更快；CPU 兼容性更好）")
    print("  1) GPU（默认）")
    print("  2) CPU")
    device = _prompt_choice("  选择", {"1": "gpu", "2": "cpu"}, "1")
    choices.device = device
    cfg["separation"]["device"] = device
    cfg["asr"]["device"] = device
    cfg["persons"]["device"] = device
    cfg["describe"]["gpu_layers"] = 999 if device == "gpu" else 0

    # 第二步：分割时长
    print("[2/8] 分割时长（片段时长范围，单位：秒）")
    choices.min_sec = _prompt_float("  最短时长", DEFAULT_MIN_SEC, 0.1, None)
    choices.max_sec = _prompt_float("  最长时长", DEFAULT_MAX_SEC,
                                    choices.min_sec, None)
    if choices.max_sec < choices.min_sec:
        choices.max_sec = choices.min_sec
    cfg["segmentation"]["min_segment_sec"] = float(choices.min_sec)
    cfg["segmentation"]["max_segment_sec"] = float(choices.max_sec)

    # 第三步：是否执行前半段流程
    print("[3/8] 是否执行：镜头分割 -> 故事图 -> 人声分离 -> 语音识别")
    choices.split = _prompt_yes_no("  执行", True)
    if not choices.split:
        print("  已选择不执行分割：原视频将整体作为单个片段，继续生成故事图 /"
              " 人声分离 / 语音识别。")
    cfg["segmentation"]["enabled"] = bool(choices.split)

    # 第四步：人声分离强度
    print("[4/8] 人声分离强度（逐源视频整轨分离人声 / 伴奏）")
    print("  1) 低（fast，最快，UVR-MDX-Net）")
    print("  2) 中（balanced，质量与速度均衡，BS-Roformer）")
    print("  3) 高（best，质量最好但最慢，Mel-Band Roformer）")
    tier_default = {"fast": "1", "balanced": "2",
                    "best": "3"}.get(cfg["separation"].get("model"), "1")
    tier = _prompt_choice(
        "  选择", {"1": "fast", "2": "balanced", "3": "best"}, tier_default)
    choices.separate_tier = tier
    cfg["separation"]["model"] = tier

    # 第五步：人物提取与归类的模型选型
    print("[5/8] 人物提取与归类（整片，在各阶段之后）")
    print("  1) 动漫模型（默认，2D/3D 动画，CLIP 嵌入 + ByteTrack）")
    print("  2) 真人模型（写实视频）")
    print("  3) 不执行")
    person_choice = _prompt_choice(
        "  选择", {"1": "anime", "2": "real", "3": "off"}, "1")
    choices.persons = person_choice != "off"
    choices.persons_domain = ("anime" if person_choice == "off" else person_choice)
    cfg["persons"]["enabled"] = bool(choices.persons)
    cfg["persons"]["domain"] = choices.persons_domain

    # 第六步：是否生成视频描述（默认生成）
    print("[6/8] 是否生成视频描述（逐片段中文 prompt）")
    choices.describe = _prompt_yes_no("  生成", True)
    cfg["describe"]["enabled"] = bool(choices.describe)
    if not choices.describe:
        print("  已选择不生成视频描述：跳过输入图像与片段选择，"
              "也不加载描述模型。")
        choices.describe_all = False
        choices.describe_none = True
        print(BANNER)
        logger.info(
            "interactive: device=%s split=%s separation=%s persons=%s/%s "
            "duration=[%ss, %ss] describe=none", choices.device, choices.split,
            choices.separate_tier, choices.persons, choices.persons_domain,
            _fmt(choices.min_sec), _fmt(choices.max_sec))
        return choices

    # 第七步：视频描述输入图像
    print("[7/8] 视频描述使用的输入图像")
    print("  1) 故事图识别（storyboard，默认）")
    print("  2) 帧识别（frames）")
    mode = _prompt_choice("  选择", {"1": "storyboard", "2": "frames"}, "1")
    choices.input_mode = mode
    cfg["describe"]["input_mode"] = mode

    # 第八步：枚举片段并选择需要生成描述的片段
    print("[8/8] 选择需要生成视频描述的片段")
    entries = []
    for path in videos:
        data = enumerate_fn(path)
        if data is None:
            print("  跳过（已处理，加 --overwrite 可重跑）：%s"
                  % os.path.basename(path))
            continue
        choices.precomputed[path] = data
        for seg in data.get("segments", []) or []:
            entries.append(_entry_for(seg, path))

    if not entries:
        print("  没有可用于生成描述的片段，将跳过视频描述。")
        choices.describe_all = False
        choices.describe_none = True
        print(BANNER)
        return choices

    _print_entries(entries)
    selected = _prompt_selection(len(entries))
    if selected is None:
        choices.describe_all = True
    elif not selected:
        choices.describe_all = False
        choices.describe_none = True
    else:
        choices.describe_all = False
        choices.describe_none = False
        for index in selected:
            entry = entries[index - 1]
            choices.describe_shots.setdefault(entry["path"], set()).add(
                entry["shot_id"])

    print(BANNER)
    logger.info(
        "interactive: device=%s split=%s separation=%s persons=%s/%s "
        "duration=[%ss, %ss] input_mode=%s describe=%s", choices.device,
        choices.split, choices.separate_tier, choices.persons,
        choices.persons_domain, _fmt(choices.min_sec),
        _fmt(choices.max_sec), choices.input_mode,
        "none" if choices.describe_none
        else ("all" if choices.describe_all else "selected"))
    return choices
