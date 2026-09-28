"""一键清理运行生成的产物（不影响模型、运行时与源码）。

清理范围（可分别关闭）：
1. <包根>/output/ 下全部内容（segments、voiceprint_center.json、run.log 等）；
2. <包根>/input/ 下由识别生成的 *.srt（整视频字幕，写在输入视频同目录）；
3. 系统临时目录下本项目验收脚本的工作目录（ref_forge_*）。

保护范围（绝不删除）：
- models/（离线模型）、python/（运行时）、bin/、app/、scripts/、demo/；
- input/ 下的视频文件、git 已跟踪的任何文件。

默认 **dry-run（只预览不删除）**；确认后加 `--yes` 执行。

用法：
    dev.bat scripts\\clean_generated.py                # 预览
    dev.bat scripts\\clean_generated.py --yes          # 执行清理
    dev.bat scripts\\clean_generated.py --yes --no-temp
    dev.bat scripts\\clean_generated.py --yes --output <自定义输出目录>
"""

import argparse
import os
import shutil
import stat
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTECTED_TOP = {"models", "python", "bin", "app", "scripts", "demo", ".git"}


def _remove(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, onerror=_on_rm_error)
    else:
        try:
            os.remove(path)
        except OSError:
            os.chmod(path, stat.S_IWRITE)
            os.remove(path)


def _on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def _path_size(path):
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _list_output(output_dir):
    if not os.path.isdir(output_dir):
        return []
    return [os.path.join(output_dir, name)
            for name in os.listdir(output_dir)]


def _list_input_srt(input_dir):
    if not os.path.isdir(input_dir):
        return []
    return [os.path.join(input_dir, name)
            for name in os.listdir(input_dir)
            if name.lower().endswith(".srt")
            and os.path.isfile(os.path.join(input_dir, name))]


def _list_temp_dirs():
    base = tempfile.gettempdir()
    if not os.path.isdir(base):
        return []
    return [os.path.join(base, name) for name in os.listdir(base)
            if name.startswith("ref_forge_")
            and os.path.isdir(os.path.join(base, name))]


def _is_protected(path):
    absolute = os.path.abspath(path)
    if not absolute.startswith(os.path.abspath(ROOT) + os.sep):
        return True
    head = os.path.relpath(absolute, ROOT).split(os.sep)[0]
    return head in PROTECTED_TOP


def _collect(args):
    targets = []
    targets.extend(_list_output(args.output))
    targets.extend(_list_input_srt(args.input))
    targets = [path for path in targets if not _is_protected(path)]
    if args.temp:
        targets.extend(_list_temp_dirs())
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description="清理运行生成的产物")
    parser.add_argument("--yes", action="store_true", help="实际执行删除（默认仅预览）")
    parser.add_argument("--input", default=os.path.join(ROOT, "input"),
                        help="输入目录（清理其中生成的 *.srt）")
    parser.add_argument("--output", default=os.path.join(ROOT, "output"),
                        help="输出目录（清空其全部内容）")
    parser.add_argument("--no-temp", dest="temp", action="store_false",
                        default=True, help="不清理系统临时目录中的验收工作目录")
    args = parser.parse_args(argv)

    if _is_protected(args.output):
        print("[错误] 输出目录位于受保护位置，拒绝清理: %s" % args.output)
        return 1

    targets = _collect(args)
    total = sum(_path_size(path) for path in targets)
    print("将清理 %d 项，共 %.2f MB：" % (len(targets), total / (1024.0 * 1024.0)))
    for path in targets:
        print("  %-8s %s" % ("[dir]" if os.path.isdir(path) else "[file]",
                             os.path.relpath(path, ROOT)
                             if os.path.abspath(path).startswith(ROOT)
                             else path))

    if not args.yes:
        print("\n预览完成（未删除）。确认后加 --yes 执行。")
        return 0

    for path in targets:
        try:
            _remove(path)
        except OSError as exc:
            print("  [警告] 删除失败 %s: %s" % (path, exc))

    # 保证 output/ 目录仍存在，便于后续运行
    os.makedirs(args.output, exist_ok=True)
    print("\n清理完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
