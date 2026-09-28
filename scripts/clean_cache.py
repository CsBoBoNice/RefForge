"""一键删除缓存 / 中间文件，便于把整个目录压缩拷贝到其他设备。

清理内容（默认全部）：
1. Python 缓存：所有 `__pycache__/` 目录与 `*.pyc` / `*.pyo`（含包内运行时，
   重新运行时会自动重建）；
2. 下载中间文件：`*.part` / `*.tmp` / `.downloads/`；
3. HuggingFace 记账缓存：`models/**/.cache/huggingface/`（仅下载元数据，不影响推理）；
4. 日志：`*.log`；
5. 系统 / 编辑器杂项：`Thumbs.db` / `Desktop.ini` / `.DS_Store` 与
   `.pytest_cache` / `.mypy_cache` / `.ruff_cache` / `.vscode` / `.idea`；
6. 系统临时目录中的本项目工作目录：`%TEMP%\\ref_forge_*`。

可选（默认不做）：
- `--output`：同时清空 `<包根>/output/` 与 `/output_single/`（会删除已生成的输出）；
- `--git`：删除 `<包根>/.git`（大幅减小体积，但丢失版本历史）；
- `--keep-runtime`：保留包内 `python/` 下的 Python 缓存（拷贝后首次启动更快）。

保护（绝不删除）：`app/`、`scripts/`、`models/` 模型文件本体、`python/` 运行时本体、
`bin/`、`llama_cpp/`、`input/`、`config.json` 等源码与离线资产。

默认 **dry-run（只预览不删除）**；确认后加 `--yes` 执行。

用法：
    dev.bat scripts\\clean_cache.py                 # 预览
    dev.bat scripts\\clean_cache.py --yes           # 执行清理
    dev.bat scripts\\clean_cache.py --yes --output  # 连 output/ 一起清空
    dev.bat scripts\\clean_cache.py --yes --git     # 连 .git 一起删除
"""

import argparse
import os
import shutil
import stat
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_DIR = os.path.join(ROOT, "python")

CATEGORY_ORDER = [
    ("pycache", "Python 缓存（__pycache__）"),
    ("pyc", "Python 字节码（*.pyc / *.pyo）"),
    ("download", "下载中间文件（*.part / *.tmp / .downloads）"),
    ("hf", "HuggingFace 记账缓存（.cache/huggingface）"),
    ("log", "日志（*.log）"),
    ("junk", "系统 / 编辑器杂项"),
    ("output", "输出目录（output / output_single）"),
    ("git", ".git"),
    ("temp", "系统临时目录（%TEMP%\\ref_forge_*）"),
]

JUNK_FILES = {"Thumbs.db", "Desktop.ini", ".DS_Store"}
JUNK_DIRS = {".pytest_cache", ".mypy_cache", ".ruff_cache", ".vscode", ".idea"}
PYC_SUFFIXES = (".pyc", ".pyo")
DOWNLOAD_SUFFIXES = (".part", ".tmp")
OUTPUT_DIRS = ("output", "output_single")


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


def _is_hf_cache(path):
    return (os.path.basename(path) == "huggingface"
            and os.path.basename(os.path.dirname(path)) == ".cache")


def _temp_dirs():
    base = tempfile.gettempdir()
    if not os.path.isdir(base):
        return []
    return [os.path.join(base, name) for name in os.listdir(base)
            if name.startswith("ref_forge_")
            and os.path.isdir(os.path.join(base, name))]


def _collect(include_output, include_git, include_temp):
    found = []

    def add(category, path):
        found.append((category, path))

    for root, dirs, files in os.walk(ROOT, topdown=True):
        if ".git" in dirs:
            dirs.remove(".git")
        if _is_hf_cache(root):
            add("hf", root)
            dirs[:] = []
            continue
        for name in list(dirs):
            if name == "__pycache__":
                add("pycache", os.path.join(root, name))
                dirs.remove(name)
            elif name == ".downloads":
                add("download", os.path.join(root, name))
                dirs.remove(name)
            elif name in JUNK_DIRS:
                add("junk", os.path.join(root, name))
                dirs.remove(name)
        for name in files:
            lower = name.lower()
            if name in JUNK_FILES:
                add("junk", os.path.join(root, name))
            elif lower.endswith(PYC_SUFFIXES):
                add("pyc", os.path.join(root, name))
            elif lower.endswith(DOWNLOAD_SUFFIXES):
                add("download", os.path.join(root, name))
            elif lower.endswith(".log"):
                add("log", os.path.join(root, name))

    if include_output:
        for name in OUTPUT_DIRS:
            path = os.path.join(ROOT, name)
            if os.path.isdir(path):
                add("output", path)
    if include_git:
        git_dir = os.path.join(ROOT, ".git")
        if os.path.isdir(git_dir):
            add("git", git_dir)
    if include_temp:
        for path in _temp_dirs():
            add("temp", path)
    return found


def _is_within(path, parent):
    try:
        return (os.path.commonpath([os.path.abspath(path), os.path.abspath(parent)])
                == os.path.abspath(parent))
    except ValueError:
        return False


def _dedupe(items):
    kept = []
    for category, path in sorted(items, key=lambda item: os.path.abspath(item[1])):
        if any(_is_within(path, seen) for _c, seen in kept):
            continue
        kept.append((category, path))
    return kept


def _filter_runtime(items, keep_runtime):
    if not keep_runtime:
        return items
    runtime = os.path.abspath(RUNTIME_DIR)
    return [(c, p) for c, p in items if not _is_within(p, runtime)]


def _format_size(size):
    return "%.2f MB" % (size / (1024.0 * 1024.0))


def main(argv=None):
    parser = argparse.ArgumentParser(description="一键删除缓存 / 中间文件")
    parser.add_argument("--yes", action="store_true", help="实际执行删除（默认仅预览）")
    parser.add_argument("--output", action="store_true",
                        help="同时清空 output/ 与 output_single/")
    parser.add_argument("--git", action="store_true", help="同时删除 .git（丢失版本历史）")
    parser.add_argument("--keep-runtime", action="store_true",
                        help="保留包内 python/ 下的 Python 缓存")
    parser.add_argument("--no-temp", dest="temp", action="store_false",
                        default=True, help="不清理系统临时目录中的本项目工作目录")
    args = parser.parse_args(argv)

    items = _collect(args.output, args.git, args.temp)
    items = _filter_runtime(items, args.keep_runtime)
    items = _dedupe(items)

    by_category = {}
    for category, path in items:
        by_category.setdefault(category, []).append(path)

    print("缓存 / 中间文件清单：")
    total_size = 0
    for category, label in CATEGORY_ORDER:
        paths = by_category.get(category)
        if not paths:
            continue
        size = sum(_path_size(path) for path in paths)
        total_size += size
        print("  %-12s %4d 项  %10s" % ("[" + category + "]", len(paths),
                                        _format_size(size)))
        if len(paths) <= 8:
            for path in paths:
                print("               %s" % _show(path))
    print("\n共 %d 项，%s。" % (len(items), _format_size(total_size)))

    if not args.yes:
        print("预览完成（未删除）。确认后加 --yes 执行。")
        return 0

    failed = 0
    for _category, path in items:
        try:
            _remove(path)
        except OSError as exc:
            failed += 1
            print("  [警告] 删除失败 %s: %s" % (path, exc))

    if args.output:
        for name in OUTPUT_DIRS:
            os.makedirs(os.path.join(ROOT, name), exist_ok=True)

    extras = []
    if args.output:
        extras.append("output/")
    if args.git:
        extras.append(".git/")
    print("\n清理完成（失败 %d 项），目录已可打包拷贝。" % failed)
    if extras:
        print("已一并删除：%s" % "、".join(extras))
    else:
        print("（未删除 output/ 与 .git；如需请加 --output / --git）")
    return 0


def _show(path):
    absolute = os.path.abspath(path)
    if absolute.startswith(ROOT):
        return os.path.relpath(absolute, ROOT)
    return absolute


if __name__ == "__main__":
    sys.exit(main())
