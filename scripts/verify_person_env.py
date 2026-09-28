"""安装人物提取依赖后校验原环境未被破坏（供 setup_runtime.bat 调用）。

校验项：
- `opencv-python-headless` 已安装且未安装 `opencv-python`（两者冲突）；
- `cv2` / `torch` / `numpy` 版本与本项目既定版本一致；
- `ultralytics` / `lap` / `onnxruntime` 可导入。
任一不符返回非零，安装脚本随即中止。
"""

import importlib.metadata as metadata
import sys

EXPECTED_CV2 = "4.10.0"
EXPECTED_TORCH = "2.9.1+cu128"
EXPECTED_NUMPY = "2.2.6"


def main():
    errors = []
    try:
        metadata.version("opencv-python-headless")
    except metadata.PackageNotFoundError:
        errors.append("opencv-python-headless missing")
    try:
        metadata.version("opencv-python")
        errors.append("opencv-python installed (conflicts with headless)")
    except metadata.PackageNotFoundError:
        pass

    try:
        import cv2
        if cv2.__version__ != EXPECTED_CV2:
            errors.append("cv2 %s != %s" % (cv2.__version__, EXPECTED_CV2))
    except Exception as exc:
        errors.append("cv2 import failed: %s" % exc)

    try:
        import torch
        if torch.__version__ != EXPECTED_TORCH:
            errors.append("torch %s != %s" % (torch.__version__, EXPECTED_TORCH))
    except Exception as exc:
        errors.append("torch import failed: %s" % exc)

    try:
        import numpy
        if numpy.__version__ != EXPECTED_NUMPY:
            errors.append("numpy %s != %s" % (numpy.__version__, EXPECTED_NUMPY))
    except Exception as exc:
        errors.append("numpy import failed: %s" % exc)

    for module in ("ultralytics", "lap", "onnxruntime"):
        try:
            __import__(module)
        except Exception as exc:
            errors.append("%s import failed: %s" % (module, exc))

    if errors:
        for message in errors:
            print("[错误] %s" % message)
        return 1
    print("person env verified: cv2=%s torch=%s numpy=%s" % (
        EXPECTED_CV2, EXPECTED_TORCH, EXPECTED_NUMPY))
    return 0


if __name__ == "__main__":
    sys.exit(main())
