"""视频描述子系统：多模态大模型（llama.cpp + Qwen3.5）逐片段生成 H3 prompt。

对外接口：
    available(cfg) -> (ok, missing)          # llama-server / 模型是否就绪
    run_phase(jobs, output_root, cfg, logger) # 视频描述阶段
"""

from .runner import available, run_phase

__all__ = ["available", "run_phase"]
