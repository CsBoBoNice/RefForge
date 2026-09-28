"""语音子系统：人声分离（双轨）+ 语音识别（Qwen3-ASR + FunASR 声纹）。

对外接口：
    available(cfg) -> (ok, missing)                 # 语音识别是否就绪
    run_phase(jobs, output_root, cfg, logger)       # 语音识别阶段
    shutdown()

    separation_available(cfg) -> (ok, missing)      # 人声分离是否就绪
    run_separation_phase(jobs, cfg, logger)         # 人声分离阶段（先于 ASR）
    separation_shutdown()
"""

from .models import available, configure_language, shutdown
from .runner import run_phase
from .separation import available as separation_available
from .separation import run_phase as run_separation_phase

__all__ = [
    "available", "configure_language", "shutdown", "run_phase",
    "separation_available", "run_separation_phase",
]
