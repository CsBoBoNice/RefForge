"""阶段耗时统计：线程安全累加，供流水线各阶段打点。"""

import threading
import time
from contextlib import contextmanager


class StageTimer(object):
    """按阶段名累加耗时（秒）与调用次数；线程池内调用安全。

    并行阶段累加的是各 worker 自身的墙钟耗时之和，用于衡量该阶段的总工作量；
    进程级阶段（如逐源视频阶段、分离、ASR）在主线程打点，反映真实墙钟耗时。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._totals = {}
        self._counts = {}

    def add(self, name, seconds):
        with self._lock:
            self._totals[name] = self._totals.get(name, 0.0) + float(seconds)
            self._counts[name] = self._counts.get(name, 0) + 1

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - start)

    def snapshot(self):
        with self._lock:
            return dict(self._totals), dict(self._counts)

    def report(self, logger, title="stage timing"):
        totals, counts = self.snapshot()
        if not totals:
            return
        logger.info("%s (total=%.2fs wall, per-stage accumulated):",
                    title, totals.get("total", 0.0))
        for name in sorted(totals, key=lambda key: -totals[key]):
            logger.info("  %-22s %8.2fs  x%d", name, totals[name],
                        counts.get(name, 0))
