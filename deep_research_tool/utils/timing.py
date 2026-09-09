"""
Stage timing for a research run — MEASURED numbers only.

``StageTimer`` records wall-clock seconds per named stage plus arbitrary
counters (API calls, cache reuse). The report shows these as measured
facts; it never derives a "% faster" claim or a remaining-time estimate
from them (the review explicitly rejects unmeasured speed-up figures).
"""

import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional


class StageTimer:

    def __init__(self):
        self._lock = threading.Lock()
        self._stages: List[Dict[str, Any]] = []
        self._open: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        self.started_at = time.time()
        self.finished_at: Optional[float] = None

    def start(self, stage: str) -> None:
        with self._lock:
            self._open[stage] = time.time()

    def stop(self, stage: str, **meta) -> Optional[float]:
        with self._lock:
            t0 = self._open.pop(stage, None)
            if t0 is None:
                return None
            dur = round(time.time() - t0, 2)
            self._stages.append({"stage": stage, "seconds": dur, **meta})
            return dur

    @contextmanager
    def stage(self, name: str, **meta):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name, **meta)

    def count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + int(n)

    def set_count(self, name: str, value: int) -> None:
        with self._lock:
            self.counters[name] = int(value)

    def finish(self) -> None:
        with self._lock:
            # stages still open (e.g. after a cancel) are closed as-is
            for stage, t0 in list(self._open.items()):
                self._stages.append({"stage": stage,
                                     "seconds": round(time.time() - t0, 2),
                                     "interrupted": True})
            self._open.clear()
            self.finished_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            end = self.finished_at or time.time()
            return {
                "total_seconds": round(end - self.started_at, 2),
                "stages": list(self._stages),
                "counters": dict(self.counters),
            }
