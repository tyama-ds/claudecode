"""Small per-run stage timer; no prompts, document text or credentials."""

import threading
import time


class StageTimings:
    def __init__(self):
        self.started = time.perf_counter()
        self._stages = {}
        self._lock = threading.Lock()

    def call(self, stage, operation, *args, **kwargs):
        start = time.perf_counter()
        failed = False
        try:
            return operation(*args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            elapsed = time.perf_counter() - start
            with self._lock:
                entry = self._stages.setdefault(stage, {"seconds": 0.0, "calls": 0, "failures": 0})
                entry["seconds"] += elapsed
                entry["calls"] += 1
                entry["failures"] += int(failed)

    def snapshot(self):
        with self._lock:
            stages = {key: {**value, "seconds": round(value["seconds"], 4)}
                      for key, value in self._stages.items()}
        return {"total_seconds": round(time.perf_counter() - self.started, 4), "stages": stages}
