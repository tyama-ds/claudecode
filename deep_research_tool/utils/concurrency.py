"""
Shared concurrency control for the whole application.

Two problems are solved here:

1. **One knob for parallelism** — ``parallel_max_workers`` (default 8,
   allowed 1..16, hard cap 16). Stage-specific caps (deep_think
   max_workers, figure_max_workers, fast_crawl_workers,
   max_concurrent_searches, chunk workers, ...) remain as CEILINGS; the
   effective pool size of every stage is::

       effective_workers = min(parallel_max_workers, stage_cap, task_count)

2. **A real limit on concurrent I/O** — pool sizes alone cannot bound
   the SUM of several independent thread pools, or several Web UI jobs
   in one server process. ``ConcurrencyLimiter`` is a bounded semaphore
   acquired ONLY around leaf operations (an LLM call, an HTTP fetch),
   never while waiting on child tasks — so nesting pools cannot
   deadlock. Two scopes compose:

   - run scope: one limiter per research run (``parallel_max_workers``)
   - process scope: one limiter per Python process (hard cap 16),
     shared by all Web UI jobs

   "Process-wide" means THIS Python/Web-server process. Independently
   launched OS processes (separate CLI runs, multiple servers) are NOT
   coordinated — there is no cross-process semaphore.

Permits are always released in ``finally`` (the context manager), so
exceptions, timeouts and cancellations cannot leak permits. Work here is
I/O-bound (LLM/API/HTTP), so ThreadPools are used throughout; nothing is
moved to a ProcessPool without measured CPU-bound need.
"""

import threading
from contextlib import contextmanager
from typing import Optional

PARALLEL_MAX_WORKERS_DEFAULT = 8
PARALLEL_MAX_WORKERS_MIN = 1
PARALLEL_MAX_WORKERS_HARD_CAP = 16


def validate_parallel_max_workers(value, source: str = "parallel_max_workers") -> int:
    """Strictly validate a parallel_max_workers value.

    Accepts int or an integer-valued string ("8"). Rejects bool, float
    (including 8.0 — the UI must send an integer), zero, negatives,
    values above the hard cap, and non-numeric strings. NEVER clamps:
    an out-of-range value is an error returned to the caller, not a
    silently adjusted number.
    """
    if isinstance(value, bool):
        raise ValueError(f"{source} must be an integer, not a bool "
                         f"(got {value!r})")
    if isinstance(value, float):
        raise ValueError(f"{source} must be an integer, not a float "
                         f"(got {value!r})")
    if isinstance(value, str):
        text = value.strip()
        if not text.lstrip("+-").isdigit():
            raise ValueError(f"{source} must be an integer "
                             f"(got {value!r})")
        value = int(text)
    if not isinstance(value, int):
        raise ValueError(f"{source} must be an integer (got {value!r})")
    if value < PARALLEL_MAX_WORKERS_MIN or value > PARALLEL_MAX_WORKERS_HARD_CAP:
        raise ValueError(
            f"{source} must be between {PARALLEL_MAX_WORKERS_MIN} and "
            f"{PARALLEL_MAX_WORKERS_HARD_CAP} (got {value})")
    return value


def effective_workers(parallel_max_workers: Optional[int],
                      stage_cap: Optional[int],
                      task_count: Optional[int] = None) -> int:
    """min(parallel_max_workers, stage_cap, task_count), each optional."""
    candidates = [v for v in (parallel_max_workers, stage_cap, task_count)
                  if v is not None]
    return max(1, min(candidates)) if candidates else 1


class ConcurrencyLimiter:
    """Bounded semaphore with peak measurement.

    Acquire a permit ONLY around a leaf operation (one LLM call, one
    HTTP request). A parent task must NEVER hold a permit while waiting
    for child tasks, or nested pools could deadlock.
    """

    def __init__(self, limit: int, name: str = ""):
        if limit < 1:
            raise ValueError(f"limit must be >= 1 (got {limit})")
        self.limit = limit
        self.name = name
        self._sem = threading.BoundedSemaphore(limit)
        self._lock = threading.Lock()
        self._active = 0
        self._peak = 0

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    @property
    def peak(self) -> int:
        """Highest number of simultaneously held permits (for tests and
        the completion report's measured max concurrency)."""
        with self._lock:
            return self._peak

    def reset_peak(self) -> None:
        with self._lock:
            self._peak = self._active

    @contextmanager
    def permit(self, timeout: Optional[float] = None):
        acquired = self._sem.acquire(timeout=timeout) \
            if timeout is not None else self._sem.acquire()
        if not acquired:
            raise TimeoutError(
                f"could not acquire {self.name or 'concurrency'} permit "
                f"within {timeout}s (limit={self.limit})")
        with self._lock:
            self._active += 1
            self._peak = max(self._peak, self._active)
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
            self._sem.release()


# --------------------------------------------------------------------------
# Process-wide limiter (one per Python process; Web UI jobs share it)
# --------------------------------------------------------------------------

_process_limiter = ConcurrencyLimiter(PARALLEL_MAX_WORKERS_HARD_CAP,
                                      name="process")


def get_process_limiter() -> ConcurrencyLimiter:
    return _process_limiter


class RunLimits:
    """Composed run-scope + process-scope permits for one research run.

    Attach to LLM/search clients (``client.concurrency_limiter``); they
    take one composed permit around each leaf call. Acquisition order is
    always process -> run, so no lock-order inversion is possible.
    """

    def __init__(self, parallel_max_workers: int,
                 process_limiter: Optional[ConcurrencyLimiter] = None):
        self.parallel_max_workers = validate_parallel_max_workers(
            parallel_max_workers)
        self.run_limiter = ConcurrencyLimiter(self.parallel_max_workers,
                                              name="run")
        self.process_limiter = process_limiter or get_process_limiter()

    @contextmanager
    def permit(self, timeout: Optional[float] = None):
        with self.process_limiter.permit(timeout=timeout):
            with self.run_limiter.permit(timeout=timeout):
                yield

    @property
    def run_peak(self) -> int:
        return self.run_limiter.peak


@contextmanager
def maybe_permit(limiter, timeout: Optional[float] = None):
    """Take a permit when a limiter is attached; no-op otherwise.

    ``limiter`` is anything exposing ``.permit()`` (ConcurrencyLimiter
    or RunLimits) or None.
    """
    if limiter is None:
        yield
        return
    with limiter.permit(timeout=timeout):
        yield
