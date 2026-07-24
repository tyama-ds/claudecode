"""
TaskDAG — dependency-aware scheduler for ResearchTasks.

Reuses the app-wide concurrency model:
- pool size = effective_workers(parallel_max_workers, stage_cap, tasks)
- an optional limiter (RunLimits / ConcurrencyLimiter) is taken as a
  LEAF permit around each task execution, never while waiting on
  dependencies — consistent with utils.concurrency's rules.

Deterministic: ready tasks are dispatched in sorted task_id order and
results are returned keyed by task_id. Cycles are detected up front.
"""

import concurrent.futures
import threading
from typing import Any, Callable, Dict, List, Optional

from ..utils.concurrency import effective_workers, maybe_permit
from .models import ResearchTask


class TaskDAG:
    """A small deterministic DAG scheduler for gap-research tasks."""

    def __init__(self, tasks: Optional[List[ResearchTask]] = None):
        self._tasks: Dict[str, ResearchTask] = {}
        for t in tasks or []:
            self.add(t)

    def add(self, task: ResearchTask) -> ResearchTask:
        if task.task_id in self._tasks:
            raise ValueError(f"duplicate task id: {task.task_id}")
        self._tasks[task.task_id] = task
        return task

    def tasks(self) -> List[ResearchTask]:
        return [self._tasks[k] for k in sorted(self._tasks)]

    def __len__(self) -> int:
        return len(self._tasks)

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Unknown dependencies and cycles are hard errors."""
        for t in self._tasks.values():
            for dep in t.depends_on:
                if dep not in self._tasks:
                    raise ValueError(
                        f"task {t.task_id} depends on unknown task {dep}")
        # Kahn's algorithm for cycle detection
        indegree = {tid: 0 for tid in self._tasks}
        for t in self._tasks.values():
            for _ in t.depends_on:
                indegree[t.task_id] += 1
        queue = [tid for tid, d in sorted(indegree.items()) if d == 0]
        seen = 0
        dependents: Dict[str, List[str]] = {tid: [] for tid in self._tasks}
        for t in self._tasks.values():
            for dep in t.depends_on:
                dependents[dep].append(t.task_id)
        while queue:
            tid = queue.pop(0)
            seen += 1
            for child in sorted(dependents[tid]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if seen != len(self._tasks):
            cyclic = sorted(tid for tid, d in indegree.items() if d > 0)
            raise ValueError(f"task dependency cycle involving: {cyclic}")

    # -- execution ---------------------------------------------------------

    def run(
        self,
        execute: Callable[[ResearchTask], Any],
        parallel_max_workers: Optional[int] = None,
        stage_cap: Optional[int] = None,
        limiter=None,
    ) -> Dict[str, Any]:
        """Execute all tasks respecting dependencies and worker limits.

        - ``execute(task)`` runs each task; its return value lands in
          the result dict. An exception marks the task ``failed`` and
          SKIPS its dependents (marked failed with a dependency error) —
          the scheduler itself never raises for task errors.
        - ``limiter`` (RunLimits/ConcurrencyLimiter/None) is acquired as
          a leaf permit around execute() only.
        """
        self.validate()
        results: Dict[str, Any] = {}
        lock = threading.Lock()
        done = threading.Condition(lock)
        finished: Dict[str, bool] = {}      # task_id -> success
        remaining = set(self._tasks)

        workers = effective_workers(parallel_max_workers, stage_cap,
                                    len(self._tasks))

        def _ready() -> List[ResearchTask]:
            out = []
            for tid in sorted(remaining):
                t = self._tasks[tid]
                if all(dep in finished for dep in t.depends_on):
                    out.append(t)
            return out

        def _run_one(task: ResearchTask):
            failed_dep = next((d for d in task.depends_on
                               if finished.get(d) is False), None)
            if failed_dep is not None:
                task.status = "failed"
                task.error = f"dependency failed: {failed_dep}"
                return task.task_id, None, False
            task.status = "running"
            try:
                with maybe_permit(limiter):
                    value = execute(task)
                task.status = "done"
                return task.task_id, value, True
            except Exception as e:      # task errors never kill the DAG
                task.status = "failed"
                task.error = str(e)
                return task.task_id, None, False

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers) as pool:
            in_flight: Dict[concurrent.futures.Future, str] = {}
            with lock:
                for task in _ready():
                    remaining.discard(task.task_id)
                    in_flight[pool.submit(_run_one, task)] = task.task_id
            while in_flight:
                completed = next(concurrent.futures.as_completed(
                    list(in_flight)))
                in_flight.pop(completed)
                tid, value, ok = completed.result()
                with lock:
                    finished[tid] = ok
                    results[tid] = value
                    for task in _ready():
                        remaining.discard(task.task_id)
                        in_flight[pool.submit(_run_one, task)] = \
                            task.task_id
        return results
