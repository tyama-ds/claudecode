"""
Verification runtime plumbing: cache, progress, cancellation.

- VerificationCache: same-run memoization of claim extraction, claim
  judgements and coverage windows. Keys always include the MODEL name
  and PROMPT_VERSION, so a model switch or prompt change can never
  serve stale results. Unchanged sections/claims are not re-verified.
- VerificationProgress: thread-safe live state for the GUI — phase,
  per-phase and overall progress, rounds, counters, LLM calls, cache
  hits, retries, elapsed / estimated remaining time, waiting states
  (LLM response / rate limit / retry / timeout) and cancellation.
  Snapshots contain NO api keys and NO prompt bodies.
- VerificationCancelled / VerificationTimeout: raised at safe
  boundaries (between chunks, batches, retries, research and revision
  steps); no new LLM request starts after cancellation.
"""

import hashlib
import threading
import time
from typing import Any, Dict, Optional

# bump when verification prompt wording changes (cache invalidation)
PROMPT_VERSION = "v3"

PHASES = (
    "idle",              # 待機
    "extracting",        # クレーム抽出中
    "judging",           # クレーム検証中
    "coverage",          # カバレッジ確認中
    "researching",       # 不足エビデンス調査中
    "revising",          # セクション修正中
    "final_check",       # 最終検証中
    "done",              # 完了
    "cancelled",         # キャンセル
    "error",             # エラー
)

PHASE_LABELS_JA = {
    "idle": "待機中", "extracting": "クレーム抽出中",
    "judging": "クレーム検証中", "coverage": "カバレッジ確認中",
    "researching": "不足エビデンス調査中", "revising": "セクション修正中",
    "final_check": "最終検証中", "done": "完了",
    "cancelled": "キャンセル", "error": "エラー",
}

WAIT_LABELS_JA = {
    "llm": "LLM応答待ち", "rate_limit": "レート制限待ち",
    "retry": "リトライ待ち", "timeout": "タイムアウト処理中",
}


class VerificationCancelled(Exception):
    """Raised at a safe boundary after the user requested cancellation."""


class VerificationTimeout(Exception):
    """Raised at a safe boundary when the verification timeout elapsed."""


def stable_hash(*parts) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:32]


class VerificationCache:
    """Thread-safe same-run cache with hit/miss accounting."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._store: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        if not self.enabled:
            return None
        with self._lock:
            if key in self._store:
                self.hits += 1
                return self._store[key]
            self.misses += 1
            return None

    def put(self, key: str, value) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._store[key] = value

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class VerificationProgress:
    """Thread-safe verification progress for live UI polling."""

    def __init__(self):
        self._lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.started_at: Optional[float] = None
        self.deadline: Optional[float] = None
        self._d: Dict[str, Any] = {
            "phase": "idle",
            "round": 0,
            "max_rounds": 0,
            "claims_done": 0,
            "claims_total": 0,
            "chunks_done": 0,
            "chunks_total": 0,
            "llm_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "retries": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "current_label": "",
            "profile": "",
        }
        self._waiting: Optional[str] = None
        self._waiting_since: Optional[float] = None

    # -- lifecycle ------------------------------------------------------

    def start(self, profile: str, max_rounds: int,
              timeout_seconds: int = 0) -> None:
        with self._lock:
            self.started_at = time.time()
            self.deadline = (self.started_at + timeout_seconds
                             if timeout_seconds else None)
            self._d["profile"] = profile
            self._d["max_rounds"] = max_rounds
            self._d["phase"] = "extracting"

    def cancel(self) -> None:
        self.cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def checkpoint(self) -> None:
        """Called at every safe boundary (chunk / batch / retry / round).

        Raises VerificationCancelled or VerificationTimeout so no new
        LLM request starts afterwards."""
        if self.cancel_event.is_set():
            self.set_phase("cancelled")
            raise VerificationCancelled()
        if self.deadline and time.time() > self.deadline:
            self.set_waiting("timeout")
            raise VerificationTimeout()

    # -- mutation -------------------------------------------------------

    def set_phase(self, phase: str, label: str = "") -> None:
        with self._lock:
            self._d["phase"] = phase
            self._d["current_label"] = label
            self._waiting = None

    def set_round(self, round_no: int) -> None:
        with self._lock:
            self._d["round"] = round_no

    def set_counts(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if key in self._d:
                    self._d[key] = value

    def add(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if key in self._d:
                    self._d[key] += value

    def set_label(self, label: str) -> None:
        with self._lock:
            self._d["current_label"] = label

    def set_waiting(self, kind: Optional[str]) -> None:
        """kind: llm / rate_limit / retry / timeout / None."""
        with self._lock:
            self._waiting = kind
            self._waiting_since = time.time() if kind else None

    def sync_cache(self, cache: VerificationCache) -> None:
        with self._lock:
            self._d["cache_hits"] = cache.hits
            self._d["cache_misses"] = cache.misses

    # -- reading --------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """UI-safe snapshot: metrics only, never keys or prompt text."""
        with self._lock:
            d = dict(self._d)
            now = time.time()
            d["elapsed_seconds"] = round(now - self.started_at, 1) \
                if self.started_at else 0.0
            d["phase_label"] = PHASE_LABELS_JA.get(d["phase"], d["phase"])
            hits, misses = d["cache_hits"], d["cache_misses"]
            d["cache_hit_rate"] = round(hits / (hits + misses), 3) \
                if (hits + misses) else 0.0
            # estimated remaining from claim throughput of this run
            eta = None
            if d["claims_done"] and d["claims_total"] and self.started_at:
                rate = d["claims_done"] / max(now - self.started_at, 0.001)
                remaining = max(d["claims_total"] - d["claims_done"], 0)
                # remaining rounds repeat roughly the same work
                rounds_left = max(d["max_rounds"] - d["round"], 0)
                eta = remaining / max(rate, 0.001) \
                    + rounds_left * (d["claims_total"] / max(rate, 0.001)) * 0.3
            d["eta_seconds"] = round(eta, 1) if eta is not None else None
            # stall visibility: >30s in one waiting state is surfaced
            if self._waiting and self._waiting_since and \
                    now - self._waiting_since >= 30:
                d["waiting"] = self._waiting
                d["waiting_label"] = WAIT_LABELS_JA.get(
                    self._waiting, self._waiting)
                d["waiting_seconds"] = round(now - self._waiting_since, 1)
            else:
                d["waiting"] = None
            d["cancelled"] = self.cancel_event.is_set()
            return d
