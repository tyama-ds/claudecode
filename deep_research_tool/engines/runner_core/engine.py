"""Engine interface と実行コンテキスト（deep-research-orchestrator と同一）。

Python 3.10 でも動くよう ``datetime.UTC`` → ``timezone.utc``、
``TimeoutError`` → ``asyncio.TimeoutError`` に置き換えている（3.11+ では同じ型）。
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import EngineCapabilities, RunEvent, RunRequest, RunResult


class CancelledByUser(Exception):
    """ユーザー起点のキャンセル。"""


class RunContext:
    """エンジン実装がイベントを発行し、キャンセルを検知するためのコンテキスト。"""

    def __init__(self, run_id: str, request: RunRequest):
        self.run_id = run_id
        self.request = request
        self._events: List[RunEvent] = []
        self._seq = 0
        self._cancel_event = asyncio.Event()
        self._lock = asyncio.Lock()

    def emit(self, type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self._seq += 1
        self._events.append(
            RunEvent(
                seq=self._seq,
                type=type,
                payload=payload or {},
                ts=datetime.now(timezone.utc).isoformat(),
            )
        )

    @property
    def events(self) -> List[RunEvent]:
        return self._events

    @property
    def last_seq(self) -> int:
        return self._seq

    def request_cancel(self) -> None:
        self._cancel_event.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise CancelledByUser()

    async def sleep(self, seconds: float) -> None:
        """キャンセル可能な sleep。"""
        try:
            await asyncio.wait_for(self._cancel_event.wait(), timeout=seconds)
            raise CancelledByUser()
        except asyncio.TimeoutError:
            return


class Engine(ABC):
    engine_id: str

    @abstractmethod
    def capabilities(self) -> EngineCapabilities: ...

    @abstractmethod
    async def run(self, ctx: RunContext) -> RunResult: ...
