"""A tiny thread-safe publish/subscribe event bus.

The orchestrator runs a collaboration in a background thread and *publishes*
events as turns happen; the web server *subscribes* (one queue per connected
browser) and forwards events over Server-Sent Events. Pure stdlib.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Event:
    """A single event in a session's lifecycle."""

    type: str  # session_start | turn_start | turn_end | status | error | session_end
    data: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Sentinel published to every subscriber when a session is finished, so SSE
# generators know to close cleanly.
_CLOSED = object()


class EventBus:
    """Fan-out of events to any number of subscriber queues.

    New subscribers receive the full backlog of events first (so a browser that
    connects mid-run still sees the whole transcript), then live events.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: List["queue.Queue"] = []
        self._history: List[Event] = []
        self._closed = False

    def publish(self, event: Event) -> None:
        with self._lock:
            self._history.append(event)
            subscribers = list(self._subscribers)
        for q in subscribers:
            q.put(event)

    def subscribe(self) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue()
        with self._lock:
            for event in self._history:  # replay backlog
                q.put(event)
            if self._closed:
                q.put(_CLOSED)
            else:
                self._subscribers.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue") -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            subscribers = list(self._subscribers)
        for q in subscribers:
            q.put(_CLOSED)

    @staticmethod
    def is_closed_sentinel(item: Any) -> bool:
        return item is _CLOSED

    @property
    def history(self) -> List[Event]:
        with self._lock:
            return list(self._history)
