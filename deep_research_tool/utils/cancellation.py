"""
Run-level cancellation primitives shared by the pipeline layers.

``RunCancelled`` is raised at SAFE checkpoints after the user cancelled a
run (GUI Stop / Web UI cancel). It lives here (not in main.py) so the
researcher, crawlers and extractors can honor the token without a
circular import. main.py re-exports it under the same name.

Contract:
- ``cancel_check`` callables raise RunCancelled when the run's cancel
  event is set and return None otherwise; they are cheap and may be
  called before EVERY new unit of work (search, page fetch, LLM call).
- After a cancel is accepted no NEW work starts; work already in flight
  finishes (or fails) and its result is kept. Callers distinguish
  "cancel accepted, waiting for in-flight requests" from "stopped".
"""

import threading
from typing import Callable, Optional


class RunCancelled(Exception):
    """Raised at a safe checkpoint after the user cancelled the run.

    Carries the stage name; the partial artifacts produced so far stay
    on disk / in the live sink and are surfaced as a CANCELLED result —
    never as a normal completion.
    """


class CancelToken:
    """Thread-safe cancel flag with an optional acknowledgement counter.

    ``check()`` raises RunCancelled when set. ``in_flight`` lets a caller
    report how many started requests are still being waited for after
    the cancel was accepted (shown in the UI as "終了待ち").
    """

    def __init__(self, event: Optional[threading.Event] = None):
        self.event = event or threading.Event()
        self._lock = threading.Lock()
        self._in_flight = 0
        self.accepted_at: Optional[float] = None

    def set(self) -> None:
        import time
        with self._lock:
            if self.accepted_at is None:
                self.accepted_at = time.time()
        self.event.set()

    def is_set(self) -> bool:
        return self.event.is_set()

    def check(self) -> None:
        if self.event.is_set():
            raise RunCancelled("cancelled by user")

    # in-flight accounting (leaf operations that already started)
    def enter(self) -> None:
        with self._lock:
            self._in_flight += 1

    def leave(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight


def make_cancel_check(token_or_event) -> Callable[[], None]:
    """Adapt a CancelToken / threading.Event / callable into a
    ``cancel_check()`` callable (None -> no-op)."""
    if token_or_event is None:
        return lambda: None
    if isinstance(token_or_event, CancelToken):
        return token_or_event.check
    if isinstance(token_or_event, threading.Event):
        def _check():
            if token_or_event.is_set():
                raise RunCancelled("cancelled by user")
        return _check
    if callable(token_or_event):
        return token_or_event
    return lambda: None
