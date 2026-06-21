"""Session state and an in-memory session registry."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..adapters.base import AgentAdapter
from .events import Event, EventBus


@dataclass
class Turn:
    """One recorded agent turn in the transcript."""

    agent: str
    role: str
    round: int
    content: str
    ok: bool = True
    error: Optional[str] = None
    duration: float = 0.0


@dataclass
class Session:
    """A single collaboration run."""

    id: str
    task: str
    strategy: str
    rounds: int
    agents: Dict[str, AgentAdapter]  # role -> adapter
    bus: EventBus = field(default_factory=EventBus)
    status: str = "pending"  # pending | running | done | error | stopped
    transcript: List[Turn] = field(default_factory=list)
    stop_requested: bool = False

    # -- helpers used by the engine/strategies ----------------------------

    def emit(self, type: str, **data) -> None:
        self.bus.publish(Event(type=type, data=data))

    def record(self, turn: Turn) -> None:
        self.transcript.append(turn)

    def should_stop(self) -> bool:
        return self.stop_requested


class SessionManager:
    """Thread-safe registry of active sessions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, Session] = {}

    def create(
        self,
        task: str,
        strategy: str,
        rounds: int,
        agents: Dict[str, AgentAdapter],
    ) -> Session:
        session = Session(
            id=uuid.uuid4().hex[:12],
            task=task,
            strategy=strategy,
            rounds=rounds,
            agents=agents,
        )
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)

    def all(self) -> List[Session]:
        with self._lock:
            return list(self._sessions.values())
