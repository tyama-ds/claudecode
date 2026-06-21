"""Session state and an in-memory session registry."""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..adapters.base import AgentAdapter
from .events import Event, EventBus

# A line that contributes a durable note to the shared scratchpad, e.g.
#   NOTE: use a set for O(1) membership checks
# Optional leading bullet ("- " / "* ") is tolerated; "NOTE" is case-insensitive.
_NOTE_RE = re.compile(r"^\s*(?:[-*]\s*)?note\s*:\s*(.+?)\s*$", re.IGNORECASE)


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
class Note:
    """A single entry on the shared scratchpad."""

    author: str
    text: str


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
    scratchpad: List[Note] = field(default_factory=list)  # shared blackboard
    stop_requested: bool = False

    # -- helpers used by the engine/strategies ----------------------------

    def emit(self, type: str, **data) -> None:
        self.bus.publish(Event(type=type, data=data))

    def record(self, turn: Turn) -> None:
        self.transcript.append(turn)

    def should_stop(self) -> bool:
        return self.stop_requested

    # -- shared scratchpad (blackboard) -----------------------------------

    def render_scratchpad(self) -> str:
        """Render the scratchpad for inclusion in an agent's prompt."""
        if not self.scratchpad:
            return "(empty)"
        return "\n".join(f"- ({n.author}) {n.text}" for n in self.scratchpad)

    def absorb_notes(self, author: str, text: str) -> int:
        """Extract ``NOTE:`` lines from an agent's reply onto the scratchpad.

        Returns how many notes were added (duplicates of the latest identical
        text are skipped so re-stated notes don't pile up).
        """
        added = 0
        existing = {(n.text) for n in self.scratchpad}
        for line in text.splitlines():
            m = _NOTE_RE.match(line)
            if not m:
                continue
            note_text = m.group(1).strip()
            if note_text and note_text not in existing:
                self.scratchpad.append(Note(author=author, text=note_text))
                existing.add(note_text)
                added += 1
        return added

    def scratchpad_view(self) -> List[dict]:
        return [{"author": n.author, "text": n.text} for n in self.scratchpad]


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
