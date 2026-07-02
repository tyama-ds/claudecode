"""The agent adapter abstraction.

Every backend — a mock, a coding-agent CLI, or an LLM API — implements the same
small interface so the orchestrator can treat them uniformly. This is what makes
the system extensible: adding a new participant (e.g. another local LLM) means
writing one ``AgentAdapter`` subclass.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Message:
    """A single chat message in an agent's conversation history."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class AgentResult:
    """The outcome of a single agent turn."""

    text: str
    ok: bool = True
    error: Optional[str] = None
    duration: float = 0.0
    meta: dict = field(default_factory=dict)


class AgentAdapter(ABC):
    """Base class for all agent backends."""

    #: The :class:`~agent_orchestrator.config.AdapterKind` value, set by subclasses.
    kind: str = "base"

    #: Whether this backend can use a structured conversation ``history`` (a list
    #: of messages). When False, the orchestrator falls back to embedding the
    #: transcript as text in the prompt. Backends that ignore history set this
    #: False so the fallback path is used instead.
    supports_history: bool = True

    #: When set (workspace strategies only), the backend works *inside* this
    #: directory. CLI backends run there and edit files natively with their own
    #: tools; other backends ignore it and use the <FILE> protocol instead.
    workdir: Optional[str] = None

    def __init__(self, name: str, display_name: Optional[str] = None):
        self.name = name
        self.display_name = display_name or name

    # -- interface ---------------------------------------------------------

    @abstractmethod
    def _generate(
        self,
        prompt: str,
        system: Optional[str],
        history: List[Message],
    ) -> str:
        """Produce a single response. Subclasses implement this.

        Should return the assistant's text, or raise on failure.
        """

    def available(self) -> "tuple[bool, str]":
        """Return ``(is_available, reason)``.

        Adapters that depend on an external CLI, SDK, or API key override this
        to report why they cannot run, so the UI can disable them gracefully.
        """
        return True, ""

    # -- public, timed wrapper --------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        history: Optional[List[Message]] = None,
    ) -> AgentResult:
        """Run :meth:`_generate`, timing it and capturing errors."""
        start = time.monotonic()
        try:
            text = self._generate(prompt, system, history or [])
            return AgentResult(text=text.strip(), ok=True, duration=time.monotonic() - start)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
            return AgentResult(
                text="",
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                duration=time.monotonic() - start,
            )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} kind={self.kind!r}>"
