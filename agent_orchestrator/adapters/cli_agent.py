"""Adapter that drives a local coding-agent CLI via subprocess.

This is the path that makes **Codex** and **Claude Code** collaborate as their
real selves: each turn shells out to the agent's headless/non-interactive mode,
feeds it the constructed prompt on stdin, and captures stdout as the response.

The command is fully configurable; two presets are provided:

- ``claude_code`` -> ``claude -p --output-format text`` (Claude Code CLI)
- ``codex``       -> ``codex exec`` (OpenAI Codex CLI)

Prompts are passed on **stdin** (not argv) to avoid shell-escaping issues and
argv length limits.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import List, Optional, Sequence

from ..config import get_settings
from .base import AgentAdapter, Message


def _render_conversation(prompt: str, system: Optional[str], history: List[Message]) -> str:
    """Flatten system + history + prompt into a single plain-text prompt.

    Coding-agent CLIs take one prompt string, so we serialise the role context
    and prior turns into a readable transcript the agent can act on.
    """
    parts: List[str] = []
    if system:
        parts.append(f"[ROLE]\n{system}")
    for msg in history:
        label = {"assistant": "[PREVIOUS — your turn]",
                 "user": "[CONTEXT]",
                 "system": "[ROLE]"}.get(msg.role, f"[{msg.role.upper()}]")
        parts.append(f"{label}\n{msg.content}")
    parts.append(f"[TASK]\n{prompt}")
    return "\n\n".join(parts)


class CLIAgentAdapter(AgentAdapter):
    """Run a coding-agent CLI non-interactively, one turn at a time."""

    kind = "cli"

    def __init__(
        self,
        name: str,
        command: Sequence[str],
        display_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        super().__init__(name, display_name)
        self.command = list(command)
        self.timeout = timeout if timeout is not None else get_settings().cli_timeout

    @property
    def executable(self) -> str:
        return self.command[0] if self.command else ""

    def available(self) -> "tuple[bool, str]":
        if not self.command:
            return False, "no command configured"
        if shutil.which(self.executable) is None:
            return False, f"'{self.executable}' not found on PATH"
        return True, ""

    def _generate(self, prompt: str, system: Optional[str], history: List[Message]) -> str:
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(reason)
        full_prompt = _render_conversation(prompt, system, history)
        try:
            proc = subprocess.run(
                self.command,
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"timed out after {self.timeout}s") from exc

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"exit {proc.returncode}: {err[:500]}")
        out = (proc.stdout or "").strip()
        if not out:
            raise RuntimeError("empty output from CLI")
        return out


# -- presets ---------------------------------------------------------------

def claude_code_adapter(name: str = "claude_code") -> CLIAgentAdapter:
    """Claude Code CLI in headless print mode."""
    return CLIAgentAdapter(
        name=name,
        display_name="Claude Code",
        command=["claude", "-p", "--output-format", "text"],
    )


def codex_adapter(name: str = "codex") -> CLIAgentAdapter:
    """OpenAI Codex CLI in non-interactive exec mode."""
    return CLIAgentAdapter(
        name=name,
        display_name="Codex",
        command=["codex", "exec"],
    )
