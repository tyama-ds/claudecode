"""Agent Orchestrator.

An orchestrator (and web UI) that makes two coding agents — **Codex** and
**Claude Code** — collaborate on a task, with optional **local LLM** participants.

The package is intentionally dependency-light: the orchestration core and the
web server use only the Python standard library, so it runs out of the box.
The API-backed adapters use the official ``anthropic`` / ``openai`` SDKs and are
imported lazily, so they are only required when you actually select them.

Layout
------
- ``config``        : enums and runtime settings.
- ``adapters``      : pluggable agent backends (mock / CLI / API / local LLM).
- ``orchestrator``  : the collaboration engine, strategies, sessions, events.
- ``server``        : a stdlib HTTP + Server-Sent-Events web UI.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
