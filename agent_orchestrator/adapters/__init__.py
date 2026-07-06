"""Adapter registry: a catalog of selectable backends and a factory.

The web UI asks :func:`catalog` for the list of agents it can offer (and whether
each is currently usable), and the server calls :func:`build_adapter` to turn a
selection from the UI into a live :class:`AgentAdapter`.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .api_agent import AnthropicAPIAdapter, OpenAIAPIAdapter
from .base import AgentAdapter, AgentResult, Message
from .cli_agent import CLIAgentAdapter, claude_code_adapter, codex_adapter
from .mock import MockAdapter

__all__ = [
    "AgentAdapter",
    "AgentResult",
    "Message",
    "MockAdapter",
    "CLIAgentAdapter",
    "AnthropicAPIAdapter",
    "OpenAIAPIAdapter",
    "build_adapter",
    "catalog",
]


# Each catalog entry maps a stable ``id`` (used by the frontend) to a builder.
# ``name`` is the per-agent instance name; builders accept it so two roles can
# use the same backend with distinct labels.
_BUILDERS: Dict[str, Callable[[str], AgentAdapter]] = {
    "mock": lambda name: MockAdapter(name=name, display_name="Mock"),
    "claude_code": claude_code_adapter,
    "codex": codex_adapter,
    "anthropic": lambda name: AnthropicAPIAdapter(name=name),
    "openai": lambda name: OpenAIAPIAdapter(name=name),
    "local": lambda name: OpenAIAPIAdapter(name=name, local=True),
}

# Display metadata for the catalog, in presentation order.
_CATALOG_META = [
    ("claude_code", "Claude Code (CLI)", "Anthropic's `claude` CLI in headless mode."),
    ("codex", "Codex (CLI)", "OpenAI's `codex exec` CLI."),
    ("anthropic", "Claude (API)", "Claude via the official anthropic SDK."),
    ("openai", "GPT (API)", "GPT via the official openai SDK."),
    ("local", "Local LLM (Ollama/LM Studio)", "An OpenAI-compatible local endpoint."),
    ("mock", "Mock (offline)", "Deterministic, no network — for demos and tests."),
]


def build_adapter(spec: dict) -> AgentAdapter:
    """Construct an adapter from a UI selection.

    ``spec`` looks like ``{"id": "claude_code", "name": "implementer",
    "model": "...optional override..."}``.
    """
    adapter_id = spec.get("id", "mock")
    name = spec.get("name", adapter_id)
    if adapter_id not in _BUILDERS:
        raise ValueError(f"unknown adapter id: {adapter_id!r}")
    adapter = _BUILDERS[adapter_id](name)
    # Optional per-selection model override for API adapters.
    model = spec.get("model")
    if model and hasattr(adapter, "model"):
        adapter.model = model
    return adapter


def catalog() -> List[dict]:
    """Return the selectable backends with live availability information."""
    out: List[dict] = []
    for adapter_id, label, desc in _CATALOG_META:
        probe = _BUILDERS[adapter_id]("probe")
        available, reason = probe.available()
        out.append(
            {
                "id": adapter_id,
                "label": label,
                "description": desc,
                "available": available,
                "reason": reason,
            }
        )
    return out
