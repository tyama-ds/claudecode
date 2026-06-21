"""Adapters that talk to LLM APIs through their official SDKs.

- :class:`AnthropicAPIAdapter` uses the official ``anthropic`` SDK.
- :class:`OpenAIAPIAdapter` uses the official ``openai`` SDK, and — by pointing
  ``base_url`` at a local OpenAI-compatible server (Ollama, LM Studio, vLLM) —
  also powers local-LLM participants.

The SDKs are imported lazily inside the methods, so the rest of the package
(and the mock / CLI adapters) work even when they are not installed.
"""

from __future__ import annotations

from typing import List, Optional

from ..config import get_settings
from .base import AgentAdapter, Message


def _to_role_dicts(system: Optional[str], history: List[Message], prompt: str) -> list:
    """Build an OpenAI-style ``messages`` list (system handled separately for Anthropic)."""
    msgs: list = []
    for m in history:
        role = m.role if m.role in ("user", "assistant") else "user"
        msgs.append({"role": role, "content": m.content})
    msgs.append({"role": "user", "content": prompt})
    return msgs


class AnthropicAPIAdapter(AgentAdapter):
    """Claude via ``anthropic`` SDK (``client.messages``)."""

    kind = "anthropic"

    def __init__(self, name: str = "claude_api", display_name: Optional[str] = None,
                 model: Optional[str] = None):
        super().__init__(name, display_name or "Claude (API)")
        self.model = model or get_settings().anthropic_model

    def available(self) -> "tuple[bool, str]":
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "anthropic SDK not installed (pip install anthropic)"
        if not get_settings().anthropic_api_key:
            return False, "ANTHROPIC_API_KEY not set"
        return True, ""

    def _generate(self, prompt: str, system: Optional[str], history: List[Message]) -> str:
        import anthropic

        client = anthropic.Anthropic()
        messages = _to_role_dicts(None, history, prompt)
        settings = get_settings()
        # Stream and collect the final message: this is the SDK-recommended way
        # to avoid HTTP timeouts on longer generations.
        with client.messages.stream(
            model=self.model,
            max_tokens=settings.max_tokens,
            system=system or "",
            messages=messages,
        ) as stream:
            final = stream.get_final_message()
        return "".join(b.text for b in final.content if getattr(b, "type", None) == "text")


class OpenAIAPIAdapter(AgentAdapter):
    """GPT (or any OpenAI-compatible endpoint) via the ``openai`` SDK.

    Set ``base_url`` to a local server to use this for local LLMs.
    """

    kind = "openai"

    def __init__(
        self,
        name: str = "openai_api",
        display_name: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        local: bool = False,
    ):
        self.local = local
        settings = get_settings()
        super().__init__(name, display_name or ("Local LLM" if local else "GPT (API)"))
        self.model = model or (settings.local_model if local else settings.openai_model)
        self.base_url = base_url or (settings.local_base_url if local else None)

    def available(self) -> "tuple[bool, str]":
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, "openai SDK not installed (pip install openai)"
        if not self.local and not get_settings().openai_api_key:
            return False, "OPENAI_API_KEY not set"
        # For local endpoints we cannot cheaply verify reachability here; the
        # call itself will surface a clear error if the server is down.
        return True, ""

    def _generate(self, prompt: str, system: Optional[str], history: List[Message]) -> str:
        from openai import OpenAI

        kwargs = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.local:
            # Local servers usually ignore the key but the SDK requires one.
            kwargs["api_key"] = "local"
        client = OpenAI(**kwargs)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(_to_role_dicts(None, history, prompt))

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=get_settings().max_tokens,
        )
        return resp.choices[0].message.content or ""
