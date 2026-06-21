"""LLM API adapters using only the Python standard library (``urllib``).

Deliberately SDK-free: on a locked-down / audited machine this needs no
``pip install`` and pulls in no compiled native extensions — it is plain text
Python calling the providers' REST endpoints directly. The only requirement is
an API key (and, for local models, a running OpenAI-compatible server).

- :class:`AnthropicAPIAdapter` → Claude Messages API.
- :class:`OpenAIAPIAdapter` → OpenAI (or any OpenAI-compatible) chat endpoint;
  point ``base_url`` at a local server (Ollama / LM Studio) for local LLMs.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import List, Optional

from ..config import get_settings
from .base import AgentAdapter, Message

_API_TIMEOUT = 300  # seconds


def _post_json(url: str, headers: dict, payload: dict) -> dict:
    """POST JSON and return the parsed JSON response, with readable errors."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={**headers, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_API_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"connection failed: {exc.reason}") from exc


def _chat_messages(history: List[Message], prompt: str) -> list:
    msgs = [
        {"role": m.role if m.role in ("user", "assistant") else "user", "content": m.content}
        for m in history
    ]
    msgs.append({"role": "user", "content": prompt})
    return msgs


class AnthropicAPIAdapter(AgentAdapter):
    """Claude via the Messages REST API (``POST /v1/messages``)."""

    kind = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"
    VERSION = "2023-06-01"

    def __init__(self, name: str = "claude_api", display_name: Optional[str] = None,
                 model: Optional[str] = None):
        super().__init__(name, display_name or "Claude (API)")
        self.model = model or get_settings().anthropic_model

    def available(self) -> "tuple[bool, str]":
        if not get_settings().anthropic_api_key:
            return False, "ANTHROPIC_API_KEY not set"
        return True, ""

    def _generate(self, prompt: str, system: Optional[str], history: List[Message]) -> str:
        settings = get_settings()
        payload = {
            "model": self.model,
            "max_tokens": settings.max_tokens,
            "messages": _chat_messages(history, prompt),
        }
        if system:
            payload["system"] = system
        headers = {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": self.VERSION,
        }
        data = _post_json(self.URL, headers, payload)
        blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if not text:
            raise RuntimeError(f"no text in response (stop_reason={data.get('stop_reason')})")
        return text


class OpenAIAPIAdapter(AgentAdapter):
    """OpenAI-compatible chat completions (``POST /v1/chat/completions``).

    With ``local=True`` this targets a local server (default: Ollama), which is
    how local-LLM participants are powered — same wire format, different host.
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
        base = base_url or (settings.local_base_url if local else "https://api.openai.com/v1")
        self.url = base.rstrip("/") + "/chat/completions"

    def available(self) -> "tuple[bool, str]":
        if not self.local and not get_settings().openai_api_key:
            return False, "OPENAI_API_KEY not set"
        return True, ""

    def _generate(self, prompt: str, system: Optional[str], history: List[Message]) -> str:
        settings = get_settings()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(_chat_messages(history, prompt))
        payload = {"model": self.model, "messages": messages, "max_tokens": settings.max_tokens}
        # Local servers usually ignore the key; OpenAI requires it.
        key = "local" if self.local else settings.openai_api_key
        data = _post_json(self.url, {"Authorization": f"Bearer {key}"}, payload)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"no choices in response: {json.dumps(data)[:300]}")
        return choices[0].get("message", {}).get("content", "") or ""
