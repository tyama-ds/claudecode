"""LLM API adapters using only the Python standard library (``urllib``).

Deliberately SDK-free: on a locked-down / audited machine this needs no
``pip install`` and pulls in no compiled native extensions — it is plain text
Python calling the providers' REST endpoints directly. The only requirement is
an API key (and, for local models, a running OpenAI-compatible server).

Endpoints, keys, and an optional HTTP(S) proxy are read from
:class:`~agent_orchestrator.config.Settings` at call time, so values entered in
the UI Settings tab take effect immediately.

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


class ApiHTTPError(RuntimeError):
    """An HTTP error from a provider, carrying the status and response body.

    Subclasses RuntimeError so it still surfaces as a normal turn error, while
    letting adapters inspect ``.status`` / ``.body`` to recover (e.g. retry with
    a different parameter name).
    """

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body or ""


def _post_json(url: str, headers: dict, payload: dict, use_proxy: bool = True) -> dict:
    """POST JSON and return the parsed response.

    The configured proxy is applied only when ``use_proxy`` is true, so local-LLM
    calls can opt out and connect directly to a localhost endpoint.
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={**headers, "Content-Type": "application/json"},
    )
    proxy = get_settings().proxy if use_proxy else None
    try:
        if proxy:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy})
            )
            resp = opener.open(req, timeout=_API_TIMEOUT)
        else:
            resp = urllib.request.urlopen(req, timeout=_API_TIMEOUT)
        with resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise ApiHTTPError(exc.code, detail) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"connection failed: {exc.reason}") from exc


def _chat_messages(history: List[Message], prompt: str) -> list:
    msgs = [
        {"role": m.role if m.role in ("user", "assistant") else "user", "content": m.content}
        for m in history
    ]
    msgs.append({"role": "user", "content": prompt})
    return msgs


def _needs_completion_tokens(model: str) -> bool:
    """Whether a model rejects ``max_tokens`` and needs ``max_completion_tokens``.

    Newer OpenAI models — the gpt-5 family and the o-series reasoning models —
    require ``max_completion_tokens``; the 4o / 4.x families still use
    ``max_tokens``. A wrong guess is corrected by the retry in the adapter.
    """
    m = (model or "").lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


class AnthropicAPIAdapter(AgentAdapter):
    """Claude via the Messages REST API (``POST {base}/v1/messages``)."""

    kind = "anthropic"
    VERSION = "2023-06-01"

    def __init__(self, name: str = "claude_api", display_name: Optional[str] = None,
                 model: Optional[str] = None):
        super().__init__(name, display_name or "Claude (API)")
        self.model = model or get_settings().anthropic_model

    def available(self) -> "tuple[bool, str]":
        if not get_settings().anthropic_api_key:
            return False, "no API key (set ANTHROPIC_API_KEY or enter it in Settings)"
        return True, ""

    def _generate(self, prompt: str, system: Optional[str], history: List[Message]) -> str:
        settings = get_settings()
        url = settings.anthropic_base_url.rstrip("/") + "/v1/messages"
        payload = {
            "model": self.model,
            "max_tokens": settings.max_tokens,
            "messages": _chat_messages(history, prompt),
        }
        if system:
            payload["system"] = system
        headers = {"x-api-key": settings.anthropic_api_key, "anthropic-version": self.VERSION}
        data = _post_json(url, headers, payload)
        blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if not text:
            raise RuntimeError(f"no text in response (stop_reason={data.get('stop_reason')})")
        return text


class OpenAIAPIAdapter(AgentAdapter):
    """OpenAI-compatible chat completions (``POST {base}/chat/completions``).

    Works with any OpenAI-compatible provider by setting its base URL: OpenAI,
    Azure OpenAI, Together, Groq, OpenRouter, vLLM, or a local server (Ollama /
    LM Studio) when ``local=True``.
    """

    kind = "openai"

    def __init__(
        self,
        name: str = "openai_api",
        display_name: Optional[str] = None,
        model: Optional[str] = None,
        local: bool = False,
    ):
        self.local = local
        settings = get_settings()
        super().__init__(name, display_name or ("Local LLM" if local else "GPT (API)"))
        self.model = model or (settings.local_model if local else settings.openai_model)

    def available(self) -> "tuple[bool, str]":
        if not self.local and not get_settings().openai_api_key:
            return False, "no API key (set OPENAI_API_KEY or enter it in Settings)"
        return True, ""

    def _generate(self, prompt: str, system: Optional[str], history: List[Message]) -> str:
        settings = get_settings()
        base = settings.local_base_url if self.local else settings.openai_base_url
        url = base.rstrip("/") + "/chat/completions"
        key = settings.local_api_key if self.local else settings.openai_api_key

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(_chat_messages(history, prompt))
        base_payload = {"model": self.model, "messages": messages}

        # Choose the token-limit parameter for the model; if the server rejects
        # it as unsupported, retry with the other name. Covers OpenAI's 4o vs
        # gpt-5/o-series split and any OpenAI-compatible provider.
        if self.local:
            token_params = ["max_tokens"]
        elif _needs_completion_tokens(self.model):
            token_params = ["max_completion_tokens", "max_tokens"]
        else:
            token_params = ["max_tokens", "max_completion_tokens"]

        # Local endpoints connect directly unless the user opts into the proxy.
        use_proxy = settings.local_use_proxy if self.local else True

        last_error = None
        for i, param in enumerate(token_params):
            payload = {**base_payload, param: settings.max_tokens}
            try:
                data = _post_json(url, {"Authorization": f"Bearer {key}"}, payload,
                                  use_proxy=use_proxy)
            except ApiHTTPError as exc:
                last_error = exc
                body = exc.body.lower()
                can_retry = (
                    i + 1 < len(token_params)
                    and exc.status == 400
                    and param in body
                    and ("unsupported" in body or "not supported" in body)
                )
                if can_retry:
                    continue
                raise
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"no choices in response: {json.dumps(data)[:300]}")
            return choices[0].get("message", {}).get("content", "") or ""
        raise last_error  # pragma: no cover - loop always returns or raises above
