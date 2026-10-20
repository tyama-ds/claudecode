"""
Delegated LLM clients — a LOCAL server used for one role or stage inside
a run whose main provider is something else (⚙設定 → ロールルーティング,
or 工程別のAI切り替え = ローカル).

Why a wrapper instead of handing the raw LocalLLMClient around:

- every error names the role and the setting that caused the local call,
  so "Local LLM API error" never appears without its reason;
- after the first failure the role falls back to the MAIN client for the
  rest of the run (``fallback_enabled``), so an auxiliary local server
  can never abort a multi-hour run — the switch is recorded and reported;
  the switch is shared by every role that uses the same server, so a
  dead server is waited on once, not once per role;
- ``preflight()`` sends ONE tiny request before the run starts, so a dead
  or misconfigured local server is reported at start, not hours later;
- ``reset_run_state()`` re-arms the local route for the next run of a
  reused tool instance.

The wrapper only ever talks to the server the user configured; it never
invents an endpoint.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional


class DelegatedLLMUnavailable(RuntimeError):
    """The preflight could not get an answer from a delegated local server."""

    def __init__(self, message: str, *, role_label: str = "",
                 setting_label: str = "", base_url: str = "",
                 model: str = "", cause: str = ""):
        super().__init__(message)
        self.role_label = role_label
        self.setting_label = setting_label
        self.base_url = base_url
        self.model = model
        self.cause = cause


class _ServerState:
    """Trip state shared by every wrapper of the same local client."""

    def __init__(self):
        self.lock = threading.Lock()
        self.tripped = False


def _is_cancellation(exc: BaseException) -> bool:
    """Run / verification cancellations must never be turned into a
    fallback or a prefixed error."""
    try:
        from ..utils.cancellation import RunCancelled
        if isinstance(exc, RunCancelled):
            return True
    except Exception:
        pass
    try:
        from ..verification.runtime import VerificationCancelled
        if isinstance(exc, VerificationCancelled):
            return True
    except Exception:
        pass
    return False


def provider_name(client: Any) -> str:
    """Best-effort provider name of a raw client (by class)."""
    if isinstance(client, DelegatedLLMClient):
        return "local"
    name = type(client).__name__.lower()
    for key in ("local", "openai", "anthropic"):
        if key in name:
            return key
    return type(client).__name__


class DelegatedLLMClient:
    """Routes one role's LLM calls to a local client, with a recorded
    fallback to the main client.

    Attribute reads that the wrapper does not define (``model``,
    ``max_tokens``, ``base_url``, ``backend``, ``max_concurrency`` …) are
    served by the local client; the run-scoped wiring DeepResearchTool
    sets on every client (``concurrency_limiter``, ``token_stats``,
    ``cancel_check``) is forwarded to it as well.
    """

    _FORWARDED = ("concurrency_limiter", "token_stats", "cancel_check",
                  "retry_attempts", "retry_backoff_seconds")

    def __init__(self, local: Any, *, fallback: Any = None,
                 role_label: str, setting_label: str,
                 fallback_enabled: bool = True,
                 on_fallback: Optional[Callable[[str], None]] = None):
        object.__setattr__(self, "_local", local)
        self._fallback = fallback
        self.role_label = role_label
        self.setting_label = setting_label
        self.fallback_enabled = bool(fallback_enabled) and fallback is not None
        self._on_fallback = on_fallback
        self._lock = threading.Lock()
        # one trip flag per SERVER: two roles on the same local client
        # (role=all) switch over together after the first failure
        state = getattr(local, "_delegation_state", None)
        if not isinstance(state, _ServerState):
            state = _ServerState()
            try:
                setattr(local, "_delegation_state", state)
            except Exception:
                pass
        self._state = state
        self.failures = 0
        self.local_calls = 0
        self.fallback_calls = 0
        self.events: List[Dict[str, Any]] = []
        self.preflight_result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # transparent attribute access
    # ------------------------------------------------------------------
    @property
    def delegate(self) -> Any:
        """The wrapped local client."""
        return self._local

    @property
    def fallback(self) -> Any:
        return self._fallback

    @property
    def tripped(self) -> bool:
        """True once this server failed in this run and the role switched
        to the main client (shared with every wrapper of the server)."""
        return self._state.tripped

    def __getattr__(self, name: str) -> Any:
        # only reached when normal lookup fails; never recurse on privates
        if name.startswith("_"):
            raise AttributeError(name)
        local = object.__getattribute__(self, "_local")
        value = getattr(local, name)
        fallback = object.__getattribute__(self, "_fallback")
        # public LLM-call helpers (analyze_research_query, …) must go
        # through the fallback logic too, not straight to the local client
        if callable(value) and fallback is not None and \
                callable(getattr(fallback, name, None)):
            return lambda *a, **kw: self._call(name, *a, **kw)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._FORWARDED:
            # run-scoped wiring belongs to the local client; the main
            # client is wired by the tool itself
            try:
                setattr(object.__getattribute__(self, "_local"), name, value)
            except Exception:
                pass
            return
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # calls
    # ------------------------------------------------------------------
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 **kwargs) -> Any:
        return self._call("generate", prompt, system_prompt=system_prompt,
                          **kwargs)

    def chat(self, messages: Any, system_prompt: Optional[str] = None,
             **kwargs) -> Any:
        return self._call("chat", messages, system_prompt=system_prompt,
                          **kwargs)

    def _prefix(self, exc: BaseException) -> str:
        return (f"[ローカルLLM委譲: {self.role_label}（{self.setting_label}）] "
                f"{exc}")

    def _call(self, method: str, *args, **kwargs) -> Any:
        if not self._state.tripped:
            with self._lock:
                self.local_calls += 1
            try:
                return getattr(self._local, method)(*args, **kwargs)
            except Exception as e:
                if _is_cancellation(e):
                    raise
                self._record_failure(e)
                if not self.fallback_enabled:
                    raise RuntimeError(self._prefix(e)) from e
        with self._lock:
            self.fallback_calls += 1
        return getattr(self._fallback, method)(*args, **kwargs)

    def _record_failure(self, exc: BaseException) -> None:
        with self._lock:
            self.failures += 1
            self.events.append({
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "error": str(exc)[:300],
                "action": "fallback" if self.fallback_enabled else "raise",
            })
        first = False
        if self.fallback_enabled:
            with self._state.lock:
                if not self._state.tripped:
                    self._state.tripped = True
                    first = True
        if first:
            fb = provider_name(self._fallback)
            message = (
                f"ローカルLLMへ委譲した{self.role_label}が失敗したため、"
                f"以後はメインのプロバイダ（{fb}）で続行します"
                f"（{self.setting_label}）。原因: {str(exc)[:200]}")
            print(f"[RoleLLM] {message}")
            try:
                from ..utils.helpers import ResearchWarnings
                ResearchWarnings.get_instance().add(
                    ResearchWarnings.HIGH, "LocalLLM", message)
            except Exception:
                pass
            if self._on_fallback is not None:
                try:
                    self._on_fallback(message)
                except Exception:
                    pass

    def reset_run_state(self) -> None:
        """Re-arm the local route for a new run of the same tool instance
        (the server is checked again by the preflight)."""
        with self._lock:
            self.failures = 0
            self.local_calls = 0
            self.fallback_calls = 0
            self.events = []
            self.preflight_result = None
        with self._state.lock:
            self._state.tripped = False

    # ------------------------------------------------------------------
    # preflight
    # ------------------------------------------------------------------
    def preflight(self) -> Dict[str, Any]:
        """One tiny request to the delegated server, with the CONFIGURED
        timeout (a server that cold-loads its model for minutes is given
        the time the user set). Raises DelegatedLLMUnavailable (with the
        setting that caused the delegation) when it cannot answer."""
        local = self._local
        started = time.time()
        saved_tokens = getattr(local, "max_tokens", None)
        timeout = getattr(local, "timeout", None)
        try:
            try:
                local.max_tokens = 8
            except Exception:
                pass
            local.generate("ping", system_prompt="Reply with the single word OK.")
        except Exception as e:
            if _is_cancellation(e):
                raise
            seconds = round(time.time() - started, 2)
            self.preflight_result = {"ok": False, "seconds": seconds,
                                     "error": str(e)[:300]}
            base_url = str(getattr(local, "base_url", "") or "")
            model = str(getattr(local, "model", "") or "")
            waited = ""
            if isinstance(timeout, (int, float)) and "inactivity" in str(e):
                waited = (f"応答待ちは ⚙設定 → ローカルLLM → タイムアウト"
                          f"（{timeout:g} 秒）で打ち切りました。")
            message = (
                f"ローカルLLMへの委譲（{self.role_label}: {self.setting_label}）"
                f"の疎通確認に失敗しました: {base_url}"
                f"{'（モデル ' + model + '）' if model else ''} — {str(e)[:200]}。"
                f"{waited}"
                "⚙設定 → ローカルLLM のサーバーURL・バックエンド・モデル名を確認するか、"
                "委譲をやめる場合はロールルーティングを「オフ」"
                "（工程別のAI切り替えは「既定」）にしてください。"
                "調査は開始していません。")
            raise DelegatedLLMUnavailable(
                message, role_label=self.role_label,
                setting_label=self.setting_label, base_url=base_url,
                model=model, cause=str(e)[:300]) from e
        finally:
            try:
                if saved_tokens is not None:
                    local.max_tokens = saved_tokens
            except Exception:
                pass
        self.preflight_result = {"ok": True,
                                 "seconds": round(time.time() - started, 2)}
        return self.preflight_result

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        local = self._local
        return {
            "role": self.role_label,
            "setting": self.setting_label,
            "provider": "local",
            "base_url": getattr(local, "base_url", None),
            "model": getattr(local, "model", None),
            "backend": getattr(local, "backend", None),
            "fallback_enabled": self.fallback_enabled,
            "fallback_provider": (provider_name(self._fallback)
                                  if self._fallback is not None else None),
            "tripped": self.tripped,
            "failures": self.failures,
            "local_calls": self.local_calls,
            "fallback_calls": self.fallback_calls,
            "preflight": self.preflight_result,
            "events": list(self.events[-5:]),
        }
