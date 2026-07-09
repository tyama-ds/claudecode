"""Configuration: enums, defaults, and runtime settings.

Settings are resolved from environment variables (and, optionally, a ``.env``
file in the working directory) so that no extra dependency is required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class AdapterKind(str, Enum):
    """The kind of backend that powers an agent."""

    MOCK = "mock"        # Deterministic, offline. For demos and tests.
    CLI = "cli"          # A local coding-agent CLI (codex / claude) via subprocess.
    ANTHROPIC = "anthropic"  # Claude via the official `anthropic` SDK.
    OPENAI = "openai"        # GPT via the official `openai` SDK.
    LOCAL = "local"          # A local, OpenAI-compatible endpoint (Ollama, LM Studio, ...).


# Default model identifiers. All are overridable per-request or via env vars.
# Anthropic default follows the current most-capable Opus model.
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_LOCAL_MODEL = "llama3.1"
# Ollama exposes an OpenAI-compatible API at this base URL by default.
DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _load_dotenv(path: str = ".env") -> None:
    """Load ``KEY=VALUE`` pairs from a .env file into os.environ if present.

    Existing environment variables are never overwritten. This is a tiny,
    dependency-free substitute for python-dotenv.
    """
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class Settings:
    """Runtime settings for the orchestrator and server."""

    host: str = "127.0.0.1"
    port: int = 8765

    # API keys / models / endpoints (read from the environment; overridable at
    # runtime from the UI Settings tab).
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    local_api_key: str = "local"
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    openai_model: str = DEFAULT_OPENAI_MODEL
    local_model: str = DEFAULT_LOCAL_MODEL
    anthropic_base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    local_base_url: str = DEFAULT_LOCAL_BASE_URL

    # Outbound HTTP(S) proxy applied to API calls (None = direct).
    proxy: Optional[str] = None
    # Whether local-LLM calls go through the proxy too. Local endpoints are
    # usually on localhost, so they default to a direct connection.
    local_use_proxy: bool = False
    # Max simultaneous requests to the local LLM (0 = unlimited). Local servers
    # (Ollama/LM Studio) often serve only a few parallel generations well, so
    # parallel team phases are throttled to this many in-flight calls.
    local_max_concurrency: int = 0

    # User-Agent sent by the http_get / browser_get agent tools.
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )

    # Generation limits.
    max_tokens: int = 4096
    # Per-agent-turn timeout (seconds) for subprocess/CLI adapters.
    cli_timeout: int = 600

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        proxy = (
            os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
            or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
            or os.environ.get("ALL_PROXY") or os.environ.get("all_proxy")
        )
        return cls(
            host=os.environ.get("ORCHESTRATOR_HOST", "127.0.0.1"),
            port=int(os.environ.get("ORCHESTRATOR_PORT", "8765")),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            local_api_key=os.environ.get("LOCAL_LLM_API_KEY", "local"),
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
            openai_model=os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            local_model=os.environ.get("LOCAL_LLM_MODEL", DEFAULT_LOCAL_MODEL),
            anthropic_base_url=os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_ANTHROPIC_BASE_URL),
            openai_base_url=os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
            local_base_url=os.environ.get("LOCAL_LLM_BASE_URL", DEFAULT_LOCAL_BASE_URL),
            proxy=proxy,
            local_use_proxy=os.environ.get("LOCAL_LLM_USE_PROXY", "").strip().lower()
            in ("1", "true", "yes", "on"),
            local_max_concurrency=int(os.environ.get("LOCAL_LLM_MAX_CONCURRENCY", "0") or 0),
            max_tokens=int(os.environ.get("ORCHESTRATOR_MAX_TOKENS", "4096")),
            cli_timeout=int(os.environ.get("ORCHESTRATOR_CLI_TIMEOUT", "600")),
            user_agent=os.environ.get("ORCHESTRATOR_USER_AGENT") or cls.user_agent,
        )

    def apply_overrides(self, data: dict) -> None:
        """Apply runtime overrides from the UI Settings tab.

        Secret keys are only set when a non-empty value is supplied (so saving
        the form never wipes an env-provided key). ``proxy`` can be cleared by
        sending an empty string.
        """
        for attr in ("anthropic_api_key", "openai_api_key", "local_api_key"):
            v = data.get(attr)
            if v and v.strip():
                setattr(self, attr, v.strip())
        for attr in ("anthropic_model", "openai_model", "local_model",
                     "anthropic_base_url", "openai_base_url", "local_base_url"):
            v = data.get(attr)
            if v and v.strip():
                setattr(self, attr, v.strip())
        if "proxy" in data:
            p = (data.get("proxy") or "").strip()
            self.proxy = p or None
        if "local_use_proxy" in data:
            self.local_use_proxy = bool(data.get("local_use_proxy"))
        if "local_max_concurrency" in data:
            try:
                self.local_max_concurrency = max(
                    0, min(int(data.get("local_max_concurrency") or 0), 16))
            except (TypeError, ValueError):
                pass
        ua = data.get("user_agent")
        if ua and ua.strip():
            self.user_agent = ua.strip()


# A process-wide settings instance, lazily initialised.
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
