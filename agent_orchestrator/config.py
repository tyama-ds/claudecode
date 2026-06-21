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

    # API keys / models (read from the environment).
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    openai_model: str = DEFAULT_OPENAI_MODEL
    local_model: str = DEFAULT_LOCAL_MODEL
    local_base_url: str = DEFAULT_LOCAL_BASE_URL

    # Generation limits.
    max_tokens: int = 4096
    # Per-agent-turn timeout (seconds) for subprocess/CLI adapters.
    cli_timeout: int = 600

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls(
            host=os.environ.get("ORCHESTRATOR_HOST", "127.0.0.1"),
            port=int(os.environ.get("ORCHESTRATOR_PORT", "8765")),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
            openai_model=os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            local_model=os.environ.get("LOCAL_LLM_MODEL", DEFAULT_LOCAL_MODEL),
            local_base_url=os.environ.get("LOCAL_LLM_BASE_URL", DEFAULT_LOCAL_BASE_URL),
            max_tokens=int(os.environ.get("ORCHESTRATOR_MAX_TOKENS", "4096")),
            cli_timeout=int(os.environ.get("ORCHESTRATOR_CLI_TIMEOUT", "600")),
        )


# A process-wide settings instance, lazily initialised.
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
