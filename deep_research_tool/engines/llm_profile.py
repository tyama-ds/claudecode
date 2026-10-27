"""Config → run-scoped LLM / search settings for an external engine.

The engine gets exactly what the selected provider would get: the same
endpoint, model and key (OpenAI / local OpenAI-compatible server /
Anthropic). Nothing here is persisted; the key only travels inside the
request object to the engine subprocess's environment.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .runner_core.models import LlmRunConfig, SearchRunConfig

OPENAI_DEFAULT_ENDPOINT = "https://api.openai.com/v1"
ANTHROPIC_DEFAULT_ENDPOINT = "https://api.anthropic.com"

# embedding defaults per provider (GPT Researcher needs an embedding model;
# the user can override with research.engine_embedding_model)
DEFAULT_EMBEDDINGS = {
    "openai": "text-embedding-3-small",
    "local": "nomic-embed-text",
    "anthropic": None,          # no embeddings API; the engine decides / refuses
}


def _provider_name(config) -> str:
    p = getattr(config.api, "provider", "openai")
    return str(getattr(p, "value", p) or "openai").lower()


def _local_endpoint(config) -> Optional[str]:
    url = (getattr(config.api, "local_base_url", None) or "").strip()
    if not url:
        return None
    backend = str(getattr(getattr(config.api, "local_backend", ""), "value",
                          getattr(config.api, "local_backend", "")) or "").lower()
    # Ollama's OpenAI-compatible API lives under /v1
    if backend == "ollama" and not url.rstrip("/").endswith("/v1"):
        url = url.rstrip("/") + "/v1"
    return url


def llm_run_config(config, *, profile_id: Optional[str] = None) -> LlmRunConfig:
    """The active provider as an engine LLM profile. Raises ValueError
    when the provider has nothing to point the engine at (no local URL)."""
    provider = _provider_name(config)
    api = config.api
    embedding = getattr(config.research, "engine_embedding_model", None) or \
        DEFAULT_EMBEDDINGS.get(provider)
    timeout = int(getattr(api, "local_timeout", None) or 120) if provider == "local" else 120
    if provider == "openai":
        model = api.openai_model
        return LlmRunConfig(
            profile_id=profile_id or f"openai:{model}",
            api="openai-compatible",
            endpoint=(api.openai_base_url or "").strip() or OPENAI_DEFAULT_ENDPOINT,
            model=model,
            api_key=api.openai_api_key,
            embedding_model=embedding,
            timeout_seconds=timeout,
        )
    if provider == "local":
        endpoint = _local_endpoint(config)
        if not endpoint:
            raise ValueError("ローカルLLM のサーバーURL（local_base_url）が未設定のため、"
                             "外部エンジンに渡す接続先がありません")
        model = api.local_model
        return LlmRunConfig(
            profile_id=profile_id or f"local:{model}",
            api="openai-compatible",
            endpoint=endpoint,
            model=model,
            # OpenAI-compatible clients refuse an empty key; local servers ignore it
            api_key=api.local_api_key or "local",
            embedding_model=embedding,
            embedding_endpoint=endpoint,
            timeout_seconds=timeout,
        )
    if provider == "anthropic":
        model = api.anthropic_model
        return LlmRunConfig(
            profile_id=profile_id or f"anthropic:{model}",
            api="anthropic",
            endpoint=(api.anthropic_base_url or "").strip() or ANTHROPIC_DEFAULT_ENDPOINT,
            model=model,
            api_key=api.anthropic_api_key,
            embedding_model=embedding,
            timeout_seconds=timeout,
        )
    raise ValueError(f"unknown LLM provider: {provider}")


def search_run_config(config) -> SearchRunConfig:
    """The search backend the engine should use: SearXNG when a URL is
    configured, otherwise the tool's own search method (DuckDuckGo /
    Selenium) which Phase 2/3 engines translate to their own retrievers."""
    searx = (getattr(config.research, "searx_url", None) or "").strip()
    max_results = int(getattr(config.search, "max_results", 10) or 10)
    if searx:
        return SearchRunConfig(provider="searxng", endpoint=searx,
                               max_results=max_results)
    method = str(getattr(getattr(config.search, "method", ""), "value",
                         getattr(config.search, "method", "")) or "duckduckgo").lower()
    provider = "selenium" if method == "selenium" else "duckduckgo"
    return SearchRunConfig(provider=provider, endpoint=None, max_results=max_results)


def proxy_env(config) -> Dict[str, str]:
    """HTTP(S)_PROXY for the engine subprocess from the tool's proxy config."""
    out: Dict[str, str] = {}
    try:
        proxies = config.proxy.get_proxies_dict() or {}
    except Exception:
        proxies = {}
    if proxies.get("http"):
        out["HTTP_PROXY"] = proxies["http"]
    if proxies.get("https"):
        out["HTTPS_PROXY"] = proxies["https"]
    return out


def describe_profile(cfg: Optional[LlmRunConfig]) -> Dict[str, Any]:
    """Non-secret view for the UI / result (never the key)."""
    if cfg is None:
        return {}
    return {"api": cfg.api, "endpoint": cfg.endpoint, "model": cfg.model,
            "embedding_model": cfg.embedding_model,
            "has_api_key": bool(cfg.api_key)}
