"""GPT Researcher engine — Engine implementation and pure logic (ported from
deep-research-orchestrator's runners/gpt_researcher/gptr_engine.py).

This module never imports the ``gpt_researcher`` package: the server runs
on whatever Python this tool runs on, while gpt-researcher 0.16.0 needs
Python 3.12 and its own dependency set. Each run starts ``worker.py`` as a
subprocess in the engine's interpreter (``engines/venv.py``) with a
controlled environment:

- gpt-researcher is configured through process-global environment
  variables, so the worker gets a MINIMAL env built from scratch (PATH /
  HOME / proxies / the engine settings); the server's own API keys and
  search keys never leak into it, and ``proxy_env`` cannot override the
  engine settings (FORBIDDEN_ENV_KEYS)
- the worker prints JSONL to stdout (events / result / fatal); the API
  key travels only through the environment and is masked in errors
- search goes through this tool's choice: SearXNG when ``searx_url`` is
  set, otherwise DuckDuckGo (the engine's own retriever). The default
  Tavily retriever is never used and no search-provider key exists.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..jsonl_worker import (EngineConfigError, redact_secrets,
                            result_from_worker_payload, run_jsonl_worker)
from ..runner_core.engine import Engine, RunContext
from ..runner_core.models import (EngineCapabilities, RunInput, RunRequest,
                                  RunResult)
from ..venv import ENGINE_SPECS

ENGINE_ID = "gpt-researcher"
ENGINE_VERSION = ENGINE_SPECS[ENGINE_ID]["version"]   # pinned package version
ENGINE_NAME = "GPT Researcher"

# search-provider API keys are never carried into the worker (no fallback
# to Tavily & co.), and the engine settings cannot be overridden from outside
FORBIDDEN_ENV_KEYS = frozenset({
    "TAVILY_API_KEY", "EXA_API_KEY", "BING_API_KEY", "SERPER_API_KEY",
    "SERPAPI_API_KEY", "SEARCHAPI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CX_KEY",
    "BRAVE_API_KEY",
    "RETRIEVER", "SEARX_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "FAST_LLM", "SMART_LLM", "STRATEGIC_LLM", "EMBEDDING",
})

# variables of the server's environment the worker may inherit (nothing else)
_INHERITED_ENV = (
    "PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP",
    # Windows needs these for Python itself to start
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "APPDATA",
    "LOCALAPPDATA", "USERPROFILE", "PROGRAMDATA",
    # corporate CA bundles / proxy exclusions
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "NO_PROXY", "no_proxy",
    # tokenizer / model caches (first run may need the network otherwise)
    "TIKTOKEN_CACHE_DIR", "HF_HOME", "XDG_CACHE_HOME",
)

_LANGUAGE_MAP = {
    "ja": "japanese", "en": "english", "zh": "chinese", "ko": "korean",
    "de": "german", "fr": "french", "es": "spanish", "it": "italian",
    "pt": "portuguese", "ru": "russian", "th": "thai", "vi": "vietnamese",
    "id": "indonesian", "nl": "dutch", "pl": "polish", "tr": "turkish",
    "ar": "arabic", "hi": "hindi",
}

_ALLOWED_REPORT_TYPES = ("research_report", "detailed_report")

# this tool's search providers → gpt-researcher RETRIEVER names
_RETRIEVER_FOR = {"searxng": "searx", "duckduckgo": "duckduckgo"}

DEFAULT_MAX_ITERATIONS = 3


def validate_request(request: RunRequest) -> None:
    """Fail fast (Japanese message) when the request cannot drive
    gpt-researcher: it needs an OpenAI-compatible LLM endpoint, an
    embedding model and a search provider it has a retriever for."""
    llm = request.llm
    if llm is None:
        raise EngineConfigError(
            f"{ENGINE_NAME} には LLM の設定が必要です（プロバイダ・モデルを設定してください）")
    if llm.api != "openai-compatible":
        raise EngineConfigError(
            f"{ENGINE_NAME} は OpenAI 互換 API（OpenAI / ローカルLLM）だけに対応しています。"
            f"API 種別 '{llm.api}' は使えません（Anthropic は Open Deep Research で使えます）")
    if not (llm.endpoint or "").strip():
        raise EngineConfigError(f"{ENGINE_NAME} に渡す LLM のエンドポイントが未設定です")
    if not (llm.model or "").strip():
        raise EngineConfigError(f"{ENGINE_NAME} に渡す LLM のモデル名が未設定です")
    if not (llm.embedding_model or "").strip():
        raise EngineConfigError(
            f"{ENGINE_NAME} は文脈の圧縮に埋め込みモデルを必須とします。"
            "⚙設定 → 機能 → 調査エンジン の「埋め込みモデル」を設定してください")
    search = request.search
    if search is None or search.provider == "disabled":
        raise EngineConfigError(
            f"{ENGINE_NAME} には Web 検索が必要です（DuckDuckGo または SearXNG）。"
            "このエンジンにオフラインモードはありません")
    if search.provider == "searxng" and not (search.endpoint or "").strip():
        raise EngineConfigError("SearXNG の URL（searx_url）が未設定です")
    if search.provider not in ("searxng", "duckduckgo", "selenium"):
        raise EngineConfigError(f"検索プロバイダ '{search.provider}' は {ENGINE_NAME} で使えません")
    report_type = str(request.options.get("report_type", "research_report"))
    if report_type not in _ALLOWED_REPORT_TYPES:
        raise EngineConfigError(
            f"report_type '{report_type}' は未対応です。対応値: {list(_ALLOWED_REPORT_TYPES)}")


def resolve_max_iterations(request: RunRequest) -> int:
    """Search iterations: options.max_iterations (default 3), capped by
    max_searches when the request sets one; never below 1."""
    try:
        max_iterations = int(request.options.get("max_iterations", DEFAULT_MAX_ITERATIONS))
    except (TypeError, ValueError):
        max_iterations = DEFAULT_MAX_ITERATIONS
    max_iterations = max(1, max_iterations)
    if request.max_searches is not None:
        try:
            max_iterations = max(1, min(max_iterations, int(request.max_searches)))
        except (TypeError, ValueError):
            pass
    return max_iterations


def build_worker_env(request: RunRequest, *,
                     base_env: Optional[Mapping[str, str]] = None
                     ) -> Tuple[Dict[str, str], List[str]]:
    """The worker's environment, built from scratch.

    - from ``base_env`` (normally os.environ) only the names in
      _INHERITED_ENV are copied — never the server's API keys
    - ``request.proxy_env`` is merged (FORBIDDEN_ENV_KEYS refused, warned)
    - the engine settings come LAST so nothing can override them
    Returns (env, warnings)."""
    validate_request(request)
    assert request.llm is not None and request.search is not None
    warnings: List[str] = []
    base = dict(base_env) if base_env is not None else {}

    env: Dict[str, str] = {
        "PATH": base.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": base.get("HOME", base.get("USERPROFILE", "/tmp")),
        "LANG": base.get("LANG", "C.UTF-8"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    for key in _INHERITED_ENV:
        if key in base and key not in env and base[key]:
            env[key] = base[key]

    for key, value in (request.proxy_env or {}).items():
        if key.upper() in FORBIDDEN_ENV_KEYS:
            warnings.append(f"proxy_env のキー {key} は安全のため無視しました")
            continue
        env[key] = str(value)
        # requests reads the lowercase spelling, httpx either
        if key.upper() in ("HTTP_PROXY", "HTTPS_PROXY") and key.lower() not in env:
            env[key.lower()] = str(value)

    llm = request.llm
    search = request.search

    # --- search: this tool's provider; never the default Tavily ---
    provider = search.provider
    if provider == "selenium":
        warnings.append(f"{ENGINE_NAME} はブラウザ検索（Selenium）を使えないため、"
                        "DuckDuckGo で検索します")
        provider = "duckduckgo"
    env["RETRIEVER"] = _RETRIEVER_FOR[provider]
    if provider == "searxng":
        env["SEARX_URL"] = (search.endpoint or "").strip()
    env["MAX_SEARCH_RESULTS_PER_QUERY"] = str(max(1, int(search.max_results or 10)))

    # --- LLM: one OpenAI-compatible endpoint for every role ---
    env["OPENAI_BASE_URL"] = llm.endpoint.strip()
    # gpt-researcher's openai provider insists on a key; local servers ignore it
    env["OPENAI_API_KEY"] = llm.api_key or "dummy-key"
    model_spec = f"openai:{llm.model.strip()}"
    env["FAST_LLM"] = model_spec
    env["SMART_LLM"] = model_spec
    env["STRATEGIC_LLM"] = model_spec

    # --- embeddings (required for context compression, any retriever) ---
    env["EMBEDDING"] = f"openai:{llm.embedding_model.strip()}"
    if "api.openai.com" not in llm.endpoint:
        # langchain's OpenAIEmbeddings tokenises the input with tiktoken and
        # sends token ids unless told otherwise; Ollama / LM Studio / other
        # OpenAI-compatible servers only accept strings
        env["EMBEDDING_KWARGS"] = json.dumps({"check_embedding_ctx_length": False})
    if llm.embedding_endpoint and llm.embedding_endpoint.rstrip("/") != llm.endpoint.rstrip("/"):
        warnings.append(
            "埋め込みのエンドポイントが LLM と異なりますが、gpt-researcher は埋め込み用に"
            "別のエンドポイントを指定できないため、LLM のエンドポイントを使います")

    # --- the rest of the engine settings ---
    env["SCRAPER"] = "bs"
    env["MAX_ITERATIONS"] = str(resolve_max_iterations(request))
    lang = (request.input.language or "ja").lower()
    env["LANGUAGE"] = _LANGUAGE_MAP.get(lang, lang)
    env["REPORT_FORMAT"] = "markdown"
    return env, warnings


def build_query(run_input: RunInput) -> str:
    """topic + objective + instructions as the one query gpt-researcher takes."""
    parts = [run_input.topic.strip()]
    if run_input.objective:
        parts.append(f"調査目的: {run_input.objective.strip()}")
    if run_input.instructions:
        parts.append(f"追加指示: {run_input.instructions.strip()}")
    return "\n\n".join(p for p in parts if p)


def build_worker_payload(request: RunRequest) -> Tuple[Dict[str, Any], List[str]]:
    """The JSON the worker reads from stdin (no secrets). (payload, warnings)."""
    warnings: List[str] = []
    if request.input.documents:
        warnings.append(f"{ENGINE_NAME} は手元の資料を受け付けないため無視しました")
    payload: Dict[str, Any] = {
        "query": build_query(request.input),
        "report_type": str(request.options.get("report_type", "research_report")),
        "language": request.input.language,
        "input_urls": list(request.input.input_urls or []),
    }
    return payload, warnings


def worker_path() -> Path:
    return Path(__file__).with_name("worker.py")


class GptResearcherEngine(Engine):
    """Runs gpt-researcher in the engine's own interpreter via worker.py."""

    engine_id = ENGINE_ID

    def __init__(self, python_path: Optional[str] = None,
                 worker: Optional[str] = None):
        # the interpreter is resolved by the registry (engines/venv.py);
        # falling back to the server's Python is only for tests with a
        # stand-in worker
        self.python_path = str(python_path or sys.executable)
        self.worker = Path(worker) if worker else worker_path()

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine_id=self.engine_id,
            name=ENGINE_NAME,
            version=ENGINE_VERSION,
            output_kind="report",
            streaming=True,
            cancel=True,
            citations=False,    # inline citations in the body, no structured claims
            token_usage=False,  # gpt-researcher does not expose token counts
            cost=True,          # its own OpenAI price-table estimate only
            local_files=False,
            options_schema={
                "type": "object",
                "properties": {
                    "report_type": {
                        "type": "string",
                        "title": "レポートの種類",
                        "enum": list(_ALLOWED_REPORT_TYPES),
                        "default": "research_report",
                        "description": "research_report = 標準（1 回の執筆）、detailed_report = 詳細（小節ごとに執筆。時間と LLM 呼び出しが増える）",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "title": "検索イテレーション数",
                        "minimum": 1,
                        "maximum": 10,
                        "default": DEFAULT_MAX_ITERATIONS,
                        "description": "検索サブクエリの数（1〜10）。多いほど情報源が増え、時間も増えます",
                    },
                },
            },
            health="available",
            health_reason=None,
            required_config=["llm:openai-compatible+embedding", "search:duckduckgo|searxng"],
        )

    async def run(self, ctx: RunContext) -> RunResult:
        request = ctx.request
        validate_request(request)                     # EngineConfigError (ValueError)
        env, env_warnings = build_worker_env(request, base_env=os.environ)
        payload, payload_warnings = build_worker_payload(request)
        if not self.worker.is_file():
            raise EngineConfigError(f"{ENGINE_NAME} のワーカーが見つかりません: {self.worker}")
        if not Path(self.python_path).is_file():
            raise EngineConfigError(
                f"{ENGINE_NAME} の Python が見つかりません: {self.python_path}")

        ctx.emit("stage", {"stage": "starting_worker", "engine": self.engine_id,
                           "python": self.python_path})
        secrets = [request.llm.api_key if request.llm else None]
        stderr_tail: List[str] = []
        result_payload = await run_jsonl_worker(
            ctx,
            argv=[self.python_path, "-u", str(self.worker)],
            env=env,
            payload=payload,
            cwd=str(self.worker.parent),
            secrets=secrets,
            stderr_tail_out=stderr_tail,
        )
        try:
            result = result_from_worker_payload(result_payload)
        except Exception as e:  # noqa: BLE001 - a malformed result line is an engine error
            raise RuntimeError(redact_secrets(
                f"{ENGINE_NAME} の結果を読み取れませんでした: {e}", secrets)) from e
        if not result.sources:
            # gpt-researcher writes an apology instead of a report when every
            # search came back empty; that is a failed run here, never a
            # "report" to verify. The retriever prints its errors to stderr.
            hints = [line.strip() for line in stderr_tail
                     if "error" in line.lower() or "exception" in line.lower()][-3:]
            where = ("SearXNG の URL" if env.get("RETRIEVER") == "searx"
                     else "DuckDuckGo への到達（プロキシ・社内ネットワークの制限）")
            raise RuntimeError(
                f"{ENGINE_NAME} は情報源を 1 件も取得できませんでした（検索結果が空）。"
                f"{where}を確認してください"
                + (f"。ワーカーの出力: {' / '.join(hints)[:600]}" if hints else ""))
        result.warnings = [*env_warnings, *payload_warnings, *result.warnings]
        return result
