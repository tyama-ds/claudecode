"""Open Deep Research engine — Engine implementation and pure logic (ported
from deep-research-orchestrator's runners/open_deep_research/odr_engine.py).

This module never imports ``open_deep_research`` / langchain / mcp: the
server runs on whatever Python this tool runs on, while open-deep-research
0.0.16 needs Python 3.12 and its own dependency set. Each run:

1. starts the tool's in-process MCP search server (``mcp_search.py``) on
   a free localhost port — the engine has no DuckDuckGo / SearXNG
   retriever of its own, so it is told ``search_api="none"`` and given
   ``mcp_config`` pointing at that server (tools ``web_search`` /
   ``fetch_page``; 指定 URL のみ → ``fetch_page`` only)
2. starts ``worker.py`` in the engine's interpreter (``engines/venv.py``)
   with a MINIMAL environment built from scratch: PATH / HOME / proxies /
   the LLM endpoint and key. open-deep-research's ``Configuration`` reads
   environment variables BEFORE the configurable dict, so the field names
   (SEARCH_API, RESEARCH_MODEL, …) and every hosted-search key are blocked
   from the worker's environment (FORBIDDEN_ENV_KEYS); NO_PROXY covers
   127.0.0.1 so the MCP connection never goes through the proxy
3. reads the worker's JSONL (stages → progress, result → RunResult)

LLM: OpenAI-compatible (``openai:<model>``) or Anthropic
(``anthropic:<model>``); the same model for the research / summarization
/ compression / final-report roles. No embedding model is needed.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..jsonl_worker import (EngineConfigError, redact_secrets,
                            result_from_worker_payload, run_jsonl_worker)
from ..runner_core.engine import Engine, RunContext
from ..runner_core.models import (EngineCapabilities, RunInput, RunRequest,
                                  RunResult, SourceRecord)
from ..venv import ENGINE_SPECS
from .mcp_search import (ALL_TOOLS, TOOL_FETCH, TOOL_SEARCH, McpSearchServer,
                         SearchTools, backend_for)
# pure helpers shared with the worker (worker.py imports nothing of ours)
from .worker import parse_sources_section, urls_in_body

ENGINE_ID = "open-deep-research"
ENGINE_VERSION = ENGINE_SPECS[ENGINE_ID]["version"]   # pinned package version
ENGINE_NAME = "Open Deep Research"

# hosted search APIs the engine could fall back to (never), and the
# Configuration field names the environment would override
FORBIDDEN_ENV_KEYS = frozenset({
    "TAVILY_API_KEY", "EXA_API_KEY", "BING_API_KEY", "SERPER_API_KEY",
    "SERPAPI_API_KEY", "SEARCHAPI_API_KEY", "GOOGLE_API_KEY", "BRAVE_API_KEY",
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE",
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_API_URL",
    "GET_API_KEYS_FROM_CONFIG",
    "SEARCH_API", "ALLOW_CLARIFICATION", "MCP_CONFIG", "MCP_PROMPT",
    "RESEARCH_MODEL", "SUMMARIZATION_MODEL", "COMPRESSION_MODEL", "FINAL_REPORT_MODEL",
    "RESEARCH_MODEL_MAX_TOKENS", "SUMMARIZATION_MODEL_MAX_TOKENS",
    "COMPRESSION_MODEL_MAX_TOKENS", "FINAL_REPORT_MODEL_MAX_TOKENS",
    "MAX_CONCURRENT_RESEARCH_UNITS", "MAX_RESEARCHER_ITERATIONS",
    "MAX_REACT_TOOL_CALLS", "MAX_STRUCTURED_OUTPUT_RETRIES", "MAX_CONTENT_LENGTH",
})

# variables of the server's environment the worker may inherit (nothing else)
_INHERITED_ENV = (
    "PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP",
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "APPDATA",
    "LOCALAPPDATA", "USERPROFILE", "PROGRAMDATA",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "NO_PROXY", "no_proxy",
    "TIKTOKEN_CACHE_DIR", "HF_HOME", "XDG_CACHE_HOME",
)

FORBIDDEN_SEARCH_APIS = frozenset({"tavily", "openai", "anthropic"})

DEFAULT_MAX_RESEARCHER_ITERATIONS = 3
DEFAULT_MAX_REACT_TOOL_CALLS = 5
DEFAULT_MAX_CONCURRENT_UNITS = 2
DEFAULT_RECURSION_LIMIT = 50

_LANGUAGE_NAMES = {
    "ja": "Japanese", "en": "English", "zh": "Chinese", "ko": "Korean",
    "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "th": "Thai", "vi": "Vietnamese",
    "id": "Indonesian", "nl": "Dutch", "pl": "Polish", "tr": "Turkish",
    "ar": "Arabic", "hi": "Hindi",
}


def search_disabled_by_option(request: RunRequest) -> bool:
    """``options.search_api == "none"`` is the one explicit way to run
    without any search tool (LLM knowledge only)."""
    return request.options.get("search_api") == "none"


def validate_request(request: RunRequest) -> None:
    """Fail fast (Japanese message) when the request cannot drive
    open-deep-research: an LLM endpoint (OpenAI-compatible or Anthropic)
    and a search the tool can serve over MCP."""
    llm = request.llm
    if llm is None:
        raise EngineConfigError(
            f"{ENGINE_NAME} には LLM の設定が必要です（プロバイダ・モデルを設定してください）")
    if llm.api not in ("openai-compatible", "anthropic"):
        raise EngineConfigError(f"API 種別 '{llm.api}' は {ENGINE_NAME} で使えません")
    if not (llm.endpoint or "").strip():
        raise EngineConfigError(f"{ENGINE_NAME} に渡す LLM のエンドポイントが未設定です")
    if not (llm.model or "").strip():
        raise EngineConfigError(f"{ENGINE_NAME} に渡す LLM のモデル名が未設定です")
    search_api = request.options.get("search_api")
    if search_api is not None and search_api != "none":
        if search_api in FORBIDDEN_SEARCH_APIS:
            raise EngineConfigError(
                f"有料／ホスト型の検索 API（search_api={search_api}）はこのツールでは使えません。"
                "検索はこのツールの検索設定（DuckDuckGo / ブラウザ / SearXNG）を MCP 経由で渡します")
        raise EngineConfigError(f"search_api '{search_api}' は未対応です")
    if search_disabled_by_option(request):
        return
    search = request.search
    if search is None or search.provider == "disabled":
        raise EngineConfigError(
            f"{ENGINE_NAME} には Web 検索が必要です（DuckDuckGo / ブラウザ / SearXNG）。"
            "検索なしで実行するときは engine_options に search_api=\"none\" を明示してください")
    if search.provider == "searxng" and not (search.endpoint or "").strip():
        raise EngineConfigError("SearXNG の URL（searx_url）が未設定です")
    if search.provider not in ("searxng", "duckduckgo", "selenium"):
        raise EngineConfigError(f"検索プロバイダ '{search.provider}' は {ENGINE_NAME} で使えません")


def model_prefix(request: RunRequest) -> str:
    assert request.llm is not None
    return "anthropic:" if request.llm.api == "anthropic" else "openai:"


def _append_no_proxy(env: Dict[str, str], hosts: str = "127.0.0.1,localhost") -> None:
    """The MCP server is on localhost: never route it through the proxy.
    Both spellings end up with the same merged list (requests reads the
    lowercase one, httpx either)."""
    parts: List[str] = []
    for key in ("NO_PROXY", "no_proxy"):
        for p in env.get(key, "").split(","):
            p = p.strip()
            if p and p not in parts:
                parts.append(p)
    for host in hosts.split(","):
        if host not in parts:
            parts.append(host)
    env["NO_PROXY"] = env["no_proxy"] = ",".join(parts)


def build_worker_env(request: RunRequest, *,
                     base_env: Optional[Mapping[str, str]] = None
                     ) -> Tuple[Dict[str, str], List[str]]:
    """The worker's environment, built from scratch: a few inherited
    names, the proxies (FORBIDDEN_ENV_KEYS refused), then the LLM endpoint
    and key for the selected API. Returns (env, warnings)."""
    validate_request(request)
    assert request.llm is not None
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
        if key.upper() in ("HTTP_PROXY", "HTTPS_PROXY") and key.lower() not in env:
            env[key.lower()] = str(value)
    _append_no_proxy(env)

    llm = request.llm
    endpoint = llm.endpoint.strip()
    if llm.api == "anthropic":
        env["ANTHROPIC_API_KEY"] = llm.api_key or "dummy-key"
        env["ANTHROPIC_BASE_URL"] = endpoint
        env["ANTHROPIC_API_URL"] = endpoint
    else:
        env["OPENAI_API_KEY"] = llm.api_key or "dummy-key"
        env["OPENAI_BASE_URL"] = endpoint
        env["OPENAI_API_BASE"] = endpoint
    # the engine resolves keys from the environment, never from the config
    env["GET_API_KEYS_FROM_CONFIG"] = "false"
    return env, warnings


def _int_option(request: RunRequest, key: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(request.options.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def resolve_max_react_tool_calls(request: RunRequest) -> int:
    """Tool calls per researcher: options.max_react_tool_calls (default 5),
    capped by max_searches when the request sets one; never below 1."""
    value = _int_option(request, "max_react_tool_calls", DEFAULT_MAX_REACT_TOOL_CALLS, 1, 20)
    if request.max_searches is not None:
        try:
            value = max(1, min(value, int(request.max_searches)))
        except (TypeError, ValueError):
            pass
    return value


def mcp_tools_for(request: RunRequest) -> List[str]:
    """Which search tools the engine may use: 指定 URL のみ
    (input_urls without complement) → fetch_page only; otherwise both."""
    if request.input.input_urls and not bool(request.options.get("complement_source_urls", True)):
        return [TOOL_FETCH]
    return list(ALL_TOOLS)


def build_mcp_prompt(tools: List[str], run_input: RunInput) -> str:
    if TOOL_SEARCH in tools:
        text = (f"You MUST use the `{TOOL_SEARCH}` tool to search the web; it returns titles, "
                "URLs, snippets and page excerpts from the research tool's configured search "
                f"engine. Use `{TOOL_FETCH}` to read a page in full. Ground every finding in "
                "these pages and cite their URLs.")
    else:
        text = (f"You MUST use ONLY the pages listed as starting URLs, read with the "
                f"`{TOOL_FETCH}` tool. Do not look for other sources; if a page does not cover "
                "a point, say so. Cite the URLs you used.")
    if run_input.input_urls:
        # the researcher only sees its system prompt (where mcp_prompt is
        # formatted in) and the supervisor's research_topic — not the
        # first user message — so the URLs must be listed here
        text += (" Read every starting URL before writing. Starting URLs:\n"
                 + "\n".join(f"  - {u}" for u in run_input.input_urls))
    return text


def build_configurable(request: RunRequest, mcp_url: Optional[str],
                       tools: Optional[List[str]] = None) -> Dict[str, Any]:
    """open-deep-research's ``configurable`` dict."""
    assert request.llm is not None
    model = f"{model_prefix(request)}{request.llm.model.strip()}"
    configurable: Dict[str, Any] = {
        "allow_clarification": False,          # headless: never ask the user back
        "search_api": "none",                  # no hosted search; MCP only
        "research_model": model,
        "summarization_model": model,
        "compression_model": model,
        "final_report_model": model,
        "max_structured_output_retries": 3,
        "max_concurrent_research_units": _int_option(
            request, "max_concurrent_research_units", DEFAULT_MAX_CONCURRENT_UNITS, 1, 5),
        "max_researcher_iterations": _int_option(
            request, "max_researcher_iterations", DEFAULT_MAX_RESEARCHER_ITERATIONS, 1, 10),
        "max_react_tool_calls": resolve_max_react_tool_calls(request),
    }
    if mcp_url:
        tools = tools or mcp_tools_for(request)
        configurable["mcp_config"] = {
            "url": mcp_url,                    # the engine appends "/mcp"
            "tools": list(tools),
            "auth_required": False,
        }
        configurable["mcp_prompt"] = build_mcp_prompt(list(tools), request.input)
    return configurable


def build_prompt(run_input: RunInput, complement_urls: bool = True) -> str:
    """topic / objective / instructions / starting URLs / language as the
    one message the graph starts from."""
    parts = [f"Research topic: {run_input.topic.strip()}"]
    if run_input.objective:
        parts.append(f"Objective: {run_input.objective.strip()}")
    if run_input.instructions:
        parts.append(f"Additional instructions: {run_input.instructions.strip()}")
    if run_input.input_urls:
        urls = "\n".join(f"- {u}" for u in run_input.input_urls)
        if complement_urls:
            parts.append(f"Start from these URLs (read them first, then search for more):\n{urls}")
        else:
            parts.append(f"Use ONLY these URLs as sources (read each one; do not search elsewhere):\n{urls}")
    lang = (run_input.language or "ja").lower()
    lang_name = _LANGUAGE_NAMES.get(lang, run_input.language or "Japanese")
    parts.append(f"Write the final report in {lang_name}. End it with a '### Sources' section "
                 "listing every cited URL as '[n] Title: URL'.")
    return "\n\n".join(parts)


def build_worker_payload(request: RunRequest, mcp_url: Optional[str],
                         tools: Optional[List[str]] = None
                         ) -> Tuple[Dict[str, Any], List[str]]:
    """The JSON the worker reads from stdin (no secrets). (payload, warnings)."""
    warnings: List[str] = []
    if request.input.documents:
        warnings.append(f"{ENGINE_NAME} は手元の資料を受け付けないため無視しました")
    if search_disabled_by_option(request):
        warnings.append("search_api=\"none\" が指定されたため、Web 検索なし（LLM の知識のみ）で実行しました")
    complement = bool(request.options.get("complement_source_urls", True))
    payload = {
        "prompt": build_prompt(request.input, complement_urls=complement),
        "language": request.input.language,
        "configurable": build_configurable(request, mcp_url, tools),
        "recursion_limit": _int_option(request, "recursion_limit", DEFAULT_RECURSION_LIMIT, 10, 200),
    }
    return payload, warnings


def worker_path() -> Path:
    return Path(__file__).with_name("worker.py")


class OpenDeepResearchEngine(Engine):
    """Runs open-deep-research in the engine's own interpreter via
    worker.py, with the tool's search served over MCP."""

    engine_id = ENGINE_ID

    def __init__(self, python_path: Optional[str] = None, worker: Optional[str] = None,
                 search_client=None):
        self.python_path = str(python_path or sys.executable)
        self.worker = Path(worker) if worker else worker_path()
        # the registry / main.py hand over the tool's search client; the
        # MCP server lends it to the engine
        self.search_client = search_client
        self.last_search_log: Dict[str, Any] = {}

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine_id=self.engine_id,
            name=ENGINE_NAME,
            version=ENGINE_VERSION,
            output_kind="report",
            streaming=True,
            cancel=True,
            citations=False,     # Sources list + URLs in the body, no structured claims
            token_usage=True,    # usage_metadata when the model reports it
            cost=False,
            local_files=False,
            options_schema={
                "type": "object",
                "properties": {
                    "max_researcher_iterations": {
                        "type": "integer", "title": "監督者の反復回数",
                        "minimum": 1, "maximum": 10,
                        "default": DEFAULT_MAX_RESEARCHER_ITERATIONS,
                        "description": "監督者が調査員に調査を割り当てる回数の上限（1〜10）",
                    },
                    "max_react_tool_calls": {
                        "type": "integer", "title": "調査員あたりのツール呼び出し数",
                        "minimum": 1, "maximum": 20,
                        "default": DEFAULT_MAX_REACT_TOOL_CALLS,
                        "description": "調査員 1 人が検索・ページ取得を呼べる回数（1〜20）",
                    },
                    "max_concurrent_research_units": {
                        "type": "integer", "title": "同時に動く調査員の数",
                        "minimum": 1, "maximum": 5,
                        "default": DEFAULT_MAX_CONCURRENT_UNITS,
                        "description": "並列の調査員の数（1〜5）。多いほど速いが LLM 呼び出しが重なります",
                    },
                    "recursion_limit": {
                        "type": "integer", "title": "グラフの再帰上限",
                        "minimum": 10, "maximum": 200,
                        "default": DEFAULT_RECURSION_LIMIT,
                        "description": "LangGraph のステップ数の上限（10〜200）",
                    },
                },
            },
            health="available",
            health_reason=None,
            required_config=["llm:openai-compatible|anthropic", "search:duckduckgo|selenium|searxng"],
        )

    async def run(self, ctx: RunContext) -> RunResult:
        request = ctx.request
        validate_request(request)                      # EngineConfigError (ValueError)
        env, env_warnings = build_worker_env(request, base_env=os.environ)
        if not self.worker.is_file():
            raise EngineConfigError(f"{ENGINE_NAME} のワーカーが見つかりません: {self.worker}")
        if not Path(self.python_path).is_file():
            raise EngineConfigError(f"{ENGINE_NAME} の Python が見つかりません: {self.python_path}")

        server: Optional[McpSearchServer] = None
        tools_impl: Optional[SearchTools] = None
        mcp_url: Optional[str] = None
        tool_names: Optional[List[str]] = None
        try:
            if not search_disabled_by_option(request):
                search = request.search
                assert search is not None
                try:
                    backend = backend_for(search.provider, endpoint=search.endpoint,
                                          search_client=self.search_client,
                                          timeout=search.timeout_seconds)
                except ValueError as e:
                    raise EngineConfigError(str(e)) from e
                # the tools run on the server's handler / fetch-pool threads;
                # their events are handed to this loop so that ctx.emit (and
                # the progress callback behind it) only ever runs here
                loop = asyncio.get_running_loop()

                def _emit_on_loop(type_: str, payload: Dict[str, Any]) -> None:
                    try:
                        ctx.emit(type_, payload)
                    except Exception:   # noqa: BLE001 - a progress hiccup must not surface as a loop error
                        pass

                tools_impl = SearchTools(
                    backend, max_results=search.max_results,
                    on_event=lambda t, p: loop.call_soon_threadsafe(_emit_on_loop, t, p))
                tool_names = mcp_tools_for(request)
                server = McpSearchServer(tools_impl, enabled_tools=tool_names).start()
                mcp_url = server.url
                ctx.emit("stage", {"stage": "starting_mcp_server", "port": server.port,
                                   "tools": tool_names, "search": backend.name})

            payload, payload_warnings = build_worker_payload(request, mcp_url, tool_names)
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
        finally:
            if server is not None:
                server.stop()
            self.last_search_log = dict(tools_impl.log) if tools_impl is not None else {}

        try:
            result = result_from_worker_payload(result_payload)
        except Exception as e:  # noqa: BLE001 - a malformed result line is an engine error
            raise RuntimeError(redact_secrets(
                f"{ENGINE_NAME} の結果を読み取れませんでした: {e}", secrets)) from e

        if not result.sources and result.report_markdown:
            # a worker that could not list sources: the report's own
            # Sources section, else the URLs cited in its body
            result.sources = [SourceRecord(**d) for d in
                              (parse_sources_section(result.report_markdown)
                               or urls_in_body(result.report_markdown))]
        log = self.last_search_log
        if not result.sources and tools_impl is not None:
            # no source at all: the engine wrote from nothing it could cite
            hints = [line.strip() for line in stderr_tail
                     if "error" in line.lower() or "exception" in line.lower()][-3:]
            fetch_errors = int(log.get("fetch_errors") or 0)
            if log.get("searches", 0) == 0 and log.get("fetches", 0) == 0 and fetch_errors == 0:
                raise RuntimeError(
                    f"{ENGINE_NAME} は検索ツールを一度も呼ばずに終了し、出典がありません"
                    "（MCP 接続・LLM のツール呼び出し対応を確認してください）"
                    + (f"。ワーカーの出力: {' / '.join(hints)[:600]}" if hints else ""))
            if log.get("searches", 0) == 0 and log.get("fetches", 0) == 0:
                # fetch_page was called (指定 URL のみ) but no page could be read
                failed = list((log.get("failed_urls") or {}).items())[:3]
                detail = " / ".join(f"{u}（{why}）" for u, why in failed)
                raise RuntimeError(
                    f"{ENGINE_NAME} はページの取得を {fetch_errors} 回試みましたが 1 件も読めず、"
                    "出典がありません（指定 URL への到達・プロキシ・アクセス制限・PDF 等の解析を"
                    "確認してください）" + (f"。取得できなかったページ: {detail[:600]}" if detail else ""))
            if log.get("results", 0) == 0 and log.get("fetches", 0) == 0:
                where = ("SearXNG の URL" if request.search and request.search.provider == "searxng"
                         else "DuckDuckGo への到達（プロキシ・社内ネットワークの制限）")
                raise RuntimeError(
                    f"{ENGINE_NAME} の検索結果が空で、情報源を 1 件も取得できませんでした。"
                    f"{where}を確認してください"
                    + (f"。ワーカーの出力: {' / '.join(hints)[:600]}" if hints else ""))
        if log:
            result.metrics.searches = int(log.get("searches") or 0)
            result.raw["search_log"] = {
                "backend": getattr(tools_impl.backend, "name", None) if tools_impl else None,
                "tools": tool_names,
                "searches": log.get("searches"), "results": log.get("results"),
                "fetches": log.get("fetches"), "fetch_errors": log.get("fetch_errors"),
                "search_errors": log.get("search_errors"),
                "queries": list(log.get("queries") or [])[:50],
                "urls": dict(list((log.get("urls") or {}).items())[:200]),
                "failed_urls": dict(log.get("failed_urls") or {}),
            }
            # titles the search saw, for sources the report lists without one
            titles = log.get("urls") or {}
            for s in result.sources:
                if not s.title and titles.get(s.url):
                    s.title = titles[s.url]
        result.warnings = [*env_warnings, *payload_warnings, *result.warnings]
        return result
