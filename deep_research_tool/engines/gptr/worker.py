"""gpt-researcher worker subprocess (runs in the engine's own Python, 3.12+).

Started by engines/gptr/engine.py with:
- environment: RETRIEVER / SEARX_URL / OPENAI_BASE_URL / OPENAI_API_KEY /
  FAST_LLM / SMART_LLM / STRATEGIC_LLM / EMBEDDING / SCRAPER / MAX_ITERATIONS /
  MAX_SEARCH_RESULTS_PER_QUERY / LANGUAGE / REPORT_FORMAT (all set by the parent)
- stdin: one JSON document {"query", "report_type", "language", "input_urls"}

Prints JSONL to stdout:
  {"kind":"event","type":"log|stage|source_found|cost","payload":{...}}
  {"kind":"result", ...RunResult fields...}   or   {"kind":"fatal","error":...}

The library (and its dependencies) may print() to stdout; so that the
protocol cannot be corrupted, fd 1 is redirected to stderr right at start
and the original stdout is kept on a private duplicate used for JSONL only.

This file is self-contained on purpose (standard library + gpt_researcher):
it must not import deep_research_tool, whose dependencies are not in the
engine's environment. Ported from deep-research-orchestrator's
runners/gpt_researcher/worker.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, TextIO

_EMIT: Optional[TextIO] = None
ENGINE_VERSION = "0.16.0"


def _setup_stdio() -> TextIO:
    """Keep the original stdout for JSONL; send fd 1 to stderr."""
    emit_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(emit_fd, "w", encoding="utf-8", buffering=1)


def _emit(obj: Dict[str, Any]) -> None:
    assert _EMIT is not None
    _EMIT.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    _EMIT.flush()


def emit_event(type_: str, payload: Dict[str, Any]) -> None:
    _emit({"kind": "event", "type": type_, "payload": payload})


def _install_typing_shim() -> None:
    """Workaround for an upstream bug in gpt-researcher 0.16.0:
    gpt_researcher/actions/query_processing.py (and friends) annotate with
    ``Any`` / ``List`` without importing them, so importing the package
    raises NameError. Injecting the typing names into builtins (inside this
    worker process only) makes the import succeed."""
    import builtins
    import typing

    for name in ("Any", "List", "Dict", "Optional", "Tuple", "Set",
                 "Union", "Callable", "Iterable", "Sequence"):
        if not hasattr(builtins, name):
            setattr(builtins, name, getattr(typing, name))


class WsBridge:
    """gpt-researcher's websocket stand-in: send_json → JSONL events."""

    async def send_json(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        type_ = data.get("type")
        content = data.get("content")
        output = data.get("output")
        metadata = data.get("metadata")
        if type_ == "images":
            return
        if type_ == "logs" and content == "added_source_url" and isinstance(metadata, str):
            emit_event("source_found", {"url": metadata})
            return
        if type_ == "report":
            # a streamed chunk of the report body; the whole text comes in
            # the result, so only a heartbeat is sent
            emit_event("log", {"message": "(report chunk)", "content": "report_chunk"})
            return
        payload: Dict[str, Any] = {"message": str(output) if output is not None else ""}
        if content is not None:
            payload["content"] = str(content)
        if metadata is not None:
            payload["metadata"] = metadata
        emit_event("log", payload)


class LogBridge:
    """gpt-researcher's log_handler stand-in."""

    async def on_tool_start(self, tool_name: str, **kwargs: Any) -> None:
        emit_event("log", {"message": f"tool start: {tool_name}", "tool": str(tool_name)})

    async def on_agent_action(self, action: str, **kwargs: Any) -> None:
        emit_event("stage", {"stage": f"action:{action}"})

    async def on_research_step(self, step: str, details: Any, **kwargs: Any) -> None:
        payload: Dict[str, Any] = {"stage": str(step)}
        if isinstance(details, dict):
            payload["details"] = {k: str(v)[:300] for k, v in details.items()}
        emit_event("stage", payload)


def _excerpt(text: Any, limit: int = 200) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    text = " ".join(text.split())
    return text[:limit]


def _summary_from_report(report: str, limit: int = 400) -> Optional[str]:
    for block in (report or "").split("\n\n"):
        block = block.strip()
        if not block or block.startswith("#"):
            continue
        return " ".join(block.split())[:limit]
    return None


def collect_sources(research_sources: Any, source_urls: Any) -> List[Dict[str, Any]]:
    """get_research_sources() items (title / excerpt) first, then any
    get_source_urls() entries not seen yet; URLs de-duplicated."""
    sources: List[Dict[str, Any]] = []
    seen = set()
    for item in research_sources or []:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("href")
        if not isinstance(url, str) or not url or url in seen:
            continue
        seen.add(url)
        sources.append({
            "url": url,
            "title": item.get("title") if isinstance(item.get("title"), str) else None,
            "excerpt": _excerpt(item.get("content") or item.get("raw_content")),
            "meta": {},
        })
    for url in source_urls or []:
        if isinstance(url, str) and url and url not in seen:
            seen.add(url)
            sources.append({"url": url, "title": None, "excerpt": None, "meta": {}})
    return sources


async def _research(payload: Dict[str, Any]) -> Dict[str, Any]:
    started = time.monotonic()
    _install_typing_shim()
    from gpt_researcher import GPTResearcher  # noqa: PLC0415 (after the env is set)

    input_urls = payload.get("input_urls") or []
    complement = bool(payload.get("complement_source_urls", True)) if input_urls else False
    researcher = GPTResearcher(
        query=payload["query"],
        report_type=payload.get("report_type", "research_report"),
        report_source="web",
        source_urls=input_urls or None,
        complement_source_urls=complement,
        websocket=WsBridge(),
        log_handler=LogBridge(),
        verbose=True,
    )

    emit_event("stage", {"stage": "conduct_research"})
    await researcher.conduct_research()
    emit_event("stage", {"stage": "write_report"})
    report = await researcher.write_report()

    source_urls: List[str] = list(researcher.get_source_urls() or [])
    research_sources = researcher.get_research_sources() or []
    sources = collect_sources(research_sources, source_urls)
    for s in sources:
        emit_event("source_found", {"url": s["url"], "title": s.get("title")})

    # costs the engine cannot report stay null (never rounded to 0)
    llm_cost: Optional[float]
    try:
        raw_cost = researcher.get_costs()
        llm_cost = float(raw_cost) if raw_cost else None
    except (TypeError, ValueError, AttributeError):
        llm_cost = None
    emit_event("cost", {"llm_cost_usd": llm_cost, "estimate": True, "search_api_cost_usd": 0.0})

    warnings = [
        "このエンジンは主張単位の構造化引用を返しません（レポート本文中の引用のみ）",
        "llm_cost_usd は gpt-researcher 内蔵の OpenAI 価格表による推定値です。"
        "ローカル／独自 LLM では実費と一致しません（取得できない場合は null）",
        "トークン使用量は gpt-researcher から取得できないため null です",
    ]

    visited_urls = getattr(researcher, "visited_urls", None)
    raw: Dict[str, Any] = {
        "engine": "gpt-researcher",
        "engine_version": ENGINE_VERSION,
        "source_urls": source_urls,
        "visited_urls": sorted(visited_urls) if isinstance(visited_urls, set) else visited_urls,
    }
    try:
        raw["step_costs"] = researcher.get_step_costs()
    except Exception:  # noqa: BLE001 - optional extra
        pass

    return {
        "kind": "result",
        "output_kind": "report",
        "summary": _summary_from_report(report),
        "report_markdown": report,
        "claims": [],
        "sources": sources,
        "metrics": {
            "searches": None,
            "sources": len(sources),
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "llm_cost_usd": llm_cost,
            "llm_cost_is_estimate": True,
            "search_api_cost_usd": 0.0,
            "infra_cost": "not_measured",
            "duration_seconds": round(time.monotonic() - started, 3),
        },
        "warnings": warnings,
        "raw": raw,
    }


def main() -> None:
    global _EMIT
    _EMIT = _setup_stdio()
    try:
        payload = json.load(sys.stdin)
        result = asyncio.run(_research(payload))
        _emit(result)
    except Exception as e:  # noqa: BLE001 - everything becomes a fatal line for the parent
        _emit({"kind": "fatal", "error": f"{type(e).__name__}: {e}"})
        sys.exit(1)


if __name__ == "__main__":
    main()
