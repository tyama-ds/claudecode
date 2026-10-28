"""open-deep-research worker subprocess (runs in the engine's own Python, 3.12+).

Started by engines/odr/engine.py with:
- environment: OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_API_BASE (or the
  ANTHROPIC_* trio), proxies with NO_PROXY for 127.0.0.1 — set by the parent
- stdin: one JSON document {"prompt", "language", "configurable",
  "recursion_limit"}; ``configurable`` carries allow_clarification=False,
  search_api="none", the four model names and mcp_config (the URL of the
  tool's in-process search server)

Prints JSONL to stdout:
  {"kind":"event","type":"stage|log|token_usage","payload":{...}}
  {"kind":"result", ...RunResult fields...}   or   {"kind":"fatal","error":...}

fd 1 is redirected to stderr right at start (open-deep-research print()s
on some failures, e.g. "Error loading MCP tools: ...") and the original
stdout is kept on a private duplicate used for the JSONL only.

This file is self-contained on purpose (standard library + the engine's
packages): it must not import deep_research_tool. The pure helpers at the
top (sources parsing, usage / state extraction) are also imported by
engine.py, which is why nothing above ``_research`` touches langchain.
Ported from deep-research-orchestrator's runners/open_deep_research/.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, TextIO, Tuple

_EMIT: Optional[TextIO] = None
ENGINE_VERSION = "0.0.16"

# ----------------------------------------------------------------------
# pure helpers (shared with engine.py)
# ----------------------------------------------------------------------

_SOURCES_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*(?:sources?|references?|出典|参考文献|引用文献|参考資料)\s*(?:一覧|リスト)?\s*:?\s*$",
    re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s<>\)\]\"'\\]+")
_LINE_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s*)?(?:\[?\d+\]?[.):]?\s*)?")
_MD_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")

_STAGE_LABELS = {
    "clarify_with_user": "clarify",
    "write_research_brief": "brief",
    "research_supervisor": "supervisor",
    "supervisor": "supervisor",
    "supervisor_tools": "supervisor_tools",
    "researcher": "researcher",
    "researcher_tools": "researcher_tools",
    "compress_research": "compress",
    "final_report_generation": "final_report",
}


def parse_sources_section(report_markdown: Optional[str]) -> List[Dict[str, Any]]:
    """The report's closing ``### Sources`` list (``[n] Title: URL``) as
    source dicts; [] when there is no such section (nothing is invented)."""
    if not report_markdown:
        return []
    lines = report_markdown.splitlines()
    start = None
    for i, line in enumerate(lines):
        if _SOURCES_HEADING_RE.match(line):
            start = i + 1          # the last Sources heading wins
    if start is None:
        return []
    sources: List[Dict[str, Any]] = []
    seen = set()
    for line in lines[start:]:
        if re.match(r"^\s{0,3}#{1,6}\s", line):
            break                  # next heading ends the section
        m = _URL_RE.search(line)
        if not m:
            continue
        url = m.group(0).rstrip(".,;:")
        if url in seen:
            continue
        seen.add(url)
        title = _LINE_PREFIX_RE.sub("", line[: m.start()]).strip()
        title = title.rstrip(":-–—").strip().strip("*_").strip()
        # "[Title](URL)" lines: the title is inside the brackets
        md = _MD_LINK_RE.search(line)
        if md and md.group(2).rstrip(".,;:") == url:
            title = md.group(1).strip()
        sources.append({"url": url, "title": title or None, "excerpt": None, "meta": {}})
    return sources


def urls_in_body(report_markdown: Optional[str]) -> List[Dict[str, Any]]:
    """Every URL cited in the body (markdown links first, then bare URLs),
    for a report without a Sources section. De-duplicated, in order."""
    if not report_markdown:
        return []
    sources: List[Dict[str, Any]] = []
    seen = set()
    for m in _MD_LINK_RE.finditer(report_markdown):
        url = m.group(2).rstrip(".,;:")
        if url not in seen:
            seen.add(url)
            sources.append({"url": url, "title": m.group(1).strip() or None,
                            "excerpt": None, "meta": {"from": "body"}})
    for m in _URL_RE.finditer(report_markdown):
        url = m.group(0).rstrip(".,;:")
        if url not in seen:
            seen.add(url)
            sources.append({"url": url, "title": None, "excerpt": None, "meta": {"from": "body"}})
    return sources


def extract_usage(obj: Any, _depth: int = 0) -> Tuple[int, int]:
    """Sum of AIMessage.usage_metadata inside one update chunk
    (input_tokens, output_tokens); (0, 0) when none."""
    if _depth > 8 or obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return (0, 0)
    total_in = 0
    total_out = 0
    usage = obj.get("usage_metadata") if isinstance(obj, dict) else getattr(obj, "usage_metadata", None)
    if isinstance(usage, dict):
        try:
            total_in += int(usage.get("input_tokens") or 0)
            total_out += int(usage.get("output_tokens") or 0)
        except (TypeError, ValueError):
            pass
    children: List[Any] = []
    if isinstance(obj, dict):
        children = [v for k, v in obj.items() if k != "usage_metadata"]
    elif isinstance(obj, (list, tuple, set)):
        children = list(obj)
    else:
        content = getattr(obj, "content", None)
        if isinstance(content, (list, dict)):
            children = [content]
    for child in children:
        i, o = extract_usage(child, _depth + 1)
        total_in += i
        total_out += o
    return (total_in, total_out)


def find_state_value(obj: Any, key: str, _depth: int = 0) -> Any:
    """A state key (final_report / research_brief …) anywhere in an update chunk."""
    if _depth > 6 or obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return None
    if isinstance(obj, dict):
        if key in obj and obj[key] is not None:
            return obj[key]
        for value in obj.values():
            found = find_state_value(value, key, _depth + 1)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found = find_state_value(item, key, _depth + 1)
            if found is not None:
                return found
    return None


def content_text(value: Any) -> Optional[str]:
    """AIMessage.content as text: a str as is; a content-block list (OpenAI
    Responses API models, Anthropic multi-block / cited replies) → its text
    blocks joined (as langchain-core's AIMessage.text does); else None."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for b in value:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
                parts.append(b["text"])
        return "".join(parts)
    return None


def usage_from_llm_result(response: Any) -> Tuple[int, int]:
    """(input, output) tokens of one LLM call's result (LLMResult with
    ChatGenerations carrying AIMessage.usage_metadata)."""
    total_in = total_out = 0
    for gens in getattr(response, "generations", None) or []:
        for g in gens or []:
            u = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
            if not isinstance(u, dict):
                continue
            try:
                total_in += int(u.get("input_tokens") or 0)
                total_out += int(u.get("output_tokens") or 0)
            except (TypeError, ValueError):
                pass
    return total_in, total_out


def _summary_from_report(report: str, limit: int = 400) -> Optional[str]:
    for block in (report or "").split("\n\n"):
        block = block.strip()
        if not block or block.startswith("#"):
            continue
        return " ".join(block.split())[:limit]
    return None


# ----------------------------------------------------------------------
# the subprocess
# ----------------------------------------------------------------------

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


async def _research(payload: Dict[str, Any]) -> Dict[str, Any]:
    started = time.monotonic()
    from langchain_core.callbacks.base import BaseCallbackHandler  # noqa: PLC0415 (engine env only)
    from langchain_core.messages import HumanMessage  # noqa: PLC0415
    from open_deep_research.deep_researcher import deep_researcher  # noqa: PLC0415

    class _UsageCounter(BaseCallbackHandler):
        """Token usage of EVERY LLM call, subgraphs included. Walking the
        "updates" chunks misses the researcher's calls (its subgraph only
        outputs compressed_research / raw_notes) and the structured-output
        / compression calls, and counts the supervisor's twice."""

        def __init__(self) -> None:
            super().__init__()
            self.input = 0
            self.output = 0
            self.calls = 0

        def on_llm_end(self, response, **kwargs) -> None:
            i, o = usage_from_llm_result(response)
            self.input += i
            self.output += o
            self.calls += 1

    counter = _UsageCounter()
    config: Dict[str, Any] = {
        "configurable": payload["configurable"],
        "recursion_limit": int(payload.get("recursion_limit", 50)),
        "callbacks": [counter],
    }
    inputs = {"messages": [HumanMessage(content=payload["prompt"])]}

    final_report: Optional[str] = None
    research_brief: Optional[str] = None
    node_counts: Dict[str, int] = {}

    async for chunk in deep_researcher.astream(inputs, config=config,
                                               stream_mode="updates", subgraphs=True):
        namespace: tuple = ()
        update = chunk
        if isinstance(chunk, tuple) and len(chunk) == 2:
            namespace, update = chunk
        if not isinstance(update, dict):
            continue
        for node, delta in update.items():
            stage = _STAGE_LABELS.get(str(node), str(node))
            node_counts[stage] = node_counts.get(stage, 0) + 1
            emit_event("stage", {"stage": stage, "node": str(node),
                                 "namespace": [str(n) for n in namespace],
                                 "count": node_counts[stage]})
            fr = content_text(find_state_value(delta, "final_report"))
            if fr and fr.strip():
                final_report = fr
            rb = content_text(find_state_value(delta, "research_brief"))
            if rb and rb.strip() and rb != research_brief:
                research_brief = rb
                emit_event("log", {"message": "調査方針: " + " ".join(rb.split())[:300]})

    usage_in, usage_out = counter.input, counter.output
    got_usage = bool(usage_in or usage_out)
    if got_usage:
        emit_event("token_usage", {"prompt_tokens": usage_in, "completion_tokens": usage_out})

    if not final_report:
        raise RuntimeError("final_report が生成されませんでした（graph が完了しなかった可能性。"
                           "recursion_limit・制限時間・LLM のエラーを確認してください）")
    if final_report.startswith("Error generating final report"):
        raise RuntimeError(f"open-deep-research 内部エラー: {final_report[:500]}")

    sources = parse_sources_section(final_report)
    warnings = [
        "このエンジンは主張単位の構造化引用を返しません（レポート末尾の Sources リストと本文の URL のみ）",
        "llm_cost_usd は価格表が不明なため算出しません（null）",
    ]
    if not sources:
        sources = urls_in_body(final_report)
        if sources:
            warnings.append("レポートに Sources セクションが無いため、本文中の URL を出典として扱いました")
        else:
            warnings.append("レポートに Sources セクションも URL も無いため、出典は空です")
    if not got_usage:
        warnings.append("トークン使用量（usage_metadata）を取得できなかったため null です")

    return {
        "kind": "result",
        "output_kind": "report",
        "summary": _summary_from_report(final_report),
        "report_markdown": final_report,
        "claims": [],
        "sources": sources,
        "metrics": {
            "searches": None,
            "sources": len(sources),
            "prompt_tokens": usage_in if got_usage else None,
            "completion_tokens": usage_out if got_usage else None,
            "total_tokens": (usage_in + usage_out) if got_usage else None,
            "llm_cost_usd": None,
            "llm_cost_is_estimate": None,
            "search_api_cost_usd": 0.0,
            "infra_cost": "not_measured",
            "duration_seconds": round(time.monotonic() - started, 3),
        },
        "warnings": warnings,
        "raw": {
            "engine": "open-deep-research",
            "engine_version": ENGINE_VERSION,
            "research_brief": research_brief,
            "node_counts": node_counts,
            "llm_calls": counter.calls,
        },
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
