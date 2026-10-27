"""Engine result → ResearchSession + Evidence Locker.

Every external engine returns the same ``RunResult`` (report markdown,
sources, claims, metrics). This module turns that into exactly what the
rest of the pipeline expects from the built-in researcher:

- chapters = the report's level-2 headings (``report_loader.split_report_sections``;
  reference / sources sections are dropped)
- sources are de-duplicated by canonical URL and assigned to chapters:
  explicit citations first (``[n]``, ``[title](url)``, bare URLs in the
  chapter), then a 2-gram containment match of title + excerpt against the
  chapter text, else chapter 1 — a source is NEVER dropped
- citations are bridged to the pipeline's stable per-chapter form:
  ``[n]`` / ``[title](url)`` → ``[SOURCE k]`` where k is the source's
  position in that chapter's evidence list (the finalization runner
  registers ``[SOURCE k]`` from ``extracted_content`` order, so the two
  are built together here)
- each source becomes locker evidence: the page is fetched through the
  tool's search client; when that fails the engine's excerpt is
  registered as "excerpt only" (``metadata.excerpt_only``) so the
  verification can still cite it and the UI can show what was not read
- the raw engine output is saved under ``engine_<session>/`` (result.json
  with events, report.md) for audit / later re-normalization
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from ..evidence.locker import EvidenceLocker, EvidenceType
from ..report.report_loader import split_report_sections
from ..research.query_generator import (ResearchPlan, TableOfContents,
                                        TableOfContentsItem)
from ..research.researcher import ResearchSession, ResearchState
from ..utils.concurrency import ContextThreadPoolExecutor, effective_workers
from ..verification.claim_verifier import _bigrams, _containment
from .runner_core.models import RunEvent, RunResult, SourceRecord

# --- URL canonicalization (identical to the orchestrator's normalizer) ---

_TRACKING_PARAMS = re.compile(r"^(utm_|fbclid|gclid|ref_|mc_eid|mc_cid)")


def canonicalize_url(url: str) -> str:
    """tracking parameter・fragment 除去、host 小文字化、末尾 slash 正規化。"""
    try:
        parsed = urlparse((url or "").strip())
    except ValueError:
        return url
    if not parsed.scheme:
        return url
    query = [
        (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if not _TRACKING_PARAMS.match(k.lower())
    ]
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    try:
        port = parsed.port
    except ValueError:
        port = None
    return urlunparse(
        (
            parsed.scheme.lower(),
            (parsed.hostname or "").lower() + (f":{port}" if port else ""),
            path,
            parsed.params,
            urlencode(query),
            "",  # fragment 除去
        )
    )


# --- "### Sources" section parser (from the orchestrator's ODR runner) ---

_SOURCES_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*(?:sources?|references?|出典|参考文献|引用文献)\s*:?\s*$",
    re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s<>\)\]]+")
_LINE_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s*)?(?:\[?\d+\]?[.):]?\s*)?")


def parse_sources_section(report_markdown: Optional[str]) -> List[Dict[str, Any]]:
    """Report 末尾の Sources / 出典リスト ``[n] Title: URL`` を source dict 化する。
    セクションが無い場合は []（捏造しない）。"""
    if not report_markdown:
        return []
    lines = report_markdown.splitlines()
    start = None
    for i, line in enumerate(lines):
        if _SOURCES_HEADING_RE.match(line):
            start = i + 1  # 最後の Sources 見出しを採用
    if start is None:
        return []
    sources: List[Dict[str, Any]] = []
    seen = set()
    for line in lines[start:]:
        if re.match(r"^\s{0,3}#{1,6}\s", line):
            break  # 次の見出しでセクション終了
        url_match = _URL_RE.search(line)
        if not url_match:
            continue
        url = url_match.group(0).rstrip(".,;:")
        if url in seen:
            continue
        seen.add(url)
        title = _LINE_PREFIX_RE.sub("", line[: url_match.start()]).strip()
        title = title.rstrip(":-–—").strip().strip("*_").strip()
        sources.append({"url": url, "title": title or None, "excerpt": None, "meta": {}})
    return sources


# --- citations ---------------------------------------------------------

# "[3]" / "[1][2]" but not a markdown link "[3](http...)" and not [SOURCE 3]
_NUM_CITE_RE = re.compile(r"\[(\d{1,3})\](?!\()")
_MD_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")


def dedupe_sources(sources: List[SourceRecord]
                   ) -> Tuple[List[SourceRecord], Dict[int, int], Dict[str, int]]:
    """Unique sources by canonical URL.

    Returns (unique list, {engine's 1-based number → unique index},
    {canonical url → unique index}). The first occurrence keeps its
    title / excerpt; later duplicates only fill in blanks."""
    unique: List[SourceRecord] = []
    index_map: Dict[int, int] = {}
    by_canon: Dict[str, int] = {}
    for n, src in enumerate(sources, 1):
        url = (src.url or "").strip()
        if not url:
            continue
        key = canonicalize_url(url)
        idx = by_canon.get(key)
        if idx is None:
            idx = len(unique)
            unique.append(src)
            by_canon[key] = idx
        else:
            kept = unique[idx]
            if not kept.title and src.title:
                kept.title = src.title
            if not kept.excerpt and src.excerpt:
                kept.excerpt = src.excerpt
        index_map[n] = idx
    return unique, index_map, by_canon


def _cited_in(text: str, index_map: Dict[int, int],
              by_canon: Dict[str, int]) -> List[int]:
    """Unique-source indexes cited in a chapter, in first-appearance order."""
    events: List[Tuple[int, int]] = []
    for m in _NUM_CITE_RE.finditer(text):
        idx = index_map.get(int(m.group(1)))
        if idx is not None:
            events.append((m.start(), idx))
    for m in _MD_LINK_RE.finditer(text):
        idx = by_canon.get(canonicalize_url(m.group(2)))
        if idx is not None:
            events.append((m.start(), idx))
    for m in _URL_RE.finditer(text):
        idx = by_canon.get(canonicalize_url(m.group(0).rstrip(".,;:)")))
        if idx is not None:
            events.append((m.start(), idx))
    ordered: List[int] = []
    for _, idx in sorted(events):
        if idx not in ordered:
            ordered.append(idx)
    return ordered


def assign_sources_to_sections(sections: List[Dict[str, str]],
                               sources: List[SourceRecord],
                               index_map: Dict[int, int],
                               by_canon: Dict[str, int],
                               min_containment: float = 0.35
                               ) -> Dict[int, List[int]]:
    """{chapter position → ordered unique-source indexes}.

    Explicit citations win; an uncited source goes to the chapter whose
    text contains most of the source's title + excerpt 2-grams (at least
    ``min_containment``), else to chapter 1. Every source lands somewhere."""
    assignments: Dict[int, List[int]] = {}
    for pos, sec in enumerate(sections):
        assignments[pos] = _cited_in(sec.get("content", ""), index_map, by_canon)
    placed = {idx for v in assignments.values() for idx in v}
    grams = [_bigrams(sec.get("content", "")) for sec in sections]
    for idx, src in enumerate(sources):
        if idx in placed:
            continue
        needle = _bigrams(f"{src.title or ''} {src.excerpt or ''}".strip())
        best, best_score = 0, 0.0
        for pos, hay in enumerate(grams):
            score = _containment(needle, hay) if needle else 0.0
            if score > best_score:
                best, best_score = pos, score
        target = best if best_score >= min_containment else 0
        assignments.setdefault(target, []).append(idx)
    return assignments


def bridge_citations(text: str, ordered: List[int], index_map: Dict[int, int],
                     by_canon: Dict[str, int]) -> str:
    """``[n]`` → ``[SOURCE k]`` and ``[title](url)`` → ``title [SOURCE k]``
    for the sources of this chapter (k = 1-based position in ``ordered``).
    Unknown numbers / links are left untouched."""
    position = {idx: k + 1 for k, idx in enumerate(ordered)}

    def _num(m):
        idx = index_map.get(int(m.group(1)))
        k = position.get(idx) if idx is not None else None
        return f"[SOURCE {k}]" if k else m.group(0)

    def _link(m):
        idx = by_canon.get(canonicalize_url(m.group(2)))
        k = position.get(idx) if idx is not None else None
        return f"{m.group(1)} [SOURCE {k}]" if k else m.group(0)

    text = _NUM_CITE_RE.sub(_num, text or "")
    return _MD_LINK_RE.sub(_link, text)


# --- session -----------------------------------------------------------

def build_session_from_result(result: RunResult, *, query: str,
                              requirements: str = "", language: str = "ja",
                              engine_id: str, engine_name: str,
                              session_id: Optional[str] = None
                              ) -> Tuple[ResearchSession, List[SourceRecord],
                                         Dict[str, List[int]]]:
    """A COMPLETED ResearchSession whose chapters are the engine report's
    sections (citations bridged), the unique sources, and
    {section id → ordered unique-source indexes} for the evidence step.
    Raises ValueError when the engine returned no body."""
    text = (result.report_markdown or "").strip()
    if not text and (result.summary or "").strip():
        text = (result.summary or "").strip()
    if not text:
        raise ValueError(f"{engine_name} は本文を返しませんでした")
    title, sections = split_report_sections(text, fallback_title=query)
    if not sections:
        raise ValueError(f"{engine_name} の本文に章が見つかりません")

    sources = list(result.sources or [])
    if not sources:
        sources = [SourceRecord(**d) for d in parse_sources_section(result.report_markdown)]
    unique, index_map, by_canon = dedupe_sources(sources)
    assignments = assign_sources_to_sections(sections, unique, index_map, by_canon)

    items = [TableOfContentsItem(section=str(i + 1), title=s["title"],
                                 description=s["title"])
             for i, s in enumerate(sections)]
    plan_title = title or query or "report"
    toc = TableOfContents(title=plan_title, items=items)
    plan = ResearchPlan(title=plan_title, summary=(result.summary or "").strip(),
                        table_of_contents=toc, search_queries=[],
                        methodology_notes=f"external engine: {engine_id}")
    session = ResearchSession(query=query, requirements=requirements or "",
                              state=ResearchState.COMPLETED, research_plan=plan)
    if session_id:
        session.session_id = session_id
    section_sources: Dict[str, List[int]] = {}
    for pos, (item, s) in enumerate(zip(items, sections)):
        ordered = list(assignments.get(pos, []))
        content = bridge_citations(s["content"], ordered, index_map, by_canon)
        session.section_contents[item.section] = {
            "title": s["title"], "content": content,
            "summary": "", "confidence": "unknown",
            "sources": [unique[i].url for i in ordered],
            "extracted_content": [],
        }
        item.content = content
        section_sources[item.section] = ordered
    session.completed_sections = [item.section for item in items]
    session.run_config = {"source": "engine", "research_engine": engine_id,
                          "language": language}
    return session, unique, section_sources


# --- evidence ----------------------------------------------------------

def seed_evidence_from_sources(session: ResearchSession,
                               evidence_locker: EvidenceLocker,
                               sources: List[SourceRecord],
                               section_sources: Dict[str, List[int]], *,
                               search_client=None, engine_id: str,
                               engine_name: str, topic: str,
                               parallel_max_workers: int = 4,
                               fetch: bool = True, min_chars: int = 200,
                               cancel_check=None) -> Dict[str, int]:
    """Register every source as chapter evidence, in the chapters'
    citation order (so ``[SOURCE k]`` resolves). Pages are fetched
    concurrently through the search client; a page that cannot be read
    is registered from the engine's excerpt and marked
    ``metadata.excerpt_only`` (never dropped, never invented)."""
    def _fetch(idx: int):
        src = sources[idx]
        out = {"text": "", "title": src.title or src.url, "error": None,
               "metadata": {}}
        if cancel_check is not None:
            cancel_check()
        if not fetch:
            out["error"] = "取得を省略（設定）"
        elif search_client is None or not hasattr(search_client, "get_page_content"):
            out["error"] = "検索クライアントなし"
        else:
            try:
                page = search_client.get_page_content(src.url)
                text = (getattr(page, "text_content", "") or "").strip()
                meta = dict(getattr(page, "metadata", {}) or {})
                if meta.get("error"):
                    out["error"] = str(meta["error"])
                elif len(text) >= min_chars:
                    out["text"] = text
                    out["title"] = getattr(page, "title", "") or out["title"]
                    out["metadata"] = meta
                else:
                    out["error"] = f"本文が短すぎます（{len(text)} 文字）"
            except Exception as e:  # noqa: BLE001 - one page must not kill the run
                out["error"] = f"{type(e).__name__}: {e}"
        return idx, out

    fetched: Dict[int, Dict[str, Any]] = {}
    serial = bool(getattr(search_client, "requires_serial_access", False))
    workers = 1 if serial else effective_workers(parallel_max_workers, 8, len(sources))
    if sources:
        if workers > 1:
            with ContextThreadPoolExecutor(max_workers=workers) as ex:
                for idx, out in ex.map(_fetch, range(len(sources))):
                    fetched[idx] = out
        else:
            for idx in range(len(sources)):
                fetched[idx] = _fetch(idx)[1]

    search_query = f"{engine_name}: {topic}"
    for sid, ordered in section_sources.items():
        sdata = session.section_contents.get(sid)
        if sdata is None:
            continue
        sdata.setdefault("extracted_content", [])
        sdata.setdefault("sources", [])
        for idx in ordered:
            src = sources[idx]
            f = fetched.get(idx) or {"text": "", "title": src.title or src.url,
                                     "error": "not fetched", "metadata": {}}
            excerpt_only = not f["text"]
            body = f["text"] or (src.excerpt or "").strip() or (src.title or src.url)
            metadata = dict(f["metadata"])
            metadata.update({"engine": engine_id, "engine_rank": idx + 1,
                             "excerpt_only": excerpt_only})
            if f.get("error"):
                metadata["fetch_error"] = str(f["error"])
            evidence_locker.add_evidence(
                url=src.url, title=f["title"], content_excerpt=body[:500],
                extracted_text=body, evidence_type=EvidenceType.WEB_PAGE,
                search_query=search_query, section_reference=sid,
                metadata=metadata)
            sdata["extracted_content"].append({
                "title": f["title"], "url": src.url,
                "content": body[:2000], "raw_content": body,
                "key_points": [], "relevance_score": 0.3 if excerpt_only else 0.5,
                "excerpt_only": excerpt_only,
            })
            if src.url not in sdata["sources"]:
                sdata["sources"].append(src.url)
    full = sum(1 for f in fetched.values() if f["text"])
    return {"sources": len(sources), "fetched": full,
            "excerpt_only": len(sources) - full}


# --- artifacts ---------------------------------------------------------

def save_engine_artifacts(output_dir, session_id: str, engine_id: str,
                          result: RunResult, events: List[RunEvent],
                          request_summary: Optional[Dict[str, Any]] = None
                          ) -> Dict[str, str]:
    """``engine_<session>/result.json`` (result + events + non-secret
    request) and ``report.md`` (the engine's own markdown)."""
    d = Path(output_dir) / f"engine_{session_id}"
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "engine_id": engine_id,
        "saved_at": datetime.now().isoformat(),
        "request": request_summary or {},
        "result": result.model_dump(mode="json"),
        "events": [e.model_dump(mode="json") for e in events],
    }
    result_path = d / "result.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    report_path = d / "report.md"
    report_path.write_text(result.report_markdown or "", encoding="utf-8")
    return {"engine_result": str(result_path), "engine_report": str(report_path)}
