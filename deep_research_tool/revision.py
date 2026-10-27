"""Feedback re-research (レポートへのフィードバック → 再調査・追加調査).

The user reads a finished report and answers with opinions, URLs,
earlier research to lean on and a way of working (add / revise / full).
This module turns that into a REVISION PLAN — which chapters to keep,
which to research again with what instructions and queries, which to
add — and prepares a NEW session (a new job, the original untouched)
whose kept chapters are reused verbatim and whose other chapters the
normal research loop processes through ``Researcher.resume_research``.
Reference research contributes its evidence pages as local documents,
so the same URL is never fetched twice and the chapters can cite them.

Nothing here calls the network; the LLM is used once for the plan (with
a deterministic fallback when its answer is unusable).
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .evidence.locker import EvidenceLocker
from .research.query_generator import (ResearchPlan, TableOfContents,
                                       TableOfContentsItem)
from .research.researcher import ResearchSession, ResearchState
from .utils.helpers import ResearchWarnings, extract_json_from_response

MODES = ("add", "revise", "full")


class NoChangesRequested(ValueError):
    """The planner read the feedback and found nothing to revise or add
    (approving remarks only). Not a planner failure: no fallback."""
MODE_LABELS = {"add": "追加調査", "revise": "再調査", "full": "全面再調査"}
ACTION_LABELS = {"keep": "そのまま", "revise": "直す", "add": "追加"}
MAX_FEEDBACK_CHARS = 6000
MAX_REFERENCE_DOCS = 80
REFERENCE_MIN_CHARS = 200


@dataclass
class RevisionAction:
    section: str
    action: str                     # keep | revise | add
    title: str
    instructions: str = ""
    queries: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"section": self.section, "action": self.action, "title": self.title,
                "instructions": self.instructions, "queries": list(self.queries),
                "reason": self.reason}


@dataclass
class RevisionPlan:
    mode: str
    actions: List[RevisionAction] = field(default_factory=list)
    requirements_addendum: str = ""
    acknowledged: List[str] = field(default_factory=list)
    notes: str = ""
    source: str = "llm"             # llm | fallback | none (full mode)

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "actions": [a.to_dict() for a in self.actions],
                "requirements_addendum": self.requirements_addendum,
                "acknowledged": list(self.acknowledged), "notes": self.notes,
                "source": self.source}

    def by_action(self, name: str) -> List[RevisionAction]:
        return [a for a in self.actions if a.action == name]

    @property
    def changes(self) -> List[RevisionAction]:
        return [a for a in self.actions if a.action in ("revise", "add")]


# ----------------------------------------------------------------------
# the parent report as the planner sees it
# ----------------------------------------------------------------------

def flat_sections(session: ResearchSession) -> List[TableOfContentsItem]:
    plan = getattr(session, "research_plan", None)
    if plan is None:
        return []
    return list(plan.table_of_contents.get_flat_sections())


def section_digest(session: ResearchSession, item: TableOfContentsItem,
                   limit: int = 240) -> str:
    data = session.section_contents.get(item.section) or {}
    text = (data.get("summary") or "").strip() or (data.get("content") or "").strip()
    text = " ".join(text.split())
    return text[:limit]


def _next_section_number(sections: Sequence[TableOfContentsItem]) -> int:
    top = []
    for s in sections:
        head = str(s.section).split(".")[0]
        if head.isdigit():
            top.append(int(head))
    return (max(top) + 1) if top else 1


# ----------------------------------------------------------------------
# the plan: LLM first, heuristics as the fallback
# ----------------------------------------------------------------------

def build_revision_prompt(session: ResearchSession, feedback: str, mode: str, *,
                          language: str = "ja", k: int = 3,
                          extra_instructions: str = "",
                          reference_summaries: Optional[List[str]] = None) -> str:
    sections = flat_sections(session)
    toc_lines = []
    for item in sections:
        toc_lines.append(f"- {item.section}: {item.title} — {section_digest(session, item) or '(本文なし)'}")
    refs = ""
    if reference_summaries:
        refs = "\n".join(f"- {r}" for r in reference_summaries)
    allow_add = mode == "add"
    if language == "ja":
        return f"""あなたは調査レポートの編集者です。読者からのフィードバックを、章ごとの修正計画（JSON）に変換してください。

【調査テーマ】{session.query}
【調査要件】{session.requirements or "なし"}

【現在のレポートの章と要旨】
{chr(10).join(toc_lines) or "(章なし)"}

【読者のフィードバック】
{feedback.strip()[:MAX_FEEDBACK_CHARS]}
{("【追加の指示】" + chr(10) + extra_instructions.strip()) if extra_instructions.strip() else ""}
{("【参照できる過去の調査】" + chr(10) + refs) if refs else ""}

やり方: {MODE_LABELS.get(mode, mode)}（{"指摘された章を直し、必要なら章を追加する" if allow_add else "指摘された章を調べ直して書き直す。章の追加はしない"}）

規則:
- フィードバックで指摘・要求された章だけを "revise" にし、それ以外は "keep" にする。「妥当」「問題ない」などの肯定は acknowledged に記録して章は keep にする。
- revise の instructions は、その章で何をどう変えるか（更新する数値・加える比較・使う資料）を具体的に 1〜3 文で書く。
- queries は、その指示を満たすための Web 検索クエリを {k} 件まで、検索語として自然な短い形（固有名詞・年・指標を含める）で書く。
{"- フィードバックが新しい話題を求めていれば additions に章を追加する（title・instructions・queries）。" if allow_add else "- additions は空の配列にする。"}
- requirements_addendum には、今後の執筆全体に効く要件（例: 2025 年の数値を優先、3 社比較）を 1〜2 文で書く。無ければ空文字。

JSON のみを出力:
{{"items": [{{"section": "2", "action": "revise", "instructions": "...", "queries": ["...", "..."], "reason": "..."}}, {{"section": "1", "action": "keep"}}],
 "additions": [{{"title": "...", "instructions": "...", "queries": ["..."]}}],
 "requirements_addendum": "...",
 "acknowledged": ["..."],
 "notes": "..."}}"""
    return f"""You are the editor of a research report. Convert the reader's feedback into a per-chapter revision plan (JSON).

Topic: {session.query}
Requirements: {session.requirements or "none"}

Current chapters and digests:
{chr(10).join(toc_lines) or "(none)"}

Reader feedback:
{feedback.strip()[:MAX_FEEDBACK_CHARS]}
{("Additional instructions:" + chr(10) + extra_instructions.strip()) if extra_instructions.strip() else ""}
{("Earlier research available:" + chr(10) + refs) if refs else ""}

Mode: {mode} ({"revise the chapters the feedback points at and add chapters when it asks for new topics" if allow_add else "revise the chapters the feedback points at; never add chapters"})

Rules:
- Only chapters the feedback criticises or asks to change become "revise"; everything else is "keep". Approving remarks go to acknowledged and keep the chapter.
- instructions: 1-3 concrete sentences (numbers to update, comparisons to add, sources to use).
- queries: up to {k} short web search queries (names, years, metrics).
{"- additions: new chapters (title, instructions, queries) when the feedback asks for new topics." if allow_add else "- additions must be an empty array."}
- requirements_addendum: 1-2 sentences that apply to all writing from now on, or an empty string.

Output JSON only:
{{"items": [{{"section": "2", "action": "revise", "instructions": "...", "queries": ["..."], "reason": "..."}}, {{"section": "1", "action": "keep"}}],
 "additions": [{{"title": "...", "instructions": "...", "queries": ["..."]}}],
 "requirements_addendum": "...", "acknowledged": ["..."], "notes": "..."}}"""


def _clean_queries(raw, title: str, instructions: str, k: int) -> List[str]:
    out: List[str] = []
    for q in (raw or []) if isinstance(raw, list) else []:
        q = " ".join(str(q).split())
        if q and q not in out:
            out.append(q)
    if not out:
        hint = " ".join(instructions.split())[:40]
        out.append(f"{title} {hint}".strip() if hint else title)
    return out[:max(1, k)]


def parse_revision_response(text: str, session: ResearchSession, mode: str,
                            k: int = 3) -> RevisionPlan:
    """The LLM's JSON → RevisionPlan. Raises ValueError when there is no
    usable plan (no JSON, or nothing to change)."""
    data = extract_json_from_response(text or "")
    if not isinstance(data, dict):
        raise ValueError("no JSON in the revision answer")
    sections = flat_sections(session)
    by_id = {str(s.section): s for s in sections}
    decided: Dict[str, RevisionAction] = {}
    for item in data.get("items") or []:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("section") or "").strip()
        if sid not in by_id:
            continue
        action = str(item.get("action") or "keep").strip().lower()
        if action not in ("keep", "revise"):
            action = "revise" if action in ("update", "rewrite", "fix", "change") else "keep"
        title = by_id[sid].title
        instructions = " ".join(str(item.get("instructions") or "").split())
        decided[sid] = RevisionAction(
            section=sid, action=action, title=title,
            instructions=instructions if action == "revise" else "",
            queries=_clean_queries(item.get("queries"), title, instructions, k) if action == "revise" else [],
            reason=" ".join(str(item.get("reason") or "").split())[:300])
    actions = [decided.get(str(s.section)) or RevisionAction(section=str(s.section), action="keep", title=s.title)
               for s in sections]
    if mode == "add":
        next_num = _next_section_number(sections)
        for add in data.get("additions") or []:
            if not isinstance(add, dict):
                continue
            title = " ".join(str(add.get("title") or "").split())
            if not title:
                continue
            instructions = " ".join(str(add.get("instructions") or "").split())
            actions.append(RevisionAction(
                section=str(next_num), action="add", title=title, instructions=instructions,
                queries=_clean_queries(add.get("queries"), title, instructions, k),
                reason=" ".join(str(add.get("reason") or "").split())[:300]))
            next_num += 1
    plan = RevisionPlan(
        mode=mode, actions=actions,
        requirements_addendum=" ".join(str(data.get("requirements_addendum") or "").split())[:800],
        acknowledged=[" ".join(str(a).split())[:300] for a in (data.get("acknowledged") or [])
                      if str(a).strip()][:20],
        notes=" ".join(str(data.get("notes") or "").split())[:600],
        source="llm")
    if not plan.changes:
        raise NoChangesRequested("the plan changes nothing")
    return plan


_SECTION_REF = re.compile(r"(?:第\s*(\d+)\s*[章節]|(\d+)\s*章|section\s*(\d+)|chapter\s*(\d+)|(?<![\d.])(\d+)\.\s)",
                          re.IGNORECASE)


def fallback_revision_plan(session: ResearchSession, feedback: str, mode: str,
                           k: int = 3) -> RevisionPlan:
    """No LLM plan: chapters the feedback names (第2章 / 2章 / section 2 /
    the chapter title) are revised with the whole feedback as the
    instruction; when nothing is named, every chapter is revised."""
    sections = flat_sections(session)
    text = feedback or ""
    named = set()
    for m in _SECTION_REF.finditer(text):
        num = next((g for g in m.groups() if g), None)
        if num:
            named.add(num)
    low = text.lower()
    for s in sections:
        title = (s.title or "").strip()
        if title and len(title) >= 2 and title.lower() in low:
            named.add(str(s.section).split(".")[0])
    instructions = " ".join(text.split())[:600]
    hint = instructions[:40]
    actions: List[RevisionAction] = []
    for s in sections:
        head = str(s.section).split(".")[0]
        hit = (head in named or str(s.section) in named) if named else True
        if hit:
            actions.append(RevisionAction(section=str(s.section), action="revise", title=s.title,
                                          instructions=instructions,
                                          queries=[f"{s.title} {hint}".strip()][:k],
                                          reason="フィードバックで言及（規則による判定）" if named else
                                                 "章の指定が無いため全章を対象"))
        else:
            actions.append(RevisionAction(section=str(s.section), action="keep", title=s.title))
    return RevisionPlan(mode=mode, actions=actions, requirements_addendum="",
                        notes="LLM の修正計画が使えなかったため、規則で対象章を決めました",
                        source="fallback")


def plan_revision(llm, session: ResearchSession, feedback: str, mode: str = "add", *,
                  language: str = "ja", k: int = 3, extra_instructions: str = "",
                  reference_summaries: Optional[List[str]] = None) -> RevisionPlan:
    """The revision plan for this feedback. ``full`` needs no plan: the
    feedback becomes an addendum to the requirements and the research
    starts from scratch."""
    if mode not in MODES:
        raise ValueError(f"unknown revision mode: {mode}")
    feedback = (feedback or "").strip()
    if not feedback:
        raise ValueError("feedback text is required")
    if mode == "full":
        return RevisionPlan(mode="full", actions=[], requirements_addendum=feedback[:2000],
                            source="none")
    prompt = build_revision_prompt(session, feedback, mode, language=language, k=k,
                                   extra_instructions=extra_instructions,
                                   reference_summaries=reference_summaries)
    try:
        response = llm.generate(prompt)
        text = getattr(response, "content", response)
        return parse_revision_response(str(text), session, mode, k)
    except NoChangesRequested:
        # a valid answer that keeps every chapter: the caller reports it
        return RevisionPlan(mode=mode, actions=[
            RevisionAction(section=str(s.section), action="keep", title=s.title)
            for s in flat_sections(session)], source="llm")
    except Exception as e:  # noqa: BLE001 - fall back, never abort on the planner
        ResearchWarnings.get_instance().add(
            "MEDIUM", "Revision",
            f"修正計画を LLM から得られなかったため規則で決めました（{e}）")
        return fallback_revision_plan(session, feedback, mode, k)


# ----------------------------------------------------------------------
# what the plan review shows
# ----------------------------------------------------------------------

def review_plan_for(rp: RevisionPlan, session: ResearchSession) -> ResearchPlan:
    """The revision plan as a ResearchPlan: the plan review panel lists
    every chapter with its action in the title and the instructions as
    the description, and the new search queries."""
    items: List[TableOfContentsItem] = []
    for a in rp.actions:
        label = ACTION_LABELS.get(a.action, a.action)
        desc = a.instructions or ("変更なし（本文と根拠を再利用）" if a.action == "keep" else "")
        if a.reason and a.action != "keep":
            desc = f"{desc}（{a.reason}）" if desc else a.reason
        items.append(TableOfContentsItem(section=a.section, title=f"【{label}】{a.title}",
                                         description=desc))
    parent_title = getattr(getattr(session, "research_plan", None), "title", "") or session.query
    summary_bits = []
    if rp.acknowledged:
        summary_bits.append("記録のみ: " + " / ".join(rp.acknowledged))
    if rp.requirements_addendum:
        summary_bits.append("追加要件: " + rp.requirements_addendum)
    if rp.notes:
        summary_bits.append(rp.notes)
    queries = [q for a in rp.changes for q in a.queries]
    return ResearchPlan(
        title=f"修正計画（{MODE_LABELS.get(rp.mode, rp.mode)}）: {parent_title}",
        summary="\n".join(summary_bits),
        table_of_contents=TableOfContents(title=parent_title, items=items),
        search_queries=queries,
        methodology_notes="feedback revision")


# ----------------------------------------------------------------------
# the new session
# ----------------------------------------------------------------------

def _pad_queries(queries: List[str], title: str, k: int) -> List[str]:
    out = [q for q in queries if q][:k]
    fillers = [f"{title} 最新", f"{title} 動向", f"{title} 統計", f"{title} 事例"]
    i = 0
    while len(out) < k:
        cand = fillers[i % len(fillers)] if i < len(fillers) else f"{title} {i}"
        if cand not in out:
            out.append(cand)
        i += 1
    return out[:k]


def build_revised_session(parent: ResearchSession, rp: RevisionPlan, *, k: int = 3,
                          session_id: Optional[str] = None,
                          feedback: str = "") -> ResearchSession:
    """A new session that reuses the kept chapters and leaves the revised
    / added ones for the research loop (resume_research semantics).

    The plan's search_queries are laid out in per-section slices of k
    (the loop hands section i the i-th slice), so revised and added
    chapters get their own queries and kept chapters harmless fillers."""
    if parent.research_plan is None:
        raise ValueError("the original research has no plan to revise")
    plan = ResearchPlan.from_dict(copy.deepcopy(parent.research_plan.to_dict()))
    by_action = {a.section: a for a in rp.actions}
    for item in plan.table_of_contents.get_flat_sections():
        a = by_action.get(str(item.section))
        item.status = "completed" if (a is None or a.action == "keep") else "pending"
        if a is not None and a.action == "revise":
            item.description = (item.description or "").strip()
            if a.instructions:
                item.description = (item.description + "\n" if item.description else "") + \
                    f"【修正指示】{a.instructions}"
            item.content = ""
            item.sources = []
    for a in rp.by_action("add"):
        plan.table_of_contents.items.append(TableOfContentsItem(
            section=a.section, title=a.title,
            description=(f"【追加の章】{a.instructions}" if a.instructions else "【追加の章】"),
            status="pending"))
    ordered = plan.table_of_contents.get_flat_sections()
    queries: List[str] = []
    for item in ordered:
        a = by_action.get(str(item.section))
        if a is not None and a.action in ("revise", "add"):
            queries.extend(_pad_queries(list(a.queries), item.title, k))
        else:
            queries.extend(_pad_queries([], item.title, k))
    plan.search_queries = queries
    plan.methodology_notes = (plan.methodology_notes + "\n" if plan.methodology_notes else "") + \
        f"feedback revision ({MODE_LABELS.get(rp.mode, rp.mode)}) of session {parent.session_id}"

    requirements = parent.requirements or ""
    if rp.requirements_addendum:
        requirements = (requirements + "\n\n" if requirements else "") + \
            f"【追加要件（フィードバック）】{rp.requirements_addendum}"
    session = ResearchSession(query=parent.query, requirements=requirements,
                              state=ResearchState.RESEARCHING, research_plan=plan)
    if session_id:
        session.session_id = session_id
    kept: List[str] = []
    for item in ordered:
        sid = str(item.section)
        a = by_action.get(sid)
        if a is None or a.action == "keep":
            data = parent.section_contents.get(sid)
            if data is not None:
                session.section_contents[sid] = copy.deepcopy(data)
                kept.append(sid)
            else:
                session.section_contents[sid] = {"title": item.title, "content": "", "summary": "",
                                                 "confidence": "unknown", "sources": [],
                                                 "extracted_content": []}
        else:
            previous = (parent.section_contents.get(sid) or {}).get("content", "")
            session.section_contents[sid] = {
                "title": item.title, "content": "", "summary": "", "confidence": "unknown",
                "sources": [], "extracted_content": [],
                "previous_content": previous[:4000] if a.action == "revise" else "",
                "revision_instructions": a.instructions,
            }
    session.completed_sections = kept
    session.run_config = dict(parent.run_config or {})
    session.run_config.update({
        "parent_session_id": parent.session_id,
        "revision_mode": rp.mode,
        "revision": rp.to_dict(),
        "feedback": (feedback or "")[:MAX_FEEDBACK_CHARS],
        "revised_at": datetime.now().isoformat(),
    })
    return session


def seed_locker(parent_evidence_path: Optional[Path], session_id: str,
                output_dir: Path) -> EvidenceLocker:
    """The parent's evidence under the new session id (kept chapters cite
    it; revised chapters may cite it again without refetching)."""
    ev_dir = Path(output_dir) / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    locker = None
    if parent_evidence_path and Path(parent_evidence_path).is_file():
        try:
            locker = EvidenceLocker.load_from_json(Path(parent_evidence_path))
        except Exception as e:  # noqa: BLE001 - a broken export is reported, not fatal
            ResearchWarnings.get_instance().add(
                "MEDIUM", "Revision", f"元の調査の根拠 JSON を読めませんでした（{e}）")
            locker = None
    if locker is None:
        locker = EvidenceLocker(research_id=session_id, output_dir=ev_dir)
    locker.research_id = session_id
    locker.output_dir = ev_dir
    return locker


def evidence_path_for(session_path: Path, session: Optional[ResearchSession] = None
                      ) -> Optional[Path]:
    """Where a saved session's evidence JSON is: the checkpoint's record,
    else <dir>/evidence/evidence_<id>.json."""
    session_path = Path(session_path)
    if session is None:
        try:
            session = ResearchSession.load(session_path)
        except Exception:
            return None
    saved = getattr(session, "saved_artifacts", {}) or {}
    cand = saved.get("evidence_json")
    if cand and Path(cand).is_file():
        return Path(cand)
    cand = session_path.parent / "evidence" / f"evidence_{session.session_id}.json"
    return cand if cand.is_file() else None


# ----------------------------------------------------------------------
# earlier research as reference material
# ----------------------------------------------------------------------

def reference_documents(session_paths: Sequence[str], *, min_chars: int = REFERENCE_MIN_CHARS,
                        max_docs: int = MAX_REFERENCE_DOCS
                        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evidence pages of earlier research as local documents (source
    "reference"), de-duplicated by URL, plus a per-session note."""
    docs: List[Dict[str, Any]] = []
    info: List[Dict[str, Any]] = []
    seen = set()
    for raw in session_paths or []:
        path = Path(raw)
        entry = {"session_path": str(path), "session_id": None, "query": None,
                 "evidence": 0, "used": 0, "error": None}
        try:
            session = ResearchSession.load(path)
            entry["session_id"] = session.session_id
            entry["query"] = session.query
            ev_path = evidence_path_for(path, session)
            if ev_path is None:
                entry["error"] = "根拠 JSON が見つかりません"
                info.append(entry)
                continue
            locker = EvidenceLocker.load_from_json(ev_path)
            for ev in locker.get_all_evidence():
                entry["evidence"] += 1
                url = (ev.url or "").strip()
                text = (ev.extracted_text or "").strip()
                if not url or url.startswith("local://") or len(text) < min_chars:
                    continue
                key = url.rstrip("/").lower()
                if key in seen or len(docs) >= max_docs:
                    continue
                seen.add(key)
                docs.append({
                    "path": url, "title": ev.title or url, "content": text,
                    "metadata": {"source": "reference", "source_url": url,
                                 "reference_session_id": session.session_id,
                                 "reference_query": session.query,
                                 "published_date": getattr(ev, "published_date", None),
                                 "fetched_at": getattr(ev, "accessed_at", None)
                                 or getattr(ev, "created_at", None)},
                })
                entry["used"] += 1
        except Exception as e:  # noqa: BLE001 - one unreadable reference is reported
            entry["error"] = f"{type(e).__name__}: {e}"[:200]
        info.append(entry)
    return docs, info


def reference_summaries(session_paths: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in session_paths or []:
        try:
            s = ResearchSession.load(Path(raw))
        except Exception:
            continue
        titles = [f"{it.section} {it.title}" for it in flat_sections(s)][:8]
        digest = " ".join(((s.section_contents.get("_executive_summary") or {})
                           .get("executive_summary") or "").split())[:200]
        when = (s.completed_at or s.started_at or "")[:10]
        out.append(f"「{s.query}」({when}) 章: {'、'.join(titles)}" + (f" 要旨: {digest}" if digest else ""))
    return out


# ----------------------------------------------------------------------
# what changed (for the result)
# ----------------------------------------------------------------------

def revision_summary(parent: ResearchSession, new: ResearchSession, rp: RevisionPlan,
                     locker: Optional[EvidenceLocker], *, parent_job_id: Optional[str] = None,
                     feedback: str = "", references: Optional[List[Dict[str, Any]]] = None,
                     parent_evidence_ids: Optional[set] = None) -> Dict[str, Any]:
    by_action = {a.section: a for a in rp.actions}
    rows = []
    for item in flat_sections(new):
        sid = str(item.section)
        a = by_action.get(sid)
        action = a.action if a is not None else "keep"
        before = (parent.section_contents.get(sid) or {}).get("content", "") or ""
        after = (new.section_contents.get(sid) or {}).get("content", "") or ""
        row = {"section": sid, "title": item.title, "action": action,
               "chars_before": len(before), "chars_after": len(after),
               "instructions": a.instructions if a is not None else ""}
        if locker is not None:
            try:
                evs = locker.get_section_evidence(sid)
                row["evidence_after"] = len(evs)
                if parent_evidence_ids is not None:
                    row["new_evidence"] = sum(1 for e in evs if e.id not in parent_evidence_ids)
                row["url_evidence"] = sum(1 for e in evs if getattr(e, "access_method", "") == "url")
                row["reference_evidence"] = sum(
                    1 for e in evs if getattr(e, "access_method", "") == "reference")
            except Exception:
                pass
        rows.append(row)
    return {
        "mode": rp.mode,
        "mode_label": MODE_LABELS.get(rp.mode, rp.mode),
        "parent_session_id": parent.session_id,
        "parent_job_id": parent_job_id,
        "parent_query": parent.query,
        "feedback": (feedback or "")[:MAX_FEEDBACK_CHARS],
        "plan_source": rp.source,
        "requirements_addendum": rp.requirements_addendum,
        "acknowledged": list(rp.acknowledged),
        "notes": rp.notes,
        "sections": rows,
        "kept": [r["section"] for r in rows if r["action"] == "keep"],
        "revised": [r["section"] for r in rows if r["action"] == "revise"],
        "added": [r["section"] for r in rows if r["action"] == "add"],
        "references": list(references or []),
    }
