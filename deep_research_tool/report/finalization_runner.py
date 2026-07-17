"""
FinalizationRunner - the ONE production finalization path.

Every report version (V1 / V2 / V3) and ``run_manual_research`` convert
their generation output to the common chapter form ``{section_id:
markdown_text}`` and run it through this runner:

    generation result -> common chapters -> FinalizationController
    (verify -> decide -> act -> re-verify -> freeze) -> deterministic
    citation rendering -> back to each renderer

Guarantees implemented here (audit items 1-3, 5, 7, 12, 13):

- Evidence is read LIVE from the Evidence Locker on every verification
  pass — no snapshot of URLs or evidence taken at loop start.
- Citations use canonical Evidence ids. During editing the body carries
  per-section ``[SOURCE N]`` tags whose registry maps to Evidence.id;
  display numbers ``[n]`` are substituted only at the final render, and
  the locker is reordered so every renderer's references list matches.
- Targeted final research updates, in the same round: the locker, the
  section-evidence relations, the citation registry (append-only), the
  generator's chapter citations (via callback), and the length-planner
  units. Reposts of already-collected content (even under a different
  URL) are detected by content similarity and are NOT counted as new
  independent sources.
- Re-edit prompts contain the FULL section evidence within a character
  budget (no fixed first-15 cap), newly researched evidence first, each
  block labeled with its stable [SOURCE N] number.
- URLs already fetched are never fetched again (LoopBudget.seen_urls is
  seeded from the live locker and consulted before every fetch).
"""

import html as _html
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .citations import CitationManager
from .finalization import (
    FinalizationController,
    LoopBudget,
)
from .length_planner import LengthPlanner, _jaccard_bigram
from ..verification.claim_verifier import ClaimVerifier

# Character budget for evidence blocks inside one edit prompt
EDIT_EVIDENCE_CHAR_BUDGET = 24000
EDIT_EVIDENCE_ITEM_CHARS = 1600

# Recency markers: when the query/requirements demand up-to-date or
# primary information, the primary/freshness gate becomes REQUIRED
_FRESHNESS_MARKERS = re.compile(
    r"最新|直近|現時点|今年|本年|足元|現在の|一次情報|一次資料|"
    r"20\d{2}年(?:時点|現在)|latest|most recent|up[- ]to[- ]date|current",
    re.IGNORECASE)


class FinalizationRunner:
    """Builds and runs the finalization loop for any report version."""

    def __init__(
        self,
        *,
        evidence_locker,
        session_contents: Dict[str, Any],
        research_plan=None,
        query: str = "",
        requirements: str = "",
        language: str = "ja",
        llm_client=None,
        eval_llm=None,
        writing_llm=None,
        search_client=None,
        report_config=None,
        research_config=None,
        output_dir=None,
        session_id: str = "",
        progress_callback: Optional[Callable] = None,
        chapter_citation_callback: Optional[Callable] = None,
    ):
        """
        Args:
            evidence_locker: live EvidenceLocker (queried on every pass)
            session_contents: {sid: {"content", "extracted_content",
                "sources", ...}} — the research-side section store; new
                evidence is written back into it
            chapter_citation_callback: callable(section_id, evidence)
                invoked for every newly researched evidence item so the
                generator-side chapter (ChapterContent.citations) stays
                in sync with the registry
        """
        self.locker = evidence_locker
        self.session_contents = session_contents or {}
        self.research_plan = research_plan
        self.query = query
        self.requirements = requirements or ""
        self.language = language
        self.llm_client = llm_client
        self.eval_llm = eval_llm or llm_client
        self.writing_llm = writing_llm or llm_client
        self.search_client = search_client
        self.rp = report_config
        self.rc = research_config
        self.output_dir = Path(output_dir or "./output")
        self.session_id = session_id
        self.progress_callback = progress_callback
        self.chapter_citation_callback = chapter_citation_callback

        self.citation_mgr = CitationManager(
            evidence_ids_exist=self._evidence_id_exists)
        self.planner = self._build_planner()
        self.budget = self._build_budget()
        self.claim_verifier = ClaimVerifier(
            llm_client=self.eval_llm, language=self.language)
        self.section_evidence: Dict[str, List[Dict]] = {}
        self._url_to_id: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def _rp(self, name, default=None):
        return getattr(self.rp, name, default) if self.rp else default

    def _rc(self, name, default=None):
        return getattr(self.rc, name, default) if self.rc else default

    def _build_planner(self) -> LengthPlanner:
        length_mode = (self._rp("length_mode", "adaptive") or
                       "adaptive").lower()
        # target_characters remains a SOFT preference in both modes;
        # only user-set hard_min/hard_max are ever absolute, so a legacy
        # target alone can never trigger research (audit item 11)
        preferred = self._rp("preferred_body_chars") or \
            self._rp("target_characters")
        return LengthPlanner(
            language=self.language,
            length_mode=length_mode,
            preferred_body_chars=preferred,
            hard_min_body_chars=self._rp("hard_min_body_chars"),
            hard_max_body_chars=self._rp("hard_max_body_chars"),
            length_tolerance=self._rp("length_tolerance", 0.20),
        )

    def _build_budget(self) -> LoopBudget:
        budget = LoopBudget(
            max_final_research_rounds=self._rc("max_final_research_rounds", 2),
            max_final_revision_rounds=self._rc("max_final_revision_rounds", 2),
            max_no_improvement_rounds=self._rc("max_no_improvement_rounds", 1),
            min_score_improvement=self._rc("min_score_improvement", 0.03),
            min_new_independent_sources=self._rc(
                "min_new_independent_sources", 1),
            min_claim_support_score=self._rc("min_claim_support_score", 0.85),
            required_critical_coverage=self._rc(
                "required_critical_coverage", 1.0),
        )
        # every URL already in the locker is "seen": the final loop never
        # re-fetches a URL the research already collected
        for ev in self.locker.get_all_evidence():
            if getattr(ev, "url", ""):
                budget.seen_urls.add(ev.url)
        return budget

    def _evidence_id_exists(self, eid: str) -> bool:
        """LIVE existence check: canonical id first, URL fallback."""
        if not eid:
            return False
        if self.locker.get_evidence(eid) is not None:
            return True
        # fallback for entries whose URL never made it into the locker
        get_by_url = getattr(self.locker, "get_by_url", None)
        if callable(get_by_url):
            return get_by_url(eid) is not None
        return any(getattr(e, "url", "") == eid
                   for e in self.locker.get_all_evidence())

    def _eid_for_url(self, url: str) -> str:
        """Resolve a URL to the canonical Evidence.id (live)."""
        if url in self._url_to_id:
            return self._url_to_id[url]
        get_by_url = getattr(self.locker, "get_by_url", None)
        ev = get_by_url(url) if callable(get_by_url) else None
        if ev is None:
            for e in self.locker.get_all_evidence():
                if getattr(e, "url", "") == url:
                    ev = e
                    break
        eid = getattr(ev, "id", "") if ev is not None else ""
        eid = eid or url    # URL fallback keeps the registry aligned
        self._url_to_id[url] = eid
        return eid

    # ------------------------------------------------------------------
    # registry / planner setup
    # ------------------------------------------------------------------

    def _register_sections(self, chapters: Dict[str, str]) -> None:
        for sid in chapters:
            sdata = self.session_contents.get(sid) or {}
            extracted = sdata.get("extracted_content") or []
            urls = [ec.get("url", "") for ec in extracted if ec.get("url")]
            if not urls:
                urls = [u for u in (sdata.get("sources") or []) if u]
            ids = [self._eid_for_url(u) for u in urls]
            self.citation_mgr.register_section(sid, ids)
            self.section_evidence[sid] = list(extracted)

    def _plan_lengths(self) -> None:
        plan_sections = []
        if self.research_plan is not None:
            try:
                for item in (self.research_plan.table_of_contents
                             .get_flat_sections()):
                    plan_sections.append(
                        {"section": item.section, "title": item.title})
            except Exception:
                plan_sections = []
        if not plan_sections:
            plan_sections = [{"section": sid, "title": ""}
                             for sid in self.section_evidence]
        self.planner.initial_allocation(plan_sections)
        self._recalc_units()

    def _recalc_units(self) -> None:
        units = {sid: self.planner.extract_units(sid, evs)
                 for sid, evs in self.section_evidence.items()}
        if units:
            self.planner.recalc_after_research(units)

    def _critical_questions(self) -> List[str]:
        """Plan questions + explicit user requirements (audit item 8)."""
        questions: List[str] = []
        if self.research_plan is not None:
            try:
                for item in self.research_plan.table_of_contents.items:
                    q = f"{item.title} {getattr(item, 'description', '')}" \
                        .strip()
                    if q:
                        questions.append(q)
            except Exception:
                pass
        for line in re.split(r"[\n。]", self.requirements):
            line = line.strip()
            if len(line) >= 8:
                questions.append(line)
        return questions[:15]

    def _freshness_required(self) -> bool:
        return bool(_FRESHNESS_MARKERS.search(
            f"{self.query} {self.requirements}"))

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------

    def run(self, chapters: Dict[str, str]) -> Dict[str, Any]:
        """Run the loop on common-form chapters; returns the outcome dict.

        outcome keys: chapters (RENDERED, display-numbered), raw_chapters
        (frozen [SOURCE N] form), verdict, decision, history,
        limitations, over_hard_max, ordered_evidence_ids, html_path.
        """
        if self.progress_callback:
            self.progress_callback("Verifying final report body...", 96)

        self._register_sections(chapters)
        self._plan_lengths()
        critical_questions = self._critical_questions()
        freshness_required = self._freshness_required()

        current = dict(chapters)   # the shared, mutating body

        def verify_fn(body):
            # LIVE evidence on every pass — never a loop-start snapshot
            return self.claim_verifier.verify_report(
                chapters=body,
                evidence_list=self.locker.get_all_evidence(),
                citation_manager=self.citation_mgr,
                critical_questions=critical_questions,
                length_plans=self.planner.plans,
                preferred_body_chars=self.planner.preferred_body_chars,
                exclude_references=self._rp(
                    "exclude_references_from_count", True),
                primary_freshness_required=freshness_required,
            )

        def research_fn(issues, queries):
            return self._research_round(issues, queries)

        controller = FinalizationController(
            verify_fn=verify_fn,
            research_fn=research_fn if self.search_client else None,
            rewrite_fn=lambda sid, issues: self._edit_section(
                sid, issues, current, mode="rewrite"),
            compress_fn=lambda sid, issues: self._edit_section(
                sid, issues, current, mode="compress"),
            hedge_fn=lambda sid, issues: self._edit_section(
                sid, issues, current, mode="hedge"),
            validate_citations_fn=self.citation_mgr.validate,
            budget=self.budget,
            hard_max_body_chars=self._rp("hard_max_body_chars"),
            hard_min_body_chars=self._rp("hard_min_body_chars"),
            language=self.language,
        )

        outcome = controller.run(current)

        # FREEZE, then deterministic display numbering ([SOURCE N] -> [n])
        frozen = outcome["chapters"]
        rendered, ordered_ids = self.citation_mgr.render_numbering(frozen)
        # references follow first-use order in every renderer
        try:
            self.locker.reorder(ordered_ids)
        except Exception as e:
            print(f"[Finalize] locker reorder failed: {e}")

        outcome["raw_chapters"] = frozen
        outcome["chapters"] = rendered
        outcome["ordered_evidence_ids"] = ordered_ids

        verdict = outcome["verdict"]
        m = verdict.metrics
        print(f"[Finalize] decision={outcome['decision']} "
              f"support={m.claim_support_score:.2f} "
              f"claims={m.claims_total} "
              f"unsupported={m.unsupported_count} "
              f"contradicted={m.contradicted_count} "
              f"uncertain={m.uncertain_count} "
              f"coverage={m.critical_question_coverage:.2f} "
              f"freshness={m.primary_freshness} "
              f"body_chars={m.actual_body_chars}")

        html_path = (self.output_dir /
                     f"verification_{self.session_id or 'report'}.html")
        try:
            self._write_html(verdict, outcome, html_path)
        except Exception as e:
            print(f"[Finalize] verification HTML failed: {e}")
            html_path = None
        outcome["html_path"] = html_path
        return outcome

    # ------------------------------------------------------------------
    # research round (live locker updates, dedup, no refetch)
    # ------------------------------------------------------------------

    def _is_duplicate_content(self, text: str) -> bool:
        """Reposts (URL-different copies) never count as new sources."""
        head = (text or "")[:1500]
        if not head.strip():
            return True
        for ev in self.locker.get_all_evidence():
            existing = (getattr(ev, "extracted_text", "") or
                        getattr(ev, "content_excerpt", "") or "")[:1500]
            if existing and _jaccard_bigram(head, existing) > 0.85:
                return True
        return False

    def _research_round(self, issues, queries) -> Dict[str, Any]:
        from ..evidence.locker import EvidenceType

        new_sources = 0
        changed_sections: List[str] = []
        per_issue_sections: Dict[str, List] = {}
        for issue in issues:
            if issue.section_id:
                per_issue_sections.setdefault(
                    issue.section_id, []).append(issue)

        for q in queries[:6]:
            try:
                results = self.search_client.search(q)
            except Exception as e:
                print(f"[Finalize] research query failed '{q[:40]}': {e}")
                continue
            for r in results[:2]:
                url = getattr(r, "url", "")
                # seen_urls is wired to the actual fetch: a URL is fetched
                # at most once across the whole run (locker-seeded)
                if not self.budget.is_novel_url(url):
                    continue
                try:
                    page = self.search_client.get_page_content(url)
                except Exception:
                    continue
                text = getattr(page, "text_content", "") or ""
                if len(text) < 200:
                    continue
                # content-level dedup: same substance under another URL
                # is NOT a new independent source
                if self._is_duplicate_content(text):
                    print(f"[Finalize] duplicate content skipped: {url[:60]}")
                    continue
                target_sections = list(per_issue_sections.keys()) or \
                    list(self.section_evidence.keys())[:1]
                evidence = self.locker.add_evidence(
                    url=url, title=getattr(page, "title", "") or url,
                    content_excerpt=text[:500], extracted_text=text,
                    evidence_type=EvidenceType.WEB_PAGE,
                    search_query=q,
                    section_reference=target_sections[0]
                    if target_sections else "",
                )
                self._url_to_id[url] = evidence.id
                new_sources += 1
                for sid in target_sections:
                    entry = {"title": getattr(page, "title", "") or url,
                             "url": url, "content": text[:2000],
                             "raw_content": text, "key_points": [],
                             "relevance_score": 0.5, "is_new": True}
                    # section-evidence relations
                    self.section_evidence.setdefault(sid, []).append(entry)
                    sdata = self.session_contents.get(sid)
                    if sdata is not None:
                        sdata.setdefault("extracted_content", []).append(entry)
                        sdata.setdefault("sources", []).append(url)
                    # citation registry: APPEND-ONLY, numbers never shift
                    self.citation_mgr.append_evidence(sid, evidence.id)
                    # generator-side chapter citations stay in sync
                    if self.chapter_citation_callback:
                        try:
                            self.chapter_citation_callback(sid, evidence)
                        except Exception:
                            pass
                    if sid not in changed_sections:
                        changed_sections.append(sid)

        # length-planner units follow the evidence (audit item 2)
        if changed_sections:
            self._recalc_units()
        return {"new_sources": new_sources,
                "changed_sections": changed_sections}

    # ------------------------------------------------------------------
    # section edits (full evidence, new-first, stable SOURCE numbers)
    # ------------------------------------------------------------------

    def _evidence_blocks(self, sid: str) -> str:
        """All section evidence within a char budget, NEW evidence first,
        each block labeled with its stable [SOURCE N] number."""
        extracted = self.section_evidence.get(sid) or \
            (self.session_contents.get(sid) or {}).get(
                "extracted_content") or []
        # stable numbers come from the citation registry
        id_to_number = {eid: n for n, eid in
                        self.citation_mgr.mapping(sid).items()}
        entries = []
        for i, ec in enumerate(extracted):
            eid = self._eid_for_url(ec.get("url", ""))
            number = id_to_number.get(eid, i + 1)
            entries.append((bool(ec.get("is_new")), number, ec))
        # newly researched evidence first (audit item 7)
        entries.sort(key=lambda t: (not t[0], t[1]))

        blocks = []
        budget = EDIT_EVIDENCE_CHAR_BUDGET
        for is_new, number, ec in entries:
            body = (ec.get("raw_content") or ec.get("content") or "")
            body = body[:EDIT_EVIDENCE_ITEM_CHARS]
            tag = ("【新規】" if self.language == "ja" else "[NEW] ") \
                if is_new else ""
            block = f"[SOURCE {number}] {tag}{ec.get('title', '')}\n{body}"
            if budget - len(block) < 0 and blocks:
                break
            budget -= len(block)
            blocks.append(block)
        return "\n---\n".join(blocks)

    def _edit_section(self, sid: str, issues, current: Dict[str, str],
                      mode: str = "rewrite") -> Optional[str]:
        """LLM edit of ONE section from its EXISTING evidence.

        mode: rewrite  (deepen using unused evidence, no new facts)
              compress (remove redundancy, keep claims & citations)
              hedge    (soften unresolved assertions, state limitations)
        The controller machine-validates citations before accepting.
        """
        text = current.get(sid, "")
        if not text:
            text = (self.session_contents.get(sid) or {}).get("content", "")
        if not text:
            return None

        evidence_text = self._evidence_blocks(sid)
        issue_text = "\n".join(
            f"- ({i.type}) {i.claim or i.reason}" for i in issues[:8])

        lang_ja = self.language == "ja"
        if mode == "rewrite":
            instruction = (
                "既存のエビデンスだけを使い、以下の不足点を解消するように本文を増補・再構成してください。"
                "【新規】と記されたエビデンスを優先的に反映する。"
                "外部の新しい情報を創作しない。[SOURCE N] は上記の番号のみ使用し、既存の引用は維持する。"
                if lang_ja else
                "Using ONLY the evidence above, expand/restructure the text to fix the issues. "
                "Prefer the evidence marked [NEW]. Do not invent facts. "
                "Use only the [SOURCE N] numbers listed; keep existing citations.")
        elif mode == "compress":
            instruction = (
                "重要な主張と引用 [SOURCE N] をすべて維持したまま、重複・冗長・低重要度の記述を圧縮してください。"
                "引用だけ残して主張を消さない。新しい情報を加えない。"
                if lang_ja else
                "Compress redundancy while KEEPING every important claim and its [SOURCE N] citations. "
                "Never leave a citation whose claim was removed. Add nothing new.")
        else:  # hedge
            instruction = (
                "以下の未解決の主張について、断定を避けた限定表現（「〜の可能性がある」「公開情報では確認できなかった」等）に"
                "修正してください。それ以外の本文・引用は変更しない。"
                if lang_ja else
                "Soften the unresolved assertions below into hedged statements. "
                "Change nothing else; keep all citations.")

        prompt = (f"{'あなたはレポート編集者です。' if lang_ja else 'You are a report editor.'}\n\n"
                  f"{'【対象セクション本文】' if lang_ja else '[SECTION TEXT]'}\n{text}\n\n"
                  f"{'【エビデンス】' if lang_ja else '[EVIDENCE]'}\n{evidence_text}\n\n"
                  f"{'【指摘事項】' if lang_ja else '[ISSUES]'}\n{issue_text or '-'}\n\n"
                  f"{'【指示】' if lang_ja else '[INSTRUCTIONS]'}\n{instruction}\n\n"
                  f"{'修正後の本文のみを出力（見出しを含め、JSON・前置き不要）:' if lang_ja else 'Output only the revised section text:'}")

        try:
            response = self.writing_llm.generate(prompt)
            edited = (response.content or "").strip()
            return edited or None
        except Exception as e:
            print(f"[Finalize] section edit failed ({mode}, {sid}): {e}")
            return None

    # ------------------------------------------------------------------
    # verification report (deterministic HTML)
    # ------------------------------------------------------------------

    def _write_html(self, verdict, outcome, path) -> None:
        m = verdict.metrics
        rows = "".join(
            f"<tr><td>{_html.escape(i.section_id)}</td>"
            f"<td>{_html.escape(i.claim_id)}</td>"
            f"<td>{_html.escape(i.type)}</td>"
            f"<td>{_html.escape(i.severity)}</td>"
            f"<td>{_html.escape((i.claim or '')[:120])}</td>"
            f"<td>{_html.escape((i.reason or '')[:160])}</td></tr>"
            for i in verdict.issues)
        history = "".join(
            f"<li>round {h['round']}: {h['decision']} "
            f"(score={h['score']:.2f}, issues={h['issues']})</li>"
            for h in outcome.get("history", []))
        errors = "".join(
            f"<li>{_html.escape(e)}</li>" for e in m.extraction_errors)
        doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Final Verification</title>
<style>body{{font-family:sans-serif;margin:2em}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px}}</style></head><body>
<h1>最終検証レポート</h1>
<p>判定: <b>{_html.escape(outcome.get('decision', ''))}</b></p>
<ul>
<li>claim support score: {m.claim_support_score:.3f}
 （claims: {m.claims_total}, chunks: {m.chunks_total},
 chunk failures: {m.chunks_failed}）</li>
<li>unsupported: {m.unsupported_count}（うちcritical: {m.unsupported_critical_claims}）</li>
<li>contradicted: {m.contradicted_count} / uncertain: {m.uncertain_count}</li>
<li>critical question coverage: {m.critical_question_coverage:.2f}</li>
<li>citations valid: {m.citations_valid}</li>
<li>primary/freshness: {_html.escape(m.primary_freshness)}</li>
<li>verification failed: {m.verification_failed}</li>
<li>body chars: {m.actual_body_chars}
 (recommended {m.recommended_min_chars}–{m.recommended_max_chars})</li>
</ul>
<h2>判定履歴</h2><ol>{history}</ol>
<h2>抽出エラー ({len(m.extraction_errors)})</h2><ul>{errors}</ul>
<h2>Issues ({len(verdict.issues)})</h2>
<table><tr><th>section</th><th>claim</th><th>type</th><th>severity</th>
<th>claim text</th><th>reason</th></tr>{rows}</table>
</body></html>"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(doc, encoding="utf-8")
