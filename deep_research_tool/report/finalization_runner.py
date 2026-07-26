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
        verification_progress=None,
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

        # --- verification profile: the ONE backend resolution path ---
        from ..verification.profiles import (
            VerificationSettings, settings_from_research_config)
        from ..verification.runtime import (
            VerificationCache, VerificationProgress)
        if self.rc is not None:
            self.verification_settings = settings_from_research_config(self.rc)
        else:
            self.verification_settings = VerificationSettings()
        self.verification_cache = VerificationCache(
            enabled=self.verification_settings.cache_enabled)
        self.verification_progress = (verification_progress
                                      or VerificationProgress())

        self.citation_mgr = CitationManager(
            evidence_ids_exist=self._evidence_id_exists)
        self.planner = self._build_planner()
        self.budget = self._build_budget()
        self.claim_verifier = ClaimVerifier(
            llm_client=self.eval_llm, language=self.language,
            settings=self.verification_settings,
            cache=self.verification_cache,
            progress=self.verification_progress)
        self.section_evidence: Dict[str, List[Dict]] = {}
        self._url_to_id: Dict[str, str] = {}
        # Sections that are VERIFIED but never LLM-edited (rendered
        # verbatim from their frozen source, e.g. figure semantics)
        self.render_only_sections: set = set()
        # Sections verified WITHOUT the per-claim citation requirement
        # (executive summary — a digest of already-cited body claims)
        self.citation_exempt_sections: set = set()

        # --- adaptive coverage: requirement-granular gap control ------
        from ..adaptive import CoverageLedger, StopController
        from ..utils.audit_log import AuditLog
        self.adaptive_enabled = bool(self._rc("adaptive_coverage", True))
        self.ledger = CoverageLedger(
            max_search_attempts=self._rc(
                "requirement_max_search_attempts", 2) or 2) \
            if self.adaptive_enabled else None
        self.stop_controller = StopController(
            max_rounds=self.verification_settings.max_final_research_rounds,
            max_stall_rounds=self._rc("max_stall_rounds", 2) or 2) \
            if self.adaptive_enabled else None
        if self.adaptive_enabled:
            # ONE stopping authority: with the ledger active, the
            # StopController decides stalls. The legacy LoopBudget
            # no-improvement counter is aligned to the SAME threshold so
            # it can never stop a round earlier than the configured
            # max_stall_rounds (previously max_no_improvement_rounds=1
            # ended the loop after a single stall regardless).
            self.budget.max_no_improvement_rounds = max(
                self.budget.max_no_improvement_rounds,
                self.stop_controller.max_stall_rounds)
        # default ON in production (a research_config exists); OFF for
        # bare test/manual constructions so no files appear unrequested
        audit_enabled = bool(self._rc("audit_log_enabled",
                                      self.rc is not None))
        self.audit = AuditLog(
            path=(self.output_dir /
                  f"audit_{self.session_id or 'report'}.jsonl"),
            enabled=audit_enabled,
            session_id=self.session_id)
        # round-in-progress bookkeeping (finalized on the next verify)
        self._pending_round = None
        self._last_claim_counts = None
        self._req_seq = 0

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
        vs = self.verification_settings
        budget = LoopBudget(
            max_final_research_rounds=vs.max_final_research_rounds,
            max_final_revision_rounds=vs.max_final_revision_rounds,
            max_no_improvement_rounds=vs.max_no_improvement_rounds,
            min_score_improvement=self._rc("min_score_improvement", 0.03),
            min_new_independent_sources=self._rc(
                "min_new_independent_sources", 1),
            min_claim_support_score=vs.min_claim_support_score,
            required_critical_coverage=vs.required_critical_coverage,
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
        # shift budget from low- to high-importance sections
        self.planner.rebalance()

    def _recalc_units(self) -> None:
        units = {sid: self.planner.extract_units(sid, evs)
                 for sid, evs in self.section_evidence.items()}
        if units:
            self.planner.recalc_after_research(units)

    def _requirement_sources(self) -> List[Dict[str, str]]:
        """ALL requirement sources, none silently dropped.

        - explicit USER requirements come FIRST (highest priority);
        - EVERY level of the research plan is included
          (get_flat_sections walks subsections too);
        - duplicates are merged after normalization;
        - there is NO count cap — requirements that cannot fit one LLM
          call are batched downstream, never discarded.
        """
        sources: List[Dict[str, str]] = []
        seen: set = set()

        def _add(text: str, origin: str, section: str = "") -> None:
            text = re.sub(r"\s+", " ", (text or "").strip())
            key = re.sub(r"\s+", "", text.lower())
            if len(text) < 8 or key in seen:
                return
            seen.add(key)
            sources.append({"text": text, "origin": origin,
                            "section": section})

        # explicit user requirements FIRST — they are never dropped
        for line in re.split(r"[\n。]", self.requirements):
            _add(line, "user")

        if self.research_plan is not None:
            try:
                flat = list(self.research_plan.table_of_contents
                            .get_flat_sections())
            except Exception:
                flat = []
            if not flat:
                try:
                    flat = list(self.research_plan.table_of_contents.items)
                except Exception:
                    flat = []
            for item in flat:
                q = (f"{getattr(item, 'title', '')} "
                     f"{getattr(item, 'description', '')}").strip()
                _add(q, "plan", str(getattr(item, "section", "") or ""))
        return sources

    def _critical_questions(self) -> List[str]:
        """Plan questions (all hierarchy levels) + explicit user
        requirements — ALL of them, user requirements first. The old
        silent ``[:15]`` cap is gone; coverage judging batches the
        questions per LLM call instead of discarding them."""
        return [s["text"] for s in self._requirement_sources()]

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

        if self.ledger is not None:
            self._build_requirements(critical_questions, chapters)
        self.audit.event(
            "finalization_started",
            profile=self.verification_settings.profile,
            adaptive=self.ledger is not None,
            sections=sorted(chapters), questions=len(critical_questions))

        current = dict(chapters)   # the shared, mutating body

        from ..verification.runtime import stable_hash
        last_verify = {"fingerprint": None, "verdict": None}

        def _fingerprint(body):
            # report body + sections + claims-bearing text + evidence +
            # citation relations: when NOTHING substantive changed, the
            # previous verdict stands without any LLM call
            evidence_sig = stable_hash(
                "ev", *sorted(
                    f"{getattr(e, 'id', '')}:"
                    f"{stable_hash(getattr(e, 'extracted_text', '') or getattr(e, 'content_excerpt', '') or '')}"
                    for e in self.locker.get_all_evidence()))
            registry_sig = stable_hash(
                "reg",
                *[f"{sid}:" + ",".join(
                    str(eid) for eid in self.citation_mgr.evidence_ids(sid))
                  for sid in sorted(self.citation_mgr.sections(), key=str)])
            body_sig = stable_hash(
                "body", *[f"{sid}\x1e{text}"
                          for sid, text in sorted(body.items())])
            return stable_hash(body_sig, evidence_sig, registry_sig,
                               *critical_questions)

        def verify_fn(body):
            # differential verification: an IDENTICAL body/evidence/
            # citation state is never re-verified with the LLM
            fp = _fingerprint(body)
            if (self.verification_settings.cache_enabled
                    and last_verify["fingerprint"] == fp
                    and last_verify["verdict"] is not None):
                self.verification_cache.hits += 1
                self.verification_progress.sync_cache(
                    self.verification_cache)
                print("[Finalize] body/evidence/citations unchanged — "
                      "skipping full re-verification")
                self._after_verify(last_verify["verdict"])
                return last_verify["verdict"]

            # LIVE evidence on every pass — never a loop-start snapshot
            verdict = self.claim_verifier.verify_report(
                chapters=body,
                evidence_list=self.locker.get_all_evidence(),
                citation_manager=self.citation_mgr,
                critical_questions=critical_questions,
                length_plans=self.planner.plans,
                preferred_body_chars=self.planner.preferred_body_chars,
                exclude_references=self._rp(
                    "exclude_references_from_count", True),
                primary_freshness_required=freshness_required,
                citation_exempt_sections=self.citation_exempt_sections,
            )
            last_verify["fingerprint"] = fp
            last_verify["verdict"] = verdict
            self._after_verify(verdict)
            return verdict

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
            # fixed mode: the explicit quota ± tolerance is evaluated in
            # production (planner.fixed_quota() is None in adaptive mode)
            fixed_target_chars=self.planner.fixed_quota(),
            length_tolerance=self._rp("length_tolerance", 0.20),
            language=self.language,
            progress=self.verification_progress,
        )

        import time as _time
        from ..verification.runtime import (
            VerificationCancelled, VerificationTimeout)
        vs = self.verification_settings
        self.verification_progress.start(
            vs.profile,
            max_rounds=(vs.max_final_research_rounds
                        + vs.max_final_revision_rounds + 2),
            timeout_seconds=vs.timeout_seconds)
        started = _time.time()

        try:
            outcome = controller.run(current)
            self.verification_progress.set_phase("done")
        except VerificationCancelled:
            # SAFE cancel: no new LLM request after the boundary; the
            # partial body, last verdict and metrics are preserved
            outcome = self._interrupted_outcome(
                controller, current, last_verify["verdict"], "cancelled")
            self.verification_progress.set_phase("cancelled")
        except VerificationTimeout:
            outcome = self._interrupted_outcome(
                controller, current, last_verify["verdict"], "timeout")
            self.verification_progress.set_phase("error", "timeout")

        outcome["verification_summary"] = self._build_summary(
            outcome, started)

        # settle the coverage ledger: nothing stays silently "open" —
        # remaining gaps are explicitly closed as budget_exhausted
        if self.ledger is not None:
            closed = self.ledger.close_budget_exhausted(
                "最終ループ終了時に未解決" if self.language == "ja"
                else "unresolved at end of finalization loop")
            if not self.ledger.termination_reason:
                self.ledger.termination_reason = (
                    "all_requirements_terminal" if not closed
                    else "loop_budget_exhausted")
            outcome["coverage_ledger"] = self.ledger.to_dict()
            outcome["termination_reason"] = self.ledger.termination_reason
            outcome["verification_summary"]["termination_reason"] = \
                self.ledger.termination_reason
            self.audit.event("finalization_done",
                             decision=outcome["decision"],
                             termination_reason=(
                                 self.ledger.termination_reason),
                             coverage=self.ledger.coverage(),
                             counts=self.ledger.counts(),
                             budget_exhausted=closed)
        else:
            self.audit.event("finalization_done",
                             decision=outcome["decision"])

        # FREEZE, then deterministic display numbering ([SOURCE N] -> [n])
        frozen = outcome["chapters"]
        self.audit.event("body_frozen", sections=sorted(frozen))
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
    # interruption / summary
    # ------------------------------------------------------------------

    @staticmethod
    def _interrupted_outcome(controller, chapters, verdict, decision):
        """Partial outcome after cancel/timeout — an unverified body is
        NEVER silently promoted: without a verdict the outcome carries a
        verification_failed one."""
        from .finalization import StructuredVerdict
        if verdict is None:
            verdict = StructuredVerdict()
            verdict.metrics.verification_failed = True
            verdict.metrics.claim_support_score = 0.0
        return {
            "chapters": chapters,
            "verdict": verdict,
            "decision": decision,
            "history": controller.history,
            "limitations": controller.limitations,
            "over_hard_max": False,
        }

    def _build_summary(self, outcome, started) -> Dict[str, Any]:
        """Completion summary for the GUI (mode, timing, counters,
        unresolved claims, skipped work). Metrics only — never api keys
        or prompt text."""
        import time as _time
        verdict = outcome["verdict"]
        m = verdict.metrics
        snap = self.verification_progress.snapshot()
        vs = self.verification_settings

        unresolved = [
            {"section_id": i.section_id, "type": i.type,
             "severity": i.severity,
             "claim": (i.claim or i.reason or "")[:160]}
            for i in verdict.issues
            if i.type in ("unsupported_claim", "contradicted_claim",
                          "uncertain_claim", "invalid_citation",
                          "unanswered_critical_question")][:20]

        skipped_work = []
        if vs.max_final_research_rounds == 0:
            skipped_work.append("追加調査（0回設定）")
        if vs.max_final_revision_rounds == 0:
            skipped_work.append("自動修正（0回設定）")
        if m.skipped_minor_claims:
            skipped_work.append(
                f"軽微クレームのサンプル検証で{m.skipped_minor_claims}件を省略"
                f"（検証率{int(vs.minor_claim_sample_rate * 100)}%）")

        supported = max(m.claims_total - m.skipped_minor_claims
                        - m.unsupported_count - m.contradicted_count
                        - m.uncertain_count, 0)
        counters = {"claims": snap["claims_total"],
                    "chunks": snap["chunks_total"],
                    "llm": snap["llm_calls"]}
        bottleneck = ("クレーム検証"
                      if snap["claims_total"] >= snap["chunks_total"]
                      else "クレーム抽出")

        return {
            "profile": vs.profile,
            "decision": outcome["decision"],
            "duration_seconds": round(_time.time() - started, 1),
            "rounds": len(outcome.get("history", [])),
            "claims_total": m.claims_total,
            "supported": supported,
            "partially_supported": m.uncertain_count,
            "unsupported": m.unsupported_count + m.contradicted_count,
            "skipped_minor": m.skipped_minor_claims,
            "claim_support_score": round(m.claim_support_score, 3),
            "critical_coverage": round(m.critical_question_coverage, 3),
            "llm_calls": snap["llm_calls"],
            "cache_hits": snap["cache_hits"],
            "cache_hit_rate": snap["cache_hit_rate"],
            "retries": snap["retries"],
            "bottleneck": bottleneck,
            "unresolved_claims": unresolved,
            "skipped_work": skipped_work,
        }

    # ------------------------------------------------------------------
    # research round (live locker updates, dedup, no refetch)
    # ------------------------------------------------------------------

    _DATE_RE = re.compile(
        r"((?:19|20)\d{2})[年/\-.](\d{1,2})[月/\-.]?(?:(\d{1,2})日?)?")

    def _classify_new_evidence(self, url: str, text: str):
        """(published_date, source_type, is_primary) for researched pages."""
        from ..evidence.locker import SourceType
        published = ""
        m = self._DATE_RE.search((text or "")[:3000])
        if m:
            month = int(m.group(2))
            day = int(m.group(3) or 1)
            if 1 <= month <= 12 and 1 <= day <= 31:
                published = f"{m.group(1)}-{month:02d}-{day:02d}"
        host = re.sub(r"^https?://", "", url or "").split("/")[0].lower()
        if re.search(r"\.(go\.jp|gov(\.[a-z]{2})?)$|\.gouv\.", host):
            return published, SourceType.OFFICIAL, True
        if re.search(r"\.(ac\.jp|edu(\.[a-z]{2})?)$", host):
            return published, SourceType.ACADEMIC, True
        return published, SourceType.UNKNOWN, False

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

    def _fetch_and_register(self, url: str, query: str,
                            target_sections: List[str],
                            changed_sections: List[str],
                            meta: Optional[Dict[str, str]] = None):
        """Fetch one novel URL and register it as evidence (shared by the
        legacy and adaptive research rounds). Returns the Evidence or
        None (already seen / too short / duplicate content / fetch
        error). seen_urls is wired to the actual fetch: a URL is fetched
        at most once across the whole run (locker-seeded).

        ``target_sections`` is the ONLY place the evidence is registered
        (citations + session contents). An EMPTY list keeps the evidence
        in the locker as UNASSIGNED — it is never sprayed across
        sections it was not searched for. ``meta`` (query/requirement/
        section/claim ids) is stored on the evidence so every
        EvidenceChunk built from it carries the full provenance chain.
        """
        from ..evidence.locker import EvidenceType

        if not self.budget.is_novel_url(url):
            return None
        try:
            page = self.search_client.get_page_content(url)
        except Exception:
            return None
        text = getattr(page, "text_content", "") or ""
        if len(text) < 200:
            return None
        # content-level dedup: same substance under another URL is NOT a
        # new independent source
        if self._is_duplicate_content(text):
            print(f"[Finalize] duplicate content skipped: {url[:60]}")
            return None
        # freshness metadata for the NEW evidence: published date
        # (best-effort from the page text) and a deterministic
        # source-type/primary classification from the domain, so the
        # primary/freshness gate can re-evaluate correctly after the
        # additional research
        published, src_type, is_primary = \
            self._classify_new_evidence(url, text)
        evidence = self.locker.add_evidence(
            url=url, title=getattr(page, "title", "") or url,
            content_excerpt=text[:500], extracted_text=text,
            evidence_type=EvidenceType.WEB_PAGE,
            search_query=query,
            section_reference=target_sections[0] if target_sections else "",
            published_date=published,
            source_type=src_type,
        )
        if is_primary:
            evidence.quality_indicators.is_primary_source = True
        if meta:
            try:
                evidence.adaptive_meta = dict(meta)
            except Exception:
                pass
        self._url_to_id[url] = evidence.id
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
        return evidence

    def _research_round(self, issues, queries) -> Dict[str, Any]:
        if self.ledger is not None:
            return self._adaptive_research_round(issues, queries)

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
                target_sections = list(per_issue_sections.keys()) or \
                    list(self.section_evidence.keys())[:1]
                if self._fetch_and_register(url, q, target_sections,
                                            changed_sections) is not None:
                    new_sources += 1

        # length-planner units follow the evidence (audit item 2)
        if changed_sections:
            self._recalc_units()
        return {"new_sources": new_sources,
                "changed_sections": changed_sections}

    # ------------------------------------------------------------------
    # adaptive research round (requirement-granular, gap-only)
    # ------------------------------------------------------------------

    # tasks actually RUN per research round; the rest is DEFERRED (their
    # requirements keep their attempts and run in the next round)
    ROUND_TASK_CAP = 12

    def _adaptive_research_round(self, issues, queries) -> Dict[str, Any]:
        """Gap-only research at REQUIREMENT granularity.

        - the ONE stopping authority (StopController over the Coverage
          Ledger) is consulted first: a stalled or exhausted run performs
          no search at all;
        - only gap requirements (open / conflicted) get queries —
          supported requirements are NEVER re-searched;
        - the round's task list is CAPPED FIRST: only the scheduled
          tasks consume their requirement's search attempt; deferred
          tasks consume nothing and stay runnable next round;
        - multi_hop requirements get a dependent second task whose query
          is refined from the first task's top result;
        - evidence found by a task is registered ONLY to that task's
          target section — section-less results are kept as UNASSIGNED
          evidence in the locker, never sprayed across sections.
        """
        from ..adaptive import TaskDAG, build_intent_queries, rrf_merge
        from ..adaptive.models import (
            TASK_DEFERRED, TASK_EXECUTED, TASK_FAILED, TASK_SCHEDULED,
            ResearchTask)

        stop, stop_reason = self.stop_controller.should_stop(self.ledger)
        if stop:
            print(f"[Adaptive] research stopped: {stop_reason}")
            self.ledger.termination_reason = stop_reason
            self.audit.event("research_stopped", reason=stop_reason,
                             coverage=self.ledger.coverage())
            return {"new_sources": 0, "changed_sections": []}

        gaps = self.ledger.gap_requirements()
        if not gaps:
            self.audit.event("research_skipped",
                             reason="no_gap_requirements")
            return {"new_sources": 0, "changed_sections": []}

        # queries handed in by the controller are ALREADY novel-filtered;
        # map them back to requirement AND section via their issue
        query_meta: Dict[str, Dict[str, str]] = {}
        for issue in issues:
            req = self._requirement_for_issue(issue)
            if req is None:
                continue
            for q in issue.search_queries or []:
                query_meta.setdefault(q, {
                    "req_id": req.req_id,
                    "section_id": issue.section_id or req.section_id,
                    "issue_id": issue.claim_id or issue.type,
                    "claim_id": issue.claim_id,
                })

        gap_ids = {r.req_id for r in gaps}
        tasks: List[ResearchTask] = []
        seq = 0

        def _new_task(req, query, meta=None, depends_on=None):
            nonlocal seq
            seq += 1
            meta = meta or {}
            return ResearchTask(
                task_id=f"T{seq:03d}", req_id=req.req_id, query=query,
                intent=req.intent,
                issue_id=str(meta.get("issue_id", "") or ""),
                section_id=str(meta.get("section_id", "")
                               or req.section_id or ""),
                claim_id=str(meta.get("claim_id", "") or ""),
                depends_on=list(depends_on or []))

        for q in queries:
            meta = query_meta.get(q)
            req_id = (meta or {}).get("req_id") or gaps[0].req_id
            if req_id not in gap_ids:
                continue        # supported requirements are never searched
            tasks.append(_new_task(self.ledger.get(req_id), q, meta))
        # intent-templated diversification for the gap requirements;
        # multi_hop requirements chain a DEPENDENT refinement task whose
        # query is completed from the first task's top result
        for req in gaps[:4]:
            # a requirement demanding PRIMARY evidence searches with the
            # primary_source intent regardless of its textual intent
            intent = ("primary_source"
                      if req.expected_evidence_type == "primary"
                      else req.intent)
            extra = build_intent_queries(req.text, intent,
                                         self.language, max_queries=3)
            fresh = self.budget.novel_queries(extra)[:2]
            prev_task = None
            for q in fresh:
                task = _new_task(
                    req, q,
                    depends_on=[prev_task.task_id]
                    if (req.intent == "multi_hop"
                        and prev_task is not None) else None)
                tasks.append(task)
                prev_task = task
        if not tasks:
            return {"new_sources": 0, "changed_sections": []}

        # ---- cap FIRST, then account: only SCHEDULED tasks consume the
        #      requirement's attempt; deferred tasks consume nothing ----
        scheduled = tasks[:self.ROUND_TASK_CAP]
        deferred = tasks[self.ROUND_TASK_CAP:]
        # a dependency cut by the cap defers its dependents too
        scheduled_ids = {t.task_id for t in scheduled}
        keep: List[ResearchTask] = []
        for t in scheduled:
            if all(dep in scheduled_ids for dep in t.depends_on):
                keep.append(t)
            else:
                deferred.append(t)
                scheduled_ids.discard(t.task_id)
        scheduled = keep
        for t in scheduled:
            t.execution_state = TASK_SCHEDULED
        for t in deferred:
            t.execution_state = TASK_DEFERRED
            # deferred queries stay novel for the NEXT round
            key = re.sub(r"\s+", "", (t.query or "").lower())
            self.budget.seen_queries.discard(key)
        attempted = sorted({t.req_id for t in scheduled})
        for req_id in attempted:
            self.ledger.record_search_attempt(req_id)

        # snapshot BEFORE the round (deltas are measured, not self-reported)
        coverage_before = self.ledger.coverage()
        counts_before = self._claim_counts()

        # parallel search through the DAG scheduler. Permit ownership is
        # ONE-SIDED: the search client takes its own leaf permit inside
        # search(), so the DAG passes NO limiter (double acquisition of
        # the same RunLimits would deadlock at limit=1).
        dag = TaskDAG(scheduled)
        results_map: Dict[str, List] = {}

        def _execute(task: ResearchTask):
            if task.depends_on:
                # multi-hop refinement: the dependent query is completed
                # with the top result of its prerequisite task
                prev = results_map.get(task.depends_on[0]) or []
                top_title = str(getattr(prev[0], "title", "") or "") \
                    if prev else ""
                if top_title:
                    task.query = f"{task.query} {top_title[:48]}"
            value = self.search_client.search(task.query)
            results_map[task.task_id] = value or []
            return value

        results_by_task = dag.run(
            _execute,
            parallel_max_workers=self._rc("parallel_max_workers", 8),
            stage_cap=self.verification_settings.max_workers,
        )
        for task in dag.tasks():
            value = results_by_task.get(task.task_id)
            task.result_count = len(value or [])
            task.execution_state = TASK_FAILED if task.status == "failed" \
                else TASK_EXECUTED

        # ---- per (requirement, section) RRF fusion: evidence follows
        #      its OWN task's target, never other sections ----
        by_target: Dict[tuple, List] = {}
        task_of_target: Dict[tuple, ResearchTask] = {}
        for task in dag.tasks():
            value = results_by_task.get(task.task_id)
            if not value:
                continue
            key = (task.req_id, task.section_id)
            by_target.setdefault(key, []).append(value)
            task_of_target.setdefault(key, task)

        new_sources = 0
        changed_sections: List[str] = []
        new_evidence_ids: List[str] = []
        unassigned: List[str] = []
        for key in sorted(by_target):
            req_id, section_id = key
            req = self.ledger.get(req_id)
            task = task_of_target[key]
            fused = rrf_merge(by_target[key], limit=4)
            targets = [section_id] if section_id else []
            for r in fused:
                url = getattr(r, "url", "")
                evidence = self._fetch_and_register(
                    url, task.query, targets, changed_sections,
                    meta={"query_id": task.task_id,
                          "requirement_id": req_id,
                          "section_id": section_id,
                          "claim_id": task.claim_id})
                if evidence is not None:
                    new_sources += 1
                    new_evidence_ids.append(evidence.id)
                    req.evidence_ids.append(evidence.id)
                    if not targets:
                        unassigned.append(evidence.id)

        if changed_sections:
            self._recalc_units()

        # attempt budget spent and still a gap -> unavailable_after_search
        exhausted = self.ledger.close_exhausted()

        self._pending_round = {
            "round_index": len(self.ledger.rounds) + 1,
            "new_unique_evidence": new_sources,
            "queries_run": len(scheduled),
            "coverage_before": coverage_before,
            "counts_before": counts_before,
            "scheduled_tasks": [t.task_id for t in scheduled],
            "executed_tasks": [t.task_id for t in scheduled
                               if t.execution_state == TASK_EXECUTED],
            "failed_tasks": [t.task_id for t in scheduled
                             if t.execution_state == TASK_FAILED],
            "deferred_tasks": [t.task_id for t in deferred],
        }
        self.audit.event(
            "research_round",
            round=self._pending_round["round_index"],
            gap_requirements=attempted,
            scheduled=[t.to_dict() for t in scheduled],
            deferred=[t.to_dict() for t in deferred],
            new_sources=new_sources,
            new_evidence_ids=new_evidence_ids,
            unassigned_evidence_ids=unassigned,
            unavailable_after_search=exhausted,
        )
        return {"new_sources": new_sources,
                "changed_sections": changed_sections}

    # ------------------------------------------------------------------
    # coverage ledger plumbing
    # ------------------------------------------------------------------

    def _build_requirements(self, critical_questions: List[str],
                            chapters: Dict[str, str]) -> None:
        """Decompose the run into RequirementLeafs (deterministic).

        - one requirement per source question — explicit USER
          requirements first (REQ-U*, highest priority), then EVERY
          research-plan item at EVERY hierarchy level (REQ-Q*). Nothing
          is capped or silently dropped;
        - one claims-support requirement per section (REQ-S*);
        - freshness/primary constraints are attached PER REQUIREMENT
          (text markers or a run-level freshness demand), so a stale
          gate reopens exactly the affected requirement.
        """
        from ..adaptive import classify_intent
        from ..adaptive.models import RequirementLeaf

        run_freshness = self._freshness_required()
        sources = self._requirement_sources()
        u_seq = q_seq = 0
        for src in sources:
            text = src["text"]
            fresh = run_freshness or bool(_FRESHNESS_MARKERS.search(text))
            if src["origin"] == "user":
                u_seq += 1
                rid = f"REQ-U{u_seq}"
            else:
                q_seq += 1
                rid = f"REQ-Q{q_seq}"
            self.ledger.add(RequirementLeaf(
                req_id=rid, text=text,
                section_id=src.get("section", "") or "",
                intent=classify_intent(text),
                priority="critical",
                origin=src["origin"],
                freshness_requirement=fresh,
                primary_source_required=fresh,
                expected_evidence_type="primary" if fresh else ""))
        for sid in chapters:
            text = (f"セクション{sid}の事実主張がエビデンスに支持されている"
                    if self.language == "ja" else
                    f"every factual claim in section {sid} is "
                    f"evidence-backed")
            self.ledger.add(RequirementLeaf(
                req_id=f"REQ-S{sid}", text=text, section_id=sid,
                intent="background", priority="important",
                origin="section",
                freshness_requirement=run_freshness,
                primary_source_required=run_freshness))
        self.audit.event("requirements_decomposed",
                         count=len(self.ledger),
                         requirements=[
                             {"req_id": r.req_id, "intent": r.intent,
                              "priority": r.priority, "origin": r.origin,
                              "freshness": r.freshness_requirement,
                              "text": r.text[:80]}
                             for r in self.ledger.requirements()])

    def _requirement_for_issue(self, issue):
        """Deterministic issue -> requirement mapping."""
        if self.ledger is None:
            return None
        from ..report.finalization import ISSUE_UNANSWERED_QUESTION
        if issue.type == ISSUE_UNANSWERED_QUESTION and issue.claim:
            for req in self.ledger.requirements():
                if req.req_id.startswith(("REQ-U", "REQ-Q")) \
                        and req.text == issue.claim:
                    return req
        if issue.section_id:
            return self.ledger.get(f"REQ-S{issue.section_id}")
        return None

    def _claim_counts(self) -> Dict[str, int]:
        """Measured claim counters from the LAST verdict (for deltas)."""
        return dict(self._last_claim_counts or
                    {"supported": 0, "contradicted": 0})

    def _update_ledger_from_verdict(self, verdict) -> None:
        """Deterministic requirement-state update from one verify pass.

        Freshness is REQUIREMENT-scoped: a stale_or_non_primary_sources
        issue keeps/reopens EXACTLY the requirements it names (by
        section) — a freshness-failing requirement is never marked
        supported just because its claims verified textually.
        """
        from ..adaptive.models import (
            REQ_BUDGET_EXHAUSTED, REQ_CONFLICTED, REQ_NOT_APPLICABLE,
            REQ_OPEN, REQ_SUPPORTED, REQ_UNAVAILABLE)
        from ..report.finalization import (
            FRESHNESS_FAIL, ISSUE_CONTRADICTED, ISSUE_STALE_OR_NON_PRIMARY,
            ISSUE_UNANSWERED_QUESTION, ISSUE_UNSUPPORTED)

        m = verdict.metrics
        supported_claims = max(
            0, m.claims_total - m.skipped_minor_claims - m.unsupported_count
            - m.contradicted_count - m.uncertain_count)
        self._last_claim_counts = {"supported": supported_claims,
                                   "contradicted": m.contradicted_count}

        unanswered = {i.claim for i in verdict.issues
                      if i.type == ISSUE_UNANSWERED_QUESTION and i.claim}
        sec_open: Dict[str, List[str]] = {}
        sec_conflicted: Dict[str, List[str]] = {}
        stale_sections: set = set()
        stale_global = False
        for i in verdict.issues:
            if i.type == ISSUE_STALE_OR_NON_PRIMARY:
                if i.section_id:
                    stale_sections.add(i.section_id)
                else:
                    stale_global = True
                continue
            if not i.section_id:
                continue
            if i.type == ISSUE_UNSUPPORTED:
                sec_open.setdefault(i.section_id, []).append(i.claim_id)
            elif i.type == ISSUE_CONTRADICTED:
                sec_conflicted.setdefault(i.section_id, []).append(i.claim_id)

        freshness_failed = m.primary_freshness == FRESHNESS_FAIL

        def _freshness_blocks(req) -> bool:
            """True when THIS requirement's freshness demand is unmet."""
            if not req.freshness_requirement or not freshness_failed:
                return False
            if req.section_id:
                return req.section_id in stale_sections or (
                    stale_global and not stale_sections)
            # section-less (question) requirements: blocked while the
            # freshness gate fails anywhere the run demands freshness
            return stale_global or bool(stale_sections)

        for req in self.ledger.requirements():
            if req.status in (REQ_NOT_APPLICABLE, REQ_BUDGET_EXHAUSTED):
                continue
            if req.req_id.startswith(("REQ-U", "REQ-Q")):
                target = REQ_OPEN if req.text in unanswered \
                    else REQ_SUPPORTED
            else:
                sid = req.section_id
                if sid in sec_conflicted:
                    target = REQ_CONFLICTED
                    req.claim_ids = sorted(
                        set(sec_conflicted[sid] + sec_open.get(sid, [])))
                elif sid in sec_open:
                    target = REQ_OPEN
                    req.claim_ids = sorted(set(sec_open[sid]))
                else:
                    target = REQ_SUPPORTED
            # freshness gate: an unmet freshness/primary demand keeps
            # the requirement OPEN even when its claims verified
            if target == REQ_SUPPORTED and _freshness_blocks(req):
                target = REQ_OPEN
                reason = ("一次情報・鮮度要件が未充足"
                          if self.language == "ja"
                          else "freshness/primary-source demand unmet")
            elif target == REQ_SUPPORTED and req.section_id and \
                    req.min_independent_sources > 1 and \
                    len(set(self.citation_mgr.evidence_ids(
                        req.section_id))
                        | set(req.evidence_ids)) < \
                    req.min_independent_sources:
                # independent-source floor: fewer registered sources
                # than the requirement demands -> stays open
                target = REQ_OPEN
                reason = (f"独立ソースが{req.min_independent_sources}件"
                          f"未満" if self.language == "ja" else
                          f"fewer than {req.min_independent_sources} "
                          f"independent sources")
            else:
                reason = ("検証パスの結果" if self.language == "ja"
                          else "verification pass")
            if target == req.status:
                continue
            # a searched-out requirement stays unavailable unless the
            # new pass actually SUPPORTS it
            if req.status == REQ_UNAVAILABLE and target != REQ_SUPPORTED:
                continue
            self.ledger.transition(req.req_id, target, reason=reason)
            self.audit.event("requirement_transition",
                             req_id=req.req_id, to=target, reason=reason)

    def _after_verify(self, verdict) -> None:
        """Ledger update + pending-round settlement after ANY verify."""
        if self.ledger is None:
            return
        self._update_ledger_from_verdict(verdict)
        pending = self._pending_round
        if pending is not None:
            from ..adaptive.models import ProgressRound
            counts_now = self._claim_counts()
            counts_before = pending["counts_before"]
            progress = ProgressRound(
                round_index=pending["round_index"],
                new_unique_evidence=pending["new_unique_evidence"],
                new_supported_claims=max(
                    0, counts_now["supported"]
                    - counts_before.get("supported", 0)),
                resolved_conflicts=max(
                    0, counts_before.get("contradicted", 0)
                    - counts_now["contradicted"]),
                coverage_delta=(self.ledger.coverage()
                                - pending["coverage_before"]),
                queries_run=pending["queries_run"],
                scheduled_tasks=pending.get("scheduled_tasks", []),
                executed_tasks=pending.get("executed_tasks", []),
                failed_tasks=pending.get("failed_tasks", []),
                deferred_tasks=pending.get("deferred_tasks", []),
            )
            self.ledger.record_round(progress)
            # the single stopping authority marks termination candidates
            stop_next, why = self.stop_controller.should_stop(self.ledger)
            progress.termination_candidate = stop_next
            if stop_next and not self.ledger.termination_reason:
                self.ledger.termination_reason = why
            self._pending_round = None
            self.audit.event("round_measured", **progress.to_dict())
        self.audit.event(
            "verify_pass",
            claims_total=verdict.metrics.claims_total,
            unsupported=verdict.metrics.unsupported_count,
            contradicted=verdict.metrics.contradicted_count,
            uncertain=verdict.metrics.uncertain_count,
            support_score=round(verdict.metrics.claim_support_score, 3),
            coverage=self.ledger.coverage(),
            counts=self.ledger.counts(),
        )

    # ------------------------------------------------------------------
    # section edits (full evidence, new-first, stable SOURCE numbers)
    # ------------------------------------------------------------------

    def _register_replacements(self, sid: str, issues) -> Dict[int, str]:
        """Make citation-replacement candidates citable in this section.

        For invalid-citation issues carrying replacement evidence ids,
        the ids are APPENDED to the section registry (existing numbers
        never shift) and the evidence body joins the section's evidence
        so the rewrite prompt can quote it. Returns {id(issue): hint}.
        """
        hints: Dict[int, str] = {}
        for issue in issues:
            if issue.type != "invalid_citation" or \
                    not issue.supporting_source_ids:
                continue
            numbers = []
            for eid in issue.supporting_source_ids:
                n = self.citation_mgr.append_evidence(sid, eid)
                numbers.append(n)
                # ensure the evidence text is available to the rewrite
                entry_urls = {ec.get("url") for ec in
                              self.section_evidence.get(sid, [])}
                ev = self.locker.get_evidence(eid)
                if ev is not None and getattr(ev, "url", "") not in entry_urls:
                    text = (getattr(ev, "extracted_text", "") or
                            getattr(ev, "content_excerpt", "") or "")
                    entry = {"title": getattr(ev, "title", "") or eid,
                             "url": getattr(ev, "url", ""),
                             "content": text[:2000], "raw_content": text,
                             "key_points": [], "relevance_score": 0.6,
                             "is_new": True}
                    self.section_evidence.setdefault(sid, []).append(entry)
            if numbers:
                hints[id(issue)] = ", ".join(
                    f"[SOURCE {n}]" for n in numbers)
        return hints

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

    # issue types whose claim text pins an exact body sentence — these
    # are patched LOCALLY (sentence-level), never by section regeneration
    _CLAIM_LEVEL_TYPES = (
        "unsupported_claim", "contradicted_claim", "uncertain_claim",
        "invalid_citation", "claim_without_citation",
        "citation_association_failure",
    )

    @staticmethod
    def _sentence_spans(text: str) -> List:
        """(start, end, sentence) for every sentence, exact offsets."""
        spans = []
        pos = 0
        for m in re.finditer(r"[^。．！？!?.\n]*[。．！？!?.]|[^。．！？!?.\n]+",
                             text):
            sent = m.group(0)
            if sent.strip():
                spans.append((m.start(), m.end(), sent))
            pos = m.end()
        return spans

    @staticmethod
    def _sent_hash(sentence: str) -> str:
        import hashlib
        return hashlib.sha256(sentence.encode("utf-8")).hexdigest()[:24]

    def _locate_claim_span(self, text: str, claim_text: str):
        """Best-matching sentence span for a claim, or None."""
        from ..verification.claim_verifier import _bigrams, _containment
        claim_bi = _bigrams(claim_text or "")
        best, best_score = None, 0.0
        for start, end, sent in self._sentence_spans(text):
            if len(sent.strip()) < 8:
                continue
            score = _containment(claim_bi, _bigrams(sent))
            if score > best_score:
                best, best_score = (start, end, sent), score
        return best if best_score >= 0.3 else None

    def _patch_claims(self, sid: str, text: str, issues,
                      mode: str) -> Optional[str]:
        """LOCALIZED repair: rewrite ONLY the sentences of the failing
        claims; every other sentence is byte-preserved (proved by hash
        comparison, not assumed). Returns the patched section or None
        when any claim cannot be located / any patch is unusable — the
        caller then falls back to the audited safe regeneration.
        """
        from ..adaptive.models import AtomicClaim

        # locate every claim FIRST: one unlocatable claim -> no patching
        located: Dict[tuple, List] = {}
        for issue in issues:
            span = self._locate_claim_span(text, issue.claim)
            if span is None:
                self.audit.event("patch_fallback", section_id=sid,
                                 reason="claim_not_locatable",
                                 claim=(issue.claim or "")[:80])
                return None
            located.setdefault((span[0], span[1]),
                               []).append((issue, span[2]))

        spans_all = self._sentence_spans(text)
        patch_starts = {start for start, _end in located}
        protected = [(start, end, sent) for start, end, sent in spans_all
                     if start not in patch_starts]
        protected_hashes = [self._sent_hash(s) for _s, _e, s in protected]

        replacement_hints = self._register_replacements(sid, issues)
        evidence_text = self._evidence_blocks(sid)
        atomic_claims: List[AtomicClaim] = []
        patches: List[tuple] = []
        for (start, end), group in sorted(located.items()):
            issue, sentence = group[0]
            hint = replacement_hints.get(id(issue)) or ""
            new_sentence = self._patch_sentence(
                sid, sentence, [i for i, _s in group], hint,
                evidence_text, mode)
            if not new_sentence:
                self.audit.event("patch_fallback", section_id=sid,
                                 reason="llm_patch_unusable",
                                 claim=(issue.claim or "")[:80])
                return None
            patches.append((start, end, new_sentence))
            neighbor_hashes = []
            idx = next(i for i, sp in enumerate(spans_all)
                       if sp[0] == start)
            for j in (idx - 1, idx + 1):
                if 0 <= j < len(spans_all) and \
                        spans_all[j][0] not in patch_starts:
                    neighbor_hashes.append(
                        self._sent_hash(spans_all[j][2]))
            atomic_claims.append(AtomicClaim(
                claim_id=issue.claim_id or f"P-{sid}.{start}",
                req_id=f"REQ-S{sid}", section_id=sid,
                text=issue.claim or sentence,
                status=issue.type,
                span_start=start, span_end=end,
                cited_source_ids=list(issue.supporting_source_ids or []),
                protected_neighbor_hashes=neighbor_hashes))

        # splice back-to-front so earlier offsets stay valid
        patched = text
        for start, end, new_sentence in sorted(patches, reverse=True):
            patched = patched[:start] + new_sentence + patched[end:]

        # PROOF, not assumption: every protected sentence must survive
        # byte-identically (its hash is still present in the result)
        result_hashes = {self._sent_hash(s) for _s, _e, s
                         in self._sentence_spans(patched)}
        missing = [h for h in protected_hashes if h not in result_hashes]
        if missing:
            self.audit.event("patch_fallback", section_id=sid,
                             reason="protected_sentence_changed",
                             missing_hashes=missing[:5])
            return None

        self.audit.event(
            "claims_patched", section_id=sid, mode=mode,
            patched=[c.to_dict() for c in atomic_claims],
            protected_sentences=len(protected_hashes))
        return patched

    def _patch_sentence(self, sid: str, sentence: str, issues,
                        hint: str, evidence_text: str,
                        mode: str) -> Optional[str]:
        """Ask the LLM for a replacement of ONE sentence only.

        Output that does not look like a single sentence (headings,
        multi-paragraph text, extreme length) is rejected — the model
        must never smuggle a whole-section rewrite through a patch.
        """
        lang_ja = self.language == "ja"
        issue_text = "\n".join(
            f"- ({i.type}) {i.reason or i.claim}" for i in issues[:4])
        if mode == "hedge":
            instruction = (
                "この一文だけを、断定を避けた限定表現（「〜の可能性がある」"
                "「公開情報では確認できなかった」等）に修正してください。"
                if lang_ja else
                "Rewrite ONLY this sentence into a hedged statement.")
        else:
            instruction = (
                "この一文だけを、上記エビデンスに基づいて正確な内容に修正して"
                "ください。引用 [SOURCE N] は提示された番号のみ使用する。"
                if lang_ja else
                "Rewrite ONLY this sentence so it is supported by the "
                "evidence above. Use only the listed [SOURCE N] numbers.")
        if hint:
            instruction += (f"　正しい出典候補: {hint}" if lang_ja
                            else f" Replacement citation: {hint}")
        prompt = (
            f"{'あなたはレポート編集者です。対象は一文のみです。' if lang_ja else 'You are a report editor. ONE sentence only.'}\n\n"
            f"{'【エビデンス】' if lang_ja else '[EVIDENCE]'}\n{evidence_text}\n\n"
            f"{'【指摘事項】' if lang_ja else '[ISSUES]'}\n{issue_text or '-'}\n\n"
            f"{'【修正対象の一文】' if lang_ja else '[SENTENCE TO FIX]'}\n{sentence}\n\n"
            f"{'【指示】' if lang_ja else '[INSTRUCTIONS]'}\n{instruction}\n\n"
            f"{'修正後の一文のみを出力（見出し・前置き・複数文は不可）:' if lang_ja else 'Output ONLY the corrected sentence:'}")
        try:
            response = self.writing_llm.generate(prompt)
        except Exception as e:
            print(f"[Finalize] sentence patch failed ({sid}): {e}")
            return None
        out = (response.content or "").strip()
        if not out:
            return None
        # reject whole-section output masquerading as a patch
        if out.lstrip().startswith("#") or "\n\n" in out or \
                len(out) > max(400, len(sentence) * 4):
            return None
        return out

    def _edit_section(self, sid: str, issues, current: Dict[str, str],
                      mode: str = "rewrite") -> Optional[str]:
        """LLM edit of ONE section from its EXISTING evidence.

        mode: rewrite  (repair failing claims; deepen from evidence)
              compress (remove redundancy, keep claims & citations)
              hedge    (soften unresolved assertions, state limitations)

        Claim-level defects are patched LOCALLY (sentence spans) so
        supported text never changes; whole-section regeneration is the
        EXPLICIT, audited fallback when a claim cannot be located or a
        patch is unusable. The controller machine-validates citations
        before accepting any result.
        """
        if sid in self.render_only_sections:
            # render-only section (figure semantics): verified, but its
            # text always mirrors the frozen source — never LLM-edited
            return None

        text = current.get(sid, "")
        if not text:
            text = (self.session_contents.get(sid) or {}).get("content", "")
        if not text:
            return None

        # ---- localized patching first (rewrite / hedge) --------------
        if mode in ("rewrite", "hedge"):
            claim_issues = [i for i in issues
                            if i.claim and i.type in self._CLAIM_LEVEL_TYPES]
            if claim_issues:
                patched = self._patch_claims(sid, text, claim_issues, mode)
                if patched is not None:
                    return patched
                # fall through: audited safe regeneration below

        # Misattributed citations: register the replacement candidates in
        # the section registry (append-only) so the rewrite can cite them
        # with a valid, stable [SOURCE N] number.
        replacement_hints = self._register_replacements(sid, issues)

        evidence_text = self._evidence_blocks(sid)
        issue_lines = []
        for i in issues[:8]:
            line = f"- ({i.type}) {i.claim or i.reason}"
            hint = replacement_hints.get(id(i))
            if hint:
                line += ("　→ 正しい出典候補: " if self.language == "ja"
                         else " -> replacement citation: ") + hint
            issue_lines.append(line)
        issue_text = "\n".join(issue_lines)

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
