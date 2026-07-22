"""
Finalization pipeline - state-based decisions for the final report loop.

Implements the pipeline tail:

    draft -> verify -> decide -> (research | rewrite | compress |
    finalize-with-limitations) -> re-verify -> accept -> render

Design principles (from the pipeline specification):
- The text that ships is the text that was verified: any LLM edit to the
  body triggers re-verification, and nothing edits the body after the
  final verification (only deterministic rendering).
- "Not enough" never automatically means "search more": evidence
  shortage (RESEARCH) is separated from explanation shortage
  (REWRITE_FROM_EVIDENCE) and from redundancy (COMPRESS_FROM_EVIDENCE).
- Length is an adaptive range derived from information units, not a
  fixed quota. Short-but-complete is acceptable; useful overshoot within
  the hard maximum is never trimmed. hard_min / hard_max are absolute
  ONLY when the user explicitly set them.
- Loops are bounded; when search cannot resolve an issue the run ends
  with limitations stated in the body (FINALIZE_WITH_LIMITATIONS).
  FINALIZE_WITH_LIMITATIONS order: hedge edits -> deterministic
  limitations section -> citation machine check -> full re-verification
  of the final body -> freeze. The safety-net exit follows the same
  order, so no body ever ships without a verification pass over its
  final form.
- An unverifiable body (verifier failure, zero claims on factual text)
  is NEVER accepted: it ends in FINALIZE_WITH_LIMITATIONS or an error.
- The LLM produces structured assessments; the state choice and every
  threshold comparison happen in Python.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ResearchDecision(Enum):
    ACCEPT = "accept"
    RESEARCH = "research"
    REWRITE_FROM_EVIDENCE = "rewrite_from_evidence"
    COMPRESS_FROM_EVIDENCE = "compress_from_evidence"
    FINALIZE_WITH_LIMITATIONS = "finalize_with_limitations"


# Issue types produced by verification / assessment
ISSUE_UNSUPPORTED = "unsupported_claim"
ISSUE_CONTRADICTED = "contradicted_claim"
ISSUE_UNCERTAIN = "uncertain_claim"
ISSUE_UNCITED_CLAIM = "claim_without_citation"
ISSUE_UNANSWERED_QUESTION = "unanswered_critical_question"
ISSUE_INSUFFICIENT_EXPLANATION = "insufficient_explanation"
ISSUE_UNUSED_EVIDENCE = "unused_high_importance_evidence"
ISSUE_REDUNDANCY = "redundant_content"
ISSUE_INVALID_CITATION = "invalid_citation"
ISSUE_ORPHAN_CITATION = "orphan_citation"
ISSUE_ALL_CITATIONS_DELETED = "all_citations_deleted"
ISSUE_OVER_HARD_MAX = "over_hard_max"
ISSUE_UNDER_HARD_MIN = "under_hard_min"
ISSUE_VERIFICATION_FAILURE = "verification_failure"
ISSUE_STALE_OR_NON_PRIMARY = "stale_or_non_primary_sources"

# primary/freshness gate states (3-state, never a silent boolean)
FRESHNESS_PASS = "pass"
FRESHNESS_FAIL = "fail"
FRESHNESS_NOT_REQUIRED = "not_required"


@dataclass
class VerificationIssue:
    """One structured problem found in the candidate body."""
    section_id: str = ""
    claim_id: str = ""
    type: str = ""
    severity: str = "important"       # critical / important / minor
    claim: str = ""
    reason: str = ""
    supporting_source_ids: List[str] = field(default_factory=list)
    needed_evidence: Optional[str] = None
    search_queries: List[str] = field(default_factory=list)
    fallback_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "claim_id": self.claim_id,
            "type": self.type,
            "severity": self.severity,
            "claim": self.claim,
            "reason": self.reason,
            "supporting_source_ids": self.supporting_source_ids,
            "needed_evidence": self.needed_evidence,
            "search_queries": self.search_queries,
            "fallback_action": self.fallback_action,
        }


@dataclass
class SectionAssessment:
    """Per-section length/content evaluation (spec section 5)."""
    section_id: str
    importance: str = "normal"         # high / normal / low
    actual_chars: int = 0
    recommended_min_chars: int = 0
    recommended_chars: int = 0
    recommended_max_chars: int = 0
    missing_content_units: List[str] = field(default_factory=list)
    unused_evidence_ids: List[str] = field(default_factory=list)
    redundant_passages: List[str] = field(default_factory=list)
    recommended_action: str = ResearchDecision.ACCEPT.value
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "importance": self.importance,
            "actual_chars": self.actual_chars,
            "recommended_min_chars": self.recommended_min_chars,
            "recommended_chars": self.recommended_chars,
            "recommended_max_chars": self.recommended_max_chars,
            "missing_content_units": self.missing_content_units,
            "unused_evidence_ids": self.unused_evidence_ids,
            "redundant_passages": self.redundant_passages,
            "recommended_action": self.recommended_action,
            "reason": self.reason,
        }


@dataclass
class VerificationMetrics:
    """Aggregated metrics for one verification pass."""
    critical_question_coverage: float = 1.0
    claim_support_score: float = 1.0
    unsupported_critical_claims: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    uncertain_count: int = 0
    unresolved_contradictions: int = 0
    citations_valid: bool = True
    # primary/freshness gate: pass / fail / not_required (3-state)
    primary_freshness: str = FRESHNESS_NOT_REQUIRED
    actual_body_chars: int = 0
    preferred_body_chars: Optional[int] = None
    recommended_min_chars: int = 0
    recommended_chars: int = 0
    recommended_max_chars: int = 0
    missing_content_units: List[str] = field(default_factory=list)
    unused_high_importance_evidence_ids: List[str] = field(default_factory=list)
    redundant_passages: List[str] = field(default_factory=list)
    # fail-closed accounting: the verifier records HOW MUCH it verified
    claims_total: int = 0
    skipped_minor_claims: int = 0     # fast profile: sampled-out minors
    chunks_total: int = 0
    chunks_failed: int = 0
    extraction_errors: List[str] = field(default_factory=list)
    # True when verification could not actually verify the body
    # (0 claims on factual text, all chunks failed, verifier crash):
    # such a body must never be ACCEPTed.
    verification_failed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "critical_question_coverage": self.critical_question_coverage,
            "claim_support_score": self.claim_support_score,
            "unsupported_critical_claims": self.unsupported_critical_claims,
            "unsupported_count": self.unsupported_count,
            "contradicted_count": self.contradicted_count,
            "uncertain_count": self.uncertain_count,
            "unresolved_contradictions": self.unresolved_contradictions,
            "citations_valid": self.citations_valid,
            "primary_freshness": self.primary_freshness,
            "actual_body_chars": self.actual_body_chars,
            "preferred_body_chars": self.preferred_body_chars,
            "recommended_min_chars": self.recommended_min_chars,
            "recommended_chars": self.recommended_chars,
            "recommended_max_chars": self.recommended_max_chars,
            "missing_content_units": self.missing_content_units,
            "unused_high_importance_evidence_ids":
                self.unused_high_importance_evidence_ids,
            "redundant_passages": self.redundant_passages,
            "claims_total": self.claims_total,
            "skipped_minor_claims": self.skipped_minor_claims,
            "chunks_total": self.chunks_total,
            "chunks_failed": self.chunks_failed,
            "extraction_errors": self.extraction_errors,
            "verification_failed": self.verification_failed,
        }


@dataclass
class StructuredVerdict:
    """The structured verification output (spec section 3)."""
    decision: str = ResearchDecision.ACCEPT.value   # advisory; Python re-decides
    issues: List[VerificationIssue] = field(default_factory=list)
    metrics: VerificationMetrics = field(default_factory=VerificationMetrics)
    section_assessments: Dict[str, SectionAssessment] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics.to_dict(),
            "section_assessments": {
                k: v.to_dict() for k, v in self.section_assessments.items()},
        }

    def issues_of(self, *types: str) -> List[VerificationIssue]:
        return [i for i in self.issues if i.type in types]

    def critical_issues(self) -> List[VerificationIssue]:
        return [i for i in self.issues if i.severity == "critical"]


# ---------------------------------------------------------------------------
# Body character counting (markdown syntax / TOC / references excluded)
# ---------------------------------------------------------------------------

_REFERENCE_HEADINGS = (
    "参考文献", "引用文献", "references", "sources", "bibliography",
    "出典一覧", "用語集", "glossary", "処理中の警告",
)


def count_body_chars(text: str, exclude_references: bool = True) -> int:
    """Count body characters, excluding markdown syntax, TOC and references.

    Used for every length judgement so that markup and reference lists
    never inflate the perceived body size.
    """
    if not text:
        return 0
    lines = []
    in_code = False
    in_reference = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip().lower()
            if exclude_references and any(
                    h in title for h in _REFERENCE_HEADINGS):
                in_reference = True
                continue
            in_reference = False
            continue    # headings are structure, not body
        if in_reference:
            continue
        if stripped.startswith("|"):        # tables: keep cell text only
            stripped = re.sub(r"[|:\-\s]+", "", stripped)
        lines.append(stripped)
    body = "\n".join(lines)
    body = re.sub(r"\[SOURCE:?\s*\d+\]", "", body)        # citation tags
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", body)      # images
    body = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", body)  # links -> text
    body = re.sub(r"[*_`>#]+", "", body)                  # md tokens
    body = re.sub(r"\s+", "", body)                       # whitespace
    return len(body)


# ---------------------------------------------------------------------------
# Decision logic (pure Python; spec sections 2, 5, 7)
# ---------------------------------------------------------------------------

@dataclass
class LoopBudget:
    """Round limits and stagnation state for the finalization loop."""
    max_final_research_rounds: int = 2
    max_final_revision_rounds: int = 2
    max_no_improvement_rounds: int = 1
    min_score_improvement: float = 0.03
    min_new_independent_sources: int = 1
    min_claim_support_score: float = 0.85
    required_critical_coverage: float = 1.0

    research_rounds: int = 0
    revision_rounds: int = 0
    no_improvement_rounds: int = 0
    prev_score: Optional[float] = None
    seen_queries: set = field(default_factory=set)
    seen_urls: set = field(default_factory=set)

    def research_allowed(self) -> bool:
        # Strict comparisons: max_no_improvement_rounds=1 means ONE
        # stagnant round stops further research (was off-by-one).
        return (self.research_rounds < self.max_final_research_rounds
                and self.no_improvement_rounds < max(
                    1, self.max_no_improvement_rounds))

    def revision_allowed(self) -> bool:
        return self.revision_rounds < self.max_final_revision_rounds

    def novel_queries(self, queries: List[str]) -> List[str]:
        """Drop queries already tried (identical or near-identical)."""
        fresh = []
        for q in queries:
            key = re.sub(r"\s+", "", (q or "").lower())
            if not key or key in self.seen_queries:
                continue
            # near-duplicate: an already-seen query containing / contained by
            if any(key in s or s in key for s in self.seen_queries):
                continue
            self.seen_queries.add(key)
            fresh.append(q)
        return fresh

    def is_novel_url(self, url: str) -> bool:
        """True exactly once per URL; registers the URL as seen."""
        if not url or url in self.seen_urls:
            return False
        self.seen_urls.add(url)
        return True

    def register_score(self, score: float, resolved_issues: int,
                       new_independent_sources: int) -> None:
        """Track improvement; stagnation feeds the stop conditions."""
        improved = False
        if self.prev_score is not None:
            if (score - self.prev_score) >= self.min_score_improvement:
                improved = True
        if resolved_issues > 0:
            improved = True
        if new_independent_sources >= self.min_new_independent_sources:
            improved = True
        if self.prev_score is None:
            improved = True     # first round is not stagnation
        self.prev_score = score
        if improved:
            self.no_improvement_rounds = 0
        else:
            self.no_improvement_rounds += 1


def passes_hard_gates(verdict: StructuredVerdict, budget: LoopBudget) -> bool:
    """Hard gates that averages and length can never override (spec 7)."""
    m = verdict.metrics
    if m.verification_failed:
        return False
    if m.chunks_failed > 0:
        # unverified body ranges remain -> never a clean ACCEPT
        return False
    if m.critical_question_coverage < budget.required_critical_coverage:
        return False
    if m.unsupported_critical_claims > 0:
        return False
    if not m.citations_valid:
        return False
    if m.primary_freshness == FRESHNESS_FAIL:
        return False
    return True


def decide(
    verdict: StructuredVerdict,
    budget: LoopBudget,
    hard_max_body_chars: Optional[int] = None,
    hard_min_body_chars: Optional[int] = None,
    fixed_target_chars: Optional[int] = None,
    length_tolerance: float = 0.20,
) -> ResearchDecision:
    """Choose the next state. Thresholds live here, not in the LLM.

    Priority: unverifiable body > hard length ceiling > evidence
    problems > citation defects > hard floor > redundancy > shallow
    explanation > accept.

    hard_min / hard_max apply ONLY when the caller passes them (i.e. the
    user explicitly set them); a legacy target_characters value in
    adaptive mode never reaches this function as a hard bound and can
    therefore never trigger research on its own.
    """
    m = verdict.metrics

    # --- unverifiable body is never accepted ---
    if m.verification_failed:
        return ResearchDecision.FINALIZE_WITH_LIMITATIONS

    # --- absolute ceiling: compress (keep claims & citations) ---
    if hard_max_body_chars and m.actual_body_chars > hard_max_body_chars:
        if budget.revision_allowed():
            return ResearchDecision.COMPRESS_FROM_EVIDENCE
        return ResearchDecision.FINALIZE_WITH_LIMITATIONS

    # --- evidence-level problems -> research (bounded) ---
    needs_research = (
        m.critical_question_coverage < budget.required_critical_coverage
        or m.unsupported_critical_claims > 0
        or m.primary_freshness == FRESHNESS_FAIL
        or any(i.needed_evidence for i in verdict.issues)
        or bool(verdict.issues_of(ISSUE_UNSUPPORTED, ISSUE_CONTRADICTED,
                                  ISSUE_UNANSWERED_QUESTION))
    )
    if needs_research:
        if budget.research_allowed():
            return ResearchDecision.RESEARCH
        return ResearchDecision.FINALIZE_WITH_LIMITATIONS

    # --- misattributed citations with KNOWN replacement candidates are a
    #     body defect: rewrite swaps the citation; research would be waste ---
    if not m.citations_valid:
        fixable = any(i.type == ISSUE_INVALID_CITATION
                      and i.supporting_source_ids
                      for i in verdict.issues)
        if fixable and budget.revision_allowed():
            return ResearchDecision.REWRITE_FROM_EVIDENCE

    # --- support score below threshold but nothing actionable by search ---
    if m.claim_support_score < budget.min_claim_support_score:
        if budget.research_allowed():
            return ResearchDecision.RESEARCH
        return ResearchDecision.FINALIZE_WITH_LIMITATIONS

    # --- invalid citations are a body defect: fix from evidence, not search ---
    if not m.citations_valid:
        if budget.revision_allowed():
            return ResearchDecision.REWRITE_FROM_EVIDENCE
        return ResearchDecision.FINALIZE_WITH_LIMITATIONS

    # --- absolute floor (user-set only): expand from evidence, then
    #     research, NEVER padding. An evidence-poor short body ends in
    #     RESEARCH or limitations, not in inflated prose. ---
    if hard_min_body_chars and m.actual_body_chars < hard_min_body_chars:
        has_untapped_evidence = bool(
            m.unused_high_importance_evidence_ids or m.missing_content_units
            or verdict.issues_of(ISSUE_UNUSED_EVIDENCE,
                                 ISSUE_INSUFFICIENT_EXPLANATION))
        if has_untapped_evidence and budget.revision_allowed():
            return ResearchDecision.REWRITE_FROM_EVIDENCE
        if budget.research_allowed():
            return ResearchDecision.RESEARCH
        return ResearchDecision.FINALIZE_WITH_LIMITATIONS

    # --- fixed mode: the user-set target ± tolerance is evaluated in
    #     production. Overshoot compresses; shortfall is fixed from
    #     EVIDENCE (rewrite) or by research — never by padding, and an
    #     over-target body is compressed, never mechanically truncated ---
    if fixed_target_chars:
        upper = int(fixed_target_chars * (1 + length_tolerance))
        lower = int(fixed_target_chars * (1 - length_tolerance))
        if m.actual_body_chars > upper and budget.revision_allowed():
            return ResearchDecision.COMPRESS_FROM_EVIDENCE
        if m.actual_body_chars < lower:
            has_untapped = bool(
                m.unused_high_importance_evidence_ids
                or m.missing_content_units)
            if has_untapped and budget.revision_allowed():
                return ResearchDecision.REWRITE_FROM_EVIDENCE
            if budget.research_allowed():
                return ResearchDecision.RESEARCH
            return ResearchDecision.FINALIZE_WITH_LIMITATIONS

    # --- redundancy / imbalance -> compress (must carry reasons) ---
    if m.redundant_passages and budget.revision_allowed():
        return ResearchDecision.COMPRESS_FROM_EVIDENCE

    # --- shallow explanation with sufficient evidence -> rewrite, no search ---
    shallow = (
        bool(m.missing_content_units)
        or bool(m.unused_high_importance_evidence_ids)
        or bool(verdict.issues_of(ISSUE_INSUFFICIENT_EXPLANATION,
                                  ISSUE_UNUSED_EVIDENCE,
                                  ISSUE_UNCITED_CLAIM))
    )
    if shallow and budget.revision_allowed():
        return ResearchDecision.REWRITE_FROM_EVIDENCE

    # --- unverified body ranges (chunk extraction failed after retries):
    #     no edit can fix this; end EXPLICITLY with limitations instead of
    #     looping or accepting a partially verified body ---
    if m.chunks_failed > 0:
        return ResearchDecision.FINALIZE_WITH_LIMITATIONS

    # Short but complete is acceptable; useful overshoot under the hard
    # ceiling is acceptable — no length-only rejections here by design.
    if passes_hard_gates(verdict, budget):
        return ResearchDecision.ACCEPT

    # Gates failed but nothing actionable remains
    if budget.research_allowed():
        return ResearchDecision.RESEARCH
    return ResearchDecision.FINALIZE_WITH_LIMITATIONS


def decide_section_action(assessment: SectionAssessment,
                          evidence_sufficient: bool) -> str:
    """Per-section recommendation (spec section 5, 初稿後判定)."""
    a = assessment
    if a.redundant_passages:
        return ResearchDecision.COMPRESS_FROM_EVIDENCE.value
    if a.missing_content_units:
        if evidence_sufficient:
            return ResearchDecision.REWRITE_FROM_EVIDENCE.value
        return ResearchDecision.RESEARCH.value
    if a.actual_chars < a.recommended_min_chars:
        # short: fine when nothing required is missing
        if evidence_sufficient or not a.missing_content_units:
            return ResearchDecision.ACCEPT.value
        return ResearchDecision.RESEARCH.value
    return ResearchDecision.ACCEPT.value


# ---------------------------------------------------------------------------
# Finalization controller
# ---------------------------------------------------------------------------

class FinalizationController:
    """Runs the bounded verify → decide → act → re-verify loop.

    All actions are injected callables so the controller stays pure and
    testable without a network or a real LLM:

      verify_fn(chapters)                 -> StructuredVerdict
      research_fn(issues, queries)        -> {"new_sources": int}
      rewrite_fn(section_id, issues)      -> new_text or None
      compress_fn(section_id, issues)     -> new_text or None
      hedge_fn(section_id, issues)        -> new_text or None  (limitations)
      validate_citations_fn(section_id, text[, previous_text]) -> bool

    Invariants enforced here:
    - after ANY body change (LLM edit OR deterministic limitations
      append), verify_fn runs again before the loop returns
    - once the loop returns, no further body-changing callable is invoked
      (rendering afterwards must be deterministic)
    - a RESEARCH round with no novel queries is never executed and never
      consumes a research round
    """

    def __init__(
        self,
        verify_fn: Callable,
        research_fn: Callable = None,
        rewrite_fn: Callable = None,
        compress_fn: Callable = None,
        hedge_fn: Callable = None,
        validate_citations_fn: Callable = None,
        budget: LoopBudget = None,
        hard_max_body_chars: Optional[int] = None,
        hard_min_body_chars: Optional[int] = None,
        fixed_target_chars: Optional[int] = None,
        length_tolerance: float = 0.20,
        language: str = "ja",
        progress=None,
    ):
        self.verify_fn = verify_fn
        self.research_fn = research_fn
        self.rewrite_fn = rewrite_fn
        self.compress_fn = compress_fn
        self.hedge_fn = hedge_fn
        self.validate_citations_fn = validate_citations_fn or (lambda s, t: True)
        self.budget = budget or LoopBudget()
        self.hard_max_body_chars = hard_max_body_chars
        self.hard_min_body_chars = hard_min_body_chars
        self.fixed_target_chars = fixed_target_chars
        self.length_tolerance = length_tolerance
        self.language = language
        # optional verification.runtime.VerificationProgress: phase and
        # round reporting plus SAFE cancellation/timeout boundaries
        self.progress = progress
        self.history: List[Dict[str, Any]] = []
        self.limitations: List[str] = []
        self._last_new_sources = 0

    def _checkpoint(self) -> None:
        if self.progress is not None:
            self.progress.checkpoint()

    def _phase(self, phase: str, label: str = "") -> None:
        if self.progress is not None:
            self.progress.set_phase(phase, label)

    # -- helpers ----------------------------------------------------------

    def _sections_with_issues(self, verdict: StructuredVerdict,
                              *types: str) -> List[str]:
        secs = []
        for issue in verdict.issues:
            if types and issue.type not in types:
                continue
            if issue.section_id and issue.section_id not in secs:
                secs.append(issue.section_id)
        return secs

    def _validate_citations(self, section_id: str, text: str,
                            previous_text: Optional[str] = None) -> bool:
        """Call the injected validator; 2-arg validators stay supported."""
        try:
            return self.validate_citations_fn(section_id, text, previous_text)
        except TypeError:
            return self.validate_citations_fn(section_id, text)

    def _apply_edit(self, chapters: Dict[str, str], section_id: str,
                    new_text: Optional[str]) -> bool:
        """Apply an edited section iff its citations survive validation.

        The previous text is passed so an edit that deletes every
        citation from a previously-cited section is rejected too.
        """
        if not new_text or not new_text.strip():
            return False
        if not self._validate_citations(section_id, new_text,
                                        chapters.get(section_id)):
            print(f"[Finalize] rejected edit for section {section_id}: "
                  f"invalid citations in LLM output")
            return False
        chapters[section_id] = new_text
        return True

    def _append_limitations(self, chapters: Dict[str, str],
                            verdict: StructuredVerdict) -> bool:
        """Deterministically record unresolved issues in the body.

        Returns True when the body changed (the caller must re-verify)."""
        unresolved = [i for i in verdict.issues]
        if not unresolved and not self.limitations:
            return False
        if self.language == "ja":
            lines = ["", "### 調査上の限界", ""]
            for note in self.limitations:
                lines.append(f"- {note}")
            for i in unresolved:
                what = i.claim or i.reason or i.type
                lines.append(f"- 未解決: {what}（{i.reason or i.type}）")
        else:
            lines = ["", "### Research Limitations", ""]
            for note in self.limitations:
                lines.append(f"- {note}")
            for i in unresolved:
                what = i.claim or i.reason or i.type
                lines.append(f"- Unresolved: {what} ({i.reason or i.type})")
        last_key = list(chapters.keys())[-1] if chapters else None
        if last_key is None:
            return False
        chapters[last_key] = chapters[last_key].rstrip() + "\n" + "\n".join(lines)
        return True

    # -- main loop --------------------------------------------------------

    def run(self, chapters: Dict[str, str]) -> Dict[str, Any]:
        """Run the loop; returns final chapters + verdict + decision trail.

        The returned chapters are FROZEN: they have been verified after
        the last body change (including the deterministic limitations
        section), and callers must only apply deterministic rendering
        afterwards.
        """
        verdict = self.verify_fn(chapters)
        max_total_rounds = (self.budget.max_final_research_rounds
                            + self.budget.max_final_revision_rounds + 2)

        for _round in range(max_total_rounds + 1):
            self._checkpoint()      # safe cancel/timeout boundary
            if self.progress is not None:
                self.progress.set_round(_round + 1)
            decision = decide(verdict, self.budget, self.hard_max_body_chars,
                              self.hard_min_body_chars,
                              fixed_target_chars=self.fixed_target_chars,
                              length_tolerance=self.length_tolerance)
            self.history.append({
                "round": _round,
                "decision": decision.value,
                "score": verdict.metrics.claim_support_score,
                "issues": len(verdict.issues),
            })

            if decision == ResearchDecision.ACCEPT:
                return self._finish(chapters, verdict, decision)

            if decision == ResearchDecision.FINALIZE_WITH_LIMITATIONS:
                return self._exit_with_limitations(chapters, verdict)

            changed_any = False
            prev_issue_count = len(verdict.issues)

            if decision == ResearchDecision.RESEARCH:
                self._phase("researching")
                changed_any = self._do_research(chapters, verdict)
            elif decision == ResearchDecision.REWRITE_FROM_EVIDENCE:
                self._phase("revising")
                changed_any = self._do_revision(
                    chapters, verdict, self.rewrite_fn,
                    (ISSUE_INSUFFICIENT_EXPLANATION, ISSUE_UNUSED_EVIDENCE,
                     ISSUE_UNCITED_CLAIM))
            elif decision == ResearchDecision.COMPRESS_FROM_EVIDENCE:
                self._phase("revising")
                changed_any = self._do_revision(
                    chapters, verdict, self.compress_fn,
                    (ISSUE_REDUNDANCY, ISSUE_OVER_HARD_MAX))

            # Any body change (or even a no-op action) demands re-verification
            verdict = self.verify_fn(chapters)
            resolved = max(0, prev_issue_count - len(verdict.issues))
            self.budget.register_score(
                verdict.metrics.claim_support_score,
                resolved_issues=resolved,
                new_independent_sources=self._last_new_sources,
            )
            if not changed_any:
                # action produced nothing -> counts as stagnation
                self.budget.no_improvement_rounds = max(
                    self.budget.no_improvement_rounds, 1)

        # Safety net: bounded loop exhausted. Same exit contract as
        # FINALIZE_WITH_LIMITATIONS: the post-limitations body is
        # re-verified before it ships.
        return self._exit_with_limitations(chapters, verdict,
                                           note_hedge=False)

    def _exit_with_limitations(self, chapters: Dict[str, str],
                               verdict: StructuredVerdict,
                               note_hedge: bool = True) -> Dict[str, Any]:
        """FINALIZE_WITH_LIMITATIONS in the mandated order:
        hedge -> limitations append -> citation machine check ->
        full re-verification -> freeze."""
        changed = False
        if note_hedge:
            changed = self._finalize_with_limitations(chapters, verdict)
        else:
            if self.language == "ja":
                self.limitations.append(
                    "検証ループの上限に達したため、未解決の論点が残っています。")
            else:
                self.limitations.append(
                    "The verification loop budget was exhausted; some "
                    "points remain unresolved.")
        appended = self._append_limitations(chapters, verdict)

        # Citation machine check over the final body (defense in depth;
        # the re-verification below also validates via the registry)
        for sid, text in chapters.items():
            if not self._validate_citations(sid, text):
                print(f"[Finalize] citation check failed on final body "
                      f"for section {sid}")

        if changed or appended:
            verdict = self.verify_fn(chapters)   # verify the FINAL body
        return self._finish(chapters, verdict,
                            ResearchDecision.FINALIZE_WITH_LIMITATIONS)

    def _do_research(self, chapters: Dict[str, str],
                     verdict: StructuredVerdict) -> bool:
        """Targeted additional research; never repeats queries/URLs.

        A round with no novel queries is NOT executed and does NOT
        consume the research budget — it registers as stagnation via the
        caller instead of burning rounds on empty searches.
        """
        self._last_new_sources = 0
        issues = verdict.issues_of(
            ISSUE_UNSUPPORTED, ISSUE_CONTRADICTED, ISSUE_UNANSWERED_QUESTION
        ) or verdict.issues
        queries = []
        for issue in issues:
            queries.extend(issue.search_queries or [])
        queries = self.budget.novel_queries(queries)
        if not queries or self.research_fn is None:
            return False    # no round consumed
        self._checkpoint()          # safe boundary before research I/O
        self.budget.research_rounds += 1
        try:
            outcome = self.research_fn(issues, queries) or {}
        except Exception as e:
            print(f"[Finalize] research round failed: {e}")
            return False
        self._last_new_sources = int(outcome.get("new_sources", 0))
        changed = bool(outcome.get("changed_sections"))
        # After research, affected sections are rewritten from evidence
        if self.rewrite_fn:
            for sid in outcome.get("changed_sections", []) or \
                    self._sections_with_issues(verdict):
                new_text = self.rewrite_fn(sid, issues)
                if self._apply_edit(chapters, sid, new_text):
                    changed = True
        return changed or self._last_new_sources > 0

    def _do_revision(self, chapters: Dict[str, str],
                     verdict: StructuredVerdict, fn: Callable,
                     issue_types) -> bool:
        """Rewrite/compress affected sections from EXISTING evidence."""
        if fn is None:
            return False
        self.budget.revision_rounds += 1
        sections = self._sections_with_issues(verdict, *issue_types)
        if not sections:
            # metrics-level signal without per-section issue: touch the
            # sections named in metrics, else all
            sections = list(chapters.keys())
        changed = False
        issues = verdict.issues_of(*issue_types) or verdict.issues
        for sid in sections:
            if sid not in chapters:
                continue
            self._checkpoint()      # safe boundary between section edits
            try:
                new_text = fn(sid, [i for i in issues
                                    if i.section_id in ("", sid)])
            except Exception as e:
                print(f"[Finalize] revision failed for {sid}: {e}")
                continue
            if self._apply_edit(chapters, sid, new_text):
                changed = True
        return changed

    def _finalize_with_limitations(self, chapters: Dict[str, str],
                                   verdict: StructuredVerdict) -> bool:
        """Soften unresolved assertions (LLM); caller re-verifies."""
        if self.language == "ja":
            self.limitations.append(
                "追加調査の上限に達したか、新しい独立ソースが得られなかったため、"
                "一部の論点は未確認のままです。")
        else:
            self.limitations.append(
                "Research limits were reached; some points remain unverified.")
        if self.hedge_fn is None:
            return False
        changed = False
        for sid in self._sections_with_issues(verdict):
            if sid not in chapters:
                continue
            try:
                new_text = self.hedge_fn(
                    sid, [i for i in verdict.issues if i.section_id == sid])
            except Exception as e:
                print(f"[Finalize] hedging failed for {sid}: {e}")
                continue
            if self._apply_edit(chapters, sid, new_text):
                changed = True
        return changed

    def _finish(self, chapters, verdict, decision):
        over_hard_max = bool(
            self.hard_max_body_chars
            and verdict.metrics.actual_body_chars > self.hard_max_body_chars)
        if over_hard_max and decision == ResearchDecision.ACCEPT:
            # never report a hard-max violation as a clean completion
            decision = ResearchDecision.FINALIZE_WITH_LIMITATIONS
        return {
            "chapters": chapters,
            "verdict": verdict,
            "decision": decision.value,
            "history": self.history,
            "limitations": self.limitations,
            "over_hard_max": over_hard_max,
        }
