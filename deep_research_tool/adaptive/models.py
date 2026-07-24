"""
Typed models for adaptive Deep Research (requirement granularity).

These dataclasses replace ad-hoc dicts so gap state can be tracked and
tested per REQUIREMENT / CLAIM / ROUND instead of one document-wide
boolean. They are plain data — all behavior (transitions, stopping,
scheduling) lives in ledger.py / dag.py.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Requirement coverage states (the ledger's vocabulary)
# --------------------------------------------------------------------------

REQ_OPEN = "open"                              # not yet supported
REQ_SUPPORTED = "supported"                    # verified against evidence
REQ_CONFLICTED = "conflicted"                  # evidence disagrees
REQ_UNAVAILABLE = "unavailable_after_search"   # searched, nothing usable
REQ_NOT_APPLICABLE = "not_applicable"          # explicitly out of scope
REQ_BUDGET_EXHAUSTED = "budget_exhausted"      # stopped by budget/stall

REQUIREMENT_STATES = (
    REQ_OPEN, REQ_SUPPORTED, REQ_CONFLICTED,
    REQ_UNAVAILABLE, REQ_NOT_APPLICABLE, REQ_BUDGET_EXHAUSTED,
)

# states that need no further work (terminal for the research loop)
TERMINAL_STATES = (REQ_SUPPORTED, REQ_NOT_APPLICABLE,
                   REQ_UNAVAILABLE, REQ_BUDGET_EXHAUSTED)
# states that legitimately trigger gap research
GAP_STATES = (REQ_OPEN, REQ_CONFLICTED)


@dataclass
class RequirementLeaf:
    """One atomic requirement the report must satisfy.

    Produced by decomposing the research plan / user requirements; the
    unit at which coverage, gap research and stopping are decided.
    """
    req_id: str
    text: str
    section_id: str = ""
    intent: str = "background"          # see intent.QUERY_INTENTS
    priority: str = "important"         # critical / important / minor
    origin: str = "plan"                # user / plan / section
    status: str = REQ_OPEN
    status_reason: str = ""
    # WHY the requirement ended in a terminal state (shown in outcome,
    # audit and GUI; empty while non-terminal)
    terminal_reason: str = ""
    # evidence constraints, evaluated PER REQUIREMENT (never from the
    # union of the whole document's evidence)
    freshness_requirement: bool = False
    primary_source_required: bool = False
    min_independent_sources: int = 1
    expected_evidence_type: str = ""    # e.g. "statistics", "primary"
    claim_ids: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    search_attempts: int = 0
    history: List[str] = field(default_factory=list)   # state trail

    @property
    def criticality(self) -> str:       # spec alias for priority
        return self.priority

    def to_dict(self) -> Dict[str, Any]:
        return {
            "req_id": self.req_id,
            "text": self.text,
            "section_id": self.section_id,
            "intent": self.intent,
            "priority": self.priority,
            "criticality": self.priority,
            "origin": self.origin,
            "status": self.status,
            "status_reason": self.status_reason,
            "terminal_reason": self.terminal_reason,
            "freshness_requirement": self.freshness_requirement,
            "primary_source_required": self.primary_source_required,
            "min_independent_sources": self.min_independent_sources,
            "expected_evidence_type": self.expected_evidence_type,
            "claim_ids": list(self.claim_ids),
            "evidence_ids": list(self.evidence_ids),
            "search_attempts": self.search_attempts,
            "history": list(self.history),
        }


# execution states a task moves through in ONE research round.
# scheduled: picked for this round (consumes the requirement's attempt)
# executed:  the search actually ran (successfully)
# failed:    the search ran and errored
# deferred:  cut by the round's task cap — attempt NOT consumed,
#            runnable in the next round
TASK_SCHEDULED = "scheduled"
TASK_EXECUTED = "executed"
TASK_FAILED = "failed"
TASK_DEFERRED = "deferred"


@dataclass
class ResearchTask:
    """One schedulable unit of gap research (a query for a requirement).

    Carries the FULL provenance chain (requirement -> issue -> section
    -> claim) so evidence found by this task is registered ONLY where
    the task aimed — never sprayed across unrelated sections.
    """
    task_id: str
    req_id: str
    query: str
    intent: str = "background"
    issue_id: str = ""                  # claim_id / issue key it serves
    section_id: str = ""                # ONLY registration target
    claim_id: str = ""
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"             # DAG-internal: pending/running/done/failed
    execution_state: str = "pending"    # scheduled/executed/failed/deferred
    result_count: int = 0
    error: str = ""

    @property
    def requirement_id(self) -> str:    # spec alias
        return self.req_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "req_id": self.req_id,
            "requirement_id": self.req_id,
            "issue_id": self.issue_id,
            "section_id": self.section_id,
            "claim_id": self.claim_id,
            "query": self.query,
            "intent": self.intent,
            "depends_on": list(self.depends_on),
            "status": self.status,
            "execution_state": self.execution_state,
            "result_count": self.result_count,
            "error": self.error,
        }


@dataclass
class AtomicClaim:
    """A verified factual claim linked back to its requirement.

    Carries its exact body location (span) and the hashes of its
    PROTECTED neighbor sentences so localized patching can prove that
    supported text was not touched.
    """
    claim_id: str
    req_id: str
    section_id: str
    text: str
    status: str = "uncertain"    # supported/unsupported/contradicted/uncertain
    # locator: character span of the claim's sentence in the section body
    span_start: int = -1
    span_end: int = -1
    cited_source_ids: List[str] = field(default_factory=list)
    supporting_evidence_ids: List[str] = field(default_factory=list)
    # sha256 hashes of the neighboring PROTECTED sentences (must be
    # byte-identical after a localized patch)
    protected_neighbor_hashes: List[str] = field(default_factory=list)

    @property
    def verification_state(self) -> str:    # spec alias
        return self.status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "req_id": self.req_id,
            "section_id": self.section_id,
            "text": self.text,
            "status": self.status,
            "verification_state": self.status,
            "span": [self.span_start, self.span_end],
            "cited_source_ids": list(self.cited_source_ids),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "protected_neighbor_hashes":
                list(self.protected_neighbor_hashes),
        }


@dataclass
class ProgressRound:
    """MEASURED deltas of one research round (stall detection input).

    Stall is decided from these numbers — never from an LLM's own claim
    of progress. Task accounting distinguishes what was actually RUN
    from what was merely queued (deferred tasks consumed nothing).
    """
    round_index: int
    new_unique_evidence: int = 0
    new_supported_claims: int = 0
    resolved_conflicts: int = 0
    coverage_delta: float = 0.0
    queries_run: int = 0
    scheduled_tasks: List[str] = field(default_factory=list)
    executed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    deferred_tasks: List[str] = field(default_factory=list)
    # set by the StopController when this round makes the run a
    # candidate for termination (stall/budget)
    termination_candidate: bool = False
    notes: str = ""

    def is_stalled(self, min_coverage_delta: float = 0.005) -> bool:
        """No measurable progress in ANY dimension."""
        return (self.new_unique_evidence <= 0
                and self.new_supported_claims <= 0
                and self.resolved_conflicts <= 0
                and self.coverage_delta < min_coverage_delta)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_index": self.round_index,
            "new_unique_evidence": self.new_unique_evidence,
            "new_supported_claims": self.new_supported_claims,
            "resolved_conflicts": self.resolved_conflicts,
            "coverage_delta": round(self.coverage_delta, 4),
            "queries_run": self.queries_run,
            "scheduled_tasks": list(self.scheduled_tasks),
            "executed_tasks": list(self.executed_tasks),
            "failed_tasks": list(self.failed_tasks),
            "deferred_tasks": list(self.deferred_tasks),
            "termination_candidate": self.termination_candidate,
            "stalled": self.is_stalled(),
            "notes": self.notes,
        }
