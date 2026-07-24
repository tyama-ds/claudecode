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
    status: str = REQ_OPEN
    status_reason: str = ""
    claim_ids: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    search_attempts: int = 0
    history: List[str] = field(default_factory=list)   # state trail

    def to_dict(self) -> Dict[str, Any]:
        return {
            "req_id": self.req_id,
            "text": self.text,
            "section_id": self.section_id,
            "intent": self.intent,
            "priority": self.priority,
            "status": self.status,
            "status_reason": self.status_reason,
            "claim_ids": list(self.claim_ids),
            "evidence_ids": list(self.evidence_ids),
            "search_attempts": self.search_attempts,
            "history": list(self.history),
        }


@dataclass
class ResearchTask:
    """One schedulable unit of gap research (a query for a requirement)."""
    task_id: str
    req_id: str
    query: str
    intent: str = "background"
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"             # pending / running / done / failed
    result_count: int = 0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "req_id": self.req_id,
            "query": self.query,
            "intent": self.intent,
            "depends_on": list(self.depends_on),
            "status": self.status,
            "result_count": self.result_count,
            "error": self.error,
        }


@dataclass
class AtomicClaim:
    """A verified factual claim linked back to its requirement."""
    claim_id: str
    req_id: str
    section_id: str
    text: str
    status: str = "uncertain"    # supported/unsupported/contradicted/uncertain
    supporting_evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "req_id": self.req_id,
            "section_id": self.section_id,
            "text": self.text,
            "status": self.status,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
        }


@dataclass
class ProgressRound:
    """MEASURED deltas of one research round (stall detection input).

    Stall is decided from these numbers — never from an LLM's own claim
    of progress.
    """
    round_index: int
    new_unique_evidence: int = 0
    new_supported_claims: int = 0
    resolved_conflicts: int = 0
    coverage_delta: float = 0.0
    queries_run: int = 0
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
            "stalled": self.is_stalled(),
            "notes": self.notes,
        }
