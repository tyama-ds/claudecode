"""
Coverage Ledger — per-requirement coverage state with validated
transitions, plus the deterministic StopController.

The ledger is the single source of truth for WHICH requirements still
need research. Gap research is issued ONLY for requirements in a gap
state (open / conflicted); supported requirements are never re-searched
and never re-drafted.

All transitions are validated against an explicit matrix; an illegal
transition raises instead of silently corrupting coverage state.
"""

import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import (
    GAP_STATES,
    REQ_BUDGET_EXHAUSTED,
    REQ_CONFLICTED,
    REQ_NOT_APPLICABLE,
    REQ_OPEN,
    REQ_SUPPORTED,
    REQ_UNAVAILABLE,
    REQUIREMENT_STATES,
    TERMINAL_STATES,
    ProgressRound,
    RequirementLeaf,
)

# Allowed transitions. Notes:
# - supported -> open/conflicted: later verification may invalidate a
#   requirement (new contradicting evidence) — coverage must reopen.
# - unavailable_after_search -> open: new evidence found incidentally
#   (another requirement's search) legitimately reopens it.
# - not_applicable is terminal (only an explicit caller decision made it,
#   and only an explicit reopen() call may undo it).
# - budget_exhausted is terminal for the run.
_ALLOWED: Dict[str, Tuple[str, ...]] = {
    REQ_OPEN: (REQ_SUPPORTED, REQ_CONFLICTED, REQ_UNAVAILABLE,
               REQ_NOT_APPLICABLE, REQ_BUDGET_EXHAUSTED),
    REQ_CONFLICTED: (REQ_SUPPORTED, REQ_OPEN, REQ_UNAVAILABLE,
                     REQ_BUDGET_EXHAUSTED),
    REQ_SUPPORTED: (REQ_OPEN, REQ_CONFLICTED),
    REQ_UNAVAILABLE: (REQ_OPEN, REQ_SUPPORTED, REQ_BUDGET_EXHAUSTED),
    REQ_NOT_APPLICABLE: (),
    REQ_BUDGET_EXHAUSTED: (),
}


class CoverageLedger:
    """Thread-safe per-requirement coverage bookkeeping."""

    def __init__(self, max_search_attempts: int = 2):
        self._lock = threading.RLock()
        self._reqs: Dict[str, RequirementLeaf] = {}
        self.max_search_attempts = max(1, int(max_search_attempts))
        self.rounds: List[ProgressRound] = []

    # -- requirement management -------------------------------------------

    def add(self, req: RequirementLeaf) -> RequirementLeaf:
        with self._lock:
            if req.req_id in self._reqs:
                return self._reqs[req.req_id]
            if req.status not in REQUIREMENT_STATES:
                raise ValueError(f"unknown requirement state: {req.status}")
            req.history.append(req.status)
            self._reqs[req.req_id] = req
            return req

    def get(self, req_id: str) -> Optional[RequirementLeaf]:
        with self._lock:
            return self._reqs.get(req_id)

    def requirements(self) -> List[RequirementLeaf]:
        with self._lock:
            return list(self._reqs.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._reqs)

    # -- transitions ---------------------------------------------------------

    def transition(self, req_id: str, new_state: str,
                   reason: str = "") -> RequirementLeaf:
        """Validated state transition; raises on an illegal move."""
        if new_state not in REQUIREMENT_STATES:
            raise ValueError(f"unknown requirement state: {new_state}")
        with self._lock:
            req = self._reqs.get(req_id)
            if req is None:
                raise KeyError(f"unknown requirement: {req_id}")
            if new_state == req.status:
                if reason:
                    req.status_reason = reason
                return req
            if new_state not in _ALLOWED[req.status]:
                raise ValueError(
                    f"illegal transition {req.status} -> {new_state} "
                    f"({req_id})")
            req.status = new_state
            req.status_reason = reason
            req.history.append(new_state)
            return req

    def reopen(self, req_id: str, reason: str = "") -> RequirementLeaf:
        """Explicit caller decision: reopen a not_applicable requirement."""
        with self._lock:
            req = self._reqs.get(req_id)
            if req is None:
                raise KeyError(f"unknown requirement: {req_id}")
            req.status = REQ_OPEN
            req.status_reason = reason or "explicitly reopened"
            req.history.append(REQ_OPEN)
            return req

    def record_search_attempt(self, req_id: str) -> RequirementLeaf:
        """Count one gap-search attempt; flip to unavailable_after_search
        when the attempt budget is exhausted and the gap remains."""
        with self._lock:
            req = self._reqs.get(req_id)
            if req is None:
                raise KeyError(f"unknown requirement: {req_id}")
            req.search_attempts += 1
            return req

    def close_exhausted(self, reason: str = "") -> List[str]:
        """Move gap requirements whose search attempts are used up to
        unavailable_after_search. Returns the affected req_ids."""
        moved = []
        with self._lock:
            for req in self._reqs.values():
                if req.status in GAP_STATES and \
                        req.search_attempts >= self.max_search_attempts:
                    self.transition(
                        req.req_id, REQ_UNAVAILABLE,
                        reason or (f"no usable evidence after "
                                   f"{req.search_attempts} searches"))
                    moved.append(req.req_id)
        return moved

    def close_budget_exhausted(self, reason: str = "") -> List[str]:
        """End of run/budget: every remaining gap requirement is closed
        as budget_exhausted (never silently left 'open')."""
        moved = []
        with self._lock:
            for req in self._reqs.values():
                if req.status in GAP_STATES:
                    self.transition(req.req_id, REQ_BUDGET_EXHAUSTED,
                                    reason or "research budget exhausted")
                    moved.append(req.req_id)
        return moved

    # -- queries / metrics -----------------------------------------------

    def gap_requirements(self) -> List[RequirementLeaf]:
        """Requirements that legitimately trigger research, ordered by
        priority (critical first) then id — deterministic."""
        order = {"critical": 0, "important": 1, "minor": 2}
        with self._lock:
            gaps = [r for r in self._reqs.values()
                    if r.status in GAP_STATES
                    and r.search_attempts < self.max_search_attempts]
        return sorted(gaps, key=lambda r: (order.get(r.priority, 1),
                                           r.req_id))

    def coverage(self) -> float:
        """supported + not_applicable over all requirements."""
        with self._lock:
            reqs = list(self._reqs.values())
        if not reqs:
            return 1.0
        done = sum(1 for r in reqs
                   if r.status in (REQ_SUPPORTED, REQ_NOT_APPLICABLE))
        return done / len(reqs)

    def counts(self) -> Dict[str, int]:
        with self._lock:
            counts = {state: 0 for state in REQUIREMENT_STATES}
            for r in self._reqs.values():
                counts[r.status] += 1
            return counts

    def all_terminal(self) -> bool:
        with self._lock:
            return all(r.status in TERMINAL_STATES
                       for r in self._reqs.values())

    def record_round(self, progress: ProgressRound) -> None:
        with self._lock:
            self.rounds.append(progress)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "requirements": [r.to_dict() for r in self._reqs.values()],
                "counts": self.counts(),
                "coverage": round(self.coverage(), 4),
                "rounds": [p.to_dict() for p in self.rounds],
                "max_search_attempts": self.max_search_attempts,
            }


class StopController:
    """Deterministic stopping decision for the adaptive research loop.

    Stops when ANY of:
    - every requirement is terminal (nothing left to research)
    - the round budget is spent
    - ``max_stall_rounds`` consecutive rounds show no MEASURED progress
      (new unique evidence, newly supported claims, resolved conflicts,
      coverage delta — all zero/below threshold)

    The LLM has no vote here: rounds are populated from counted
    artifacts (locker growth, verdict deltas), not model self-reports.
    """

    def __init__(self, max_rounds: int = 3, max_stall_rounds: int = 2,
                 min_coverage_delta: float = 0.005):
        self.max_rounds = max(0, int(max_rounds))
        self.max_stall_rounds = max(1, int(max_stall_rounds))
        self.min_coverage_delta = float(min_coverage_delta)

    def should_stop(self, ledger: CoverageLedger) -> Tuple[bool, str]:
        if ledger.all_terminal():
            return True, "all_requirements_terminal"
        rounds = ledger.rounds
        if len(rounds) >= self.max_rounds:
            return True, "round_budget_exhausted"
        stall = 0
        for p in reversed(rounds):
            if p.is_stalled(self.min_coverage_delta):
                stall += 1
            else:
                break
        if stall >= self.max_stall_rounds:
            return True, f"stalled_{stall}_rounds"
        return False, ""
