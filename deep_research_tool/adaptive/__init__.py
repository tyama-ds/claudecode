"""
Adaptive Deep Research: requirement/claim/evidence-granular gap control.

This package upgrades the document-level "need_more_research" boolean
into typed, per-requirement state:

- models:  RequirementLeaf / ResearchTask / AtomicClaim / ProgressRound
           (EvidenceChunk is reused from verification.claim_verifier)
- ledger:  CoverageLedger — per-requirement coverage states with a
           validated transition matrix, plus StopController (stall
           detection from measured deltas, never LLM self-report)
- dag:     TaskDAG — dependency-aware scheduler reusing the app-wide
           parallel_max_workers limit
- intent:  deterministic query-intent classification, intent-templated
           query building, and INTERNAL Reciprocal Rank Fusion over the
           existing search clients (no external search/rerank APIs)

Design contract (mirrors the pipeline specification):
- The LLM produces structured output only (gap reasons, query
  candidates, verdicts); loops, budgets, worker limits and every state
  transition are deterministic Python.
- Nothing here talks to any network service directly: searching stays
  on the existing DuckDuckGo/Selenium/requests clients.
"""

from .models import (            # noqa: F401
    REQ_BUDGET_EXHAUSTED,
    REQ_CONFLICTED,
    REQ_NOT_APPLICABLE,
    REQ_OPEN,
    REQ_SUPPORTED,
    REQ_UNAVAILABLE,
    REQUIREMENT_STATES,
    AtomicClaim,
    ProgressRound,
    RequirementLeaf,
    ResearchTask,
)
from .ledger import CoverageLedger, StopController      # noqa: F401
from .dag import TaskDAG                                # noqa: F401
from .intent import (            # noqa: F401
    QUERY_INTENTS,
    build_intent_queries,
    classify_intent,
    rrf_merge,
)

# EvidenceChunk lives in the verifier (it is the retrieval unit of the
# whole verification stack); re-export it as part of the typed model set
from ..verification.claim_verifier import EvidenceChunk  # noqa: F401
