"""Runner API v1 — the engine contract shared with deep-research-orchestrator.

``models`` and ``engine`` are kept identical to the orchestrator's
``runners/common/runner_core`` (apart from the wider search provider
list and Python 3.10 compatibility) so an engine written for one side
runs on the other and a future HTTP transport needs no re-mapping.
"""

from .engine import CancelledByUser, Engine, RunContext
from .models import (
    TERMINAL_STATES,
    ClaimRecord,
    EngineCapabilities,
    EvidenceRecord,
    LlmRunConfig,
    RunEvent,
    RunInput,
    RunMetrics,
    RunRequest,
    RunResult,
    RunState,
    RunStatusResponse,
    SearchRunConfig,
    SourceRecord,
)

__all__ = [
    "CancelledByUser", "Engine", "RunContext", "TERMINAL_STATES",
    "ClaimRecord", "EngineCapabilities", "EvidenceRecord", "LlmRunConfig",
    "RunEvent", "RunInput", "RunMetrics", "RunRequest", "RunResult",
    "RunState", "RunStatusResponse", "SearchRunConfig", "SourceRecord",
]
