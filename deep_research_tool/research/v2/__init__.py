"""
Research V2 - Enhanced research orchestration.

Version 2.0 adds:
- Think Tool (strategic reflection during research iterations)
- Pre-research clarification flow
- Parallel section processing via asyncio

Usage:
    from deep_research_tool.research.v2 import (
        ResearcherV2,
        ResearchReflector,
        ResearchClarifier,
        AsyncResearchOrchestrator,
    )

    researcher = ResearcherV2(
        llm_client=llm,
        search_client=search,
        enable_think_tool=True,
        enable_parallel=True,
    )

    session = researcher.conduct_research("AI market analysis 2025")
"""

# Think Tool (strategic reflection)
from .reflector import (
    ResearchReflector,
    ReflectionResult,
    OverallReflection,
)

# Pre-research clarification
from .clarifier import (
    ResearchClarifier,
    ClarificationResult,
)

# Parallel section processing
from .async_orchestrator import (
    AsyncResearchOrchestrator,
    SectionGroup,
    ParallelResearchResult,
)

# Enhanced researcher
from .researcher import ResearcherV2

__all__ = [
    # Reflector
    "ResearchReflector",
    "ReflectionResult",
    "OverallReflection",
    # Clarifier
    "ResearchClarifier",
    "ClarificationResult",
    # Async orchestrator
    "AsyncResearchOrchestrator",
    "SectionGroup",
    "ParallelResearchResult",
    # Researcher
    "ResearcherV2",
]

__version__ = "2.0.0"
