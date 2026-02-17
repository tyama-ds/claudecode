"""Patent research logic modules."""

from .patent_query_generator import PatentQueryGenerator
from .claim_analyzer import ClaimAnalyzer
from .auxiliary_trigger import AuxiliaryTrigger, TriggerResult


def __getattr__(name):
    """Lazy imports for modules with circular dependency potential."""
    if name == "PatentResearcher":
        from .patent_researcher import PatentResearcher
        return PatentResearcher
    if name == "PatentResearchSession":
        from .patent_researcher import PatentResearchSession
        return PatentResearchSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PatentResearcher",
    "PatentResearchSession",
    "PatentQueryGenerator",
    "ClaimAnalyzer",
    "AuxiliaryTrigger",
    "TriggerResult",
]
