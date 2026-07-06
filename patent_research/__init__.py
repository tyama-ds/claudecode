"""
Patent Research Tool - AI-powered patent analysis with multi-layer search.

Provides patent-centric research with auxiliary searches for:
- Technical papers (CiNii, J-STAGE, Google Scholar)
- Patent examination documents
- Business/market evidence
"""

from .config import (
    PatentResearchConfig,
    PatentSearchConfig,
    AuxiliarySearchConfig,
    PatentReportConfig,
    create_patent_config,
)
from .models.patent import (
    Patent,
    PatentClaim,
    IPCClassification,
    PatentFamily,
)
from .models.analysis import (
    ClaimChart,
    ClaimChartEntry,
    TechnologyLandscape,
    PriorArtRecord,
)


def __getattr__(name):
    """Lazy imports for modules with circular dependency potential."""
    if name == "PatentResearchTool":
        from .main import PatentResearchTool
        return PatentResearchTool
    if name == "run_patent_research":
        from .main import run_patent_research
        return run_patent_research
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PatentResearchTool",
    "run_patent_research",
    "PatentResearchConfig",
    "PatentSearchConfig",
    "AuxiliarySearchConfig",
    "PatentReportConfig",
    "create_patent_config",
    "Patent",
    "PatentClaim",
    "IPCClassification",
    "PatentFamily",
    "ClaimChart",
    "ClaimChartEntry",
    "TechnologyLandscape",
    "PriorArtRecord",
]
