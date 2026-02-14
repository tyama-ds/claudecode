"""
Information Gathering Agent - Automated information collection and synthesis.

A focused tool for gathering information from web sources,
extracting relevant content, and producing structured evidence
with per-section summaries and executive summaries.

Derived from deep_research_tool, focused purely on information
gathering without report generation, verification, or DeepThink.
"""

from .config import Config, create_config
from .main import (
    InfoGatheringAgent,
    GatheringResult,
    run_gathering,
)

__all__ = [
    # Main interface
    "InfoGatheringAgent",
    "GatheringResult",
    "run_gathering",
    # Configuration
    "Config",
    "create_config",
]
