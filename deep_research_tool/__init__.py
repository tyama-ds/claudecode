"""
Deep Research Tool
==================

A comprehensive research tool that leverages OpenAI and Anthropic APIs
for automated web research, information extraction, and report generation.

Features:
- Multi-LLM support (OpenAI, Anthropic)
- Web search via DuckDuckGo or Selenium browser
- Automated research loop with configurable iterations
- Evidence tracking and citation management
- Hallucination verification
- Report generation (docx, pdf, markdown)
- DeepThink reasoning enhancement
- Manual search mode with CSV/XLSX evidence files
- GUI for easy configuration
"""

__version__ = "0.1.0"
__author__ = "Deep Research Tool Team"

from .config import Config, ResearchConfig, create_config, ProxyConfig
from .main import (
    DeepResearchTool,
    run_research,
    run_manual_research,
    diagnose_session,
    ManualTableOfContents,
)
from .api.base import get_token_stats, reset_token_stats, TokenUsageStats
from .evidence.manual_loader import ManualEvidenceLoader, load_evidence_file
from .research.manual_researcher import ManualResearcher


def launch_gui():
    """Launch the graphical user interface."""
    from .gui import main as gui_main
    gui_main()


__all__ = [
    "Config",
    "ResearchConfig",
    "ProxyConfig",
    "create_config",
    "DeepResearchTool",
    "run_research",
    "run_manual_research",
    "diagnose_session",
    "ManualTableOfContents",
    "ManualEvidenceLoader",
    "ManualResearcher",
    "load_evidence_file",
    "launch_gui",
    "get_token_stats",
    "reset_token_stats",
    "TokenUsageStats",
    "__version__",
]
