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
- GUI for easy configuration
"""

__version__ = "0.2.0"
__author__ = "Deep Research Tool Team"

from .config import Config, ResearchConfig, create_config, ProxyConfig
from .main import DeepResearchTool, run_research, diagnose_session
from .api.base import get_token_stats, reset_token_stats, TokenUsageStats


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
    "diagnose_session",
    "launch_gui",
    "get_token_stats",
    "reset_token_stats",
    "TokenUsageStats",
    "__version__",
]
