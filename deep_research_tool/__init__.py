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

__version__ = "0.1.0"
__author__ = "Deep Research Tool Team"

from .config import Config, ResearchConfig, create_config, ProxyConfig
from .main import DeepResearchTool, run_research


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
    "launch_gui",
    "__version__",
]
