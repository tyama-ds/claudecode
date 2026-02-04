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
"""

__version__ = "0.1.0"
__author__ = "Deep Research Tool Team"

from .config import Config, ResearchConfig
from .main import DeepResearchTool

__all__ = [
    "Config",
    "ResearchConfig",
    "DeepResearchTool",
    "__version__",
]
