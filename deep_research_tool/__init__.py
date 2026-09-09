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

__version__ = "1.4.0"
__author__ = "Deep Research Tool Team"

# Harden console output as early as possible: on Japanese Windows the console
# defaults to cp932, which cannot encode characters like U+2011 (non-breaking
# hyphen) or U+2013 (en dash) that appear in web page titles and LLM output, so
# a bare print() would raise UnicodeEncodeError. Doing this at package import
# guarantees it runs before any submodule prints, on every entry path (CLI,
# Web UI, notebook, direct import).
from .utils.helpers import ensure_utf8_output as _ensure_utf8_output
_ensure_utf8_output()

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


def launch_gui(host: str = "127.0.0.1", port: int = 8765,
               output_dir: str = "./output", open_browser: bool = True):
    """Launch the graphical user interface.

    The GUI is the browser-based Web UI (HTML/JS served by the local
    stdlib HTTP server) — no Tkinter is required. Blocks until Ctrl+C.
    """
    from .webui.server import run_server
    run_server(host=host, port=port, output_dir=output_dir,
               open_browser=open_browser)


def launch_fermi_gui(host: str = "127.0.0.1", port: int = 8765,
                     output_dir: str = "./output", open_browser: bool = True):
    """Launch the Fermi estimation GUI (the Web UI opened on its
    フェルミ推定 panel). No Tkinter is required."""
    from .webui.server import run_server
    run_server(host=host, port=port, output_dir=output_dir,
               open_browser=open_browser, fragment="fermi")


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
    "launch_fermi_gui",
    "get_token_stats",
    "reset_token_stats",
    "TokenUsageStats",
    "__version__",
]
