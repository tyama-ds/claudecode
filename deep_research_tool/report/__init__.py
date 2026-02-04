"""
Report generation module for Deep Research Tool.
"""

from .generator import ReportGenerator, ReportFormat
from .length_controller import (
    ContentLengthController,
    LengthTarget,
    LengthInfo,
    estimate_page_count,
    get_length_summary,
)

__all__ = [
    "ReportGenerator",
    "ReportFormat",
    "ContentLengthController",
    "LengthTarget",
    "LengthInfo",
    "estimate_page_count",
    "get_length_summary",
]
