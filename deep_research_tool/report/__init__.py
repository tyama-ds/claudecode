"""
Report generation module for Deep Research Tool.
"""

from .generator import ReportGenerator, ReportFormat
from .length_controller import (
    ContentLengthController,
    LengthTarget,
    LengthInfo,
    ExpansionRequirement,
    estimate_page_count,
    get_length_summary,
)
from .figure_table_generator import (
    FigureTableGenerator,
    FigureTableCollection,
    Figure,
    TableData,
    FigureType,
    ChartType,
    add_figures_to_report,
)

__all__ = [
    "ReportGenerator",
    "ReportFormat",
    "ContentLengthController",
    "LengthTarget",
    "LengthInfo",
    "ExpansionRequirement",
    "estimate_page_count",
    "get_length_summary",
    "FigureTableGenerator",
    "FigureTableCollection",
    "Figure",
    "TableData",
    "FigureType",
    "ChartType",
    "add_figures_to_report",
]
