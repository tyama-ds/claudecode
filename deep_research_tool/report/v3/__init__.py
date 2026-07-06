"""
Report Generator V3 - DOCX-native document generator.

Version 3.0:
- Direct DOCX construction via python-docx official API (no Markdown intermediate)
- Reuses V2 consistency features (glossary, context, two-phase generation)
- Figures, tables, and charts inserted via python-docx API
- Robust XML sanitization inherited from V2

Usage:
    from deep_research_tool.report.v3 import DocxReportGeneratorV3

    generator = DocxReportGeneratorV3(
        llm_client=llm,
        writing_style=WritingStyle.BUSINESS,
        target_audience=TargetAudience.BUSINESS,
    )

    result = generator.generate_report(
        research_topic="Research Topic",
        research_plan=plan,
        section_contents=section_contents,
    )

    report_path = generator.generate_and_save(
        result,
        output_dir=Path("./output"),
        filename="report",
        evidence_locker=evidence_locker,
    )
"""

from .docx_generator import DocxReportGeneratorV3

# Re-export V2 components used by V3
from ..v2.context import (
    ReportContext,
    WritingStyle,
    TargetAudience,
)
from ..v2.generator import (
    ReportFormatError,
    ChapterContent,
    GenerationResult,
)

__all__ = [
    "DocxReportGeneratorV3",
    "ReportContext",
    "WritingStyle",
    "TargetAudience",
    "ReportFormatError",
    "ChapterContent",
    "GenerationResult",
]

__version__ = "3.0.0"
