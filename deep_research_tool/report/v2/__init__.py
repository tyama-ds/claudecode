"""
Report Generator V2 - Enhanced report generation with consistency features.

Version 2.0 adds:
- Global context maintenance across chapters
- Terminology consistency via glossary
- Previous chapter summaries for continuity
- Fact tracking to prevent contradictions
- Post-generation consistency checking
- Two-phase generation (draft + refinement)

Usage:
    from deep_research_tool.report.v2 import (
        ReportGeneratorV2,
        ReportContext,
        WritingStyle,
        TargetAudience,
    )

    generator = ReportGeneratorV2(
        llm_client=llm,
        writing_style=WritingStyle.BUSINESS,
        target_audience=TargetAudience.BUSINESS,
    )

    result = generator.generate_report(
        research_topic="Research Topic",
        research_plan=plan,
        section_contents=section_contents,
    )

    final_doc = generator.generate_final_document(result)
"""

# Context and style
from .context import (
    ReportContext,
    WritingStyle,
    TargetAudience,
    GlossaryEntry,
    EstablishedFact,
    ChapterSummary,
    CrossReference,
)

# Consistency checking
from .consistency import (
    ConsistencyChecker,
    ConsistencyReport,
    ConsistencyIssue,
    IssueType,
)

# Glossary management
from .glossary import (
    GlossaryManager,
    TermCandidate,
)

# Report generation
from .generator import (
    ReportGeneratorV2,
    ReportFormatError,
    ChapterContent,
    GenerationResult,
)

__all__ = [
    # Context
    "ReportContext",
    "WritingStyle",
    "TargetAudience",
    "GlossaryEntry",
    "EstablishedFact",
    "ChapterSummary",
    "CrossReference",
    # Consistency
    "ConsistencyChecker",
    "ConsistencyReport",
    "ConsistencyIssue",
    "IssueType",
    # Glossary
    "GlossaryManager",
    "TermCandidate",
    # Generator
    "ReportGeneratorV2",
    "ReportFormatError",
    "ChapterContent",
    "GenerationResult",
]

__version__ = "2.0.0"
