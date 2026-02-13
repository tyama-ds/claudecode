"""
Evidence Locker module for tracking and managing research sources.
"""

from .locker import (
    EvidenceLocker,
    Evidence,
    EvidenceType,
    QualityCategory,
    SourceType,
    QualityIndicators,
)
from .quality_evaluator import (
    QualityEvaluator,
    QualityEvaluation,
    categorize_by_quality,
    get_quality_summary,
)
from .manual_loader import (
    ManualEvidenceLoader,
    load_evidence_file,
)
from .content_filter import (
    ContentFilter,
    ContentFilterConfig,
    FilterResult,
    create_strict_filter,
    create_moderate_filter,
    create_minimal_filter,
)

__all__ = [
    "EvidenceLocker",
    "Evidence",
    "EvidenceType",
    "QualityCategory",
    "SourceType",
    "QualityIndicators",
    "QualityEvaluator",
    "QualityEvaluation",
    "categorize_by_quality",
    "get_quality_summary",
    "ManualEvidenceLoader",
    "load_evidence_file",
    "ContentFilter",
    "ContentFilterConfig",
    "FilterResult",
    "create_strict_filter",
    "create_moderate_filter",
    "create_minimal_filter",
]
