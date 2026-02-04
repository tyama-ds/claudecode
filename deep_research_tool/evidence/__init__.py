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
]
