"""
Reasoning chain definitions for DeepThink.

This module defines the data structures for representing reasoning steps,
chains, and consistency check results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class ReasoningType(str, Enum):
    """Types of reasoning steps."""
    FACT_EXTRACTION = "fact_extraction"
    INFERENCE = "inference"
    SYNTHESIS = "synthesis"
    CONCLUSION = "conclusion"
    VERIFICATION = "verification"


class ConsistencyMode(str, Enum):
    """Modes for handling consistency issues."""
    WARN = "warn"       # Log warning but continue
    REVISE = "revise"   # Attempt to revise problematic conclusions
    STRICT = "strict"   # Reject conclusions that fail consistency check


@dataclass
class ReasoningStep:
    """
    Represents a single step in the reasoning chain.

    Attributes:
        step_id: Unique identifier for this step
        step_type: Type of reasoning (fact, inference, etc.)
        premises: Input facts/statements for this step
        conclusion: Output conclusion from this step
        source_references: References to original sources
        confidence: Confidence score for this step (0-1)
        metrics: Quantitative metrics for this step
        timestamp: When this step was created
    """
    step_id: str
    step_type: ReasoningType
    premises: List[str]
    conclusion: str
    source_references: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "premises": self.premises,
            "conclusion": self.conclusion,
            "source_references": self.source_references,
            "confidence": self.confidence,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ReasoningChain:
    """
    Represents a complete reasoning chain from facts to conclusion.

    Attributes:
        chain_id: Unique identifier for this chain
        steps: Ordered list of reasoning steps
        initial_facts: Original facts extracted from sources
        final_conclusion: Final synthesized conclusion
        overall_confidence: Aggregate confidence score
        deviation_score: How much the conclusion deviates from facts
        is_consistent: Whether the chain passes consistency check
    """
    chain_id: str
    steps: List[ReasoningStep] = field(default_factory=list)
    initial_facts: List[str] = field(default_factory=list)
    final_conclusion: str = ""
    overall_confidence: float = 0.0
    deviation_score: float = 0.0
    is_consistent: bool = True

    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the chain."""
        self.steps.append(step)

    def get_all_premises(self) -> List[str]:
        """Get all premises used throughout the chain."""
        all_premises = []
        for step in self.steps:
            all_premises.extend(step.premises)
        return all_premises

    def get_all_conclusions(self) -> List[str]:
        """Get all conclusions from the chain."""
        return [step.conclusion for step in self.steps]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chain_id": self.chain_id,
            "steps": [step.to_dict() for step in self.steps],
            "initial_facts": self.initial_facts,
            "final_conclusion": self.final_conclusion,
            "overall_confidence": self.overall_confidence,
            "deviation_score": self.deviation_score,
            "is_consistent": self.is_consistent,
        }


@dataclass
class ConsistencyCheckResult:
    """
    Result of a consistency check between facts and conclusion.

    Attributes:
        is_consistent: Whether the conclusion is consistent with facts
        deviation_score: Quantitative measure of deviation (0-1)
        semantic_deviation: Semantic distance from original facts
        logical_gap: Gap in logical reasoning
        contradiction_score: Degree of contradiction detected
        problematic_areas: Specific areas with consistency issues
        suggestions: Suggestions for improvement
        confidence_score: Overall confidence in the conclusion
    """
    is_consistent: bool
    deviation_score: float
    semantic_deviation: float = 0.0
    logical_gap: float = 0.0
    contradiction_score: float = 0.0
    problematic_areas: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_consistent": self.is_consistent,
            "deviation_score": self.deviation_score,
            "semantic_deviation": self.semantic_deviation,
            "logical_gap": self.logical_gap,
            "contradiction_score": self.contradiction_score,
            "problematic_areas": self.problematic_areas,
            "suggestions": self.suggestions,
            "confidence_score": self.confidence_score,
        }


@dataclass
class DeepThinkResult:
    """
    Complete result of DeepThink processing.

    Attributes:
        original_content: Original input content
        processed_content: Content after DeepThink processing
        reasoning_chains: All reasoning chains used
        consistency_result: Final consistency check result
        metrics_summary: Summary of all quantitative metrics
        processing_time: Time taken for processing
        deep_think_level: The level used for processing
    """
    original_content: str
    processed_content: str
    reasoning_chains: List[ReasoningChain] = field(default_factory=list)
    consistency_result: Optional[ConsistencyCheckResult] = None
    metrics_summary: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    deep_think_level: float = 0.5

    @property
    def is_valid(self) -> bool:
        """Check if the result is valid (passes consistency check)."""
        if self.consistency_result is None:
            return True
        return self.consistency_result.is_consistent

    @property
    def overall_confidence(self) -> float:
        """Get overall confidence score."""
        if self.consistency_result:
            return self.consistency_result.confidence_score
        return self.metrics_summary.get("confidence_score", 0.5)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "original_content": self.original_content[:500] + "..." if len(self.original_content) > 500 else self.original_content,
            "processed_content": self.processed_content[:500] + "..." if len(self.processed_content) > 500 else self.processed_content,
            "reasoning_chains": [chain.to_dict() for chain in self.reasoning_chains],
            "consistency_result": self.consistency_result.to_dict() if self.consistency_result else None,
            "metrics_summary": self.metrics_summary,
            "processing_time": self.processing_time,
            "deep_think_level": self.deep_think_level,
            "is_valid": self.is_valid,
            "overall_confidence": self.overall_confidence,
        }
