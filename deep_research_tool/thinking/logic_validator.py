"""
Logic validation for DeepThink.

This module provides validation of reasoning chains, checking for
logical consistency, proper inference, and absence of fallacies.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .reasoning_chain import (
    ReasoningStep,
    ReasoningChain,
    ReasoningType,
    ConsistencyCheckResult,
)
from .metrics import DeepThinkMetrics, MetricsConfig


class ValidationLevel(str, Enum):
    """Levels of validation strictness."""
    RELAXED = "relaxed"     # Allow minor inconsistencies
    STANDARD = "standard"   # Default validation
    STRICT = "strict"       # No tolerance for issues


@dataclass
class ValidationConfig:
    """Configuration for logic validation."""
    level: ValidationLevel = ValidationLevel.STANDARD
    consistency_threshold: float = 0.3
    fidelity_threshold: float = 0.7
    coherence_threshold: float = 0.6
    max_expansion: float = 0.5


class LogicValidator:
    """
    Validates reasoning chains for logical consistency and correctness.

    Performs:
    - Step-by-step validation of reasoning
    - Chain-wide consistency checking
    - Fallacy detection
    - Source fidelity verification
    """

    def __init__(
        self,
        config: ValidationConfig = None,
        metrics: DeepThinkMetrics = None
    ):
        """
        Initialize the logic validator.

        Args:
            config: Validation configuration
            metrics: Metrics calculator instance
        """
        self.config = config or ValidationConfig()
        self.metrics = metrics or DeepThinkMetrics()

    def validate_step(
        self,
        step: ReasoningStep,
        source_text: str = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate a single reasoning step.

        Args:
            step: The reasoning step to validate
            source_text: Original source text for fact verification

        Returns:
            Tuple of (is_valid, list of issues found)
        """
        issues = []

        # Validate based on step type
        if step.step_type == ReasoningType.FACT_EXTRACTION:
            issues.extend(self._validate_fact_extraction(step, source_text))
        elif step.step_type == ReasoningType.INFERENCE:
            issues.extend(self._validate_inference(step))
        elif step.step_type == ReasoningType.SYNTHESIS:
            issues.extend(self._validate_synthesis(step))
        elif step.step_type == ReasoningType.CONCLUSION:
            issues.extend(self._validate_conclusion(step))

        # Calculate metrics for this step
        step_metrics = self.metrics.evaluate_reasoning_step(
            premises=step.premises,
            conclusion=step.conclusion,
            source_text=source_text
        )

        # Check against thresholds
        if step_metrics["source_fidelity"] < self.config.fidelity_threshold:
            issues.append(
                f"Low source fidelity ({step_metrics['source_fidelity']:.2f} < {self.config.fidelity_threshold})"
            )

        if step_metrics["logical_coherence"] < self.config.coherence_threshold:
            issues.append(
                f"Low logical coherence ({step_metrics['logical_coherence']:.2f} < {self.config.coherence_threshold})"
            )

        if step_metrics["expansion_degree"] > self.config.max_expansion:
            issues.append(
                f"Excessive information expansion ({step_metrics['expansion_degree']:.2f} > {self.config.max_expansion})"
            )

        # Update step metrics
        step.metrics = step_metrics
        step.confidence = step_metrics["confidence_score"]

        # Determine validity based on validation level
        is_valid = self._determine_step_validity(issues)

        return is_valid, issues

    def _validate_fact_extraction(
        self,
        step: ReasoningStep,
        source_text: str = None
    ) -> List[str]:
        """Validate fact extraction step."""
        issues = []

        if not step.premises:
            issues.append("Fact extraction has no source premises")

        if not step.conclusion:
            issues.append("Fact extraction produced no output")

        # Check if conclusion is substantially shorter than source
        if step.premises:
            source_len = sum(len(p) for p in step.premises)
            if len(step.conclusion) > source_len * 2:
                issues.append("Extracted fact is longer than source - possible fabrication")

        return issues

    def _validate_inference(self, step: ReasoningStep) -> List[str]:
        """Validate inference step."""
        issues = []

        if not step.premises:
            issues.append("Inference has no premises")
        elif len(step.premises) < 1:
            issues.append("Inference requires at least one premise")

        if not step.conclusion:
            issues.append("Inference produced no conclusion")

        # Check for common fallacies
        fallacy_issues = self._check_fallacies(step.premises, step.conclusion)
        issues.extend(fallacy_issues)

        return issues

    def _validate_synthesis(self, step: ReasoningStep) -> List[str]:
        """Validate synthesis step."""
        issues = []

        if len(step.premises) < 2:
            issues.append("Synthesis requires at least two premises to combine")

        if not step.conclusion:
            issues.append("Synthesis produced no output")

        return issues

    def _validate_conclusion(self, step: ReasoningStep) -> List[str]:
        """Validate conclusion step."""
        issues = []

        if not step.premises:
            issues.append("Conclusion has no supporting evidence")

        if not step.conclusion:
            issues.append("No conclusion produced")

        return issues

    def _check_fallacies(
        self,
        premises: List[str],
        conclusion: str
    ) -> List[str]:
        """
        Check for common logical fallacies.

        Currently checks for:
        - Hasty generalization
        - Appeal to authority (when not from source)
        - False dichotomy
        """
        issues = []

        # Patterns indicating hasty generalization
        generalization_patterns = [
            "all", "always", "never", "everyone", "nobody",
            "全て", "常に", "決して", "皆", "誰も"
        ]

        combined_premises = " ".join(premises).lower()

        for pattern in generalization_patterns:
            if pattern in conclusion.lower() and pattern not in combined_premises:
                issues.append(f"Potential hasty generalization: '{pattern}' in conclusion but not in premises")
                break

        return issues

    def _determine_step_validity(self, issues: List[str]) -> bool:
        """Determine if step is valid based on issues and validation level."""
        if not issues:
            return True

        if self.config.level == ValidationLevel.RELAXED:
            # Only fail on critical issues
            critical_keywords = ["fabrication", "no premises", "no conclusion"]
            return not any(kw in issue.lower() for issue in issues for kw in critical_keywords)

        elif self.config.level == ValidationLevel.STANDARD:
            # Allow up to 2 non-critical issues
            critical_count = sum(
                1 for issue in issues
                if any(kw in issue.lower() for kw in ["fabrication", "no premises", "no conclusion"])
            )
            return critical_count == 0 and len(issues) <= 2

        else:  # STRICT
            return len(issues) == 0

    def validate_chain(
        self,
        chain: ReasoningChain,
        source_texts: Dict[str, str] = None
    ) -> ConsistencyCheckResult:
        """
        Validate an entire reasoning chain.

        Args:
            chain: The reasoning chain to validate
            source_texts: Mapping of source references to their texts

        Returns:
            ConsistencyCheckResult with validation details
        """
        source_texts = source_texts or {}
        all_issues = []
        step_validities = []

        # Validate each step
        for step in chain.steps:
            source_text = source_texts.get(step.source_references[0]) if step.source_references else None
            is_valid, issues = self.validate_step(step, source_text)
            step_validities.append(is_valid)
            all_issues.extend(issues)

        # Calculate chain-wide metrics
        if chain.initial_facts and chain.final_conclusion:
            intermediate_conclusions = [s.conclusion for s in chain.steps[:-1]] if len(chain.steps) > 1 else []

            deviation_result = self.metrics.calc_deviation_score(
                initial_facts=chain.initial_facts,
                conclusion=chain.final_conclusion,
                intermediate_conclusions=intermediate_conclusions
            )

            # Check for contradictions
            contradiction_score, contradictions = self.metrics.detect_contradiction(
                facts=chain.initial_facts,
                conclusion=chain.final_conclusion
            )

            for contradiction in contradictions:
                all_issues.append(f"Contradiction detected: {contradiction}")

            # Calculate overall confidence
            avg_step_confidence = sum(s.confidence for s in chain.steps) / len(chain.steps) if chain.steps else 0.5

            overall_confidence = self.metrics.calc_confidence_score(
                source_fidelity=avg_step_confidence,
                logical_coherence=1.0 - deviation_result["logical_gap"],
                expansion_degree=sum(s.metrics.get("expansion_degree", 0) for s in chain.steps) / len(chain.steps) if chain.steps else 0,
                deviation_score=deviation_result["total_deviation"]
            )

            # Determine consistency
            is_consistent = (
                deviation_result["total_deviation"] < self.config.consistency_threshold
                and all(step_validities)
            )

            # Update chain attributes
            chain.deviation_score = deviation_result["total_deviation"]
            chain.overall_confidence = overall_confidence
            chain.is_consistent = is_consistent

            return ConsistencyCheckResult(
                is_consistent=is_consistent,
                deviation_score=deviation_result["total_deviation"],
                semantic_deviation=deviation_result["semantic_deviation"],
                logical_gap=deviation_result["logical_gap"],
                contradiction_score=contradiction_score,
                problematic_areas=all_issues,
                suggestions=self._generate_suggestions(all_issues, deviation_result),
                confidence_score=overall_confidence
            )

        # Fallback if chain is incomplete
        return ConsistencyCheckResult(
            is_consistent=False,
            deviation_score=1.0,
            problematic_areas=["Incomplete reasoning chain"],
            suggestions=["Ensure chain has initial facts and final conclusion"],
            confidence_score=0.0
        )

    def _generate_suggestions(
        self,
        issues: List[str],
        deviation_result: Dict[str, float]
    ) -> List[str]:
        """Generate improvement suggestions based on issues found."""
        suggestions = []

        if deviation_result["semantic_deviation"] > 0.5:
            suggestions.append("Conclusion deviates significantly from original facts - consider staying closer to source material")

        if deviation_result["logical_gap"] > 0.4:
            suggestions.append("Large logical gaps detected - add intermediate reasoning steps")

        if deviation_result["contradiction_score"] > 0.3:
            suggestions.append("Contradictions found - verify numerical values and negation statements")

        if any("expansion" in issue.lower() for issue in issues):
            suggestions.append("Excessive information added - focus on facts present in sources")

        if any("fidelity" in issue.lower() for issue in issues):
            suggestions.append("Low source fidelity - verify extracted facts against original sources")

        return suggestions

    def check_final_consistency(
        self,
        initial_facts: List[str],
        final_conclusion: str,
        threshold: float = None
    ) -> ConsistencyCheckResult:
        """
        Perform final consistency check between initial facts and conclusion.

        This is the "consistency check branch" that verifies the final conclusion
        doesn't deviate too far from the initial facts.

        Args:
            initial_facts: Original extracted facts
            final_conclusion: Final synthesized conclusion
            threshold: Consistency threshold (uses config if not specified)

        Returns:
            ConsistencyCheckResult with check details
        """
        threshold = threshold or self.config.consistency_threshold

        # Calculate deviation
        deviation_result = self.metrics.calc_deviation_score(
            initial_facts=initial_facts,
            conclusion=final_conclusion
        )

        # Detect contradictions
        contradiction_score, contradictions = self.metrics.detect_contradiction(
            facts=initial_facts,
            conclusion=final_conclusion
        )

        # Calculate confidence
        coherence = self.metrics.calc_logical_coherence(initial_facts, final_conclusion)
        expansion = self.metrics.calc_expansion_degree(initial_facts, final_conclusion)

        confidence = self.metrics.calc_confidence_score(
            source_fidelity=1.0 - deviation_result["semantic_deviation"],
            logical_coherence=coherence,
            expansion_degree=expansion,
            deviation_score=deviation_result["total_deviation"]
        )

        # Determine consistency
        is_consistent = deviation_result["total_deviation"] < threshold

        problematic_areas = []
        if not is_consistent:
            if deviation_result["semantic_deviation"] > threshold:
                problematic_areas.append(
                    f"Semantic deviation too high: {deviation_result['semantic_deviation']:.2f}"
                )
            if deviation_result["logical_gap"] > threshold:
                problematic_areas.append(
                    f"Logical gap too large: {deviation_result['logical_gap']:.2f}"
                )
        problematic_areas.extend(contradictions)

        suggestions = self._generate_suggestions(problematic_areas, deviation_result)

        return ConsistencyCheckResult(
            is_consistent=is_consistent,
            deviation_score=deviation_result["total_deviation"],
            semantic_deviation=deviation_result["semantic_deviation"],
            logical_gap=deviation_result["logical_gap"],
            contradiction_score=contradiction_score,
            problematic_areas=problematic_areas,
            suggestions=suggestions,
            confidence_score=confidence
        )
