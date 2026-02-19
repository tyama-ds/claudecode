"""
Validation for Fermi estimation results.

Cross-checks estimates against evidence and performs sanity checks.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A single validation issue found."""
    severity: str = "warning"  # "info", "warning", "error"
    category: str = ""         # "order_of_magnitude", "cross_check", "sanity", "range"
    description: str = ""
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Complete validation result."""
    is_valid: bool = True
    overall_confidence: float = 0.5
    issues: List[ValidationIssue] = field(default_factory=list)
    cross_check_results: List[Dict[str, Any]] = field(default_factory=list)
    sanity_checks_passed: int = 0
    sanity_checks_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "overall_confidence": self.overall_confidence,
            "issues": [i.to_dict() for i in self.issues],
            "cross_check_results": self.cross_check_results,
            "sanity_checks_passed": self.sanity_checks_passed,
            "sanity_checks_total": self.sanity_checks_total,
        }


class Validator:
    """
    Validates Fermi estimation results.

    Performs:
    1. Sanity checks (positive values, low <= base <= high, etc.)
    2. Cross-checks against NumericalDataStore data
    3. LLM-based reasonableness check (optional)
    """

    VALIDATION_PROMPT_JA = """以下のフェルミ推定結果を検証してください。

## 推定対象
{target_metric}

## 推定結果
ベースケース: {base_value} {unit}
悲観シナリオ: {low_value} {unit}
楽観シナリオ: {high_value} {unit}

## 使用した前提
{assumptions_summary}

## 既知の参考データ
{reference_data}

## 検証項目
1. 推定値のオーダー（桁数）は妥当か？
2. 各前提は合理的か？
3. 既知のデータとの整合性はあるか？
4. 明らかな見落としや誤りはないか？

## 出力形式 (JSON)
{{
    "overall_assessment": "妥当/要注意/問題あり",
    "confidence": 0.0から1.0,
    "issues": [
        {{"severity": "info/warning/error", "category": "カテゴリ", "description": "説明", "suggestion": "提案"}}
    ],
    "cross_checks": [
        {{"reference": "参考データ", "expected_range": "期待範囲", "actual": "推定値", "consistent": true}}
    ]
}}

JSONのみを出力:"""

    VALIDATION_PROMPT_EN = """Validate the following Fermi estimation result.

## Target Metric
{target_metric}

## Estimation Result
Base case: {base_value} {unit}
Pessimistic: {low_value} {unit}
Optimistic: {high_value} {unit}

## Assumptions Used
{assumptions_summary}

## Known Reference Data
{reference_data}

## Validation Items
1. Is the order of magnitude reasonable?
2. Are the assumptions reasonable?
3. Is the result consistent with known data?
4. Are there obvious errors or omissions?

## Output (JSON only)
{{
    "overall_assessment": "reasonable/caution/problematic",
    "confidence": 0.0 to 1.0,
    "issues": [
        {{"severity": "info/warning/error", "category": "category", "description": "...", "suggestion": "..."}}
    ],
    "cross_checks": [
        {{"reference": "...", "expected_range": "...", "actual": "...", "consistent": true}}
    ]
}}

Output only JSON:"""

    def __init__(
        self,
        llm_client=None,
        data_store=None,
        evidence_locker=None,
        language: str = "ja",
    ):
        self.llm_client = llm_client
        self.data_store = data_store
        self.evidence_locker = evidence_locker
        self.language = language

    def validate(
        self,
        target_metric: str,
        base_value: float,
        low_value: float,
        high_value: float,
        unit: str,
        assumptions: List = None,
    ) -> ValidationResult:
        """Validate a Fermi estimation result."""
        all_issues = []
        cross_checks = []

        # Sanity checks
        sanity_issues = self._sanity_checks(base_value, low_value, high_value)
        all_issues.extend(sanity_issues)
        sanity_passed = len([i for i in sanity_issues if i.severity != "error"])
        sanity_total = 4  # number of sanity checks

        # Cross-check with evidence
        if self.data_store:
            cc_results = self._cross_check_with_evidence(target_metric, base_value, unit)
            cross_checks.extend(cc_results)
            for cc in cc_results:
                if not cc.get("consistent", True):
                    all_issues.append(ValidationIssue(
                        severity="warning",
                        category="cross_check",
                        description=f"Inconsistent with reference: {cc.get('reference', '')}",
                        suggestion=cc.get("suggestion", "Review the estimation"),
                    ))

        # LLM validation
        confidence = 0.5
        if self.llm_client:
            llm_issues, llm_confidence = self._llm_validation(
                target_metric, base_value, low_value, high_value,
                unit, assumptions or [],
            )
            all_issues.extend(llm_issues)
            confidence = llm_confidence

        has_errors = any(i.severity == "error" for i in all_issues)

        return ValidationResult(
            is_valid=not has_errors,
            overall_confidence=confidence,
            issues=all_issues,
            cross_check_results=cross_checks,
            sanity_checks_passed=sanity_total - len([i for i in sanity_issues if i.severity == "error"]),
            sanity_checks_total=sanity_total,
        )

    def _sanity_checks(
        self, base: float, low: float, high: float,
    ) -> List[ValidationIssue]:
        """Basic sanity checks."""
        issues = []

        # Check: low <= base <= high
        if low > base:
            issues.append(ValidationIssue(
                severity="error",
                category="sanity",
                description=f"Low estimate ({low}) > base estimate ({base})",
                suggestion="Low estimate should be less than or equal to base",
            ))

        if base > high:
            issues.append(ValidationIssue(
                severity="error",
                category="sanity",
                description=f"Base estimate ({base}) > high estimate ({high})",
                suggestion="High estimate should be greater than or equal to base",
            ))

        # Check: values are finite
        for label, val in [("base", base), ("low", low), ("high", high)]:
            if not isinstance(val, (int, float)) or val != val:  # NaN check
                issues.append(ValidationIssue(
                    severity="error",
                    category="sanity",
                    description=f"{label} value is not a valid number: {val}",
                ))

        # Check: range reasonableness (high/low ratio)
        if low > 0 and high > 0:
            ratio = high / low
            if ratio > 100:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="range",
                    description=f"Very wide estimate range: high/low ratio = {ratio:.1f}x",
                    suggestion="Consider narrowing assumptions for more precise estimate",
                ))

        return issues

    def _cross_check_with_evidence(
        self, target_metric: str, value: float, unit: str,
    ) -> List[Dict[str, Any]]:
        """Cross-check with data in NumericalDataStore."""
        if not self.data_store:
            return []

        results = []
        metric_lower = target_metric.lower()

        for dp in self.data_store.get_high_confidence(threshold=0.5):
            # Simple keyword matching
            dp_text = f"{dp.metric_name} {dp.subject}".lower()
            overlap = sum(
                1 for word in metric_lower.split()
                if len(word) > 1 and word in dp_text
            )
            if overlap < 2:
                continue

            # Compare order of magnitude
            ref_value = dp.normalized_value or dp.value
            if ref_value and ref_value > 0 and value > 0:
                import math
                magnitude_diff = abs(math.log10(value) - math.log10(ref_value))
                consistent = magnitude_diff < 2  # Within 2 orders of magnitude

                results.append({
                    "reference": f"{dp.metric_name}: {dp.raw_text} ({dp.source_title})",
                    "reference_value": ref_value,
                    "estimated_value": value,
                    "magnitude_diff": magnitude_diff,
                    "consistent": consistent,
                    "suggestion": "" if consistent else
                        f"Estimate differs by {10**magnitude_diff:.0f}x from reference",
                })

        return results

    def _llm_validation(
        self,
        target_metric: str,
        base: float, low: float, high: float,
        unit: str,
        assumptions: List,
    ) -> tuple:
        """Use LLM for reasonableness check. Returns (issues, confidence)."""
        # Format assumptions
        assumptions_summary = "\n".join(
            f"- {a.parameter_name}: {a.value} {a.unit} ({a.source.value})"
            for a in assumptions
        ) if assumptions else "なし" if self.language == "ja" else "None"

        # Get reference data
        reference_data = "なし" if self.language == "ja" else "None"
        if self.data_store:
            refs = []
            for dp in self.data_store.get_high_confidence(threshold=0.5)[:10]:
                refs.append(f"- {dp.metric_name}: {dp.raw_text}")
            if refs:
                reference_data = "\n".join(refs)

        template = (
            self.VALIDATION_PROMPT_JA
            if self.language == "ja"
            else self.VALIDATION_PROMPT_EN
        )
        prompt = template.format(
            target_metric=target_metric,
            base_value=base,
            low_value=low,
            high_value=high,
            unit=unit,
            assumptions_summary=assumptions_summary,
            reference_data=reference_data,
        )

        try:
            response = self.llm_client.generate(prompt)
            if not response or not response.content:
                return [], 0.5

            data = self._parse_json(response.content)
            if not data:
                return [], 0.5

            confidence = float(data.get("confidence", 0.5))
            issues = []
            for issue_data in data.get("issues", []):
                issues.append(ValidationIssue(
                    severity=issue_data.get("severity", "info"),
                    category=issue_data.get("category", "llm_check"),
                    description=issue_data.get("description", ""),
                    suggestion=issue_data.get("suggestion", ""),
                ))

            return issues, confidence

        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return [], 0.5

    def _parse_json(self, content: str) -> Optional[Dict]:
        """Parse JSON from LLM response."""
        content = content.strip()
        if "```" in content:
            parts = content.split("```")
            for part in parts[1:]:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
                if cleaned.startswith("{"):
                    content = cleaned
                    break
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(content[start:end])
                except json.JSONDecodeError:
                    return None
        return None
