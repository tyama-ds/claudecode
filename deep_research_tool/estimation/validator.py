"""
Validation for Fermi estimation results.

Cross-checks estimates against evidence and performs sanity checks.
Includes a 4-layer domain prior system that establishes expected order-of-magnitude
ranges before validating estimates.
"""

import json
import logging
import math
import re
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
    domain_prior: Optional["DomainPrior"] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "overall_confidence": self.overall_confidence,
            "issues": [i.to_dict() for i in self.issues],
            "cross_check_results": self.cross_check_results,
            "sanity_checks_passed": self.sanity_checks_passed,
            "sanity_checks_total": self.sanity_checks_total,
            "domain_prior": self.domain_prior.to_dict() if self.domain_prior else None,
        }


@dataclass
class DomainPrior:
    """Expected order-of-magnitude range for a target metric."""
    expected_order_low: int = 0
    expected_order_high: int = 20
    reference_points: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    source: str = ""  # "evidence", "static_rule", "llm", "unit_fallback"
    evidence_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_order_low": self.expected_order_low,
            "expected_order_high": self.expected_order_high,
            "reference_points": self.reference_points,
            "reasoning": self.reasoning,
            "source": self.source,
            "evidence_count": self.evidence_count,
        }

    @property
    def margin(self) -> float:
        """Tolerance margin (in orders of magnitude) based on source quality."""
        if self.source == "evidence" and self.evidence_count >= 3:
            return 0.5
        if self.source == "evidence":
            return 1.0
        if self.source == "static_rule":
            return 1.0
        if self.source == "llm":
            return 1.5
        return 3.0


class DomainPriorProvider:
    """
    4-layer domain prior system for establishing expected order-of-magnitude ranges.

    Layer 0: Collected evidence (data_store + evidence_locker) — highest priority
    Layer 1: Static rule table
    Layer 2: LLM query
    Layer 3: Unit-based fallback
    """

    # Layer 1: Static rules mapping keyword patterns to expected order ranges
    STATIC_RULES = [
        # (keywords, unit_patterns, order_low, order_high, description)
        (["日本", "人口"], None, 7, 9, "Japan population"),
        (["世界", "人口"], None, 9, 11, "World population"),
        (["日本", "gdp"], ["円", "yen"], 14, 15, "Japan GDP in JPY"),
        (["日本", "gdp"], ["ドル", "usd", "$"], 12, 13, "Japan GDP in USD"),
        (["世界", "gdp"], ["ドル", "usd", "$"], 13, 14, "World GDP in USD"),
        (["市場規模", "日本"], ["円", "yen"], 9, 14, "Japan domestic market in JPY"),
        (["market", "size", "japan"], ["usd", "$", "ドル"], 8, 12, "Japan market in USD"),
        (["市場規模", "世界"], ["ドル", "usd", "$"], 10, 14, "Global market in USD"),
        (["市場規模", "世界"], ["円", "yen"], 12, 16, "Global market in JPY"),
        (["世帯数", "日本"], None, 7, 8, "Japan households"),
        (["企業数", "日本"], None, 5, 7, "Japan companies"),
        (["売上", "年間"], ["円", "yen"], 6, 14, "Annual revenue in JPY"),
    ]

    # Layer 3: Unit-based fallback ranges
    UNIT_FALLBACK = {
        "人": (0, 11),       # 1 person to world population
        "people": (0, 11),
        "世帯": (0, 8),
        "households": (0, 8),
        "円": (0, 16),       # 1 yen to Japan GDP
        "yen": (0, 16),
        "jpy": (0, 16),
        "ドル": (0, 14),     # 1 USD to world GDP
        "usd": (0, 14),
        "$": (0, 14),
        "%": (-2, 2),        # 0.01% to 100%
        "ratio": (-3, 1),
        "kg": (-6, 12),
        "m": (-9, 8),
        "l": (-3, 12),
    }

    # Scale multipliers for extracting numbers from evidence text
    _SCALE_MAP = {
        "兆": 1e12, "trillion": 1e12,
        "億": 1e8, "billion": 1e9,
        "万": 1e4, "million": 1e6,
        "千": 1e3, "thousand": 1e3,
    }

    def __init__(self, llm_client=None, language: str = "ja"):
        self.llm_client = llm_client
        self.language = language

    def get_domain_prior(
        self,
        target_metric: str,
        unit: str,
        data_store=None,
        evidence_locker=None,
    ) -> DomainPrior:
        """
        Get domain prior through 4-layer fallback chain.

        Layer 0: Collected evidence (highest priority, zero token cost)
        Layer 1: Static rules
        Layer 2: LLM query
        Layer 3: Unit-based fallback
        """
        # Layer 0: Collected evidence
        prior = self._prior_from_evidence(target_metric, unit, data_store, evidence_locker)
        if prior:
            logger.info(
                f"Domain prior from evidence ({prior.evidence_count} data points): "
                f"10^{prior.expected_order_low}..10^{prior.expected_order_high}"
            )
            return prior

        # Layer 1: Static rules
        prior = self._prior_from_static_rules(target_metric, unit)
        if prior:
            logger.info(
                f"Domain prior from static rule: "
                f"10^{prior.expected_order_low}..10^{prior.expected_order_high}"
            )
            return prior

        # Layer 2: LLM query
        if self.llm_client:
            prior = self._prior_from_llm(target_metric, unit)
            if prior:
                logger.info(
                    f"Domain prior from LLM: "
                    f"10^{prior.expected_order_low}..10^{prior.expected_order_high}"
                )
                return prior

        # Layer 3: Unit-based fallback
        prior = self._prior_from_unit_fallback(target_metric, unit)
        logger.info(
            f"Domain prior from unit fallback: "
            f"10^{prior.expected_order_low}..10^{prior.expected_order_high}"
        )
        return prior

    def _prior_from_evidence(
        self,
        target_metric: str,
        unit: str,
        data_store,
        evidence_locker,
    ) -> Optional[DomainPrior]:
        """Layer 0: Derive domain prior from collected research data."""
        related_values = []

        # A. Search NumericalDataStore
        if data_store and hasattr(data_store, "data_points") and data_store.data_points:
            related_values.extend(
                self._search_data_store(target_metric, unit, data_store)
            )

        # B. Search EvidenceLocker if data_store didn't yield enough
        if len(related_values) < 2 and evidence_locker:
            related_values.extend(
                self._search_evidence_locker(target_metric, unit, evidence_locker)
            )

        if not related_values:
            return None

        return self._compute_prior_from_values(related_values)

    def _search_data_store(
        self, target_metric: str, unit: str, data_store,
    ) -> List[Dict[str, Any]]:
        """Search NumericalDataStore for related numerical data points."""
        metric_lower = target_metric.lower()
        results = []

        for dp in data_store.data_points:
            dp_text = f"{dp.metric_name} {dp.subject} {dp.raw_text}".lower()

            # Score relevance by keyword overlap
            score = 0
            for word in metric_lower.split():
                if len(word) > 1 and word in dp_text:
                    score += 1

            # Bonus for unit match
            if dp.unit and unit:
                dp_unit = dp.unit.lower()
                target_unit = unit.lower()
                if dp_unit in target_unit or target_unit in dp_unit:
                    score += 2

            ref_value = dp.normalized_value or dp.value
            if score >= 2 and ref_value and ref_value > 0:
                results.append({
                    "value": ref_value,
                    "confidence": dp.combined_confidence or 0.5,
                    "source": f"{dp.metric_name}: {dp.raw_text}",
                    "score": score,
                })

        return results

    def _search_evidence_locker(
        self, target_metric: str, unit: str, evidence_locker,
    ) -> List[Dict[str, Any]]:
        """Search EvidenceLocker text content for numerical hints."""
        results = []
        metric_lower = target_metric.lower()

        try:
            all_evidence = evidence_locker.get_all_evidence()
        except (AttributeError, TypeError):
            return results

        for evidence in all_evidence:
            text = (
                getattr(evidence, "content_excerpt", "") or
                getattr(evidence, "extracted_text", "") or
                ""
            ).lower()

            # Check relevance
            overlap = sum(
                1 for word in metric_lower.split()
                if len(word) > 1 and word in text
            )
            if overlap < 2:
                continue

            # Extract numbers with scale suffixes from text
            pattern = r'([\d,]+\.?\d*)\s*(兆|億|万|千|trillion|billion|million|thousand)?'
            for match in re.finditer(pattern, text):
                num_str, scale = match.group(1), match.group(2)
                try:
                    val = float(num_str.replace(",", ""))
                    multiplier = self._SCALE_MAP.get(scale, 1) if scale else 1
                    final_val = val * multiplier
                    if final_val > 0:
                        confidence = getattr(evidence, "reliability_score", 0.4) or 0.4
                        results.append({
                            "value": final_val,
                            "confidence": confidence,
                            "source": getattr(evidence, "title", "") or "evidence",
                            "score": overlap,
                        })
                except ValueError:
                    continue

        return results

    def _compute_prior_from_values(
        self, related_values: List[Dict[str, Any]],
    ) -> Optional[DomainPrior]:
        """Compute expected order-of-magnitude range from collected values."""
        # Sort by relevance (score * confidence)
        weighted = sorted(
            related_values,
            key=lambda x: x["confidence"] * x["score"],
            reverse=True,
        )
        top_n = weighted[:10]

        orders = []
        for v in top_n:
            if v["value"] > 0:
                orders.append(math.log10(v["value"]))

        if not orders:
            return None

        reference_points = [
            {"name": v["source"], "value": v["value"]}
            for v in top_n[:3]
        ]

        if len(orders) == 1:
            center = orders[0]
            return DomainPrior(
                expected_order_low=int(center - 2),
                expected_order_high=int(center + 2),
                reference_points=reference_points,
                reasoning=f"Based on {len(top_n)} collected data point(s) "
                          f"near 10^{center:.1f}",
                source="evidence",
                evidence_count=len(top_n),
            )

        order_min = min(orders)
        order_max = max(orders)
        order_median = sorted(orders)[len(orders) // 2]

        return DomainPrior(
            expected_order_low=int(order_min - 1),
            expected_order_high=int(order_max + 1),
            reference_points=reference_points,
            reasoning=f"Based on {len(top_n)} collected data points: "
                      f"range 10^{order_min:.1f}..10^{order_max:.1f}, "
                      f"median 10^{order_median:.1f}",
            source="evidence",
            evidence_count=len(top_n),
        )

    def _prior_from_static_rules(
        self, target_metric: str, unit: str,
    ) -> Optional[DomainPrior]:
        """Layer 1: Match against static rule table."""
        metric_lower = target_metric.lower()
        unit_lower = (unit or "").lower()

        best_match = None
        best_score = 0

        for keywords, unit_patterns, order_low, order_high, desc in self.STATIC_RULES:
            # Check keyword match
            keyword_hits = sum(1 for kw in keywords if kw.lower() in metric_lower)
            if keyword_hits < 2:
                continue

            # Check unit match (if unit_patterns specified)
            if unit_patterns:
                unit_hit = any(up.lower() in unit_lower for up in unit_patterns)
                if not unit_hit:
                    continue

            score = keyword_hits + (1 if unit_patterns else 0)
            if score > best_score:
                best_score = score
                best_match = (order_low, order_high, desc)

        if best_match:
            order_low, order_high, desc = best_match
            return DomainPrior(
                expected_order_low=order_low,
                expected_order_high=order_high,
                reasoning=f"Static rule: {desc}",
                source="static_rule",
            )

        return None

    def _prior_from_llm(
        self, target_metric: str, unit: str,
    ) -> Optional[DomainPrior]:
        """Layer 2: Ask LLM for expected order of magnitude."""
        if self.language == "ja":
            prompt = (
                f"「{target_metric}」（単位: {unit}）の妥当な桁数範囲を教えてください。\n"
                f"JSONのみ出力: {{\"order_low\": N, \"order_high\": M, \"reasoning\": \"...\"}}\n"
                f"order_low/order_highは10のべき乗の指数（例: 10^9なら9）。"
            )
        else:
            prompt = (
                f"What is the expected order of magnitude range for "
                f"\"{target_metric}\" (unit: {unit})?\n"
                f"Output JSON only: {{\"order_low\": N, \"order_high\": M, \"reasoning\": \"...\"}}\n"
                f"order_low/order_high are exponents of 10 (e.g., 9 for 10^9)."
            )

        try:
            response = self.llm_client.generate(prompt)
            if not response or not response.content:
                return None

            data = self._parse_json(response.content)
            if not data:
                return None

            order_low = int(data.get("order_low", 0))
            order_high = int(data.get("order_high", 20))
            reasoning = data.get("reasoning", "")

            if order_low > order_high:
                order_low, order_high = order_high, order_low

            return DomainPrior(
                expected_order_low=order_low,
                expected_order_high=order_high,
                reasoning=f"LLM estimate: {reasoning}",
                source="llm",
            )

        except Exception as e:
            logger.error(f"LLM domain prior query failed: {e}")
            return None

    def _prior_from_unit_fallback(
        self, target_metric: str, unit: str,
    ) -> DomainPrior:
        """Layer 3: Fallback based on unit alone."""
        unit_lower = (unit or "").lower()

        for unit_key, (order_low, order_high) in self.UNIT_FALLBACK.items():
            if unit_key in unit_lower:
                return DomainPrior(
                    expected_order_low=order_low,
                    expected_order_high=order_high,
                    reasoning=f"Unit-based fallback for '{unit}'",
                    source="unit_fallback",
                )

        # Ultimate fallback: extremely wide range
        return DomainPrior(
            expected_order_low=0,
            expected_order_high=20,
            reasoning="No matching rule; using universal fallback",
            source="unit_fallback",
        )

    @staticmethod
    def _parse_json(content: str) -> Optional[Dict]:
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


class Validator:
    """
    Validates Fermi estimation results.

    Performs:
    1. Sanity checks (positive values, low <= base <= high, etc.)
    2. Domain prior check (order-of-magnitude range from evidence/rules/LLM)
    3. Cross-checks against NumericalDataStore data
    4. LLM-based reasonableness check (optional)
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

        # 1. Sanity checks
        sanity_issues = self._sanity_checks(base_value, low_value, high_value)
        all_issues.extend(sanity_issues)
        sanity_total = 5  # 4 basic sanity checks + 1 domain prior check

        # 2. Domain prior check (order-of-magnitude range validation)
        prior_provider = DomainPriorProvider(
            llm_client=self.llm_client, language=self.language,
        )
        domain_prior = prior_provider.get_domain_prior(
            target_metric=target_metric,
            unit=unit,
            data_store=self.data_store,
            evidence_locker=self.evidence_locker,
        )
        prior_issues = self._check_against_prior(base_value, domain_prior)
        all_issues.extend(prior_issues)

        # 3. Cross-check with evidence
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

        # 4. LLM validation
        confidence = 0.5
        if self.llm_client:
            llm_issues, llm_confidence = self._llm_validation(
                target_metric, base_value, low_value, high_value,
                unit, assumptions or [],
            )
            all_issues.extend(llm_issues)
            confidence = llm_confidence

        has_errors = any(i.severity == "error" for i in all_issues)
        sanity_errors = len(
            [i for i in sanity_issues if i.severity == "error"]
        ) + len(
            [i for i in prior_issues if i.severity == "error"]
        )

        return ValidationResult(
            is_valid=not has_errors,
            overall_confidence=confidence,
            issues=all_issues,
            cross_check_results=cross_checks,
            sanity_checks_passed=sanity_total - sanity_errors,
            sanity_checks_total=sanity_total,
            domain_prior=domain_prior,
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

    def _check_against_prior(
        self, base_value: float, prior: DomainPrior,
    ) -> List[ValidationIssue]:
        """Check estimate against domain prior expected range."""
        issues = []

        if not base_value or base_value <= 0:
            return issues

        actual_order = math.log10(base_value)
        margin = prior.margin
        low_bound = prior.expected_order_low - margin
        high_bound = prior.expected_order_high + margin

        if actual_order < low_bound:
            deviation = low_bound - actual_order
            severity = "error" if deviation > 2 else "warning"
            issues.append(ValidationIssue(
                severity=severity,
                category="domain_prior",
                description=(
                    f"Estimate 10^{actual_order:.1f} is below expected range "
                    f"10^{prior.expected_order_low}..10^{prior.expected_order_high} "
                    f"(margin ±{margin})"
                ),
                suggestion=(
                    f"Value appears {10**deviation:.0f}x too small. "
                    f"Prior source: {prior.source} ({prior.reasoning})"
                ),
            ))
        elif actual_order > high_bound:
            deviation = actual_order - high_bound
            severity = "error" if deviation > 2 else "warning"
            issues.append(ValidationIssue(
                severity=severity,
                category="domain_prior",
                description=(
                    f"Estimate 10^{actual_order:.1f} is above expected range "
                    f"10^{prior.expected_order_low}..10^{prior.expected_order_high} "
                    f"(margin ±{margin})"
                ),
                suggestion=(
                    f"Value appears {10**deviation:.0f}x too large. "
                    f"Prior source: {prior.source} ({prior.reasoning})"
                ),
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
