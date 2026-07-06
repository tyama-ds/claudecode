"""
Assumption management for Fermi estimation.

Manages sourcing, validation, and tracking of assumptions
used in each tree node.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .decomposer import TreeNode

logger = logging.getLogger(__name__)


class AssumptionSource(str, Enum):
    """Source of an assumption value."""
    EVIDENCE_DIRECT = "evidence_direct"
    EVIDENCE_DERIVED = "evidence_derived"
    LLM_ESTIMATE = "llm_estimate"
    USER_INPUT = "user_input"
    COMMON_KNOWLEDGE = "common_knowledge"
    DEFAULT = "default"


@dataclass
class Assumption:
    """A single assumption used in the estimation."""
    assumption_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    node_id: str = ""
    parameter_name: str = ""

    value: float = 0.0
    value_low: float = 0.0
    value_high: float = 0.0
    unit: str = ""

    source: AssumptionSource = AssumptionSource.LLM_ESTIMATE
    source_data_id: str = ""
    source_url: str = ""
    reasoning: str = ""

    confidence: float = 0.5
    sensitivity_rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assumption_id": self.assumption_id,
            "node_id": self.node_id,
            "parameter_name": self.parameter_name,
            "value": self.value,
            "value_low": self.value_low,
            "value_high": self.value_high,
            "unit": self.unit,
            "source": self.source.value,
            "source_data_id": self.source_data_id,
            "source_url": self.source_url,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "sensitivity_rank": self.sensitivity_rank,
        }


class AssumptionManager:
    """
    Manages assumptions for Fermi estimation.

    For each leaf node:
    1. Search NumericalDataStore for matching data
    2. If found, use evidence-backed value
    3. If not, use LLM to estimate with reasoning
    4. Generate pessimistic/optimistic ranges
    """

    ESTIMATION_PROMPT_JA = """あなたはフェルミ推定の専門家です。
以下のパラメータの値を推定してください。

## 推定対象
パラメータ名: {parameter_name}
説明: {description}
単位: {unit}
文脈: {context}

## 参考情報
{reference_info}

## 指示
以下の3つのシナリオで値を推定し、推論過程を説明してください：
1. ベースケース（最も可能性が高い値）
2. 悲観シナリオ（低めの見積もり）
3. 楽観シナリオ（高めの見積もり）

## 出力形式 (JSON)
{{
    "value": 数値,
    "value_low": 数値,
    "value_high": 数値,
    "unit": "単位",
    "reasoning": "推論過程の説明",
    "confidence": 0.0から1.0の信頼度
}}

JSONのみを出力:"""

    ESTIMATION_PROMPT_EN = """You are a Fermi estimation expert.
Estimate the value of the following parameter.

## Parameter
Name: {parameter_name}
Description: {description}
Unit: {unit}
Context: {context}

## Reference Information
{reference_info}

## Instructions
Estimate the value for three scenarios with reasoning:
1. Base case (most likely value)
2. Pessimistic scenario (low estimate)
3. Optimistic scenario (high estimate)

## Output (JSON only)
{{
    "value": number,
    "value_low": number,
    "value_high": number,
    "unit": "unit",
    "reasoning": "Reasoning explanation",
    "confidence": 0.0 to 1.0
}}

Output only JSON:"""

    def __init__(
        self,
        llm_client,
        data_store,
        language: str = "ja",
    ):
        self.llm_client = llm_client
        self.data_store = data_store
        self.language = language
        self._assumptions: List[Assumption] = []

    def resolve_leaf_node(
        self,
        node: TreeNode,
        context: str = "",
    ) -> Assumption:
        """Resolve the value for a leaf node."""
        # Try evidence first
        matched = self._search_data_store(node)
        if matched is not None:
            # Apply +/- 20% for range if not available
            value = matched.normalized_value or matched.value
            assumption = Assumption(
                node_id=node.node_id,
                parameter_name=node.name,
                value=value,
                value_low=value * 0.8,
                value_high=value * 1.2,
                unit=matched.unit or node.unit,
                source=AssumptionSource.EVIDENCE_DIRECT,
                source_data_id=matched.data_id,
                source_url=matched.source_url,
                reasoning=f"Evidence: {matched.raw_text} (from {matched.source_title})",
                confidence=matched.combined_confidence or 0.7,
            )
            # Update the node
            node.value = assumption.value
            node.value_low = assumption.value_low
            node.value_high = assumption.value_high
            node.is_evidence_backed = True
            node.evidence_data_id = matched.data_id
            node.confidence = assumption.confidence

            self._assumptions.append(assumption)
            logger.info(f"Node '{node.name}' resolved from evidence: {value} {assumption.unit}")
            return assumption

        # Fall back to LLM estimation
        reference_data = self._get_reference_data(node)
        assumption = self._estimate_with_llm(node, reference_data, context)
        self._assumptions.append(assumption)
        return assumption

    def _search_data_store(self, node: TreeNode) -> Optional[Any]:
        """Search NumericalDataStore for data matching this node."""
        if not self.data_store:
            return None

        name_lower = node.name.lower()
        name_en_lower = node.name_en.lower() if node.name_en else ""

        best_match = None
        best_score = 0.0

        for dp in self.data_store.data_points:
            score = 0.0
            metric_lower = (dp.metric_name or "").lower()
            subject_lower = (dp.subject or "").lower()
            raw_lower = (dp.raw_text or "").lower()

            # Check name overlap
            for keyword in name_lower.split():
                if len(keyword) > 1:
                    if keyword in metric_lower or keyword in subject_lower:
                        score += 0.3
                    if keyword in raw_lower:
                        score += 0.1

            # Check English name overlap
            if name_en_lower:
                for keyword in name_en_lower.split():
                    if len(keyword) > 2:
                        if keyword in metric_lower or keyword in subject_lower:
                            score += 0.3

            # Unit match bonus
            if node.unit and dp.unit and node.unit.lower() == dp.unit.lower():
                score += 0.2

            # Confidence weighting
            score *= (dp.combined_confidence or 0.5)

            if score > best_score and score >= 0.3:
                best_score = score
                best_match = dp

        return best_match

    def _get_reference_data(self, node: TreeNode) -> str:
        """Get related reference data as text for LLM context."""
        if not self.data_store:
            return "なし" if self.language == "ja" else "None"

        references = []
        for dp in self.data_store.get_high_confidence(threshold=0.4):
            references.append(
                f"- {dp.metric_name}: {dp.raw_text} "
                f"({dp.subject}, {dp.date_context})"
            )
            if len(references) >= 10:
                break

        return "\n".join(references) if references else (
            "なし" if self.language == "ja" else "None"
        )

    def _estimate_with_llm(
        self,
        node: TreeNode,
        reference_info: str,
        context: str = "",
    ) -> Assumption:
        """Use LLM to estimate the value for a leaf node."""
        template = (
            self.ESTIMATION_PROMPT_JA
            if self.language == "ja"
            else self.ESTIMATION_PROMPT_EN
        )
        prompt = template.format(
            parameter_name=node.name,
            description=node.description,
            unit=node.unit,
            context=context,
            reference_info=reference_info,
        )

        try:
            response = self.llm_client.generate(prompt)
            if not response or not response.content:
                return self._default_assumption(node)

            data = self._parse_json(response.content)
            if not data:
                return self._default_assumption(node)

            value = float(data.get("value", 0))
            value_low = float(data.get("value_low", value * 0.5))
            value_high = float(data.get("value_high", value * 2.0))
            confidence = float(data.get("confidence", 0.4))

            assumption = Assumption(
                node_id=node.node_id,
                parameter_name=node.name,
                value=value,
                value_low=value_low,
                value_high=value_high,
                unit=data.get("unit", node.unit),
                source=AssumptionSource.LLM_ESTIMATE,
                reasoning=data.get("reasoning", ""),
                confidence=confidence,
            )

            node.value = value
            node.value_low = value_low
            node.value_high = value_high
            node.estimation_reasoning = assumption.reasoning
            node.confidence = confidence

            logger.info(f"Node '{node.name}' estimated by LLM: {value} {assumption.unit}")
            return assumption

        except Exception as e:
            logger.error(f"LLM estimation failed for '{node.name}': {e}")
            return self._default_assumption(node)

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

    def _default_assumption(self, node: TreeNode) -> Assumption:
        """Create a default assumption when estimation fails."""
        assumption = Assumption(
            node_id=node.node_id,
            parameter_name=node.name,
            value=1.0,
            value_low=0.5,
            value_high=2.0,
            unit=node.unit,
            source=AssumptionSource.DEFAULT,
            reasoning="Default value (estimation failed)",
            confidence=0.1,
        )
        node.value = 1.0
        node.value_low = 0.5
        node.value_high = 2.0
        node.confidence = 0.1
        return assumption

    def get_all_assumptions(self) -> List[Assumption]:
        return self._assumptions.copy()

    def get_evidence_backed_count(self) -> int:
        return sum(
            1 for a in self._assumptions
            if a.source in (AssumptionSource.EVIDENCE_DIRECT, AssumptionSource.EVIDENCE_DERIVED)
        )
