"""
Fermi Estimator - Order-of-magnitude estimation for Deep Research Tool.

Decomposes a quantitative question into multiplicative factors,
estimates each factor with the LLM (optionally grounded on known
values), and combines them into a final estimate with an
uncertainty range.

Example usage:
    from deep_research_tool.thinking import FermiEstimator
    from deep_research_tool.api import get_client

    llm = get_client(provider="openai")
    estimator = FermiEstimator(llm_client=llm, language="ja")

    result = estimator.estimate("日本国内のピアノ調律師の人数は？")
    print(result.to_markdown())
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from deep_research_tool.utils.helpers import extract_json_from_response


logger = logging.getLogger(__name__)


@dataclass
class FermiFactor:
    """A single factor in a Fermi decomposition."""

    name: str
    description: str = ""
    low: float = 0.0
    mid: float = 0.0
    high: float = 0.0
    unit: str = ""
    operation: str = "multiply"  # "multiply" or "divide"
    basis: str = "assumption"    # assumption / known_value / llm_knowledge

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "low": self.low,
            "mid": self.mid,
            "high": self.high,
            "unit": self.unit,
            "operation": self.operation,
            "basis": self.basis,
        }


@dataclass
class FermiEstimate:
    """Result of a Fermi estimation."""

    question: str
    factors: List[FermiFactor] = field(default_factory=list)
    value: float = 0.0
    low: float = 0.0
    high: float = 0.0
    unit: str = ""
    formula: str = ""
    reasoning: str = ""
    assumptions: List[str] = field(default_factory=list)
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "factors": [f.to_dict() for f in self.factors],
            "value": self.value,
            "low": self.low,
            "high": self.high,
            "unit": self.unit,
            "formula": self.formula,
            "reasoning": self.reasoning,
            "assumptions": self.assumptions,
            "confidence": self.confidence,
        }

    def to_markdown(self) -> str:
        """Render the estimate as a markdown summary."""
        lines = [
            f"## フェルミ推定: {self.question}",
            "",
            f"**推定値: {format_number(self.value)} {self.unit}**",
            f"（範囲: {format_number(self.low)} 〜 {format_number(self.high)} {self.unit}、"
            f"確信度: {self.confidence:.0%}）",
            "",
            f"**計算式**: {self.formula}",
            "",
            "### 分解した要素",
            "",
            "| 要素 | 中央値 | 範囲 | 単位 | 演算 | 根拠 |",
            "|------|--------|------|------|------|------|",
        ]
        for f in self.factors:
            op = "×" if f.operation == "multiply" else "÷"
            lines.append(
                f"| {f.name} | {format_number(f.mid)} | "
                f"{format_number(f.low)}〜{format_number(f.high)} | "
                f"{f.unit} | {op} | {f.basis} |"
            )
        if self.assumptions:
            lines += ["", "### 前提条件", ""]
            lines += [f"- {a}" for a in self.assumptions]
        if self.reasoning:
            lines += ["", "### 推論過程", "", self.reasoning]
        return "\n".join(lines)


def format_number(value: float) -> str:
    """Format a number for display (scientific notation when large/small)."""
    if value == 0:
        return "0"
    abs_v = abs(value)
    if abs_v >= 1e5 or abs_v < 1e-3:
        exponent = math.floor(math.log10(abs_v))
        mantissa = value / (10 ** exponent)
        return f"{mantissa:.2f}×10^{exponent}"
    if abs_v >= 100:
        return f"{value:,.0f}"
    return f"{value:,.3g}"


class FermiEstimator:
    """
    LLM-driven Fermi estimation.

    Decomposes a question into factors, estimates each with a
    low/mid/high range, and multiplies (or divides) them together.
    """

    def __init__(self, llm_client, language: str = "ja"):
        """
        Args:
            llm_client: LLM client with a generate(prompt, system_prompt) method
            language: Output language ("ja" or "en")
        """
        self.llm = llm_client
        self.language = language

    def estimate(
        self,
        question: str,
        context: str = "",
        known_values: Optional[Dict[str, float]] = None,
    ) -> FermiEstimate:
        """
        Perform a Fermi estimation for a quantitative question.

        Args:
            question: The quantitative question (e.g., "日本のEV充電器の総数は？")
            context: Additional context (e.g., excerpts from research evidence)
            known_values: Known values to ground factors on (name -> value)

        Returns:
            FermiEstimate with factors and combined estimate

        Raises:
            ValueError: If the LLM response cannot be parsed into factors
        """
        prompt = self._build_prompt(question, context, known_values)
        response = self.llm.generate(prompt, system_prompt=self._system_prompt())
        data = self._parse_response(response.content)

        factors = []
        for item in data.get("factors", []):
            try:
                factors.append(FermiFactor(
                    name=str(item.get("name", "")),
                    description=str(item.get("description", "")),
                    low=float(item.get("low", 0)),
                    mid=float(item.get("mid", 0)),
                    high=float(item.get("high", 0)),
                    unit=str(item.get("unit", "")),
                    operation=item.get("operation", "multiply"),
                    basis=str(item.get("basis", "assumption")),
                ))
            except (TypeError, ValueError) as e:
                logger.debug(f"Skipping unparsable factor: {e}")

        if not factors:
            raise ValueError(
                f"LLM returned no usable factors for question: {question}"
            )

        value, low, high = self._combine(factors)

        return FermiEstimate(
            question=question,
            factors=factors,
            value=value,
            low=low,
            high=high,
            unit=str(data.get("unit", "")),
            formula=str(data.get("formula", "")),
            reasoning=str(data.get("reasoning", "")),
            assumptions=[str(a) for a in data.get("assumptions", [])],
            confidence=self._confidence(factors),
        )

    def _combine(self, factors: List[FermiFactor]) -> tuple:
        """Combine factors into (mid, low, high).

        For divisors, the low bound uses the divisor's high value and
        vice versa, so the range stays consistent.
        """
        mid = low = high = 1.0
        for f in factors:
            if f.operation == "divide":
                if f.mid == 0 or f.low == 0 or f.high == 0:
                    logger.warning(f"Skipping zero divisor factor: {f.name}")
                    continue
                mid /= f.mid
                low /= f.high
                high /= f.low
            else:
                mid *= f.mid
                low *= f.low
                high *= f.high
        return mid, low, high

    def _confidence(self, factors: List[FermiFactor]) -> float:
        """Heuristic confidence: narrower ranges and grounded values score higher."""
        if not factors:
            return 0.0
        spread_scores = []
        basis_scores = []
        for f in factors:
            if f.low > 0 and f.high > 0:
                ratio = f.high / f.low
                # ratio 1 -> 1.0, ratio 100 -> ~0.0
                spread_scores.append(max(0.0, 1.0 - math.log10(ratio) / 2))
            else:
                spread_scores.append(0.3)
            basis_scores.append(0.9 if f.basis == "known_value" else 0.5)
        return round(
            0.6 * (sum(spread_scores) / len(spread_scores))
            + 0.4 * (sum(basis_scores) / len(basis_scores)),
            2,
        )

    def _system_prompt(self) -> str:
        if self.language == "ja":
            return (
                "あなたはフェルミ推定の専門家です。定量的な質問を掛け算・割り算で"
                "組み合わせられる要素に分解し、各要素を保守的な範囲付きで推定します。"
                "必ず指定されたJSON形式のみで回答してください。"
            )
        return (
            "You are an expert in Fermi estimation. Decompose quantitative "
            "questions into multiplicative factors with conservative ranges. "
            "Respond only in the specified JSON format."
        )

    def _build_prompt(
        self,
        question: str,
        context: str,
        known_values: Optional[Dict[str, float]],
    ) -> str:
        known_text = ""
        if known_values:
            rows = "\n".join(f"- {k}: {v}" for k, v in known_values.items())
            known_text = (
                f"\n【既知の値（basis を known_value としてそのまま使うこと）】\n{rows}\n"
                if self.language == "ja"
                else f"\n[Known values - use as-is with basis=known_value]\n{rows}\n"
            )

        context_text = ""
        if context:
            context_text = (
                f"\n【参考情報】\n{context[:3000]}\n"
                if self.language == "ja"
                else f"\n[Reference context]\n{context[:3000]}\n"
            )

        if self.language == "ja":
            return f"""以下の定量的な質問についてフェルミ推定を行ってください。

質問: {question}
{known_text}{context_text}
質問の答えを掛け算・割り算で組み合わせられる3〜7個の要素に分解し、
各要素について low（下限）、mid（中央推定値）、high（上限）を推定してください。

JSON形式で出力:
```json
{{
  "unit": "最終推定値の単位",
  "formula": "要素A × 要素B ÷ 要素C の形式の計算式",
  "factors": [
    {{
      "name": "要素名",
      "description": "この要素の説明",
      "low": 1000,
      "mid": 5000,
      "high": 10000,
      "unit": "単位",
      "operation": "multiply",
      "basis": "assumption"
    }}
  ],
  "assumptions": ["前提条件1", "前提条件2"],
  "reasoning": "分解の考え方と各推定の根拠の説明"
}}
```

注意:
- operation は "multiply"（掛ける）または "divide"（割る）
- basis は "known_value"（既知の値）、"llm_knowledge"（一般知識）、"assumption"（仮定）
- 数値は数値型で出力（文字列にしない）"""

        return f"""Perform a Fermi estimation for the following quantitative question.

Question: {question}
{known_text}{context_text}
Decompose the answer into 3-7 factors combinable via multiplication/division,
estimating low, mid, and high values for each.

Output JSON:
```json
{{
  "unit": "unit of the final estimate",
  "formula": "factor A × factor B ÷ factor C",
  "factors": [
    {{
      "name": "factor name",
      "description": "what this factor represents",
      "low": 1000,
      "mid": 5000,
      "high": 10000,
      "unit": "unit",
      "operation": "multiply",
      "basis": "assumption"
    }}
  ],
  "assumptions": ["assumption 1", "assumption 2"],
  "reasoning": "explanation of the decomposition and estimates"
}}
```

Notes:
- operation is "multiply" or "divide"
- basis is "known_value", "llm_knowledge", or "assumption"
- Output numbers as JSON numbers, not strings"""

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse the LLM response into a dict, with a lenient fallback."""
        try:
            return extract_json_from_response(content)
        except (ValueError, json.JSONDecodeError):
            pass

        # Lenient fallback: find outermost braces
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not parse Fermi estimation JSON from LLM response")
