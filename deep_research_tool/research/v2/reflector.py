"""
Research Reflector - Strategic thinking tool for research loops.

Inspired by open_deep_research's think_tool, this module provides
meta-cognitive reflection during research iterations. It evaluates
coverage, quality, and strategic direction of ongoing research.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ..query_generator import TableOfContentsItem
from ..content_extractor import ExtractedContent


@dataclass
class ReflectionResult:
    """Output of a Think Tool reflection step."""

    # Coverage assessment
    coverage_assessment: str = ""
    coverage_score: float = 0.0  # 0.0-1.0

    # Quality assessment
    quality_assessment: str = ""
    quality_score: float = 0.0  # 0.0-1.0

    # Blind spots and gaps
    identified_blind_spots: List[str] = field(default_factory=list)

    # Strategic direction
    strategic_direction: str = ""
    should_pivot: bool = False
    pivot_reason: Optional[str] = None

    # Recommended next actions
    recommended_queries: List[str] = field(default_factory=list)

    # Decision
    stop_research: bool = False
    stop_reason: Optional[str] = None

    # Overall confidence
    confidence: float = 0.0  # 0.0-1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage_assessment": self.coverage_assessment,
            "coverage_score": self.coverage_score,
            "quality_assessment": self.quality_assessment,
            "quality_score": self.quality_score,
            "identified_blind_spots": self.identified_blind_spots,
            "strategic_direction": self.strategic_direction,
            "should_pivot": self.should_pivot,
            "pivot_reason": self.pivot_reason,
            "recommended_queries": self.recommended_queries,
            "stop_research": self.stop_research,
            "stop_reason": self.stop_reason,
            "confidence": self.confidence,
        }


@dataclass
class OverallReflection:
    """Reflection on the entire research session after all sections."""

    is_complete: bool = False
    overall_quality: float = 0.0
    cross_section_gaps: List[str] = field(default_factory=list)
    redundancies: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_complete": self.is_complete,
            "overall_quality": self.overall_quality,
            "cross_section_gaps": self.cross_section_gaps,
            "redundancies": self.redundancies,
            "suggestions": self.suggestions,
        }


class ResearchReflector:
    """
    Strategic thinking tool for research loops.

    Provides meta-cognitive reflection during research iterations,
    evaluating coverage, quality, and strategic direction. Inspired
    by open_deep_research's think_tool but with structured output.
    """

    def __init__(self, llm_client, language: str = "ja"):
        self.llm = llm_client
        self.language = language

    def reflect_on_section(
        self,
        section: TableOfContentsItem,
        collected_evidence: List[ExtractedContent],
        iteration: int,
        max_iterations: int,
        research_topic: str,
        previous_sections_summary: Dict[str, str] = None,
    ) -> ReflectionResult:
        """
        Reflect on the current state of section research.

        Called after each research iteration (from iteration 2+) to decide
        whether to continue, pivot, or stop researching this section.
        """
        # Build evidence summary
        evidence_summary = self._summarize_evidence(collected_evidence)
        prev_summary = self._format_previous_sections(previous_sections_summary or {})

        if self.language == "ja":
            prompt = self._build_ja_prompt(
                section, evidence_summary, iteration, max_iterations,
                research_topic, prev_summary,
            )
        else:
            prompt = self._build_en_prompt(
                section, evidence_summary, iteration, max_iterations,
                research_topic, prev_summary,
            )

        try:
            response = self.llm.generate(prompt)
            return self._parse_reflection(response.content)
        except Exception as e:
            print(f"[Reflector] Reflection failed: {e}")
            # Default: continue research
            return ReflectionResult(
                coverage_assessment="Reflection failed",
                quality_assessment="Unknown",
                confidence=0.5,
            )

    def reflect_on_overall(
        self,
        section_contents: Dict[str, Dict],
        research_topic: str,
    ) -> OverallReflection:
        """
        Reflect on the entire research session after all sections are done.

        Identifies cross-section gaps, redundancies, and overall quality.
        """
        sections_text = []
        for section_id, content in section_contents.items():
            if section_id.startswith("_"):
                continue
            title = content.get("title", section_id)
            text = content.get("content", "")[:400]
            sections_text.append(f"[{section_id}] {title}\n{text}")

        sections_combined = "\n---\n".join(sections_text)

        if self.language == "ja":
            prompt = f"""あなたは研究アドバイザーです。以下の研究レポートの全セクションを振り返り、全体的な評価を行ってください。

研究テーマ: {research_topic}

全セクション概要:
{sections_combined}

以下をJSON形式で回答してください:
{{
    "is_complete": true/false,
    "overall_quality": 0.0-1.0,
    "cross_section_gaps": ["セクション間のギャップ"],
    "redundancies": ["セクション間の重複"],
    "suggestions": ["改善提案"]
}}"""
        else:
            prompt = f"""You are a research advisor. Review all sections below and provide an overall assessment.

Research topic: {research_topic}

All sections:
{sections_combined}

Return JSON:
{{
    "is_complete": true/false,
    "overall_quality": 0.0-1.0,
    "cross_section_gaps": ["gaps between sections"],
    "redundancies": ["redundant content across sections"],
    "suggestions": ["improvement suggestions"]
}}"""

        try:
            response = self.llm.generate(prompt)
            return self._parse_overall_reflection(response.content)
        except Exception as e:
            print(f"[Reflector] Overall reflection failed: {e}")
            return OverallReflection(is_complete=True, overall_quality=0.5)

    def _build_ja_prompt(
        self, section, evidence_summary, iteration, max_iterations,
        research_topic, prev_summary,
    ) -> str:
        return f"""あなたは研究戦略アドバイザーです。現在のセクション調査の進捗を分析し、次のステップを戦略的に計画してください。

## 研究テーマ
{research_topic}

## 現在のセクション
セクション {section.section}: {section.title}
説明: {section.description}

## 調査の進捗
イテレーション: {iteration}/{max_iterations}
収集済みエビデンス数: {len(evidence_summary.get('sources', []))}

## 収集済み情報の要約
{evidence_summary.get('summary', '情報なし')}

## 他セクションの概要
{prev_summary or 'なし（最初のセクション）'}

## 分析指示
以下の観点で現状を分析し、JSON形式で回答してください:

1. **網羅性**: セクションの要件をどの程度カバーしているか
2. **情報の質**: 収集した情報は信頼できるか、数値データはあるか
3. **見落とし**: 調査すべきだが未調査の領域
4. **戦略**: 次にどうすべきか（続行/方向転換/終了）

JSON形式:
{{
    "coverage_assessment": "網羅性の評価（1-2文）",
    "coverage_score": 0.0-1.0,
    "quality_assessment": "情報の質の評価（1-2文）",
    "quality_score": 0.0-1.0,
    "identified_blind_spots": ["見落とし1", "見落とし2"],
    "strategic_direction": "次のステップの方針（1文）",
    "should_pivot": false,
    "pivot_reason": null,
    "recommended_queries": ["推奨クエリ1", "推奨クエリ2"],
    "stop_research": false,
    "stop_reason": null,
    "confidence": 0.0-1.0
}}

**判断基準**:
- coverage_score >= 0.8 かつ quality_score >= 0.7 → stop_research: true を検討
- イテレーション {iteration}/{max_iterations} を考慮し、残り回数が少なければ効率的な方針を
- 見落としが重大ならば should_pivot: true とし、recommended_queries で方向修正"""

    def _build_en_prompt(
        self, section, evidence_summary, iteration, max_iterations,
        research_topic, prev_summary,
    ) -> str:
        return f"""You are a research strategy advisor. Analyze the current section research progress and plan the next steps strategically.

## Research Topic
{research_topic}

## Current Section
Section {section.section}: {section.title}
Description: {section.description}

## Research Progress
Iteration: {iteration}/{max_iterations}
Evidence collected: {len(evidence_summary.get('sources', []))}

## Collected Information Summary
{evidence_summary.get('summary', 'No information yet')}

## Other Sections Summary
{prev_summary or 'None (first section)'}

## Analysis Instructions
Analyze the current state from these perspectives and return JSON:

1. **Coverage**: How well does the collected data cover section requirements?
2. **Quality**: Is the information reliable? Does it include numerical data?
3. **Blind spots**: Areas that should be investigated but haven't been
4. **Strategy**: What should we do next? (continue/pivot/stop)

JSON format:
{{
    "coverage_assessment": "Coverage evaluation (1-2 sentences)",
    "coverage_score": 0.0-1.0,
    "quality_assessment": "Quality evaluation (1-2 sentences)",
    "quality_score": 0.0-1.0,
    "identified_blind_spots": ["blind spot 1", "blind spot 2"],
    "strategic_direction": "Next step strategy (1 sentence)",
    "should_pivot": false,
    "pivot_reason": null,
    "recommended_queries": ["recommended query 1", "recommended query 2"],
    "stop_research": false,
    "stop_reason": null,
    "confidence": 0.0-1.0
}}

**Decision criteria**:
- coverage_score >= 0.8 and quality_score >= 0.7 → consider stop_research: true
- Consider iteration {iteration}/{max_iterations} and prioritize efficiency
- If blind spots are critical, set should_pivot: true with recommended_queries"""

    def _summarize_evidence(self, evidence: List[ExtractedContent]) -> Dict[str, Any]:
        """Create a summary of collected evidence for the prompt."""
        if not evidence:
            return {"summary": "", "sources": []}

        sources = []
        key_points_all = []
        for ec in evidence:
            sources.append(ec.source_title or ec.source_url)
            if ec.key_points:
                key_points_all.extend(ec.key_points[:3])

        # Build concise summary
        summary_parts = []
        for ec in evidence[:5]:
            if ec.processed_content:
                summary_parts.append(ec.processed_content[:200])

        return {
            "summary": "\n".join(summary_parts) if summary_parts else "",
            "sources": sources,
            "key_points": key_points_all[:10],
        }

    def _format_previous_sections(self, summaries: Dict[str, str]) -> str:
        """Format previous section summaries for context."""
        if not summaries:
            return ""
        parts = []
        for section_id, summary in list(summaries.items())[-3:]:
            parts.append(f"- {section_id}: {summary[:150]}")
        return "\n".join(parts)

    def _parse_reflection(self, content: str) -> ReflectionResult:
        """Parse LLM response into ReflectionResult."""
        result = ReflectionResult()
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                result.coverage_assessment = data.get("coverage_assessment", "")
                result.coverage_score = float(data.get("coverage_score", 0.0))
                result.quality_assessment = data.get("quality_assessment", "")
                result.quality_score = float(data.get("quality_score", 0.0))
                result.identified_blind_spots = data.get("identified_blind_spots", [])
                result.strategic_direction = data.get("strategic_direction", "")
                result.should_pivot = bool(data.get("should_pivot", False))
                result.pivot_reason = data.get("pivot_reason")
                result.recommended_queries = data.get("recommended_queries", [])
                result.stop_research = bool(data.get("stop_research", False))
                result.stop_reason = data.get("stop_reason")
                result.confidence = float(data.get("confidence", 0.5))
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[Reflector] Failed to parse reflection: {e}")
        return result

    def _parse_overall_reflection(self, content: str) -> OverallReflection:
        """Parse LLM response into OverallReflection."""
        result = OverallReflection()
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                result.is_complete = bool(data.get("is_complete", True))
                result.overall_quality = float(data.get("overall_quality", 0.5))
                result.cross_section_gaps = data.get("cross_section_gaps", [])
                result.redundancies = data.get("redundancies", [])
                result.suggestions = data.get("suggestions", [])
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[Reflector] Failed to parse overall reflection: {e}")
        return result
