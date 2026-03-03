"""
Research Clarifier - Pre-research confirmation flow.

Inspired by open_deep_research's clarify_with_user(), this module
analyzes a user's research query and determines if clarification
questions should be asked before research begins.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable


@dataclass
class ClarificationResult:
    """Result of query analysis for clarification."""

    needs_clarification: bool = False
    questions: List[str] = field(default_factory=list)
    interpreted_scope: str = ""
    suggested_requirements: str = ""
    estimated_sections: int = 5

    def to_dict(self) -> Dict:
        return {
            "needs_clarification": self.needs_clarification,
            "questions": self.questions,
            "interpreted_scope": self.interpreted_scope,
            "suggested_requirements": self.suggested_requirements,
            "estimated_sections": self.estimated_sections,
        }


class ResearchClarifier:
    """
    Pre-research query clarification.

    Analyzes a user's research query to determine if it's specific enough
    to produce high-quality results, or if clarification questions should
    be asked first.
    """

    def __init__(self, llm_client, language: str = "ja"):
        self.llm = llm_client
        self.language = language

    def analyze_query(
        self,
        query: str,
        requirements: str = "",
    ) -> ClarificationResult:
        """
        Analyze a research query and determine if clarification is needed.

        Returns ClarificationResult with questions if the query is ambiguous.
        """
        if self.language == "ja":
            prompt = self._build_ja_prompt(query, requirements)
        else:
            prompt = self._build_en_prompt(query, requirements)

        try:
            response = self.llm.generate(prompt)
            return self._parse_result(response.content)
        except Exception as e:
            print(f"[Clarifier] Analysis failed: {e}")
            return ClarificationResult(needs_clarification=False)

    def merge_clarification(
        self,
        original_requirements: str,
        answers: Dict[str, str],
    ) -> str:
        """
        Merge user's clarification answers into the requirements string.

        Args:
            original_requirements: The original requirements string.
            answers: Dict mapping question -> answer.

        Returns:
            Updated requirements string with clarification answers appended.
        """
        if not answers:
            return original_requirements

        clarification_text = "\n".join(
            f"- {q}: {a}" for q, a in answers.items() if a
        )

        if original_requirements:
            return f"{original_requirements}\n\n[Clarification]\n{clarification_text}"
        return f"[Clarification]\n{clarification_text}"

    def _build_ja_prompt(self, query: str, requirements: str) -> str:
        return f"""あなたは研究コンサルタントです。以下の研究クエリを分析し、研究を開始する前に確認すべき事項があるか判断してください。

## 研究クエリ
{query}

## 要件
{requirements or '（指定なし）'}

## 分析観点
1. **スコープの明確さ**: 広すぎないか、狭すぎないか
2. **時間軸**: いつの情報が必要か明確か
3. **地理的範囲**: 対象地域が明確か
4. **技術レベル**: 専門家向けか一般向けか
5. **期待する深さ**: 概要レベルか詳細分析か

## 判断基準
- クエリが十分に具体的であれば、needs_clarification: false
- 質問は最大3つまで。本当に必要な場合のみ
- 要件で既にカバーされている情報は質問しない

JSON形式で回答:
{{
    "needs_clarification": true/false,
    "questions": ["質問1（あれば）", "質問2（あれば）"],
    "interpreted_scope": "このクエリをこう解釈しました（1-2文）",
    "suggested_requirements": "推奨する追加要件（あれば）",
    "estimated_sections": 5
}}"""

    def _build_en_prompt(self, query: str, requirements: str) -> str:
        return f"""You are a research consultant. Analyze the research query below and determine if clarification questions should be asked before starting research.

## Research Query
{query}

## Requirements
{requirements or '(none specified)'}

## Analysis Criteria
1. **Scope clarity**: Too broad or too narrow?
2. **Time frame**: Is the relevant time period clear?
3. **Geographic scope**: Is the target region clear?
4. **Technical level**: Expert audience or general audience?
5. **Expected depth**: Overview level or detailed analysis?

## Guidelines
- If the query is specific enough, set needs_clarification: false
- Maximum 3 questions. Only ask if truly necessary
- Don't ask about information already covered in requirements

Return JSON:
{{
    "needs_clarification": true/false,
    "questions": ["question 1 (if needed)", "question 2 (if needed)"],
    "interpreted_scope": "Interpreted scope (1-2 sentences)",
    "suggested_requirements": "Suggested additional requirements (if any)",
    "estimated_sections": 5
}}"""

    def _parse_result(self, content: str) -> ClarificationResult:
        """Parse LLM response into ClarificationResult."""
        result = ClarificationResult()
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                result.needs_clarification = bool(data.get("needs_clarification", False))
                result.questions = data.get("questions", [])[:3]
                result.interpreted_scope = data.get("interpreted_scope", "")
                result.suggested_requirements = data.get("suggested_requirements", "")
                result.estimated_sections = int(data.get("estimated_sections", 5))
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[Clarifier] Failed to parse result: {e}")
        return result
