"""
Patent-specific query generator.

Generates patent search queries with IPC/CPC code awareness,
jurisdiction targeting, and multi-layer search query planning.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ..models.patent import Patent

logger = logging.getLogger(__name__)


@dataclass
class PatentSearchPlan:
    """A complete patent research plan."""

    title: str
    summary: str
    patent_queries: List[str] = field(default_factory=list)
    academic_queries: List[str] = field(default_factory=list)
    business_queries: List[str] = field(default_factory=list)
    suggested_ipc_codes: List[str] = field(default_factory=list)
    key_terms: List[str] = field(default_factory=list)
    target_patents: List[str] = field(default_factory=list)
    report_sections: List[Dict[str, str]] = field(default_factory=list)
    methodology_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "patent_queries": self.patent_queries,
            "academic_queries": self.academic_queries,
            "business_queries": self.business_queries,
            "suggested_ipc_codes": self.suggested_ipc_codes,
            "key_terms": self.key_terms,
            "target_patents": self.target_patents,
            "report_sections": self.report_sections,
            "methodology_notes": self.methodology_notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatentSearchPlan":
        return cls(**data)


class PatentQueryGenerator:
    """Generate patent-specific research queries using LLM."""

    def __init__(self, llm_client, language: str = "ja"):
        self.llm = llm_client
        self.language = language

    def create_patent_search_plan(
        self,
        query: str,
        requirements: str = "",
        ipc_codes: List[str] = None,
        target_patents: List[str] = None,
    ) -> PatentSearchPlan:
        """
        Create a comprehensive patent search plan.

        Uses LLM to analyze the query and generate structured search queries
        for patent databases, academic sources, and business evidence.

        Args:
            query: User's research query
            requirements: Additional requirements
            ipc_codes: User-specified IPC codes
            target_patents: Specific patent numbers to focus on

        Returns:
            PatentSearchPlan with queries for all layers
        """
        ipc_hint = ""
        if ipc_codes:
            ipc_hint = f"\nユーザー指定IPC分類: {', '.join(ipc_codes)}"

        target_hint = ""
        if target_patents:
            target_hint = f"\n対象特許番号: {', '.join(target_patents)}"

        prompt = f"""あなたは特許調査の専門家です。以下の調査テーマについて、特許検索計画を策定してください。

【調査テーマ】{query}
【追加要件】{requirements or 'なし'}{ipc_hint}{target_hint}

以下のJSON形式で回答してください：
{{
    "title": "調査レポートのタイトル",
    "summary": "調査の概要（2-3文）",
    "patent_queries": [
        "特許DB検索クエリ1（キーワード＋技術用語）",
        "特許DB検索クエリ2",
        "特許DB検索クエリ3"
    ],
    "academic_queries": [
        "学術論文検索クエリ1（技術的背景調査用）",
        "学術論文検索クエリ2"
    ],
    "business_queries": [
        "ビジネスエビデンス検索クエリ1（市場・売上データ用）"
    ],
    "suggested_ipc_codes": ["H01L21/00", "G06F3/00"],
    "key_terms": ["技術用語1", "技術用語2"],
    "target_patents": ["既知の重要特許番号があれば"],
    "report_sections": [
        {{"section": "1", "title": "セクションタイトル", "description": "内容説明"}},
        {{"section": "2", "title": "セクションタイトル", "description": "内容説明"}}
    ],
    "methodology_notes": "調査方法に関する補足"
}}"""

        try:
            response = self.llm.generate(prompt)
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])

                # Preserve user-specified data
                if target_patents:
                    existing_targets = data.get("target_patents", [])
                    for tp in target_patents:
                        if tp not in existing_targets:
                            existing_targets.append(tp)
                    data["target_patents"] = existing_targets

                if ipc_codes:
                    existing_ipc = data.get("suggested_ipc_codes", [])
                    for code in ipc_codes:
                        if code not in existing_ipc:
                            existing_ipc.insert(0, code)
                    data["suggested_ipc_codes"] = existing_ipc

                return PatentSearchPlan.from_dict(data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"[PatentQueryGenerator] Failed to parse LLM response: {e}")

        # Fallback plan
        return PatentSearchPlan(
            title=f"特許調査: {query}",
            summary=f"{query}に関する特許調査",
            patent_queries=[
                query,
                f"{query} 特許",
                f"{query} patent",
            ],
            academic_queries=[f"{query} 技術 論文"],
            suggested_ipc_codes=ipc_codes or [],
            key_terms=[query],
            target_patents=target_patents or [],
        )

    def generate_follow_up_queries(
        self,
        original_query: str,
        patents_found: List[Patent],
        gaps: List[str] = None,
    ) -> List[str]:
        """
        Generate follow-up queries based on initial patent search results.

        Args:
            original_query: Original research query
            patents_found: Patents found so far
            gaps: Information gaps identified

        Returns:
            List of follow-up search queries
        """
        patent_summary = ""
        for p in patents_found[:5]:
            ipc_str = ", ".join(c.full_code for c in p.ipc_classifications[:3])
            patent_summary += f"- {p.patent_number}: {p.title} (IPC: {ipc_str})\n"

        gaps_text = "\n".join(f"- {g}" for g in (gaps or []))

        prompt = f"""特許調査の追加検索クエリを生成してください。

【調査テーマ】{original_query}

【発見済みの特許】
{patent_summary or '（なし）'}

【情報ギャップ】
{gaps_text or '（なし）'}

以下のJSON形式で、追加の特許検索クエリを3つ生成してください：
["追加クエリ1", "追加クエリ2", "追加クエリ3"]"""

        try:
            response = self.llm.generate(prompt)
            content = response.content
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                queries = json.loads(content[start:end])
                return queries[:3]
        except Exception as e:
            logger.error(f"[PatentQueryGenerator] Follow-up query generation failed: {e}")

        return [
            f"{original_query} 最新 特許",
            f"{original_query} 関連技術",
        ]

    def generate_ipc_suggestions(
        self,
        query: str,
        existing_patents: List[Patent] = None,
    ) -> List[str]:
        """
        Suggest IPC codes relevant to the research query.

        Args:
            query: Research query
            existing_patents: Already found patents for context

        Returns:
            List of suggested IPC codes
        """
        existing_ipc = set()
        if existing_patents:
            for p in existing_patents:
                for ipc in p.ipc_classifications:
                    existing_ipc.add(ipc.full_code)

        existing_text = ", ".join(sorted(existing_ipc)) if existing_ipc else "なし"

        prompt = f"""以下の技術テーマに関連するIPC（国際特許分類）コードを提案してください。

【テーマ】{query}
【既知のIPC】{existing_text}

以下のJSON形式で回答してください：
[
    {{"code": "H01L21/027", "description": "半導体装置の製造方法"}},
    {{"code": "G06F3/041", "description": "タッチパネル"}}
]"""

        try:
            response = self.llm.generate(prompt)
            content = response.content
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                suggestions = json.loads(content[start:end])
                return [s["code"] for s in suggestions if "code" in s]
        except Exception:
            pass

        return list(existing_ipc)
