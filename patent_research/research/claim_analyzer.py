"""
Patent claim analyzer.

Parses, analyzes, and compares patent claims using LLM.
Extracts technical elements for matching and claim chart generation.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from ..models.patent import Patent, PatentClaim
from ..models.analysis import ClaimChart, ClaimChartEntry

logger = logging.getLogger(__name__)


class ClaimAnalyzer:
    """Analyze patent claims using LLM."""

    def __init__(self, llm_client, language: str = "ja"):
        self.llm = llm_client
        self.language = language

    def parse_claims_text(self, claims_text: str) -> List[PatentClaim]:
        """
        Parse raw claims text into structured PatentClaim objects using LLM.

        Args:
            claims_text: Raw claims text from a patent

        Returns:
            List of structured PatentClaim objects
        """
        prompt = f"""以下の特許請求項テキストを構造化してください。

【請求項テキスト】
{claims_text[:5000]}

以下のJSON形式で回答してください：
[
    {{
        "claim_number": 1,
        "claim_text": "請求項の全文",
        "claim_type": "independent",
        "depends_on": null,
        "technical_elements": ["技術要素1", "技術要素2"]
    }},
    {{
        "claim_number": 2,
        "claim_text": "請求項の全文",
        "claim_type": "dependent",
        "depends_on": 1,
        "technical_elements": ["技術要素1"]
    }}
]"""

        try:
            response = self.llm.generate(prompt)
            content = response.content
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                claims_data = json.loads(content[start:end])
                return [PatentClaim.from_dict(c) for c in claims_data]
        except Exception as e:
            logger.error(f"[ClaimAnalyzer] Failed to parse claims: {e}")

        return []

    def extract_technical_elements(self, patent: Patent) -> List[str]:
        """
        Extract key technical elements from a patent's claims.

        Args:
            patent: Patent object with claims

        Returns:
            List of technical element strings
        """
        if not patent.claims:
            return []

        # Focus on independent claims
        claims_text = "\n".join(
            f"請求項{c.claim_number}: {c.claim_text[:500]}"
            for c in patent.independent_claims[:5]
        )

        if not claims_text:
            claims_text = "\n".join(
                f"請求項{c.claim_number}: {c.claim_text[:500]}"
                for c in patent.claims[:5]
            )

        prompt = f"""以下の特許請求項から、主要な技術要素を抽出してください。

【特許】{patent.patent_number}: {patent.title}

【請求項】
{claims_text}

技術要素のリストをJSON形式で回答してください：
["技術要素1", "技術要素2", "技術要素3"]

技術要素とは、特許の請求項に記載された構成要件、方法ステップ、材料、装置の構成部品などです。"""

        try:
            response = self.llm.generate(prompt)
            content = response.content
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                elements = json.loads(content[start:end])
                return elements
        except Exception as e:
            logger.error(f"[ClaimAnalyzer] Failed to extract elements: {e}")

        return []

    def generate_claim_chart(
        self,
        target_patent: Patent,
        reference_patents: List[Patent],
        chart_type: str = "prior_art",
        detail_level: str = "detailed",
    ) -> ClaimChart:
        """
        Generate a claim chart comparing target patent against references.

        Args:
            target_patent: The patent being analyzed
            reference_patents: Reference patents for comparison
            chart_type: "prior_art", "product", "freedom_to_operate"
            detail_level: "summary", "detailed", "comprehensive"

        Returns:
            ClaimChart with comparison entries
        """
        # Build target claims summary
        target_claims = "\n".join(
            f"請求項{c.claim_number} ({c.claim_type}): {c.claim_text[:300]}"
            for c in target_patent.claims[:10]
        )

        # Build reference patents summary
        ref_summaries = []
        for ref in reference_patents[:5]:
            ref_claims = "\n".join(
                f"  請求項{c.claim_number}: {c.claim_text[:200]}"
                for c in ref.claims[:5]
            )
            ref_summaries.append(
                f"【{ref.patent_number}】{ref.title}\n{ref_claims}"
            )
        ref_text = "\n\n".join(ref_summaries)

        detail_instruction = {
            "summary": "各マッピングは1-2文で簡潔に",
            "detailed": "各マッピングは技術的な対応関係を具体的に記述",
            "comprehensive": "各マッピングは技術的な対応関係を詳細に分析し、差異点も含めて記述",
        }.get(detail_level, "各マッピングは技術的な対応関係を具体的に記述")

        prompt = f"""以下の対象特許と参照特許を比較し、クレームチャートを作成してください。

【対象特許】{target_patent.patent_number}: {target_patent.title}
{target_claims}

【参照特許】
{ref_text}

比較タイプ: {chart_type}
詳細度: {detail_instruction}

以下のJSON形式で回答してください：
{{
    "entries": [
        {{
            "claim_element": "対象特許の請求項要素",
            "patent_number": "参照特許番号",
            "mapping": "対応関係の説明",
            "confidence": 0.8,
            "source_excerpt": "参照特許からの引用"
        }}
    ],
    "summary": "クレームチャート全体のサマリー"
}}"""

        chart = ClaimChart(
            target_patent=target_patent.patent_number,
            comparison_type=chart_type,
        )

        try:
            response = self.llm.generate(prompt)
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                chart.summary = data.get("summary", "")
                for entry_data in data.get("entries", []):
                    chart.entries.append(ClaimChartEntry.from_dict(entry_data))
        except Exception as e:
            logger.error(f"[ClaimAnalyzer] Failed to generate claim chart: {e}")

        return chart

    def compare_claims(
        self,
        claim_a: PatentClaim,
        claim_b: PatentClaim,
        patent_a_title: str = "",
        patent_b_title: str = "",
    ) -> Dict[str, Any]:
        """
        Compare two patent claims for overlap and similarity.

        Args:
            claim_a: First claim
            claim_b: Second claim
            patent_a_title: Title of patent A for context
            patent_b_title: Title of patent B for context

        Returns:
            Dict with comparison results
        """
        prompt = f"""以下の2つの特許請求項を比較分析してください。

【請求項A】({patent_a_title})
{claim_a.claim_text[:1000]}

【請求項B】({patent_b_title})
{claim_b.claim_text[:1000]}

以下のJSON形式で回答してください：
{{
    "similarity_score": 0.7,
    "overlapping_elements": ["共通する技術要素"],
    "unique_to_a": ["Aにのみ存在する要素"],
    "unique_to_b": ["Bにのみ存在する要素"],
    "analysis": "比較分析の詳細説明"
}}"""

        try:
            response = self.llm.generate(prompt)
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except Exception as e:
            logger.error(f"[ClaimAnalyzer] Claim comparison failed: {e}")

        return {
            "similarity_score": 0.0,
            "overlapping_elements": [],
            "unique_to_a": [],
            "unique_to_b": [],
            "analysis": "比較分析に失敗しました",
        }
