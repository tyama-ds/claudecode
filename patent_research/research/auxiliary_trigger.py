"""
Auxiliary search trigger logic.

Analyzes patent content using LLM to decide when secondary (academic/technical)
and tertiary (business/market) searches should be triggered.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

from ..models.patent import Patent
from ..config import AuxiliarySearchConfig

logger = logging.getLogger(__name__)


@dataclass
class TriggerItem:
    """A single trigger item with search context."""

    term: str
    context: str
    search_query: str
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "context": self.context,
            "search_query": self.search_query,
            "confidence": self.confidence,
        }


@dataclass
class TriggerResult:
    """Result of auxiliary trigger analysis for a patent."""

    patent_number: str = ""
    technical_terms: List[TriggerItem] = field(default_factory=list)
    academic_references: List[TriggerItem] = field(default_factory=list)
    business_indicators: List[TriggerItem] = field(default_factory=list)
    standards_references: List[TriggerItem] = field(default_factory=list)

    @property
    def needs_academic_search(self) -> bool:
        """Whether academic paper search should be triggered."""
        return len(self.technical_terms) > 0 or len(self.academic_references) > 0

    @property
    def needs_business_search(self) -> bool:
        """Whether business evidence search should be triggered."""
        return len(self.business_indicators) > 0

    @property
    def needs_standards_search(self) -> bool:
        """Whether standards/specification search should be triggered."""
        return len(self.standards_references) > 0

    def get_academic_queries(self) -> List[str]:
        """Get all academic search queries from triggers."""
        queries = []
        for item in self.technical_terms + self.academic_references:
            if item.search_query:
                queries.append(item.search_query)
        return queries

    def get_business_queries(self) -> List[str]:
        """Get all business search queries from triggers."""
        return [item.search_query for item in self.business_indicators if item.search_query]

    def get_standards_queries(self) -> List[str]:
        """Get all standards search queries from triggers."""
        return [item.search_query for item in self.standards_references if item.search_query]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patent_number": self.patent_number,
            "technical_terms": [t.to_dict() for t in self.technical_terms],
            "academic_references": [a.to_dict() for a in self.academic_references],
            "business_indicators": [b.to_dict() for b in self.business_indicators],
            "standards_references": [s.to_dict() for s in self.standards_references],
        }


class AuxiliaryTrigger:
    """
    Analyzes patent content to decide when auxiliary searches are needed.

    Uses LLM to identify:
    - Technical terms that need academic paper corroboration
    - Academic concepts referenced in the patent
    - Business/market indicators for evidence gathering
    - Standards and specifications referenced
    """

    def __init__(
        self,
        llm_client,
        config: AuxiliarySearchConfig,
        language: str = "ja",
    ):
        self.llm = llm_client
        self.config = config
        self.language = language

    def analyze_patent(self, patent: Patent) -> TriggerResult:
        """
        Analyze a single patent and return trigger decisions.

        Args:
            patent: Patent to analyze

        Returns:
            TriggerResult with trigger decisions for each category
        """
        result = TriggerResult(patent_number=patent.patent_number)

        # Build patent content for analysis
        claims_text = "\n".join(
            f"請求項{c.claim_number}: {c.claim_text[:300]}"
            for c in patent.claims[:5]
        )

        prompt = f"""以下の特許情報を分析し、補助検索が必要な要素を特定してください。

【特許番号】{patent.patent_number}
【タイトル】{patent.title}
【要約】{patent.abstract[:1000]}
【請求項】
{claims_text or '（請求項なし）'}

以下を特定してください：
1. 技術用語: 学術論文や技術資料で裏付けが必要な技術的概念
2. 学術参照: 特許が参照している可能性のある学術的な概念・理論
3. ビジネス指標: 市場規模、売上、シェアなどのビジネスデータが関連する記載
4. 規格参照: 参照されている業界規格・標準

以下のJSON形式で回答してください：
{{
    "technical_terms": [
        {{"term": "技術用語", "context": "特許内での文脈", "search_query": "CiNii/J-STAGE用の検索クエリ", "confidence": 0.8}}
    ],
    "academic_references": [
        {{"term": "学術概念", "context": "参照理由", "search_query": "論文検索クエリ", "confidence": 0.7}}
    ],
    "business_indicators": [
        {{"term": "市場データ参照", "context": "特許内での文脈", "search_query": "ビジネス検索クエリ", "confidence": 0.6}}
    ],
    "standards_references": [
        {{"term": "規格名", "context": "文脈", "search_query": "規格検索クエリ", "confidence": 0.8}}
    ]
}}

もし該当する要素がない場合は空のリストを返してください。"""

        try:
            response = self.llm.generate(prompt)
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])

                # Parse technical terms
                for item in data.get("technical_terms", []):
                    confidence = item.get("confidence", 0.5)
                    if confidence >= self.config.technical_term_threshold:
                        result.technical_terms.append(TriggerItem(
                            term=item.get("term", ""),
                            context=item.get("context", ""),
                            search_query=item.get("search_query", ""),
                            confidence=confidence,
                        ))

                # Parse academic references
                for item in data.get("academic_references", []):
                    result.academic_references.append(TriggerItem(
                        term=item.get("term", ""),
                        context=item.get("context", ""),
                        search_query=item.get("search_query", ""),
                        confidence=item.get("confidence", 0.5),
                    ))

                # Parse business indicators
                for item in data.get("business_indicators", []):
                    confidence = item.get("confidence", 0.5)
                    if confidence >= self.config.business_term_threshold:
                        result.business_indicators.append(TriggerItem(
                            term=item.get("term", ""),
                            context=item.get("context", ""),
                            search_query=item.get("search_query", ""),
                            confidence=confidence,
                        ))

                # Parse standards references
                for item in data.get("standards_references", []):
                    result.standards_references.append(TriggerItem(
                        term=item.get("term", ""),
                        context=item.get("context", ""),
                        search_query=item.get("search_query", ""),
                        confidence=item.get("confidence", 0.5),
                    ))

        except Exception as e:
            logger.error(
                f"[AuxiliaryTrigger] Analysis failed for {patent.patent_number}: {e}"
            )

        logger.info(
            f"[AuxiliaryTrigger] {patent.patent_number}: "
            f"tech={len(result.technical_terms)}, "
            f"academic={len(result.academic_references)}, "
            f"business={len(result.business_indicators)}, "
            f"standards={len(result.standards_references)}"
        )
        return result

    def analyze_patents_batch(self, patents: List[Patent]) -> List[TriggerResult]:
        """
        Analyze multiple patents for auxiliary triggers.

        Args:
            patents: List of patents to analyze

        Returns:
            List of TriggerResult objects
        """
        results = []
        for patent in patents:
            result = self.analyze_patent(patent)
            results.append(result)
        return results

    def aggregate_triggers(self, trigger_results: List[TriggerResult]) -> TriggerResult:
        """
        Aggregate trigger results from multiple patents into a combined result.

        Deduplicates queries and merges all trigger items.

        Args:
            trigger_results: List of individual trigger results

        Returns:
            Combined TriggerResult with deduplicated queries
        """
        combined = TriggerResult(patent_number="aggregated")
        seen_queries = set()

        for result in trigger_results:
            for item in result.technical_terms:
                if item.search_query not in seen_queries:
                    combined.technical_terms.append(item)
                    seen_queries.add(item.search_query)

            for item in result.academic_references:
                if item.search_query not in seen_queries:
                    combined.academic_references.append(item)
                    seen_queries.add(item.search_query)

            for item in result.business_indicators:
                if item.search_query not in seen_queries:
                    combined.business_indicators.append(item)
                    seen_queries.add(item.search_query)

            for item in result.standards_references:
                if item.search_query not in seen_queries:
                    combined.standards_references.append(item)
                    seen_queries.add(item.search_query)

        return combined
