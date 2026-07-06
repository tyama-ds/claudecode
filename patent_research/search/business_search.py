"""
Business evidence search client.

Searches for market data, revenue figures, market share,
and other business evidence related to patent technologies.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

from deep_research_tool.search.duckduckgo import DuckDuckGoSearch

logger = logging.getLogger(__name__)


@dataclass
class BusinessEvidence:
    """A piece of business evidence (market data, revenue, etc.)."""

    title: str
    url: str
    evidence_type: str = ""  # "market_size", "revenue", "market_share", "growth_rate", "general"
    content_excerpt: str = ""
    data_points: List[Dict[str, Any]] = field(default_factory=list)
    source_credibility: str = "unknown"  # "high", "medium", "low", "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "evidence_type": self.evidence_type,
            "content_excerpt": self.content_excerpt,
            "data_points": self.data_points,
            "source_credibility": self.source_credibility,
        }


class BusinessSearchClient:
    """Search client for business and market evidence."""

    # Keywords that indicate business relevance
    BUSINESS_KEYWORDS_JA = [
        "市場規模", "売上", "シェア", "市場シェア", "成長率",
        "市場動向", "業界分析", "レポート", "統計",
    ]
    BUSINESS_KEYWORDS_EN = [
        "market size", "revenue", "market share", "growth rate",
        "market analysis", "industry report", "statistics",
    ]

    def __init__(
        self,
        language: str = "ja",
        max_results: int = 5,
        proxies: dict = None,
        verify_ssl: bool = True,
    ):
        self.language = language
        self.max_results = max_results
        self.ddg = DuckDuckGoSearch(
            max_results=max_results,
            region="jp-jp" if language == "ja" else "wt-wt",
            proxies=proxies,
            verify_ssl=verify_ssl,
        )

    def search(
        self,
        query: str,
        evidence_types: List[str] = None,
        max_results: int = None,
    ) -> List[BusinessEvidence]:
        """
        Search for business evidence related to a query.

        Args:
            query: Search query (typically technology or market related)
            evidence_types: Types of evidence to focus on
            max_results: Maximum results to return

        Returns:
            List of BusinessEvidence objects
        """
        max_res = max_results or self.max_results
        all_evidence = []
        seen_urls = set()

        # Generate business-oriented search queries
        search_queries = self._generate_business_queries(query, evidence_types)

        for search_query in search_queries:
            try:
                results = self.ddg.search(search_query)
                for result in results[:max_res]:
                    url = result.url if hasattr(result, "url") else ""
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)

                    title = result.title if hasattr(result, "title") else ""
                    snippet = result.snippet if hasattr(result, "snippet") else ""

                    evidence = BusinessEvidence(
                        title=title,
                        url=url,
                        evidence_type=self._classify_evidence_type(title + " " + snippet),
                        content_excerpt=snippet,
                        source_credibility=self._assess_credibility(url),
                    )
                    all_evidence.append(evidence)

            except Exception as e:
                logger.warning(f"[BusinessSearch] Search failed for '{search_query}': {e}")

            if len(all_evidence) >= max_res:
                break

        logger.info(f"[BusinessSearch] Found {len(all_evidence)} business evidence items")
        return all_evidence[:max_res]

    def search_and_extract(
        self,
        query: str,
        max_results: int = None,
    ) -> List[BusinessEvidence]:
        """Search and extract content from business sources."""
        evidence_list = self.search(query, max_results=max_results)

        for evidence in evidence_list[:self.max_results]:
            if evidence.url:
                try:
                    page = self.ddg.get_page_content(evidence.url)
                    if page and page.text_content:
                        evidence.content_excerpt = page.text_content[:3000]
                except Exception as e:
                    logger.warning(
                        f"[BusinessSearch] Content fetch failed for {evidence.url}: {e}"
                    )

        return evidence_list

    def _generate_business_queries(
        self,
        query: str,
        evidence_types: List[str] = None,
    ) -> List[str]:
        """Generate business-oriented search queries."""
        queries = []

        if self.language == "ja":
            keywords = self.BUSINESS_KEYWORDS_JA
        else:
            keywords = self.BUSINESS_KEYWORDS_EN

        if evidence_types:
            type_keywords = {
                "market_size": "市場規模" if self.language == "ja" else "market size",
                "revenue": "売上 収益" if self.language == "ja" else "revenue",
                "market_share": "市場シェア" if self.language == "ja" else "market share",
                "growth_rate": "成長率 CAGR" if self.language == "ja" else "growth rate CAGR",
            }
            for et in evidence_types:
                if et in type_keywords:
                    queries.append(f"{query} {type_keywords[et]}")
        else:
            # Default: generate a mix of business queries
            queries.append(f"{query} 市場規模 {self._get_current_year()}")
            queries.append(f"{query} 市場動向 業界分析")

        # Always add a general business query
        if not queries:
            queries.append(f"{query} market analysis")

        return queries

    def _classify_evidence_type(self, text: str) -> str:
        """Classify the type of business evidence from text."""
        text_lower = text.lower()

        if any(kw in text for kw in ["市場規模", "market size"]):
            return "market_size"
        elif any(kw in text for kw in ["売上", "revenue", "収益"]):
            return "revenue"
        elif any(kw in text for kw in ["シェア", "share"]):
            return "market_share"
        elif any(kw in text for kw in ["成長率", "growth", "CAGR"]):
            return "growth_rate"

        return "general"

    def _assess_credibility(self, url: str) -> str:
        """Assess source credibility based on URL domain."""
        high_credibility = [
            "statista.com", "idc.com", "gartner.com",
            "meti.go.jp", "stat.go.jp",  # Japanese government
            "nikkei.com", "reuters.com", "bloomberg.com",
        ]
        medium_credibility = [
            "techcrunch.com", "zdnet.com",
            "itmedia.co.jp", "nikkeibp.co.jp",
        ]

        url_lower = url.lower()
        for domain in high_credibility:
            if domain in url_lower:
                return "high"
        for domain in medium_credibility:
            if domain in url_lower:
                return "medium"

        return "unknown"

    def _get_current_year(self) -> str:
        """Get current year for date-aware queries."""
        from datetime import datetime
        return str(datetime.now().year)
