"""
Academic search client for CiNii, J-STAGE, and Google Scholar.

Uses DuckDuckGo site: queries to search academic databases.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from deep_research_tool.search.duckduckgo import DuckDuckGoSearch

logger = logging.getLogger(__name__)


@dataclass
class AcademicPaper:
    """An academic paper search result."""

    title: str
    url: str
    source: str = ""  # "cinii", "jstage", "google_scholar"
    authors: str = ""
    abstract: str = ""
    publication_date: str = ""
    journal: str = ""
    content_excerpt: str = ""
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "authors": self.authors,
            "abstract": self.abstract,
            "publication_date": self.publication_date,
            "journal": self.journal,
            "content_excerpt": self.content_excerpt,
            "relevance_score": self.relevance_score,
        }


class AcademicSearchClient:
    """Search client for academic papers across CiNii, J-STAGE, and Google Scholar."""

    SITE_PREFIXES = {
        "cinii": "site:ci.nii.ac.jp",
        "jstage": "site:jstage.jst.go.jp",
        "google_scholar": "site:scholar.google.com",
    }

    def __init__(
        self,
        sources: List[str] = None,
        max_results_per_source: int = 5,
        language: str = "ja",
        proxies: dict = None,
        verify_ssl: bool = True,
    ):
        self.sources = sources or ["cinii", "jstage", "google_scholar"]
        self.max_results_per_source = max_results_per_source
        self.language = language
        self.ddg = DuckDuckGoSearch(
            max_results=max_results_per_source,
            region="jp-jp" if language == "ja" else "wt-wt",
            proxies=proxies,
            verify_ssl=verify_ssl,
        )

    def search(
        self,
        query: str,
        sources: List[str] = None,
        max_results: int = None,
    ) -> List[AcademicPaper]:
        """
        Search for academic papers across configured sources.

        Args:
            query: Search query
            sources: Override which sources to search
            max_results: Override max results per source

        Returns:
            List of AcademicPaper results from all sources
        """
        search_sources = sources or self.sources
        max_per_source = max_results or self.max_results_per_source
        all_papers = []

        for source in search_sources:
            if source not in self.SITE_PREFIXES:
                logger.warning(f"[AcademicSearch] Unknown source: {source}")
                continue

            papers = self._search_source(
                query=query,
                source=source,
                max_results=max_per_source,
            )
            all_papers.extend(papers)

        logger.info(
            f"[AcademicSearch] Found {len(all_papers)} papers "
            f"across {len(search_sources)} sources"
        )
        return all_papers

    def search_and_extract(
        self,
        query: str,
        sources: List[str] = None,
        max_results: int = None,
    ) -> List[AcademicPaper]:
        """
        Search and extract content from academic papers.

        Args:
            query: Search query
            sources: Override which sources to search
            max_results: Override max results per source

        Returns:
            List of AcademicPaper results with content excerpts
        """
        papers = self.search(query, sources, max_results)

        # Fetch content for top papers
        for paper in papers[:self.max_results_per_source]:
            if paper.url:
                try:
                    page = self.ddg.get_page_content(paper.url)
                    if page and page.text_content:
                        paper.content_excerpt = page.text_content[:3000]
                except Exception as e:
                    logger.warning(
                        f"[AcademicSearch] Content fetch failed for {paper.url}: {e}"
                    )

        return papers

    def _search_source(
        self,
        query: str,
        source: str,
        max_results: int,
    ) -> List[AcademicPaper]:
        """Search a single academic source."""
        site_prefix = self.SITE_PREFIXES.get(source, "")
        search_query = f"{site_prefix} {query}"

        logger.info(f"[AcademicSearch] Searching {source}: {search_query}")

        try:
            results = self.ddg.search(search_query)
        except Exception as e:
            logger.error(f"[AcademicSearch] Search failed for {source}: {e}")
            return []

        papers = []
        for result in results[:max_results]:
            title = result.title if hasattr(result, "title") else ""
            url = result.url if hasattr(result, "url") else ""
            snippet = result.snippet if hasattr(result, "snippet") else ""

            papers.append(
                AcademicPaper(
                    title=title,
                    url=url,
                    source=source,
                    abstract=snippet,
                    content_excerpt=snippet,
                )
            )

        return papers
