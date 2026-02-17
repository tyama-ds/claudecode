"""
Patent examination document search client.

Searches for patent office examination records including:
- Office actions (拒絶理由通知)
- Examination history (審査経過情報)
- Amendments and responses
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

from deep_research_tool.search.duckduckgo import DuckDuckGoSearch

logger = logging.getLogger(__name__)


@dataclass
class ExaminationDocument:
    """A patent examination document."""

    patent_number: str
    document_type: str = ""  # "office_action", "response", "amendment", "decision"
    title: str = ""
    url: str = ""
    date: str = ""
    content_excerpt: str = ""
    source: str = ""  # "jplatpat", "web"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patent_number": self.patent_number,
            "document_type": self.document_type,
            "title": self.title,
            "url": self.url,
            "date": self.date,
            "content_excerpt": self.content_excerpt,
            "source": self.source,
        }


class ExaminationSearchClient:
    """Search client for patent examination documents."""

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

    def search_examination_history(
        self,
        patent_number: str,
        max_results: int = None,
    ) -> List[ExaminationDocument]:
        """
        Search for examination history of a specific patent.

        Args:
            patent_number: Patent number to search for
            max_results: Maximum results to return

        Returns:
            List of ExaminationDocument objects
        """
        max_res = max_results or self.max_results
        documents = []

        # Search J-PlatPat examination documents
        queries = [
            f"site:j-platpat.inpit.go.jp {patent_number} 審査経過",
            f"{patent_number} 拒絶理由通知",
            f"{patent_number} 審査 特許庁",
        ]

        seen_urls = set()

        for query in queries:
            try:
                results = self.ddg.search(query)
                for result in results[:max_res]:
                    url = result.url if hasattr(result, "url") else ""
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)

                    title = result.title if hasattr(result, "title") else ""
                    snippet = result.snippet if hasattr(result, "snippet") else ""

                    doc_type = self._classify_document_type(title + " " + snippet)

                    documents.append(
                        ExaminationDocument(
                            patent_number=patent_number,
                            document_type=doc_type,
                            title=title,
                            url=url,
                            content_excerpt=snippet,
                            source="web",
                        )
                    )
            except Exception as e:
                logger.warning(
                    f"[ExaminationSearch] Search failed for query '{query}': {e}"
                )

            if len(documents) >= max_res:
                break

        logger.info(
            f"[ExaminationSearch] Found {len(documents)} examination documents "
            f"for {patent_number}"
        )
        return documents[:max_res]

    def search_examination_batch(
        self,
        patent_numbers: List[str],
        max_per_patent: int = 3,
    ) -> Dict[str, List[ExaminationDocument]]:
        """
        Search examination history for multiple patents.

        Args:
            patent_numbers: List of patent numbers
            max_per_patent: Maximum documents per patent

        Returns:
            Dict mapping patent number to list of ExaminationDocument
        """
        results = {}
        for patent_number in patent_numbers:
            docs = self.search_examination_history(
                patent_number, max_results=max_per_patent
            )
            results[patent_number] = docs
        return results

    def fetch_document_content(
        self,
        document: ExaminationDocument,
    ) -> ExaminationDocument:
        """
        Fetch the full content of an examination document.

        Args:
            document: ExaminationDocument with URL

        Returns:
            Same document with content_excerpt populated
        """
        if not document.url:
            return document

        try:
            page = self.ddg.get_page_content(document.url)
            if page and page.text_content:
                document.content_excerpt = page.text_content[:5000]
        except Exception as e:
            logger.warning(
                f"[ExaminationSearch] Content fetch failed for {document.url}: {e}"
            )

        return document

    def _classify_document_type(self, text: str) -> str:
        """Classify the type of examination document from text."""
        text_lower = text.lower()

        if any(term in text for term in ["拒絶理由", "拒絶査定", "rejection"]):
            return "office_action"
        elif any(term in text for term in ["意見書", "response", "補正"]):
            return "response"
        elif any(term in text for term in ["補正書", "amendment"]):
            return "amendment"
        elif any(term in text for term in ["審決", "decision", "特許査定"]):
            return "decision"
        elif any(term in text for term in ["審査経過", "examination", "file wrapper"]):
            return "examination_history"

        return "other"
