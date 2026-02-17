"""
Google Patents search client.

Uses DuckDuckGo site: search to find patents on Google Patents,
then scrapes patent pages for structured data.
"""

import re
import logging
from typing import List, Optional, Tuple

from deep_research_tool.search.duckduckgo import DuckDuckGoSearch

from .patent_search import PatentSearchClient
from ..models.patent import Patent, PatentClaim, IPCClassification
from ..models.search_result import PatentSearchResult

logger = logging.getLogger(__name__)


class GooglePatentsClient(PatentSearchClient):
    """Google Patents search client using DuckDuckGo site: queries."""

    GOOGLE_PATENTS_DOMAIN = "patents.google.com"

    def __init__(
        self,
        language: str = "ja",
        max_results: int = 10,
        proxies: dict = None,
        verify_ssl: bool = True,
    ):
        super().__init__(language=language)
        self.ddg = DuckDuckGoSearch(
            max_results=max_results,
            region="wt-wt",
            proxies=proxies,
            verify_ssl=verify_ssl,
        )

    def search_patents(
        self,
        query: str,
        ipc_codes: List[str] = None,
        jurisdictions: List[str] = None,
        date_range: Tuple[str, str] = None,
        max_results: int = 20,
    ) -> List[PatentSearchResult]:
        """Search Google Patents via DuckDuckGo site: query."""
        # Build search query
        search_query = f"site:{self.GOOGLE_PATENTS_DOMAIN} {query}"

        if ipc_codes:
            search_query += f" {' '.join(ipc_codes)}"

        if jurisdictions:
            # Add jurisdiction hints
            jurisdiction_terms = " OR ".join(jurisdictions)
            search_query += f" ({jurisdiction_terms})"

        logger.info(f"[GooglePatents] Searching: {search_query}")

        try:
            results = self.ddg.search(search_query)
        except Exception as e:
            logger.error(f"[GooglePatents] Search failed: {e}")
            return []

        patent_results = []
        for result in results[:max_results]:
            parsed = self._parse_search_result(result)
            if parsed:
                patent_results.append(parsed)

        logger.info(f"[GooglePatents] Found {len(patent_results)} patents")
        return patent_results

    def get_patent_detail(self, patent_number: str) -> Optional[Patent]:
        """Fetch patent detail page from Google Patents."""
        # Construct direct URL
        clean_number = re.sub(r"[\s\-,]", "", patent_number)
        url = f"https://{self.GOOGLE_PATENTS_DOMAIN}/patent/{clean_number}"

        logger.info(f"[GooglePatents] Fetching detail: {url}")

        try:
            page = self.ddg.get_page_content(url)
            if not page or not page.text_content:
                return None

            return self._parse_patent_page(
                url=url,
                content=page.text_content,
                patent_number=patent_number,
            )
        except Exception as e:
            logger.error(f"[GooglePatents] Detail fetch failed for {patent_number}: {e}")
            return None

    def _parse_search_result(self, result) -> Optional[PatentSearchResult]:
        """Parse a DuckDuckGo search result into a PatentSearchResult."""
        url = result.url if hasattr(result, "url") else ""
        title = result.title if hasattr(result, "title") else ""
        snippet = result.snippet if hasattr(result, "snippet") else ""

        if self.GOOGLE_PATENTS_DOMAIN not in url:
            return None

        # Try to extract patent number from URL
        patent_number = self._extract_patent_number_from_url(url)
        if not patent_number:
            return None

        # Detect jurisdiction from patent number
        jurisdiction = self._detect_jurisdiction(patent_number)

        return PatentSearchResult(
            patent_number=patent_number,
            title=title,
            snippet=snippet,
            url=url,
            source_database="google_patents",
            jurisdiction=jurisdiction,
        )

    def _parse_patent_page(
        self,
        url: str,
        content: str,
        patent_number: str,
    ) -> Patent:
        """Parse Google Patents page content into a Patent object."""
        patent = Patent(
            patent_number=patent_number,
            title="",
            source_url=url,
            source_database="google_patents",
            source_databases=["google_patents"],
        )

        # Extract title (usually the first significant line)
        lines = content.strip().split("\n")
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) > 5 and not line.startswith("http"):
                patent.title = line
                break

        # Extract abstract
        abstract_match = re.search(
            r"(?:Abstract|要約|概要)\s*[:\n](.*?)(?:\n\n|\nClaims|\n請求項)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if abstract_match:
            patent.abstract = abstract_match.group(1).strip()[:2000]

        # Extract claims
        patent.claims = self._extract_claims(content)

        # Extract IPC codes
        ipc_matches = re.findall(
            r"([A-H]\d{2}[A-Z]\d+/\d+)", content
        )
        for ipc_code in set(ipc_matches):
            patent.ipc_classifications.append(IPCClassification(full_code=ipc_code))

        # Extract applicant/inventor
        applicant_match = re.search(
            r"(?:Applicant|出願人|Current Assignee)[:\s]*(.*?)(?:\n|$)",
            content,
            re.IGNORECASE,
        )
        if applicant_match:
            patent.applicant = applicant_match.group(1).strip()

        inventor_match = re.search(
            r"(?:Inventor|発明者)[:\s]*(.*?)(?:\n|$)",
            content,
            re.IGNORECASE,
        )
        if inventor_match:
            patent.inventor = inventor_match.group(1).strip()

        # Extract dates
        filing_match = re.search(
            r"(?:Filing date|出願日)[:\s]*([\d\-/]+)",
            content,
            re.IGNORECASE,
        )
        if filing_match:
            patent.filing_date = filing_match.group(1).strip()

        pub_match = re.search(
            r"(?:Publication date|公開日|公表日)[:\s]*([\d\-/]+)",
            content,
            re.IGNORECASE,
        )
        if pub_match:
            patent.publication_date = pub_match.group(1).strip()

        patent.jurisdiction = self._detect_jurisdiction(patent_number)
        patent.full_text = content[:50000]

        return patent

    def _extract_claims(self, content: str) -> List[PatentClaim]:
        """Extract patent claims from page content."""
        claims = []

        # Try to find claims section
        claims_section = re.search(
            r"(?:Claims|請求項)\s*(?:\(\d+\))?\s*\n(.*?)(?:\nDescription|\n明細書|\n$)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if not claims_section:
            return claims

        claims_text = claims_section.group(1)

        # Parse individual claims
        # Pattern for numbered claims: "1." or "【請求項1】"
        claim_patterns = [
            re.compile(r"(\d+)\.\s+(.*?)(?=\n\d+\.\s|\Z)", re.DOTALL),
            re.compile(r"【請求項(\d+)】\s*(.*?)(?=【請求項\d+】|\Z)", re.DOTALL),
        ]

        for pattern in claim_patterns:
            matches = pattern.findall(claims_text)
            if matches:
                for num_str, text in matches:
                    claim_num = int(num_str)
                    claim_text = text.strip()

                    # Determine if dependent
                    is_dependent = bool(
                        re.search(
                            r"(?:claim|請求項)\s*(\d+)",
                            claim_text,
                            re.IGNORECASE,
                        )
                    )

                    depends_on = None
                    if is_dependent:
                        dep_match = re.search(
                            r"(?:claim|請求項)\s*(\d+)",
                            claim_text,
                            re.IGNORECASE,
                        )
                        if dep_match:
                            depends_on = int(dep_match.group(1))

                    claims.append(
                        PatentClaim(
                            claim_number=claim_num,
                            claim_text=claim_text,
                            claim_type="dependent" if is_dependent else "independent",
                            depends_on=depends_on,
                        )
                    )
                break  # Use the first pattern that matches

        return claims

    def _extract_patent_number_from_url(self, url: str) -> Optional[str]:
        """Extract patent number from a Google Patents URL."""
        # URLs like: https://patents.google.com/patent/US11234567B2
        match = re.search(
            r"patents\.google\.com/patent/([A-Z]{2}\d+[A-Z]?\d*)",
            url,
        )
        if match:
            return match.group(1)
        return None

    def _detect_jurisdiction(self, patent_number: str) -> str:
        """Detect jurisdiction from patent number prefix."""
        number = patent_number.strip().upper()
        if number.startswith("JP"):
            return "JP"
        elif number.startswith("US"):
            return "US"
        elif number.startswith("EP"):
            return "EP"
        elif number.startswith("WO"):
            return "WO"
        elif number.startswith("CN"):
            return "CN"
        elif number.startswith("KR"):
            return "KR"
        elif number.startswith("DE"):
            return "DE"
        return ""

    @property
    def source_name(self) -> str:
        return "google_patents"
