"""
J-PlatPat (Japanese Patent Office) search client.

Uses DuckDuckGo site: search for discovery, with optional Selenium
for JavaScript-rendered pages.
"""

import re
import logging
from typing import List, Optional, Tuple

from deep_research_tool.search.duckduckgo import DuckDuckGoSearch

from .patent_search import PatentSearchClient
from ..models.patent import Patent, PatentClaim, IPCClassification
from ..models.search_result import PatentSearchResult

logger = logging.getLogger(__name__)


class JPlatPatClient(PatentSearchClient):
    """J-PlatPat search client for Japanese patent database."""

    JPLATPAT_DOMAIN = "j-platpat.inpit.go.jp"

    def __init__(
        self,
        language: str = "ja",
        max_results: int = 10,
        proxies: dict = None,
        verify_ssl: bool = True,
        use_selenium: bool = False,
    ):
        super().__init__(language=language)
        self.ddg = DuckDuckGoSearch(
            max_results=max_results,
            region="jp-jp",  # Japanese region for better JP patent results
            proxies=proxies,
            verify_ssl=verify_ssl,
        )
        self.use_selenium = use_selenium
        self._selenium_client = None

    def _get_selenium_client(self):
        """Lazily initialize Selenium client if needed."""
        if self._selenium_client is None and self.use_selenium:
            try:
                from deep_research_tool.search.selenium_browser import SeleniumSearch
                self._selenium_client = SeleniumSearch(
                    headless=True,
                    browser="chrome",
                )
            except ImportError:
                logger.warning(
                    "[JPlatPat] Selenium not available, falling back to DuckDuckGo"
                )
        return self._selenium_client

    def search_patents(
        self,
        query: str,
        ipc_codes: List[str] = None,
        jurisdictions: List[str] = None,
        date_range: Tuple[str, str] = None,
        max_results: int = 20,
    ) -> List[PatentSearchResult]:
        """Search J-PlatPat via DuckDuckGo site: query."""
        # Build search query targeted at J-PlatPat
        search_query = f"site:{self.JPLATPAT_DOMAIN} {query}"

        if ipc_codes:
            search_query += f" {' '.join(ipc_codes)}"

        # Also search for Japanese patent results generally
        # (J-PlatPat pages may not always appear via site: search)
        alt_query = f"特許 {query} J-PlatPat"

        logger.info(f"[JPlatPat] Searching: {search_query}")

        patent_results = []

        # Primary search
        try:
            results = self.ddg.search(search_query)
            for result in results[:max_results]:
                parsed = self._parse_search_result(result)
                if parsed:
                    patent_results.append(parsed)
        except Exception as e:
            logger.error(f"[JPlatPat] Primary search failed: {e}")

        # Alternative search for more results
        if len(patent_results) < max_results // 2:
            try:
                alt_results = self.ddg.search(alt_query)
                seen_numbers = {r.patent_number for r in patent_results}
                for result in alt_results[:max_results]:
                    parsed = self._parse_search_result(result)
                    if parsed and parsed.patent_number not in seen_numbers:
                        patent_results.append(parsed)
                        seen_numbers.add(parsed.patent_number)
            except Exception as e:
                logger.error(f"[JPlatPat] Alt search failed: {e}")

        logger.info(f"[JPlatPat] Found {len(patent_results)} patents")
        return patent_results[:max_results]

    def get_patent_detail(self, patent_number: str) -> Optional[Patent]:
        """Fetch patent details from J-PlatPat."""
        # Try direct DuckDuckGo search for the specific patent
        search_query = f"site:{self.JPLATPAT_DOMAIN} {patent_number}"

        try:
            results = self.ddg.search(search_query)
            if not results:
                # Fallback: search without site restriction
                results = self.ddg.search(f"J-PlatPat {patent_number} 特許")

            for result in results[:3]:
                url = result.url if hasattr(result, "url") else ""
                if not url:
                    continue

                try:
                    page = self.ddg.get_page_content(url)
                    if page and page.text_content and len(page.text_content) > 100:
                        return self._parse_patent_page(
                            url=url,
                            content=page.text_content,
                            patent_number=patent_number,
                        )
                except Exception as e:
                    logger.warning(f"[JPlatPat] Page fetch failed: {e}")
                    continue

        except Exception as e:
            logger.error(
                f"[JPlatPat] Detail fetch failed for {patent_number}: {e}"
            )

        return None

    def _parse_search_result(self, result) -> Optional[PatentSearchResult]:
        """Parse a search result into a PatentSearchResult."""
        url = result.url if hasattr(result, "url") else ""
        title = result.title if hasattr(result, "title") else ""
        snippet = result.snippet if hasattr(result, "snippet") else ""

        # Try to extract Japanese patent number from title or snippet
        patent_number = self._extract_jp_patent_number(title + " " + snippet + " " + url)
        if not patent_number:
            return None

        return PatentSearchResult(
            patent_number=patent_number,
            title=title,
            snippet=snippet,
            url=url,
            source_database="jplatpat",
            jurisdiction="JP",
        )

    def _parse_patent_page(
        self,
        url: str,
        content: str,
        patent_number: str,
    ) -> Patent:
        """Parse J-PlatPat page content into a Patent object."""
        patent = Patent(
            patent_number=patent_number,
            title="",
            source_url=url,
            source_database="jplatpat",
            source_databases=["jplatpat"],
            jurisdiction="JP",
        )

        # Extract title
        title_match = re.search(
            r"(?:【発明の名称】|発明の名称)\s*(.*?)(?:\n|$)",
            content,
        )
        if title_match:
            patent.title = title_match.group(1).strip()
        elif content.strip():
            # Use first non-empty line as title
            for line in content.strip().split("\n")[:10]:
                line = line.strip()
                if line and len(line) > 3:
                    patent.title = line
                    break

        # Extract abstract
        abstract_match = re.search(
            r"(?:【要約】|要約)\s*(.*?)(?:\n【|\n\n)",
            content,
            re.DOTALL,
        )
        if abstract_match:
            patent.abstract = abstract_match.group(1).strip()[:2000]

        # Extract claims
        patent.claims = self._extract_jp_claims(content)

        # Extract applicant
        applicant_match = re.search(
            r"(?:【出願人】|出願人)\s*(.*?)(?:\n|$)",
            content,
        )
        if applicant_match:
            patent.applicant = applicant_match.group(1).strip()

        # Extract inventor
        inventor_match = re.search(
            r"(?:【発明者】|発明者)\s*(.*?)(?:\n|$)",
            content,
        )
        if inventor_match:
            patent.inventor = inventor_match.group(1).strip()

        # Extract IPC codes
        ipc_matches = re.findall(
            r"([A-H]\d{2}[A-Z]\d+/\d+)",
            content,
        )
        for ipc_code in set(ipc_matches):
            patent.ipc_classifications.append(IPCClassification(full_code=ipc_code))

        # Extract filing date
        filing_match = re.search(
            r"(?:【出願日】|出願日)[:\s]*([\d\.\-/年月日]+)",
            content,
        )
        if filing_match:
            patent.filing_date = filing_match.group(1).strip()

        patent.full_text = content[:50000]

        return patent

    def _extract_jp_claims(self, content: str) -> List[PatentClaim]:
        """Extract claims from Japanese patent text."""
        claims = []

        # Find claims section
        claims_section = re.search(
            r"(?:【特許請求の範囲】|請求の範囲)(.*?)(?:【発明の詳細な説明】|【明細書】|\Z)",
            content,
            re.DOTALL,
        )
        if not claims_section:
            return claims

        claims_text = claims_section.group(1)

        # Parse 【請求項N】 format
        claim_matches = re.findall(
            r"【請求項(\d+)】\s*(.*?)(?=【請求項\d+】|\Z)",
            claims_text,
            re.DOTALL,
        )

        for num_str, text in claim_matches:
            claim_num = int(num_str)
            claim_text = text.strip()

            # Check if dependent
            dep_match = re.search(r"請求項(\d+)", claim_text)
            is_dependent = dep_match is not None and dep_match.group(1) != num_str
            depends_on = int(dep_match.group(1)) if is_dependent else None

            claims.append(
                PatentClaim(
                    claim_number=claim_num,
                    claim_text=claim_text,
                    claim_type="dependent" if is_dependent else "independent",
                    depends_on=depends_on,
                )
            )

        return claims

    def _extract_jp_patent_number(self, text: str) -> Optional[str]:
        """Extract Japanese patent number from text."""
        # Patterns for Japanese patent numbers
        patterns = [
            r"(JP\d{4}[\-]?\d{6}[A-Z]?\d*)",  # JP2024-123456A
            r"(特開\d{4}[\-]?\d{6})",  # 特開2024-123456
            r"(特許第?\d{7}号?)",  # 特許第1234567号
            r"(WO\d{4}/\d{6})",  # WO2024/123456
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return None

    @property
    def source_name(self) -> str:
        return "jplatpat"
