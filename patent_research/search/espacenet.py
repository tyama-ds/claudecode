"""
Espacenet (European Patent Office) search client.

Uses the OPS (Open Patent Services) REST API for structured data access.
Fallback to DuckDuckGo site: search when API is not configured.
"""

import re
import logging
import time
from typing import List, Optional, Tuple, Dict, Any

import requests
from deep_research_tool.search.duckduckgo import DuckDuckGoSearch

from .patent_search import PatentSearchClient
from ..models.patent import Patent, PatentClaim, IPCClassification, PatentFamily
from ..models.search_result import PatentSearchResult

logger = logging.getLogger(__name__)


class EspacenetClient(PatentSearchClient):
    """Espacenet search client using OPS API with DuckDuckGo fallback."""

    OPS_BASE_URL = "https://ops.epo.org/3.2/rest-services"
    ESPACENET_DOMAIN = "worldwide.espacenet.com"

    def __init__(
        self,
        language: str = "ja",
        consumer_key: str = None,
        consumer_secret: str = None,
        max_results: int = 10,
        proxies: dict = None,
        verify_ssl: bool = True,
    ):
        super().__init__(language=language)
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self._access_token = None
        self._token_expires = 0
        self.proxies = proxies
        self.verify_ssl = verify_ssl

        # DuckDuckGo fallback
        self.ddg = DuckDuckGoSearch(
            max_results=max_results,
            region="wt-wt",
            proxies=proxies,
            verify_ssl=verify_ssl,
        )

    @property
    def _has_api_credentials(self) -> bool:
        return bool(self.consumer_key and self.consumer_secret)

    def _get_access_token(self) -> Optional[str]:
        """Get OAuth access token from OPS API."""
        if self._access_token and time.time() < self._token_expires:
            return self._access_token

        if not self._has_api_credentials:
            return None

        try:
            response = requests.post(
                "https://ops.epo.org/3.2/auth/accesstoken",
                data={"grant_type": "client_credentials"},
                auth=(self.consumer_key, self.consumer_secret),
                proxies=self.proxies,
                verify=self.verify_ssl,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            self._access_token = data.get("access_token")
            # Token typically valid for 20 minutes
            self._token_expires = time.time() + int(data.get("expires_in", 1200)) - 60
            return self._access_token
        except Exception as e:
            logger.error(f"[Espacenet] Token request failed: {e}")
            return None

    def _api_request(self, endpoint: str, params: dict = None) -> Optional[Dict]:
        """Make an authenticated request to the OPS API."""
        token = self._get_access_token()
        if not token:
            return None

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

        url = f"{self.OPS_BASE_URL}/{endpoint}"

        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                proxies=self.proxies,
                verify=self.verify_ssl,
                timeout=30,
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                logger.warning("[Espacenet] API rate limit or access denied")
                return None
            else:
                logger.warning(
                    f"[Espacenet] API returned status {response.status_code}"
                )
                return None
        except Exception as e:
            logger.error(f"[Espacenet] API request failed: {e}")
            return None

    def search_patents(
        self,
        query: str,
        ipc_codes: List[str] = None,
        jurisdictions: List[str] = None,
        date_range: Tuple[str, str] = None,
        max_results: int = 20,
    ) -> List[PatentSearchResult]:
        """Search patents via OPS API or DuckDuckGo fallback."""
        # Try OPS API first
        if self._has_api_credentials:
            results = self._search_via_ops(
                query, ipc_codes, jurisdictions, date_range, max_results
            )
            if results:
                return results

        # Fallback to DuckDuckGo site: search
        return self._search_via_ddg(query, ipc_codes, max_results)

    def _search_via_ops(
        self,
        query: str,
        ipc_codes: List[str] = None,
        jurisdictions: List[str] = None,
        date_range: Tuple[str, str] = None,
        max_results: int = 20,
    ) -> List[PatentSearchResult]:
        """Search via OPS API."""
        # Build CQL query
        cql_parts = [f'txt="{query}"']

        if ipc_codes:
            ipc_query = " OR ".join(f'ic="{code}"' for code in ipc_codes)
            cql_parts.append(f"({ipc_query})")

        if date_range and date_range[0]:
            cql_parts.append(f'pd>="{date_range[0].replace("-", "")}"')
        if date_range and len(date_range) > 1 and date_range[1]:
            cql_parts.append(f'pd<="{date_range[1].replace("-", "")}"')

        cql = " AND ".join(cql_parts)

        data = self._api_request(
            f"published-data/search/biblio",
            params={"q": cql, "Range": f"1-{min(max_results, 100)}"},
        )

        if not data:
            return []

        results = []
        try:
            search_results = (
                data.get("ops:world-patent-data", {})
                .get("ops:biblio-search", {})
                .get("ops:search-result", {})
                .get("ops:publication-reference", [])
            )
            if isinstance(search_results, dict):
                search_results = [search_results]

            for item in search_results[:max_results]:
                doc_id = item.get("document-id", {})
                if isinstance(doc_id, list):
                    doc_id = doc_id[0]

                country = doc_id.get("country", {}).get("$", "")
                doc_number = doc_id.get("doc-number", {}).get("$", "")
                kind = doc_id.get("kind", {}).get("$", "")
                patent_number = f"{country}{doc_number}{kind}"

                results.append(
                    PatentSearchResult(
                        patent_number=patent_number,
                        title="",  # OPS search doesn't return title inline
                        url=f"https://{self.ESPACENET_DOMAIN}/patent/search?q={patent_number}",
                        source_database="espacenet",
                        jurisdiction=country,
                    )
                )
        except (KeyError, TypeError) as e:
            logger.warning(f"[Espacenet] Failed to parse OPS results: {e}")

        return results

    def _search_via_ddg(
        self,
        query: str,
        ipc_codes: List[str] = None,
        max_results: int = 20,
    ) -> List[PatentSearchResult]:
        """Fallback search via DuckDuckGo."""
        search_query = f"site:{self.ESPACENET_DOMAIN} {query}"
        if ipc_codes:
            search_query += f" {' '.join(ipc_codes)}"

        logger.info(f"[Espacenet] DuckDuckGo fallback: {search_query}")

        try:
            results = self.ddg.search(search_query)
        except Exception as e:
            logger.error(f"[Espacenet] DuckDuckGo search failed: {e}")
            return []

        patent_results = []
        for result in results[:max_results]:
            url = result.url if hasattr(result, "url") else ""
            title = result.title if hasattr(result, "title") else ""
            snippet = result.snippet if hasattr(result, "snippet") else ""

            patent_number = self._extract_patent_number(title + " " + snippet + " " + url)
            if patent_number:
                patent_results.append(
                    PatentSearchResult(
                        patent_number=patent_number,
                        title=title,
                        snippet=snippet,
                        url=url,
                        source_database="espacenet",
                        jurisdiction=patent_number[:2] if len(patent_number) >= 2 else "",
                    )
                )

        return patent_results

    def get_patent_detail(self, patent_number: str) -> Optional[Patent]:
        """Get patent detail via OPS API or page scraping."""
        # Try OPS API first
        if self._has_api_credentials:
            patent = self._get_detail_via_ops(patent_number)
            if patent:
                return patent

        # Fallback to page scraping
        return self._get_detail_via_scraping(patent_number)

    def _get_detail_via_ops(self, patent_number: str) -> Optional[Patent]:
        """Get patent details via OPS API."""
        clean = re.sub(r"[\s\-,]", "", patent_number)

        # Biblio data
        data = self._api_request(f"published-data/publication/docdb/{clean}/biblio")
        if not data:
            return None

        patent = Patent(
            patent_number=patent_number,
            title="",
            source_database="espacenet",
            source_databases=["espacenet"],
        )

        try:
            exchange = (
                data.get("ops:world-patent-data", {})
                .get("exchange-documents", {})
                .get("exchange-document", {})
            )
            if isinstance(exchange, list):
                exchange = exchange[0]

            # Extract bibliographic data
            biblio = exchange.get("bibliographic-data", {})

            # Title
            titles = biblio.get("invention-title", [])
            if isinstance(titles, dict):
                titles = [titles]
            for t in titles:
                lang = t.get("@lang", "")
                if lang == "en" or not patent.title:
                    patent.title = t.get("$", "")

            # Abstract
            abstracts = exchange.get("abstract", [])
            if isinstance(abstracts, dict):
                abstracts = [abstracts]
            for a in abstracts:
                text = a.get("p", {})
                if isinstance(text, dict):
                    text = text.get("$", "")
                elif isinstance(text, list):
                    text = " ".join(p.get("$", "") for p in text if isinstance(p, dict))
                if text:
                    patent.abstract = str(text)[:2000]
                    break

            # IPC classifications
            ipc_data = biblio.get("classifications-ipcr", {}).get("classification-ipcr", [])
            if isinstance(ipc_data, dict):
                ipc_data = [ipc_data]
            for ipc in ipc_data:
                text = ipc.get("text", {}).get("$", "")
                if text:
                    patent.ipc_classifications.append(
                        IPCClassification(full_code=text.strip())
                    )

            # Applicant
            applicants = (
                biblio.get("parties", {})
                .get("applicants", {})
                .get("applicant", [])
            )
            if isinstance(applicants, dict):
                applicants = [applicants]
            names = []
            for app in applicants:
                name = app.get("applicant-name", {}).get("name", {}).get("$", "")
                if name:
                    names.append(name)
            patent.applicant = "; ".join(names)

            # Filing date
            app_ref = biblio.get("application-reference", {})
            doc_id = app_ref.get("document-id", {})
            if isinstance(doc_id, list):
                doc_id = doc_id[0]
            patent.filing_date = doc_id.get("date", {}).get("$", "")

            # Publication date
            pub_ref = biblio.get("publication-reference", {})
            pub_doc_id = pub_ref.get("document-id", {})
            if isinstance(pub_doc_id, list):
                pub_doc_id = pub_doc_id[0]
            patent.publication_date = pub_doc_id.get("date", {}).get("$", "")

            # Jurisdiction
            country = pub_doc_id.get("country", {}).get("$", "")
            patent.jurisdiction = country

        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"[Espacenet] Failed to parse OPS detail: {e}")

        return patent

    def _get_detail_via_scraping(self, patent_number: str) -> Optional[Patent]:
        """Get patent details via page scraping (fallback)."""
        url = f"https://{self.ESPACENET_DOMAIN}/patent/search?q={patent_number}"
        try:
            page = self.ddg.get_page_content(url)
            if page and page.text_content:
                patent = Patent(
                    patent_number=patent_number,
                    title="",
                    source_url=url,
                    source_database="espacenet",
                    source_databases=["espacenet"],
                    full_text=page.text_content[:50000],
                )
                # Basic parsing from page content
                lines = page.text_content.strip().split("\n")
                for line in lines[:10]:
                    line = line.strip()
                    if line and len(line) > 5:
                        patent.title = line
                        break
                return patent
        except Exception as e:
            logger.error(f"[Espacenet] Scraping failed for {patent_number}: {e}")
        return None

    def get_patent_family(self, patent_number: str) -> Optional[PatentFamily]:
        """Get patent family via OPS API."""
        if not self._has_api_credentials:
            return None

        clean = re.sub(r"[\s\-,]", "", patent_number)
        data = self._api_request(f"family/publication/docdb/{clean}/biblio")
        if not data:
            return None

        family = PatentFamily(family_id=clean, members=[])

        try:
            family_members = (
                data.get("ops:world-patent-data", {})
                .get("ops:patent-family", {})
                .get("ops:family-member", [])
            )
            if isinstance(family_members, dict):
                family_members = [family_members]

            jurisdictions = set()
            for member in family_members:
                pub_ref = member.get("publication-reference", {})
                doc_id = pub_ref.get("document-id", {})
                if isinstance(doc_id, list):
                    doc_id = doc_id[0]

                country = doc_id.get("country", {}).get("$", "")
                doc_number = doc_id.get("doc-number", {}).get("$", "")
                kind = doc_id.get("kind", {}).get("$", "")
                member_number = f"{country}{doc_number}{kind}"

                family.members.append(
                    Patent(
                        patent_number=member_number,
                        title="",
                        jurisdiction=country,
                        source_database="espacenet",
                    )
                )
                jurisdictions.add(country)

            family.jurisdictions = sorted(jurisdictions)

        except (KeyError, TypeError) as e:
            logger.warning(f"[Espacenet] Failed to parse family data: {e}")

        return family

    def _extract_patent_number(self, text: str) -> Optional[str]:
        """Extract a patent number from text."""
        patterns = [
            r"([A-Z]{2}\d{5,}[A-Z]?\d*)",
            r"(WO\d{4}/?\d{6})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    @property
    def source_name(self) -> str:
        return "espacenet"
