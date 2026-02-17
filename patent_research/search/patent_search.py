"""
Abstract base class for patent search clients.

All patent database clients (Google Patents, J-PlatPat, Espacenet)
implement this interface.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from ..models.patent import Patent, PatentClaim, PatentFamily
from ..models.search_result import PatentSearchResult


class PatentSearchClient(ABC):
    """Abstract base class for patent database clients."""

    def __init__(self, language: str = "ja"):
        self.language = language

    @abstractmethod
    def search_patents(
        self,
        query: str,
        ipc_codes: List[str] = None,
        jurisdictions: List[str] = None,
        date_range: Tuple[str, str] = None,
        max_results: int = 20,
    ) -> List[PatentSearchResult]:
        """
        Search for patents matching the query.

        Args:
            query: Search query string
            ipc_codes: IPC classification codes to filter by
            jurisdictions: Patent jurisdictions to search (e.g., ["JP", "US"])
            date_range: Tuple of (start_date, end_date) as strings
            max_results: Maximum number of results to return

        Returns:
            List of PatentSearchResult objects
        """
        ...

    @abstractmethod
    def get_patent_detail(self, patent_number: str) -> Optional[Patent]:
        """
        Get full patent details including claims, classifications, etc.

        Args:
            patent_number: Patent number to look up

        Returns:
            Patent object with full details, or None if not found
        """
        ...

    def get_patent_claims(self, patent_number: str) -> List[PatentClaim]:
        """
        Get claims for a specific patent.

        Default implementation calls get_patent_detail and extracts claims.

        Args:
            patent_number: Patent number

        Returns:
            List of PatentClaim objects
        """
        patent = self.get_patent_detail(patent_number)
        if patent:
            return patent.claims
        return []

    def get_patent_family(self, patent_number: str) -> Optional[PatentFamily]:
        """
        Get patent family information.

        Args:
            patent_number: Patent number

        Returns:
            PatentFamily object, or None if not available
        """
        return None

    def get_citing_patents(self, patent_number: str) -> List[str]:
        """
        Get patents that cite the given patent.

        Args:
            patent_number: Patent number

        Returns:
            List of patent numbers that cite this patent
        """
        patent = self.get_patent_detail(patent_number)
        if patent:
            return patent.citing_patents
        return []

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this patent database source."""
        ...
