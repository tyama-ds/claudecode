"""
Patent search result model.

Represents a search result from patent databases before full patent
detail retrieval.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class PatentSearchResult:
    """A search result from a patent database."""

    patent_number: str
    title: str
    snippet: str = ""
    url: str = ""
    source_database: str = ""  # "google_patents", "jplatpat", "espacenet"

    # Metadata from the search result (before full detail retrieval)
    applicant: str = ""
    filing_date: str = ""
    publication_date: str = ""
    ipc_codes: list = field(default_factory=list)
    jurisdiction: str = ""

    # Relevance scoring
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patent_number": self.patent_number,
            "title": self.title,
            "snippet": self.snippet,
            "url": self.url,
            "source_database": self.source_database,
            "applicant": self.applicant,
            "filing_date": self.filing_date,
            "publication_date": self.publication_date,
            "ipc_codes": self.ipc_codes,
            "jurisdiction": self.jurisdiction,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatentSearchResult":
        return cls(**data)
