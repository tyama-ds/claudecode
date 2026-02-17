"""
Patent data models.

Core data structures for representing patents, claims, classifications,
and patent families.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class IPCClassification:
    """International Patent Classification code."""

    full_code: str  # e.g., "H01L21/027"
    section: str = ""  # e.g., "H" (Electricity)
    class_code: str = ""  # e.g., "H01"
    subclass: str = ""  # e.g., "H01L"
    main_group: str = ""  # e.g., "H01L21"
    subgroup: str = ""  # e.g., "H01L21/027"
    description: str = ""

    def __post_init__(self):
        """Parse full_code into components if not provided."""
        if self.full_code and not self.section:
            self._parse_code()

    def _parse_code(self) -> None:
        """Parse IPC code into components."""
        code = self.full_code.strip()
        if len(code) >= 1:
            self.section = code[0]
        if len(code) >= 3:
            self.class_code = code[:3]
        if len(code) >= 4:
            self.subclass = code[:4]
        if "/" in code:
            self.main_group = code.split("/")[0]
            self.subgroup = code

    def to_dict(self) -> Dict[str, str]:
        return {
            "full_code": self.full_code,
            "section": self.section,
            "class_code": self.class_code,
            "subclass": self.subclass,
            "main_group": self.main_group,
            "subgroup": self.subgroup,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "IPCClassification":
        return cls(**data)


@dataclass
class PatentClaim:
    """A single patent claim."""

    claim_number: int
    claim_text: str
    claim_type: str = "independent"  # "independent" or "dependent"
    depends_on: Optional[int] = None
    technical_elements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_number": self.claim_number,
            "claim_text": self.claim_text,
            "claim_type": self.claim_type,
            "depends_on": self.depends_on,
            "technical_elements": self.technical_elements,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatentClaim":
        return cls(**data)


def normalize_patent_number(patent_number: str) -> str:
    """
    Normalize patent number to a canonical format for deduplication.

    Handles variations like:
    - "JP2024-123456A" -> "JP2024123456A"
    - "JP 2024-123456 A" -> "JP2024123456A"
    - "US11,234,567B2" -> "US11234567B2"
    - "US 11,234,567 B2" -> "US11234567B2"
    """
    # Remove spaces, hyphens, commas
    normalized = patent_number.strip()
    normalized = re.sub(r"[\s,\-]", "", normalized)
    # Uppercase
    normalized = normalized.upper()
    return normalized


@dataclass
class Patent:
    """A patent document."""

    patent_number: str
    title: str
    abstract: str = ""
    applicant: str = ""
    inventor: str = ""
    filing_date: str = ""
    publication_date: str = ""
    grant_date: str = ""
    status: str = ""  # "granted", "pending", "expired"
    jurisdiction: str = ""  # "JP", "US", "EP", "WO"

    claims: List[PatentClaim] = field(default_factory=list)
    ipc_classifications: List[IPCClassification] = field(default_factory=list)
    cpc_classifications: List[str] = field(default_factory=list)

    cited_patents: List[str] = field(default_factory=list)  # Patents this cites
    citing_patents: List[str] = field(default_factory=list)  # Patents that cite this

    family_id: str = ""
    family_members: List[str] = field(default_factory=list)

    source_url: str = ""
    source_database: str = ""  # "google_patents", "jplatpat", "espacenet"
    source_databases: List[str] = field(default_factory=list)  # All DBs where found
    full_text: str = ""

    # Auxiliary data gathered from secondary/tertiary searches
    related_papers: List[Dict[str, Any]] = field(default_factory=list)
    business_evidence: List[Dict[str, Any]] = field(default_factory=list)
    examination_records: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def normalized_number(self) -> str:
        """Get normalized patent number for deduplication."""
        return normalize_patent_number(self.patent_number)

    @property
    def independent_claims(self) -> List[PatentClaim]:
        """Get only independent claims."""
        return [c for c in self.claims if c.claim_type == "independent"]

    @property
    def dependent_claims(self) -> List[PatentClaim]:
        """Get only dependent claims."""
        return [c for c in self.claims if c.claim_type == "dependent"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patent_number": self.patent_number,
            "title": self.title,
            "abstract": self.abstract,
            "applicant": self.applicant,
            "inventor": self.inventor,
            "filing_date": self.filing_date,
            "publication_date": self.publication_date,
            "grant_date": self.grant_date,
            "status": self.status,
            "jurisdiction": self.jurisdiction,
            "claims": [c.to_dict() for c in self.claims],
            "ipc_classifications": [i.to_dict() for i in self.ipc_classifications],
            "cpc_classifications": self.cpc_classifications,
            "cited_patents": self.cited_patents,
            "citing_patents": self.citing_patents,
            "family_id": self.family_id,
            "family_members": self.family_members,
            "source_url": self.source_url,
            "source_database": self.source_database,
            "source_databases": self.source_databases,
            "related_papers": self.related_papers,
            "business_evidence": self.business_evidence,
            "examination_records": self.examination_records,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Patent":
        data = data.copy()
        if "claims" in data:
            data["claims"] = [PatentClaim.from_dict(c) for c in data["claims"]]
        if "ipc_classifications" in data:
            data["ipc_classifications"] = [
                IPCClassification.from_dict(i) for i in data["ipc_classifications"]
            ]
        # Remove full_text from dict to avoid huge memory in deserialization
        data.pop("full_text", None)
        return cls(**data)


@dataclass
class PatentFamily:
    """A patent family (same invention filed in multiple jurisdictions)."""

    family_id: str
    members: List[Patent] = field(default_factory=list)
    earliest_priority_date: str = ""
    jurisdictions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family_id": self.family_id,
            "members": [m.to_dict() for m in self.members],
            "earliest_priority_date": self.earliest_priority_date,
            "jurisdictions": self.jurisdictions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatentFamily":
        data = data.copy()
        if "members" in data:
            data["members"] = [Patent.from_dict(m) for m in data["members"]]
        return cls(**data)
