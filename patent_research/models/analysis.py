"""
Analysis models for patent research.

Data structures for claim charts, technology landscapes, and prior art records.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ClaimChartEntry:
    """A single row in a claim chart."""

    claim_element: str
    patent_number: str
    mapping: str  # How the claim element maps to prior art / product
    confidence: float = 0.0
    source_excerpt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_element": self.claim_element,
            "patent_number": self.patent_number,
            "mapping": self.mapping,
            "confidence": self.confidence,
            "source_excerpt": self.source_excerpt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimChartEntry":
        return cls(**data)


@dataclass
class ClaimChart:
    """A complete claim chart comparing claims against prior art or products."""

    target_patent: str
    comparison_type: str = "prior_art"  # "prior_art", "product", "freedom_to_operate"
    entries: List[ClaimChartEntry] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_patent": self.target_patent,
            "comparison_type": self.comparison_type,
            "entries": [e.to_dict() for e in self.entries],
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimChart":
        data = data.copy()
        if "entries" in data:
            data["entries"] = [ClaimChartEntry.from_dict(e) for e in data["entries"]]
        return cls(**data)


@dataclass
class PriorArtRecord:
    """A prior art record found during patent research."""

    patent_number: str = ""
    title: str = ""
    source_type: str = ""  # "patent", "paper", "standard", "product"
    source_url: str = ""
    relevance_score: float = 0.0
    relevant_claims: List[int] = field(default_factory=list)
    description: str = ""
    key_overlap: str = ""  # Brief description of overlap with target patent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patent_number": self.patent_number,
            "title": self.title,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "relevance_score": self.relevance_score,
            "relevant_claims": self.relevant_claims,
            "description": self.description,
            "key_overlap": self.key_overlap,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PriorArtRecord":
        return cls(**data)


@dataclass
class TechnologyLandscape:
    """Technology landscape analysis results."""

    topic: str
    total_patents_analyzed: int = 0
    date_range: str = ""

    # IPC distribution: code -> count
    ipc_distribution: Dict[str, int] = field(default_factory=dict)

    # Top applicants: [{name, count, key_patents}]
    top_applicants: List[Dict[str, Any]] = field(default_factory=list)

    # Filing trend: year -> count
    filing_trend: Dict[str, int] = field(default_factory=dict)

    # Key technologies identified
    key_technologies: List[str] = field(default_factory=list)

    # Technology clusters: [{name, patents, description}]
    clusters: List[Dict[str, Any]] = field(default_factory=list)

    # Geographic distribution: jurisdiction -> count
    geographic_distribution: Dict[str, int] = field(default_factory=dict)

    # Summary
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "total_patents_analyzed": self.total_patents_analyzed,
            "date_range": self.date_range,
            "ipc_distribution": self.ipc_distribution,
            "top_applicants": self.top_applicants,
            "filing_trend": self.filing_trend,
            "key_technologies": self.key_technologies,
            "clusters": self.clusters,
            "geographic_distribution": self.geographic_distribution,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TechnologyLandscape":
        return cls(**data)
