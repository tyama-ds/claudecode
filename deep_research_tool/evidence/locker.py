"""
Evidence Locker - Track and manage research sources and citations.
"""

import json
import csv
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import uuid4


class EvidenceType(str, Enum):
    """Types of evidence sources."""
    WEB_PAGE = "web_page"
    PDF_DOCUMENT = "pdf_document"
    RESEARCH_PAPER = "research_paper"
    NEWS_ARTICLE = "news_article"
    OFFICIAL_DOCUMENT = "official_document"
    IMAGE = "image"
    VIDEO = "video"
    API_RESPONSE = "api_response"
    USER_PROVIDED = "user_provided"
    OTHER = "other"


@dataclass
class Evidence:
    """A single piece of evidence/source."""

    # Core identification
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    evidence_type: EvidenceType = EvidenceType.WEB_PAGE

    # Source information
    url: str = ""
    title: str = ""
    author: str = ""
    publisher: str = ""
    published_date: str = ""

    # Content
    content_excerpt: str = ""
    content_hash: str = ""

    # Access information
    accessed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    access_method: str = ""  # e.g., "duckduckgo", "selenium", "direct"

    # Citation information
    citation_key: str = ""
    citation_text: str = ""

    # Research context
    search_query: str = ""
    section_reference: str = ""  # Which report section this supports

    # Quality indicators
    reliability_score: float = 0.0  # 0-1 scale
    relevance_score: float = 0.0  # 0-1 scale

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate derived fields after initialization."""
        if not self.citation_key:
            self.citation_key = self._generate_citation_key()

        if not self.content_hash and self.content_excerpt:
            self.content_hash = self._generate_content_hash()

        if not self.citation_text:
            self.citation_text = self._generate_citation_text()

    def _generate_citation_key(self) -> str:
        """Generate a citation key for this evidence."""
        # Format: [Author/Publisher][Year][TitleWord]
        parts = []

        # Author or publisher
        source = self.author or self.publisher or "Unknown"
        parts.append(source.split()[0][:10] if source else "Anon")

        # Year from published_date or accessed_at
        date_str = self.published_date or self.accessed_at
        try:
            if "-" in date_str:
                year = date_str.split("-")[0]
            else:
                year = datetime.now().strftime("%Y")
        except Exception:
            year = datetime.now().strftime("%Y")
        parts.append(year)

        # First significant word from title
        if self.title:
            words = [w for w in self.title.split() if len(w) > 3]
            if words:
                parts.append(words[0][:8])

        return "_".join(parts)

    def _generate_content_hash(self) -> str:
        """Generate a hash of the content for deduplication."""
        content = (self.url + self.title + self.content_excerpt).encode("utf-8")
        return hashlib.md5(content).hexdigest()[:12]

    def _generate_citation_text(self) -> str:
        """Generate formatted citation text."""
        parts = []

        # Author/Publisher
        if self.author:
            parts.append(self.author)
        elif self.publisher:
            parts.append(self.publisher)

        # Year
        if self.published_date:
            try:
                year = self.published_date.split("-")[0]
                parts.append(f"({year})")
            except Exception:
                pass

        # Title
        if self.title:
            parts.append(f'"{self.title}"')

        # URL
        if self.url:
            parts.append(f"Retrieved from {self.url}")

        # Access date
        if self.accessed_at:
            try:
                accessed = datetime.fromisoformat(self.accessed_at)
                parts.append(f"(Accessed: {accessed.strftime('%Y-%m-%d')})")
            except Exception:
                pass

        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["evidence_type"] = self.evidence_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        """Create from dictionary."""
        data = data.copy()
        if "evidence_type" in data:
            data["evidence_type"] = EvidenceType(data["evidence_type"])
        return cls(**data)

    def to_bibtex(self) -> str:
        """Generate BibTeX entry for this evidence."""
        entry_type = "misc"
        if self.evidence_type == EvidenceType.RESEARCH_PAPER:
            entry_type = "article"
        elif self.evidence_type == EvidenceType.PDF_DOCUMENT:
            entry_type = "techreport"
        elif self.evidence_type == EvidenceType.NEWS_ARTICLE:
            entry_type = "article"

        lines = [f"@{entry_type}{{{self.citation_key},"]
        if self.author:
            lines.append(f"  author = {{{self.author}}},")
        if self.title:
            lines.append(f"  title = {{{self.title}}},")
        if self.publisher:
            lines.append(f"  publisher = {{{self.publisher}}},")
        if self.published_date:
            lines.append(f"  year = {{{self.published_date.split('-')[0]}}},")
        if self.url:
            lines.append(f"  url = {{{self.url}}},")
        lines.append(f"  note = {{Accessed: {self.accessed_at}}},")
        lines.append("}")

        return "\n".join(lines)


class EvidenceLocker:
    """
    Manages collection and organization of research evidence.

    Tracks all sources accessed during research, generates citations,
    and exports evidence in various formats for integration with reports.
    """

    def __init__(
        self,
        research_id: str = None,
        output_dir: Path = None,
    ):
        """
        Initialize Evidence Locker.

        Args:
            research_id: Unique identifier for this research session
            output_dir: Directory for evidence exports
        """
        self.research_id = research_id or str(uuid4())[:8]
        self.output_dir = output_dir or Path("./output/evidence")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._evidence: Dict[str, Evidence] = {}
        self._citation_map: Dict[str, str] = {}  # citation_key -> evidence_id
        self._section_evidence: Dict[str, List[str]] = {}  # section -> evidence_ids

    def add_evidence(
        self,
        url: str,
        title: str,
        content_excerpt: str = "",
        evidence_type: EvidenceType = EvidenceType.WEB_PAGE,
        search_query: str = "",
        section_reference: str = "",
        **kwargs
    ) -> Evidence:
        """
        Add new evidence to the locker.

        Args:
            url: Source URL
            title: Source title
            content_excerpt: Excerpt of content used
            evidence_type: Type of evidence
            search_query: Query that found this source
            section_reference: Report section this supports
            **kwargs: Additional evidence fields

        Returns:
            Created Evidence object
        """
        evidence = Evidence(
            url=url,
            title=title,
            content_excerpt=content_excerpt,
            evidence_type=evidence_type,
            search_query=search_query,
            section_reference=section_reference,
            **kwargs
        )

        # Check for duplicates by content hash
        for existing in self._evidence.values():
            if existing.content_hash == evidence.content_hash:
                # Return existing evidence instead
                return existing

        # Add to collection
        self._evidence[evidence.id] = evidence
        self._citation_map[evidence.citation_key] = evidence.id

        # Track section association
        if section_reference:
            if section_reference not in self._section_evidence:
                self._section_evidence[section_reference] = []
            self._section_evidence[section_reference].append(evidence.id)

        return evidence

    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        """Get evidence by ID."""
        return self._evidence.get(evidence_id)

    def get_by_citation_key(self, citation_key: str) -> Optional[Evidence]:
        """Get evidence by citation key."""
        evidence_id = self._citation_map.get(citation_key)
        return self._evidence.get(evidence_id) if evidence_id else None

    def get_section_evidence(self, section: str) -> List[Evidence]:
        """Get all evidence for a specific section."""
        evidence_ids = self._section_evidence.get(section, [])
        return [self._evidence[eid] for eid in evidence_ids if eid in self._evidence]

    def get_all_evidence(self) -> List[Evidence]:
        """Get all evidence items."""
        return list(self._evidence.values())

    def update_reliability(self, evidence_id: str, score: float) -> None:
        """Update reliability score for evidence."""
        if evidence_id in self._evidence:
            self._evidence[evidence_id].reliability_score = max(0, min(1, score))

    def update_relevance(self, evidence_id: str, score: float) -> None:
        """Update relevance score for evidence."""
        if evidence_id in self._evidence:
            self._evidence[evidence_id].relevance_score = max(0, min(1, score))

    def generate_citation(self, evidence_id: str, style: str = "apa") -> str:
        """
        Generate formatted citation for evidence.

        Args:
            evidence_id: Evidence ID
            style: Citation style (apa, mla, chicago, bibtex)

        Returns:
            Formatted citation string
        """
        evidence = self._evidence.get(evidence_id)
        if not evidence:
            return ""

        if style.lower() == "bibtex":
            return evidence.to_bibtex()

        # Default APA-style citation
        return evidence.citation_text

    def export_to_json(self, filepath: Path = None) -> Path:
        """
        Export all evidence to JSON file.

        Args:
            filepath: Output file path (default: auto-generated)

        Returns:
            Path to exported file
        """
        filepath = filepath or self.output_dir / f"evidence_{self.research_id}.json"

        export_data = {
            "research_id": self.research_id,
            "exported_at": datetime.now().isoformat(),
            "total_evidence": len(self._evidence),
            "evidence": [e.to_dict() for e in self._evidence.values()],
            "sections": {
                section: [self._evidence[eid].to_dict() for eid in eids if eid in self._evidence]
                for section, eids in self._section_evidence.items()
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        return filepath

    def export_to_csv(self, filepath: Path = None) -> Path:
        """
        Export all evidence to CSV file.

        Args:
            filepath: Output file path (default: auto-generated)

        Returns:
            Path to exported file
        """
        filepath = filepath or self.output_dir / f"evidence_{self.research_id}.csv"

        fieldnames = [
            "id", "citation_key", "evidence_type", "url", "title",
            "author", "publisher", "published_date", "content_excerpt",
            "accessed_at", "search_query", "section_reference",
            "reliability_score", "relevance_score", "citation_text"
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for evidence in self._evidence.values():
                row = {
                    "id": evidence.id,
                    "citation_key": evidence.citation_key,
                    "evidence_type": evidence.evidence_type.value,
                    "url": evidence.url,
                    "title": evidence.title,
                    "author": evidence.author,
                    "publisher": evidence.publisher,
                    "published_date": evidence.published_date,
                    "content_excerpt": evidence.content_excerpt[:500],
                    "accessed_at": evidence.accessed_at,
                    "search_query": evidence.search_query,
                    "section_reference": evidence.section_reference,
                    "reliability_score": evidence.reliability_score,
                    "relevance_score": evidence.relevance_score,
                    "citation_text": evidence.citation_text,
                }
                writer.writerow(row)

        return filepath

    def export_bibliography(
        self,
        filepath: Path = None,
        style: str = "apa"
    ) -> Path:
        """
        Export bibliography/references list.

        Args:
            filepath: Output file path
            style: Citation style

        Returns:
            Path to exported file
        """
        filepath = filepath or self.output_dir / f"bibliography_{self.research_id}.txt"

        citations = []
        for i, evidence in enumerate(sorted(
            self._evidence.values(),
            key=lambda e: e.citation_key
        ), 1):
            citation = self.generate_citation(evidence.id, style)
            citations.append(f"[{i}] {citation}")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("References\n")
            f.write("=" * 50 + "\n\n")
            f.write("\n\n".join(citations))

        return filepath

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected evidence."""
        evidence_list = list(self._evidence.values())

        type_counts = {}
        for e in evidence_list:
            type_name = e.evidence_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        avg_reliability = (
            sum(e.reliability_score for e in evidence_list) / len(evidence_list)
            if evidence_list else 0
        )
        avg_relevance = (
            sum(e.relevance_score for e in evidence_list) / len(evidence_list)
            if evidence_list else 0
        )

        return {
            "total_evidence": len(evidence_list),
            "evidence_by_type": type_counts,
            "sections_covered": list(self._section_evidence.keys()),
            "average_reliability": round(avg_reliability, 2),
            "average_relevance": round(avg_relevance, 2),
            "unique_domains": len(set(
                self._extract_domain(e.url) for e in evidence_list if e.url
            )),
        }

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return ""

    def merge_from(self, other: "EvidenceLocker") -> int:
        """
        Merge evidence from another locker.

        Args:
            other: Another EvidenceLocker to merge from

        Returns:
            Number of new evidence items added
        """
        added = 0
        for evidence in other.get_all_evidence():
            # Check for duplicates
            if evidence.content_hash not in [e.content_hash for e in self._evidence.values()]:
                self._evidence[evidence.id] = evidence
                self._citation_map[evidence.citation_key] = evidence.id
                added += 1

                # Merge section mappings
                if evidence.section_reference:
                    if evidence.section_reference not in self._section_evidence:
                        self._section_evidence[evidence.section_reference] = []
                    self._section_evidence[evidence.section_reference].append(evidence.id)

        return added

    def to_dict(self) -> Dict[str, Any]:
        """Convert locker to dictionary for serialization."""
        return {
            "research_id": self.research_id,
            "evidence": [e.to_dict() for e in self._evidence.values()],
            "section_evidence": self._section_evidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], output_dir: Path = None) -> "EvidenceLocker":
        """Create locker from dictionary."""
        locker = cls(
            research_id=data.get("research_id"),
            output_dir=output_dir,
        )

        for e_data in data.get("evidence", []):
            evidence = Evidence.from_dict(e_data)
            locker._evidence[evidence.id] = evidence
            locker._citation_map[evidence.citation_key] = evidence.id

        locker._section_evidence = data.get("section_evidence", {})

        return locker

    @classmethod
    def load_from_json(cls, filepath: Path) -> "EvidenceLocker":
        """Load locker from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, output_dir=filepath.parent)
