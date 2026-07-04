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


class QualityCategory(str, Enum):
    """Information quality categories."""
    AUTHORITATIVE = "authoritative"  # Government, academic, established research
    HIGH = "high"  # Major news outlets, verified experts, peer-reviewed
    MEDIUM = "medium"  # Reputable blogs, industry publications
    LOW = "low"  # User-generated, forums, unverified
    UNVERIFIED = "unverified"  # Sources that couldn't be evaluated


class SourceType(str, Enum):
    """Types of source origins."""
    OFFICIAL = "official"  # Government, official organizations
    ACADEMIC = "academic"  # Universities, research institutions
    NEWS = "news"  # News media outlets
    BLOG = "blog"  # Personal or company blogs
    SOCIAL = "social"  # Social media
    COMMERCIAL = "commercial"  # Commercial/product sites
    WIKI = "wiki"  # Wikipedia and similar
    FORUM = "forum"  # Forums, Q&A sites
    UNKNOWN = "unknown"


@dataclass
class QualityIndicators:
    """Indicators for source quality assessment."""
    has_author: bool = False
    has_date: bool = False
    has_citations: bool = False
    has_professional_tone: bool = False
    is_primary_source: bool = False
    is_peer_reviewed: bool = False
    domain_authority: float = 0.0  # 0-1 scale
    content_depth: float = 0.0  # 0-1 scale
    factual_consistency: float = 0.0  # 0-1 scale

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_author": self.has_author,
            "has_date": self.has_date,
            "has_citations": self.has_citations,
            "has_professional_tone": self.has_professional_tone,
            "is_primary_source": self.is_primary_source,
            "is_peer_reviewed": self.is_peer_reviewed,
            "domain_authority": self.domain_authority,
            "content_depth": self.content_depth,
            "factual_consistency": self.factual_consistency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityIndicators":
        """Create from dictionary."""
        return cls(**data)

    def calculate_score(self) -> float:
        """Calculate overall quality score from indicators."""
        score = 0.0
        weights = {
            "has_author": 0.1,
            "has_date": 0.05,
            "has_citations": 0.15,
            "has_professional_tone": 0.1,
            "is_primary_source": 0.15,
            "is_peer_reviewed": 0.2,
            "domain_authority": 0.1,
            "content_depth": 0.075,
            "factual_consistency": 0.075,
        }
        if self.has_author:
            score += weights["has_author"]
        if self.has_date:
            score += weights["has_date"]
        if self.has_citations:
            score += weights["has_citations"]
        if self.has_professional_tone:
            score += weights["has_professional_tone"]
        if self.is_primary_source:
            score += weights["is_primary_source"]
        if self.is_peer_reviewed:
            score += weights["is_peer_reviewed"]
        score += self.domain_authority * weights["domain_authority"]
        score += self.content_depth * weights["content_depth"]
        score += self.factual_consistency * weights["factual_consistency"]
        return min(1.0, score)


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
    relevance_score: float = 0.0  # 0-1 scale (relevance to the section)
    importance_score: float = 0.0  # 0-1 scale (importance to the research purpose)

    # Quality categorization
    quality_category: QualityCategory = QualityCategory.UNVERIFIED
    source_type: SourceType = SourceType.UNKNOWN
    quality_indicators: QualityIndicators = field(default_factory=QualityIndicators)
    quality_notes: str = ""  # Notes about the quality assessment
    potential_biases: List[str] = field(default_factory=list)

    # Multilingual support
    source_language: str = ""  # ISO 639-1 code (e.g., "ja", "en", "zh")
    original_title: str = ""  # Title in original language
    original_content: str = ""  # Content in original language
    translated_title: str = ""  # Translated title (if applicable)
    translated_content: str = ""  # Translated content (if applicable)
    translation_confidence: float = 1.0  # Confidence in translation (0-1)
    is_translated: bool = False  # Whether content was translated

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate derived fields after initialization."""
        if not self.citation_key:
            self.citation_key = self._generate_citation_key()

        # Always generate content_hash for deduplication (using url+title at minimum)
        if not self.content_hash:
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
        data["quality_category"] = self.quality_category.value
        data["source_type"] = self.source_type.value
        data["quality_indicators"] = self.quality_indicators.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        """Create from dictionary."""
        data = data.copy()
        if "evidence_type" in data:
            data["evidence_type"] = EvidenceType(data["evidence_type"])
        if "quality_category" in data:
            data["quality_category"] = QualityCategory(data["quality_category"])
        if "source_type" in data:
            data["source_type"] = SourceType(data["source_type"])
        if "quality_indicators" in data and isinstance(data["quality_indicators"], dict):
            data["quality_indicators"] = QualityIndicators.from_dict(data["quality_indicators"])
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

    def update_importance(self, evidence_id: str, score: float) -> None:
        """Update importance score (relevance to the research purpose)."""
        if evidence_id in self._evidence:
            self._evidence[evidence_id].importance_score = max(0, min(1, score))

    def update_importance_by_url(
        self, url: str, score: float, section: str = None,
    ) -> int:
        """
        Update importance for all evidence matching a URL.

        Args:
            url: Source URL to match
            score: Importance score (0-1)
            section: If given, only update evidence tied to this section

        Returns:
            Number of evidence items updated
        """
        updated = 0
        for evidence in self._evidence.values():
            if evidence.url != url:
                continue
            if section and evidence.section_reference != section:
                continue
            evidence.importance_score = max(0, min(1, score))
            updated += 1
        return updated

    def get_evidence_by_importance(
        self, min_importance: float = 0.0, section: str = None,
    ) -> List[Evidence]:
        """
        Get evidence sorted by importance (descending).

        Args:
            min_importance: Minimum importance score to include
            section: If given, restrict to evidence for this section

        Returns:
            Evidence items sorted by importance_score descending
        """
        items = (
            self.get_section_evidence(section) if section
            else self.get_all_evidence()
        )
        filtered = [e for e in items if e.importance_score >= min_importance]
        return sorted(filtered, key=lambda e: e.importance_score, reverse=True)

    def update_quality(
        self,
        evidence_id: str,
        quality_category: QualityCategory = None,
        source_type: SourceType = None,
        quality_indicators: QualityIndicators = None,
        quality_notes: str = None,
        potential_biases: List[str] = None,
    ) -> None:
        """
        Update quality information for evidence.

        Args:
            evidence_id: Evidence ID
            quality_category: Quality category
            source_type: Source type
            quality_indicators: Quality indicators
            quality_notes: Notes about quality
            potential_biases: List of potential biases
        """
        if evidence_id not in self._evidence:
            return

        evidence = self._evidence[evidence_id]
        if quality_category is not None:
            evidence.quality_category = quality_category
        if source_type is not None:
            evidence.source_type = source_type
        if quality_indicators is not None:
            evidence.quality_indicators = quality_indicators
        if quality_notes is not None:
            evidence.quality_notes = quality_notes
        if potential_biases is not None:
            evidence.potential_biases = potential_biases

    def get_evidence_by_quality(
        self,
        min_quality: QualityCategory = None,
        quality_categories: List[QualityCategory] = None,
    ) -> List[Evidence]:
        """
        Get evidence filtered by quality category.

        Args:
            min_quality: Minimum quality category (AUTHORITATIVE > HIGH > MEDIUM > LOW > UNVERIFIED)
            quality_categories: Specific quality categories to include

        Returns:
            List of evidence matching the quality criteria
        """
        quality_order = [
            QualityCategory.AUTHORITATIVE,
            QualityCategory.HIGH,
            QualityCategory.MEDIUM,
            QualityCategory.LOW,
            QualityCategory.UNVERIFIED,
        ]

        if quality_categories:
            return [
                e for e in self._evidence.values()
                if e.quality_category in quality_categories
            ]

        if min_quality:
            min_index = quality_order.index(min_quality)
            allowed_categories = quality_order[:min_index + 1]
            return [
                e for e in self._evidence.values()
                if e.quality_category in allowed_categories
            ]

        return list(self._evidence.values())

    def get_evidence_by_source_type(
        self,
        source_types: List[SourceType],
    ) -> List[Evidence]:
        """
        Get evidence filtered by source type.

        Args:
            source_types: List of source types to include

        Returns:
            List of evidence matching the source types
        """
        return [
            e for e in self._evidence.values()
            if e.source_type in source_types
        ]

    def get_high_quality_evidence(self) -> List[Evidence]:
        """Get only AUTHORITATIVE and HIGH quality evidence."""
        return self.get_evidence_by_quality(
            quality_categories=[QualityCategory.AUTHORITATIVE, QualityCategory.HIGH]
        )

    def get_verified_evidence(self) -> List[Evidence]:
        """Get all evidence except UNVERIFIED."""
        return self.get_evidence_by_quality(min_quality=QualityCategory.LOW)

    def filter_evidence(
        self,
        min_quality: QualityCategory = None,
        quality_categories: List[QualityCategory] = None,
        source_types: List[SourceType] = None,
        min_reliability: float = None,
        min_relevance: float = None,
        section: str = None,
    ) -> List[Evidence]:
        """
        Filter evidence by multiple criteria.

        Args:
            min_quality: Minimum quality category
            quality_categories: Specific quality categories to include
            source_types: Source types to include
            min_reliability: Minimum reliability score (0-1)
            min_relevance: Minimum relevance score (0-1)
            section: Filter by section reference

        Returns:
            List of evidence matching all criteria
        """
        # Start with all evidence or section evidence
        if section:
            evidence_list = self.get_section_evidence(section)
        else:
            evidence_list = list(self._evidence.values())

        # Apply quality category filter
        if quality_categories:
            evidence_list = [e for e in evidence_list if e.quality_category in quality_categories]
        elif min_quality:
            quality_order = [
                QualityCategory.AUTHORITATIVE,
                QualityCategory.HIGH,
                QualityCategory.MEDIUM,
                QualityCategory.LOW,
                QualityCategory.UNVERIFIED,
            ]
            min_index = quality_order.index(min_quality)
            allowed_categories = quality_order[:min_index + 1]
            evidence_list = [e for e in evidence_list if e.quality_category in allowed_categories]

        # Apply source type filter
        if source_types:
            evidence_list = [e for e in evidence_list if e.source_type in source_types]

        # Apply reliability filter
        if min_reliability is not None:
            evidence_list = [e for e in evidence_list if e.reliability_score >= min_reliability]

        # Apply relevance filter
        if min_relevance is not None:
            evidence_list = [e for e in evidence_list if e.relevance_score >= min_relevance]

        return evidence_list

    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get statistics about evidence quality distribution."""
        evidence_list = list(self._evidence.values())

        if not evidence_list:
            return {
                "total_evidence": 0,
                "quality_distribution": {},
                "source_type_distribution": {},
                "average_quality_score": 0,
                "high_quality_count": 0,
                "high_quality_percentage": 0,
            }

        # Quality distribution
        quality_counts = {}
        for e in evidence_list:
            cat = e.quality_category.value
            quality_counts[cat] = quality_counts.get(cat, 0) + 1

        # Source type distribution
        source_type_counts = {}
        for e in evidence_list:
            st = e.source_type.value
            source_type_counts[st] = source_type_counts.get(st, 0) + 1

        # Calculate average quality score from indicators
        quality_scores = [e.quality_indicators.calculate_score() for e in evidence_list]
        avg_quality_score = sum(quality_scores) / len(quality_scores)

        # High quality count
        high_quality_count = sum(
            1 for e in evidence_list
            if e.quality_category in [QualityCategory.AUTHORITATIVE, QualityCategory.HIGH]
        )

        return {
            "total_evidence": len(evidence_list),
            "quality_distribution": quality_counts,
            "source_type_distribution": source_type_counts,
            "average_quality_score": round(avg_quality_score, 3),
            "high_quality_count": high_quality_count,
            "high_quality_percentage": round(high_quality_count / len(evidence_list) * 100, 1),
        }

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
            "reliability_score", "relevance_score", "importance_score",
            "quality_category", "source_type", "quality_score", "quality_notes",
            "citation_text"
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
                    "importance_score": evidence.importance_score,
                    "quality_category": evidence.quality_category.value,
                    "source_type": evidence.source_type.value,
                    "quality_score": evidence.quality_indicators.calculate_score(),
                    "quality_notes": evidence.quality_notes,
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
