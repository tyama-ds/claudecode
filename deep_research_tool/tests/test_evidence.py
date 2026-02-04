"""
Tests for Evidence Locker module.
"""

import json
import tempfile
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deep_research_tool.evidence.locker import (
    EvidenceLocker,
    Evidence,
    EvidenceType,
)


class TestEvidence:
    """Tests for Evidence dataclass."""

    def test_evidence_creation(self):
        """Test creating an evidence item."""
        evidence = Evidence(
            url="https://example.com/article",
            title="Test Article",
            content_excerpt="This is test content.",
        )
        assert evidence.url == "https://example.com/article"
        assert evidence.title == "Test Article"
        assert evidence.id is not None
        assert evidence.citation_key is not None

    def test_evidence_to_dict(self):
        """Test converting evidence to dictionary."""
        evidence = Evidence(
            url="https://example.com",
            title="Test",
        )
        data = evidence.to_dict()
        assert "url" in data
        assert "title" in data
        assert "evidence_type" in data
        assert data["evidence_type"] == "web_page"

    def test_evidence_from_dict(self):
        """Test creating evidence from dictionary."""
        data = {
            "id": "test123",
            "url": "https://example.com",
            "title": "Test",
            "evidence_type": "web_page",
            "content_excerpt": "Content",
        }
        evidence = Evidence.from_dict(data)
        assert evidence.id == "test123"
        assert evidence.url == "https://example.com"

    def test_citation_generation(self):
        """Test citation text generation."""
        evidence = Evidence(
            url="https://example.com/article",
            title="Important Research",
            author="John Smith",
            published_date="2024-01-15",
        )
        assert "John Smith" in evidence.citation_text
        assert "Important Research" in evidence.citation_text

    def test_bibtex_generation(self):
        """Test BibTeX entry generation."""
        evidence = Evidence(
            url="https://example.com",
            title="Research Paper",
            author="Jane Doe",
            published_date="2024-06-01",
            evidence_type=EvidenceType.RESEARCH_PAPER,
        )
        bibtex = evidence.to_bibtex()
        assert "@article" in bibtex
        assert "Jane Doe" in bibtex


class TestEvidenceLocker:
    """Tests for EvidenceLocker class."""

    def test_locker_creation(self):
        """Test creating an evidence locker."""
        locker = EvidenceLocker()
        assert locker.research_id is not None
        assert len(locker.get_all_evidence()) == 0

    def test_add_evidence(self):
        """Test adding evidence to locker."""
        locker = EvidenceLocker()
        evidence = locker.add_evidence(
            url="https://example.com",
            title="Test Article",
            content_excerpt="Test content",
        )
        assert evidence is not None
        assert len(locker.get_all_evidence()) == 1

    def test_add_duplicate_evidence(self):
        """Test that duplicates are detected."""
        locker = EvidenceLocker()
        e1 = locker.add_evidence(
            url="https://example.com",
            title="Test",
            content_excerpt="Content",
        )
        e2 = locker.add_evidence(
            url="https://example.com",
            title="Test",
            content_excerpt="Content",
        )
        # Should return existing evidence
        assert e1.id == e2.id
        assert len(locker.get_all_evidence()) == 1

    def test_get_evidence_by_id(self):
        """Test retrieving evidence by ID."""
        locker = EvidenceLocker()
        evidence = locker.add_evidence(
            url="https://example.com",
            title="Test",
        )
        retrieved = locker.get_evidence(evidence.id)
        assert retrieved is not None
        assert retrieved.url == "https://example.com"

    def test_section_evidence_tracking(self):
        """Test tracking evidence by section."""
        locker = EvidenceLocker()
        locker.add_evidence(
            url="https://example1.com",
            title="Source 1",
            section_reference="1.1",
        )
        locker.add_evidence(
            url="https://example2.com",
            title="Source 2",
            section_reference="1.1",
        )
        locker.add_evidence(
            url="https://example3.com",
            title="Source 3",
            section_reference="2.1",
        )

        section_evidence = locker.get_section_evidence("1.1")
        assert len(section_evidence) == 2

    def test_export_to_json(self):
        """Test exporting evidence to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            locker = EvidenceLocker(output_dir=Path(tmpdir))
            locker.add_evidence(url="https://example.com", title="Test")

            filepath = locker.export_to_json()
            assert filepath.exists()

            with open(filepath) as f:
                data = json.load(f)
            assert "evidence" in data
            assert len(data["evidence"]) == 1

    def test_export_to_csv(self):
        """Test exporting evidence to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            locker = EvidenceLocker(output_dir=Path(tmpdir))
            locker.add_evidence(url="https://example.com", title="Test")

            filepath = locker.export_to_csv()
            assert filepath.exists()

    def test_statistics(self):
        """Test getting statistics."""
        locker = EvidenceLocker()
        locker.add_evidence(
            url="https://example1.com",
            title="Web Article",
            evidence_type=EvidenceType.WEB_PAGE,
        )
        locker.add_evidence(
            url="https://example2.com",
            title="News",
            evidence_type=EvidenceType.NEWS_ARTICLE,
        )

        stats = locker.get_statistics()
        assert stats["total_evidence"] == 2
        assert "evidence_by_type" in stats

    def test_merge_lockers(self):
        """Test merging two evidence lockers."""
        locker1 = EvidenceLocker()
        locker1.add_evidence(url="https://example1.com", title="Source 1")

        locker2 = EvidenceLocker()
        locker2.add_evidence(url="https://example2.com", title="Source 2")

        added = locker1.merge_from(locker2)
        assert added == 1
        assert len(locker1.get_all_evidence()) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
