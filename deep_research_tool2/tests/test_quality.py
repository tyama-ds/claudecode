"""
Tests for quality categorization functionality.
"""

import pytest
import tempfile
from pathlib import Path

from deep_research_tool2.evidence.locker import (
    Evidence,
    EvidenceLocker,
    EvidenceType,
    QualityCategory,
    SourceType,
    QualityIndicators,
)
from deep_research_tool2.evidence.quality_evaluator import (
    QualityEvaluator,
    QualityEvaluation,
    categorize_by_quality,
    get_quality_summary,
)


class TestQualityIndicators:
    """Test QualityIndicators dataclass."""

    def test_default_values(self):
        """Test default values."""
        indicators = QualityIndicators()
        assert indicators.has_author is False
        assert indicators.has_date is False
        assert indicators.domain_authority == 0.0

    def test_calculate_score_empty(self):
        """Test score calculation with default values."""
        indicators = QualityIndicators()
        score = indicators.calculate_score()
        assert score == 0.0

    def test_calculate_score_full(self):
        """Test score calculation with all indicators positive."""
        indicators = QualityIndicators(
            has_author=True,
            has_date=True,
            has_citations=True,
            has_professional_tone=True,
            is_primary_source=True,
            is_peer_reviewed=True,
            domain_authority=1.0,
            content_depth=1.0,
            factual_consistency=1.0,
        )
        score = indicators.calculate_score()
        assert score == pytest.approx(1.0)

    def test_calculate_score_partial(self):
        """Test score calculation with some indicators."""
        indicators = QualityIndicators(
            has_author=True,
            has_citations=True,
            domain_authority=0.5,
        )
        score = indicators.calculate_score()
        # has_author (0.1) + has_citations (0.15) + domain_authority (0.5 * 0.1)
        assert score == pytest.approx(0.3, rel=0.1)

    def test_to_dict_and_from_dict(self):
        """Test serialization."""
        indicators = QualityIndicators(
            has_author=True,
            has_citations=True,
            domain_authority=0.8,
        )
        data = indicators.to_dict()
        restored = QualityIndicators.from_dict(data)
        assert restored.has_author is True
        assert restored.has_citations is True
        assert restored.domain_authority == 0.8


class TestEvidenceQuality:
    """Test Evidence with quality fields."""

    def test_evidence_with_quality(self):
        """Test creating evidence with quality fields."""
        evidence = Evidence(
            url="https://example.gov/report",
            title="Official Report",
            quality_category=QualityCategory.AUTHORITATIVE,
            source_type=SourceType.OFFICIAL,
            quality_indicators=QualityIndicators(
                has_author=True,
                has_date=True,
                is_primary_source=True,
            ),
        )
        assert evidence.quality_category == QualityCategory.AUTHORITATIVE
        assert evidence.source_type == SourceType.OFFICIAL
        assert evidence.quality_indicators.has_author is True

    def test_evidence_to_dict_with_quality(self):
        """Test serialization with quality fields."""
        evidence = Evidence(
            url="https://example.edu/paper",
            title="Academic Paper",
            quality_category=QualityCategory.HIGH,
            source_type=SourceType.ACADEMIC,
            quality_notes="Peer-reviewed journal",
            potential_biases=["Funding from industry"],
        )
        data = evidence.to_dict()
        assert data["quality_category"] == "high"
        assert data["source_type"] == "academic"
        assert data["quality_notes"] == "Peer-reviewed journal"

    def test_evidence_from_dict_with_quality(self):
        """Test deserialization with quality fields."""
        data = {
            "url": "https://news.com/article",
            "title": "News Article",
            "evidence_type": "news_article",
            "quality_category": "medium",
            "source_type": "news",
            "quality_indicators": {
                "has_author": True,
                "has_date": True,
                "has_citations": False,
                "has_professional_tone": True,
                "is_primary_source": False,
                "is_peer_reviewed": False,
                "domain_authority": 0.6,
                "content_depth": 0.4,
                "factual_consistency": 0.5,
            },
        }
        evidence = Evidence.from_dict(data)
        assert evidence.quality_category == QualityCategory.MEDIUM
        assert evidence.source_type == SourceType.NEWS
        assert evidence.quality_indicators.has_author is True


class TestEvidenceLockerQuality:
    """Test EvidenceLocker quality methods."""

    @pytest.fixture
    def locker_with_mixed_quality(self, tmp_path):
        """Create locker with evidence of varying quality."""
        locker = EvidenceLocker(output_dir=tmp_path)

        # Authoritative evidence
        locker.add_evidence(
            url="https://who.int/report",
            title="WHO Health Report",
            quality_category=QualityCategory.AUTHORITATIVE,
            source_type=SourceType.OFFICIAL,
        )

        # High quality evidence
        locker.add_evidence(
            url="https://nature.com/article",
            title="Nature Research",
            quality_category=QualityCategory.HIGH,
            source_type=SourceType.ACADEMIC,
        )

        # Medium quality evidence
        locker.add_evidence(
            url="https://techcrunch.com/news",
            title="Tech News",
            quality_category=QualityCategory.MEDIUM,
            source_type=SourceType.NEWS,
        )

        # Low quality evidence
        locker.add_evidence(
            url="https://reddit.com/discussion",
            title="Reddit Discussion",
            quality_category=QualityCategory.LOW,
            source_type=SourceType.FORUM,
        )

        # Unverified evidence
        locker.add_evidence(
            url="https://unknown-site.com/post",
            title="Unknown Post",
            quality_category=QualityCategory.UNVERIFIED,
            source_type=SourceType.UNKNOWN,
        )

        return locker

    def test_filter_by_quality_categories(self, locker_with_mixed_quality):
        """Test filtering by specific quality categories."""
        high_quality = locker_with_mixed_quality.get_evidence_by_quality(
            quality_categories=[QualityCategory.AUTHORITATIVE, QualityCategory.HIGH]
        )
        assert len(high_quality) == 2

    def test_filter_by_min_quality(self, locker_with_mixed_quality):
        """Test filtering by minimum quality."""
        medium_or_better = locker_with_mixed_quality.get_evidence_by_quality(
            min_quality=QualityCategory.MEDIUM
        )
        assert len(medium_or_better) == 3  # authoritative, high, medium

    def test_get_high_quality_evidence(self, locker_with_mixed_quality):
        """Test get_high_quality_evidence method."""
        high_quality = locker_with_mixed_quality.get_high_quality_evidence()
        assert len(high_quality) == 2
        for e in high_quality:
            assert e.quality_category in [QualityCategory.AUTHORITATIVE, QualityCategory.HIGH]

    def test_get_verified_evidence(self, locker_with_mixed_quality):
        """Test get_verified_evidence method."""
        verified = locker_with_mixed_quality.get_verified_evidence()
        assert len(verified) == 4  # All except UNVERIFIED
        for e in verified:
            assert e.quality_category != QualityCategory.UNVERIFIED

    def test_filter_by_source_type(self, locker_with_mixed_quality):
        """Test filtering by source type."""
        academic_sources = locker_with_mixed_quality.get_evidence_by_source_type(
            [SourceType.ACADEMIC, SourceType.OFFICIAL]
        )
        assert len(academic_sources) == 2

    def test_filter_evidence_combined(self, locker_with_mixed_quality):
        """Test combined filtering."""
        # High quality from academic/official sources
        filtered = locker_with_mixed_quality.filter_evidence(
            quality_categories=[QualityCategory.AUTHORITATIVE, QualityCategory.HIGH],
            source_types=[SourceType.ACADEMIC, SourceType.OFFICIAL],
        )
        assert len(filtered) == 2

    def test_quality_statistics(self, locker_with_mixed_quality):
        """Test quality statistics."""
        stats = locker_with_mixed_quality.get_quality_statistics()
        assert stats["total_evidence"] == 5
        assert stats["high_quality_count"] == 2
        assert stats["high_quality_percentage"] == 40.0
        assert "authoritative" in stats["quality_distribution"]
        assert "academic" in stats["source_type_distribution"]

    def test_update_quality(self, locker_with_mixed_quality):
        """Test updating quality information."""
        evidence = locker_with_mixed_quality.get_all_evidence()[0]

        locker_with_mixed_quality.update_quality(
            evidence.id,
            quality_category=QualityCategory.HIGH,
            quality_notes="Downgraded after review",
            potential_biases=["Potential conflict of interest"],
        )

        updated = locker_with_mixed_quality.get_evidence(evidence.id)
        assert updated.quality_category == QualityCategory.HIGH
        assert updated.quality_notes == "Downgraded after review"
        assert "Potential conflict of interest" in updated.potential_biases


class TestQualityEvaluator:
    """Test QualityEvaluator class."""

    def test_evaluate_url_government(self):
        """Test URL evaluation for government sites."""
        evaluator = QualityEvaluator()
        result = evaluator.evaluate_url("https://www.cdc.gov/health-report")
        assert result["quality_category"] == QualityCategory.AUTHORITATIVE
        assert result["source_type"] == SourceType.OFFICIAL

    def test_evaluate_url_academic(self):
        """Test URL evaluation for academic sites."""
        evaluator = QualityEvaluator()
        result = evaluator.evaluate_url("https://mit.edu/research/paper")
        assert result["quality_category"] == QualityCategory.AUTHORITATIVE
        assert result["source_type"] == SourceType.ACADEMIC

    def test_evaluate_url_news(self):
        """Test URL evaluation for news sites."""
        evaluator = QualityEvaluator()
        result = evaluator.evaluate_url("https://www.bbc.com/news/article")
        assert result["quality_category"] == QualityCategory.HIGH
        assert result["source_type"] == SourceType.NEWS

    def test_evaluate_url_social(self):
        """Test URL evaluation for social media."""
        evaluator = QualityEvaluator()
        result = evaluator.evaluate_url("https://twitter.com/user/post")
        assert result["quality_category"] == QualityCategory.LOW
        assert result["source_type"] == SourceType.SOCIAL

    def test_evaluate_url_unknown(self):
        """Test URL evaluation for unknown sites."""
        evaluator = QualityEvaluator()
        result = evaluator.evaluate_url("https://random-site.xyz/page")
        assert result["quality_category"] == QualityCategory.UNVERIFIED

    def test_evaluate_content_with_author(self):
        """Test content evaluation detecting author."""
        evaluator = QualityEvaluator()
        content = """
        This article was written by Dr. John Smith, Professor at MIT.
        Published on 2024-01-15.

        According to recent research [1], the findings show...

        References:
        [1] Nature Journal, 2023
        """
        evaluation = evaluator.evaluate_content(
            url="https://example.com/article",
            title="Research Article",
            content=content,
        )
        assert evaluation.quality_indicators.has_author is True
        assert evaluation.quality_indicators.has_date is True
        assert evaluation.quality_indicators.has_citations is True


class TestQualityUtilities:
    """Test quality utility functions."""

    def test_categorize_by_quality(self, tmp_path):
        """Test categorize_by_quality function."""
        locker = EvidenceLocker(output_dir=tmp_path)
        locker.add_evidence(
            url="https://gov.example/1",
            title="Gov 1",
            quality_category=QualityCategory.AUTHORITATIVE,
        )
        locker.add_evidence(
            url="https://news.example/1",
            title="News 1",
            quality_category=QualityCategory.MEDIUM,
        )
        locker.add_evidence(
            url="https://news.example/2",
            title="News 2",
            quality_category=QualityCategory.MEDIUM,
        )

        categorized = categorize_by_quality(locker)
        assert len(categorized[QualityCategory.AUTHORITATIVE]) == 1
        assert len(categorized[QualityCategory.MEDIUM]) == 2
        assert len(categorized[QualityCategory.LOW]) == 0

    def test_get_quality_summary_japanese(self, tmp_path):
        """Test Japanese quality summary."""
        locker = EvidenceLocker(output_dir=tmp_path)
        locker.add_evidence(
            url="https://gov.example/1",
            title="Gov 1",
            quality_category=QualityCategory.AUTHORITATIVE,
        )
        locker.add_evidence(
            url="https://news.example/1",
            title="News 1",
            quality_category=QualityCategory.HIGH,
        )

        summary = get_quality_summary(locker, language="ja")
        assert "情報品質サマリー" in summary
        assert "権威的" in summary or "高品質" in summary

    def test_get_quality_summary_english(self, tmp_path):
        """Test English quality summary."""
        locker = EvidenceLocker(output_dir=tmp_path)
        locker.add_evidence(
            url="https://gov.example/1",
            title="Gov 1",
            quality_category=QualityCategory.AUTHORITATIVE,
        )

        summary = get_quality_summary(locker, language="en")
        assert "Quality Summary" in summary


class TestCSVExportWithQuality:
    """Test CSV export with quality fields."""

    def test_export_csv_includes_quality(self, tmp_path):
        """Test that CSV export includes quality fields."""
        locker = EvidenceLocker(output_dir=tmp_path)
        locker.add_evidence(
            url="https://example.com/article",
            title="Test Article",
            quality_category=QualityCategory.HIGH,
            source_type=SourceType.NEWS,
            quality_notes="Good source",
        )

        csv_path = locker.export_to_csv()
        content = csv_path.read_text()

        assert "quality_category" in content
        assert "source_type" in content
        assert "quality_score" in content
        assert "high" in content
        assert "news" in content
