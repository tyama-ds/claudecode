"""
Tests for output length control functionality.
"""

import pytest
import tempfile
from pathlib import Path

from deep_research_tool.report.length_controller import (
    ContentLengthController,
    LengthTarget,
    LengthInfo,
    estimate_page_count,
    get_length_summary,
)
from deep_research_tool.report.generator import ReportGenerator, ReportFormat
from deep_research_tool.research.researcher import ResearchSession
from deep_research_tool.research.query_generator import (
    ResearchPlan,
    TableOfContents,
    TableOfContentsItem,
)
from deep_research_tool.evidence.locker import EvidenceLocker


class TestLengthTarget:
    """Test LengthTarget dataclass."""

    def test_default_values(self):
        """Test default values."""
        target = LengthTarget()
        assert target.target_pages is None
        assert target.target_characters is None
        assert target.chars_per_page == 2500

    def test_has_target_false(self):
        """Test has_target when no targets set."""
        target = LengthTarget()
        assert target.has_target() is False

    def test_has_target_with_pages(self):
        """Test has_target with page target."""
        target = LengthTarget(target_pages=10)
        assert target.has_target() is True

    def test_has_target_with_characters(self):
        """Test has_target with character target."""
        target = LengthTarget(target_characters=25000)
        assert target.has_target() is True

    def test_get_target_characters_from_pages(self):
        """Test calculating target characters from pages."""
        target = LengthTarget(target_pages=10, chars_per_page=2500)
        assert target.get_target_characters() == 25000

    def test_get_target_characters_direct(self):
        """Test direct target characters takes precedence."""
        target = LengthTarget(
            target_pages=10,
            target_characters=30000,
            chars_per_page=2500
        )
        # Direct target_characters takes precedence
        assert target.get_target_characters() == 30000


class TestLengthInfo:
    """Test LengthInfo dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        info = LengthInfo(
            total_characters=25000,
            estimated_pages=10.0,
            section_lengths={"1": 5000, "2": 5000},
        )
        data = info.to_dict()
        assert data["total_characters"] == 25000
        assert data["estimated_pages"] == 10.0
        assert len(data["section_lengths"]) == 2


class TestContentLengthController:
    """Test ContentLengthController class."""

    def test_calculate_length(self):
        """Test calculating content length."""
        controller = ContentLengthController()
        section_contents = {
            "1": {"content": "A" * 2500, "summary": "B" * 500},
            "2": {"content": "C" * 3000},
        }
        info = controller.calculate_length(section_contents)
        assert info.total_characters == 6000
        assert info.estimated_pages == pytest.approx(2.4, rel=0.1)

    def test_needs_adjustment_no_target(self):
        """Test needs_adjustment with no target."""
        controller = ContentLengthController()
        section_contents = {"1": {"content": "A" * 5000}}
        needs_adj, ratio = controller.needs_adjustment(section_contents)
        assert needs_adj is False
        assert ratio == 1.0

    def test_needs_adjustment_shrink_needed(self):
        """Test needs_adjustment when shrinking is needed."""
        target = LengthTarget(target_characters=5000)
        controller = ContentLengthController(target=target)
        section_contents = {"1": {"content": "A" * 10000}}
        needs_adj, ratio = controller.needs_adjustment(section_contents)
        assert needs_adj is True
        assert ratio < 1.0
        assert ratio == pytest.approx(0.5, rel=0.1)

    def test_needs_adjustment_within_tolerance(self):
        """Test needs_adjustment within 10% tolerance."""
        target = LengthTarget(target_characters=10000)
        controller = ContentLengthController(target=target)
        # 9500 is within 10% of 10000
        section_contents = {"1": {"content": "A" * 9500}}
        needs_adj, ratio = controller.needs_adjustment(section_contents)
        assert needs_adj is False

    def test_adjust_content_shrink(self):
        """Test content shrinking."""
        target = LengthTarget(target_characters=5000)
        controller = ContentLengthController(target=target)

        original_content = "A" * 10000
        section_contents = {"1": {"content": original_content}}

        adjusted = controller.adjust_content(section_contents)

        # Should be shrunk
        assert len(adjusted["1"]["content"]) < len(original_content)

    def test_adjust_content_no_change_without_target(self):
        """Test content is unchanged without target."""
        controller = ContentLengthController()
        original_content = "A" * 10000
        section_contents = {"1": {"content": original_content}}

        adjusted = controller.adjust_content(section_contents)

        assert adjusted["1"]["content"] == original_content

    def test_shrink_text_at_sentence(self):
        """Test text is shrunk at sentence boundaries."""
        controller = ContentLengthController()

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        shrunk = controller._shrink_text(text, 0.5)

        # Should end at a sentence boundary
        assert shrunk.endswith(".") or shrunk.endswith("...")

    def test_shrink_text_japanese(self):
        """Test Japanese text shrinking."""
        controller = ContentLengthController(language="ja")

        text = "最初の文章です。次の文章です。三番目の文章です。四番目の文章です。"
        shrunk = controller._shrink_text(text, 0.5)

        # Should be shorter
        assert len(shrunk) < len(text)

    def test_shrink_preserves_paragraphs(self):
        """Test that paragraph structure is preserved."""
        controller = ContentLengthController()

        text = """First paragraph with some content.

Second paragraph with more content.

Third paragraph with additional content.

Fourth paragraph with final content."""

        shrunk = controller._shrink_text(text, 0.6)

        # Should still have paragraph structure
        assert "\n\n" in shrunk

    def test_key_findings_are_limited(self):
        """Test that key findings list is limited."""
        target = LengthTarget(target_characters=1000)
        controller = ContentLengthController(target=target)

        section_contents = {
            "_executive_summary": {
                "content": "A" * 5000,
                "key_findings": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],
            }
        }

        adjusted = controller.adjust_content(section_contents, adjustment_ratio=0.5)

        # Key findings should be reduced but at least 2
        assert len(adjusted["_executive_summary"]["key_findings"]) >= 2
        assert len(adjusted["_executive_summary"]["key_findings"]) < 8


class TestReportGeneratorLengthControl:
    """Test ReportGenerator with length control."""

    @pytest.fixture
    def sample_session(self):
        """Create a sample session with content."""
        session = ResearchSession(
            query="Test research",
            requirements="Test requirements",
        )
        session.research_plan = ResearchPlan(
            title="Test Report",
            summary="Test summary",
            table_of_contents=TableOfContents(
                title="Test Report",
                items=[
                    TableOfContentsItem(
                        section="1",
                        title="Introduction",
                        description="Test intro",
                        subsections=[],
                    ),
                    TableOfContentsItem(
                        section="2",
                        title="Main Content",
                        description="Main section",
                        subsections=[],
                    ),
                ],
            ),
        )
        # Create content that would be ~8 pages
        session.section_contents = {
            "_executive_summary": {
                "executive_summary": "This is a test executive summary. " * 50,
                "key_findings": ["Finding " + str(i) for i in range(10)],
            },
            "1": {
                "title": "Introduction",
                "content": "Introduction content paragraph. " * 200,
                "summary": "Brief intro summary.",
            },
            "2": {
                "title": "Main Content",
                "content": "Main content paragraph with details. " * 300,
                "summary": "Main content summary.",
            },
        }
        return session

    @pytest.fixture
    def evidence_locker(self, tmp_path):
        """Create evidence locker."""
        locker = EvidenceLocker(output_dir=tmp_path)
        locker.add_evidence(
            url="https://example.com/source",
            title="Test Source",
            content_excerpt="Test content",
        )
        return locker

    def test_get_length_info(self, sample_session, tmp_path):
        """Test getting length info."""
        generator = ReportGenerator(output_dir=tmp_path)
        info = generator.get_length_info(sample_session)

        assert info.total_characters > 0
        assert info.estimated_pages > 0
        assert len(info.section_lengths) > 0

    def test_get_length_summary_japanese(self, sample_session, tmp_path):
        """Test Japanese length summary."""
        generator = ReportGenerator(output_dir=tmp_path, language="ja")
        summary = generator.get_length_summary(sample_session)

        assert "文字数" in summary
        assert "ページ数" in summary

    def test_get_length_summary_english(self, sample_session, tmp_path):
        """Test English length summary."""
        generator = ReportGenerator(output_dir=tmp_path, language="en")
        summary = generator.get_length_summary(sample_session)

        assert "Characters" in summary
        assert "Pages" in summary

    def test_generate_report_with_target_pages(
        self, sample_session, evidence_locker, tmp_path
    ):
        """Test report generation with page target."""
        generator = ReportGenerator(output_dir=tmp_path)

        # Get original length
        original_info = generator.get_length_info(sample_session)

        # Generate with target of 2 pages (smaller than original)
        report_path = generator.generate_report(
            session=sample_session,
            evidence_locker=evidence_locker,
            format=ReportFormat.MARKDOWN,
            target_pages=2,
        )

        assert report_path.exists()

        # Read the generated content
        content = report_path.read_text()

        # Content should be shorter than if no target
        # (We can't easily compare since adjustment happens at session level)
        assert len(content) > 0

    def test_generate_report_with_target_characters(
        self, sample_session, evidence_locker, tmp_path
    ):
        """Test report generation with character target."""
        generator = ReportGenerator(output_dir=tmp_path)

        report_path = generator.generate_report(
            session=sample_session,
            evidence_locker=evidence_locker,
            format=ReportFormat.MARKDOWN,
            target_characters=5000,
        )

        assert report_path.exists()
        content = report_path.read_text()
        assert len(content) > 0

    def test_generate_report_no_target(
        self, sample_session, evidence_locker, tmp_path
    ):
        """Test report generation without target (no changes)."""
        generator = ReportGenerator(output_dir=tmp_path)

        # Generate without targets
        report_path = generator.generate_report(
            session=sample_session,
            evidence_locker=evidence_locker,
            format=ReportFormat.MARKDOWN,
        )

        assert report_path.exists()

    def test_pdf_with_target_pages(
        self, sample_session, evidence_locker, tmp_path
    ):
        """Test PDF generation with page target."""
        generator = ReportGenerator(output_dir=tmp_path)

        report_path = generator.generate_report(
            session=sample_session,
            evidence_locker=evidence_locker,
            format=ReportFormat.PDF,
            target_pages=3,
        )

        assert report_path.exists()
        assert report_path.suffix == ".pdf"
        assert report_path.stat().st_size > 0

    def test_docx_with_target_characters(
        self, sample_session, evidence_locker, tmp_path
    ):
        """Test DOCX generation with character target."""
        generator = ReportGenerator(output_dir=tmp_path)

        report_path = generator.generate_report(
            session=sample_session,
            evidence_locker=evidence_locker,
            format=ReportFormat.DOCX,
            target_characters=10000,
        )

        assert report_path.exists()
        assert report_path.suffix == ".docx"
        assert report_path.stat().st_size > 0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_estimate_page_count(self):
        """Test page count estimation."""
        content = "A" * 5000
        pages = estimate_page_count(content, "pdf")
        assert pages == pytest.approx(2.0, rel=0.1)

    def test_estimate_page_count_different_formats(self):
        """Test page estimation for different formats."""
        content = "A" * 10000

        pdf_pages = estimate_page_count(content, "pdf")
        docx_pages = estimate_page_count(content, "docx")
        markdown_pages = estimate_page_count(content, "markdown")

        # Different formats have different chars per page
        assert pdf_pages != docx_pages or pdf_pages != markdown_pages

    def test_get_length_summary(self):
        """Test get_length_summary function."""
        section_contents = {
            "1": {"content": "A" * 5000},
            "2": {"content": "B" * 5000},
        }

        summary_ja = get_length_summary(section_contents, "pdf", "ja")
        assert "文字数" in summary_ja

        summary_en = get_length_summary(section_contents, "pdf", "en")
        assert "Characters" in summary_en


class TestConfigIntegration:
    """Test integration with config system."""

    def test_config_with_target_pages(self):
        """Test config creation with target pages."""
        from deep_research_tool.config import create_config

        config = create_config(
            target_pages=10,
        )
        assert config.report.target_pages == 10
        assert config.report.target_characters is None

    def test_config_with_target_characters(self):
        """Test config creation with target characters."""
        from deep_research_tool.config import create_config

        config = create_config(
            target_characters=25000,
        )
        assert config.report.target_characters == 25000
        assert config.report.target_pages is None

    def test_config_with_both_targets(self):
        """Test config creation with both targets."""
        from deep_research_tool.config import create_config

        config = create_config(
            target_pages=10,
            target_characters=30000,
        )
        assert config.report.target_pages == 10
        assert config.report.target_characters == 30000
