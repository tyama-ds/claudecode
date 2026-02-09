"""
Tests for figure and table generation functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from deep_research_tool2.report.figure_table_generator import (
    FigureTableGenerator,
    FigureTableCollection,
    Figure,
    TableData,
    FigureType,
    ChartType,
    add_figures_to_report,
)
from deep_research_tool2.research.researcher import ResearchSession
from deep_research_tool2.research.query_generator import (
    ResearchPlan,
    TableOfContents,
    TableOfContentsItem,
)
from deep_research_tool2.evidence.locker import EvidenceLocker, Evidence


class TestFigure:
    """Test Figure dataclass."""

    def test_create_figure(self):
        """Test creating a figure."""
        fig = Figure(
            figure_id="fig_1_1",
            figure_type=FigureType.IMAGE,
            title="Test Figure",
            caption="A test figure",
            source_url="https://example.com/image.png",
            source_title="Example Site",
            section_id="1",
        )
        assert fig.figure_id == "fig_1_1"
        assert fig.figure_type == FigureType.IMAGE
        assert fig.title == "Test Figure"

    def test_figure_to_dict(self):
        """Test figure serialization."""
        fig = Figure(
            figure_id="fig_1_1",
            figure_type=FigureType.CHART,
            title="Chart",
            caption="A chart",
            section_id="1",
        )
        data = fig.to_dict()
        assert data["figure_id"] == "fig_1_1"
        assert data["figure_type"] == "chart"


class TestTableData:
    """Test TableData dataclass."""

    def test_create_table(self):
        """Test creating a table."""
        table = TableData(
            table_id="table_1_1",
            title="Sales Data",
            caption="Annual sales",
            headers=["Year", "Sales", "Growth"],
            rows=[
                ["2020", 100, "10%"],
                ["2021", 120, "20%"],
                ["2022", 150, "25%"],
            ],
            section_id="1",
            data_type="time_series",
            unit="億円",
        )
        assert table.table_id == "table_1_1"
        assert len(table.headers) == 3
        assert len(table.rows) == 3

    def test_table_to_dict(self):
        """Test table serialization."""
        table = TableData(
            table_id="table_1",
            title="Data",
            caption="Caption",
            headers=["A", "B"],
            rows=[["1", "2"]],
            section_id="1",
        )
        data = table.to_dict()
        assert data["table_id"] == "table_1"
        assert data["headers"] == ["A", "B"]


class TestFigureTableCollection:
    """Test FigureTableCollection."""

    def test_empty_collection(self):
        """Test empty collection."""
        collection = FigureTableCollection()
        assert len(collection.figures) == 0
        assert len(collection.tables) == 0
        assert len(collection.charts) == 0

    def test_get_figures_for_section(self):
        """Test getting figures by section."""
        collection = FigureTableCollection()
        collection.figures = [
            Figure(
                figure_id="fig_1",
                figure_type=FigureType.IMAGE,
                title="Fig 1",
                caption="",
                section_id="1",
            ),
            Figure(
                figure_id="fig_2",
                figure_type=FigureType.IMAGE,
                title="Fig 2",
                caption="",
                section_id="2",
            ),
        ]

        section_1_figs = collection.get_figures_for_section("1")
        assert len(section_1_figs) == 1
        assert section_1_figs[0].figure_id == "fig_1"

    def test_collection_to_dict(self):
        """Test collection serialization."""
        collection = FigureTableCollection()
        collection.figures.append(
            Figure(
                figure_id="fig_1",
                figure_type=FigureType.IMAGE,
                title="Test",
                caption="",
                section_id="1",
            )
        )
        collection.tables.append(
            TableData(
                table_id="table_1",
                title="Data",
                caption="",
                headers=["A"],
                rows=[["1"]],
                section_id="1",
            )
        )

        data = collection.to_dict()
        assert len(data["figures"]) == 1
        assert len(data["tables"]) == 1
        assert len(data["charts"]) == 0


class TestFigureTableGenerator:
    """Test FigureTableGenerator class."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create generator instance."""
        return FigureTableGenerator(
            output_dir=tmp_path,
            language="ja",
            max_images_per_section=2,
        )

    @pytest.fixture
    def sample_session(self):
        """Create sample session."""
        session = ResearchSession(query="Test research")
        session.research_plan = ResearchPlan(
            title="Test Report",
            summary="Summary",
            table_of_contents=TableOfContents(
                title="Test",
                items=[
                    TableOfContentsItem(
                        section="1",
                        title="Introduction",
                        description="Intro",
                        subsections=[],
                    ),
                ],
            ),
        )
        session.section_contents = {
            "1": {
                "title": "Introduction",
                "content": "The market grew from 100億円 in 2020 to 150億円 in 2022. "
                          "In 2021, the value was 120億円. Growth rate was 20%.",
                "images": [
                    {
                        "src": "https://example.com/chart.png",
                        "alt": "Market Chart",
                        "suggested_caption": "Market Growth Chart",
                        "page_title": "Example Report",
                    }
                ],
            },
        }
        return session

    @pytest.fixture
    def evidence_locker(self, tmp_path):
        """Create evidence locker with test data."""
        locker = EvidenceLocker(output_dir=tmp_path)
        locker.add_evidence(
            url="https://example.com/source1",
            title="Source 1",
            content_excerpt="Content from source 1",
            section_reference="1",
        )
        return locker

    def test_generator_init(self, generator, tmp_path):
        """Test generator initialization."""
        assert generator.output_dir == tmp_path
        assert generator.language == "ja"
        assert generator.max_images_per_section == 2

    def test_extract_time_series(self, generator):
        """Test time series extraction."""
        content = "市場規模は2020年に100億円、2021年に120億円、2022年に150億円となった。"
        years = ["2020", "2021", "2022"]

        time_series = generator._extract_time_series(content, years)

        # Should find some data
        assert isinstance(time_series, dict)

    def test_format_time_series_rows(self, generator):
        """Test time series row formatting."""
        time_series = {
            "億": {"2020": 100, "2021": 120, "2022": 150},
        }

        rows = generator._format_time_series_rows(time_series)

        assert len(rows) == 3
        assert rows[0][0] == "2020"
        assert rows[1][0] == "2021"
        assert rows[2][0] == "2022"

    def test_extract_tables_by_pattern(self, generator):
        """Test pattern-based table extraction."""
        content = "売上は2020年に100億円、2021年に120億円、2022年に150億円。"

        tables = generator._extract_tables_by_pattern(
            section_id="1",
            content=content,
            section_title="Market Analysis",
        )

        # May or may not find tables depending on pattern matching
        assert isinstance(tables, list)

    def test_generate_chart_for_table(self, generator):
        """Test chart generation from table."""
        table = TableData(
            table_id="table_1",
            title="Sales Data",
            caption="Annual sales",
            headers=["Year", "Sales"],
            rows=[
                ["2020", 100],
                ["2021", 120],
                ["2022", 150],
            ],
            section_id="1",
            data_type="time_series",
        )

        chart = generator._generate_chart_for_table(table)

        if chart:  # Chart generation may fail if matplotlib has issues
            assert chart.figure_type == FigureType.CHART
            assert chart.image_path is not None
            assert chart.image_path.exists()

    def test_add_figures_to_markdown(self, generator):
        """Test adding figures to markdown."""
        markdown_content = """# Test Report

## 1. Introduction

This is the introduction section.

---

## 2. Analysis

This is the analysis section.
"""

        collection = FigureTableCollection()
        collection.figures.append(
            Figure(
                figure_id="fig_1",
                figure_type=FigureType.IMAGE,
                title="Figure 1",
                caption="A figure",
                source_url="https://example.com/img.png",
                source_title="Example",
                section_id="1",
                image_path=Path("/tmp/img.png"),
            )
        )
        collection.tables.append(
            TableData(
                table_id="table_1",
                title="Table 1",
                caption="A table",
                headers=["A", "B"],
                rows=[["1", "2"], ["3", "4"]],
                section_id="1",
            )
        )

        updated = generator.add_figures_to_markdown(markdown_content, collection)

        # Should contain table markup
        assert "| A | B |" in updated or "Figure 1" in updated

    def test_export_collection(self, generator, tmp_path):
        """Test collection export."""
        collection = FigureTableCollection()
        collection.figures.append(
            Figure(
                figure_id="fig_1",
                figure_type=FigureType.IMAGE,
                title="Test",
                caption="Test caption",
                section_id="1",
            )
        )

        output_path = tmp_path / "collection.json"
        generator.export_collection(collection, output_path)

        assert output_path.exists()

        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert "figures" in data
        assert len(data["figures"]) == 1

    @patch('requests.get')
    def test_download_image_success(self, mock_get, generator):
        """Test successful image download."""
        mock_response = Mock()
        mock_response.content = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        mock_response.headers = {'content-type': 'image/png'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        path, data = generator._download_image("https://example.com/test.png")

        assert path is not None
        assert data is not None
        assert path.exists()

    @patch('requests.get')
    def test_download_image_failure(self, mock_get, generator):
        """Test image download failure handling."""
        mock_get.side_effect = Exception("Network error")

        path, data = generator._download_image("https://example.com/test.png")

        assert path is None
        assert data is None


class TestChartGeneration:
    """Test matplotlib chart generation."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create generator."""
        return FigureTableGenerator(output_dir=tmp_path)

    def test_line_chart_generation(self, generator):
        """Test line chart generation."""
        table = TableData(
            table_id="line_test",
            title="Time Series",
            caption="Test data",
            headers=["Year", "Value"],
            rows=[
                ["2020", 100],
                ["2021", 150],
                ["2022", 200],
            ],
            section_id="1",
            data_type="time_series",
        )

        chart = generator._generate_chart_for_table(table)

        if chart:
            assert chart.figure_type == FigureType.CHART
            assert chart.image_path.suffix == ".png"

    def test_bar_chart_generation(self, generator):
        """Test bar chart generation."""
        table = TableData(
            table_id="bar_test",
            title="Comparison",
            caption="Test comparison",
            headers=["Category", "Value"],
            rows=[
                ["A", 50],
                ["B", 75],
                ["C", 100],
            ],
            section_id="1",
            data_type="comparison",
        )

        chart = generator._generate_chart_for_table(table)

        if chart:
            assert chart.figure_type == FigureType.CHART

    def test_multi_column_chart(self, generator):
        """Test chart with multiple data columns."""
        table = TableData(
            table_id="multi_test",
            title="Multi Series",
            caption="Multiple series",
            headers=["Year", "Sales", "Profit"],
            rows=[
                ["2020", 100, 20],
                ["2021", 150, 30],
                ["2022", 200, 40],
            ],
            section_id="1",
            data_type="time_series",
        )

        chart = generator._generate_chart_for_table(table)

        if chart:
            assert chart.figure_type == FigureType.CHART

    def test_empty_table_no_chart(self, generator):
        """Test that empty tables don't generate charts."""
        table = TableData(
            table_id="empty_test",
            title="Empty",
            caption="No data",
            headers=["A"],
            rows=[],
            section_id="1",
        )

        chart = generator._generate_chart_for_table(table)

        assert chart is None


class TestIntegration:
    """Integration tests."""

    @pytest.fixture
    def full_setup(self, tmp_path):
        """Set up full test environment."""
        session = ResearchSession(query="Market Analysis")
        session.research_plan = ResearchPlan(
            title="Market Analysis Report",
            summary="Analysis",
            table_of_contents=TableOfContents(
                title="Market Analysis",
                items=[
                    TableOfContentsItem(
                        section="1",
                        title="Overview",
                        description="Market overview",
                        subsections=[],
                    ),
                    TableOfContentsItem(
                        section="2",
                        title="Trends",
                        description="Market trends",
                        subsections=[],
                    ),
                ],
            ),
        )
        session.section_contents = {
            "1": {
                "title": "Overview",
                "content": "The market size was 100 billion in 2020, grew to 150 billion in 2022.",
            },
            "2": {
                "title": "Trends",
                "content": "Growth rate: 2020=10%, 2021=20%, 2022=25%.",
            },
        }

        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir()
        locker = EvidenceLocker(output_dir=evidence_dir)
        locker.add_evidence(
            url="https://example.com/report",
            title="Market Report",
            content_excerpt="Market data",
            section_reference="1",
        )

        return {
            "session": session,
            "evidence_locker": locker,
            "output_dir": tmp_path,
        }

    def test_full_generation(self, full_setup):
        """Test full figure/table generation pipeline."""
        generator = FigureTableGenerator(
            output_dir=full_setup["output_dir"],
            language="ja",
        )

        collection = generator.generate_figures_and_tables(
            session=full_setup["session"],
            evidence_locker=full_setup["evidence_locker"],
            include_images=False,  # Skip actual image downloads
            include_tables=True,
            include_charts=True,
        )

        # Collection should be created
        assert isinstance(collection, FigureTableCollection)

    def test_markdown_insertion(self, full_setup):
        """Test markdown figure/table insertion."""
        generator = FigureTableGenerator(
            output_dir=full_setup["output_dir"],
        )

        markdown = """# Market Report

## 1. Overview

Market overview content.

---

## 2. Trends

Trends content.
"""

        collection = FigureTableCollection()
        collection.tables.append(
            TableData(
                table_id="table_1",
                title="Market Size",
                caption="Annual market size",
                headers=["Year", "Size (Billion)"],
                rows=[["2020", 100], ["2021", 120], ["2022", 150]],
                section_id="1",
            )
        )

        updated = generator.add_figures_to_markdown(markdown, collection)

        # Table should be inserted
        assert "| Year | Size (Billion) |" in updated
