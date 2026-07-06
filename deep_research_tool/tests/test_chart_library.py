"""
Tests for chart library selection (matplotlib / seaborn).
"""

import pytest
from unittest.mock import Mock

from deep_research_tool.report.figure_table_generator import FigureTableGenerator


class TestChartLibrary:
    def test_default_is_matplotlib(self, tmp_path):
        gen = FigureTableGenerator(output_dir=tmp_path)
        assert gen.chart_library == "matplotlib"
        assert gen._get_seaborn() is None  # never imports seaborn

    def test_seaborn_selection_stored(self, tmp_path):
        gen = FigureTableGenerator(output_dir=tmp_path, chart_library="seaborn")
        assert gen.chart_library == "seaborn"

    def test_apply_chart_style_never_raises(self, tmp_path):
        """Style application works with either library selection.

        With seaborn missing it must fall back to matplotlib silently.
        """
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        for library in ("matplotlib", "seaborn"):
            gen = FigureTableGenerator(output_dir=tmp_path, chart_library=library)
            sns = gen._apply_chart_style(plt, fm)
            try:
                import seaborn  # noqa: F401
                expected_module = library == "seaborn"
            except ImportError:
                expected_module = False
            assert (sns is not None) == expected_module

    def test_config_wiring(self):
        from deep_research_tool.config import create_config
        config = create_config(chart_library="seaborn")
        assert config.report.chart_library == "seaborn"
