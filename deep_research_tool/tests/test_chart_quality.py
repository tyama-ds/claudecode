"""
Tests for the chart quality gate.

The first three test classes are the field-reported symptoms, fixed as
permanent regression cases:
  1. year-vs-year identity lines (years extracted as values)
  2. bar charts of identical quantities
  3. flat series from duplicated same-X points
"""

import unittest

from deep_research_tool.evidence.numerical_extractor import NumericalDataPoint
from deep_research_tool.report.chart_quality import (
    ChartQualityGate, Reason, is_year_like,
)


def dp(value, year=None, unit="億円", metric="市場規模", subject="国内市場",
       source="http://a.example", conf=0.8, derived=False):
    p = NumericalDataPoint(
        value=value, unit=unit, metric_name=metric, subject=subject,
        year=year, source_url=source, extraction_confidence=conf,
    )
    p.is_derived = derived
    return p


class TestSymptom1YearAsValue(unittest.TestCase):
    """症状①: X軸もY軸も 2018, 2019, 2020 の折れ線."""

    def test_year_vs_year_identity_line_rejected(self):
        gate = ChartQualityGate()
        points = [dp(2018, year=2018, unit=""), dp(2019, year=2019, unit=""),
                  dp(2020, year=2020, unit="")]
        result = gate.evaluate(points, purpose="show_trend", title="年 vs 年")
        self.assertFalse(result.passed)
        self.assertIn(Reason.YEAR_AS_VALUE, result.reasons)
        self.assertFalse(result.demote_to_table)  # poisoned data: no table either

    def test_identity_detected_even_with_year_unit(self):
        gate = ChartQualityGate()
        points = [dp(2021, year=2021, unit="年"), dp(2022, year=2022, unit="年")]
        result = gate.evaluate(points, purpose="show_trend")
        self.assertFalse(result.passed)

    def test_is_year_like(self):
        self.assertTrue(is_year_like(2020, ""))
        self.assertTrue(is_year_like(2020, "年"))
        self.assertFalse(is_year_like(2020, "億円"))   # real unit -> real value
        self.assertFalse(is_year_like(2020.5, ""))
        self.assertFalse(is_year_like(120, ""))

    def test_values_that_happen_to_be_year_sized_with_units_pass(self):
        """2018億円 etc. must NOT be mistaken for years."""
        gate = ChartQualityGate()
        points = [dp(1950, year=2021), dp(2020, year=2022), dp(2110, year=2023)]
        result = gate.evaluate(points, purpose="show_trend")
        self.assertTrue(result.passed)


class TestSymptom2IdenticalBars(unittest.TestCase):
    """症状②: 明らかに同量の棒グラフ."""

    def test_all_equal_bars_rejected(self):
        gate = ChartQualityGate()
        points = [dp(100, subject="A社"), dp(100, subject="B社"),
                  dp(100, subject="C社")]
        result = gate.evaluate(points, purpose="compare_values")
        self.assertFalse(result.passed)
        self.assertIn(Reason.ZERO_VARIANCE, result.reasons)
        self.assertFalse(result.demote_to_table)

    def test_near_constant_rejected(self):
        gate = ChartQualityGate()
        points = [dp(1000.0, subject="A社"), dp(1000.1, subject="B社"),
                  dp(999.9, subject="C社")]
        result = gate.evaluate(points, purpose="compare_values")
        self.assertFalse(result.passed)
        self.assertIn(Reason.NEAR_CONSTANT, result.reasons)

    def test_meaningful_comparison_passes(self):
        gate = ChartQualityGate()
        points = [dp(120, subject="A社"), dp(80, subject="B社"),
                  dp(45, subject="C社")]
        result = gate.evaluate(points, purpose="compare_values")
        self.assertTrue(result.passed)
        self.assertGreater(result.info_score, 0.3)


class TestSymptom3DuplicateFlatSeries(unittest.TestCase):
    """症状③: 同じ年(X軸)でY軸が一定."""

    def test_same_year_duplicates_collapse_then_reject(self):
        gate = ChartQualityGate()
        # the same fact extracted 4 times -> one aggregated point -> too few
        points = [dp(500, year=2023) for _ in range(4)]
        result = gate.evaluate(points, purpose="show_trend")
        self.assertFalse(result.passed)
        self.assertEqual(len(result.points), 1)   # collapsed
        self.assertIn(Reason.TOO_FEW_POINTS, result.reasons)

    def test_duplicates_collapse_but_real_trend_survives(self):
        gate = ChartQualityGate()
        points = [dp(100, year=2021), dp(100, year=2021),   # duplicate
                  dp(140, year=2022), dp(190, year=2023)]
        result = gate.evaluate(points, purpose="show_trend")
        self.assertTrue(result.passed)
        self.assertEqual(len(result.points), 3)
        self.assertEqual([p.year for p in result.points], [2021, 2022, 2023])

    def test_conflicting_duplicates_prefer_confidence(self):
        gate = ChartQualityGate()
        low = dp(999, year=2021, conf=0.3)
        high = dp(100, year=2021, conf=0.9)
        points = [low, high, dp(140, year=2022), dp(190, year=2023)]
        result = gate.evaluate(points, purpose="show_trend")
        self.assertTrue(result.passed)
        y2021 = next(p for p in result.points if p.year == 2021)
        self.assertEqual(y2021.value, 100)   # high-confidence wins


class TestStructuralGates(unittest.TestCase):
    def test_two_point_trend_demoted_to_table(self):
        gate = ChartQualityGate()
        points = [dp(100, year=2021), dp(150, year=2022)]
        result = gate.evaluate(points, purpose="show_trend")
        self.assertFalse(result.passed)
        self.assertIn(Reason.TOO_FEW_POINTS, result.reasons)
        self.assertTrue(result.demote_to_table)   # numbers still table-worthy

    def test_mixed_units_rejected(self):
        gate = ChartQualityGate()
        points = [dp(100, year=2021, unit="億円"), dp(15, year=2022, unit="%"),
                  dp(120, year=2023, unit="億円")]
        result = gate.evaluate(points, purpose="show_trend")
        self.assertFalse(result.passed)
        self.assertIn(Reason.MIXED_UNITS, result.reasons)

    def test_derived_only_rejected(self):
        gate = ChartQualityGate()
        points = [dp(100, year=2021), dp(110, year=2022, derived=True),
                  dp(121, year=2023, derived=True), dp(133, year=2024, derived=True)]
        result = gate.evaluate(points, purpose="show_trend")
        self.assertFalse(result.passed)
        self.assertIn(Reason.DERIVED_ONLY, result.reasons)

    def test_pie_must_sum_to_100(self):
        gate = ChartQualityGate()
        bad = [dp(40, subject="A", unit="%"), dp(30, subject="B", unit="%")]
        result = gate.evaluate(bad, purpose="show_composition")
        self.assertFalse(result.passed)
        self.assertIn(Reason.PIE_SUM_INVALID, result.reasons)

        good = [dp(55, subject="A", unit="%"), dp(30, subject="B", unit="%"),
                dp(15, subject="C", unit="%")]
        result = gate.evaluate(good, purpose="show_composition")
        self.assertTrue(result.passed)

    def test_healthy_time_series_passes(self):
        gate = ChartQualityGate()
        points = [dp(100, year=2020, source="http://a"),
                  dp(135, year=2021, source="http://b"),
                  dp(180, year=2022, source="http://c"),
                  dp(220, year=2023, source="http://a")]
        result = gate.evaluate(points, purpose="show_trend")
        self.assertTrue(result.passed)
        self.assertGreater(result.info_score, 0.5)

    def test_rejection_summary(self):
        gate = ChartQualityGate()
        gate.evaluate([dp(2020, year=2020, unit="")], purpose="show_trend",
                      title="bad1")
        gate.evaluate([dp(1, subject="A"), dp(1, subject="B")],
                      purpose="compare_values", title="bad2")
        summary = gate.rejection_summary()
        self.assertIn("2件", summary)
        self.assertIn("年そのもの", summary)


if __name__ == "__main__":
    unittest.main()
