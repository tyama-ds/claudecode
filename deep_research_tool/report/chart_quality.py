"""
Chart quality gate - deterministic tests that reject uninformative charts.

The chart pipeline used to draw anything with >= 2 data points, which
produced meaningless output: year-vs-year identity lines (the year itself
extracted as the value), bar charts of identical quantities, and flat
series built from duplicated points. This module sits between chart
recommendation and rendering and only lets through candidates that carry
actual information. All tests are deterministic (no LLM), fast, and each
rejection carries a machine-readable reason code so a zero-chart run can
explain itself.

Usage:
    gate = ChartQualityGate()
    result = gate.evaluate(points, purpose="show_trend")
    if result.passed:
        render(result.points)       # points are aggregated/cleaned
    elif result.demote_to_table:
        make_table(result.points)   # numeric value without chart value
"""

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple


# --- thresholds -----------------------------------------------------------

YEAR_MIN, YEAR_MAX = 1900, 2100
MIN_CV = 0.01                    # coefficient of variation below this = flat
MIN_LINE_POINTS = 3              # line charts need at least 3 distinct X
MIN_BAR_CATEGORIES = 2           # bar charts need at least 2 categories
VALUE_ABS_MIN, VALUE_ABS_MAX = 1e-6, 1e15   # sane numeric range
PIE_SUM_MIN, PIE_SUM_MAX = 90.0, 110.0      # composition must sum to ~100%
MAX_DERIVED_RATIO = 0.5          # at most half the points may be derived


# --- rejection reason codes ------------------------------------------------

class Reason:
    """Machine-readable rejection reason codes."""
    IDENTITY_XY = "identity_xy"              # Y values are the X values (year-vs-year)
    YEAR_AS_VALUE = "year_as_value"          # values look like calendar years
    ZERO_VARIANCE = "zero_variance"          # every Y identical
    NEAR_CONSTANT = "near_constant"          # CV below threshold
    TOO_FEW_POINTS = "too_few_points"        # not enough distinct X after aggregation
    MIXED_UNITS = "mixed_units"              # more than one unit in a series
    VALUE_RANGE = "value_range_abnormal"     # magnitudes outside sane range
    DERIVED_ONLY = "derived_only"            # mostly interpolated/derived points
    PIE_SUM_INVALID = "pie_sum_invalid"      # composition doesn't sum to ~100%
    NO_DATA = "no_data"

    # Human-readable (Japanese) descriptions for warnings
    DESCRIPTIONS = {
        IDENTITY_XY: "X軸とY軸が同じ値（年 vs 年など）",
        YEAR_AS_VALUE: "年そのものが値として抽出されている",
        ZERO_VARIANCE: "全ての値が同一（情報量ゼロ）",
        NEAR_CONSTANT: "値がほぼ一定（変動が閾値未満）",
        TOO_FEW_POINTS: "集約後のデータ点が不足",
        MIXED_UNITS: "1つの系列に複数の単位が混在",
        VALUE_RANGE: "値の桁が異常",
        DERIVED_ONLY: "補間・派生値が過半",
        PIE_SUM_INVALID: "構成比の合計が100%前後でない",
        NO_DATA: "データなし",
    }


def is_year_like(value: float, unit: str = "") -> bool:
    """Whether a numeric value is (almost certainly) a calendar year.

    Integer in [1900, 2100] with no unit, or explicitly unit '年'/'year'.
    Values with a real unit ('億円', '%', 'users') are never year-like.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if v != int(v):
        return False
    if not (YEAR_MIN <= v <= YEAR_MAX):
        return False
    u = (unit or "").strip().lower()
    return u in ("", "年", "年度", "year", "fy")


@dataclass
class GateResult:
    """Outcome of the quality gate for one chart candidate."""
    passed: bool
    reasons: List[str] = field(default_factory=list)
    points: List = field(default_factory=list)   # aggregated/cleaned points
    info_score: float = 0.0
    demote_to_table: bool = False

    def describe(self) -> str:
        return ", ".join(
            Reason.DESCRIPTIONS.get(r, r) for r in self.reasons) or "OK"


class ChartQualityGate:
    """Deterministic per-candidate quality tests.

    evaluate() runs the applicable tests for the chart purpose and returns
    a GateResult whose `points` are the aggregated series (same-X duplicates
    collapsed) that should be used for rendering when passed.
    """

    def __init__(self, min_cv: float = MIN_CV):
        self.min_cv = min_cv
        self.rejections: List[Dict] = []   # log of rejected candidates

    # --- aggregation -----------------------------------------------------

    @staticmethod
    def _x_key(dp, purpose: str):
        """The X identity of a point for duplicate collapsing."""
        if purpose in ("show_trend", "show_growth", "forecast"):
            return (dp.year, dp.quarter or "")
        # comparisons / compositions: one bar per subject
        return (dp.subject or "").strip().lower()

    def aggregate(self, points: List, purpose: str) -> List:
        """Collapse duplicate-X points before judging the series.

        Duplicates with the same X (same year, or same subject) are reduced
        to a single point: the highest-confidence one wins; ties average.
        Without this, repeated extractions of the same fact rendered as a
        flat multi-point series.
        """
        groups: Dict = {}
        for dp in points:
            groups.setdefault(self._x_key(dp, purpose), []).append(dp)

        aggregated = []
        for group in groups.values():
            if len(group) == 1:
                aggregated.append(group[0])
                continue
            best_conf = max(g.combined_confidence for g in group)
            best = [g for g in group
                    if g.combined_confidence >= best_conf - 1e-9]
            chosen = best[0]
            if len(best) > 1:
                values = [g.value for g in best]
                if len(set(values)) > 1:
                    chosen.value = mean(values)
            aggregated.append(chosen)

        aggregated.sort(key=lambda d: (d.year or 0, d.quarter or "",
                                       (d.subject or "")))
        return aggregated

    # --- individual tests -------------------------------------------------

    @staticmethod
    def _test_year_as_value(points: List) -> bool:
        """Fail when the majority of values are calendar years."""
        year_like = sum(1 for dp in points if is_year_like(dp.value, dp.unit))
        return year_like <= len(points) / 2

    @staticmethod
    def _test_identity_xy(points: List) -> bool:
        """Fail when the Y values ARE the X values (year-vs-year line)."""
        xs = [dp.year for dp in points if dp.year is not None]
        if len(xs) != len(points) or not xs:
            return True
        ys = [dp.value for dp in points]
        return not all(abs(float(y) - float(x)) < 1e-9 for x, y in zip(xs, ys))

    def _test_variation(self, points: List) -> Optional[str]:
        """Return a reason code when the series carries no variation."""
        values = [float(dp.value) for dp in points]
        if len(set(values)) <= 1:
            return Reason.ZERO_VARIANCE
        m = mean(values)
        if abs(m) > 1e-12:
            cv = pstdev(values) / abs(m)
            if cv < self.min_cv:
                return Reason.NEAR_CONSTANT
        return None

    @staticmethod
    def _test_units(points: List) -> bool:
        units = {(dp.unit or "").strip() for dp in points}
        return len(units) <= 1

    @staticmethod
    def _test_value_range(points: List) -> bool:
        for dp in points:
            v = abs(float(dp.value))
            if v != 0 and not (VALUE_ABS_MIN <= v <= VALUE_ABS_MAX):
                return False
        return True

    @staticmethod
    def _test_derived_ratio(points: List) -> bool:
        derived = sum(1 for dp in points if getattr(dp, "is_derived", False))
        return derived <= len(points) * MAX_DERIVED_RATIO

    # --- scoring ----------------------------------------------------------

    @staticmethod
    def info_score(points: List) -> float:
        """0-1 informativeness: variation + point count + source diversity."""
        values = [float(dp.value) for dp in points]
        m = mean(values)
        cv = (pstdev(values) / abs(m)) if abs(m) > 1e-12 else 1.0
        variation = min(1.0, cv / 0.5)                 # CV 50%+ = full marks
        count = min(1.0, len(points) / 6.0)            # 6+ points = full marks
        sources = len({dp.source_url for dp in points if dp.source_url})
        diversity = min(1.0, sources / 3.0)            # 3+ sources = full marks
        return round(0.5 * variation + 0.3 * count + 0.2 * diversity, 3)

    # --- main entry ---------------------------------------------------------

    def evaluate(self, points: List, purpose: str,
                 title: str = "", section_id: str = "") -> GateResult:
        """Run all applicable tests; aggregate first, then judge."""
        if not points:
            return self._reject([], [Reason.NO_DATA], title, section_id)

        reasons: List[str] = []

        # Layer order matters: year-as-value / identity are data poisoning
        # (never demote those to tables), the rest are shape problems.
        if not self._test_year_as_value(points):
            reasons.append(Reason.YEAR_AS_VALUE)
        if purpose in ("show_trend", "show_growth", "forecast") and \
                not self._test_identity_xy(points):
            reasons.append(Reason.IDENTITY_XY)
        if reasons:
            return self._reject(points, reasons, title, section_id)

        if not self._test_units(points):
            return self._reject(points, [Reason.MIXED_UNITS], title, section_id)

        aggregated = self.aggregate(points, purpose)

        # structural minimums (after aggregation)
        min_points = (MIN_LINE_POINTS
                      if purpose in ("show_trend", "forecast")
                      else MIN_BAR_CATEGORIES)
        demote = False
        if len(aggregated) < min_points:
            # 2 aggregated points of a trend are worth a table, not a chart
            demote = len(aggregated) >= 2
            reasons.append(Reason.TOO_FEW_POINTS)

        variation_reason = self._test_variation(aggregated)
        if variation_reason:
            reasons.append(variation_reason)
            demote = False   # identical values aren't table-worthy either

        if not self._test_value_range(aggregated):
            reasons.append(Reason.VALUE_RANGE)
            demote = False
        if not self._test_derived_ratio(aggregated):
            reasons.append(Reason.DERIVED_ONLY)

        if purpose == "show_composition":
            total = sum(float(dp.value) for dp in aggregated)
            unit = (aggregated[0].unit or "") if aggregated else ""
            if "%" in unit or "％" in unit:
                if not (PIE_SUM_MIN <= total <= PIE_SUM_MAX):
                    reasons.append(Reason.PIE_SUM_INVALID)

        if reasons:
            return self._reject(aggregated, reasons, title, section_id,
                                demote_to_table=demote)

        return GateResult(
            passed=True,
            points=aggregated,
            info_score=self.info_score(aggregated),
        )

    def _reject(self, points: List, reasons: List[str], title: str,
                section_id: str, demote_to_table: bool = False) -> GateResult:
        self.rejections.append({
            "title": title,
            "section_id": section_id,
            "reasons": reasons,
            "n_points": len(points),
        })
        return GateResult(
            passed=False,
            reasons=reasons,
            points=points,
            demote_to_table=demote_to_table,
        )

    # --- reporting ----------------------------------------------------------

    def rejection_summary(self) -> str:
        """Japanese summary of why candidates were rejected (for warnings)."""
        if not self.rejections:
            return ""
        counts: Dict[str, int] = {}
        for r in self.rejections:
            for code in r["reasons"]:
                counts[code] = counts.get(code, 0) + 1
        parts = [f"{Reason.DESCRIPTIONS.get(code, code)}: {n}件"
                 for code, n in sorted(counts.items(), key=lambda x: -x[1])]
        return (f"{len(self.rejections)}件のチャート候補を品質検定で棄却"
                f"（{'、'.join(parts)}）")
