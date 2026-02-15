"""
Intelligent Chart Analyzer for Deep Research Tool.

Analyzes numerical data to determine meaningful charts and insights.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from ..evidence.numerical_extractor import (
    NumericalDataStore,
    NumericalDataPoint,
    DataType,
    MetricCategory,
    DerivedMetricsCalculator,
)


logger = logging.getLogger(__name__)


class ChartPurpose(str, Enum):
    """Purpose/story of a chart."""
    SHOW_TREND = "show_trend"           # Time series trend
    COMPARE_VALUES = "compare_values"    # Comparison between entities
    SHOW_COMPOSITION = "show_composition"  # Parts of a whole
    SHOW_GROWTH = "show_growth"          # Growth/change
    SHOW_CORRELATION = "show_correlation"  # Relationship between variables
    SHOW_DISTRIBUTION = "show_distribution"  # Distribution of values
    SHOW_RANKING = "show_ranking"        # Ranking/ordering
    FORECAST = "forecast"                # Future projection


class RecommendedChartType(str, Enum):
    """Chart types that can be recommended."""
    LINE = "line"
    BAR = "bar"
    HORIZONTAL_BAR = "horizontal_bar"
    STACKED_BAR = "stacked_bar"
    PIE = "pie"
    AREA = "area"
    SCATTER = "scatter"
    COMBO = "combo"  # Line + Bar combination


@dataclass
class ChartInsight:
    """An insight that can be derived from a chart."""
    insight_type: str           # "trend", "comparison", "peak", "growth", etc.
    description: str            # Human-readable description
    key_values: Dict[str, Any]  # Key numbers supporting this insight
    confidence: float           # Confidence in this insight
    language: str = "ja"

    def to_narrative(self) -> str:
        """Convert insight to narrative text."""
        return self.description


@dataclass
class ChartRecommendation:
    """A recommended chart with its data and insights."""

    # Identification
    chart_id: str
    title: str
    subtitle: str = ""

    # Chart specification
    chart_type: RecommendedChartType = RecommendedChartType.BAR
    purpose: ChartPurpose = ChartPurpose.COMPARE_VALUES

    # Data
    data_points: List[NumericalDataPoint] = field(default_factory=list)
    x_axis_label: str = ""
    y_axis_label: str = ""
    unit: str = ""

    # Insights
    insights: List[ChartInsight] = field(default_factory=list)
    main_message: str = ""      # Primary takeaway

    # Quality metrics
    data_confidence: float = 0.0    # Average confidence of data
    informativeness: float = 0.0    # How informative this chart is
    priority_score: float = 0.0     # Overall priority for inclusion

    # Source tracking
    section_id: str = ""
    source_urls: List[str] = field(default_factory=list)

    # Missing data info
    has_missing_data: bool = False
    missing_data_filled: bool = False
    filled_data_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chart_id": self.chart_id,
            "title": self.title,
            "subtitle": self.subtitle,
            "chart_type": self.chart_type.value,
            "purpose": self.purpose.value,
            "data_points": [dp.to_dict() for dp in self.data_points],
            "x_axis_label": self.x_axis_label,
            "y_axis_label": self.y_axis_label,
            "unit": self.unit,
            "insights": [
                {
                    "type": ins.insight_type,
                    "description": ins.description,
                    "key_values": ins.key_values,
                    "confidence": ins.confidence,
                }
                for ins in self.insights
            ],
            "main_message": self.main_message,
            "data_confidence": self.data_confidence,
            "informativeness": self.informativeness,
            "priority_score": self.priority_score,
            "section_id": self.section_id,
            "source_urls": self.source_urls,
            "has_missing_data": self.has_missing_data,
            "missing_data_filled": self.missing_data_filled,
        }


class ChartAnalyzer:
    """
    Analyzes numerical data to recommend meaningful charts.

    Uses LLM for intelligent analysis of what stories the data tells.
    """

    def __init__(
        self,
        llm_client=None,
        language: str = "ja",
        min_confidence: float = 0.5,
        use_llm_analysis: bool = True,
        fill_missing_data: bool = True,
        max_charts_per_section: int = 3,
    ):
        """
        Initialize analyzer.

        Args:
            llm_client: LLM client for intelligent analysis
            language: Output language
            min_confidence: Minimum data confidence for inclusion
            use_llm_analysis: Use LLM for insight generation
            fill_missing_data: Fill missing data points (CAGR, interpolation)
            max_charts_per_section: Maximum charts per section
        """
        self.llm = llm_client
        self.language = language
        self.min_confidence = min_confidence
        self.use_llm_analysis = use_llm_analysis and llm_client is not None
        self.fill_missing_data = fill_missing_data
        self.max_charts_per_section = max_charts_per_section

    def analyze(
        self,
        store: NumericalDataStore,
        research_topic: str = "",
    ) -> List[ChartRecommendation]:
        """
        Analyze data store and generate chart recommendations.

        Args:
            store: Numerical data store
            research_topic: Research topic for context

        Returns:
            List of chart recommendations, prioritized
        """
        if not store.data_points:
            return []

        recommendations = []

        # Fill missing data if enabled
        if self.fill_missing_data:
            calculator = DerivedMetricsCalculator(store)
            derived = calculator.calculate_all()
            store.add_many(derived)

        # Filter by confidence
        high_conf_data = store.get_high_confidence(self.min_confidence)
        if not high_conf_data:
            high_conf_data = store.data_points  # Fall back to all data

        # 1. Find time series charts
        time_series_recs = self._analyze_time_series(store, research_topic)
        recommendations.extend(time_series_recs)

        # 2. Find comparison charts
        comparison_recs = self._analyze_comparisons(store, research_topic)
        recommendations.extend(comparison_recs)

        # 3. Find composition charts (pie/donut)
        composition_recs = self._analyze_compositions(store, research_topic)
        recommendations.extend(composition_recs)

        # 4. Find growth/change charts
        growth_recs = self._analyze_growth(store, research_topic)
        recommendations.extend(growth_recs)

        # Calculate priority scores
        for rec in recommendations:
            rec.priority_score = self._calculate_priority(rec)

        # Sort by priority
        recommendations.sort(key=lambda x: x.priority_score, reverse=True)

        # Use LLM to refine insights if available
        if self.use_llm_analysis and recommendations:
            recommendations = self._refine_with_llm(
                recommendations,
                research_topic,
            )

        return recommendations

    def _analyze_time_series(
        self,
        store: NumericalDataStore,
        research_topic: str,
    ) -> List[ChartRecommendation]:
        """Analyze for time series chart opportunities."""
        recommendations = []

        time_series_groups = store.get_time_series(min_points=2)

        for series in time_series_groups:
            if len(series) < 2:
                continue

            # Sort by year
            series = sorted(series, key=lambda x: x.year or 0)

            first = series[0]
            last = series[-1]

            # Determine chart type
            if len(series) >= 4:
                chart_type = RecommendedChartType.LINE
            else:
                chart_type = RecommendedChartType.BAR

            # Check if this is growth data (use area chart)
            if first.category == MetricCategory.GROWTH_RATE:
                chart_type = RecommendedChartType.BAR

            # Generate insights
            insights = []

            # Trend insight
            if first.normalized_value > 0 and last.normalized_value > 0:
                change = (last.normalized_value - first.normalized_value) / first.normalized_value * 100
                trend = "増加" if change > 0 else "減少"
                trend_en = "increased" if change > 0 else "decreased"

                if self.language == "ja":
                    desc = f"{first.subject}の{first.metric_name}は{first.year}年から{last.year}年にかけて{abs(change):.1f}%{trend}しました。"
                else:
                    desc = f"{first.subject} {first.metric_name} {trend_en} by {abs(change):.1f}% from {first.year} to {last.year}."

                insights.append(ChartInsight(
                    insight_type="trend",
                    description=desc,
                    key_values={
                        "start_year": first.year,
                        "end_year": last.year,
                        "start_value": first.normalized_value,
                        "end_value": last.normalized_value,
                        "change_percent": change,
                    },
                    confidence=min(first.combined_confidence, last.combined_confidence),
                    language=self.language,
                ))

            # Calculate CAGR if not already present
            if first.year and last.year and last.year > first.year:
                years = last.year - first.year
                if first.normalized_value > 0:
                    cagr = ((last.normalized_value / first.normalized_value) ** (1 / years) - 1) * 100

                    if self.language == "ja":
                        desc = f"年平均成長率（CAGR）は{cagr:.1f}%です。"
                    else:
                        desc = f"Compound Annual Growth Rate (CAGR) is {cagr:.1f}%."

                    insights.append(ChartInsight(
                        insight_type="cagr",
                        description=desc,
                        key_values={"cagr": cagr, "years": years},
                        confidence=0.9,
                        language=self.language,
                    ))

            # Title
            if self.language == "ja":
                title = f"{first.subject}の{first.metric_name}推移"
                if first.year and last.year:
                    title += f"（{first.year}-{last.year}年）"
            else:
                title = f"{first.subject} {first.metric_name} Trend"
                if first.year and last.year:
                    title += f" ({first.year}-{last.year})"

            rec = ChartRecommendation(
                chart_id=f"ts_{first.data_id}",
                title=title,
                chart_type=chart_type,
                purpose=ChartPurpose.SHOW_TREND,
                data_points=series,
                x_axis_label="Year" if self.language != "ja" else "年",
                y_axis_label=first.metric_name,
                unit=first.unit,
                insights=insights,
                main_message=insights[0].description if insights else "",
                data_confidence=sum(dp.combined_confidence for dp in series) / len(series),
                section_id=first.section_id,
                source_urls=list(set(dp.source_url for dp in series if dp.source_url)),
            )

            recommendations.append(rec)

        return recommendations

    def _analyze_comparisons(
        self,
        store: NumericalDataStore,
        research_topic: str,
    ) -> List[ChartRecommendation]:
        """Analyze for comparison chart opportunities."""
        recommendations = []

        # Group by (metric_name, year, unit)
        groups: Dict[Tuple[str, int, str], List[NumericalDataPoint]] = {}
        for dp in store.data_points:
            if dp.data_type == DataType.RATE:  # Skip rates for comparison
                continue
            key = (dp.metric_name.lower(), dp.year or 0, dp.unit)
            if key[0]:  # Only if metric_name exists
                if key not in groups:
                    groups[key] = []
                groups[key].append(dp)

        for key, data_points in groups.items():
            # Need at least 2 different subjects to compare
            subjects = set(dp.subject for dp in data_points)
            if len(subjects) < 2:
                continue

            # Use the best data point per subject
            subject_best: Dict[str, NumericalDataPoint] = {}
            for dp in data_points:
                if dp.subject not in subject_best:
                    subject_best[dp.subject] = dp
                elif dp.combined_confidence > subject_best[dp.subject].combined_confidence:
                    subject_best[dp.subject] = dp

            comparison_data = list(subject_best.values())
            if len(comparison_data) < 2:
                continue

            # Sort by value descending
            comparison_data.sort(key=lambda x: x.normalized_value, reverse=True)

            # Determine chart type
            if len(comparison_data) > 6:
                chart_type = RecommendedChartType.HORIZONTAL_BAR
            else:
                chart_type = RecommendedChartType.BAR

            # Generate insights
            insights = []
            first = comparison_data[0]
            metric_name, year, unit = key

            # Leader insight
            if self.language == "ja":
                desc = f"{first.subject}が{first.metric_name}でトップ（{first.raw_text}）です。"
            else:
                desc = f"{first.subject} leads in {first.metric_name} ({first.raw_text})."

            insights.append(ChartInsight(
                insight_type="leader",
                description=desc,
                key_values={
                    "leader": first.subject,
                    "value": first.normalized_value,
                },
                confidence=first.combined_confidence,
                language=self.language,
            ))

            # Gap insight (if significant)
            if len(comparison_data) >= 2:
                second = comparison_data[1]
                if second.normalized_value > 0:
                    gap = (first.normalized_value - second.normalized_value) / second.normalized_value * 100

                    if gap > 10:  # Significant gap
                        if self.language == "ja":
                            desc = f"{first.subject}は2位の{second.subject}を{gap:.0f}%上回っています。"
                        else:
                            desc = f"{first.subject} is {gap:.0f}% ahead of {second.subject}."

                        insights.append(ChartInsight(
                            insight_type="gap",
                            description=desc,
                            key_values={"gap_percent": gap},
                            confidence=min(first.combined_confidence, second.combined_confidence),
                            language=self.language,
                        ))

            # Title
            year_str = f"（{year}年）" if year else ""
            year_str_en = f" ({year})" if year else ""

            if self.language == "ja":
                title = f"{metric_name}の比較{year_str}"
            else:
                title = f"{metric_name.title()} Comparison{year_str_en}"

            rec = ChartRecommendation(
                chart_id=f"cmp_{hash(key) % 10000:04d}",
                title=title,
                chart_type=chart_type,
                purpose=ChartPurpose.COMPARE_VALUES,
                data_points=comparison_data,
                x_axis_label="",
                y_axis_label=metric_name,
                unit=unit,
                insights=insights,
                main_message=insights[0].description if insights else "",
                data_confidence=sum(dp.combined_confidence for dp in comparison_data) / len(comparison_data),
                section_id=comparison_data[0].section_id,
                source_urls=list(set(dp.source_url for dp in comparison_data if dp.source_url)),
            )

            recommendations.append(rec)

        return recommendations

    def _analyze_compositions(
        self,
        store: NumericalDataStore,
        research_topic: str,
    ) -> List[ChartRecommendation]:
        """Analyze for composition/pie chart opportunities."""
        recommendations = []

        # Look for market share or percentage data
        share_data = [
            dp for dp in store.data_points
            if dp.category == MetricCategory.MARKET_SHARE
            or (dp.data_type == DataType.PERCENTAGE and dp.unit == "%")
        ]

        # Group by (metric_name, year)
        groups: Dict[Tuple[str, int], List[NumericalDataPoint]] = {}
        for dp in share_data:
            key = (dp.metric_name.lower(), dp.year or 0)
            if key not in groups:
                groups[key] = []
            groups[key].append(dp)

        for key, data_points in groups.items():
            if len(data_points) < 2:
                continue

            # Check if percentages roughly sum to 100 (allow ±20%)
            total = sum(dp.value for dp in data_points)
            if not (80 <= total <= 120):
                # Not a composition, skip
                continue

            # Normalize to 100%
            if total != 100:
                for dp in data_points:
                    dp.normalized_value = dp.value / total

            # Sort by value descending
            data_points.sort(key=lambda x: x.value, reverse=True)

            # Limit to 8 slices (combine "Others")
            if len(data_points) > 8:
                main_data = data_points[:7]
                others_value = sum(dp.value for dp in data_points[7:])
                # We don't create a new "Others" data point here,
                # just note that some data was combined
            else:
                main_data = data_points

            # Generate insights
            insights = []
            first = main_data[0]
            metric_name, year = key

            # Top share insight
            if self.language == "ja":
                desc = f"{first.subject}が{first.value:.1f}%で最大シェアを占めています。"
            else:
                desc = f"{first.subject} holds the largest share at {first.value:.1f}%."

            insights.append(ChartInsight(
                insight_type="top_share",
                description=desc,
                key_values={
                    "leader": first.subject,
                    "share": first.value,
                },
                confidence=first.combined_confidence,
                language=self.language,
            ))

            # Concentration insight
            top3_share = sum(dp.value for dp in main_data[:3])
            if top3_share > 70:
                if self.language == "ja":
                    desc = f"上位3社で市場の{top3_share:.0f}%を占める集中型市場です。"
                else:
                    desc = f"Top 3 players control {top3_share:.0f}% of the market."

                insights.append(ChartInsight(
                    insight_type="concentration",
                    description=desc,
                    key_values={"top3_share": top3_share},
                    confidence=0.85,
                    language=self.language,
                ))

            # Title
            year_str = f"（{year}年）" if year else ""
            year_str_en = f" ({year})" if year else ""

            if self.language == "ja":
                title = f"{metric_name or 'シェア'}構成{year_str}"
            else:
                title = f"{(metric_name or 'Share').title()} Composition{year_str_en}"

            rec = ChartRecommendation(
                chart_id=f"pie_{hash(key) % 10000:04d}",
                title=title,
                chart_type=RecommendedChartType.PIE,
                purpose=ChartPurpose.SHOW_COMPOSITION,
                data_points=main_data,
                unit="%",
                insights=insights,
                main_message=insights[0].description if insights else "",
                data_confidence=sum(dp.combined_confidence for dp in main_data) / len(main_data),
                section_id=main_data[0].section_id,
                source_urls=list(set(dp.source_url for dp in main_data if dp.source_url)),
            )

            recommendations.append(rec)

        return recommendations

    def _analyze_growth(
        self,
        store: NumericalDataStore,
        research_topic: str,
    ) -> List[ChartRecommendation]:
        """Analyze for growth rate chart opportunities."""
        recommendations = []

        # Find growth rate data
        growth_data = [
            dp for dp in store.data_points
            if dp.category == MetricCategory.GROWTH_RATE
            or dp.data_type == DataType.RATE
            or 'cagr' in dp.metric_name.lower()
            or 'growth' in dp.metric_name.lower()
            or '成長' in dp.metric_name.lower()
        ]

        if len(growth_data) < 2:
            return recommendations

        # Group by time period or subject
        # If multiple subjects have growth rates, compare them
        subjects = set(dp.subject for dp in growth_data)

        if len(subjects) >= 2:
            # Compare growth rates across subjects
            subject_best: Dict[str, NumericalDataPoint] = {}
            for dp in growth_data:
                if dp.subject not in subject_best:
                    subject_best[dp.subject] = dp
                elif dp.combined_confidence > subject_best[dp.subject].combined_confidence:
                    subject_best[dp.subject] = dp

            comparison_data = list(subject_best.values())
            comparison_data.sort(key=lambda x: x.value, reverse=True)

            # Generate insights
            insights = []
            first = comparison_data[0]

            if self.language == "ja":
                desc = f"{first.subject}が{first.value:.1f}%で最も高い成長率を示しています。"
            else:
                desc = f"{first.subject} shows the highest growth rate at {first.value:.1f}%."

            insights.append(ChartInsight(
                insight_type="highest_growth",
                description=desc,
                key_values={
                    "leader": first.subject,
                    "growth_rate": first.value,
                },
                confidence=first.combined_confidence,
                language=self.language,
            ))

            if self.language == "ja":
                title = "成長率比較"
            else:
                title = "Growth Rate Comparison"

            rec = ChartRecommendation(
                chart_id=f"growth_{hash(tuple(subjects)) % 10000:04d}",
                title=title,
                chart_type=RecommendedChartType.HORIZONTAL_BAR,
                purpose=ChartPurpose.SHOW_GROWTH,
                data_points=comparison_data,
                x_axis_label="Growth Rate (%)" if self.language != "ja" else "成長率 (%)",
                unit="%",
                insights=insights,
                main_message=insights[0].description if insights else "",
                data_confidence=sum(dp.combined_confidence for dp in comparison_data) / len(comparison_data),
                section_id=comparison_data[0].section_id,
                source_urls=list(set(dp.source_url for dp in comparison_data if dp.source_url)),
            )

            recommendations.append(rec)

        return recommendations

    def _calculate_priority(self, rec: ChartRecommendation) -> float:
        """Calculate priority score for a chart recommendation."""
        score = 0.0

        # Data confidence weight (0-30)
        score += rec.data_confidence * 30

        # Number of data points (0-20)
        n_points = len(rec.data_points)
        if n_points >= 5:
            score += 20
        elif n_points >= 3:
            score += 15
        else:
            score += n_points * 5

        # Number of insights (0-20)
        n_insights = len(rec.insights)
        score += min(n_insights * 10, 20)

        # Purpose bonus (0-15)
        purpose_weights = {
            ChartPurpose.SHOW_TREND: 15,
            ChartPurpose.COMPARE_VALUES: 12,
            ChartPurpose.SHOW_GROWTH: 12,
            ChartPurpose.SHOW_COMPOSITION: 10,
            ChartPurpose.SHOW_RANKING: 8,
            ChartPurpose.SHOW_CORRELATION: 8,
            ChartPurpose.SHOW_DISTRIBUTION: 6,
            ChartPurpose.FORECAST: 5,
        }
        score += purpose_weights.get(rec.purpose, 5)

        # Has sources (0-10)
        if rec.source_urls:
            score += 10

        # Penalty for missing data
        if rec.has_missing_data and not rec.missing_data_filled:
            score -= 10

        rec.informativeness = score / 95  # Normalize to 0-1

        return score

    def _refine_with_llm(
        self,
        recommendations: List[ChartRecommendation],
        research_topic: str,
    ) -> List[ChartRecommendation]:
        """Use LLM to refine insights and messages."""
        if not self.llm or not recommendations:
            return recommendations

        # Process top recommendations
        top_recs = recommendations[:5]

        for rec in top_recs:
            try:
                # Prepare data summary
                data_summary = []
                for dp in rec.data_points[:10]:
                    data_summary.append(f"- {dp.subject}: {dp.raw_text} ({dp.year or 'N/A'})")

                data_str = "\n".join(data_summary)

                if self.language == "ja":
                    prompt = f"""以下のグラフデータについて、ビジネスレポート向けの簡潔な洞察を1-2文で記述してください。

研究テーマ: {research_topic}
グラフタイトル: {rec.title}
グラフの目的: {rec.purpose.value}

データ:
{data_str}

既存の洞察:
{rec.main_message}

【指示】
- データから読み取れる重要なポイントを強調
- 具体的な数字を含める
- ビジネス上の示唆があれば言及
- 1-2文で簡潔に

洞察:"""
                else:
                    prompt = f"""Write 1-2 concise business insights for the following chart data.

Research Topic: {research_topic}
Chart Title: {rec.title}
Chart Purpose: {rec.purpose.value}

Data:
{data_str}

Existing insight:
{rec.main_message}

Instructions:
- Highlight key takeaways from the data
- Include specific numbers
- Mention business implications if relevant
- Keep to 1-2 sentences

Insight:"""

                response = self.llm.generate(prompt)
                refined_message = response.content.strip()

                if refined_message and len(refined_message) > 10:
                    rec.main_message = refined_message

            except Exception as e:
                logger.warning(f"LLM insight refinement failed: {e}")
                continue

        return recommendations

    def get_charts_for_section(
        self,
        store: NumericalDataStore,
        section_id: str,
        research_topic: str = "",
    ) -> List[ChartRecommendation]:
        """
        Get chart recommendations for a specific section.

        Args:
            store: Numerical data store
            section_id: Section ID to filter by
            research_topic: Research topic for context

        Returns:
            List of chart recommendations for this section
        """
        # Create a filtered store for this section
        section_data = store.get_by_section(section_id)
        if not section_data:
            return []

        section_store = NumericalDataStore(research_topic=research_topic)
        section_store.add_many(section_data)

        # Analyze
        recs = self.analyze(section_store, research_topic)

        # Limit per section
        return recs[:self.max_charts_per_section]
