"""
Technology landscape visualizer.

Generates text-based and structured visualizations of patent technology landscapes.
"""

import logging
from typing import Dict, List, Any

from ..models.analysis import TechnologyLandscape

logger = logging.getLogger(__name__)


class LandscapeVisualizer:
    """Visualize technology landscape data."""

    def __init__(self, language: str = "ja"):
        self.language = language

    def to_markdown(self, landscape: TechnologyLandscape) -> str:
        """
        Convert a TechnologyLandscape to markdown format.

        Args:
            landscape: TechnologyLandscape object

        Returns:
            Markdown formatted string
        """
        lines = []

        lines.append(f"# 技術ランドスケープ: {landscape.topic}")
        lines.append("")
        lines.append(f"**分析特許数**: {landscape.total_patents_analyzed}件")
        if landscape.date_range:
            lines.append(f"**対象期間**: {landscape.date_range}")
        lines.append("")

        if landscape.summary:
            lines.append("## 概要")
            lines.append("")
            lines.append(landscape.summary)
            lines.append("")

        # IPC Distribution
        if landscape.ipc_distribution:
            lines.append("## IPC分類分布")
            lines.append("")
            sorted_ipc = sorted(
                landscape.ipc_distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            lines.append("| IPC分類 | 件数 | 割合 |")
            lines.append("|:---|:---:|:---:|")
            total = sum(v for _, v in sorted_ipc)
            for code, count in sorted_ipc[:15]:
                pct = (count / total * 100) if total > 0 else 0
                bar = "█" * int(pct / 5)  # Simple bar chart
                lines.append(f"| {code} | {count} | {pct:.1f}% {bar} |")
            lines.append("")

        # Top Applicants
        if landscape.top_applicants:
            lines.append("## 上位出願人")
            lines.append("")
            lines.append("| 順位 | 出願人 | 件数 |")
            lines.append("|:---:|:---|:---:|")
            for i, app in enumerate(landscape.top_applicants[:15], 1):
                name = app.get("name", "")
                count = app.get("count", 0)
                lines.append(f"| {i} | {name} | {count} |")
            lines.append("")

        # Filing Trend
        if landscape.filing_trend:
            lines.append("## 出願トレンド")
            lines.append("")
            sorted_years = sorted(landscape.filing_trend.items())
            lines.append("| 年 | 件数 | トレンド |")
            lines.append("|:---:|:---:|:---|")
            max_count = max(v for _, v in sorted_years) if sorted_years else 1
            for year, count in sorted_years:
                bar_len = int(count / max_count * 20) if max_count > 0 else 0
                bar = "█" * bar_len
                lines.append(f"| {year} | {count} | {bar} |")
            lines.append("")

        # Geographic Distribution
        if landscape.geographic_distribution:
            lines.append("## 地域分布")
            lines.append("")
            lines.append("| 管轄区域 | 件数 |")
            lines.append("|:---|:---:|")
            sorted_geo = sorted(
                landscape.geographic_distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for jurisdiction, count in sorted_geo:
                lines.append(f"| {jurisdiction} | {count} |")
            lines.append("")

        # Key Technologies
        if landscape.key_technologies:
            lines.append("## 主要技術")
            lines.append("")
            for tech in landscape.key_technologies:
                lines.append(f"- {tech}")
            lines.append("")

        # Clusters
        if landscape.clusters:
            lines.append("## 技術クラスター")
            lines.append("")
            for cluster in landscape.clusters:
                name = cluster.get("name", "")
                desc = cluster.get("description", "")
                patents = cluster.get("patents", [])
                lines.append(f"### {name}")
                if desc:
                    lines.append(desc)
                if patents:
                    lines.append(f"関連特許: {', '.join(patents[:5])}")
                lines.append("")

        return "\n".join(lines)

    def to_csv(self, landscape: TechnologyLandscape) -> str:
        """Generate CSV data from landscape."""
        lines = []

        # IPC distribution
        lines.append("# IPC Distribution")
        lines.append("IPC Code,Count")
        for code, count in sorted(
            landscape.ipc_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            lines.append(f"{code},{count}")

        lines.append("")

        # Top applicants
        lines.append("# Top Applicants")
        lines.append("Applicant,Count")
        for app in landscape.top_applicants:
            lines.append(f"{app.get('name', '')},{app.get('count', 0)}")

        lines.append("")

        # Filing trend
        lines.append("# Filing Trend")
        lines.append("Year,Count")
        for year, count in sorted(landscape.filing_trend.items()):
            lines.append(f"{year},{count}")

        return "\n".join(lines)
