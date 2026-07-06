"""
Claim chart generator.

Generates formatted claim charts for patent comparison,
including Markdown tables and structured output.
"""

import logging
from typing import List, Dict, Any

from ..models.patent import Patent
from ..models.analysis import ClaimChart, ClaimChartEntry

logger = logging.getLogger(__name__)


class ClaimChartGenerator:
    """Generate formatted claim charts."""

    def __init__(self, language: str = "ja"):
        self.language = language

    def to_markdown(self, chart: ClaimChart) -> str:
        """
        Convert a ClaimChart to markdown table format.

        Args:
            chart: ClaimChart object

        Returns:
            Markdown formatted string
        """
        lines = []

        lines.append(f"### クレームチャート: {chart.target_patent}")
        lines.append(f"**比較タイプ**: {chart.comparison_type}")
        lines.append("")

        if chart.summary:
            lines.append(chart.summary)
            lines.append("")

        if chart.entries:
            lines.append("| クレーム要素 | 参照特許 | 対応関係 | 確信度 |")
            lines.append("|:---|:---|:---|:---:|")

            for entry in chart.entries:
                element = entry.claim_element.replace("|", "\\|")
                mapping = entry.mapping.replace("|", "\\|")
                confidence = f"{entry.confidence:.0%}"
                lines.append(
                    f"| {element} | {entry.patent_number} "
                    f"| {mapping} | {confidence} |"
                )

            lines.append("")

        return "\n".join(lines)

    def to_csv(self, chart: ClaimChart) -> str:
        """
        Convert a ClaimChart to CSV format.

        Args:
            chart: ClaimChart object

        Returns:
            CSV formatted string
        """
        lines = ["クレーム要素,参照特許,対応関係,確信度,引用"]
        for entry in chart.entries:
            element = entry.claim_element.replace(",", "、")
            mapping = entry.mapping.replace(",", "、")
            excerpt = entry.source_excerpt.replace(",", "、")
            lines.append(
                f'"{element}","{entry.patent_number}","{mapping}",'
                f'{entry.confidence:.2f},"{excerpt}"'
            )
        return "\n".join(lines)

    def generate_comparison_matrix(
        self,
        target_patent: Patent,
        reference_patents: List[Patent],
    ) -> str:
        """
        Generate a high-level comparison matrix between patents.

        Args:
            target_patent: Target patent to compare
            reference_patents: Reference patents

        Returns:
            Markdown formatted comparison matrix
        """
        lines = []
        lines.append("### 特許比較マトリクス")
        lines.append("")

        # Header
        header = "| 項目 | " + " | ".join(
            p.patent_number[:15] for p in [target_patent] + reference_patents[:5]
        ) + " |"
        separator = "|" + "|".join(
            "---" for _ in range(len(reference_patents[:5]) + 2)
        ) + "|"

        lines.append(header)
        lines.append(separator)

        # Title row
        row = "| タイトル | " + " | ".join(
            p.title[:30] for p in [target_patent] + reference_patents[:5]
        ) + " |"
        lines.append(row)

        # Applicant row
        row = "| 出願人 | " + " | ".join(
            p.applicant[:20] for p in [target_patent] + reference_patents[:5]
        ) + " |"
        lines.append(row)

        # IPC row
        row = "| IPC | " + " | ".join(
            (p.ipc_classifications[0].full_code if p.ipc_classifications else "-")
            for p in [target_patent] + reference_patents[:5]
        ) + " |"
        lines.append(row)

        # Claims count row
        row = "| 請求項数 | " + " | ".join(
            str(len(p.claims)) for p in [target_patent] + reference_patents[:5]
        ) + " |"
        lines.append(row)

        # Filing date row
        row = "| 出願日 | " + " | ".join(
            p.filing_date or "-"
            for p in [target_patent] + reference_patents[:5]
        ) + " |"
        lines.append(row)

        lines.append("")
        return "\n".join(lines)
