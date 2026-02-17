"""
Patent report generator.

Generates patent research reports in various formats (Markdown, DOCX, PDF).
Follows the 10-section report structure defined in the patent research plan.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..research.patent_researcher import PatentResearchSession

logger = logging.getLogger(__name__)


class PatentReportGenerator:
    """Generate patent research reports."""

    def __init__(
        self,
        llm_client=None,
        language: str = "ja",
        output_dir: Path = None,
    ):
        self.llm = llm_client
        self.language = language
        self.output_dir = output_dir or Path("./output/patent_research")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        session: PatentResearchSession,
        format_type: str = "markdown",
    ) -> Path:
        """
        Generate a patent research report.

        Args:
            session: Completed PatentResearchSession
            format_type: Report format ('markdown', 'docx', 'pdf', 'html')

        Returns:
            Path to generated report file
        """
        # Build markdown content
        markdown = self._build_markdown(session)

        if format_type == "markdown":
            return self._save_markdown(session, markdown)
        elif format_type == "docx":
            return self._save_docx(session, markdown)
        elif format_type == "pdf":
            return self._save_pdf(session, markdown)
        elif format_type == "html":
            return self._save_html(session, markdown)
        else:
            return self._save_markdown(session, markdown)

    def _build_markdown(self, session: PatentResearchSession) -> str:
        """Build the complete report as markdown."""
        lines = []

        # Title
        plan_title = (
            session.search_plan.title
            if session.search_plan
            else f"特許調査: {session.query}"
        )
        lines.append(f"# {plan_title}")
        lines.append("")
        lines.append(f"**調査日**: {datetime.now().strftime('%Y年%m月%d日')}")
        lines.append(f"**調査テーマ**: {session.query}")
        if session.requirements:
            lines.append(f"**要件**: {session.requirements}")
        lines.append(
            f"**発見特許数**: {len(session.patents_found)}件"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

        # Table of Contents
        lines.append("## 目次")
        lines.append("")
        section_order = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        for section_id in section_order:
            if section_id in session.section_contents:
                title = session.section_contents[section_id].get("title", "")
                lines.append(f"- [{section_id}. {title}](#{section_id})")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Report sections
        for section_id in section_order:
            if section_id not in session.section_contents:
                continue

            section = session.section_contents[section_id]
            title = section.get("title", f"セクション {section_id}")
            content = section.get("content", "")

            lines.append(f"## {section_id}. {title}")
            lines.append("")
            lines.append(content)
            lines.append("")
            lines.append("---")
            lines.append("")

        # Appendix: Patent List
        if session.patents_found:
            lines.append("## 付録: 発見特許一覧")
            lines.append("")
            lines.append(
                "| No. | 特許番号 | タイトル | 出願人 | IPC | 出願日 |"
            )
            lines.append("|-----|---------|---------|--------|-----|--------|")
            for i, patent in enumerate(session.patents_found, 1):
                ipc = ", ".join(
                    c.full_code for c in patent.ipc_classifications[:3]
                )
                lines.append(
                    f"| {i} | {patent.patent_number} | {patent.title[:50]} | "
                    f"{patent.applicant[:30]} | {ipc} | {patent.filing_date} |"
                )
            lines.append("")

        # Appendix: Sources
        lines.append("## 付録: 検索ソース情報")
        lines.append("")
        lines.append(f"- **セッションID**: {session.session_id}")
        lines.append(f"- **調査開始**: {session.started_at}")
        lines.append(f"- **調査完了**: {session.completed_at or 'N/A'}")
        if session.search_plan:
            lines.append(
                f"- **使用クエリ**: {', '.join(session.search_plan.patent_queries[:5])}"
            )

        return "\n".join(lines)

    def _save_markdown(self, session: PatentResearchSession, markdown: str) -> Path:
        """Save report as Markdown file."""
        filename = f"patent_report_{session.session_id}.md"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown)

        logger.info(f"[Report] Markdown report saved: {filepath}")
        return filepath

    def _save_docx(self, session: PatentResearchSession, markdown: str) -> Path:
        """Save report as DOCX file."""
        try:
            from deep_research_tool.report.generator import ReportGenerator

            filename = f"patent_report_{session.session_id}.docx"
            filepath = self.output_dir / filename

            # Use existing ReportGenerator's DOCX capability
            generator = ReportGenerator(
                output_dir=self.output_dir,
                language=self.language,
            )
            generator.save_docx(markdown, filepath)
            logger.info(f"[Report] DOCX report saved: {filepath}")
            return filepath
        except Exception as e:
            logger.warning(f"[Report] DOCX generation failed, falling back to Markdown: {e}")
            return self._save_markdown(session, markdown)

    def _save_pdf(self, session: PatentResearchSession, markdown: str) -> Path:
        """Save report as PDF file."""
        try:
            from deep_research_tool.report.generator import ReportGenerator

            filename = f"patent_report_{session.session_id}.pdf"
            filepath = self.output_dir / filename

            generator = ReportGenerator(
                output_dir=self.output_dir,
                language=self.language,
            )
            generator.save_pdf(markdown, filepath)
            logger.info(f"[Report] PDF report saved: {filepath}")
            return filepath
        except Exception as e:
            logger.warning(f"[Report] PDF generation failed, falling back to Markdown: {e}")
            return self._save_markdown(session, markdown)

    def _save_html(self, session: PatentResearchSession, markdown: str) -> Path:
        """Save report as HTML file."""
        try:
            import markdown as md

            html_content = md.markdown(markdown, extensions=["tables", "toc"])

            filename = f"patent_report_{session.session_id}.html"
            filepath = self.output_dir / filename

            html_full = f"""<!DOCTYPE html>
<html lang="{self.language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>特許調査レポート - {session.session_id}</title>
    <style>
        body {{ font-family: 'Noto Sans JP', sans-serif; max-width: 900px; margin: 0 auto; padding: 2rem; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; }}
        h2 {{ color: #444; border-bottom: 1px solid #ccc; }}
        hr {{ margin: 2rem 0; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_full)

            logger.info(f"[Report] HTML report saved: {filepath}")
            return filepath
        except Exception as e:
            logger.warning(f"[Report] HTML generation failed, falling back to Markdown: {e}")
            return self._save_markdown(session, markdown)
