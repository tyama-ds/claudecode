"""
Report Generator - Generate research reports in multiple formats.
"""

import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List

from ..evidence.locker import EvidenceLocker
from ..research.researcher import ResearchSession
from ..verification.verifier import VerificationResult


class ReportFormat(str, Enum):
    """Supported report formats."""
    MARKDOWN = "markdown"
    DOCX = "docx"
    PDF = "pdf"
    HTML = "html"


class ReportGenerator:
    """
    Generate research reports in various formats.

    Supports:
    - Markdown (.md)
    - Word Document (.docx)
    - PDF (.pdf)
    - HTML (.html)
    """

    def __init__(
        self,
        output_dir: Path = None,
        include_toc: bool = True,
        include_citations: bool = True,
        include_images: bool = True,
        language: str = "ja",
    ):
        """
        Initialize ReportGenerator.

        Args:
            output_dir: Directory for output files
            include_toc: Include table of contents
            include_citations: Include citations/references
            include_images: Include images in report
            language: Report language
        """
        self.output_dir = output_dir or Path("./output/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_toc = include_toc
        self.include_citations = include_citations
        self.include_images = include_images
        self.language = language

    def generate_report(
        self,
        session: ResearchSession,
        evidence_locker: EvidenceLocker,
        format: ReportFormat = ReportFormat.MARKDOWN,
        verification_result: VerificationResult = None,
        filename: str = None,
    ) -> Path:
        """
        Generate a complete research report.

        Args:
            session: Research session with content
            evidence_locker: Evidence locker with sources
            format: Output format
            verification_result: Optional verification results
            filename: Custom filename (without extension)

        Returns:
            Path to generated report
        """
        if not filename:
            filename = f"research_report_{session.session_id}"

        if format == ReportFormat.MARKDOWN:
            return self._generate_markdown(
                session, evidence_locker, verification_result, filename
            )
        elif format == ReportFormat.DOCX:
            return self._generate_docx(
                session, evidence_locker, verification_result, filename
            )
        elif format == ReportFormat.PDF:
            return self._generate_pdf(
                session, evidence_locker, verification_result, filename
            )
        elif format == ReportFormat.HTML:
            return self._generate_html(
                session, evidence_locker, verification_result, filename
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_markdown(
        self,
        session: ResearchSession,
        evidence_locker: EvidenceLocker,
        verification_result: VerificationResult,
        filename: str,
    ) -> Path:
        """Generate Markdown report."""
        plan = session.research_plan
        content_parts = []

        # Title and metadata
        content_parts.append(f"# {plan.title if plan else session.query}\n")
        content_parts.append(f"**Research Date:** {session.started_at[:10]}\n")
        content_parts.append(f"**Session ID:** {session.session_id}\n")
        content_parts.append("\n---\n")

        # Executive Summary
        exec_summary = session.section_contents.get("_executive_summary", {})
        if exec_summary:
            content_parts.append("## Executive Summary\n")
            content_parts.append(exec_summary.get("executive_summary", "") + "\n")

            if exec_summary.get("key_findings"):
                content_parts.append("\n### Key Findings\n")
                for finding in exec_summary["key_findings"]:
                    content_parts.append(f"- {finding}\n")

            content_parts.append("\n---\n")

        # Table of Contents
        if self.include_toc and plan:
            content_parts.append("## Table of Contents\n")
            for item in plan.table_of_contents.items:
                content_parts.append(f"- [{item.section}. {item.title}](#{self._slugify(item.title)})\n")
                for sub in item.subsections:
                    content_parts.append(f"  - [{sub.section}. {sub.title}](#{self._slugify(sub.title)})\n")
            content_parts.append("\n---\n")

        # Main Content Sections
        if plan:
            for item in plan.table_of_contents.items:
                section_content = session.section_contents.get(item.section, {})

                content_parts.append(f"\n## {item.section}. {item.title}\n")

                if section_content:
                    # Section content with citation markers converted to footnotes
                    main_content = section_content.get("content", "")
                    content_parts.append(self._process_citations(main_content) + "\n")

                    # Images
                    if self.include_images and section_content.get("images"):
                        content_parts.append("\n### Related Figures\n")
                        for i, img in enumerate(section_content["images"][:3]):
                            caption = img.get("suggested_caption", f"Figure {i+1}")
                            content_parts.append(f"![{caption}]({img.get('src', '')})\n")
                            content_parts.append(f"*{caption}*\n\n")

                # Subsections
                for sub in item.subsections:
                    sub_content = session.section_contents.get(sub.section, {})
                    content_parts.append(f"\n### {sub.section}. {sub.title}\n")
                    if sub_content:
                        content_parts.append(self._process_citations(
                            sub_content.get("content", "")
                        ) + "\n")

        # Verification Summary
        if verification_result:
            content_parts.append("\n---\n")
            content_parts.append("## Verification Summary\n")
            content_parts.append(f"**Overall Reliability:** {verification_result.overall_reliability_score:.1%}\n\n")
            content_parts.append("| Confidence Level | Count |\n")
            content_parts.append("|-----------------|-------|\n")
            content_parts.append(f"| High | {verification_result.high_confidence_count} |\n")
            content_parts.append(f"| Medium | {verification_result.medium_confidence_count} |\n")
            content_parts.append(f"| Low | {verification_result.low_confidence_count} |\n")
            content_parts.append(f"| Unsupported | {verification_result.unsupported_count} |\n")

            if verification_result.hallucination_risk_count > 0:
                content_parts.append(f"\n**Hallucination Risks:** {verification_result.hallucination_risk_count} claims flagged\n")

        # References
        if self.include_citations:
            content_parts.append("\n---\n")
            content_parts.append("## References\n")

            for i, evidence in enumerate(evidence_locker.get_all_evidence(), 1):
                content_parts.append(f"{i}. {evidence.citation_text}\n")

        # Write file
        filepath = self.output_dir / f"{filename}.md"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("".join(content_parts))

        return filepath

    def _generate_docx(
        self,
        session: ResearchSession,
        evidence_locker: EvidenceLocker,
        verification_result: VerificationResult,
        filename: str,
    ) -> Path:
        """Generate Word document report."""
        try:
            from docx import Document
            from docx.shared import Inches, Pt
            from docx.enum.text import WD_ALIGN_PARAGRAPH
        except ImportError:
            raise ImportError(
                "python-docx not installed. Install with: pip install python-docx"
            )

        doc = Document()
        plan = session.research_plan

        # Title
        title = doc.add_heading(plan.title if plan else session.query, 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Metadata
        meta = doc.add_paragraph()
        meta.add_run(f"Research Date: {session.started_at[:10]}\n").bold = True
        meta.add_run(f"Session ID: {session.session_id}")

        doc.add_paragraph()  # Spacing

        # Executive Summary
        exec_summary = session.section_contents.get("_executive_summary", {})
        if exec_summary:
            doc.add_heading("Executive Summary", level=1)
            doc.add_paragraph(exec_summary.get("executive_summary", ""))

            if exec_summary.get("key_findings"):
                doc.add_heading("Key Findings", level=2)
                for finding in exec_summary["key_findings"]:
                    doc.add_paragraph(finding, style="List Bullet")

        # Table of Contents placeholder
        if self.include_toc and plan:
            doc.add_heading("Table of Contents", level=1)
            for item in plan.table_of_contents.items:
                doc.add_paragraph(
                    f"{item.section}. {item.title}",
                    style="TOC Heading"
                )

        # Main Content
        if plan:
            for item in plan.table_of_contents.items:
                section_content = session.section_contents.get(item.section, {})

                doc.add_heading(f"{item.section}. {item.title}", level=1)

                if section_content:
                    content = section_content.get("content", "")
                    # Split into paragraphs
                    paragraphs = content.split("\n\n")
                    for para in paragraphs:
                        if para.strip():
                            doc.add_paragraph(self._strip_citations(para))

                # Subsections
                for sub in item.subsections:
                    sub_content = session.section_contents.get(sub.section, {})
                    doc.add_heading(f"{sub.section}. {sub.title}", level=2)
                    if sub_content:
                        doc.add_paragraph(self._strip_citations(
                            sub_content.get("content", "")
                        ))

        # Verification Summary
        if verification_result:
            doc.add_heading("Verification Summary", level=1)
            doc.add_paragraph(
                f"Overall Reliability: {verification_result.overall_reliability_score:.1%}"
            )

            table = doc.add_table(rows=5, cols=2)
            table.style = "Table Grid"
            table.rows[0].cells[0].text = "Confidence Level"
            table.rows[0].cells[1].text = "Count"
            table.rows[1].cells[0].text = "High"
            table.rows[1].cells[1].text = str(verification_result.high_confidence_count)
            table.rows[2].cells[0].text = "Medium"
            table.rows[2].cells[1].text = str(verification_result.medium_confidence_count)
            table.rows[3].cells[0].text = "Low"
            table.rows[3].cells[1].text = str(verification_result.low_confidence_count)
            table.rows[4].cells[0].text = "Unsupported"
            table.rows[4].cells[1].text = str(verification_result.unsupported_count)

        # References
        if self.include_citations:
            doc.add_heading("References", level=1)
            for i, evidence in enumerate(evidence_locker.get_all_evidence(), 1):
                doc.add_paragraph(f"{i}. {evidence.citation_text}")

        # Save
        filepath = self.output_dir / f"{filename}.docx"
        doc.save(filepath)

        return filepath

    def _generate_pdf(
        self,
        session: ResearchSession,
        evidence_locker: EvidenceLocker,
        verification_result: VerificationResult,
        filename: str,
    ) -> Path:
        """Generate PDF report using reportlab."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                PageBreak
            )
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
        except ImportError:
            raise ImportError(
                "reportlab not installed. Install with: pip install reportlab"
            )

        filepath = self.output_dir / f"{filename}.pdf"
        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        plan = session.research_plan

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
        )

        heading1_style = ParagraphStyle(
            'CustomH1',
            parent=styles['Heading1'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
        )

        heading2_style = ParagraphStyle(
            'CustomH2',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
        )

        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceBefore=6,
            spaceAfter=6,
        )

        # Title
        story.append(Paragraph(
            plan.title if plan else session.query,
            title_style
        ))
        story.append(Spacer(1, 12))
        story.append(Paragraph(
            f"<b>Research Date:</b> {session.started_at[:10]}",
            body_style
        ))
        story.append(Paragraph(
            f"<b>Session ID:</b> {session.session_id}",
            body_style
        ))
        story.append(Spacer(1, 30))

        # Executive Summary
        exec_summary = session.section_contents.get("_executive_summary", {})
        if exec_summary:
            story.append(Paragraph("Executive Summary", heading1_style))
            story.append(Paragraph(
                exec_summary.get("executive_summary", ""),
                body_style
            ))
            story.append(Spacer(1, 20))

        # Main Content
        if plan:
            for item in plan.table_of_contents.items:
                section_content = session.section_contents.get(item.section, {})

                story.append(Paragraph(
                    f"{item.section}. {item.title}",
                    heading1_style
                ))

                if section_content:
                    content = self._strip_citations(
                        section_content.get("content", "")
                    )
                    # Split long content into paragraphs
                    for para in content.split("\n\n"):
                        if para.strip():
                            # Escape special characters for reportlab
                            safe_para = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                            try:
                                story.append(Paragraph(safe_para, body_style))
                            except Exception:
                                story.append(Paragraph(para[:500] + "...", body_style))

                # Subsections
                for sub in item.subsections:
                    sub_content = session.section_contents.get(sub.section, {})
                    story.append(Paragraph(
                        f"{sub.section}. {sub.title}",
                        heading2_style
                    ))
                    if sub_content:
                        safe_content = self._strip_citations(
                            sub_content.get("content", "")
                        ).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        try:
                            story.append(Paragraph(safe_content[:2000], body_style))
                        except Exception:
                            story.append(Paragraph("Content unavailable", body_style))

        # Verification Summary
        if verification_result:
            story.append(PageBreak())
            story.append(Paragraph("Verification Summary", heading1_style))
            story.append(Paragraph(
                f"<b>Overall Reliability:</b> {verification_result.overall_reliability_score:.1%}",
                body_style
            ))
            story.append(Spacer(1, 10))

            # Verification table
            table_data = [
                ["Confidence Level", "Count"],
                ["High", str(verification_result.high_confidence_count)],
                ["Medium", str(verification_result.medium_confidence_count)],
                ["Low", str(verification_result.low_confidence_count)],
                ["Unsupported", str(verification_result.unsupported_count)],
            ]

            table = Table(table_data, colWidths=[3*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(table)

        # References
        if self.include_citations:
            story.append(PageBreak())
            story.append(Paragraph("References", heading1_style))

            for i, evidence in enumerate(evidence_locker.get_all_evidence(), 1):
                safe_citation = evidence.citation_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                try:
                    story.append(Paragraph(f"{i}. {safe_citation[:200]}", body_style))
                except Exception:
                    story.append(Paragraph(f"{i}. [Citation formatting error]", body_style))

        # Build PDF
        doc.build(story)

        return filepath

    def _generate_html(
        self,
        session: ResearchSession,
        evidence_locker: EvidenceLocker,
        verification_result: VerificationResult,
        filename: str,
    ) -> Path:
        """Generate HTML report."""
        plan = session.research_plan

        # Build content sections
        sections_html = []
        if plan:
            for item in plan.table_of_contents.items:
                section_content = session.section_contents.get(item.section, {})

                section_html = f"""
                <section id="{self._slugify(item.title)}">
                    <h2>{item.section}. {item.title}</h2>
                    <div class="section-content">
                        {self._markdown_to_html(section_content.get('content', ''))}
                    </div>
                """

                # Images
                if self.include_images and section_content.get("images"):
                    section_html += '<div class="figures">'
                    for img in section_content["images"][:3]:
                        section_html += f"""
                        <figure>
                            <img src="{img.get('src', '')}" alt="{img.get('suggested_caption', '')}" loading="lazy">
                            <figcaption>{img.get('suggested_caption', '')}</figcaption>
                        </figure>
                        """
                    section_html += '</div>'

                # Subsections
                for sub in item.subsections:
                    sub_content = session.section_contents.get(sub.section, {})
                    section_html += f"""
                    <section id="{self._slugify(sub.title)}">
                        <h3>{sub.section}. {sub.title}</h3>
                        <div class="section-content">
                            {self._markdown_to_html(sub_content.get('content', ''))}
                        </div>
                    </section>
                    """

                section_html += "</section>"
                sections_html.append(section_html)

        # Executive summary
        exec_summary = session.section_contents.get("_executive_summary", {})
        exec_html = ""
        if exec_summary:
            findings_html = "".join(
                f"<li>{f}</li>" for f in exec_summary.get("key_findings", [])
            )
            exec_html = f"""
            <section class="executive-summary">
                <h2>Executive Summary</h2>
                <p>{exec_summary.get('executive_summary', '')}</p>
                <h3>Key Findings</h3>
                <ul>{findings_html}</ul>
            </section>
            """

        # Verification summary
        verification_html = ""
        if verification_result:
            verification_html = f"""
            <section class="verification-summary">
                <h2>Verification Summary</h2>
                <div class="reliability-score">
                    <span class="score">{verification_result.overall_reliability_score:.1%}</span>
                    <span class="label">Overall Reliability</span>
                </div>
                <table class="verification-table">
                    <tr><th>Confidence Level</th><th>Count</th></tr>
                    <tr><td>High</td><td>{verification_result.high_confidence_count}</td></tr>
                    <tr><td>Medium</td><td>{verification_result.medium_confidence_count}</td></tr>
                    <tr><td>Low</td><td>{verification_result.low_confidence_count}</td></tr>
                    <tr><td>Unsupported</td><td>{verification_result.unsupported_count}</td></tr>
                </table>
                {f'<p class="warning">Hallucination Risks: {verification_result.hallucination_risk_count} claims flagged</p>' if verification_result.hallucination_risk_count > 0 else ''}
            </section>
            """

        # References
        references_html = ""
        if self.include_citations:
            refs = "".join(
                f"<li>{e.citation_text}</li>"
                for e in evidence_locker.get_all_evidence()
            )
            references_html = f"""
            <section class="references">
                <h2>References</h2>
                <ol>{refs}</ol>
            </section>
            """

        # TOC
        toc_html = ""
        if self.include_toc and plan:
            toc_items = "".join(
                f'<li><a href="#{self._slugify(item.title)}">{item.section}. {item.title}</a></li>'
                for item in plan.table_of_contents.items
            )
            toc_html = f'<nav class="toc"><h2>Table of Contents</h2><ol>{toc_items}</ol></nav>'

        html_content = f"""
<!DOCTYPE html>
<html lang="{self.language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{plan.title if plan else session.query}</title>
    <style>
        :root {{
            --primary-color: #2c3e50;
            --accent-color: #3498db;
            --text-color: #333;
            --bg-color: #fff;
            --border-color: #e1e1e1;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.8;
            color: var(--text-color);
            background: var(--bg-color);
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid var(--accent-color);
        }}
        h1 {{ color: var(--primary-color); font-size: 2.5em; margin-bottom: 10px; }}
        h2 {{ color: var(--primary-color); margin: 30px 0 15px; padding-bottom: 10px; border-bottom: 1px solid var(--border-color); }}
        h3 {{ color: var(--primary-color); margin: 20px 0 10px; }}
        p {{ margin-bottom: 15px; }}
        .meta {{ color: #666; font-size: 0.9em; }}
        .toc {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 30px 0; }}
        .toc ol {{ padding-left: 25px; }}
        .toc li {{ margin: 8px 0; }}
        .toc a {{ color: var(--accent-color); text-decoration: none; }}
        .toc a:hover {{ text-decoration: underline; }}
        .executive-summary {{ background: #e8f4fd; padding: 25px; border-radius: 8px; margin: 30px 0; }}
        .section-content {{ padding: 10px 0; }}
        .figures {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
        figure {{ flex: 1; min-width: 200px; text-align: center; }}
        figure img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        figcaption {{ font-size: 0.9em; color: #666; margin-top: 8px; }}
        .verification-summary {{ background: #fff3cd; padding: 25px; border-radius: 8px; margin: 30px 0; }}
        .reliability-score {{ text-align: center; margin: 20px 0; }}
        .reliability-score .score {{ font-size: 3em; font-weight: bold; color: var(--primary-color); display: block; }}
        .reliability-score .label {{ font-size: 0.9em; color: #666; }}
        .verification-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .verification-table th, .verification-table td {{ padding: 12px; border: 1px solid var(--border-color); text-align: left; }}
        .verification-table th {{ background: var(--primary-color); color: white; }}
        .warning {{ color: #dc3545; font-weight: bold; }}
        .references {{ margin-top: 40px; padding-top: 20px; border-top: 2px solid var(--border-color); }}
        .references ol {{ padding-left: 25px; }}
        .references li {{ margin: 10px 0; font-size: 0.9em; }}
        @media print {{
            body {{ max-width: none; }}
            .toc {{ page-break-after: always; }}
            section {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>{plan.title if plan else session.query}</h1>
        <p class="meta">Research Date: {session.started_at[:10]} | Session: {session.session_id}</p>
    </header>

    {toc_html}
    {exec_html}
    {''.join(sections_html)}
    {verification_html}
    {references_html}

    <footer style="margin-top: 40px; text-align: center; color: #666; font-size: 0.8em;">
        <p>Generated by Deep Research Tool</p>
    </footer>
</body>
</html>
"""

        filepath = self.output_dir / f"{filename}.html"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        return filepath

    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug."""
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[\s_-]+', '-', text)
        return text.strip('-')

    def _process_citations(self, text: str) -> str:
        """Convert citation markers to markdown footnotes."""
        # Convert [SOURCE: N] to [^N]
        text = re.sub(r'\[SOURCE:\s*(\d+)\]', r'[^\1]', text)
        # Convert [ANALYSIS] to italic marker
        text = text.replace('[ANALYSIS]', '*[Analysis]*')
        return text

    def _strip_citations(self, text: str) -> str:
        """Remove citation markers from text."""
        text = re.sub(r'\[SOURCE:\s*\d+\]', '', text)
        text = text.replace('[ANALYSIS]', '')
        return text.strip()

    def _markdown_to_html(self, text: str) -> str:
        """Simple markdown to HTML conversion."""
        if not text:
            return ""

        # Process citations first
        text = self._process_citations(text)

        # Basic markdown conversion
        # Headers (already handled in structure)
        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        # Italic
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        # Links
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
        # Line breaks
        text = text.replace('\n\n', '</p><p>')
        text = text.replace('\n', '<br>')

        return f"<p>{text}</p>"
