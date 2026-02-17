"""
Report Generator V2 - Enhanced report generation with consistency features.

Version 2.0 adds:
- Global context maintenance across chapters
- Terminology consistency via glossary
- Previous chapter summaries for continuity
- Fact tracking to prevent contradictions
- Post-generation consistency checking
- Two-phase generation (draft + refinement)
"""

import json
import re
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable, Tuple
from datetime import datetime

from .context import ReportContext, WritingStyle, TargetAudience, ChapterSummary
from .consistency import ConsistencyChecker, ConsistencyReport
from .glossary import GlossaryManager

logger = logging.getLogger(__name__)


def _section_sort_key(item) -> Tuple:
    """
    Natural sort key for section numbers like "1", "1.1", "2", "10", "10.1".

    Converts section number strings into tuples of integers for proper
    numerical ordering:
        "1"    -> (1,)
        "1.1"  -> (1, 1)
        "2"    -> (2,)
        "10"   -> (10,)
        "10.1" -> (10, 1)

    This ensures "2" comes before "10" (numerical order), not after "1.9"
    and before "10" (lexicographic order).
    """
    key = item[0] if isinstance(item, tuple) else item
    parts = []
    for part in str(key).split('.'):
        try:
            parts.append(int(part))
        except ValueError:
            # Non-numeric parts (e.g., "A", "_executive") sort after numbers
            parts.append(float('inf'))
    return tuple(parts)


def _sanitize_for_xml(text: str) -> str:
    """
    Remove characters illegal in XML 1.0 from text.

    DOCX files are XML internally. XML 1.0 forbids these characters:
    - 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F (control chars except tab/newline/carriage return)
    - 0xFFFE, 0xFFFF

    If these characters are present (e.g., from LLM output), doc.save()
    will raise an XML serialization error.
    """
    if not text:
        return text
    # Remove illegal XML 1.0 characters
    return re.sub(
        r'[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]',
        '',
        text,
    )


@dataclass
class ChapterContent:
    """Generated chapter content with metadata."""
    section_number: str
    section_title: str
    content: str
    word_count: int = 0
    key_points: List[str] = field(default_factory=list)
    terms_used: List[str] = field(default_factory=list)
    facts_stated: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    is_draft: bool = True
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class GenerationResult:
    """Result of report generation."""
    chapters: Dict[str, ChapterContent]
    context: ReportContext
    consistency_report: Optional[ConsistencyReport] = None
    total_word_count: int = 0
    generation_time_seconds: float = 0.0


class ReportGeneratorV2:
    """
    Enhanced report generator with consistency features.

    This is the V2 implementation that ensures consistency across all chapters
    by maintaining global context, terminology, and fact tracking.

    Usage:
        generator = ReportGeneratorV2(
            llm_client=llm,
            writing_style=WritingStyle.BUSINESS,
            target_audience=TargetAudience.BUSINESS,
        )

        result = generator.generate_report(
            research_topic="炭素繊維の市場調査",
            research_plan=plan,
            section_contents=section_contents,
        )
    """

    def __init__(
        self,
        llm_client,
        language: str = "ja",
        writing_style: WritingStyle = WritingStyle.BUSINESS,
        target_audience: TargetAudience = TargetAudience.BUSINESS,
        technical_level: int = 3,
        enable_consistency_check: bool = True,
        enable_two_phase: bool = True,
        progress_callback: Callable[[str, int, int], None] = None,
    ):
        """
        Initialize ReportGeneratorV2.

        Args:
            llm_client: LLM API client
            language: Target language
            writing_style: Writing style for the report
            target_audience: Target audience
            technical_level: Technical level (1-5)
            enable_consistency_check: Run consistency check after generation
            enable_two_phase: Enable two-phase generation (draft + refinement)
            progress_callback: Progress callback function
        """
        self.llm = llm_client
        self.language = language
        self.writing_style = writing_style
        self.target_audience = target_audience
        self.technical_level = technical_level
        self.enable_consistency_check = enable_consistency_check
        self.enable_two_phase = enable_two_phase
        self.progress_callback = progress_callback

        # Initialize components
        self.glossary_manager = GlossaryManager(llm_client, language)
        self.consistency_checker = ConsistencyChecker(llm_client, language)

    def generate_report(
        self,
        research_topic: str,
        research_plan: Any,
        section_contents: Dict[str, Any],
        report_title: str = "",
    ) -> GenerationResult:
        """
        Generate a complete report with consistency features.

        Args:
            research_topic: Original research topic
            research_plan: ResearchPlan object
            section_contents: Dict of section_number -> extracted content
            report_title: Optional report title

        Returns:
            GenerationResult with all chapters and metadata
        """
        import time
        start_time = time.time()

        # Initialize context
        context = ReportContext(
            research_topic=research_topic,
            report_title=report_title or research_topic,
            language=self.language,
            writing_style=self.writing_style,
            target_audience=self.target_audience,
            technical_level=self.technical_level,
        )

        # Phase 0: Initialize glossary
        self._report_progress("Initializing glossary...", 0, 100)
        initial_glossary = self.glossary_manager.create_initial_glossary(
            research_topic=research_topic,
            research_plan=research_plan,
        )
        for key, entry in initial_glossary.items():
            context.add_glossary_term(
                term=entry.get("term", key),
                definition=entry.get("definition", ""),
                aliases=entry.get("aliases", []),
                preferred_form=entry.get("preferred_form", ""),
            )

        # Get sections from plan
        sections = self._get_sections_from_plan(research_plan)
        total_sections = len(sections)

        # Phase 1: Generate drafts
        chapters: Dict[str, ChapterContent] = {}

        for i, section in enumerate(sections):
            section_num = section.get("section", str(i + 1))
            section_title = section.get("title", f"Section {i + 1}")
            section_desc = section.get("description", "")

            self._report_progress(
                f"Generating chapter {section_num}: {section_title}",
                int(10 + (i / total_sections) * 60), 100
            )

            # Get content for this section
            content_data = section_contents.get(section_num, {})

            # Generate chapter with context
            chapter = self._generate_chapter(
                section_number=section_num,
                section_title=section_title,
                section_description=section_desc,
                content_data=content_data,
                context=context,
            )

            chapters[section_num] = chapter

            # Update context with chapter summary
            summary = self._extract_chapter_summary(chapter)
            context.add_chapter_summary(
                section_number=section_num,
                section_title=section_title,
                summary=summary,
                key_points=chapter.key_points,
                terms_introduced=chapter.terms_used,
                facts_established=chapter.facts_stated,
                word_count=chapter.word_count,
            )

            # Extract and add any new terms
            new_terms = self.glossary_manager.extract_terms_from_content(
                chapter.content, section_num
            )
            for term_candidate in new_terms:
                if term_candidate.term.lower() not in context.glossary:
                    context.add_glossary_term(
                        term=term_candidate.term,
                        definition="",
                        aliases=[term_candidate.expanded_form] if term_candidate.expanded_form else [],
                        section=section_num,
                    )

        # Phase 2: Consistency check and refinement
        consistency_report = None
        if self.enable_consistency_check:
            self._report_progress("Checking consistency...", 75, 100)
            chapter_texts = {k: v.content for k, v in chapters.items()}
            consistency_report = self.consistency_checker.check_all(chapter_texts, context)

            if self.enable_two_phase and not consistency_report.is_consistent:
                self._report_progress("Refining for consistency...", 85, 100)
                chapters = self._refine_chapters(chapters, consistency_report, context)

        # Calculate totals
        total_word_count = sum(c.word_count for c in chapters.values())
        context.total_word_count = total_word_count

        self._report_progress("Generation complete", 100, 100)

        return GenerationResult(
            chapters=chapters,
            context=context,
            consistency_report=consistency_report,
            total_word_count=total_word_count,
            generation_time_seconds=time.time() - start_time,
        )

    def _generate_chapter(
        self,
        section_number: str,
        section_title: str,
        section_description: str,
        content_data: Any,
        context: ReportContext,
    ) -> ChapterContent:
        """Generate a single chapter with context awareness."""

        # Build the prompt with full context
        context_prompt = context.get_full_context_prompt()

        # Format content data
        if isinstance(content_data, dict):
            sources = content_data.get("sources", [])
            extracted = content_data.get("extracted_content", [])
            content_summary = "\n".join([
                f"- {e.get('title', 'N/A')}: {e.get('content', '')[:500]}"
                for e in extracted[:10]
            ]) if extracted else "情報なし"
        else:
            sources = []
            content_summary = str(content_data)[:2000] if content_data else "情報なし"

        if self.language == "ja":
            prompt = f"""{context_prompt}

【現在執筆中のセクション】
セクション番号: {section_number}
タイトル: {section_title}
説明: {section_description}

【収集された情報】
{content_summary}

【執筆指示】
上記の情報を基に、このセクションの内容を執筆してください。

重要な注意点：
1. 用語統一ルールに従い、一貫した用語を使用する
2. 前章までの内容と矛盾しないようにする
3. 文体・スタイル指示に従う
4. 事実に基づいた記述を行い、推測は明示する
5. 必要に応じて前章を参照・引用する

出力形式（JSON）:
{{
    "content": "本文（マークダウン形式）",
    "key_points": ["要点1", "要点2", "要点3"],
    "terms_used": ["使用した専門用語1", "専門用語2"],
    "facts_stated": ["記述した事実1", "事実2"],
    "word_count": 文字数
}}

JSONのみを出力:"""
        else:
            prompt = f"""{context_prompt}

[CURRENT SECTION]
Section: {section_number}
Title: {section_title}
Description: {section_description}

[GATHERED INFORMATION]
{content_summary}

[WRITING INSTRUCTIONS]
Write this section based on the above information.

Important notes:
1. Follow terminology consistency rules
2. Do not contradict previous chapters
3. Follow style instructions
4. Base writing on facts; clearly mark speculation
5. Reference previous chapters when appropriate

Output format (JSON):
{{
    "content": "main text (markdown format)",
    "key_points": ["point1", "point2", "point3"],
    "terms_used": ["technical term1", "term2"],
    "facts_stated": ["fact1", "fact2"],
    "word_count": word_count
}}

Output only JSON:"""

        try:
            response = self.llm.generate(prompt)

            # Guard against None/empty response (OpenAI API can return content=None)
            if not response or not response.content:
                print(f"[ReportGeneratorV2] WARNING: Empty response for section {section_number} '{section_title}'")
                return ChapterContent(
                    section_number=section_number,
                    section_title=section_title,
                    content=f"## {section_number}. {section_title}\n\n"
                            f"このセクションの内容を生成できませんでした（LLM応答が空でした）。",
                    word_count=0,
                    is_draft=True,
                )

            content = response.content.strip()

            # Parse JSON
            if "```" in content:
                content = content.split("```")[1].split("```")[0]
                if content.startswith("json"):
                    content = content[4:]

            data = json.loads(content)

            chapter_content = data.get("content", "")
            if not chapter_content:
                print(f"[ReportGeneratorV2] WARNING: LLM returned empty 'content' field for section {section_number}")

            return ChapterContent(
                section_number=section_number,
                section_title=section_title,
                content=chapter_content,
                word_count=data.get("word_count", len(chapter_content)),
                key_points=data.get("key_points", []),
                terms_used=data.get("terms_used", []),
                facts_stated=data.get("facts_stated", []),
                is_draft=True,
            )

        except json.JSONDecodeError as e:
            print(f"[ReportGeneratorV2] JSON parse failed for section {section_number}: {e}")
            # If the response looks like direct markdown (not JSON), use it as-is
            raw = response.content.strip() if response and response.content else ""
            if raw and not raw.startswith("{"):
                print(f"[ReportGeneratorV2] Using raw response as chapter content (non-JSON)")
                return ChapterContent(
                    section_number=section_number,
                    section_title=section_title,
                    content=f"## {section_number}. {section_title}\n\n{raw}",
                    word_count=len(raw),
                    is_draft=True,
                )
            return ChapterContent(
                section_number=section_number,
                section_title=section_title,
                content=f"## {section_number}. {section_title}\n\n（JSON解析エラー: {e}）",
                word_count=0,
                is_draft=True,
            )

        except Exception as e:
            print(f"[ReportGeneratorV2] Chapter generation failed for section {section_number}: {e}")
            return ChapterContent(
                section_number=section_number,
                section_title=section_title,
                content=f"## {section_number}. {section_title}\n\n（生成エラー: {e}）",
                word_count=0,
                is_draft=True,
            )

    def _extract_chapter_summary(self, chapter: ChapterContent) -> str:
        """Extract a summary from chapter content."""
        content = chapter.content

        # Try to get first paragraph or first 300 chars
        paragraphs = content.split("\n\n")
        for p in paragraphs:
            p = p.strip()
            if len(p) > 50 and not p.startswith("#"):
                return p[:300] + ("..." if len(p) > 300 else "")

        return content[:300] + ("..." if len(content) > 300 else "")

    def _refine_chapters(
        self,
        chapters: Dict[str, ChapterContent],
        consistency_report: ConsistencyReport,
        context: ReportContext,
    ) -> Dict[str, ChapterContent]:
        """Refine chapters based on consistency issues."""

        # Get sections that need refinement
        sections_to_refine = set()
        for issue in consistency_report.issues:
            if issue.section and issue.section != "全体":
                sections_to_refine.add(issue.section)
            for related in issue.related_sections:
                sections_to_refine.add(related)

        # Refine each affected section
        for section_num in sections_to_refine:
            if section_num not in chapters:
                continue

            chapter = chapters[section_num]
            issues_for_section = consistency_report.get_issues_by_section(section_num)

            if not issues_for_section:
                continue

            # Build refinement prompt
            issues_text = "\n".join([
                f"- [{i.issue_type.value}] {i.description}"
                for i in issues_for_section
            ])

            if self.language == "ja":
                prompt = f"""以下の章を修正してください。

【現在の内容】
{chapter.content}

【検出された問題】
{issues_text}

【修正指示】
上記の問題を解決するように内容を修正してください。
用語の統一、矛盾の解消、文体の調整を行ってください。

修正後の本文のみを出力してください（JSON不要）:"""
            else:
                prompt = f"""Please revise the following chapter.

[CURRENT CONTENT]
{chapter.content}

[DETECTED ISSUES]
{issues_text}

[REVISION INSTRUCTIONS]
Fix the above issues. Unify terminology, resolve contradictions, adjust style.

Output only the revised content (no JSON):"""

            try:
                response = self.llm.generate(prompt)
                refined_content = response.content.strip()

                # Update chapter
                chapter.content = refined_content
                chapter.word_count = len(refined_content)
                chapter.is_draft = False

            except Exception as e:
                print(f"[ReportGeneratorV2] Refinement failed for {section_num}: {e}")

        return chapters

    def _get_sections_from_plan(self, research_plan: Any) -> List[Dict[str, str]]:
        """Extract sections from research plan."""
        sections = []

        if hasattr(research_plan, 'table_of_contents'):
            toc = research_plan.table_of_contents
            if hasattr(toc, 'items'):
                for item in toc.items:
                    sections.append({
                        "section": item.section,
                        "title": item.title,
                        "description": getattr(item, 'description', ''),
                    })
                    # Add subsections
                    if hasattr(item, 'subsections'):
                        for sub in item.subsections:
                            sections.append({
                                "section": sub.section,
                                "title": sub.title,
                                "description": getattr(sub, 'description', ''),
                            })

        return sections

    def _report_progress(self, message: str, current: int, total: int) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(message, current, total)
        else:
            print(f"[ReportGeneratorV2] {message} ({current}/{total})")

    def generate_final_document(
        self,
        result: GenerationResult,
        include_glossary: bool = True,
        include_consistency_summary: bool = False,
    ) -> str:
        """
        Generate the final document from generation result.

        Args:
            result: GenerationResult from generate_report
            include_glossary: Include glossary section
            include_consistency_summary: Include consistency check summary

        Returns:
            Complete document as markdown string
        """
        lines = []

        # Title
        lines.append(f"# {result.context.report_title}")
        lines.append("")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d')}*")
        lines.append("")

        # Table of Contents (use natural sort for proper section ordering)
        lines.append("## 目次" if self.language == "ja" else "## Table of Contents")
        lines.append("")
        for section_num, chapter in sorted(result.chapters.items(), key=_section_sort_key):
            lines.append(f"- {section_num}. {chapter.section_title}")
        lines.append("")

        # Chapters (use natural sort: 1, 1.1, 2, 2.1, ... 10, 10.1)
        for section_num, chapter in sorted(result.chapters.items(), key=_section_sort_key):
            lines.append(chapter.content)
            lines.append("")

        # Glossary
        if include_glossary and result.context.glossary:
            glossary_section = self.glossary_manager.generate_glossary_section(
                {k: {
                    "term": v.term,
                    "definition": v.definition,
                    "aliases": v.aliases,
                } for k, v in result.context.glossary.items()},
                title="用語集" if self.language == "ja" else "Glossary"
            )
            lines.append(glossary_section)

        # Consistency summary
        if include_consistency_summary and result.consistency_report:
            summary = self.consistency_checker.generate_summary(result.consistency_report)
            lines.append(summary)

        return "\n".join(lines)

    def save_report(
        self,
        markdown_content: str,
        output_dir: Path,
        filename: str,
        format: str = "markdown",
    ) -> Path:
        """
        Save the generated report in the specified format.

        Args:
            markdown_content: The report content as markdown string
            output_dir: Directory to save the report
            filename: Base filename (without extension)
            format: Output format ('markdown', 'docx', 'pdf', 'html')

        Returns:
            Path to the saved report file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        format_lower = format.lower() if isinstance(format, str) else format.value.lower()

        if format_lower == "docx":
            return self._save_as_docx(markdown_content, output_dir, filename)
        elif format_lower == "html":
            return self._save_as_html(markdown_content, output_dir, filename)
        elif format_lower == "pdf":
            # PDF: save as markdown with note (PDF generation requires additional setup)
            md_path = self._save_as_markdown(markdown_content, output_dir, filename)
            print("[ReportGeneratorV2] PDF format is not yet supported in V2. Saved as markdown.")
            return md_path
        else:
            return self._save_as_markdown(markdown_content, output_dir, filename)

    def _save_as_markdown(self, content: str, output_dir: Path, filename: str) -> Path:
        """Save report as markdown file."""
        filepath = output_dir / f"{filename}.md"
        filepath.write_text(content, encoding="utf-8")
        return filepath

    def _save_as_docx(self, markdown_content: str, output_dir: Path, filename: str) -> Path:
        """Convert markdown content to DOCX and save."""
        try:
            from docx import Document
            from docx.shared import Pt, Inches, Cm
            from docx.enum.text import WD_ALIGN_PARAGRAPH
        except ImportError:
            logger.error("[ReportGeneratorV2] python-docx not installed. Falling back to markdown.")
            return self._save_as_markdown(markdown_content, output_dir, filename)

        # Sanitize content to remove illegal XML characters BEFORE processing
        markdown_content = _sanitize_for_xml(markdown_content)

        doc = Document()
        lines = markdown_content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Empty line
            if not stripped:
                i += 1
                continue

            # Headings
            if stripped.startswith("#"):
                level, text = self._parse_heading(stripped)
                try:
                    doc.add_heading(text, level=level)
                except Exception as e:
                    logger.warning(f"[DOCX] Heading add failed: {e}, adding as paragraph")
                    doc.add_paragraph(text)
                i += 1
                continue

            # Markdown table detection (lines starting with |)
            if stripped.startswith("|") and "|" in stripped[1:]:
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i].strip())
                    i += 1
                if table_lines:
                    self._add_markdown_table_to_docx(doc, table_lines)
                continue

            # Unordered list items
            if re.match(r'^[-*+]\s', stripped):
                text = re.sub(r'^[-*+]\s+', '', stripped)
                try:
                    para = doc.add_paragraph(style='List Bullet')
                except KeyError:
                    para = doc.add_paragraph()
                    text = "- " + text
                self._add_formatted_runs(para, text)
                i += 1
                continue

            # Ordered list items
            if re.match(r'^\d+\.\s', stripped):
                text = re.sub(r'^\d+\.\s+', '', stripped)
                try:
                    para = doc.add_paragraph(style='List Number')
                except KeyError:
                    para = doc.add_paragraph()
                    text = stripped  # Keep the number prefix
                self._add_formatted_runs(para, text)
                i += 1
                continue

            # Italic metadata line (e.g., *Generated: 2024-01-01*)
            if stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**"):
                para = doc.add_paragraph()
                run = para.add_run(stripped.strip("*"))
                run.italic = True
                run.font.size = Pt(9)
                para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                i += 1
                continue

            # Image reference: ![alt](path)
            img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', stripped)
            if img_match:
                alt_text = img_match.group(1)
                img_path = img_match.group(2)
                self._add_image_to_docx(doc, img_path, alt_text)
                i += 1
                continue

            # Regular paragraph: collect consecutive non-empty, non-special lines
            para_lines = []
            start_i = i  # Track starting position to prevent infinite loop
            while i < len(lines):
                l = lines[i].strip()
                if not l or l.startswith("#") or re.match(r'^[-*+]\s', l) or re.match(r'^\d+\.\s', l) or l.startswith("|"):
                    break
                # Check for image reference
                if re.match(r'!\[([^\]]*)\]\(([^)]+)\)', l):
                    break
                para_lines.append(l)
                i += 1

            # Safety: if no lines were consumed, force advance to avoid infinite loop
            if i == start_i:
                i += 1

            if para_lines:
                para = doc.add_paragraph()
                self._add_formatted_runs(para, " ".join(para_lines))
            continue

        filepath = output_dir / f"{filename}.docx"
        try:
            doc.save(filepath)
        except Exception as save_error:
            logger.error(f"[ReportGeneratorV2] doc.save() failed: {save_error}")
            # Second attempt: sanitize more aggressively and retry
            try:
                logger.info("[ReportGeneratorV2] Retrying with aggressive sanitization...")
                self._sanitize_docx_paragraphs(doc)
                doc.save(filepath)
            except Exception as retry_error:
                logger.error(f"[ReportGeneratorV2] Retry also failed: {retry_error}. Falling back to markdown.")
                return self._save_as_markdown(markdown_content, output_dir, filename)
        return filepath

    @staticmethod
    def _sanitize_docx_paragraphs(doc):
        """Sanitize all paragraph text in a Document to remove illegal XML chars."""
        for paragraph in doc.paragraphs:
            for run in paragraph.runs:
                if run.text:
                    run.text = _sanitize_for_xml(run.text)

    def _add_markdown_table_to_docx(self, doc, table_lines: list):
        """Convert markdown table lines to a DOCX table."""
        try:
            from docx.shared import Pt

            # Parse header row
            rows_data = []
            for line in table_lines:
                cells = [c.strip() for c in line.strip("|").split("|")]
                # Skip separator row (contains only -, :, spaces)
                if all(re.match(r'^[-:]+$', c.strip()) for c in cells if c.strip()):
                    continue
                rows_data.append(cells)

            if not rows_data:
                return

            num_cols = max(len(row) for row in rows_data)
            table = doc.add_table(rows=len(rows_data), cols=num_cols)
            table.style = "Table Grid"

            for row_idx, row_data in enumerate(rows_data):
                for col_idx, cell_text in enumerate(row_data):
                    if col_idx < num_cols:
                        cell = table.rows[row_idx].cells[col_idx]
                        cell.text = cell_text.strip()
                        # Bold header row
                        if row_idx == 0:
                            for paragraph in cell.paragraphs:
                                for run in paragraph.runs:
                                    run.bold = True
                                    run.font.size = Pt(9)

            doc.add_paragraph()  # Spacing after table
        except Exception as e:
            print(f"[ReportGeneratorV2] Table insertion failed: {e}")

    def _add_image_to_docx(self, doc, img_path: str, alt_text: str = ""):
        """Add an image to the DOCX document."""
        try:
            from docx.shared import Inches
            import os

            # Check if file exists (handle both absolute and relative paths)
            if os.path.exists(img_path):
                doc.add_picture(img_path, width=Inches(5.5))
            elif os.path.exists(os.path.join(str(self.output_dir if hasattr(self, 'output_dir') else '.'), img_path)):
                full_path = os.path.join(str(self.output_dir if hasattr(self, 'output_dir') else '.'), img_path)
                doc.add_picture(full_path, width=Inches(5.5))

            if alt_text:
                para = doc.add_paragraph()
                run = para.add_run(alt_text)
                run.italic = True
                from docx.shared import Pt
                run.font.size = Pt(9)
        except Exception as e:
            # If image can't be added, add alt text as placeholder
            if alt_text:
                doc.add_paragraph(f"[Image: {alt_text}]")

    def _save_as_html(self, markdown_content: str, output_dir: Path, filename: str) -> Path:
        """Convert markdown content to HTML and save."""
        # Simple markdown to HTML conversion
        html_body = self._markdown_to_html(markdown_content)
        html_doc = f"""<!DOCTYPE html>
<html lang="{self.language}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body {{ font-family: 'Helvetica Neue', Arial, 'Hiragino Kaku Gothic ProN', sans-serif; max-width: 900px; margin: 0 auto; padding: 2em; line-height: 1.8; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }}
ul, ol {{ padding-left: 1.5em; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""
        filepath = output_dir / f"{filename}.html"
        filepath.write_text(html_doc, encoding="utf-8")
        return filepath

    @staticmethod
    def _parse_heading(line: str) -> tuple:
        """Parse a markdown heading line into (level, text)."""
        match = re.match(r'^(#{1,6})\s+(.*)', line)
        if match:
            level = min(len(match.group(1)), 4)  # docx supports levels 0-4
            return level - 1, match.group(2).strip()
        return 0, line.strip("#").strip()

    @staticmethod
    def _add_formatted_runs(paragraph, text: str):
        """Add text with bold/italic formatting as runs to a paragraph."""
        # Split by bold (**text**) and italic (*text*) markers
        parts = re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*)', text)
        for part in parts:
            if not part:
                continue
            if part.startswith("**") and part.endswith("**"):
                run = paragraph.add_run(part[2:-2])
                run.bold = True
            elif part.startswith("*") and part.endswith("*"):
                run = paragraph.add_run(part[1:-1])
                run.italic = True
            else:
                paragraph.add_run(part)

    @staticmethod
    def _markdown_to_html(markdown_text: str) -> str:
        """Simple markdown to HTML conversion."""
        lines = markdown_text.split("\n")
        html_lines = []
        in_list = False
        list_type = None

        for line in lines:
            stripped = line.strip()

            if not stripped:
                if in_list:
                    html_lines.append(f"</{list_type}>")
                    in_list = False
                    list_type = None
                html_lines.append("")
                continue

            # Headings
            heading_match = re.match(r'^(#{1,6})\s+(.*)', stripped)
            if heading_match:
                if in_list:
                    html_lines.append(f"</{list_type}>")
                    in_list = False
                level = len(heading_match.group(1))
                text = heading_match.group(2)
                html_lines.append(f"<h{level}>{text}</h{level}>")
                continue

            # Unordered list
            if re.match(r'^[-*+]\s', stripped):
                text = re.sub(r'^[-*+]\s+', '', stripped)
                if not in_list or list_type != "ul":
                    if in_list:
                        html_lines.append(f"</{list_type}>")
                    html_lines.append("<ul>")
                    in_list = True
                    list_type = "ul"
                html_lines.append(f"  <li>{text}</li>")
                continue

            # Ordered list
            ol_match = re.match(r'^\d+\.\s+(.*)', stripped)
            if ol_match:
                text = ol_match.group(1)
                if not in_list or list_type != "ol":
                    if in_list:
                        html_lines.append(f"</{list_type}>")
                    html_lines.append("<ol>")
                    in_list = True
                    list_type = "ol"
                html_lines.append(f"  <li>{text}</li>")
                continue

            # Regular paragraph
            if in_list:
                html_lines.append(f"</{list_type}>")
                in_list = False
                list_type = None
            # Apply inline formatting
            formatted = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', stripped)
            formatted = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', formatted)
            html_lines.append(f"<p>{formatted}</p>")

        if in_list:
            html_lines.append(f"</{list_type}>")

        return "\n".join(html_lines)
