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
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime

from .context import ReportContext, WritingStyle, TargetAudience, ChapterSummary
from .consistency import ConsistencyChecker, ConsistencyReport
from .glossary import GlossaryManager
from ...utils.helpers import split_prose_and_meta

# Delimiter between chapter prose and metadata JSON in generation responses
CHAPTER_META_DELIMITER = "===CHAPTER_META==="


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
        enable_polish: bool = True,
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
            enable_polish: Run a final naturalness polish pass over all chapters
            progress_callback: Progress callback function
        """
        self.llm = llm_client
        self.language = language
        self.writing_style = writing_style
        self.target_audience = target_audience
        self.technical_level = technical_level
        self.enable_consistency_check = enable_consistency_check
        self.enable_two_phase = enable_two_phase
        self.enable_polish = enable_polish
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

        # Phase 3: Naturalness polish over all chapters
        if self.enable_polish:
            self._report_progress("Polishing chapters for naturalness...", 90, 100)
            chapters = self._polish_chapters(chapters, context)

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

        # Format content data. The Researcher stores section_contents entries
        # with keys title/content/summary/sources; extracted_content is kept
        # as a preferred source when a caller provides per-source material.
        no_info = "情報なし" if self.language == "ja" else "No information"
        if isinstance(content_data, dict):
            sources = content_data.get("sources", [])
            extracted = content_data.get("extracted_content", [])
            if extracted:
                content_summary = "\n".join([
                    f"- {e.get('title', 'N/A')}: {e.get('content', '')[:1200]}"
                    for e in extracted[:12]
                ])
            else:
                parts = []
                body = content_data.get("content", "")
                if body:
                    parts.append(body[:4000])
                summary = content_data.get("summary", "")
                if summary:
                    label = "要約: " if self.language == "ja" else "Summary: "
                    parts.append(label + summary[:600])
                if sources:
                    label = "情報源:" if self.language == "ja" else "Sources:"
                    parts.append(label + "\n" + "\n".join(
                        f"- {s}" for s in sources[:10]
                    ))
                content_summary = "\n\n".join(parts) if parts else no_info
        else:
            sources = []
            content_summary = str(content_data)[:4000] if content_data else no_info

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
4. 事実に基づいた記述を行い、推測は「〜と考えられる」などの表現で明示する
5. 必要に応じて前章を参照する

【出力形式】
まず本文をマークダウンでそのまま書く（章見出し「## {section_number}. {section_title}」から始める）。
JSONやコードブロックで囲まない。

本文の後、最後に次の区切り記号とメタ情報を必ず出力する:

{CHAPTER_META_DELIMITER}
{{"key_points": ["要点1", "要点2", "要点3"], "terms_used": ["使用した専門用語1"], "facts_stated": ["記述した重要な事実1"]}}"""
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
4. Base writing on facts; phrase speculation naturally as your assessment
5. Reference previous chapters when appropriate

[OUTPUT FORMAT]
Write the body directly as markdown (start with the heading "## {section_number}. {section_title}").
Do not wrap it in JSON or a code block.

After the body, output this delimiter followed by metadata:

{CHAPTER_META_DELIMITER}
{{"key_points": ["point1", "point2", "point3"], "terms_used": ["technical term1"], "facts_stated": ["fact1"]}}"""

        try:
            response = self.llm.generate(prompt)
            body, meta = split_prose_and_meta(response.content, CHAPTER_META_DELIMITER)

            if not body or not body.strip():
                raise ValueError("Empty chapter body in LLM response")

            return ChapterContent(
                section_number=section_number,
                section_title=section_title,
                content=body,
                word_count=len(body),
                key_points=meta.get("key_points", []),
                terms_used=meta.get("terms_used", []),
                facts_stated=meta.get("facts_stated", []),
                is_draft=True,
            )

        except Exception as e:
            print(f"[ReportGeneratorV2] Chapter generation failed: {e}")
            # Return minimal chapter
            return ChapterContent(
                section_number=section_number,
                section_title=section_title,
                content=f"# {section_number}. {section_title}\n\n（生成エラー: {e}）",
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

    def _polish_chapters(
        self,
        chapters: Dict[str, ChapterContent],
        context: ReportContext,
    ) -> Dict[str, ChapterContent]:
        """
        Polish all chapters for naturalness (one LLM call per chapter).

        Facts, numbers, citations and heading structure are kept intact; only
        the prose quality is improved. A rewrite whose length falls outside
        60%-150% of the original is rejected to guard against summarizing or
        inflating rewrites.
        """
        style_instructions = context.get_style_instructions()
        prev_tail = ""

        for section_num in sorted(chapters.keys()):
            chapter = chapters[section_num]
            if not chapter.content or not chapter.content.strip():
                continue

            if self.language == "ja":
                prompt = f"""あなたは日本語の報告書の推敲者です。以下の章の内容・事実・構成は変えずに、日本語としての自然さだけを磨いてください。

{style_instructions}

【直前の章の末尾（つながりの参考。書き換え対象ではない）】
{prev_tail or "（これが最初の章）"}

【推敲対象の章】
{chapter.content}

【推敲ルール】
1. 事実・数値・固有名詞・出典表記・見出し構成は一切変更しない。情報の追加・削除もしない。
2. 翻訳調・単調な文末・不自然な語順・冗長表現を直し、段落内と段落間の流れを滑らかにする。
3. 文体（です・ます調／である調）は上記スタイル指示に統一する。
4. 修正が不要な文はそのまま残してよい。

推敲後の本文のみを出力（前置き・JSON・コードブロック不要）:"""
            else:
                prompt = f"""You are a prose editor for a research report. Polish the following chapter for naturalness only, without changing its content, facts, or structure.

{style_instructions}

[END OF PREVIOUS CHAPTER (for transition context; not to be rewritten)]
{prev_tail or "(this is the first chapter)"}

[CHAPTER TO POLISH]
{chapter.content}

[EDITING RULES]
1. Do not change facts, numbers, proper nouns, citations, or heading structure. Do not add or remove information.
2. Fix awkward phrasing, monotonous sentence endings, and redundancy; smooth the flow within and between paragraphs.
3. Keep the style consistent with the instructions above.
4. Sentences that need no edits may be kept as-is.

Output only the polished body (no preamble, no JSON, no code block):"""

            try:
                response = self.llm.generate(prompt)
                polished = response.content.strip()

                original_len = len(chapter.content)
                if polished and 0.6 * original_len <= len(polished) <= 1.5 * original_len:
                    chapter.content = polished
                    chapter.word_count = len(polished)
                    chapter.is_draft = False
                else:
                    print(f"[ReportGeneratorV2] Polish rejected for {section_num} "
                          f"(length {len(polished)} vs original {original_len})")
            except Exception as e:
                print(f"[ReportGeneratorV2] Polish failed for {section_num}: {e}")

            prev_tail = chapter.content[-300:]

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

        # Table of Contents
        lines.append("## 目次" if self.language == "ja" else "## Table of Contents")
        lines.append("")
        for section_num, chapter in sorted(result.chapters.items()):
            lines.append(f"- {section_num}. {chapter.section_title}")
        lines.append("")

        # Chapters
        for section_num, chapter in sorted(result.chapters.items()):
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
