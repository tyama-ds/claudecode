"""
Manual Researcher - Research orchestration using pre-loaded evidence.

This module provides the ManualResearcher class which conducts research
using evidence loaded from CSV/XLSX files instead of web search.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from uuid import uuid4

from ..evidence.locker import EvidenceLocker, Evidence, EvidenceType
from ..evidence.manual_loader import ManualEvidenceLoader, load_evidence_file
from .query_generator import (
    QueryGenerator,
    ResearchPlan,
    TableOfContents,
    TableOfContentsItem,
)
from .content_extractor import ContentExtractor, ExtractedContent
from .researcher import ResearchState, ResearchSession, ResearchIteration


@dataclass
class ManualTableOfContents:
    """
    Manual table of contents specification.

    Used when auto_toc=False to provide a pre-defined structure.
    """
    title: str
    sections: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManualTableOfContents":
        """Create from dictionary."""
        return cls(
            title=data.get("title", "Research Report"),
            sections=data.get("sections", []),
        )

    @classmethod
    def from_list(cls, title: str, sections: List[str]) -> "ManualTableOfContents":
        """
        Create from simple list of section titles.

        Args:
            title: Report title
            sections: List of section titles

        Example:
            toc = ManualTableOfContents.from_list(
                "Market Analysis Report",
                ["Executive Summary", "Market Overview", "Competitive Analysis", "Recommendations"]
            )
        """
        section_dicts = []
        for i, section_title in enumerate(sections, 1):
            section_dicts.append({
                "section": str(i),
                "title": section_title,
                "description": "",
            })
        return cls(title=title, sections=section_dicts)

    def to_table_of_contents(self) -> TableOfContents:
        """Convert to standard TableOfContents object."""
        items = []
        for sec in self.sections:
            item = TableOfContentsItem(
                section=sec.get("section", str(len(items) + 1)),
                title=sec.get("title", ""),
                description=sec.get("description", ""),
            )

            # Handle subsections
            for sub in sec.get("subsections", []):
                sub_item = TableOfContentsItem(
                    section=sub.get("section", ""),
                    title=sub.get("title", ""),
                    description=sub.get("description", ""),
                )
                item.subsections.append(sub_item)

            items.append(item)

        return TableOfContents(title=self.title, items=items)


class ManualResearcher:
    """
    Research orchestrator using pre-loaded evidence.

    Instead of conducting web searches, this class uses evidence
    loaded from CSV/XLSX files to generate research reports.

    Features:
    - Load evidence from CSV/XLSX files
    - Auto-generate or use manual table of contents
    - Match evidence to sections based on content relevance
    - Generate section content using LLM synthesis
    - Support all report formats (docx, pdf, markdown, html)
    """

    def __init__(
        self,
        llm_client,
        evidence_locker: EvidenceLocker = None,
        language: str = "ja",
        output_dir: Path = None,
        progress_callback: Callable[[str, float], None] = None,
        use_enhanced_synthesis: bool = True,
    ):
        """
        Initialize ManualResearcher.

        Args:
            llm_client: LLM API client
            evidence_locker: Pre-loaded EvidenceLocker (optional, can be set later)
            language: Target language
            output_dir: Directory for output files
            progress_callback: Callback for progress updates (message, percentage)
            use_enhanced_synthesis: Use multi-pass content generation
        """
        self.llm = llm_client
        self.evidence_locker = evidence_locker
        self.language = language
        self.output_dir = output_dir or Path("./output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.query_generator = QueryGenerator(llm_client, language)
        self.content_extractor = ContentExtractor(llm_client, language)

        self.use_enhanced_synthesis = use_enhanced_synthesis
        self.session: Optional[ResearchSession] = None
        self.progress_callback = progress_callback

    def _report_progress(self, message: str, percentage: float) -> None:
        """Report progress to callback if available."""
        if self.progress_callback:
            self.progress_callback(message, percentage)
        print(f"[{percentage:.1f}%] {message}")

    def load_evidence_from_file(
        self,
        file_path: str | Path,
        column_mapping: Optional[Dict[str, str]] = None,
        encoding: str = "utf-8",
    ) -> EvidenceLocker:
        """
        Load evidence from a CSV/XLSX file.

        Args:
            file_path: Path to evidence file
            column_mapping: Custom column name mapping
            encoding: File encoding for CSV files

        Returns:
            Loaded EvidenceLocker
        """
        self.evidence_locker = load_evidence_file(
            file_path=file_path,
            research_id=str(uuid4())[:8],
            output_dir=self.output_dir / "evidence",
            column_mapping=column_mapping,
            encoding=encoding,
        )
        return self.evidence_locker

    def conduct_research(
        self,
        topic: str,
        requirements: str = "",
        auto_toc: bool = True,
        manual_toc: ManualTableOfContents = None,
        additional_context: str = "",
    ) -> ResearchSession:
        """
        Conduct research using pre-loaded evidence.

        Args:
            topic: The research topic/query
            requirements: Specific research requirements
            auto_toc: Whether to auto-generate table of contents
            manual_toc: Manual table of contents (required if auto_toc=False)
            additional_context: Additional context information

        Returns:
            Completed ResearchSession
        """
        if not self.evidence_locker:
            raise ValueError(
                "Evidence locker not loaded. Use load_evidence_from_file() first."
            )

        if not auto_toc and not manual_toc:
            raise ValueError(
                "manual_toc is required when auto_toc=False"
            )

        # Initialize session
        self.session = ResearchSession(query=topic, requirements=requirements)
        self._report_progress("Initializing manual research session...", 5)

        try:
            # Phase 1: Create or use table of contents
            self.session.state = ResearchState.PLANNING
            self._report_progress("Creating research plan...", 10)

            if auto_toc:
                # Auto-generate TOC based on evidence content
                self.session.research_plan = self._generate_auto_toc(
                    topic=topic,
                    requirements=requirements,
                    additional_context=additional_context,
                )
            else:
                # Use manual TOC
                self.session.research_plan = self._create_plan_from_manual_toc(
                    topic=topic,
                    manual_toc=manual_toc,
                    requirements=requirements,
                )

            self._report_progress(
                f"Research plan created with "
                f"{len(self.session.research_plan.table_of_contents.items)} sections",
                20
            )

            # Phase 2: Match evidence to sections
            self.session.state = ResearchState.RESEARCHING
            self._report_progress("Matching evidence to sections...", 25)
            self._match_evidence_to_sections()

            # Phase 3: Generate content for each section
            self._report_progress("Generating section content...", 40)
            self._generate_section_contents()

            # Phase 4: Synthesize findings
            self._report_progress("Synthesizing findings...", 85)
            self.session.state = ResearchState.SYNTHESIZING
            self._synthesize_findings()

            # Mark completion
            self.session.state = ResearchState.COMPLETED
            self.session.completed_at = datetime.now().isoformat()
            self._report_progress("Research completed!", 100)

            # Save session
            session_path = self.output_dir / f"session_{self.session.session_id}.json"
            self.session.save(session_path)

            # Export evidence
            self.evidence_locker.export_to_json()
            self.evidence_locker.export_to_csv()

        except Exception as e:
            self.session.state = ResearchState.ERROR
            self.session.error_message = str(e)
            self._report_progress(f"Error: {e}", -1)
            raise

        return self.session

    def _generate_auto_toc(
        self,
        topic: str,
        requirements: str,
        additional_context: str,
    ) -> ResearchPlan:
        """Generate table of contents automatically based on evidence content."""

        # Get summary of all evidence
        evidence_summaries = []
        for evidence in self.evidence_locker.get_all_evidence():
            summary = f"- {evidence.title}: {evidence.content_excerpt[:200]}..."
            evidence_summaries.append(summary)

        evidence_overview = "\n".join(evidence_summaries[:20])  # Limit to 20 items

        prompt = f"""Based on the research topic and available evidence, create a table of contents
for a comprehensive research report.

Research Topic: {topic}
Requirements: {requirements}
Additional Context: {additional_context}

Available Evidence Summary:
{evidence_overview}

Total evidence items: {len(self.evidence_locker.get_all_evidence())}

Create a logical table of contents that:
1. Covers the main aspects of the topic
2. Groups related evidence logically
3. Follows a clear narrative structure
4. Uses {"Japanese" if self.language == "ja" else "English"} for section titles

Return as JSON:
{{
    "title": "Report title",
    "summary": "Brief summary of the research scope",
    "table_of_contents": {{
        "title": "Report title",
        "items": [
            {{
                "section": "1",
                "title": "Section title",
                "description": "What this section covers",
                "subsections": [
                    {{"section": "1.1", "title": "Subsection title", "description": "..."}}
                ]
            }}
        ]
    }},
    "key_terms": ["term1", "term2"]
}}"""

        response = self.llm.generate(prompt)

        try:
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                plan_data = json.loads(content[start:end])
                return ResearchPlan.from_dict(plan_data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARNING] Failed to parse auto-generated TOC: {e}")

        # Fallback: create simple TOC
        return self._create_fallback_toc(topic)

    def _create_plan_from_manual_toc(
        self,
        topic: str,
        manual_toc: ManualTableOfContents,
        requirements: str,
    ) -> ResearchPlan:
        """Create research plan from manual table of contents."""
        toc = manual_toc.to_table_of_contents()

        return ResearchPlan(
            title=manual_toc.title,
            summary=f"Research on: {topic}",
            table_of_contents=toc,
            search_queries=[],  # Not used in manual mode
            key_terms=[],
            methodology_notes="Manual evidence-based research",
        )

    def _create_fallback_toc(self, topic: str) -> ResearchPlan:
        """Create a simple fallback TOC when auto-generation fails."""
        if self.language == "ja":
            sections = [
                ("1", "はじめに", "研究の背景と目的"),
                ("2", "現状分析", "現在の状況の概要"),
                ("3", "詳細分析", "詳細な分析と考察"),
                ("4", "結論", "まとめと今後の展望"),
            ]
        else:
            sections = [
                ("1", "Introduction", "Background and objectives"),
                ("2", "Current State Analysis", "Overview of current situation"),
                ("3", "Detailed Analysis", "In-depth analysis and discussion"),
                ("4", "Conclusion", "Summary and future outlook"),
            ]

        items = [
            TableOfContentsItem(section=s[0], title=s[1], description=s[2])
            for s in sections
        ]

        return ResearchPlan(
            title=topic,
            summary=f"Research on: {topic}",
            table_of_contents=TableOfContents(title=topic, items=items),
            search_queries=[],
            key_terms=[],
        )

    def _match_evidence_to_sections(self) -> None:
        """Match evidence items to relevant sections using LLM."""
        if not self.session or not self.session.research_plan:
            return

        toc = self.session.research_plan.table_of_contents
        sections = toc.get_flat_sections()
        all_evidence = self.evidence_locker.get_all_evidence()

        # Prepare section descriptions
        section_info = []
        for section in sections:
            section_info.append(f"{section.section}. {section.title}: {section.description}")

        section_list = "\n".join(section_info)

        # Match each evidence to sections
        for evidence in all_evidence:
            # If evidence already has section reference, use it
            if evidence.section_reference:
                continue

            prompt = f"""Match this evidence to the most relevant section(s) of the report.

Evidence:
Title: {evidence.title}
Content: {evidence.content_excerpt[:500]}

Report Sections:
{section_list}

Return the section number(s) that this evidence is most relevant to.
Return as JSON: {{"sections": ["1", "2.1"]}}
Only include sections where the evidence is directly relevant."""

            try:
                response = self.llm.generate(prompt)
                content = response.content
                start = content.find("{")
                end = content.rfind("}") + 1
                if start != -1 and end > start:
                    result = json.loads(content[start:end])
                    matched_sections = result.get("sections", [])

                    # Update evidence with section reference
                    if matched_sections:
                        evidence.section_reference = matched_sections[0]
                        # Update in locker
                        self.evidence_locker._section_evidence.setdefault(
                            matched_sections[0], []
                        ).append(evidence.id)

            except Exception as e:
                print(f"[WARNING] Evidence matching failed: {e}")
                continue

    def _generate_section_contents(self) -> None:
        """Generate content for each section using matched evidence."""
        if not self.session or not self.session.research_plan:
            return

        toc = self.session.research_plan.table_of_contents
        sections = toc.get_flat_sections()
        total_sections = len(sections)

        for idx, section in enumerate(sections):
            progress = 40 + (idx / total_sections) * 40
            self._report_progress(
                f"Generating content for: {section.section}. {section.title}",
                progress
            )

            section.status = "in_progress"

            # Get evidence for this section
            section_evidence = self.evidence_locker.get_section_evidence(section.section)

            # Also check for evidence without explicit section reference
            # that might be relevant based on content
            if not section_evidence:
                section_evidence = self._find_relevant_evidence(section)

            # Convert evidence to ExtractedContent for synthesis
            extracted_contents = []
            for evidence in section_evidence:
                ec = ExtractedContent(
                    source_url=evidence.url,
                    source_title=evidence.title,
                    processed_content=evidence.content_excerpt,
                    relevance_score=evidence.relevance_score or 0.5,
                    key_points=[],
                    images=[],
                )
                extracted_contents.append(ec)

            # Generate section content
            self._generate_and_save_section_content(section, extracted_contents)
            section.status = "completed"

    def _find_relevant_evidence(
        self,
        section: TableOfContentsItem,
        max_items: int = 5,
    ) -> List[Evidence]:
        """Find evidence relevant to a section based on content matching."""
        all_evidence = self.evidence_locker.get_all_evidence()

        # Use LLM to score relevance
        relevant = []
        section_context = f"{section.section}. {section.title}: {section.description}"

        for evidence in all_evidence:
            # Skip if already assigned to another section
            if evidence.section_reference and evidence.section_reference != section.section:
                continue

            # Simple keyword matching as fallback
            section_keywords = set(
                section.title.lower().split() + section.description.lower().split()
            )
            evidence_text = (evidence.title + " " + evidence.content_excerpt).lower()

            matches = sum(1 for kw in section_keywords if kw in evidence_text and len(kw) > 2)
            if matches >= 2:
                relevant.append(evidence)

        return relevant[:max_items]

    def _generate_and_save_section_content(
        self,
        section: TableOfContentsItem,
        section_content_parts: List[ExtractedContent],
    ) -> None:
        """Generate and save content for a section."""
        print(f"[DEBUG] Generating content for section {section.section}...")

        if section_content_parts:
            # Use enhanced multi-pass synthesis
            if self.use_enhanced_synthesis:
                print(f"[DEBUG] Using enhanced multi-pass synthesis")
                synthesized = self.content_extractor.synthesize_section_content_enhanced(
                    section_title=section.title,
                    section_description=section.description,
                    extracted_contents=section_content_parts,
                    requirements=self.session.requirements if self.session else "",
                )
            else:
                synthesized = self.content_extractor.synthesize_section_content(
                    section_title=section.title,
                    section_description=section.description,
                    extracted_contents=section_content_parts,
                    requirements=self.session.requirements if self.session else "",
                )

            content = synthesized.get("content", "")
            print(f"[DEBUG] Generated content length: {len(content)} chars")

            if not content or len(content) < 50:
                print(f"[WARNING] Synthesis returned empty, using fallback")
                content = self._create_fallback_content(section, section_content_parts)

            self.session.section_contents[section.section] = {
                "title": section.title,
                "content": content,
                "summary": synthesized.get("summary", ""),
                "confidence": synthesized.get("confidence_level", "medium"),
                "sources": [ec.source_url for ec in section_content_parts],
                "images": [],
                "gaps": synthesized.get("information_gaps", []),
            }

            section.content = content
            section.sources = [ec.source_url for ec in section_content_parts]
            print(f"[DEBUG] Section {section.section} saved with {len(content)} chars")

        else:
            # No content extracted
            print(f"[WARNING] No evidence found for section {section.section}")
            placeholder = self._create_placeholder_content(section)

            self.session.section_contents[section.section] = {
                "title": section.title,
                "content": placeholder,
                "summary": "",
                "confidence": "low",
                "sources": [],
                "images": [],
                "gaps": [f"All content for section '{section.title}'"],
            }
            section.content = placeholder

    def _create_fallback_content(
        self,
        section: TableOfContentsItem,
        parts: List[ExtractedContent],
    ) -> str:
        """Create fallback content from raw extracted parts."""
        if not parts:
            return self._create_placeholder_content(section)

        content_pieces = []
        for i, part in enumerate(parts[:5], 1):
            if part.processed_content:
                content_pieces.append(f"{part.processed_content[:500]}")

        if content_pieces:
            combined = "\n\n".join(content_pieces)
            if self.language == "ja":
                return f"【{section.title}】\n\n{combined}"
            return f"**{section.title}**\n\n{combined}"

        return self._create_placeholder_content(section)

    def _create_placeholder_content(self, section: TableOfContentsItem) -> str:
        """Create placeholder content for sections with no data."""
        if self.language == "ja":
            return f"このセクション「{section.title}」に関するエビデンスが見つかりませんでした。"
        return f"No evidence found for section '{section.title}'."

    def _synthesize_findings(self) -> None:
        """Synthesize all findings into cohesive content."""
        if not self.session:
            return

        # Create overall summary
        sections_summary = []
        for section_num, content in self.session.section_contents.items():
            sections_summary.append(
                f"Section {section_num}: {content.get('title', '')}\n"
                f"Summary: {content.get('summary', '')[:200]}"
            )

        summary_prompt = f"""Research Topic: {self.session.query}

Section Summaries:
{chr(10).join(sections_summary)}

Requirements: {self.session.requirements}

Create an executive summary of the research findings (300-500 words).
Include:
1. Key findings across all sections
2. Main conclusions
3. Recommendations for further research
4. Confidence level assessment

Return as JSON:
{{"executive_summary": "...", "key_findings": [...], "recommendations": [...], "overall_confidence": "high/medium/low"}}"""

        response = self.llm.generate(summary_prompt)

        try:
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                synthesis = json.loads(content[start:end])
                self.session.section_contents["_executive_summary"] = synthesis
        except (json.JSONDecodeError, ValueError):
            self.session.section_contents["_executive_summary"] = {
                "executive_summary": "Summary generation failed",
                "key_findings": [],
                "recommendations": [],
                "overall_confidence": "low",
            }

    def get_session(self) -> Optional[ResearchSession]:
        """Get current research session."""
        return self.session

    def get_evidence_locker(self) -> Optional[EvidenceLocker]:
        """Get evidence locker."""
        return self.evidence_locker
