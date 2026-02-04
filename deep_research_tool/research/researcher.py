"""
Researcher - Main research orchestration module.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from uuid import uuid4

from ..evidence.locker import EvidenceLocker, Evidence, EvidenceType
from ..search.base import SearchResult
from .query_generator import QueryGenerator, ResearchPlan, TableOfContents, TableOfContentsItem
from .content_extractor import ContentExtractor, ExtractedContent


class ResearchState(str, Enum):
    """States of the research process."""
    INITIALIZED = "initialized"
    PLANNING = "planning"
    RESEARCHING = "researching"
    SYNTHESIZING = "synthesizing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ResearchIteration:
    """Record of a single research iteration."""
    iteration_number: int
    section: str
    queries_executed: List[str] = field(default_factory=list)
    sources_found: int = 0
    content_extracted: int = 0
    gaps_identified: List[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration_number": self.iteration_number,
            "section": self.section,
            "queries_executed": self.queries_executed,
            "sources_found": self.sources_found,
            "content_extracted": self.content_extracted,
            "gaps_identified": self.gaps_identified,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class ResearchSession:
    """Complete research session data."""
    session_id: str = field(default_factory=lambda: str(uuid4())[:8])
    query: str = ""
    requirements: str = ""
    state: ResearchState = ResearchState.INITIALIZED
    research_plan: Optional[ResearchPlan] = None
    iterations: List[ResearchIteration] = field(default_factory=list)
    section_contents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "requirements": self.requirements,
            "state": self.state.value,
            "research_plan": self.research_plan.to_dict() if self.research_plan else None,
            "iterations": [i.to_dict() for i in self.iterations],
            "section_contents": self.section_contents,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
        }

    def save(self, filepath: Path) -> None:
        """Save session to file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "ResearchSession":
        """Load session from file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        session = cls(
            session_id=data.get("session_id"),
            query=data.get("query", ""),
            requirements=data.get("requirements", ""),
            state=ResearchState(data.get("state", "initialized")),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error_message=data.get("error_message"),
        )

        if data.get("research_plan"):
            session.research_plan = ResearchPlan.from_dict(data["research_plan"])

        session.iterations = [
            ResearchIteration(**i) for i in data.get("iterations", [])
        ]
        session.section_contents = data.get("section_contents", {})

        return session


class Researcher:
    """
    Main research orchestrator.

    Coordinates the research process including:
    - Query analysis and plan creation
    - Iterative web search and content extraction
    - Information synthesis
    - Evidence tracking
    """

    def __init__(
        self,
        llm_client,
        search_client,
        min_iterations: int = 3,
        max_iterations: int = 10,
        language: str = "ja",
        output_dir: Path = None,
        progress_callback: Callable[[str, float], None] = None,
    ):
        """
        Initialize Researcher.

        Args:
            llm_client: LLM API client
            search_client: Web search client
            min_iterations: Minimum research iterations per section
            max_iterations: Maximum research iterations per section
            language: Target language
            output_dir: Directory for output files
            progress_callback: Callback for progress updates (message, percentage)
        """
        self.llm = llm_client
        self.search = search_client
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.language = language
        self.output_dir = output_dir or Path("./output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.query_generator = QueryGenerator(llm_client, language)
        self.content_extractor = ContentExtractor(llm_client, language)

        self.session: Optional[ResearchSession] = None
        self.evidence_locker: Optional[EvidenceLocker] = None
        self.progress_callback = progress_callback

    def _report_progress(self, message: str, percentage: float) -> None:
        """Report progress to callback if available."""
        if self.progress_callback:
            self.progress_callback(message, percentage)
        print(f"[{percentage:.1f}%] {message}")

    def conduct_research(
        self,
        query: str,
        requirements: str = "",
        additional_context: str = "",
        additional_documents: List[Dict[str, str]] = None,
    ) -> ResearchSession:
        """
        Conduct complete research process.

        Args:
            query: The research query/topic
            requirements: Specific research requirements
            additional_context: Additional context information
            additional_documents: List of additional documents with 'path' and 'content'

        Returns:
            Completed ResearchSession
        """
        # Initialize session
        self.session = ResearchSession(query=query, requirements=requirements)
        self.evidence_locker = EvidenceLocker(
            research_id=self.session.session_id,
            output_dir=self.output_dir / "evidence",
        )

        try:
            # Phase 1: Planning
            self._report_progress("Creating research plan...", 5)
            self.session.state = ResearchState.PLANNING

            # Include additional documents in context
            context = additional_context
            if additional_documents:
                doc_summaries = []
                for doc in additional_documents:
                    # Add to evidence locker as user-provided
                    self.evidence_locker.add_evidence(
                        url=doc.get("path", ""),
                        title=doc.get("title", Path(doc.get("path", "")).name),
                        content_excerpt=doc.get("content", "")[:1000],
                        evidence_type=EvidenceType.USER_PROVIDED,
                    )
                    doc_summaries.append(
                        f"Document: {doc.get('title', 'Unknown')}\n"
                        f"Content: {doc.get('content', '')[:2000]}"
                    )
                context += "\n\nAdditional Reference Documents:\n" + "\n---\n".join(doc_summaries)

            self.session.research_plan = self.query_generator.create_research_plan(
                query=query,
                requirements=requirements,
                additional_context=context,
            )

            self._report_progress(
                f"Research plan created with {len(self.session.research_plan.table_of_contents.items)} sections",
                10
            )

            # Phase 2: Research Loop
            self.session.state = ResearchState.RESEARCHING
            self._conduct_research_loop()

            # Phase 3: Synthesis
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

    def _conduct_research_loop(self) -> None:
        """Execute the main research loop."""
        if not self.session or not self.session.research_plan:
            raise ValueError("Research session not properly initialized")

        toc = self.session.research_plan.table_of_contents
        sections = toc.get_flat_sections()
        total_sections = len(sections)

        # Initial queries from plan
        available_queries = list(self.session.research_plan.search_queries)

        for section_idx, section in enumerate(sections):
            section_progress_base = 10 + (section_idx / total_sections) * 70

            self._report_progress(
                f"Researching: {section.section}. {section.title}",
                section_progress_base
            )

            section.status = "in_progress"
            section_content_parts: List[ExtractedContent] = []

            # Research iterations for this section
            iteration = 0
            while iteration < self.max_iterations:
                iteration += 1
                iter_record = ResearchIteration(
                    iteration_number=iteration,
                    section=section.section,
                )

                # Get queries for this iteration
                if iteration == 1 and available_queries:
                    # Use initial queries from plan
                    queries = available_queries[:3]
                    available_queries = available_queries[3:]
                else:
                    # Generate follow-up queries
                    current_content = "\n".join(
                        ec.processed_content for ec in section_content_parts
                    )
                    gaps = self.query_generator.identify_gaps(
                        section, current_content, self.session.requirements
                    )
                    iter_record.gaps_identified = gaps

                    queries = self.query_generator.generate_follow_up_queries(
                        section, current_content, gaps
                    )

                iter_record.queries_executed = queries

                # Execute searches
                for query in queries[:3]:  # Limit queries per iteration
                    try:
                        results = self.search.search(query)
                        iter_record.sources_found += len(results)

                        # Extract content from top results
                        for result in results[:3]:  # Top 3 results per query
                            try:
                                # Get full page content
                                page = self.search.get_page_content(result.url)

                                # Extract relevant content
                                extracted = self.content_extractor.extract_relevant_content(
                                    raw_content=page.text_content,
                                    source_url=result.url,
                                    source_title=result.title,
                                    section_context=f"{section.section}. {section.title}",
                                    research_query=query,
                                )

                                if extracted.relevance_score >= 0.3:
                                    section_content_parts.append(extracted)
                                    iter_record.content_extracted += 1

                                    # Add to evidence locker
                                    self.evidence_locker.add_evidence(
                                        url=result.url,
                                        title=result.title,
                                        content_excerpt=extracted.processed_content[:500],
                                        evidence_type=EvidenceType.WEB_PAGE,
                                        search_query=query,
                                        section_reference=section.section,
                                        relevance_score=extracted.relevance_score,
                                    )

                                    # Handle images
                                    if page.images:
                                        relevant_images = self.content_extractor.extract_images_with_context(
                                            page.images[:5],
                                            page.text_content[:1000],
                                            f"{section.section}. {section.title}",
                                        )
                                        extracted.images = [
                                            img for img in relevant_images
                                            if img.get("relevance") in ["high", "medium"]
                                        ]

                            except Exception as e:
                                print(f"Content extraction error for {result.url}: {e}")
                                continue

                        # Small delay to avoid rate limiting
                        time.sleep(0.5)

                    except Exception as e:
                        print(f"Search error for query '{query}': {e}")
                        continue

                iter_record.completed_at = datetime.now().isoformat()
                self.session.iterations.append(iter_record)

                # Check if we have enough content
                if (iteration >= self.min_iterations and
                    len(section_content_parts) >= 3 and
                    len(iter_record.gaps_identified) <= 2):
                    break

            # Synthesize section content
            if section_content_parts:
                synthesized = self.content_extractor.synthesize_section_content(
                    section_title=section.title,
                    section_description=section.description,
                    extracted_contents=section_content_parts,
                    requirements=self.session.requirements,
                )

                self.session.section_contents[section.section] = {
                    "title": section.title,
                    "content": synthesized.get("content", ""),
                    "summary": synthesized.get("summary", ""),
                    "confidence": synthesized.get("confidence_level", "medium"),
                    "sources": [ec.source_url for ec in section_content_parts],
                    "images": [
                        img for ec in section_content_parts
                        for img in ec.images
                    ][:5],  # Limit images per section
                    "gaps": synthesized.get("information_gaps", []),
                }

                section.content = synthesized.get("content", "")
                section.sources = [ec.source_url for ec in section_content_parts]

            section.status = "completed"

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

    def resume_research(
        self,
        session_path: Path,
        additional_iterations: int = 2,
    ) -> ResearchSession:
        """
        Resume a previous research session.

        Args:
            session_path: Path to saved session file
            additional_iterations: Additional iterations to run

        Returns:
            Updated ResearchSession
        """
        self.session = ResearchSession.load(session_path)

        # Load evidence if available
        evidence_path = self.output_dir / "evidence" / f"evidence_{self.session.session_id}.json"
        if evidence_path.exists():
            self.evidence_locker = EvidenceLocker.load_from_json(evidence_path)
        else:
            self.evidence_locker = EvidenceLocker(
                research_id=self.session.session_id,
                output_dir=self.output_dir / "evidence",
            )

        # Continue research with additional iterations
        if self.session.state != ResearchState.COMPLETED:
            self.min_iterations = additional_iterations
            self.session.state = ResearchState.RESEARCHING
            self._conduct_research_loop()
            self._synthesize_findings()
            self.session.state = ResearchState.COMPLETED
            self.session.completed_at = datetime.now().isoformat()

        return self.session
