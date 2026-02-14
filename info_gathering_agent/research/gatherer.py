"""
Gatherer - Main information gathering orchestration module.

Derived from deep_research_tool's Researcher, focused purely on
information collection and synthesis without report generation or verification.
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
from ..evidence.content_filter import (
    ContentFilter,
    ContentFilterConfig,
    create_moderate_filter,
)
from ..search.base import SearchResult
from ..config import CrawlMode
from .query_generator import QueryGenerator, ResearchPlan, TableOfContents, TableOfContentsItem
from .content_extractor import ContentExtractor, ExtractedContent
from .site_crawler import SiteCrawler, CrawlResult, extract_keywords_from_topic
from .fast_crawler import FastCrawler, EvaluationMode, CrawlResult as FastCrawlResult


class GatheringState(str, Enum):
    """States of the information gathering process."""
    INITIALIZED = "initialized"
    PLANNING = "planning"
    GATHERING = "gathering"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class GatheringIteration:
    """Record of a single gathering iteration."""
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
class GatheringSession:
    """Complete gathering session data."""
    session_id: str = field(default_factory=lambda: str(uuid4())[:8])
    query: str = ""
    requirements: str = ""
    state: GatheringState = GatheringState.INITIALIZED
    research_plan: Optional[ResearchPlan] = None
    iterations: List[GatheringIteration] = field(default_factory=list)
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
    def load(cls, filepath: Path) -> "GatheringSession":
        """Load session from file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        session = cls(
            session_id=data.get("session_id"),
            query=data.get("query", ""),
            requirements=data.get("requirements", ""),
            state=GatheringState(data.get("state", "initialized")),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error_message=data.get("error_message"),
        )

        if data.get("research_plan"):
            session.research_plan = ResearchPlan.from_dict(data["research_plan"])

        session.iterations = [
            GatheringIteration(**i) for i in data.get("iterations", [])
        ]
        session.section_contents = data.get("section_contents", {})

        return session


class Gatherer:
    """
    Main information gathering orchestrator.

    Coordinates the gathering process including:
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
        max_queries_per_iteration: int = 3,
        max_pages_per_query: int = 3,
        language: str = "ja",
        output_dir: Path = None,
        progress_callback: Callable[[str, float], None] = None,
        extended_mode: bool = False,
        crawl_max_pages: int = 10,
        crawl_max_depth: int = 2,
        crawl_max_sites: int = 3,
        crawl_relevance_threshold: float = 0.3,
        use_enhanced_synthesis: bool = True,
        content_filter: ContentFilter = None,
        filter_mode: str = "moderate",
        crawl_mode: CrawlMode = CrawlMode.STANDARD,
        fast_crawl_workers: int = 10,
        fast_crawl_batch_size: int = 5,
    ):
        """
        Initialize Gatherer.

        Args:
            llm_client: LLM API client
            search_client: Web search client
            min_iterations: Minimum gathering iterations per section
            max_iterations: Maximum gathering iterations per section
            max_queries_per_iteration: Maximum queries to execute per iteration
            max_pages_per_query: Maximum pages to process per search query
            language: Target language
            output_dir: Directory for output files
            progress_callback: Callback for progress updates (message, percentage)
            extended_mode: Enable extended mode (deep site crawling)
            crawl_max_pages: Max pages to crawl per site in extended mode
            crawl_max_depth: Max link depth from seed URL
            crawl_max_sites: Max sites to crawl per search
            crawl_relevance_threshold: Min relevance score to include page
            use_enhanced_synthesis: Use multi-pass content generation
            content_filter: Content filter instance
            filter_mode: Filter strictness
            crawl_mode: Crawl mode (standard, fast_batch, fast_parallel)
            fast_crawl_workers: Max parallel workers for fast crawl mode
            fast_crawl_batch_size: Pages per batch in batch evaluation mode
        """
        self.llm = llm_client
        self.search = search_client
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.max_queries_per_iteration = max_queries_per_iteration
        self.max_pages_per_query = max_pages_per_query
        self.language = language
        self.output_dir = output_dir or Path("./output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.query_generator = QueryGenerator(llm_client, language)
        self.content_extractor = ContentExtractor(llm_client, language)

        self.use_enhanced_synthesis = use_enhanced_synthesis

        self.session: Optional[GatheringSession] = None
        self.evidence_locker: Optional[EvidenceLocker] = None
        self.progress_callback = progress_callback

        # Extended mode settings
        self.extended_mode = extended_mode
        self.site_crawler: Optional[SiteCrawler] = None
        if extended_mode:
            self.site_crawler = SiteCrawler(
                search_client=search_client,
                llm_client=llm_client,
                max_pages=crawl_max_pages,
                max_depth=crawl_max_depth,
                relevance_threshold=crawl_relevance_threshold,
                language=language,
            )
            self.crawl_max_sites = crawl_max_sites

        # Content filter for ad/spam removal
        self.filter_mode = filter_mode
        if content_filter:
            self.content_filter = content_filter
        elif filter_mode == "strict":
            from ..evidence.content_filter import create_strict_filter
            self.content_filter = create_strict_filter()
        elif filter_mode == "moderate":
            self.content_filter = create_moderate_filter()
        elif filter_mode == "minimal":
            from ..evidence.content_filter import create_minimal_filter
            self.content_filter = create_minimal_filter()
        else:  # "none"
            self.content_filter = None

        # Fast crawl mode settings
        self.crawl_mode = crawl_mode
        self.fast_crawler: Optional[FastCrawler] = None
        if crawl_mode in (CrawlMode.FAST_BATCH, CrawlMode.FAST_PARALLEL):
            eval_mode = (
                EvaluationMode.BATCH if crawl_mode == CrawlMode.FAST_BATCH
                else EvaluationMode.PARALLEL
            )
            self.fast_crawler = FastCrawler(
                search_client=search_client,
                llm_client=llm_client,
                evaluation_mode=eval_mode,
                content_filter=self.content_filter,
                max_workers=fast_crawl_workers,
                batch_size=fast_crawl_batch_size,
                language=language,
            )

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
    ) -> GatheringSession:
        """
        Conduct complete information gathering process.

        Args:
            query: The research query/topic
            requirements: Specific research requirements
            additional_context: Additional context information

        Returns:
            Completed GatheringSession
        """
        # Initialize session
        self.session = GatheringSession(query=query, requirements=requirements)
        self.evidence_locker = EvidenceLocker(
            research_id=self.session.session_id,
            output_dir=self.output_dir / "evidence",
        )

        try:
            # Phase 1: Planning
            self._report_progress("Creating research plan...", 5)
            self.session.state = GatheringState.PLANNING

            self.session.research_plan = self.query_generator.create_research_plan(
                query=query,
                requirements=requirements,
                additional_context=additional_context,
            )

            self._report_progress(
                f"Research plan created with {len(self.session.research_plan.table_of_contents.items)} sections",
                10
            )

            # Phase 2: Gathering Loop
            self.session.state = GatheringState.GATHERING
            self._conduct_research_loop()

            # Phase 3: Synthesis
            self._report_progress("Synthesizing findings...", 85)
            self.session.state = GatheringState.SYNTHESIZING
            self._synthesize_findings()

            # Mark completion
            self.session.state = GatheringState.COMPLETED
            self.session.completed_at = datetime.now().isoformat()
            self._report_progress("Information gathering completed!", 100)

            # Save session
            session_path = self.output_dir / f"session_{self.session.session_id}.json"
            self.session.save(session_path)

            # Export evidence
            self.evidence_locker.export_to_json()
            self.evidence_locker.export_to_csv()

        except Exception as e:
            self.session.state = GatheringState.ERROR
            self.session.error_message = str(e)
            self._report_progress(f"Error: {e}", -1)
            raise

        return self.session

    def _conduct_research_loop(self) -> None:
        """Execute the main gathering loop."""
        if not self.session or not self.session.research_plan:
            raise ValueError("Gathering session not properly initialized")

        toc = self.session.research_plan.table_of_contents
        sections = toc.get_flat_sections()
        total_sections = len(sections)

        print(f"[DEBUG] Gathering loop starting with {total_sections} sections:")
        for s in sections:
            print(f"  - {s.section}: {s.title}")

        if total_sections == 0:
            print("[ERROR] No sections found in table of contents!")
            return

        # Initial queries from plan
        available_queries = list(self.session.research_plan.search_queries)
        print(f"[DEBUG] Initial search queries: {len(available_queries)}")
        print(f"[DEBUG] Using crawl mode: {self.crawl_mode.value}")

        for section_idx, section in enumerate(sections):
            section_progress_base = 10 + (section_idx / total_sections) * 70

            self._report_progress(
                f"Gathering: {section.section}. {section.title}",
                section_progress_base
            )

            section.status = "in_progress"

            # Choose processing method based on crawl mode
            if self.fast_crawler and self.crawl_mode in (CrawlMode.FAST_BATCH, CrawlMode.FAST_PARALLEL):
                self._process_section_with_fast_crawler(
                    section=section,
                    available_queries=available_queries,
                    section_idx=section_idx,
                    total_sections=total_sections,
                )
            else:
                self._process_section_with_immediate_generation(
                    section=section,
                    available_queries=available_queries,
                    section_idx=section_idx,
                    total_sections=total_sections,
                )

            # Remove used queries
            if available_queries:
                available_queries = available_queries[self.max_queries_per_iteration:]

            section.status = "completed"

        print(f"[DEBUG] Gathering loop completed. Section contents keys: {list(self.session.section_contents.keys())}")

        # Final coherence check
        self._check_report_coherence()

    def _process_section_with_fast_crawler(
        self,
        section: TableOfContentsItem,
        available_queries: List[str],
        section_idx: int,
        total_sections: int,
    ) -> None:
        """Process a section using fast crawler mode."""
        if not self.fast_crawler:
            return self._process_section_with_immediate_generation(
                section, available_queries, section_idx, total_sections
            )

        section_content_parts: List[ExtractedContent] = []

        def fast_progress(msg: str, current: int, total: int):
            self._report_progress(
                f"{section.section}: {msg}",
                10 + (section_idx / total_sections) * 70 + (current / total) * 10
            )

        queries = available_queries[:self.max_queries_per_iteration]
        if not queries:
            queries = self.query_generator.generate_follow_up_queries(
                section, "", []
            )

        print(f"[FastCrawler] Processing section {section.section} with {len(queries)} queries")

        crawl_result: FastCrawlResult = self.fast_crawler.crawl_and_evaluate(
            queries=queries,
            section_context=f"{section.section}. {section.title}: {section.description}",
            max_pages_per_query=self.max_pages_per_query,
            min_relevance_score=0.2,
            progress_callback=fast_progress,
        )

        print(f"[FastCrawler] Found {len(crawl_result.pages)} relevant pages "
              f"(fetch: {crawl_result.total_fetch_time:.1f}s, eval: {crawl_result.total_eval_time:.1f}s)")

        for page in crawl_result.pages:
            extracted = ExtractedContent(
                source_url=page.url,
                source_title=page.title,
                raw_content=page.content,
                processed_content=page.processed_content or page.content[:2000],
                key_points=page.key_points,
                relevance_score=page.relevance_score,
                extraction_notes=f"fast_crawler ({self.crawl_mode.value})",
            )
            section_content_parts.append(extracted)

            self.evidence_locker.add_evidence(
                url=page.url,
                title=page.title,
                content_excerpt=page.processed_content[:500] if page.processed_content else page.snippet,
                evidence_type=EvidenceType.WEB_PAGE,
                search_query=page.metadata.get("query", ""),
                section_reference=section.section,
                relevance_score=page.relevance_score,
            )

        iter_record = GatheringIteration(
            iteration_number=1,
            section=section.section,
            queries_executed=queries,
            sources_found=crawl_result.pages_fetched,
            content_extracted=len(crawl_result.pages),
        )
        iter_record.completed_at = datetime.now().isoformat()
        self.session.iterations.append(iter_record)

        print(f"[FastCrawler] Section {section.section} complete. Parts: {len(section_content_parts)}")
        self._generate_and_save_section_content(section, section_content_parts)

    def _process_section_with_immediate_generation(
        self,
        section: TableOfContentsItem,
        available_queries: List[str],
        section_idx: int,
        total_sections: int,
    ) -> None:
        """Process a section with immediate content generation after research."""
        section_content_parts: List[ExtractedContent] = []

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            iter_record = GatheringIteration(
                iteration_number=iteration,
                section=section.section,
            )

            # Get queries for this iteration
            if iteration == 1 and available_queries:
                queries = available_queries[:self.max_queries_per_iteration]
            else:
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

            if not queries:
                print(f"[WARNING] No queries generated for section {section.section}")
                break

            iter_record.queries_executed = queries
            print(f"[DEBUG] Section {section.section} iteration {iteration}: executing {len(queries[:self.max_queries_per_iteration])} queries")

            # Execute searches and extract content
            for query in queries[:self.max_queries_per_iteration]:
                print(f"[DEBUG] Searching: {query[:50]}...")
                try:
                    results = self.search.search(query)
                    print(f"[DEBUG] Search returned {len(results)} results")
                    iter_record.sources_found += len(results)

                    for result in results[:self.max_pages_per_query]:
                        print(f"[DEBUG] Processing: {result.url[:60]}...")

                        # Apply content filter to URL first
                        if self.content_filter:
                            url_filter_result = self.content_filter.filter_url(result.url)
                            if not url_filter_result.should_include:
                                print(f"[FILTER] Skipped (URL): {url_filter_result.reason}")
                                continue

                        try:
                            page = self.search.get_page_content(result.url)

                            # Apply content filter to page content
                            if self.content_filter:
                                content_filter_result = self.content_filter.filter_content(
                                    url=result.url,
                                    title=result.title,
                                    content=page.text_content,
                                )
                                if not content_filter_result.should_include:
                                    print(f"[FILTER] Skipped (content): {content_filter_result.reason}")
                                    continue
                                print(f"[FILTER] Quality score: {content_filter_result.quality_score:.2f}")

                            extracted = self.content_extractor.extract_relevant_content(
                                raw_content=page.text_content,
                                source_url=result.url,
                                source_title=result.title,
                                section_context=f"{section.section}. {section.title}",
                                research_query=query,
                            )

                            print(f"[DEBUG] Extracted relevance_score: {extracted.relevance_score}")

                            if extracted.relevance_score >= 0.2:
                                section_content_parts.append(extracted)
                                iter_record.content_extracted += 1
                                print(f"[DEBUG] Content added. Total parts: {len(section_content_parts)}")

                                self.evidence_locker.add_evidence(
                                    url=result.url,
                                    title=result.title,
                                    content_excerpt=extracted.processed_content[:500],
                                    evidence_type=EvidenceType.WEB_PAGE,
                                    search_query=query,
                                    section_reference=section.section,
                                    relevance_score=extracted.relevance_score,
                                )
                            else:
                                if extracted.processed_content and len(extracted.processed_content) > 100:
                                    print(f"[DEBUG] Low relevance but adding anyway: {extracted.relevance_score}")
                                    section_content_parts.append(extracted)

                        except Exception as e:
                            print(f"[ERROR] Content extraction error for {result.url}: {e}")
                            continue

                    time.sleep(0.3)

                except Exception as e:
                    print(f"[ERROR] Search error for query '{query}': {e}")
                    continue

            iter_record.completed_at = datetime.now().isoformat()
            self.session.iterations.append(iter_record)

            # Check if we have enough content
            if iteration >= self.min_iterations and len(section_content_parts) >= 2:
                break

        print(f"[DEBUG] Section {section.section} gathering complete. Parts: {len(section_content_parts)}")
        self._generate_and_save_section_content(section, section_content_parts)

    def _generate_and_save_section_content(
        self,
        section: TableOfContentsItem,
        section_content_parts: List[ExtractedContent],
    ) -> None:
        """Immediately generate and save content for a section."""
        print(f"[DEBUG] Generating content for section {section.section}...")

        if section_content_parts:
            if self.use_enhanced_synthesis:
                print(f"[DEBUG] Using enhanced multi-pass synthesis")
                synthesized = self.content_extractor.synthesize_section_content_enhanced(
                    section_title=section.title,
                    section_description=section.description,
                    extracted_contents=section_content_parts,
                    requirements=self.session.requirements,
                )
            else:
                synthesized = self.content_extractor.synthesize_section_content(
                    section_title=section.title,
                    section_description=section.description,
                    extracted_contents=section_content_parts,
                    requirements=self.session.requirements,
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
                "images": [img for ec in section_content_parts for img in ec.images][:5],
                "gaps": synthesized.get("information_gaps", []),
            }

            section.content = content
            section.sources = [ec.source_url for ec in section_content_parts]
            print(f"[DEBUG] Section {section.section} saved with {len(content)} chars")

        else:
            print(f"[WARNING] No content extracted for section {section.section}")
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
            return f"このセクション「{section.title}」の情報を自動的に収集できませんでした。追加のリサーチが必要です。"
        return f"Information for section '{section.title}' could not be gathered automatically. Additional research is needed."

    def _check_report_coherence(self) -> None:
        """Check the logical coherence of gathered content."""
        if not self.session or not self.session.section_contents:
            print("[WARNING] No content to check for coherence")
            return

        print(f"[DEBUG] Checking content coherence...")

        sections_text = []
        for section_id, content in self.session.section_contents.items():
            if section_id.startswith("_"):
                continue
            title = content.get("title", section_id)
            text = content.get("content", "")[:500]
            sections_text.append(f"Section {section_id}: {title}\n{text}...")

        if not sections_text:
            print("[WARNING] No sections to check")
            return

        coherence_prompt = f"""Review the following sections and identify any logical inconsistencies or gaps:

{chr(10).join(sections_text)}

Return JSON:
{{
    "is_coherent": true/false,
    "issues": ["issue1", "issue2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}"""

        try:
            response = self.llm.generate(coherence_prompt)
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(content[start:end])
                self.session.section_contents["_coherence_check"] = result
                print(f"[DEBUG] Coherence check: {result.get('is_coherent', 'unknown')}")
                if result.get("issues"):
                    print(f"[DEBUG] Issues found: {result['issues']}")
        except Exception as e:
            print(f"[WARNING] Coherence check failed: {e}")

    def _conduct_extended_research(
        self,
        section: TableOfContentsItem,
        initial_results: List[SearchResult],
        section_content_parts: List[ExtractedContent],
    ) -> Dict[str, Any]:
        """Conduct extended research by crawling sites from initial search results."""
        if not self.site_crawler:
            return {"crawl_results": [], "suggested_queries": []}

        self._report_progress(
            f"Extended mode: Crawling sites for {section.section}",
            -1
        )

        keywords = extract_keywords_from_topic(
            f"{section.title} {section.description}"
        )

        existing_content = "\n".join(
            ec.processed_content[:500] for ec in section_content_parts
        )

        seed_urls = [r.url for r in initial_results[:5]]

        crawl_results = self.site_crawler.crawl_multiple_sites(
            seed_urls=seed_urls,
            research_topic=f"{section.title}",
            keywords=keywords,
            section_context=f"{section.section}. {section.title}",
            existing_content=existing_content,
            max_sites=self.crawl_max_sites,
        )

        all_suggested_queries = []

        for crawl_result in crawl_results:
            self._report_progress(
                f"Found {crawl_result.pages_relevant} relevant pages at {crawl_result.root_domain}",
                -1
            )

            all_suggested_queries.extend(crawl_result.suggested_queries)

            for crawled_page in crawl_result.crawled_pages:
                if crawled_page.relevance_score >= 0.3:
                    extracted = self.content_extractor.extract_relevant_content(
                        raw_content=crawled_page.content,
                        source_url=crawled_page.url,
                        source_title=crawled_page.title,
                        section_context=f"{section.section}. {section.title}",
                        research_query=f"{section.title} (crawled)",
                    )

                    if extracted.relevance_score >= 0.3:
                        section_content_parts.append(extracted)

                        self.evidence_locker.add_evidence(
                            url=crawled_page.url,
                            title=crawled_page.title,
                            content_excerpt=extracted.processed_content[:500],
                            evidence_type=EvidenceType.WEB_PAGE,
                            search_query=f"crawled from {crawl_result.root_domain}",
                            section_reference=section.section,
                            relevance_score=extracted.relevance_score,
                        )

        return {
            "crawl_results": crawl_results,
            "suggested_queries": list(set(all_suggested_queries))[:5],
        }

    def _synthesize_findings(self) -> None:
        """Synthesize all findings into an executive summary."""
        if not self.session:
            return

        sections_summary = []
        for section_num, content in self.session.section_contents.items():
            if section_num.startswith("_"):
                continue
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

    def get_session(self) -> Optional[GatheringSession]:
        """Get current gathering session."""
        return self.session

    def get_evidence_locker(self) -> Optional[EvidenceLocker]:
        """Get evidence locker."""
        return self.evidence_locker

    def resume_gathering(
        self,
        session_path: Path,
        additional_iterations: int = 2,
    ) -> GatheringSession:
        """
        Resume a previous gathering session.

        Args:
            session_path: Path to saved session file
            additional_iterations: Additional iterations to run

        Returns:
            Updated GatheringSession
        """
        self.session = GatheringSession.load(session_path)

        # Load evidence if available
        evidence_path = self.output_dir / "evidence" / f"evidence_{self.session.session_id}.json"
        if evidence_path.exists():
            self.evidence_locker = EvidenceLocker.load_from_json(evidence_path)
        else:
            self.evidence_locker = EvidenceLocker(
                research_id=self.session.session_id,
                output_dir=self.output_dir / "evidence",
            )

        if self.session.state != GatheringState.COMPLETED:
            self.min_iterations = additional_iterations
            self.session.state = GatheringState.GATHERING
            self._conduct_research_loop()
            self._synthesize_findings()
            self.session.state = GatheringState.COMPLETED
            self.session.completed_at = datetime.now().isoformat()

        return self.session
