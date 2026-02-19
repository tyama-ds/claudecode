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
from ..evidence.content_filter import (
    ContentFilter,
    ContentFilterConfig,
    create_moderate_filter,
)
from ..utils.helpers import ResearchWarnings
from ..search.base import SearchResult
from ..search.multilingual import MultilingualSearcher, MultilingualSearchResult
from ..config import CrawlMode, MultilingualSearchConfig
from .query_generator import QueryGenerator, ResearchPlan, TableOfContents, TableOfContentsItem
from .content_extractor import ContentExtractor, ExtractedContent
from .site_crawler import SiteCrawler, CrawlResult, extract_keywords_from_topic
from .fast_crawler import FastCrawler, EvaluationMode, CrawlResult as FastCrawlResult


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
        multilingual_config: MultilingualSearchConfig = None,
        max_content_length: int = 50000,
        target_pages: int = None,
        target_characters: int = None,
    ):
        """
        Initialize Researcher.

        Args:
            llm_client: LLM API client
            search_client: Web search client
            min_iterations: Minimum research iterations per section
            max_iterations: Maximum research iterations per section
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
            use_enhanced_synthesis: Use multi-pass content generation for better quality
            content_filter: Content filter instance (None to use default based on filter_mode)
            filter_mode: Filter strictness: "strict", "moderate", "minimal", or "none"
            crawl_mode: Crawl mode (standard, fast_batch, fast_parallel)
            fast_crawl_workers: Max parallel workers for fast crawl mode
            fast_crawl_batch_size: Pages per batch in batch evaluation mode
            multilingual_config: Multilingual search configuration (None to disable)
            max_content_length: Maximum content length for extraction truncation
            target_pages: Target output page count (used for dynamic content sizing)
            target_characters: Target output character count (overrides target_pages)
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
        self.max_content_length = max_content_length

        # Store target length info (used to calculate per-section target after plan is created)
        self._target_pages = target_pages
        self._target_characters = target_characters
        # ContentExtractor is created without per-section target initially;
        # it will be updated in _execute_research_loop once we know the section count.
        self.content_extractor = ContentExtractor(llm_client, language)

        # Use enhanced multi-pass content generation for better quality
        self.use_enhanced_synthesis = use_enhanced_synthesis

        self.session: Optional[ResearchSession] = None
        self.evidence_locker: Optional[EvidenceLocker] = None
        self.progress_callback = progress_callback

        # Multilingual search settings
        self.multilingual_config = multilingual_config
        self.multilingual_searcher: Optional[MultilingualSearcher] = None
        if multilingual_config and multilingual_config.enabled:
            self.multilingual_searcher = MultilingualSearcher(
                config=multilingual_config,
                search_client=search_client,
                llm_client=llm_client,
                progress_callback=progress_callback,
            )
            print(f"[Researcher] Multilingual search enabled: languages={multilingual_config.search_languages}")

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
                    full_content = doc.get("content", "")
                    self.evidence_locker.add_evidence(
                        url=doc.get("path", ""),
                        title=doc.get("title", Path(doc.get("path", "")).name),
                        content_excerpt=full_content[:1000],
                        extracted_text=full_content,
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

        # Debug: Log sections to be processed
        print(f"[DEBUG] Research loop starting with {total_sections} sections:")
        for s in sections:
            print(f"  - {s.section}: {s.title}")

        if total_sections == 0:
            print("[ERROR] No sections found in table of contents!")
            return

        # Calculate per-section character target and propagate to ContentExtractor
        if total_sections > 0 and (self._target_pages or self._target_characters):
            if self._target_characters:
                total_target = self._target_characters
            else:
                chars_per_page = 1500 if self.language == "ja" else 2500
                total_target = self._target_pages * chars_per_page
            target_per_section = total_target // total_sections
            self.content_extractor.target_chars_per_section = target_per_section
            print(f"[DEBUG] Dynamic content sizing: {total_target:,} total chars / "
                  f"{total_sections} sections = {target_per_section:,} chars/section")

        # Initial queries from plan
        available_queries = list(self.session.research_plan.search_queries)
        print(f"[DEBUG] Initial search queries: {len(available_queries)}")

        # Log crawl mode
        print(f"[DEBUG] Using crawl mode: {self.crawl_mode.value}")

        for section_idx, section in enumerate(sections):
            section_progress_base = 10 + (section_idx / total_sections) * 70

            self._report_progress(
                f"Researching: {section.section}. {section.title}",
                section_progress_base
            )

            section.status = "in_progress"

            # Choose processing method based on crawl mode
            if self.fast_crawler and self.crawl_mode in (CrawlMode.FAST_BATCH, CrawlMode.FAST_PARALLEL):
                # Fast crawl mode: parallel fetch + batch/parallel evaluation
                self._process_section_with_fast_crawler(
                    section=section,
                    available_queries=available_queries,
                    section_idx=section_idx,
                    total_sections=total_sections,
                )
            else:
                # Standard mode: sequential processing
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

        # Debug: Log final section contents
        print(f"[DEBUG] Research loop completed. Section contents keys: {list(self.session.section_contents.keys())}")

        # Final coherence check
        self._check_report_coherence()

    def _process_section_with_fast_crawler(
        self,
        section: TableOfContentsItem,
        available_queries: List[str],
        section_idx: int,
        total_sections: int,
    ) -> None:
        """
        Process a section using fast crawler mode.

        This method:
        1. Uses FastCrawler for parallel fetching and evaluation
        2. Generates section content from evaluated pages
        """
        if not self.fast_crawler:
            # Fallback to standard processing
            return self._process_section_with_immediate_generation(
                section, available_queries, section_idx, total_sections
            )

        section_content_parts: List[ExtractedContent] = []

        # Progress callback for fast crawler
        def fast_progress(msg: str, current: int, total: int):
            self._report_progress(
                f"{section.section}: {msg}",
                10 + (section_idx / total_sections) * 70 + (current / total) * 10
            )

        # Get queries for this section
        queries = available_queries[:self.max_queries_per_iteration]
        if not queries:
            # Generate queries if none available
            queries = self.query_generator.generate_follow_up_queries(
                section, "", [],
                research_topic=self.session.query,
            )

        print(f"[FastCrawler] Processing section {section.section} with {len(queries)} queries")

        # Use FastCrawler for parallel fetch and batch/parallel evaluation
        crawl_result: FastCrawlResult = self.fast_crawler.crawl_and_evaluate(
            queries=queries,
            section_context=f"{section.section}. {section.title}: {section.description}",
            research_topic=self.session.query,  # Pass original research topic for context-aware evaluation
            max_pages_per_query=self.max_pages_per_query,
            min_relevance_score=0.2,
            progress_callback=fast_progress,
        )

        print(f"[FastCrawler] Found {len(crawl_result.pages)} relevant pages "
              f"(fetch: {crawl_result.total_fetch_time:.1f}s, eval: {crawl_result.total_eval_time:.1f}s)")

        # Convert evaluated pages to ExtractedContent
        for page in crawl_result.pages:
            # Create ExtractedContent from evaluated page
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

            # Add to evidence locker
            full_text = page.content or page.processed_content or ""
            self.evidence_locker.add_evidence(
                url=page.url,
                title=page.title,
                content_excerpt=page.processed_content[:500] if page.processed_content else page.snippet,
                extracted_text=full_text,
                evidence_type=EvidenceType.WEB_PAGE,
                search_query=page.metadata.get("query", ""),
                section_reference=section.section,
                relevance_score=page.relevance_score,
            )

        # Create research iteration record
        iter_record = ResearchIteration(
            iteration_number=1,
            section=section.section,
            queries_executed=queries,
            sources_found=crawl_result.pages_fetched,
            content_extracted=len(crawl_result.pages),
        )
        iter_record.completed_at = datetime.now().isoformat()
        self.session.iterations.append(iter_record)

        # Generate and save section content
        print(f"[FastCrawler] Section {section.section} complete. Parts: {len(section_content_parts)}")
        self._generate_and_save_section_content(section, section_content_parts)

    def _process_section_with_immediate_generation(
        self,
        section: TableOfContentsItem,
        available_queries: List[str],
        section_idx: int,
        total_sections: int,
    ) -> None:
        """
        Process a section with immediate content generation after research.

        This method:
        1. Searches for information
        2. Extracts relevant content
        3. Immediately generates section content (not waiting until the end)
        """
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
                    section, current_content, gaps,
                    research_topic=self.session.query,
                )

            if not queries:
                print(f"[WARNING] No queries generated for section {section.section}")
                break

            iter_record.queries_executed = queries
            queries_to_run = queries[:self.max_queries_per_iteration]
            print(f"\n[Search] Section {section.section} - {len(queries_to_run)} queries to execute")

            # Execute searches and extract content
            for qi, query in enumerate(queries_to_run, 1):
                print(f"[Search] ({qi}/{len(queries_to_run)}) Query: {query}")
                try:
                    # Use multilingual search if enabled, otherwise standard search
                    if self.multilingual_searcher:
                        ml_results, ml_stats = self.multilingual_searcher.search_parallel(query)
                        # Convert MultilingualSearchResult to SearchResult for unified processing
                        results = [
                            SearchResult(
                                title=mr.title,
                                url=mr.url,
                                snippet=mr.snippet,
                                metadata={
                                    "source_language": mr.source_language,
                                    "is_translated": mr.is_translated,
                                    "translation_confidence": mr.translation_confidence,
                                    "relevance_score": mr.relevance_score,
                                },
                            )
                            for mr in ml_results
                        ]
                        print(f"[DEBUG] Multilingual search returned {len(results)} results "
                              f"(deduped from {ml_stats.total_results + ml_stats.duplicates_removed}, "
                              f"languages: {ml_stats.results_by_language})")
                    else:
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

                            # Apply max_content_length truncation
                            raw_content = page.text_content
                            if len(raw_content) > self.max_content_length:
                                raw_content = raw_content[:self.max_content_length]

                            extracted = self.content_extractor.extract_relevant_content(
                                raw_content=raw_content,
                                source_url=result.url,
                                source_title=result.title,
                                section_context=f"{section.section}. {section.title}",
                                research_query=query,
                            )

                            # Capture images from page content and attach to ExtractedContent
                            page_images = getattr(page, 'images', []) or []
                            if page_images and not extracted.images:
                                extracted.images = [
                                    {"src": img.get("src", ""), "alt": img.get("alt", ""),
                                     "title": img.get("title", ""), "page_title": result.title}
                                    for img in page_images[:5]
                                    if img.get("src", "")
                                ]

                            # Follow links to PDF/XLSX/DOCX documents on the page
                            page_links = getattr(page, 'links', []) or []
                            doc_links = [
                                link for link in page_links
                                if any(link.get("url", "").lower().endswith(ext)
                                       for ext in ('.pdf', '.xlsx', '.xls', '.docx', '.csv'))
                            ]
                            for doc_link in doc_links[:2]:  # Limit to 2 document links per page
                                doc_url = doc_link.get("url", "")
                                if doc_url:
                                    try:
                                        print(f"[DEBUG] Following document link: {doc_url[:60]}...")
                                        doc_page = self.search.get_page_content(doc_url)
                                        if doc_page.text_content and len(doc_page.text_content) > 50:
                                            doc_extracted = self.content_extractor.extract_relevant_content(
                                                raw_content=doc_page.text_content[:self.max_content_length],
                                                source_url=doc_url,
                                                source_title=doc_link.get("text", "") or doc_page.title,
                                                section_context=f"{section.section}. {section.title}",
                                                research_query=query,
                                            )
                                            if doc_extracted.relevance_score >= 0.2:
                                                section_content_parts.append(doc_extracted)
                                                iter_record.content_extracted += 1
                                                # Determine evidence type from extension
                                                doc_url_lower = doc_url.lower()
                                                if doc_url_lower.endswith('.pdf'):
                                                    ev_type = EvidenceType.PDF_DOCUMENT
                                                else:
                                                    ev_type = EvidenceType.WEB_PAGE
                                                self.evidence_locker.add_evidence(
                                                    url=doc_url,
                                                    title=doc_link.get("text", "") or doc_page.title,
                                                    content_excerpt=doc_extracted.processed_content[:500],
                                                    extracted_text=doc_page.text_content or doc_extracted.processed_content,
                                                    evidence_type=ev_type,
                                                    search_query=query,
                                                    section_reference=section.section,
                                                    relevance_score=doc_extracted.relevance_score,
                                                )
                                                print(f"[DEBUG] Document content added from {doc_url[:50]}")
                                    except Exception as doc_e:
                                        print(f"[DEBUG] Failed to extract document link {doc_url[:50]}: {doc_e}")
                                        ResearchWarnings.get_instance().add(
                                            ResearchWarnings.HIGH,
                                            "Researcher",
                                            f"Document extraction failed: {doc_url[:80]}. "
                                            f"PDF/XLSX/DOCX content lost. Error: {doc_e}",
                                        )

                            print(f"[DEBUG] Extracted relevance_score: {extracted.relevance_score}")

                            # Lower threshold to 0.2 to get more content
                            if extracted.relevance_score >= 0.2:
                                section_content_parts.append(extracted)
                                iter_record.content_extracted += 1
                                print(f"[DEBUG] Content added. Total parts: {len(section_content_parts)}")

                                # Build evidence kwargs with multilingual metadata
                                evidence_kwargs = {
                                    "url": result.url,
                                    "title": result.title,
                                    "content_excerpt": extracted.processed_content[:500],
                                    "extracted_text": raw_content,
                                    "evidence_type": EvidenceType.WEB_PAGE,
                                    "search_query": query,
                                    "section_reference": section.section,
                                    "relevance_score": extracted.relevance_score,
                                }
                                if result.metadata.get("source_language"):
                                    evidence_kwargs["source_language"] = result.metadata["source_language"]
                                    evidence_kwargs["is_translated"] = result.metadata.get("is_translated", False)
                                    evidence_kwargs["translation_confidence"] = result.metadata.get("translation_confidence", 1.0)

                                self.evidence_locker.add_evidence(**evidence_kwargs)
                            else:
                                # Even low relevance content can be useful - add with note
                                if extracted.processed_content and len(extracted.processed_content) > 100:
                                    print(f"[DEBUG] Low relevance but adding anyway: {extracted.relevance_score}")
                                    section_content_parts.append(extracted)

                        except Exception as e:
                            print(f"[ERROR] Content extraction error for {result.url}: {e}")
                            ResearchWarnings.get_instance().add(
                                ResearchWarnings.HIGH,
                                "Researcher",
                                f"Content extraction failed for {result.url[:80]}. "
                                f"All evidence from this page lost. Error: {e}",
                            )
                            continue

                    time.sleep(0.3)

                except Exception as e:
                    print(f"[ERROR] Search error for query '{query}': {e}")
                    ResearchWarnings.get_instance().add(
                        ResearchWarnings.HIGH,
                        "Researcher",
                        f"Search query failed: '{query[:60]}'. "
                        f"All results for this query lost. Error: {e}",
                    )
                    continue

            iter_record.completed_at = datetime.now().isoformat()
            self.session.iterations.append(iter_record)

            # Check if we have enough content
            if iteration >= self.min_iterations and len(section_content_parts) >= 2:
                break

        # IMMEDIATE CONTENT GENERATION after research for this section
        print(f"[DEBUG] Section {section.section} research complete. Parts: {len(section_content_parts)}")
        self._generate_and_save_section_content(section, section_content_parts)

    def _generate_and_save_section_content(
        self,
        section: TableOfContentsItem,
        section_content_parts: List[ExtractedContent],
    ) -> None:
        """
        Immediately generate and save content for a section.
        This is called right after research for each section completes.
        """
        print(f"[DEBUG] Generating content for section {section.section}...")

        if section_content_parts:
            # Use enhanced multi-pass synthesis
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
                # Fallback: create content from raw extracted parts
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
                "extracted_content": [
                    {
                        "title": ec.source_title,
                        "url": ec.source_url,
                        "content": ec.processed_content,
                        "raw_content": ec.raw_content,
                        "key_points": ec.key_points,
                        "relevance_score": ec.relevance_score,
                    }
                    for ec in section_content_parts
                ],
            }

            section.content = content
            section.sources = [ec.source_url for ec in section_content_parts]
            print(f"[DEBUG] Section {section.section} saved with {len(content)} chars")

        else:
            # No content extracted - generate placeholder with explanation
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
        """
        Check the logical coherence of the entire report.
        Uses LLM to verify sections flow naturally.
        """
        if not self.session or not self.session.section_contents:
            print("[WARNING] No content to check for coherence")
            return

        print(f"[DEBUG] Checking report coherence...")

        # Prepare content summary for coherence check
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

        coherence_prompt = f"""Review the following report sections and identify any logical inconsistencies or gaps:

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
            ResearchWarnings.get_instance().add(
                ResearchWarnings.HIGH,
                "Researcher",
                f"Cross-section coherence check failed entirely. "
                f"Logical inconsistencies may exist between sections. Error: {e}",
            )

    def _conduct_extended_research(
        self,
        section: TableOfContentsItem,
        initial_results: List[SearchResult],
        section_content_parts: List[ExtractedContent],
    ) -> Dict[str, Any]:
        """
        Conduct extended research by crawling sites from initial search results.

        Args:
            section: Current section being researched
            initial_results: Initial search results with URLs to crawl
            section_content_parts: Existing content parts to add to

        Returns:
            Dictionary with crawl results and suggested queries
        """
        if not self.site_crawler:
            return {"crawl_results": [], "suggested_queries": []}

        self._report_progress(
            f"Extended mode: Crawling sites for {section.section}",
            -1
        )

        # Extract keywords from section
        keywords = extract_keywords_from_topic(
            f"{section.title} {section.description}"
        )

        # Get existing content summary
        existing_content = "\n".join(
            ec.processed_content[:500] for ec in section_content_parts
        )

        # Get seed URLs from initial results
        seed_urls = [r.url for r in initial_results[:5]]

        # Crawl sites
        crawl_results = self.site_crawler.crawl_multiple_sites(
            seed_urls=seed_urls,
            research_topic=f"{section.title}",
            keywords=keywords,
            section_context=f"{section.section}. {section.title}",
            existing_content=existing_content,
            max_sites=self.crawl_max_sites,
        )

        all_suggested_queries = []

        # Process crawl results
        for crawl_result in crawl_results:
            self._report_progress(
                f"Found {crawl_result.pages_relevant} relevant pages at {crawl_result.root_domain}",
                -1
            )

            # Add discovered topics to suggested queries
            all_suggested_queries.extend(crawl_result.suggested_queries)

            # Extract content from relevant crawled pages
            for crawled_page in crawl_result.crawled_pages:
                if crawled_page.relevance_score >= 0.3:
                    # Create ExtractedContent from crawled page
                    extracted = self.content_extractor.extract_relevant_content(
                        raw_content=crawled_page.content,
                        source_url=crawled_page.url,
                        source_title=crawled_page.title,
                        section_context=f"{section.section}. {section.title}",
                        research_query=f"{section.title} (crawled)",
                    )

                    if extracted.relevance_score >= 0.3:
                        section_content_parts.append(extracted)

                        # Add to evidence locker
                        self.evidence_locker.add_evidence(
                            url=crawled_page.url,
                            title=crawled_page.title,
                            content_excerpt=extracted.processed_content[:500],
                            extracted_text=crawled_page.content,
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
        except (json.JSONDecodeError, ValueError) as _sum_err:
            ResearchWarnings.get_instance().add(
                ResearchWarnings.HIGH,
                "Researcher",
                f"Executive summary generation failed (JSON parse error). "
                f"Summary, key_findings, and recommendations are empty placeholders. "
                f"Error: {_sum_err}",
            )
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

    def expand_section_content(
        self,
        section_ids: List[str],
        additional_iterations: int = 2,
        focus_on_gaps: bool = True,
    ) -> Dict[str, Any]:
        """
        Expand content for specific sections by running additional research.

        This is used when content is shorter than the target page/character count.
        Runs additional research iterations focused on expanding the specified sections.

        Args:
            section_ids: List of section IDs to expand
            additional_iterations: Number of additional iterations per section
            focus_on_gaps: Whether to focus on previously identified gaps

        Returns:
            Dictionary with expansion results
        """
        if not self.session or not self.session.research_plan:
            raise ValueError("Research session not initialized")

        self._report_progress(
            f"Expanding content for {len(section_ids)} sections...",
            0
        )

        expansion_results = {
            "sections_expanded": [],
            "characters_added": 0,
            "new_sources": 0,
        }

        toc = self.session.research_plan.table_of_contents
        sections_map = {s.section: s for s in toc.get_flat_sections()}

        for idx, section_id in enumerate(section_ids):
            section = sections_map.get(section_id)
            if not section:
                continue

            progress = ((idx + 1) / len(section_ids)) * 100
            self._report_progress(
                f"Expanding section {section_id}: {section.title}",
                progress * 0.8  # Leave 20% for synthesis
            )

            # Get existing content and gaps
            existing_content = self.session.section_contents.get(section_id, {})
            existing_text = existing_content.get("content", "")
            original_length = len(existing_text)
            gaps = existing_content.get("gaps", [])

            # Collect new content parts
            new_content_parts: List[ExtractedContent] = []

            # Run additional iterations
            for iteration in range(additional_iterations):
                iter_record = ResearchIteration(
                    iteration_number=len(self.session.iterations) + 1,
                    section=section_id,
                )

                # Generate queries focusing on gaps or expanding content
                if focus_on_gaps and gaps:
                    queries = self.query_generator.generate_follow_up_queries(
                        section, existing_text, gaps,
                        research_topic=self.session.query,
                    )
                    iter_record.gaps_identified = gaps
                else:
                    # Generate deeper queries for the section
                    queries = self._generate_expansion_queries(
                        section, existing_text
                    )

                iter_record.queries_executed = queries

                # Execute searches
                for query in queries[:self.max_queries_per_iteration]:
                    try:
                        # Use multilingual search if enabled
                        if self.multilingual_searcher:
                            ml_results, _ = self.multilingual_searcher.search_parallel(query)
                            results = [
                                SearchResult(
                                    title=mr.title,
                                    url=mr.url,
                                    snippet=mr.snippet,
                                    metadata={
                                        "source_language": mr.source_language,
                                        "is_translated": mr.is_translated,
                                    },
                                )
                                for mr in ml_results
                            ]
                        else:
                            results = self.search.search(query)
                        iter_record.sources_found += len(results)

                        for result in results[:self.max_pages_per_query]:
                            try:
                                # Skip if we already have this source
                                existing_sources = existing_content.get("sources", [])
                                if result.url in existing_sources:
                                    continue

                                # Apply content filter
                                if self.content_filter:
                                    url_filter_result = self.content_filter.filter_url(result.url)
                                    if not url_filter_result.should_include:
                                        continue

                                page = self.search.get_page_content(result.url)

                                # Apply content filter to page content
                                if self.content_filter:
                                    content_filter_result = self.content_filter.filter_content(
                                        url=result.url,
                                        title=result.title,
                                        content=page.text_content,
                                    )
                                    if not content_filter_result.should_include:
                                        continue

                                # Apply max_content_length truncation
                                raw_content = page.text_content
                                if len(raw_content) > self.max_content_length:
                                    raw_content = raw_content[:self.max_content_length]

                                extracted = self.content_extractor.extract_relevant_content(
                                    raw_content=raw_content,
                                    source_url=result.url,
                                    source_title=result.title,
                                    section_context=f"{section.section}. {section.title}",
                                    research_query=query,
                                )

                                if extracted.relevance_score >= 0.3:
                                    new_content_parts.append(extracted)
                                    iter_record.content_extracted += 1

                                    self.evidence_locker.add_evidence(
                                        url=result.url,
                                        title=result.title,
                                        content_excerpt=extracted.processed_content[:500],
                                        extracted_text=raw_content,
                                        evidence_type=EvidenceType.WEB_PAGE,
                                        search_query=query,
                                        section_reference=section_id,
                                        relevance_score=extracted.relevance_score,
                                    )

                                    expansion_results["new_sources"] += 1

                            except Exception as e:
                                print(f"Content extraction error: {e}")
                                continue

                        time.sleep(0.5)

                    except Exception as e:
                        print(f"Search error: {e}")
                        continue

                iter_record.completed_at = datetime.now().isoformat()
                self.session.iterations.append(iter_record)

            # Merge new content with existing
            if new_content_parts:
                merged = self._merge_expanded_content(
                    section=section,
                    existing_content=existing_content,
                    new_parts=new_content_parts,
                )

                self.session.section_contents[section_id] = merged
                characters_added = len(merged.get("content", "")) - original_length
                expansion_results["characters_added"] += max(0, characters_added)
                expansion_results["sections_expanded"].append(section_id)

        # Re-synthesize findings with new content
        self._report_progress("Re-synthesizing with expanded content...", 90)
        self._synthesize_findings()

        self._report_progress("Content expansion complete", 100)

        return expansion_results

    def _generate_expansion_queries(
        self,
        section: TableOfContentsItem,
        existing_content: str,
    ) -> List[str]:
        """
        Generate queries to expand section content with more depth.

        Args:
            section: Section to expand
            existing_content: Existing content

        Returns:
            List of expansion queries
        """
        research_topic = self.session.query

        topic_anchor = ""
        if research_topic:
            topic_anchor = f"""
Original Research Topic: {research_topic}
IMPORTANT: All queries must stay within the context of this research topic. Do not generate generic queries."""

        prompt = f"""Based on this section and its existing content, generate search queries to find additional detailed information.
{topic_anchor}
Section: {section.section}. {section.title}
Description: {section.description}

Existing Content (summary):
{existing_content[:1000]}...

Generate 3 search queries that would find:
1. More detailed data or statistics related to this topic
2. Expert opinions or analysis on this topic
3. Case studies or specific examples

Return as JSON array: ["query1", "query2", "query3"]"""

        try:
            response = self.llm.generate(prompt)
            content = response.content
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                queries = json.loads(content[start:end])
                return queries[:3]
        except Exception:
            pass

        # Fallback queries
        return [
            f"{section.title} detailed analysis",
            f"{section.title} statistics data",
            f"{section.title} examples cases",
        ]

    def _merge_expanded_content(
        self,
        section: TableOfContentsItem,
        existing_content: Dict[str, Any],
        new_parts: List[ExtractedContent],
    ) -> Dict[str, Any]:
        """
        Merge new content with existing section content.

        Args:
            section: Section being expanded
            existing_content: Existing section content
            new_parts: New extracted content parts

        Returns:
            Merged content dictionary
        """
        # Synthesize new content
        new_synthesized = self.content_extractor.synthesize_section_content(
            section_title=section.title,
            section_description=section.description,
            extracted_contents=new_parts,
            requirements=self.session.requirements if self.session else "",
        )

        existing_text = existing_content.get("content", "")
        new_text = new_synthesized.get("content", "")

        # Use LLM to merge content coherently
        merge_prompt = f"""Merge the existing content with new additional content for this section.
The result should be a coherent, well-structured section that includes all information without repetition.

Section: {section.title}

EXISTING CONTENT:
{existing_text}

NEW ADDITIONAL CONTENT:
{new_text}

Create a merged version that:
1. Integrates all information naturally
2. Avoids repetition
3. Maintains good flow and structure
4. Preserves all important facts and citations

Return the merged content as a single text block (no JSON, just the merged text):"""

        try:
            response = self.llm.generate(merge_prompt)
            merged_text = response.content.strip()
        except Exception:
            # Fallback: just append
            merged_text = existing_text + "\n\n" + new_text

        # Merge sources
        existing_sources = existing_content.get("sources", [])
        new_sources = [ec.source_url for ec in new_parts]
        all_sources = list(set(existing_sources + new_sources))

        # Merge images
        existing_images = existing_content.get("images", [])
        new_images = [img for ec in new_parts for img in ec.images]
        all_images = (existing_images + new_images)[:5]

        # Update gaps (remove gaps that may have been addressed)
        remaining_gaps = new_synthesized.get("information_gaps", [])

        return {
            "title": section.title,
            "content": merged_text,
            "summary": new_synthesized.get("summary", existing_content.get("summary", "")),
            "confidence": new_synthesized.get("confidence_level", "medium"),
            "sources": all_sources,
            "images": all_images,
            "gaps": remaining_gaps,
            "expanded": True,
        }
