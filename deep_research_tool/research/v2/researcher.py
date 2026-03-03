"""
Researcher V2 - Enhanced research orchestrator.

Extends the V1 Researcher with:
- Think Tool (strategic reflection during research iterations)
- Parallel section processing (asyncio-based concurrency)

Usage:
    from deep_research_tool.research.v2 import ResearcherV2

    researcher = ResearcherV2(
        llm_client=llm,
        search_client=search,
        enable_think_tool=True,
        enable_parallel=True,
        max_concurrent_sections=3,
    )

    session = researcher.conduct_research("AI market analysis 2025")
"""

import asyncio
import time
import json
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path

from ..researcher import (
    Researcher,
    ResearchSession,
    ResearchState,
    ResearchIteration,
)
from ..query_generator import TableOfContentsItem
from ..content_extractor import ExtractedContent
from ...evidence.locker import EvidenceLocker, EvidenceType
from ...config import CrawlMode, MultilingualSearchConfig
from ...evidence.content_filter import ContentFilter
from ...utils.helpers import ResearchWarnings

from .reflector import ResearchReflector, ReflectionResult
from .async_orchestrator import AsyncResearchOrchestrator


class ResearcherV2(Researcher):
    """
    Enhanced research orchestrator (V2).

    Inherits all V1 functionality and adds:
    - Think Tool: strategic reflection after each research iteration
    - Parallel sections: concurrent processing of independent sections
    """

    def __init__(
        self,
        llm_client,
        search_client,
        # V2-specific parameters
        enable_think_tool: bool = True,
        think_tool_start_iteration: int = 2,
        enable_parallel: bool = False,
        max_concurrent_sections: int = 3,
        # V1 parameters (passed through to parent)
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
        # Initialize V1 parent
        super().__init__(
            llm_client=llm_client,
            search_client=search_client,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            max_queries_per_iteration=max_queries_per_iteration,
            max_pages_per_query=max_pages_per_query,
            language=language,
            output_dir=output_dir,
            progress_callback=progress_callback,
            extended_mode=extended_mode,
            crawl_max_pages=crawl_max_pages,
            crawl_max_depth=crawl_max_depth,
            crawl_max_sites=crawl_max_sites,
            crawl_relevance_threshold=crawl_relevance_threshold,
            use_enhanced_synthesis=use_enhanced_synthesis,
            content_filter=content_filter,
            filter_mode=filter_mode,
            crawl_mode=crawl_mode,
            fast_crawl_workers=fast_crawl_workers,
            fast_crawl_batch_size=fast_crawl_batch_size,
            multilingual_config=multilingual_config,
            max_content_length=max_content_length,
            target_pages=target_pages,
            target_characters=target_characters,
        )

        # V2 components
        self.enable_think_tool = enable_think_tool
        self.think_tool_start_iteration = think_tool_start_iteration
        self.enable_parallel = enable_parallel
        self.max_concurrent_sections = max_concurrent_sections

        # Initialize Think Tool
        self.reflector: Optional[ResearchReflector] = None
        if enable_think_tool:
            self.reflector = ResearchReflector(llm_client, language)
            print(f"[ResearcherV2] Think Tool enabled (starts at iteration {think_tool_start_iteration})")

        # Initialize parallel orchestrator
        self.orchestrator: Optional[AsyncResearchOrchestrator] = None
        if enable_parallel:
            self.orchestrator = AsyncResearchOrchestrator(
                max_concurrent_sections=max_concurrent_sections,
                progress_callback=progress_callback,
            )
            print(f"[ResearcherV2] Parallel mode enabled (max {max_concurrent_sections} concurrent)")

        # Track reflection results for reporting
        self.reflection_history: Dict[str, List[Dict]] = {}

    def _conduct_research_loop(self) -> None:
        """
        Execute the main research loop (V2 override).

        Uses parallel processing when enabled, otherwise falls back
        to sequential processing with Think Tool integration.
        """
        if not self.session or not self.session.research_plan:
            raise ValueError("Research session not properly initialized")

        toc = self.session.research_plan.table_of_contents
        sections = toc.get_flat_sections()
        total_sections = len(sections)

        print(f"[ResearcherV2] Research loop starting with {total_sections} sections")
        for s in sections:
            print(f"  - {s.section}: {s.title}")

        if total_sections == 0:
            print("[ERROR] No sections found in table of contents!")
            return

        # Calculate per-section character target
        if total_sections > 0 and (self._target_pages or self._target_characters):
            if self._target_characters:
                total_target = self._target_characters
            else:
                chars_per_page = 1500 if self.language == "ja" else 2500
                total_target = self._target_pages * chars_per_page
            target_per_section = total_target // total_sections
            self.content_extractor.target_chars_per_section = target_per_section

        available_queries = list(self.session.research_plan.search_queries)
        print(f"[ResearcherV2] Initial search queries: {len(available_queries)}")
        print(f"[ResearcherV2] Using crawl mode: {self.crawl_mode.value}")

        # Choose processing mode
        if self.enable_parallel and self.orchestrator:
            self._conduct_parallel_research(sections, available_queries, total_sections)
        else:
            self._conduct_sequential_research(sections, available_queries, total_sections)

        # Overall reflection after all sections
        if self.reflector:
            print("[ResearcherV2] Running overall reflection...")
            overall = self.reflector.reflect_on_overall(
                self.session.section_contents,
                self.session.query,
            )
            self.session.section_contents["_v2_overall_reflection"] = overall.to_dict()
            print(f"[ResearcherV2] Overall quality: {overall.overall_quality:.2f}")
            if overall.cross_section_gaps:
                print(f"[ResearcherV2] Cross-section gaps: {overall.cross_section_gaps}")

        # Coherence check (inherited from V1)
        self._check_report_coherence()

    def _conduct_parallel_research(
        self,
        sections: List[TableOfContentsItem],
        available_queries: List[str],
        total_sections: int,
    ) -> None:
        """Run research with parallel section processing."""
        print("[ResearcherV2] Starting parallel research...")

        # Analyze dependencies
        groups = self.orchestrator.analyze_dependencies(sections)
        print(f"[ResearcherV2] {len(groups)} dependency groups identified")

        # Define the section processor (wraps V2 sequential processing)
        def process_section(section, queries, idx, total):
            section.status = "in_progress"
            if self.fast_crawler and self.crawl_mode in (CrawlMode.FAST_BATCH, CrawlMode.FAST_PARALLEL):
                self._process_section_with_fast_crawler(section, queries, idx, total)
            else:
                self._process_section_v2(section, queries, idx, total)
            section.status = "completed"

        # Run async event loop
        result = asyncio.run(
            self.orchestrator.process_sections_parallel(
                section_groups=groups,
                process_func=process_section,
                available_queries=available_queries,
                total_sections=total_sections,
            )
        )

        print(f"[ResearcherV2] Parallel research completed in {result.total_time:.1f}s "
              f"({result.parallel_groups} groups)")
        if result.errors:
            for sec_id, err in result.errors.items():
                print(f"[ResearcherV2] Error in {sec_id}: {err}")

    def _conduct_sequential_research(
        self,
        sections: List[TableOfContentsItem],
        available_queries: List[str],
        total_sections: int,
    ) -> None:
        """Run research sequentially with Think Tool integration."""
        for section_idx, section in enumerate(sections):
            section_progress_base = 10 + (section_idx / total_sections) * 70
            self._report_progress(
                f"Researching: {section.section}. {section.title}",
                section_progress_base,
            )
            section.status = "in_progress"

            if self.fast_crawler and self.crawl_mode in (CrawlMode.FAST_BATCH, CrawlMode.FAST_PARALLEL):
                self._process_section_with_fast_crawler(
                    section, available_queries, section_idx, total_sections,
                )
            else:
                self._process_section_v2(
                    section, available_queries, section_idx, total_sections,
                )

            # Remove used queries
            if available_queries:
                available_queries = available_queries[self.max_queries_per_iteration:]

            section.status = "completed"

    def _process_section_v2(
        self,
        section: TableOfContentsItem,
        available_queries: List[str],
        section_idx: int,
        total_sections: int,
    ) -> None:
        """
        Process a section with Think Tool integration.

        This overrides the V1 _process_section_with_immediate_generation
        to add strategic reflection between iterations.
        """
        section_content_parts: List[ExtractedContent] = []
        self.reflection_history[section.section] = []

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
                print(f"[WARNING] No queries for section {section.section}")
                break

            iter_record.queries_executed = queries
            queries_to_run = queries[:self.max_queries_per_iteration]

            # Execute search and extract content (reuse V1 search logic)
            self._execute_search_iteration(
                section, queries_to_run, section_content_parts, iter_record,
            )

            iter_record.completed_at = datetime.now().isoformat()
            self.session.iterations.append(iter_record)

            # Think Tool: Strategic reflection (from iteration 2+)
            if (self.enable_think_tool and self.reflector
                    and iteration >= self.think_tool_start_iteration):

                reflection = self.reflector.reflect_on_section(
                    section=section,
                    collected_evidence=section_content_parts,
                    iteration=iteration,
                    max_iterations=self.max_iterations,
                    research_topic=self.session.query,
                    previous_sections_summary=self._get_completed_summaries(),
                )

                # Record reflection
                self.reflection_history[section.section].append(reflection.to_dict())
                print(f"[ThinkTool] Section {section.section} iter {iteration}: "
                      f"coverage={reflection.coverage_score:.2f} "
                      f"quality={reflection.quality_score:.2f} "
                      f"confidence={reflection.confidence:.2f}")

                # Act on reflection
                if reflection.stop_research:
                    print(f"[ThinkTool] Stopping research for {section.section}: "
                          f"{reflection.stop_reason or 'sufficient coverage'}")
                    break

                if reflection.should_pivot and reflection.recommended_queries:
                    print(f"[ThinkTool] Pivoting research for {section.section}: "
                          f"{reflection.pivot_reason}")
                    # Override next iteration's queries with pivot queries
                    available_queries = reflection.recommended_queries + available_queries
                    continue

            # V1 stop condition: minimum iterations met and enough content
            if iteration >= self.min_iterations and len(section_content_parts) >= 2:
                break

        # Generate and save section content (inherited from V1)
        print(f"[ResearcherV2] Section {section.section} complete. "
              f"Parts: {len(section_content_parts)}")
        self._generate_and_save_section_content(section, section_content_parts)

        # Store reflection history in section metadata
        if self.reflection_history.get(section.section):
            if section.section in self.session.section_contents:
                self.session.section_contents[section.section]["_reflections"] = (
                    self.reflection_history[section.section]
                )

    def _execute_search_iteration(
        self,
        section: TableOfContentsItem,
        queries: List[str],
        section_content_parts: List[ExtractedContent],
        iter_record: ResearchIteration,
    ) -> None:
        """
        Execute a single search iteration (extracted from V1 for reuse).

        This contains the search + extract + filter logic from the V1
        _process_section_with_immediate_generation inner loop.
        """
        import time as time_module
        from ...search.base import SearchResult

        print(f"\n[Search] Section {section.section} - {len(queries)} queries")

        for qi, query in enumerate(queries, 1):
            print(f"[Search] ({qi}/{len(queries)}) Query: {query}")
            try:
                # Use multilingual search if enabled
                if self.multilingual_searcher:
                    ml_results, ml_stats = self.multilingual_searcher.search_parallel(query)
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
                else:
                    results = self.search.search(query)

                iter_record.sources_found += len(results)

                for result in results[:self.max_pages_per_query]:
                    # URL filter
                    if self.content_filter:
                        url_filter_result = self.content_filter.filter_url(result.url)
                        if not url_filter_result.should_include:
                            continue

                    try:
                        page = self.search.get_page_content(result.url)

                        # Content filter
                        if self.content_filter:
                            content_filter_result = self.content_filter.filter_content(
                                url=result.url,
                                title=result.title,
                                content=page.text_content,
                            )
                            if not content_filter_result.should_include:
                                continue

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

                        # Capture images
                        page_images = getattr(page, 'images', []) or []
                        if page_images and not extracted.images:
                            extracted.images = [
                                {"src": img.get("src", ""), "alt": img.get("alt", ""),
                                 "title": img.get("title", ""), "page_title": result.title}
                                for img in page_images[:5]
                                if img.get("src", "")
                            ]

                        if extracted.relevance_score >= 0.2:
                            section_content_parts.append(extracted)
                            iter_record.content_extracted += 1

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
                        elif extracted.processed_content and len(extracted.processed_content) > 100:
                            section_content_parts.append(extracted)

                    except Exception as e:
                        print(f"[ERROR] Content extraction error for {result.url}: {e}")
                        ResearchWarnings.get_instance().add(
                            ResearchWarnings.HIGH,
                            "ResearcherV2",
                            f"Content extraction failed for {result.url[:80]}. Error: {e}",
                        )
                        continue

                time_module.sleep(0.3)

            except Exception as e:
                print(f"[ERROR] Search error for query '{query}': {e}")
                ResearchWarnings.get_instance().add(
                    ResearchWarnings.HIGH,
                    "ResearcherV2",
                    f"Search query failed: '{query[:60]}'. Error: {e}",
                )
                continue

    def _get_completed_summaries(self) -> Dict[str, str]:
        """Get summaries of already-completed sections for context."""
        summaries = {}
        if not self.session:
            return summaries

        for section_id, content in self.session.section_contents.items():
            if section_id.startswith("_"):
                continue
            title = content.get("title", "")
            text = content.get("content", "")
            summaries[section_id] = f"{title}: {text[:200]}"

        return summaries

    def get_reflection_history(self) -> Dict[str, List[Dict]]:
        """Get the full Think Tool reflection history."""
        return self.reflection_history
