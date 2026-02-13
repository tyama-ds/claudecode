"""
Main module for Deep Research Tool.

This module provides the main interface for conducting automated research.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from .config import Config, LLMProvider, SearchMethod, ReportFormat
from .api import get_client
from .api.base import get_token_stats, reset_token_stats
from .search import get_search_client
from .research.researcher import Researcher, ResearchSession
from .research.manual_researcher import ManualResearcher, ManualTableOfContents
from .evidence.locker import EvidenceLocker
from .verification.verifier import Verifier
from .report.generator import ReportGenerator
from .report.length_controller import ContentLengthController, LengthTarget
from .utils.document_reader import DocumentReader, auto_detect_additional_documents
from .thinking import DeepThinkProcessor, DeepThinkConfig as ThinkingConfig
from .thinking.reasoning_chain import ConsistencyMode


class DeepResearchTool:
    """
    Main interface for the Deep Research Tool.

    Orchestrates the complete research workflow including:
    - Query analysis and research planning
    - Web search and content extraction
    - Information synthesis
    - Verification for hallucinations
    - Report generation

    Example usage:

        # Basic usage with environment variables for API keys
        from deep_research_tool import DeepResearchTool, Config

        config = Config()
        tool = DeepResearchTool(config)
        result = tool.run("AI trends in healthcare 2024")

        # With explicit configuration
        from deep_research_tool import create_config

        config = create_config(
            provider="anthropic",
            anthropic_api_key="your-key",
            model="claude-3-5-sonnet-20241022",
            search_method="selenium",
            research_iterations=5,
            output_format="docx",
        )

        tool = DeepResearchTool(config)
        result = tool.run(
            query="Renewable energy market analysis",
            requirements="Focus on solar and wind energy in Asia Pacific region",
            additional_documents=["reference.pdf", "previous_report.docx"],
        )

    For Jupyter Notebook usage, see the example notebooks in the examples/ directory.
    """

    def __init__(self, config: Config = None):
        """
        Initialize DeepResearchTool.

        Args:
            config: Configuration object. If not provided, uses defaults
                   with environment variables for API keys.
        """
        self.config = config or Config()
        self._validate_config()

        # Initialize components
        self.llm_client = self._create_llm_client()
        self.search_client = self._create_search_client()
        self.researcher = None
        self.verifier = None
        self.report_generator = None
        self.deep_think_processor = None

        # Initialize DeepThink if enabled
        if self.config.deep_think.enabled:
            self._init_deep_think()

    def _init_deep_think(self) -> None:
        """Initialize DeepThink processor."""
        # Map config consistency_mode string to enum
        mode_map = {
            "warn": ConsistencyMode.WARN,
            "revise": ConsistencyMode.REVISE,
            "strict": ConsistencyMode.STRICT,
        }
        consistency_mode = mode_map.get(
            self.config.deep_think.consistency_mode,
            ConsistencyMode.WARN
        )

        thinking_config = ThinkingConfig(
            enabled=True,
            level=self.config.deep_think.level,
            reasoning_iterations=self.config.deep_think.reasoning_iterations,
            consistency_threshold=self.config.deep_think.consistency_threshold,
            consistency_mode=consistency_mode,
            fidelity_threshold=self.config.deep_think.fidelity_threshold,
            _expansion_tolerance=self.config.deep_think._expansion_tolerance,
            _deviation_weights=self.config.deep_think._deviation_weights,
        )

        self.deep_think_processor = DeepThinkProcessor(
            llm_client=self.llm_client,
            config=thinking_config,
            language=self.config.research.language,
        )

    def _validate_config(self) -> None:
        """Validate configuration."""
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")

    def _create_llm_client(self):
        """Create LLM client based on configuration."""
        kwargs = {
            "provider": self.config.api.provider.value,
            "api_key": self.config.api.get_active_api_key(),
            "model": self.config.api.get_active_model(),
            "http_proxy": self.config.proxy.http_proxy,
            "https_proxy": self.config.proxy.https_proxy,
            "verify_ssl": self.config.proxy.verify_ssl,
        }

        # Add local LLM specific settings
        if self.config.api.provider == LLMProvider.LOCAL:
            kwargs["base_url"] = self.config.api.local_base_url
            kwargs["backend"] = self.config.api.local_backend.value

        return get_client(**kwargs)

    def _create_search_client(self):
        """Create search client based on configuration."""
        kwargs = {
            "max_results": self.config.search.max_results,
            "timeout": self.config.search.page_load_timeout,
            "extract_images": self.config.search.extract_images,
            "max_images": self.config.search.max_images_per_page,
        }

        # Add proxy settings
        if self.config.proxy.is_configured():
            kwargs["proxies"] = self.config.proxy.get_proxies_dict()
            kwargs["verify_ssl"] = self.config.proxy.verify_ssl

        if self.config.search.method == SearchMethod.SELENIUM:
            kwargs["headless"] = self.config.search.headless
            kwargs["browser"] = self.config.search.browser

        return get_search_client(
            method=self.config.search.method.value,
            **kwargs
        )

    def run(
        self,
        query: str,
        requirements: str = "",
        additional_context: str = "",
        additional_documents: List[str] = None,
        progress_callback: Callable[[str, float], None] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete research workflow.

        Args:
            query: The research query/topic
            requirements: Specific research requirements
            additional_context: Additional context information
            additional_documents: List of paths to additional documents
            progress_callback: Callback function for progress updates
                              (message: str, percentage: float)

        Returns:
            Dictionary containing:
            - session_id: Unique session identifier
            - report_path: Path to generated report
            - evidence_json: Path to evidence JSON file
            - evidence_csv: Path to evidence CSV file
            - verification_html: Path to verification report (if enabled)
            - session: ResearchSession object
            - evidence_locker: EvidenceLocker object
        """
        # Process additional documents
        doc_contents = []
        if additional_documents:
            reader = DocumentReader()
            for doc_path in additional_documents:
                doc = reader.read_document(Path(doc_path))
                if not doc.error:
                    doc_contents.append({
                        "path": doc.filepath,
                        "title": doc.title or doc.filename,
                        "content": doc.content,
                    })

        # Initialize researcher
        self.researcher = Researcher(
            llm_client=self.llm_client,
            search_client=self.search_client,
            min_iterations=self.config.research.min_iterations,
            max_iterations=self.config.research.max_iterations,
            max_queries_per_iteration=self.config.research.max_queries_per_iteration,
            max_pages_per_query=self.config.research.max_pages_per_query,
            language=self.config.research.language,
            output_dir=self.config.report.output_dir,
            progress_callback=progress_callback,
            extended_mode=self.config.research.extended_mode,
            crawl_max_pages=self.config.research.crawl_max_pages,
            crawl_max_depth=self.config.research.crawl_max_depth,
            crawl_max_sites=self.config.research.crawl_max_sites,
            crawl_relevance_threshold=self.config.research.crawl_relevance_threshold,
            use_enhanced_synthesis=self.config.research.use_enhanced_synthesis,
        )

        # Conduct research
        session = self.researcher.conduct_research(
            query=query,
            requirements=requirements,
            additional_context=additional_context,
            additional_documents=doc_contents,
        )

        evidence_locker = self.researcher.get_evidence_locker()

        # Check if content expansion is needed (target specified and content too short)
        session = self._expand_if_needed(
            session=session,
            progress_callback=progress_callback,
        )

        # Apply DeepThink processing if enabled
        deep_think_results = None
        if self.config.deep_think.enabled and self.deep_think_processor:
            session, deep_think_results = self._apply_deep_think(
                session=session,
                evidence_locker=evidence_locker,
                progress_callback=progress_callback,
            )

        # Export evidence
        evidence_json = evidence_locker.export_to_json()
        evidence_csv = evidence_locker.export_to_csv()

        # Verification (if enabled)
        verification_html = None
        verification_result = None
        if self.config.enable_verification:
            if progress_callback:
                progress_callback("Running verification...", 90)

            self.verifier = Verifier(
                llm_client=self.llm_client,
                language=self.config.research.language,
            )

            # Get full content for verification
            content_for_verification = self._get_content_for_verification(session)

            verification_result = self.verifier.verify_content(
                content=content_for_verification,
                evidence_locker=evidence_locker,
                document_title=session.research_plan.title if session.research_plan else query,
                strictness=self.config.verification_strictness,
            )

            verification_html = self.config.report.output_dir / f"verification_{session.session_id}.html"
            self.verifier.generate_verification_report_html(
                verification_result,
                verification_html,
            )

        # Generate report
        if progress_callback:
            progress_callback("Generating report...", 95)

        self.report_generator = ReportGenerator(
            output_dir=self.config.report.output_dir / "reports",
            include_toc=self.config.report.include_toc,
            include_citations=self.config.report.include_citations,
            include_images=self.config.report.include_images,
            language=self.config.research.language,
        )

        report_path = self.report_generator.generate_report(
            session=session,
            evidence_locker=evidence_locker,
            format=self.config.report.format,
            verification_result=verification_result,
            target_pages=self.config.report.target_pages,
            target_characters=self.config.report.target_characters,
        )

        # Clean up search client if selenium
        if hasattr(self.search_client, 'close'):
            self.search_client.close()

        # Get token usage statistics
        token_stats = get_token_stats()

        return {
            "session_id": session.session_id,
            "report_path": str(report_path),
            "evidence_json": str(evidence_json),
            "evidence_csv": str(evidence_csv),
            "verification_html": str(verification_html) if verification_html else None,
            "session": session,
            "evidence_locker": evidence_locker,
            "verification_result": verification_result,
            "deep_think_results": deep_think_results,
            "token_usage": token_stats.to_dict(),
        }

    def _get_content_for_verification(self, session: ResearchSession) -> str:
        """Extract content from session for verification."""
        content_parts = []

        for section_num, section_data in session.section_contents.items():
            if section_num.startswith("_"):
                continue
            content_parts.append(section_data.get("content", ""))

        return "\n\n".join(content_parts)

    def _apply_deep_think(
        self,
        session: ResearchSession,
        evidence_locker: EvidenceLocker,
        progress_callback: Callable[[str, float], None] = None,
    ) -> tuple:
        """
        Apply DeepThink processing to session content.

        Processes each section with DeepThink for enhanced reasoning
        and consistency checking.

        Args:
            session: Research session with content
            evidence_locker: Evidence locker with source texts
            progress_callback: Progress callback function

        Returns:
            Tuple of (updated session, dict of DeepThink results per section)
        """
        if progress_callback:
            progress_callback("DeepThink: Processing sections...", 85)

        # Get source texts from evidence locker
        source_texts = {}
        for evidence in evidence_locker.get_all_evidence():
            # Use evidence_id as the key and content_excerpt as the source text
            if evidence.evidence_id and evidence.content_excerpt:
                source_texts[evidence.evidence_id] = evidence.content_excerpt

        # Process each section
        deep_think_results = {}
        section_count = len([k for k in session.section_contents.keys() if not k.startswith("_")])

        # Handle empty session gracefully
        if section_count == 0:
            if progress_callback:
                progress_callback("DeepThink: No sections to process", 90)
            return session, deep_think_results

        processed = 0

        for section_num, section_data in session.section_contents.items():
            if section_num.startswith("_"):
                continue

            content = section_data.get("content", "")
            if not content:
                continue

            # Progress update
            processed += 1
            if progress_callback:
                progress = 85 + (processed / max(section_count, 1) * 5)
                progress_callback(f"DeepThink: Processing section {section_num}...", progress)

            # Apply DeepThink
            result = self.deep_think_processor.process(
                content=content,
                source_texts=source_texts,
            )

            # Update section content with processed content
            session.section_contents[section_num]["content"] = result.processed_content

            # Store result metrics
            deep_think_results[section_num] = {
                "is_valid": result.is_valid,
                "confidence": result.overall_confidence,
                "metrics": result.metrics_summary,
                "consistency": result.consistency_result.to_dict() if result.consistency_result else None,
            }

            # Add DeepThink info to section data
            session.section_contents[section_num]["deep_think"] = deep_think_results[section_num]

        if progress_callback:
            avg_confidence = sum(r["confidence"] for r in deep_think_results.values()) / len(deep_think_results) if deep_think_results else 0
            progress_callback(f"DeepThink: Complete (avg confidence: {avg_confidence:.2f})", 90)

        return session, deep_think_results

    def _expand_if_needed(
        self,
        session: ResearchSession,
        progress_callback: Callable[[str, float], None] = None,
    ) -> ResearchSession:
        """
        Check if content needs expansion and expand if necessary.

        This is called when target pages/characters are specified and
        the generated content is shorter than the target.

        Args:
            session: Research session with content
            progress_callback: Progress callback function

        Returns:
            Session with potentially expanded content
        """
        # Check if target is specified
        target_pages = self.config.report.target_pages
        target_characters = self.config.report.target_characters

        if target_pages is None and target_characters is None:
            return session

        # Create length controller
        length_target = LengthTarget(
            target_pages=target_pages,
            target_characters=target_characters,
        )

        controller = ContentLengthController(
            target=length_target,
            format_type=self.config.report.format.value,
            language=self.config.research.language,
        )

        # Check if expansion is needed
        expansion_req = controller.get_expansion_requirement(session.section_contents)

        if not expansion_req.needs_expansion:
            return session

        # Report progress
        if progress_callback:
            progress_callback(
                f"Content is {expansion_req.current_characters:,} chars, "
                f"target is {expansion_req.target_characters:,} chars. "
                f"Running additional research...",
                80
            )

        # Calculate how many additional iterations to run
        additional_iterations = controller.estimate_additional_iterations(expansion_req)

        if progress_callback:
            progress_callback(
                f"Expanding {len(expansion_req.sections_to_expand)} sections "
                f"with {additional_iterations} additional iterations each...",
                82
            )

        # Run expansion
        expansion_result = self.researcher.expand_section_content(
            section_ids=expansion_req.sections_to_expand,
            additional_iterations=additional_iterations,
            focus_on_gaps=True,
        )

        if progress_callback:
            progress_callback(
                f"Added {expansion_result['characters_added']:,} characters "
                f"from {expansion_result['new_sources']} new sources",
                85
            )

        # Return the updated session
        return self.researcher.get_session()

    def quick_research(
        self,
        query: str,
        max_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform a quick research without full report generation.

        Useful for quick fact-checking or getting a brief overview.

        Args:
            query: The research query
            max_results: Maximum search results to process

        Returns:
            Dictionary with quick research results
        """
        # Search
        results = self.search_client.search(query, max_results=max_results)

        # Extract content from top results
        extracted = []
        for result in results[:3]:
            try:
                page = self.search_client.get_page_content(result.url)
                extracted.append({
                    "url": result.url,
                    "title": result.title,
                    "snippet": result.snippet,
                    "content": page.text_content[:2000],
                })
            except Exception:
                extracted.append({
                    "url": result.url,
                    "title": result.title,
                    "snippet": result.snippet,
                    "content": "",
                })

        # Quick synthesis
        synthesis_prompt = f"""Based on the following search results for "{query}":

{chr(10).join(f"[{i+1}] {e['title']}: {e['snippet']}" for i, e in enumerate(extracted))}

Provide a brief summary (2-3 paragraphs) of the key findings.
Note any conflicting information or areas requiring more research."""

        response = self.llm_client.generate(synthesis_prompt)

        return {
            "query": query,
            "results_count": len(results),
            "sources": [{"url": e["url"], "title": e["title"]} for e in extracted],
            "summary": response.content,
        }


# Convenience function for programmatic usage
def run_research(
    query: str,
    provider: str = "openai",
    api_key: str = None,
    model: str = None,
    search_method: str = "duckduckgo",
    iterations: int = 3,
    output_format: str = "markdown",
    output_dir: str = "./output",
    requirements: str = "",
    additional_documents: List[str] = None,
    enable_verification: bool = True,
    verbose: bool = False,
    target_pages: int = None,
    target_characters: int = None,
    extended_mode: bool = False,
    crawl_max_pages: int = 10,
    crawl_max_depth: int = 2,
    crawl_max_sites: int = 3,
    http_proxy: str = None,
    https_proxy: str = None,
    verify_ssl: bool = True,
    # DeepThink parameters
    deep_think: bool = False,
    deep_think_level: float = 0.5,
    reasoning_iterations: int = 3,
    consistency_threshold: float = 0.3,
    consistency_mode: str = "warn",
    # Multilingual parameters
    multilingual: bool = False,
    search_languages: List[str] = None,
    results_per_language: int = 10,
    translate_results: bool = True,
    # Enhanced synthesis
    use_enhanced_synthesis: bool = True,
    # Search depth parameters
    max_queries_per_iteration: int = 3,
    max_pages_per_query: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run research with simple parameters.

    Args:
        query: Research query/topic
        provider: LLM provider ('openai' or 'anthropic')
        api_key: API key (optional, uses env var if not provided)
        model: Model name (optional, uses default)
        search_method: Search method ('duckduckgo' or 'selenium')
        iterations: Research iterations per section
        output_format: Report format ('markdown', 'docx', 'pdf', 'html')
        output_dir: Output directory
        requirements: Research requirements
        additional_documents: List of additional document paths
        enable_verification: Enable hallucination verification
        verbose: Verbose output
        target_pages: Target page count for output (approximate)
        target_characters: Target character count for output
        extended_mode: Enable extended mode (deep site crawling)
        crawl_max_pages: Max pages to crawl per site in extended mode
        crawl_max_depth: Max link depth from seed URL
        crawl_max_sites: Max sites to crawl per search
        http_proxy: HTTP proxy URL (e.g., "http://proxy.example.com:8080")
        https_proxy: HTTPS proxy URL
        verify_ssl: Verify SSL certificates (set False for self-signed certs)
        deep_think: Enable DeepThink reasoning enhancement
        deep_think_level: Reasoning depth (0.0=conservative, 1.0=exploratory)
        reasoning_iterations: Number of reasoning iterations
        consistency_threshold: Threshold for consistency check
        consistency_mode: How to handle consistency issues ('warn', 'revise', 'strict')
        multilingual: Enable multilingual search mode
        search_languages: List of language codes to search (e.g., ['ja', 'en', 'zh'])
        results_per_language: Number of results per language
        translate_results: Whether to translate results to output language
        use_enhanced_synthesis: Use multi-pass content generation for better quality
        max_queries_per_iteration: Max queries to execute per research iteration (default: 3)
        max_pages_per_query: Max pages to process per search query (default: 3)
        **kwargs: Additional configuration options

    Returns:
        Dictionary with research results

    Example:
        result = run_research(
            "Climate change impacts on agriculture",
            provider="anthropic",
            iterations=5,
            output_format="docx",
            target_pages=10,  # Target ~10 pages
        )
        print(f"Report: {result['report_path']}")

        # With extended mode for deeper research
        result = run_research(
            "AI trends 2024",
            extended_mode=True,
            crawl_max_pages=15,
            crawl_max_sites=5,
        )

        # With proxy settings
        result = run_research(
            "Market analysis",
            http_proxy="http://proxy.company.com:8080",
            https_proxy="http://proxy.company.com:8080",
        )

        # With DeepThink for enhanced reasoning
        result = run_research(
            "Complex scientific topic",
            deep_think=True,
            deep_think_level=0.7,
            consistency_mode="revise",
        )

        # With multilingual search
        result = run_research(
            "量子コンピュータの現状",
            multilingual=True,
            search_languages=["ja", "en", "zh", "de"],
            results_per_language=10,
        )
    """
    from .config import create_config

    # Determine API key parameter
    api_key_param = {}
    if api_key:
        if provider == "openai":
            api_key_param["openai_api_key"] = api_key
        else:
            api_key_param["anthropic_api_key"] = api_key

    config = create_config(
        provider=provider,
        model=model,
        search_method=search_method,
        research_iterations=iterations,
        output_format=output_format,
        output_dir=output_dir,
        additional_documents=additional_documents,
        enable_verification=enable_verification,
        verbose=verbose,
        target_pages=target_pages,
        target_characters=target_characters,
        extended_mode=extended_mode,
        crawl_max_pages=crawl_max_pages,
        crawl_max_depth=crawl_max_depth,
        crawl_max_sites=crawl_max_sites,
        http_proxy=http_proxy,
        https_proxy=https_proxy,
        verify_ssl=verify_ssl,
        deep_think=deep_think,
        deep_think_level=deep_think_level,
        reasoning_iterations=reasoning_iterations,
        consistency_threshold=consistency_threshold,
        consistency_mode=consistency_mode,
        multilingual=multilingual,
        search_languages=search_languages,
        results_per_language=results_per_language,
        translate_results=translate_results,
        use_enhanced_synthesis=use_enhanced_synthesis,
        max_queries_per_iteration=max_queries_per_iteration,
        max_pages_per_query=max_pages_per_query,
        **api_key_param,
        **kwargs,
    )

    # Reset token stats before running
    reset_token_stats()

    tool = DeepResearchTool(config)

    def progress_callback(message: str, percentage: float):
        if verbose and percentage >= 0:
            print(f"[{percentage:5.1f}%] {message}")

    result = tool.run(
        query=query,
        requirements=requirements,
        progress_callback=progress_callback if verbose else None,
    )

    # Display token usage summary if verbose
    if verbose:
        token_stats = get_token_stats()
        language = config.research.language if hasattr(config.research, 'language') else "en"
        print("\n" + token_stats.get_summary(language))

    return result


def diagnose_session(result: Dict[str, Any]) -> str:
    """
    Diagnose the research session to identify content issues.

    Args:
        result: The result dictionary from run_research()

    Returns:
        Diagnostic report as a string
    """
    lines = ["=" * 50, "Research Session Diagnostic Report", "=" * 50, ""]

    session = result.get("session")
    if not session:
        lines.append("ERROR: No session object in result")
        return "\n".join(lines)

    # Check research plan
    lines.append(f"Session ID: {session.session_id}")
    lines.append(f"State: {session.state}")
    lines.append(f"Query: {session.query}")
    lines.append("")

    if session.research_plan:
        plan = session.research_plan
        lines.append(f"Research Plan: {plan.title}")
        lines.append(f"  - Sections: {len(plan.table_of_contents.items)}")
        lines.append(f"  - Search Queries: {len(plan.search_queries)}")
        lines.append("")

        # List sections
        lines.append("Table of Contents:")
        for item in plan.table_of_contents.items:
            lines.append(f"  [{item.section}] {item.title}")
            for sub in item.subsections:
                lines.append(f"    [{sub.section}] {sub.title}")
        lines.append("")
    else:
        lines.append("WARNING: No research plan created")
        lines.append("")

    # Check section contents
    lines.append("Section Contents:")
    if session.section_contents:
        for section_key, content_data in session.section_contents.items():
            if section_key.startswith("_"):
                lines.append(f"  [{section_key}]: (metadata)")
                continue

            content = content_data.get("content", "")
            content_len = len(content) if content else 0
            sources_count = len(content_data.get("sources", []))
            confidence = content_data.get("confidence", "unknown")

            status = "OK" if content_len > 100 else "SHORT" if content_len > 0 else "EMPTY"
            lines.append(f"  [{section_key}] {content_data.get('title', 'Unknown')}")
            lines.append(f"      Content: {content_len} chars ({status})")
            lines.append(f"      Sources: {sources_count}")
            lines.append(f"      Confidence: {confidence}")

            if content_len > 0 and content_len <= 200:
                lines.append(f"      Preview: {content[:100]}...")
    else:
        lines.append("  WARNING: No section contents found!")

    lines.append("")
    lines.append("=" * 50)

    return "\n".join(lines)


# Manual search mode convenience function
def run_manual_research(
    evidence_file: str,
    topic: str,
    provider: str = "openai",
    api_key: str = None,
    model: str = None,
    output_format: str = "docx",
    output_dir: str = "./output",
    requirements: str = "",
    enable_verification: bool = True,
    verbose: bool = False,
    target_pages: int = None,
    target_characters: int = None,
    # Manual mode specific parameters
    auto_toc: bool = True,
    manual_toc: ManualTableOfContents = None,
    manual_toc_sections: List[str] = None,
    column_mapping: Dict[str, str] = None,
    file_encoding: str = "utf-8",
    # DeepThink parameters
    deep_think: bool = False,
    deep_think_level: float = 0.5,
    reasoning_iterations: int = 3,
    consistency_threshold: float = 0.3,
    consistency_mode: str = "warn",
    # Enhanced synthesis
    use_enhanced_synthesis: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Run research using pre-collected evidence from CSV/XLSX file.

    This is the "manual search mode" that uses evidence you've already
    collected instead of conducting web searches.

    Args:
        evidence_file: Path to CSV or XLSX file containing evidence
        topic: Research topic/title
        provider: LLM provider ('openai' or 'anthropic')
        api_key: API key (optional, uses env var if not provided)
        model: Model name (optional, uses default)
        output_format: Report format ('markdown', 'docx', 'pdf', 'html')
        output_dir: Output directory
        requirements: Research requirements
        enable_verification: Enable hallucination verification
        verbose: Verbose output
        target_pages: Target page count for output
        target_characters: Target character count for output
        auto_toc: Auto-generate table of contents from evidence (default: True)
        manual_toc: Manual table of contents object (for complex structures)
        manual_toc_sections: Simple list of section titles (alternative to manual_toc)
        column_mapping: Custom column name mapping for evidence file
        file_encoding: Encoding for CSV files
        deep_think: Enable DeepThink reasoning enhancement
        deep_think_level: Reasoning depth (0.0-1.0)
        reasoning_iterations: Number of reasoning iterations
        consistency_threshold: Threshold for consistency check
        consistency_mode: How to handle consistency issues
        use_enhanced_synthesis: Use multi-pass content generation
        **kwargs: Additional configuration options

    Returns:
        Dictionary with research results including report_path

    Example:
        # Basic usage with auto-generated TOC
        result = run_manual_research(
            evidence_file="collected_data.csv",
            topic="Market Analysis 2024",
            output_format="docx",
        )

        # With manual section list
        result = run_manual_research(
            evidence_file="research_data.xlsx",
            topic="Technology Trends Report",
            auto_toc=False,
            manual_toc_sections=[
                "Executive Summary",
                "Market Overview",
                "Technology Trends",
                "Competitive Analysis",
                "Recommendations",
            ],
        )

        # With detailed manual TOC
        from deep_research_tool import ManualTableOfContents

        toc = ManualTableOfContents(
            title="Comprehensive Analysis Report",
            sections=[
                {"section": "1", "title": "Introduction", "description": "Background"},
                {"section": "2", "title": "Methodology", "description": "Research approach"},
                {"section": "3", "title": "Findings", "subsections": [
                    {"section": "3.1", "title": "Key Finding 1"},
                    {"section": "3.2", "title": "Key Finding 2"},
                ]},
                {"section": "4", "title": "Conclusion"},
            ]
        )
        result = run_manual_research(
            evidence_file="data.csv",
            topic="Analysis Report",
            auto_toc=False,
            manual_toc=toc,
        )

        # With custom column mapping
        result = run_manual_research(
            evidence_file="custom_format.csv",
            topic="Research Report",
            column_mapping={
                "title": "Source Name",
                "content": "Description",
                "url": "Link",
            },
        )

    Expected CSV/XLSX columns (flexible, case-insensitive):
        - No. / ID: Evidence number/ID
        - Title: Source title
        - Contents / Content: Main content
        - URL: Source URL
        - Author: Author name
        - Date: Publication date
        - Others / Notes: Additional notes
        - Section: Section reference (optional, for pre-categorized evidence)
    """
    from .config import create_config

    # Determine API key parameter
    api_key_param = {}
    if api_key:
        if provider == "openai":
            api_key_param["openai_api_key"] = api_key
        else:
            api_key_param["anthropic_api_key"] = api_key

    # Create config
    config = create_config(
        provider=provider,
        model=model,
        output_format=output_format,
        output_dir=output_dir,
        enable_verification=enable_verification,
        verbose=verbose,
        target_pages=target_pages,
        target_characters=target_characters,
        deep_think=deep_think,
        deep_think_level=deep_think_level,
        reasoning_iterations=reasoning_iterations,
        consistency_threshold=consistency_threshold,
        consistency_mode=consistency_mode,
        use_enhanced_synthesis=use_enhanced_synthesis,
        **api_key_param,
        **kwargs,
    )

    # Reset token stats
    reset_token_stats()

    # Create LLM client
    llm_client = get_client(
        provider=config.api.provider.value,
        api_key=config.api.get_active_api_key(),
        model=config.api.get_active_model(),
    )

    # Create manual researcher
    researcher = ManualResearcher(
        llm_client=llm_client,
        language=config.research.language,
        output_dir=config.report.output_dir,
        progress_callback=lambda msg, pct: print(f"[{pct:5.1f}%] {msg}") if verbose and pct >= 0 else None,
        use_enhanced_synthesis=use_enhanced_synthesis,
    )

    # Load evidence from file
    if verbose:
        print(f"Loading evidence from: {evidence_file}")

    researcher.load_evidence_from_file(
        file_path=evidence_file,
        column_mapping=column_mapping,
        encoding=file_encoding,
    )

    evidence_count = len(researcher.get_evidence_locker().get_all_evidence())
    if verbose:
        print(f"Loaded {evidence_count} evidence items")

    # Handle manual TOC options
    toc_to_use = None
    if not auto_toc:
        if manual_toc:
            toc_to_use = manual_toc
        elif manual_toc_sections:
            toc_to_use = ManualTableOfContents.from_list(topic, manual_toc_sections)
        else:
            raise ValueError(
                "When auto_toc=False, you must provide either "
                "manual_toc or manual_toc_sections"
            )

    # Conduct research
    session = researcher.conduct_research(
        topic=topic,
        requirements=requirements,
        auto_toc=auto_toc,
        manual_toc=toc_to_use,
    )

    evidence_locker = researcher.get_evidence_locker()

    # Apply DeepThink if enabled
    deep_think_results = None
    if config.deep_think.enabled:
        from .thinking import DeepThinkProcessor, DeepThinkConfig as ThinkingConfig
        from .thinking.reasoning_chain import ConsistencyMode

        mode_map = {
            "warn": ConsistencyMode.WARN,
            "revise": ConsistencyMode.REVISE,
            "strict": ConsistencyMode.STRICT,
        }

        thinking_config = ThinkingConfig(
            enabled=True,
            level=config.deep_think.level,
            reasoning_iterations=config.deep_think.reasoning_iterations,
            consistency_threshold=config.deep_think.consistency_threshold,
            consistency_mode=mode_map.get(config.deep_think.consistency_mode, ConsistencyMode.WARN),
            fidelity_threshold=config.deep_think.fidelity_threshold,
        )

        processor = DeepThinkProcessor(
            llm_client=llm_client,
            config=thinking_config,
            language=config.research.language,
        )

        # Get source texts
        source_texts = {
            e.evidence_id: e.content_excerpt
            for e in evidence_locker.get_all_evidence()
            if e.evidence_id and e.content_excerpt
        }

        # Process sections
        deep_think_results = {}
        for section_num, section_data in session.section_contents.items():
            if section_num.startswith("_"):
                continue

            content = section_data.get("content", "")
            if not content:
                continue

            result = processor.process(content=content, source_texts=source_texts)
            session.section_contents[section_num]["content"] = result.processed_content
            deep_think_results[section_num] = {
                "is_valid": result.is_valid,
                "confidence": result.overall_confidence,
            }

    # Export evidence
    evidence_json = evidence_locker.export_to_json()
    evidence_csv = evidence_locker.export_to_csv()

    # Verification
    verification_html = None
    verification_result = None
    if enable_verification:
        verifier = Verifier(
            llm_client=llm_client,
            language=config.research.language,
        )

        content_parts = []
        for section_num, section_data in session.section_contents.items():
            if section_num.startswith("_"):
                continue
            content_parts.append(section_data.get("content", ""))

        content_for_verification = "\n\n".join(content_parts)

        verification_result = verifier.verify_content(
            content=content_for_verification,
            evidence_locker=evidence_locker,
            document_title=topic,
            strictness=config.verification_strictness,
        )

        verification_html = config.report.output_dir / f"verification_{session.session_id}.html"
        verifier.generate_verification_report_html(verification_result, verification_html)

    # Generate report
    report_generator = ReportGenerator(
        output_dir=config.report.output_dir / "reports",
        include_toc=config.report.include_toc,
        include_citations=config.report.include_citations,
        include_images=config.report.include_images,
        language=config.research.language,
    )

    report_path = report_generator.generate_report(
        session=session,
        evidence_locker=evidence_locker,
        format=config.report.format,
        verification_result=verification_result,
        target_pages=target_pages,
        target_characters=target_characters,
    )

    # Get token stats
    token_stats = get_token_stats()

    if verbose:
        print(f"\nReport generated: {report_path}")
        print(token_stats.get_summary(config.research.language))

    return {
        "session_id": session.session_id,
        "report_path": str(report_path),
        "evidence_json": str(evidence_json),
        "evidence_csv": str(evidence_csv),
        "verification_html": str(verification_html) if verification_html else None,
        "session": session,
        "evidence_locker": evidence_locker,
        "verification_result": verification_result,
        "deep_think_results": deep_think_results,
        "token_usage": token_stats.to_dict(),
        "evidence_count": evidence_count,
    }


# Export for notebook/script usage
__all__ = [
    "DeepResearchTool",
    "run_research",
    "run_manual_research",
    "diagnose_session",
    "Config",
    "LLMProvider",
    "SearchMethod",
    "ReportFormat",
    "ManualTableOfContents",
]
