"""
Main module for Deep Research Tool.

This module provides the main interface for conducting automated research.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from .config import Config, LLMProvider, SearchMethod, ReportFormat
from .api import get_client
from .search import get_search_client
from .research.researcher import Researcher, ResearchSession
from .evidence.locker import EvidenceLocker
from .verification.verifier import Verifier
from .report.generator import ReportGenerator
from .report.length_controller import ContentLengthController, LengthTarget
from .utils.document_reader import DocumentReader, auto_detect_additional_documents


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

    def _validate_config(self) -> None:
        """Validate configuration."""
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")

    def _create_llm_client(self):
        """Create LLM client based on configuration."""
        return get_client(
            provider=self.config.api.provider.value,
            api_key=self.config.api.get_active_api_key(),
            model=self.config.api.get_active_model(),
        )

    def _create_search_client(self):
        """Create search client based on configuration."""
        kwargs = {
            "max_results": self.config.search.max_results,
            "timeout": self.config.search.page_load_timeout,
            "extract_images": self.config.search.extract_images,
            "max_images": self.config.search.max_images_per_page,
        }

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
            language=self.config.research.language,
            output_dir=self.config.report.output_dir,
            progress_callback=progress_callback,
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

        return {
            "session_id": session.session_id,
            "report_path": str(report_path),
            "evidence_json": str(evidence_json),
            "evidence_csv": str(evidence_csv),
            "verification_html": str(verification_html) if verification_html else None,
            "session": session,
            "evidence_locker": evidence_locker,
            "verification_result": verification_result,
        }

    def _get_content_for_verification(self, session: ResearchSession) -> str:
        """Extract content from session for verification."""
        content_parts = []

        for section_num, section_data in session.section_contents.items():
            if section_num.startswith("_"):
                continue
            content_parts.append(section_data.get("content", ""))

        return "\n\n".join(content_parts)

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
        **api_key_param,
        **kwargs,
    )

    tool = DeepResearchTool(config)

    def progress_callback(message: str, percentage: float):
        if verbose and percentage >= 0:
            print(f"[{percentage:5.1f}%] {message}")

    return tool.run(
        query=query,
        requirements=requirements,
        progress_callback=progress_callback if verbose else None,
    )


# Export for notebook/script usage
__all__ = [
    "DeepResearchTool",
    "run_research",
    "Config",
    "LLMProvider",
    "SearchMethod",
    "ReportFormat",
]
