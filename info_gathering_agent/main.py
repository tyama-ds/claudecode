"""
Main module for Information Gathering Agent.

Provides the main interface for automated information collection,
focused on gathering evidence and synthesizing summaries
without report generation or verification.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

from .config import Config, LLMProvider, SearchMethod, ContentFilterMode
from .api import get_client
from .api.base import get_token_stats, reset_token_stats
from .search import get_search_client
from .research.gatherer import Gatherer, GatheringSession
from .evidence.locker import EvidenceLocker


@dataclass
class GatheringResult:
    """Result of an information gathering session."""
    session_id: str = ""
    query: str = ""
    research_plan: Optional[Any] = None
    evidence_locker: Optional[EvidenceLocker] = None
    section_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    executive_summary: Dict[str, Any] = field(default_factory=dict)
    quality_statistics: Dict[str, Any] = field(default_factory=dict)
    token_usage: Dict[str, Any] = field(default_factory=dict)
    evidence_json_path: Optional[str] = None
    evidence_csv_path: Optional[str] = None
    session_path: Optional[str] = None
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "section_summaries": self.section_summaries,
            "executive_summary": self.executive_summary,
            "quality_statistics": self.quality_statistics,
            "token_usage": self.token_usage,
            "evidence_json_path": self.evidence_json_path,
            "evidence_csv_path": self.evidence_csv_path,
            "session_path": self.session_path,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class InfoGatheringAgent:
    """
    Main interface for the Information Gathering Agent.

    Orchestrates information collection including:
    - Query analysis and research planning
    - Web search and content extraction
    - Information synthesis with per-section summaries
    - Evidence tracking and export

    Example usage:

        from info_gathering_agent import InfoGatheringAgent, Config

        config = Config()
        agent = InfoGatheringAgent(config)
        result = agent.gather("AI trends in healthcare 2024")

        # With explicit configuration
        from info_gathering_agent import create_config

        config = create_config(
            provider="anthropic",
            anthropic_api_key="your-key",
            model="claude-3-5-sonnet-20241022",
            search_method="selenium",
            research_iterations=5,
        )

        agent = InfoGatheringAgent(config)
        result = agent.gather(
            query="Renewable energy market analysis",
            requirements="Focus on solar and wind energy in Asia Pacific region",
        )

        # Access results
        print(result.executive_summary)
        print(result.section_summaries)
        print(f"Evidence saved to: {result.evidence_json_path}")
    """

    def __init__(self, config: Config = None):
        """
        Initialize InfoGatheringAgent.

        Args:
            config: Configuration object. If not provided, uses defaults
                   with environment variables for API keys.
        """
        self.config = config or Config()
        self._validate_config()

        # Initialize components
        self.llm_client = self._create_llm_client()
        self.search_client = self._create_search_client()
        self.gatherer = None

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

    def _create_content_filter(self):
        """Create content filter based on configuration."""
        from .evidence.content_filter import (
            ContentFilter,
            ContentFilterConfig,
            create_strict_filter,
            create_moderate_filter,
            create_minimal_filter,
        )

        filter_mode = self.config.research.content_filter_mode

        if filter_mode == ContentFilterMode.NONE:
            return None
        elif filter_mode == ContentFilterMode.STRICT:
            content_filter = create_strict_filter()
        elif filter_mode == ContentFilterMode.MINIMAL:
            content_filter = create_minimal_filter()
        else:  # MODERATE (default)
            content_filter = create_moderate_filter()

        for domain in self.config.research.custom_blocked_domains:
            content_filter.add_blocked_domain(domain)

        for domain in self.config.research.custom_whitelisted_domains:
            content_filter.add_whitelisted_domain(domain)

        return content_filter

    def gather(
        self,
        query: str,
        requirements: str = "",
        additional_context: str = "",
        progress_callback: Callable[[str, float], None] = None,
    ) -> GatheringResult:
        """
        Run the information gathering workflow.

        Args:
            query: The research query/topic
            requirements: Specific research requirements
            additional_context: Additional context information
            progress_callback: Callback function for progress updates

        Returns:
            GatheringResult with collected evidence and summaries
        """
        started_at = datetime.now().isoformat()

        # Initialize gatherer with content filter
        content_filter = self._create_content_filter()

        self.gatherer = Gatherer(
            llm_client=self.llm_client,
            search_client=self.search_client,
            min_iterations=self.config.research.min_iterations,
            max_iterations=self.config.research.max_iterations,
            max_queries_per_iteration=self.config.research.max_queries_per_iteration,
            max_pages_per_query=self.config.research.max_pages_per_query,
            language=self.config.research.language,
            output_dir=self.config.output_dir,
            progress_callback=progress_callback,
            extended_mode=self.config.research.extended_mode,
            crawl_max_pages=self.config.research.crawl_max_pages,
            crawl_max_depth=self.config.research.crawl_max_depth,
            crawl_max_sites=self.config.research.crawl_max_sites,
            crawl_relevance_threshold=self.config.research.crawl_relevance_threshold,
            use_enhanced_synthesis=self.config.research.use_enhanced_synthesis,
            content_filter=content_filter,
            filter_mode=self.config.research.content_filter_mode.value,
            crawl_mode=self.config.research.crawl_mode,
            fast_crawl_workers=self.config.research.fast_crawl_workers,
            fast_crawl_batch_size=self.config.research.fast_crawl_batch_size,
        )

        # Conduct gathering
        session = self.gatherer.conduct_research(
            query=query,
            requirements=requirements,
            additional_context=additional_context,
        )

        evidence_locker = self.gatherer.get_evidence_locker()

        # Export evidence
        evidence_json_path = None
        evidence_csv_path = None
        if evidence_locker:
            if self.config.export_evidence_json:
                evidence_json_path = str(evidence_locker.export_to_json())
            if self.config.export_evidence_csv:
                evidence_csv_path = str(evidence_locker.export_to_csv())

        # Clean up search client if selenium
        if hasattr(self.search_client, 'close'):
            self.search_client.close()

        # Get token usage statistics
        token_stats = get_token_stats()

        # Extract section summaries (excluding metadata keys)
        section_summaries = {
            k: v for k, v in session.section_contents.items()
            if not k.startswith("_")
        }

        # Extract executive summary
        executive_summary = session.section_contents.get("_executive_summary", {})

        # Build quality statistics
        quality_stats = {}
        if evidence_locker:
            all_evidence = evidence_locker.get_all_evidence()
            quality_stats = {
                "total_sources": len(all_evidence),
                "sections_with_content": len(section_summaries),
                "coherence_check": session.section_contents.get("_coherence_check", {}),
            }

        # Session path
        session_path = str(self.config.output_dir / f"session_{session.session_id}.json")

        completed_at = datetime.now().isoformat()

        return GatheringResult(
            session_id=session.session_id,
            query=query,
            research_plan=session.research_plan,
            evidence_locker=evidence_locker,
            section_summaries=section_summaries,
            executive_summary=executive_summary,
            quality_statistics=quality_stats,
            token_usage=token_stats.to_dict(),
            evidence_json_path=evidence_json_path,
            evidence_csv_path=evidence_csv_path,
            session_path=session_path,
            started_at=started_at,
            completed_at=completed_at,
        )

    def quick_gather(
        self,
        query: str,
        max_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform a quick information gathering without full research loop.

        Useful for quick fact-checking or getting a brief overview.

        Args:
            query: The research query
            max_results: Maximum search results to process

        Returns:
            Dictionary with quick gathering results
        """
        results = self.search_client.search(query, max_results=max_results)

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


def run_gathering(
    query: str,
    provider: str = "openai",
    api_key: str = None,
    model: str = None,
    search_method: str = "duckduckgo",
    iterations: int = 3,
    output_dir: str = "./output",
    requirements: str = "",
    verbose: bool = False,
    extended_mode: bool = False,
    crawl_max_pages: int = 10,
    crawl_max_depth: int = 2,
    crawl_max_sites: int = 3,
    http_proxy: str = None,
    https_proxy: str = None,
    verify_ssl: bool = True,
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
    # Content filtering parameters
    content_filter_mode: str = "moderate",
    custom_blocked_domains: List[str] = None,
    custom_whitelisted_domains: List[str] = None,
    # Fast crawl mode parameters
    crawl_mode: str = "standard",
    fast_crawl_workers: int = 10,
    fast_crawl_batch_size: int = 5,
    **kwargs
) -> GatheringResult:
    """
    Convenience function to run information gathering with simple parameters.

    Args:
        query: Research query/topic
        provider: LLM provider ('openai', 'anthropic', or 'local')
        api_key: API key (optional, uses env var if not provided)
        model: Model name (optional, uses default)
        search_method: Search method ('duckduckgo' or 'selenium')
        iterations: Research iterations per section
        output_dir: Output directory
        requirements: Research requirements
        verbose: Verbose output
        extended_mode: Enable extended mode (deep site crawling)
        crawl_max_pages: Max pages to crawl per site
        crawl_max_depth: Max link depth from seed URL
        crawl_max_sites: Max sites to crawl per search
        http_proxy: HTTP proxy URL
        https_proxy: HTTPS proxy URL
        verify_ssl: Verify SSL certificates
        multilingual: Enable multilingual search mode
        search_languages: List of language codes to search
        results_per_language: Number of results per language
        translate_results: Whether to translate results
        use_enhanced_synthesis: Use multi-pass content generation
        max_queries_per_iteration: Max queries per iteration
        max_pages_per_query: Max pages per query
        content_filter_mode: Content filter strictness
        custom_blocked_domains: List of domains to block
        custom_whitelisted_domains: List of domains to always allow
        crawl_mode: Crawl mode
        fast_crawl_workers: Max parallel workers
        fast_crawl_batch_size: Pages per batch
        **kwargs: Additional configuration options

    Returns:
        GatheringResult with collected evidence and summaries

    Example:
        result = run_gathering(
            "Climate change impacts on agriculture",
            provider="anthropic",
            iterations=5,
        )
        print(result.executive_summary)
        print(f"Evidence: {result.evidence_json_path}")
    """
    from .config import create_config

    api_key_param = {}
    if api_key:
        if provider == "openai":
            api_key_param["openai_api_key"] = api_key
        elif provider == "local":
            pass  # local doesn't need api_key
        else:
            api_key_param["anthropic_api_key"] = api_key

    config = create_config(
        provider=provider,
        model=model,
        search_method=search_method,
        research_iterations=iterations,
        output_dir=output_dir,
        verbose=verbose,
        extended_mode=extended_mode,
        crawl_max_pages=crawl_max_pages,
        crawl_max_depth=crawl_max_depth,
        crawl_max_sites=crawl_max_sites,
        http_proxy=http_proxy,
        https_proxy=https_proxy,
        verify_ssl=verify_ssl,
        multilingual=multilingual,
        search_languages=search_languages,
        results_per_language=results_per_language,
        translate_results=translate_results,
        use_enhanced_synthesis=use_enhanced_synthesis,
        max_queries_per_iteration=max_queries_per_iteration,
        max_pages_per_query=max_pages_per_query,
        content_filter_mode=content_filter_mode,
        custom_blocked_domains=custom_blocked_domains,
        custom_whitelisted_domains=custom_whitelisted_domains,
        crawl_mode=crawl_mode,
        fast_crawl_workers=fast_crawl_workers,
        fast_crawl_batch_size=fast_crawl_batch_size,
        **api_key_param,
        **kwargs,
    )

    reset_token_stats()

    agent = InfoGatheringAgent(config)

    def progress_cb(message: str, percentage: float):
        if verbose and percentage >= 0:
            print(f"[{percentage:5.1f}%] {message}")

    result = agent.gather(
        query=query,
        requirements=requirements,
        progress_callback=progress_cb if verbose else None,
    )

    if verbose:
        token_stats = get_token_stats()
        language = config.research.language if hasattr(config.research, 'language') else "en"
        print("\n" + token_stats.get_summary(language))

    return result
