"""
Main module for Deep Research Tool.

This module provides the main interface for conducting automated research.
"""

import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from .utils.helpers import ResearchWarnings, ensure_utf8_output
from .config import Config, LLMProvider, SearchMethod, ReportFormat, ReportGeneratorVersion
from .api import get_client
from .api.base import get_token_stats, reset_token_stats
from .search import get_search_client
from .research.researcher import Researcher, ResearchSession
from .research.manual_researcher import ManualResearcher, ManualTableOfContents
from .evidence.locker import EvidenceLocker
from .verification.verifier import Verifier
from .report.generator import ReportGenerator
from .report.length_controller import ContentLengthController, LengthTarget
from .report.v2 import ReportGeneratorV2, ReportFormatError, ReportContext, WritingStyle, TargetAudience
from .report.v3 import DocxReportGeneratorV3
from .report.figure_table_generator import FigureTableGenerator, add_figures_to_report
from .evidence.numerical_extractor import (
    NumericalDataExtractor,
    NumericalDataStore,
    DerivedMetricsCalculator,
)
from .report.chart_analyzer import ChartAnalyzer, ChartRecommendation
from .utils.document_reader import DocumentReader, auto_detect_additional_documents
from .thinking import DeepThinkProcessor, DeepThinkConfig as ThinkingConfig
from .thinking.reasoning_chain import ConsistencyMode
from .estimation import FermiEstimator, FermiEstimationConfig as EstimationConfig


# Style/Audience mapping for V2
_STYLE_MAP = {
    "formal": WritingStyle.FORMAL,
    "business": WritingStyle.BUSINESS,
    "technical": WritingStyle.TECHNICAL,
    "executive": WritingStyle.EXECUTIVE,
    "casual": WritingStyle.CASUAL,
}

_AUDIENCE_MAP = {
    "expert": TargetAudience.EXPERT,
    "business": TargetAudience.BUSINESS,
    "engineer": TargetAudience.ENGINEER,
    "general": TargetAudience.GENERAL,
    "student": TargetAudience.STUDENT,
}


def _create_report_generator(
    config: Config,
    llm_client,
    output_dir: Path = None,
) -> tuple:
    """
    Create appropriate report generator based on config.

    Returns:
        Tuple of (generator, version_tag)
        version_tag: "v1", "v2", or "v3"
    """
    output_dir = output_dir or config.report.output_dir / "reports"

    if config.report.generator_version == ReportGeneratorVersion.V3:
        # Create V3 generator (DOCX-native)
        writing_style = _STYLE_MAP.get(
            config.report.v2_writing_style,
            WritingStyle.BUSINESS
        )
        target_audience = _AUDIENCE_MAP.get(
            config.report.v2_target_audience,
            TargetAudience.BUSINESS
        )

        generator = DocxReportGeneratorV3(
            llm_client=llm_client,
            writing_style=writing_style,
            target_audience=target_audience,
            technical_level=config.report.v2_technical_level,
            enable_consistency_check=config.report.v2_enable_consistency_check,
            enable_two_phase=config.report.v2_enable_two_phase,
            language=config.research.language,
            target_pages=config.report.target_pages,
            target_characters=config.report.target_characters,
        )
        return generator, "v3"

    elif config.report.generator_version == ReportGeneratorVersion.V2:
        # Create V2 generator
        writing_style = _STYLE_MAP.get(
            config.report.v2_writing_style,
            WritingStyle.BUSINESS
        )
        target_audience = _AUDIENCE_MAP.get(
            config.report.v2_target_audience,
            TargetAudience.BUSINESS
        )

        generator = ReportGeneratorV2(
            llm_client=llm_client,
            writing_style=writing_style,
            target_audience=target_audience,
            technical_level=config.report.v2_technical_level,
            enable_consistency_check=config.report.v2_enable_consistency_check,
            enable_two_phase=config.report.v2_enable_two_phase,
            enable_polish=config.report.v2_enable_polish,
            language=config.research.language,
            target_pages=config.report.target_pages,
            target_characters=config.report.target_characters,
        )
        return generator, "v2"
    else:
        # Create V1 generator
        generator = ReportGenerator(
            output_dir=output_dir,
            include_toc=config.report.include_toc,
            include_citations=config.report.include_citations,
            include_images=config.report.include_images,
            language=config.research.language,
        )
        return generator, "v1"


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
        # Keep console output from crashing on non-cp932 characters (Windows)
        ensure_utf8_output()

        # Log the running version so it's always clear which installed copy
        # is executing (helps diagnose stale-install issues)
        from . import __version__
        print(f"[DeepResearchTool] version {__version__}")

        self.config = config or Config()
        self._validate_config()
        self._setup_logging()

        # Initialize components
        self.llm_client = self._create_llm_client()
        self.stage_llm_clients = self._create_stage_llm_clients()
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

        # kept for per-section cloning: the processor's validator/metrics
        # are stateful, so parallel sections each get their own instance
        self._thinking_config = thinking_config
        self.deep_think_processor = DeepThinkProcessor(
            llm_client=self.llm_client,
            config=thinking_config,
            language=self.config.research.language,
        )

    def _setup_logging(self) -> None:
        """Configure logging based on config settings."""
        logger = logging.getLogger("deep_research_tool")
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logger.setLevel(level)

        # Add file handler if log_file is configured
        if self.config.log_file:
            self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.config.log_file, encoding="utf-8")
            file_handler.setLevel(level)
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

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
            "max_tokens_limit": self.config.api.max_tokens_limit,
            "http_proxy": self.config.proxy.http_proxy,
            "https_proxy": self.config.proxy.https_proxy,
            "verify_ssl": self.config.proxy.verify_ssl,
        }

        # Custom endpoint (corporate gateway / local server); None = official
        kwargs["base_url"] = self.config.api.get_active_base_url()

        # Add local LLM specific settings
        if self.config.api.provider == LLMProvider.LOCAL:
            kwargs["backend"] = self.config.api.local_backend.value

        return get_client(**kwargs)

    def _create_stage_llm_clients(self) -> dict:
        """
        Create per-stage LLM clients from config.api.stage_overrides.

        Stages without an override are absent from the returned dict and
        fall back to the default client.
        """
        clients = {}
        for stage, spec in (self.config.api.stage_overrides or {}).items():
            provider = spec.get("provider", self.config.api.provider.value)
            api_key = spec.get("api_key") or {
                "openai": self.config.api.openai_api_key,
                "anthropic": self.config.api.anthropic_api_key,
                "local": self.config.api.local_api_key,
            }.get(provider)

            kwargs = {
                "provider": provider,
                "api_key": api_key,
                "model": spec.get("model"),
                "http_proxy": self.config.proxy.http_proxy,
                "https_proxy": self.config.proxy.https_proxy,
                "verify_ssl": self.config.proxy.verify_ssl,
                "base_url": spec.get(
                    "base_url",
                    self.config.api.get_base_url_for(provider),
                ),
            }
            if provider == "local":
                kwargs["backend"] = spec.get(
                    "backend",
                    self.config.api.local_backend.value,
                )

            clients[stage] = get_client(**kwargs)
            print(f"[StageLLM] {stage}: {provider} / {spec.get('model') or 'default model'}")

        # --- Local LLM role routing (feature flag, default off) -------
        # local_llm_role routes the verification and/or draft stages to
        # the user's OWN local server. The base_url must have been set
        # explicitly (config or LOCAL_LLM_BASE_URL) — this code never
        # invents an endpoint, and the client only ever talks to it as
        # the result of the user starting a run.
        role = getattr(self.config.research, "local_llm_role", "off")
        if role and role != "off" and \
                self.config.api.provider != LLMProvider.LOCAL:
            base_url = self.config.api.local_base_url
            if base_url:
                local_kwargs = {
                    "provider": "local",
                    "api_key": self.config.api.local_api_key,   # optional
                    "model": None,
                    "http_proxy": self.config.proxy.http_proxy,
                    "https_proxy": self.config.proxy.https_proxy,
                    "verify_ssl": self.config.proxy.verify_ssl,
                    "base_url": base_url,
                    "backend": self.config.api.local_backend.value,
                }
                try:
                    local_client = get_client(**local_kwargs)
                except Exception as e:
                    print(f"[RoleLLM] local client unavailable: {e}")
                    local_client = None
                if local_client is not None:
                    routed = {"verify": ("evaluation",),
                              "draft": ("writing",),
                              "all": ("evaluation", "writing")}[role]
                    for stage in routed:
                        # explicit stage_overrides always win over the flag
                        clients.setdefault(stage, local_client)
                    masked = "(api key set)" if \
                        self.config.api.local_api_key else "(no api key)"
                    print(f"[RoleLLM] local_llm_role={role}: "
                          f"{', '.join(routed)} -> {base_url} {masked}")
            else:
                print("[RoleLLM] local_llm_role is set but no "
                      "local_base_url is configured — routing skipped")
        return clients

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
            kwargs["implicit_wait"] = self.config.search.implicit_wait
            kwargs["driver_path"] = self.config.search.driver_path
        elif self.config.search.method == SearchMethod.DUCKDUCKGO:
            kwargs["region"] = self.config.search.region
            kwargs["safe_search"] = self.config.search.safe_search
            kwargs["simplify_min_results"] = self.config.search.query_simplify_min_results
            kwargs["simplify_max_retries"] = self.config.search.query_simplify_max_retries
            kwargs["waf_mitigation"] = self.config.search.waf_mitigation
            kwargs["per_domain_delay"] = self.config.search.per_domain_delay

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
        from .config import ContentFilterMode

        filter_mode = self.config.research.content_filter_mode

        if filter_mode == ContentFilterMode.NONE:
            return None
        elif filter_mode == ContentFilterMode.STRICT:
            content_filter = create_strict_filter()
        elif filter_mode == ContentFilterMode.MINIMAL:
            content_filter = create_minimal_filter()
        else:  # MODERATE (default)
            content_filter = create_moderate_filter()

        # Add custom blocked domains
        for domain in self.config.research.custom_blocked_domains:
            content_filter.add_blocked_domain(domain)

        # Add custom whitelisted domains
        for domain in self.config.research.custom_whitelisted_domains:
            content_filter.add_whitelisted_domain(domain)

        return content_filter

    def run(
        self,
        query: str,
        requirements: str = "",
        additional_context: str = "",
        additional_documents: List[str] = None,
        progress_callback: Callable[[str, float], None] = None,
        plan_review_callback: Callable = None,
        live_sink=None,
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
            plan_review_callback: Optional callback(plan, revise_fn) invoked
                after the research plan is generated, before research starts.
                When None and config.research.plan_review is True, a console
                prompt with a timeout (config.research.plan_review_timeout)
                is used in interactive sessions

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
        # Reset warning collector for this run (kept when another run is
        # already active, so parallel Web UI jobs don't wipe each other)
        ResearchWarnings.reset_if_idle()

        # --- run-scoped concurrency + token accounting ---
        # One RunLimits per run: every leaf LLM/HTTP call takes a composed
        # permit (run cap = parallel_max_workers, process cap = 16 shared
        # by all Web UI jobs in this server process). One run-local
        # TokenUsageStats per run: parallel jobs never mix their numbers
        # and another job's reset cannot clear them.
        from .utils.concurrency import RunLimits
        from .api.base import TokenUsageStats
        from .verification.runtime import VerificationProgress
        # live verification progress (phases / counters / cancel) — the
        # Web UI polls this and can request a safe cancellation
        self.verification_progress = VerificationProgress()
        run_limits = RunLimits(self.config.research.parallel_max_workers)
        self.run_limits = run_limits
        run_token_stats = TokenUsageStats()
        self.run_token_stats = run_token_stats
        for _client in [self.llm_client, *self.stage_llm_clients.values()]:
            try:
                _client.concurrency_limiter = run_limits
                _client.token_stats = run_token_stats
            except Exception:
                pass
        try:
            self.search_client.concurrency_limiter = run_limits
        except Exception:
            pass

        # Log informational note when both enhanced_synthesis and V2 two_phase are active
        if (self.config.research.use_enhanced_synthesis
                and self.config.report.generator_version == ReportGeneratorVersion.V2
                and self.config.report.v2_enable_two_phase):
            print("[Info] Both use_enhanced_synthesis and v2_enable_two_phase are active. "
                  "These are complementary: enhanced_synthesis improves information "
                  "gathering quality (research layer), while two_phase improves "
                  "report writing quality (report layer).")

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

        # Initialize researcher with content filter
        content_filter = self._create_content_filter()

        # Plan review: explicit callback wins; otherwise use the console
        # prompt (with auto-continue timeout) when enabled in config
        if plan_review_callback is None and self.config.research.plan_review:
            from .utils.plan_review import make_console_plan_review_callback
            plan_review_callback = make_console_plan_review_callback(
                timeout=self.config.research.plan_review_timeout,
                language=self.config.research.language,
            )

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
            content_filter=content_filter,
            filter_mode=self.config.research.content_filter_mode.value,
            crawl_mode=self.config.research.crawl_mode,
            fast_crawl_workers=self.config.research.fast_crawl_workers,
            parallel_max_workers=self.config.research.parallel_max_workers,
            fast_crawl_batch_size=self.config.research.fast_crawl_batch_size,
            ai_crawl_max_total_pages=self.config.research.ai_crawl_max_total_pages,
            ai_crawl_max_depth=self.config.research.ai_crawl_max_depth,
            ai_crawl_site_depth=self.config.research.ai_crawl_site_depth,
            ai_crawl_max_llm_calls=self.config.research.ai_crawl_max_llm_calls,
            ai_crawl_max_pages_per_domain=self.config.research.ai_crawl_max_pages_per_domain,
            ai_crawl_politeness_delay=self.config.research.ai_crawl_politeness_delay,
            selenium_headless=self.config.search.headless,
            selenium_browser=self.config.search.browser,
            selenium_proxies=self.config.proxy.get_proxies_dict(),
            selenium_verify_ssl=self.config.proxy.verify_ssl,
            selenium_driver_path=self.config.search.driver_path,
            importance_threshold=self.config.research.importance_threshold,
            min_high_importance_sources=self.config.research.min_high_importance_sources,
            max_gap_fill_rounds=self.config.research.max_gap_fill_rounds,
            planning_llm=self.stage_llm_clients.get("planning"),
            crawling_llm=self.stage_llm_clients.get("crawling"),
            evaluation_llm=self.stage_llm_clients.get("evaluation"),
            writing_llm=self.stage_llm_clients.get("writing"),
            source_mode=self.config.research.source_mode,
            multilingual_config=self.config.multilingual if self.config.multilingual.enabled else None,
            max_content_length=self.config.research.max_content_length,
            target_pages=self.config.report.target_pages,
            target_characters=self.config.report.target_characters,
            plan_review_callback=plan_review_callback,
        )

        # Conduct research
        session = self.researcher.conduct_research(
            query=query,
            requirements=requirements,
            additional_context=additional_context,
            additional_documents=doc_contents,
        )

        evidence_locker = self.researcher.get_evidence_locker()

        # --- live report sinks (Web UI preview and/or Word COM) ---
        # Everything emitted before on_finalized is a watermarked DRAFT;
        # the verified final body replaces it at freeze.
        self.finalization_outcome = None
        live_sink = self._build_live_sink(live_sink, session)
        self.live_sink = live_sink
        if live_sink is not None and session.research_plan:
            toc = [{"section": it.section, "title": it.title}
                   for it in (session.research_plan.table_of_contents
                              .get_flat_sections())]
            live_sink.on_plan(session.research_plan.title, toc)

        # Check if content expansion is needed (target specified and content too short)
        session = self._expand_if_needed(
            session=session,
            progress_callback=progress_callback,
        )

        # Save pre-DeepThink content snapshot for Fermi estimation context.
        # DeepThink modifies session.section_contents in place, which could
        # bias Fermi estimation toward DeepThink's reasoning rather than
        # raw evidence. We preserve the original content for Fermi's context.
        _pre_deep_think_content = None
        if self.config.deep_think.enabled and self.config.fermi_estimation.enabled:
            import copy
            _pre_deep_think_content = copy.deepcopy(session.section_contents)

        # Apply DeepThink processing if enabled
        deep_think_results = None
        if self.config.deep_think.enabled and self.deep_think_processor:
            session, deep_think_results = self._apply_deep_think(
                session=session,
                evidence_locker=evidence_locker,
                progress_callback=progress_callback,
            )

        # Evidence export happens AFTER report finalization (further below)
        # so that evidence added by the final research rounds is included
        # in the JSON/CSV exports and the references registry.
        evidence_json = None
        evidence_csv = None

        # ------------------------------------------------------------------
        # Pipeline ordering (finalization spec):
        #   ALL report versions (V1 / V2 / V3) run the SAME finalization
        #   pipeline on the common chapter form:
        #     generation result -> {section_id: text} -> verify -> decide
        #     -> act -> re-verify -> freeze -> deterministic render.
        #   V1: enhancement first, then finalization right before render
        #       (after the Fermi section joined the body).
        #   V2/V3: chapter generation, then finalization of the final
        #       candidate body (with Fermi/glossary/warnings already in
        #       it), then deterministic rendering.
        # ------------------------------------------------------------------
        verification_html = None
        verification_result = None
        legacy_v1 = (self.config.report.generator_version
                     == ReportGeneratorVersion.V1)

        if legacy_v1:
            if progress_callback:
                progress_callback(
                    "Reviewing all evidence for report enhancement...", 88)
            session = self._enhance_sections_with_full_evidence(
                session=session,
                evidence_locker=evidence_locker,
                query=query,
            )
            # live report: V1 section drafts become visible here
            if live_sink is not None:
                for _sid, _sdata in session.section_contents.items():
                    if _sid.startswith("_") or not _sdata.get("content"):
                        continue
                    live_sink.on_section(_sid, _sdata.get("title", _sid),
                                         _sdata["content"], draft=True)
            # V1 verification now runs through the SAME finalization
            # pipeline as V2/V3 — in the V1 branch below, after the Fermi
            # section exists, immediately before rendering.

        # Generate report
        if progress_callback:
            progress_callback("Generating report...", 95)

        generator, version_tag = _create_report_generator(
            config=self.config,
            llm_client=self.stage_llm_clients.get("writing", self.llm_client),
            output_dir=self.config.report.output_dir / "reports",
        )
        self.report_generator = generator
        # live report: V2/V3 emit every chapter draft/update as it is
        # written (V1 sections were emitted after enhancement above)
        if hasattr(generator, "live_sink"):
            generator.live_sink = live_sink

        # Stage-1 adaptive length: the pre-draft allocation from the
        # research plan feeds draft generation (audit item 10). V2/V3
        # accept per-section character targets; V1 keeps its
        # synthesis-side length control and is covered by stages 2-3 in
        # the finalization loop.
        if version_tag in ("v2", "v3") and session.research_plan:
            from .report.length_planner import LengthPlanner
            rp = self.config.report
            stage1 = LengthPlanner(
                language=self.config.research.language,
                length_mode=rp.length_mode,
                preferred_body_chars=(rp.preferred_body_chars
                                      or rp.target_characters),
                hard_min_body_chars=rp.hard_min_body_chars,
                hard_max_body_chars=rp.hard_max_body_chars,
                length_tolerance=rp.length_tolerance,
            )
            plan_sections = [
                {"section": it.section, "title": it.title}
                for it in (session.research_plan.table_of_contents
                           .get_flat_sections())]
            stage1_plans = stage1.initial_allocation(plan_sections)
            generator.section_char_targets = {
                sid: p.provisional_chars for sid, p in stage1_plans.items()}

        # --- Run Fermi estimation early (before report save) ---
        # This must happen before DOCX/PDF conversion so that the Fermi
        # section is included in non-text formats.
        fermi_results = None
        fermi_markdown = ""
        if self.config.fermi_estimation.enabled:
            if progress_callback:
                progress_callback("Running Fermi estimation...", 96)

            fermi_results, fermi_markdown = self._run_fermi_estimation(
                session=session,
                evidence_locker=evidence_locker,
                query=query,
                pre_deep_think_content=_pre_deep_think_content,
            )

        # --- ALL semantic figure work happens BEFORE finalization -------
        # Chart data, titles, captions and table cells are generated (and
        # PNGs rendered) here, so the figure semantics join the body for
        # verification and are FROZEN with it. After the freeze, figures
        # are only INSERTED deterministically.
        figure_collection = None
        fig_generator = None
        if self.config.report.auto_figures:
            if progress_callback:
                progress_callback("Generating figures and tables...", 94)
            try:
                figure_collection, fig_generator = \
                    self._generate_figure_collection(
                        session=session, evidence_locker=evidence_locker)
            except Exception as e:
                print(f"[AutoFigures] Failed with error: {e}. "
                      f"Continuing without figures.")
                ResearchWarnings.get_instance().add(
                    ResearchWarnings.MEDIUM, "AutoFigures",
                    f"Figure/table generation failed entirely. Report will "
                    f"not contain auto-generated figures or tables. "
                    f"Error: {e}",
                )
                figure_collection, fig_generator = None, None

        extras_flags = {}
        semantic_freeze_hash = None
        if version_tag == "v3":
            # V3: DOCX-native generation flow
            result = generator.generate_report(
                research_topic=query,
                research_plan=session.research_plan,
                section_contents=session.section_contents,
            )

            # Finalization loop: verify the final candidate body — with
            # Fermi / glossary / warnings / FIGURE SEMANTICS already IN
            # it — and only then freeze it (V3 rendering is deterministic)
            if self.config.enable_verification:
                extra_chapters, extras_flags = self._build_extra_chapters(
                    generator, result, fermi_markdown,
                    figure_collection=figure_collection)
                verification_result, verification_html = \
                    self._run_finalization_loop(
                        result=result,
                        session=session,
                        evidence_locker=evidence_locker,
                        query=query,
                        requirements=requirements,
                        progress_callback=progress_callback,
                        extra_chapters=extra_chapters,
                        render_only=extras_flags.get("figures_key"),
                    )
                if extras_flags.get("fermi"):
                    fermi_markdown = ""    # already in the verified body
                semantic_freeze_hash = self._freeze_semantics(
                    {sid: ch.content for sid, ch in result.chapters.items()},
                    figure_collection)

            # Build warnings text. When the warnings snapshot was already
            # verified inside the body, only warnings raised AFTER the
            # finalization are appended (deterministic rendering).
            warnings_text = ""
            warnings_collector = ResearchWarnings.get_instance()
            if warnings_collector.has_warnings():
                language = getattr(self.config.research, "language", "en")
                if extras_flags.get("warnings"):
                    warnings_text = self._late_warnings_section(
                        warnings_collector, language)
                else:
                    warnings_text = warnings_collector.to_report_section(language)

            # Build DOCX directly via python-docx API
            output_dir = self.config.report.output_dir / "reports"
            report_path = generator.generate_and_save(
                result=result,
                output_dir=output_dir,
                filename=f"report_{session.session_id}",
                evidence_locker=evidence_locker,
                figure_collection=figure_collection,
                include_glossary=(self.config.report.v2_include_glossary
                                  and not extras_flags.get("glossary")),
                fermi_markdown=fermi_markdown,
                warnings_text=warnings_text,
            )

        elif version_tag == "v2":
            # V2: Use new generation flow
            result = generator.generate_report(
                research_topic=query,
                research_plan=session.research_plan,
                section_contents=session.section_contents,
            )

            # Finalization loop: verify the final candidate body — with
            # Fermi / glossary / warnings already IN it — act on
            # structured findings (research / rewrite / compress /
            # finalize-with-limitations), re-verify after every change,
            # then FREEZE. Everything after this point is deterministic
            # rendering (markdown assembly, format conversion).
            if self.config.enable_verification:
                extra_chapters, extras_flags = self._build_extra_chapters(
                    generator, result, fermi_markdown,
                    figure_collection=figure_collection)
                verification_result, verification_html = \
                    self._run_finalization_loop(
                        result=result,
                        session=session,
                        evidence_locker=evidence_locker,
                        query=query,
                        requirements=requirements,
                        progress_callback=progress_callback,
                        extra_chapters=extra_chapters,
                        render_only=extras_flags.get("figures_key"),
                    )
                if extras_flags.get("fermi"):
                    fermi_markdown = ""    # already in the verified body
                semantic_freeze_hash = self._freeze_semantics(
                    {sid: ch.content for sid, ch in result.chapters.items()},
                    figure_collection)

            # Generate final document as markdown
            final_doc = generator.generate_final_document(
                result,
                include_glossary=(self.config.report.v2_include_glossary
                                  and not extras_flags.get("glossary")),
                evidence_locker=evidence_locker,
            )

            # Append Fermi estimation to markdown BEFORE format conversion
            # (only when the finalization did not already verify it)
            if fermi_markdown:
                final_doc += fermi_markdown

            # Append warnings to markdown BEFORE format conversion. When
            # a warnings snapshot was verified inside the body, only
            # warnings raised after finalization are appended.
            warnings_collector = ResearchWarnings.get_instance()
            if warnings_collector.has_warnings():
                language = getattr(self.config.research, "language", "en")
                if extras_flags.get("warnings"):
                    final_doc += self._late_warnings_section(
                        warnings_collector, language)
                else:
                    final_doc += warnings_collector.to_report_section(language)

            # Save to file in the configured format (DOCX/PDF/HTML/MD)
            output_dir = self.config.report.output_dir / "reports"
            report_path = generator.save_report(
                markdown_content=final_doc,
                output_dir=output_dir,
                filename=f"report_{session.session_id}",
                format=self.config.report.format,
                strict_format=self.config.report.strict_format,
            )
        else:
            # V1: common finalization on the session sections BEFORE the
            # deterministic render. The Fermi section joins the body
            # first so it is part of what gets verified (audit item 5).
            finalized = False
            if self.config.enable_verification:
                if fermi_markdown:
                    v1_keys = [k for k in session.section_contents
                               if not k.startswith("_")]
                    if v1_keys:
                        last = v1_keys[-1]
                        prev = session.section_contents[last].get("content", "")
                        session.section_contents[last]["content"] = (
                            prev.rstrip() + "\n" + fermi_markdown)
                        fermi_markdown = ""
                verification_result, verification_html = \
                    self._run_finalization_v1(
                        session=session,
                        evidence_locker=evidence_locker,
                        query=query,
                        requirements=requirements,
                        progress_callback=progress_callback,
                        figure_collection=figure_collection,
                    )
                finalized = True
                # V1 stores the executive summary / key findings OUTSIDE
                # the chapter map — they are semantic content the reader
                # sees, so they join the freeze as manifest extras
                _exec = session.section_contents.get(
                    "_executive_summary") or {}
                semantic_freeze_hash = self._freeze_semantics(
                    {sid: sd.get("content", "")
                     for sid, sd in session.section_contents.items()
                     if not sid.startswith("_") and sd.get("content")},
                    figure_collection,
                    extras={k: _exec.get(k) for k in
                            ("executive_summary", "key_findings",
                             "recommendations") if _exec.get(k)} or None)

            # After finalization the body is FROZEN: the legacy length
            # adjustment must never rewrite a verified body, so the
            # targets are only applied on the unverified path.
            report_path = generator.generate_report(
                session=session,
                evidence_locker=evidence_locker,
                format=self.config.report.format,
                verification_result=None,
                target_pages=None if finalized else self.config.report.target_pages,
                target_characters=(None if finalized
                                   else self.config.report.target_characters),
            )

            # For V1 non-markdown formats, fermi section was not included
            # in the generator. Append to markdown files only (verified
            # bodies already contain it).
            if fermi_markdown and Path(report_path).suffix.lower() == ".md":
                self._append_text_to_file(Path(report_path), fermi_markdown)

        # Export evidence AFTER finalization (audit item 13): the exports,
        # references and registry include evidence added by the final
        # research rounds, in the final citation order.
        if self.config.research.save_evidence:
            evidence_format = self.config.research.evidence_format
            if evidence_format in ("json", "both"):
                evidence_json = evidence_locker.export_to_json()
            if evidence_format in ("csv", "both"):
                evidence_csv = evidence_locker.export_to_csv()

        # Deterministic figure INSERTION (V3 embeds inline during
        # generate_and_save). The collection was generated and verified
        # BEFORE the freeze; no LLM runs here.
        figures_report_path = None
        if figure_collection is not None and version_tag != "v3":
            if progress_callback:
                progress_callback("Inserting figures and tables...", 97)
            try:
                figures_report_path = self._insert_figures_into_report(
                    report_path=Path(report_path),
                    collection=figure_collection,
                    generator=fig_generator,
                )
            except Exception as e:
                print(f"[AutoFigures] Insertion failed: {e}. "
                      f"Continuing with original report.")
                ResearchWarnings.get_instance().add(
                    ResearchWarnings.MEDIUM,
                    "AutoFigures",
                    f"Figure insertion failed. The report keeps its "
                    f"original content without figures. Error: {e}",
                )
                figures_report_path = None

            if figures_report_path and not Path(figures_report_path).exists():
                print(f"[AutoFigures] Output file not found: {figures_report_path}. "
                      f"Falling back to original report.")
                ResearchWarnings.get_instance().add(
                    ResearchWarnings.MEDIUM,
                    "AutoFigures",
                    f"Figure-enhanced report file not found on disk. "
                    f"Falling back to original report without figures.",
                )
                figures_report_path = None

        # Determine the active report path
        active_report_path = Path(figures_report_path) if figures_report_path else Path(report_path)

        # Clean up search client if selenium
        if hasattr(self.search_client, 'close'):
            self.search_client.close()

        # Append warnings section to the report if any fallbacks occurred
        # (V2/V3 already handle warnings before/during format conversion;
        #  V1 needs post-save appending for markdown files)
        warnings_collector = ResearchWarnings.get_instance()
        if version_tag == "v1" and warnings_collector.has_warnings():
            self._append_warnings_to_report(
                active_report_path, warnings_collector)

        # Hard length bounds are ABSOLUTE constraints independent of the
        # verification toggle: with verification off there is no edit
        # loop to fix a violation (padding / mechanical truncation are
        # forbidden), so a violation is reported as a CRITICAL warning —
        # never a silent normal completion.
        rp_cfg = self.config.report
        if not self.config.enable_verification and (
                rp_cfg.hard_min_body_chars or rp_cfg.hard_max_body_chars):
            from .report.finalization import count_body_chars
            if version_tag == "v1":
                _body = "\n\n".join(
                    sd.get("content", "")
                    for sid, sd in session.section_contents.items()
                    if not sid.startswith("_"))
            else:
                _body = "\n\n".join(ch.content
                                    for ch in result.chapters.values())
            _chars = count_body_chars(
                _body, exclude_references=rp_cfg.exclude_references_from_count)
            if rp_cfg.hard_max_body_chars and \
                    _chars > rp_cfg.hard_max_body_chars:
                ResearchWarnings.get_instance().add(
                    ResearchWarnings.CRITICAL, "LengthBounds",
                    f"本文({_chars}字)が上限(hard_max_body_chars="
                    f"{rp_cfg.hard_max_body_chars})を超過しています。検証無効の"
                    f"ため自動圧縮は行われません。このレポートは正常完了では"
                    f"ありません。")
            if rp_cfg.hard_min_body_chars and \
                    _chars < rp_cfg.hard_min_body_chars:
                ResearchWarnings.get_instance().add(
                    ResearchWarnings.CRITICAL, "LengthBounds",
                    f"本文({_chars}字)が下限(hard_min_body_chars="
                    f"{rp_cfg.hard_min_body_chars})に達していません。水増しは"
                    f"行いません。追加調査で情報を増やしてください。")

        # Semantic freeze check: the content about to ship must hash
        # IDENTICALLY to the snapshot taken at freeze time. Renderers may
        # only have done non-semantic work since (layout, format
        # conversion, deterministic insertion of the frozen figures).
        semantic_output_hash = None
        semantic_artifact_check = None
        if semantic_freeze_hash:
            if version_tag == "v1":
                output_chapters = {
                    sid: sd.get("content", "")
                    for sid, sd in session.section_contents.items()
                    if not sid.startswith("_") and sd.get("content")}
                _exec = session.section_contents.get(
                    "_executive_summary") or {}
                _extras = {k: _exec.get(k) for k in
                           ("executive_summary", "key_findings",
                            "recommendations") if _exec.get(k)} or None
            else:
                output_chapters = {sid: ch.content
                                   for sid, ch in result.chapters.items()}
                _extras = None
            semantic_output_hash = self._check_semantics_at_output(
                output_chapters, figure_collection, semantic_freeze_hash,
                extras=_extras)
            # audit item D: the check must also REBUILD from the actual
            # saved artifact — not only re-hash the in-memory object
            semantic_artifact_check = self._check_semantics_in_artifact(
                active_report_path)

        # Live report: deliver the VERIFIED final body (replaces drafts,
        # removes watermarks, appends references) and release the sinks.
        if live_sink is not None:
            try:
                refs = [e.citation_text
                        for e in evidence_locker.get_all_evidence()]
                outcome = getattr(self, "finalization_outcome", None)
                if outcome is not None:
                    live_sink.on_finalized(outcome["chapters"], refs)
                else:
                    final_chapters = {
                        sid: sd.get("content", "")
                        for sid, sd in session.section_contents.items()
                        if not sid.startswith("_") and sd.get("content")}
                    live_sink.on_finalized(final_chapters, refs)
            except Exception as e:
                print(f"[LiveReport] finalize event failed: {e}")
            finally:
                try:
                    live_sink.close()
                except Exception:
                    pass

        # Get token usage statistics
        token_stats = get_token_stats()

        return {
            "session_id": session.session_id,
            "report_path": str(active_report_path),
            "report_path_original": str(report_path),
            "figures_report_path": str(figures_report_path) if figures_report_path else None,
            "evidence_json": str(evidence_json),
            "evidence_csv": str(evidence_csv),
            "verification_html": str(verification_html) if verification_html else None,
            "session": session,
            "evidence_locker": evidence_locker,
            "verification_result": verification_result,
            "deep_think_results": deep_think_results,
            "fermi_estimation_results": fermi_results,
            "token_usage": run_token_stats.to_dict(),
            "max_concurrency_observed": run_limits.run_peak,
            "semantic_manifest_hash_at_freeze": semantic_freeze_hash,
            "semantic_manifest_hash_at_output": semantic_output_hash,
            "semantic_artifact_check": semantic_artifact_check,
            "verification_summary": (getattr(self, "finalization_outcome",
                                             None) or {}).get(
                "verification_summary"),
            "verification_cancelled": (getattr(self, "finalization_outcome",
                                               None) or {}).get(
                "decision") == "cancelled",
            "warnings": warnings_collector.to_dict_list(),
            "warning_count": warnings_collector.count(),
        }

    def _run_fermi_estimation(
        self,
        session: ResearchSession,
        evidence_locker: EvidenceLocker,
        query: str,
        pre_deep_think_content: Optional[Dict] = None,
    ) -> tuple:
        """Run Fermi estimation and return results + markdown section.

        Unlike _apply_fermi_estimation, this does NOT write to any file.
        Returns the results and a markdown string that can be appended to
        the report content before format conversion.

        Returns:
            Tuple of (result_dicts_list_or_None, fermi_markdown_string)
        """
        import traceback

        try:
            config = EstimationConfig(
                enabled=True,
                target_metrics=self.config.fermi_estimation.target_metrics,
                auto_detect_targets=self.config.fermi_estimation.auto_detect_targets,
                max_tree_depth=self.config.fermi_estimation.max_tree_depth,
                max_leaf_nodes=self.config.fermi_estimation.max_leaf_nodes,
                monte_carlo_iterations=self.config.fermi_estimation.monte_carlo_iterations,
                validate_with_llm=self.config.fermi_estimation.validate_with_llm,
                min_confidence_threshold=self.config.fermi_estimation.min_confidence_threshold,
                write_to_data_store=self.config.fermi_estimation.write_to_data_store,
                include_sensitivity=self.config.fermi_estimation.include_sensitivity,
                enable_sub_decomposition=self.config.fermi_estimation.enable_sub_decomposition,
                sub_decomposition_confidence_threshold=self.config.fermi_estimation.sub_decomposition_confidence_threshold,
                sub_decomposition_max_iterations=self.config.fermi_estimation.sub_decomposition_max_iterations,
                sub_decomposition_min_sensitivity_pct=self.config.fermi_estimation.sub_decomposition_min_sensitivity_pct,
            )

            estimator = FermiEstimator(
                llm_client=self.llm_client,
                config=config,
                language=self.config.research.language,
            )

            # Extract numerical data
            data_store = NumericalDataStore(research_topic=query)
            extractor = NumericalDataExtractor(
                llm_client=self.llm_client,
                language=self.config.research.language,
            )
            for evidence in evidence_locker.get_all_evidence():
                content = evidence.content_excerpt or ""
                if content:
                    points = extractor.extract_from_content(
                        content=content[:3000],
                        source_url=evidence.url,
                        source_title=evidence.title,
                        evidence_id=evidence.id,
                    )
                    data_store.add_many(points)

            # Determine target metrics
            target_metrics = config.target_metrics
            if not target_metrics and config.auto_detect_targets:
                detected = estimator.detect_target_metrics(query, data_store)
                target_metrics = [d["metric"] for d in detected[:3]]
                print(f"[FermiEstimation] Auto-detected targets: {target_metrics}")

            if not target_metrics:
                print("[FermiEstimation] No target metrics found. Skipping.")
                return None, ""

            # Build context from pre-DeepThink content if available
            if pre_deep_think_content:
                context_parts = []
                for section_num, section_data in pre_deep_think_content.items():
                    if str(section_num).startswith("_"):
                        continue
                    context_parts.append(section_data.get("content", ""))
                context = "\n\n".join(context_parts)[:5000]
            else:
                context = self._get_content_for_verification(session)[:5000]

            results = estimator.estimate_multiple(
                target_metrics=target_metrics,
                data_store=data_store,
                evidence_locker=evidence_locker,
                context=context,
            )

            if not results:
                return None, ""

            # Build markdown section
            fermi_md = self._build_fermi_markdown(results)
            return [r.to_dict() for r in results], fermi_md

        except Exception as e:
            print(f"[FermiEstimation] Failed: {e}")
            traceback.print_exc()
            ResearchWarnings.get_instance().add(
                ResearchWarnings.MEDIUM,
                "FermiEstimation",
                f"Fermi estimation failed entirely. "
                f"No quantitative estimation section will be added to the report. Error: {e}",
            )
            return None, ""

    def _build_fermi_markdown(self, results: list) -> str:
        """Build a markdown section from Fermi estimation results."""
        section_lines = ["\n\n---\n"]
        if self.config.research.language == "ja":
            section_lines.append("## フェルミ推定\n")
        else:
            section_lines.append("## Fermi Estimation\n")

        for result in results:
            section_lines.append(result.to_summary(self.config.research.language))
            section_lines.append("")

        return "\n".join(section_lines)

    def _append_text_to_file(self, filepath: Path, text: str) -> None:
        """Atomically append text to a file (for markdown files)."""
        import os
        import tempfile

        try:
            content = filepath.read_text(encoding="utf-8")
            content += text

            fd, tmp_path = tempfile.mkstemp(
                dir=filepath.parent,
                suffix=filepath.suffix,
                prefix=".append_tmp_",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                os.replace(tmp_path, str(filepath))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            print(f"[AppendText] Failed to append to {filepath}: {e}")

    def _apply_fermi_estimation(
        self,
        session: ResearchSession,
        evidence_locker: EvidenceLocker,
        query: str,
        report_path: Path,
        pre_deep_think_content: Optional[Dict] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Run Fermi estimation and append results to the report.

        Args:
            session: Research session
            evidence_locker: Evidence locker with sources
            query: Original research query
            report_path: Path to the generated report file
            pre_deep_think_content: Original section_contents before DeepThink
                processing (if available). Used to avoid bias from DeepThink
                reasoning when building the estimation context.

        Returns:
            List of estimation result dicts, or None if failed
        """
        import traceback

        try:
            config = EstimationConfig(
                enabled=True,
                target_metrics=self.config.fermi_estimation.target_metrics,
                auto_detect_targets=self.config.fermi_estimation.auto_detect_targets,
                max_tree_depth=self.config.fermi_estimation.max_tree_depth,
                max_leaf_nodes=self.config.fermi_estimation.max_leaf_nodes,
                monte_carlo_iterations=self.config.fermi_estimation.monte_carlo_iterations,
                validate_with_llm=self.config.fermi_estimation.validate_with_llm,
                min_confidence_threshold=self.config.fermi_estimation.min_confidence_threshold,
                write_to_data_store=self.config.fermi_estimation.write_to_data_store,
                include_sensitivity=self.config.fermi_estimation.include_sensitivity,
                enable_sub_decomposition=self.config.fermi_estimation.enable_sub_decomposition,
                sub_decomposition_confidence_threshold=self.config.fermi_estimation.sub_decomposition_confidence_threshold,
                sub_decomposition_max_iterations=self.config.fermi_estimation.sub_decomposition_max_iterations,
                sub_decomposition_min_sensitivity_pct=self.config.fermi_estimation.sub_decomposition_min_sensitivity_pct,
            )

            estimator = FermiEstimator(
                llm_client=self.llm_client,
                config=config,
                language=self.config.research.language,
            )

            # Extract numerical data for the estimator
            data_store = NumericalDataStore(research_topic=query)
            extractor = NumericalDataExtractor(
                llm_client=self.llm_client,
                language=self.config.research.language,
            )
            for evidence in evidence_locker.get_all_evidence():
                content = evidence.content_excerpt or ""
                if content:
                    points = extractor.extract_from_content(
                        content=content[:3000],
                        source_url=evidence.url,
                        source_title=evidence.title,
                        evidence_id=evidence.id,
                    )
                    data_store.add_many(points)

            # Determine target metrics
            target_metrics = config.target_metrics
            if not target_metrics and config.auto_detect_targets:
                detected = estimator.detect_target_metrics(query, data_store)
                target_metrics = [d["metric"] for d in detected[:3]]
                print(f"[FermiEstimation] Auto-detected targets: {target_metrics}")

            if not target_metrics:
                print("[FermiEstimation] No target metrics found. Skipping.")
                return None

            # Build context from pre-DeepThink content if available,
            # to avoid bias from DeepThink reasoning.
            if pre_deep_think_content:
                context_parts = []
                for section_num, section_data in pre_deep_think_content.items():
                    if str(section_num).startswith("_"):
                        continue
                    context_parts.append(section_data.get("content", ""))
                context = "\n\n".join(context_parts)[:5000]
            else:
                context = self._get_content_for_verification(session)[:5000]
            results = estimator.estimate_multiple(
                target_metrics=target_metrics,
                data_store=data_store,
                evidence_locker=evidence_locker,
                context=context,
            )

            # Append Fermi estimation section to report
            if results and report_path.exists():
                self._append_fermi_to_report(results, report_path)

            return [r.to_dict() for r in results]

        except Exception as e:
            print(f"[FermiEstimation] Failed: {e}")
            traceback.print_exc()
            ResearchWarnings.get_instance().add(
                ResearchWarnings.MEDIUM,
                "FermiEstimation",
                f"Fermi estimation failed entirely. "
                f"No quantitative estimation section will be added to the report. Error: {e}",
            )
            return None

    def _append_fermi_to_report(
        self,
        results: list,
        report_path: Path,
    ) -> None:
        """Append Fermi estimation section to the report file.

        Uses atomic write (write to temp file + rename) to prevent data loss
        if the process is interrupted mid-write.
        """
        import tempfile

        try:
            content = report_path.read_text(encoding="utf-8")

            section_lines = ["\n\n---\n"]
            if self.config.research.language == "ja":
                section_lines.append("## フェルミ推定\n")
            else:
                section_lines.append("## Fermi Estimation\n")

            for result in results:
                section_lines.append(result.to_summary(self.config.research.language))
                section_lines.append("")

            content += "\n".join(section_lines)

            # Atomic write: write to temp file in the same directory, then rename.
            # os.replace() is atomic on POSIX and near-atomic on Windows.
            import os
            fd, tmp_path = tempfile.mkstemp(
                dir=report_path.parent,
                suffix=report_path.suffix,
                prefix=".fermi_tmp_",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                os.replace(tmp_path, str(report_path))
            except BaseException:
                # Clean up temp file on failure; original report is untouched.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            print(f"[FermiEstimation] Appended estimation section to {report_path}")

        except Exception as e:
            print(f"[FermiEstimation] Failed to append to report: {e}")

    def _append_warnings_to_report(
        self,
        report_path: Path,
        warnings_collector: ResearchWarnings,
    ) -> None:
        """Append a warnings section to the end of the report file.

        Only appends to markdown (.md) files. For other formats the warnings
        are still available via the ``warnings`` key in the result dict.
        """
        try:
            if not report_path.exists():
                return
            suffix = report_path.suffix.lower()
            if suffix != ".md":
                # For non-markdown formats, skip file modification but
                # warnings are still in the result dict.
                return

            language = getattr(self.config.research, "language", "en")
            section_text = warnings_collector.to_report_section(language)
            if not section_text:
                return

            import os
            import tempfile

            content = report_path.read_text(encoding="utf-8")
            content += section_text

            fd, tmp_path = tempfile.mkstemp(
                dir=report_path.parent,
                suffix=report_path.suffix,
                prefix=".warnings_tmp_",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                os.replace(tmp_path, str(report_path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            print(f"[Warnings] Appended {warnings_collector.count()} warning(s) to report")

        except Exception as e:
            print(f"[Warnings] Failed to append warnings to report: {e}")

    def _get_content_for_verification(self, session: ResearchSession) -> str:
        """Extract content from session for verification."""
        content_parts = []

        for section_num, section_data in session.section_contents.items():
            if section_num.startswith("_"):
                continue
            content_parts.append(section_data.get("content", ""))

        return "\n\n".join(content_parts)

    # ------------------------------------------------------------------
    # Live report (progressive writing while research runs)
    # ------------------------------------------------------------------

    def _build_live_sink(self, external_sink, session):
        """Combine the caller's sink (Web UI preview) with the Word COM
        sink when config.report.live_report_word is on. Returns None when
        no live output is wanted."""
        from .report.live_report import CompositeSink, WordComSink
        sinks = []
        if external_sink is not None:
            sinks.append(external_sink)
        if self.config.report.live_report_word:
            out_dir = self.config.report.output_dir / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            sinks.append(WordComSink(
                output_path=out_dir / f"report_live_{session.session_id}.docx",
                language=self.config.research.language,
            ))
        if not sinks:
            return None
        return CompositeSink(sinks)

    def _freeze_semantics(self, chapters, figure_collection,
                          extras=None) -> str:
        """Canonical semantic snapshot + hash at FREEZE time.

        ``extras`` carries semantic content stored outside the chapter
        map (V1 executive summary / key findings / recommendations) so
        the freeze covers EVERYTHING the reader will read."""
        from .report.semantic_manifest import (
            build_semantic_manifest, manifest_hash)
        manifest = build_semantic_manifest(chapters, figure_collection,
                                           extras=extras)
        digest = manifest_hash(manifest)
        self.semantic_manifest = manifest
        self.semantic_manifest_hash = digest
        print(f"[Freeze] semantic manifest hash: {digest[:16]}…")
        return digest

    def _check_semantics_at_output(self, chapters, figure_collection,
                                   freeze_hash, extras=None) -> str:
        """Recompute the manifest right before the final output.

        Any difference from the freeze-time hash means semantic content
        was generated or altered AFTER verification — a pipeline defect
        that is reported as a CRITICAL warning."""
        from .report.semantic_manifest import (
            build_semantic_manifest, manifest_hash)
        digest = manifest_hash(
            build_semantic_manifest(chapters, figure_collection,
                                    extras=extras))
        if freeze_hash and digest != freeze_hash:
            print(f"[Freeze] SEMANTIC DRIFT DETECTED: "
                  f"{freeze_hash[:16]}… -> {digest[:16]}…")
            ResearchWarnings.get_instance().add(
                ResearchWarnings.CRITICAL, "Freeze",
                "最終検証後に意味的内容が変更されました（semantic manifest "
                "hash不一致）。パイプライン不具合の可能性があります。",
            )
        return digest

    def _check_semantics_in_artifact(self, report_path) -> str:
        """Audit item D: verify the frozen semantics inside the ACTUAL
        saved artifact — re-hashing the in-memory object only proves the
        object didn't change, not that the renderer wrote it faithfully.

        Returns "pass", "fail" or "skipped:<reason>" and raises a
        CRITICAL warning on failure.
        """
        from .report.semantic_manifest import (
            extract_artifact_text, verify_frozen_in_artifact)
        manifest = getattr(self, "semantic_manifest", None)
        if not manifest:
            return "skipped:no_manifest"
        artifact_text = extract_artifact_text(report_path)
        if artifact_text is None:
            # unreadable format (e.g. PDF): SKIPPED is reported loudly,
            # never disguised as a pass
            print(f"[Freeze] artifact check skipped "
                  f"(unsupported format): {report_path}")
            return "skipped:unsupported_format"
        result = verify_frozen_in_artifact(manifest, artifact_text)
        if result["ok"]:
            print(f"[Freeze] artifact check passed "
                  f"({result['checked']} fragments verified in "
                  f"{Path(report_path).name})")
            return "pass"
        print(f"[Freeze] ARTIFACT CONTENT MISMATCH: "
              f"{len(result['missing'])}/{result['checked']} frozen "
              f"fragments missing from {report_path}")
        ResearchWarnings.get_instance().add(
            ResearchWarnings.CRITICAL, "Freeze",
            f"保存された成果物に凍結済み本文の一部が見つかりません"
            f"（{len(result['missing'])}断片欠落、例: "
            f"{result['missing'][0] if result['missing'] else ''}…）。"
            f"レンダリング段階の不具合の可能性があります。",
        )
        return "fail"

    @staticmethod
    def _emit_live_figures(live_sink, collection) -> None:
        """Emit generated charts/figures to the live report."""
        if live_sink is None or collection is None:
            return
        try:
            for fig in list(getattr(collection, "charts", []) or []) + \
                    list(getattr(collection, "figures", []) or []):
                path = getattr(fig, "image_path", None)
                if not path:
                    continue
                live_sink.on_figure(
                    getattr(fig, "section_id", "") or "",
                    str(path),
                    getattr(fig, "caption", "") or
                    getattr(fig, "title", "") or "",
                )
        except Exception as e:
            print(f"[LiveReport] figure emit failed: {e}")

    # ------------------------------------------------------------------
    # Finalization (common pipeline for V1 / V2 / V3 / manual mode)
    # ------------------------------------------------------------------

    def _build_finalization_runner(self, session, evidence_locker, query,
                                   requirements="", progress_callback=None,
                                   chapter_citation_callback=None):
        from .report.finalization_runner import FinalizationRunner
        return FinalizationRunner(
            evidence_locker=evidence_locker,
            session_contents=session.section_contents,
            research_plan=session.research_plan,
            query=query,
            requirements=requirements,
            language=self.config.research.language,
            llm_client=self.llm_client,
            eval_llm=self.stage_llm_clients.get("evaluation"),
            writing_llm=self.stage_llm_clients.get("writing"),
            search_client=self.search_client,
            report_config=self.config.report,
            research_config=self.config.research,
            output_dir=self.config.report.output_dir,
            session_id=session.session_id,
            progress_callback=progress_callback,
            chapter_citation_callback=chapter_citation_callback,
            verification_progress=getattr(self, "verification_progress",
                                          None),
        )

    def _warn_on_finalization_outcome(self, outcome) -> None:
        verdict = outcome["verdict"]
        if outcome["decision"] == "cancelled":
            ResearchWarnings.get_instance().add(
                ResearchWarnings.CRITICAL, "Finalize",
                "検証はユーザーによってキャンセルされました。本文は最後に"
                "検証された状態（または未検証）のままレンダリングされます。",
            )
        if outcome["decision"] == "timeout":
            ResearchWarnings.get_instance().add(
                ResearchWarnings.CRITICAL, "Finalize",
                "検証がタイムアウトしました（verification_timeout_seconds）。"
                "本文は最後に検証された状態のままレンダリングされます。",
            )
        if outcome["decision"] == "finalize_with_limitations":
            ResearchWarnings.get_instance().add(
                ResearchWarnings.HIGH, "Finalize",
                f"追加調査の上限内で解決できない論点が残ったため、限定表現へ"
                f"変更し「調査上の限界」を本文に記載しました"
                f"（未解決issue: {len(verdict.issues)}件）。",
            )
        if outcome.get("over_hard_max"):
            ResearchWarnings.get_instance().add(
                ResearchWarnings.CRITICAL, "Finalize",
                f"本文が上限文字数(hard_max_body_chars)を超過したまま圧縮"
                f"できませんでした（{verdict.metrics.actual_body_chars}字）。"
                f"このレポートは正常完了ではありません。",
            )
        if verdict.metrics.verification_failed:
            ResearchWarnings.get_instance().add(
                ResearchWarnings.CRITICAL, "Finalize",
                "最終本文の検証が実行できませんでした（主張抽出0件または"
                "全チャンク失敗）。本文は未検証のため取り扱いに注意して"
                "ください。",
            )

    def _run_finalization_loop(self, result, session, evidence_locker,
                               query, progress_callback=None,
                               requirements="", extra_chapters=None,
                               render_only=None):
        """Verify and finalize the exact body that will be rendered
        (V2 / V3 chapter form).

        Returns (verdict, verification_html_path). After this returns, the
        chapters inside `result` are FROZEN and already display-numbered:
        no LLM may modify them; only deterministic rendering (markdown
        assembly, DOCX/PDF/HTML conversion) is allowed downstream.
        """
        def chapter_citation_cb(sid, evidence):
            ch = result.chapters.get(sid)
            if ch is not None and evidence.url not in ch.citations:
                ch.citations.append(evidence.url)

        runner = self._build_finalization_runner(
            session, evidence_locker, query, requirements,
            progress_callback, chapter_citation_callback=chapter_citation_cb)
        if render_only:
            runner.render_only_sections.add(render_only)
        self.finalization_runner = runner
        self.claim_verifier = runner.claim_verifier

        chapters = {sid: ch.content for sid, ch in result.chapters.items()}
        for key, text in (extra_chapters or {}).items():
            if text and text.strip():
                chapters[key] = text

        outcome = runner.run(chapters)
        self.finalization_outcome = outcome

        # Freeze: write the verified, display-numbered text back into the
        # generation result (extras become chapters of their own)
        # Render-only sections (figure semantics) are NOT written back:
        # rendering uses the frozen collection itself.
        for sid, text in outcome["chapters"].items():
            if render_only and sid == render_only:
                continue
            if sid in result.chapters:
                result.chapters[sid].content = text
                result.chapters[sid].word_count = len(text)
            else:
                result.chapters[sid] = self._make_extra_chapter(sid, text)

        self._warn_on_finalization_outcome(outcome)
        return outcome["verdict"], outcome["html_path"]

    def _run_finalization_v1(self, session, evidence_locker, query,
                             requirements="", progress_callback=None,
                             figure_collection=None):
        """Common finalization for the V1 path: the session sections ARE
        the chapters; the frozen, display-numbered text is written back
        before the deterministic V1 render. Figure semantics join the
        verified body as a render-only section."""
        runner = self._build_finalization_runner(
            session, evidence_locker, query, requirements,
            progress_callback)
        self.finalization_runner = runner
        self.claim_verifier = runner.claim_verifier

        chapters = {
            sid: sdata.get("content", "")
            for sid, sdata in session.section_contents.items()
            if not sid.startswith("_") and sdata.get("content")
        }
        if figure_collection is not None:
            from .report.semantic_manifest import figure_semantics_markdown
            key = "付録D" if self.config.research.language == "ja" \
                else "Appendix D"
            fig_md = figure_semantics_markdown(
                figure_collection,
                language=self.config.research.language, section_id=key)
            if fig_md.strip():
                chapters[key] = fig_md
                runner.render_only_sections.add(key)
        if not chapters:
            return None, None

        outcome = runner.run(chapters)
        self.finalization_outcome = outcome

        for sid, text in outcome["chapters"].items():
            if sid in session.section_contents:
                session.section_contents[sid]["content"] = text

        self._warn_on_finalization_outcome(outcome)
        return outcome["verdict"], outcome["html_path"]

    def _make_extra_chapter(self, sid, text):
        """Wrap a verified extra section (Fermi / glossary / warnings)
        as a chapter so renderers treat it like any frozen chapter."""
        from .report.v2.generator import ChapterContent
        title = ""
        m = re.match(r"^##\s*(?:[^.\n]*\.\s*)?([^\n]+)", text.strip()) \
            if text else None
        if m:
            title = m.group(1).strip()
        return ChapterContent(
            section_number=sid,
            section_title=title or sid,
            content=text,
            word_count=len(text or ""),
            is_draft=False,
        )

    def _build_extra_chapters(self, generator, result, fermi_markdown,
                               figure_collection=None):
        """Body-bound extras (Fermi / glossary / warnings snapshot /
        FIGURE SEMANTICS) that must be part of the body BEFORE final
        verification.

        The figure-semantics chapter is verified but marked RENDER-ONLY:
        the finalization loop may flag issues on it, but never LLM-edits
        it — rendering always uses the frozen collection, so the verified
        text and the rendered figures cannot diverge.

        Returns (extra_chapters, flags): flags record which extras were
        included; flags["figures_key"] names the render-only section.
        """
        language = self.config.research.language
        ja = language == "ja"
        extras = {}
        flags = {}

        if figure_collection is not None:
            from .report.semantic_manifest import figure_semantics_markdown
            key = "付録D" if ja else "Appendix D"
            fig_md = figure_semantics_markdown(
                figure_collection, language=language, section_id=key)
            if fig_md.strip():
                extras[key] = fig_md
                flags["figures"] = True
                flags["figures_key"] = key

        if fermi_markdown and fermi_markdown.strip():
            num = "付録A" if ja else "Appendix A"
            title = "フェルミ推定" if ja else "Fermi Estimation"
            body = re.sub(r"^-{3,}\s*\n##[^\n]*\n", "",
                          fermi_markdown.strip())
            extras[num] = f"## {num}. {title}\n\n{body}"
            flags["fermi"] = True

        if (self.config.report.v2_include_glossary
                and getattr(getattr(result, "context", None),
                            "glossary", None)):
            glossary_md = ""
            build = getattr(generator, "build_glossary_markdown", None)
            if callable(build):
                try:
                    glossary_md = build(result) or ""
                except Exception as e:
                    print(f"[Finalize] glossary build failed: {e}")
            if glossary_md.strip():
                num = "付録B" if ja else "Appendix B"
                extras[num] = glossary_md
                flags["glossary"] = True

        collector = ResearchWarnings.get_instance()
        self._warnings_included_count = collector.count()
        if collector.has_warnings():
            num = "付録C" if ja else "Appendix C"
            extras[num] = collector.to_report_section(language).strip()
            flags["warnings"] = True

        return extras, flags

    def _late_warnings_section(self, collector, language) -> str:
        """Warnings raised AFTER finalization, rendered deterministically.

        The pre-finalization snapshot is already inside the verified body;
        only newer entries are appended here."""
        included = getattr(self, "_warnings_included_count", 0)
        entries = collector.to_dict_list()[included:]
        if not entries:
            return ""
        if language == "ja":
            lines = ["\n\n---\n", "## 処理中の警告（レポート確定後）", ""]
        else:
            lines = ["\n\n---\n", "## Processing Warnings (post-finalization)", ""]
        for w in entries:
            lines.append(f"- [{w.get('severity', '')}] "
                         f"{w.get('source', '')}: {w.get('message', '')}")
        return "\n".join(lines)

    def _enhance_sections_with_full_evidence(
        self,
        session: ResearchSession,
        evidence_locker: EvidenceLocker,
        query: str,
    ) -> ResearchSession:
        """
        Enhance report sections by reviewing ALL evidence across the entire locker.

        This method:
        1. Reviews all evidence in the locker (not just per-section)
        2. Identifies key numerical data, notable events, and important facts
        3. Re-accesses URLs that contain important data for precise extraction
        4. Regenerates section content with comprehensive evidence integration

        Args:
            session: Research session with section contents
            evidence_locker: Evidence locker with all collected evidence
            query: Original research query

        Returns:
            Enhanced ResearchSession with improved section contents
        """
        try:
            all_evidence = evidence_locker.get_all_evidence()
            if not all_evidence:
                print("[Evidence Review] No evidence available for enhancement")
                return session

            print(f"[Evidence Review] Reviewing {len(all_evidence)} evidence items across all sections")

            # Step 1: Build a comprehensive evidence summary
            evidence_summaries = []
            urls_with_key_data = []
            for ev in all_evidence:
                url = getattr(ev, 'url', '')
                title = getattr(ev, 'title', '')
                excerpt = getattr(ev, 'content_excerpt', '')
                section_ref = getattr(ev, 'section_reference', '')
                relevance = getattr(ev, 'relevance_score', 0)
                ev_type = getattr(ev, 'evidence_type', '')

                evidence_summaries.append({
                    "url": url,
                    "title": title,
                    "excerpt": excerpt[:300] if excerpt else "",
                    "section": section_ref,
                    "relevance": relevance,
                    "type": str(ev_type),
                })

                # Identify URLs likely to contain important numerical data
                if excerpt:
                    import re
                    has_numbers = bool(re.search(r'\d+[.,]?\d*\s*[%％億万兆ドル円]', excerpt))
                    has_year_data = bool(re.search(r'(19|20)\d{2}年?', excerpt))
                    if (has_numbers or has_year_data) and relevance and relevance >= 0.5:
                        urls_with_key_data.append({
                            "url": url,
                            "title": title,
                            "reason": "Contains key numerical data/dates",
                        })

            # Step 2: Re-access important URLs for precise data extraction
            re_extracted_data = {}
            if urls_with_key_data:
                print(f"[Evidence Review] Re-accessing {min(len(urls_with_key_data), 5)} URLs for precise data")
                for url_info in urls_with_key_data[:5]:  # Limit re-access
                    url = url_info["url"]
                    if not url or not url.startswith("http"):
                        continue
                    try:
                        page = self.search_client.get_page_content(url)
                        if page.text_content and len(page.text_content) > 100:
                            re_extracted_data[url] = {
                                "title": url_info["title"],
                                "content": page.text_content[:3000],
                            }
                            print(f"[Evidence Review] Re-extracted data from: {url[:60]}")
                    except Exception as e:
                        print(f"[Evidence Review] Failed to re-access {url[:60]}: {e}")

            # Step 3: Generate comprehensive evidence overview for LLM
            evidence_overview = self._build_evidence_overview(evidence_summaries, re_extracted_data)

            # Step 4: Enhance each section with full evidence context
            section_keys = [k for k in session.section_contents.keys() if not k.startswith("_")]
            if not section_keys:
                print("[Evidence Review] No sections to enhance")
                return session

            # Collect all current section contents for cross-reference
            all_sections_text = ""
            for key in section_keys:
                section_data = session.section_contents[key]
                title = section_data.get("title", key)
                content = section_data.get("content", "")
                all_sections_text += f"\n### Section {key}: {title}\n{content[:500]}...\n"

            for section_key in section_keys:
                section_data = session.section_contents[section_key]
                current_content = section_data.get("content", "")
                section_title = section_data.get("title", section_key)

                # Skip sections that already have substantial content
                if current_content and len(current_content) > 200:
                    # Still enhance with key data from re-accessed URLs
                    enhanced = self._enhance_section_content(
                        section_key=section_key,
                        section_title=section_title,
                        current_content=current_content,
                        evidence_overview=evidence_overview,
                        re_extracted_data=re_extracted_data,
                        all_sections_text=all_sections_text,
                        query=query,
                    )
                    if enhanced and len(enhanced) > len(current_content) * 0.8:
                        session.section_contents[section_key]["content"] = enhanced
                        print(f"[Evidence Review] Enhanced section {section_key}: {len(current_content)} -> {len(enhanced)} chars")
                else:
                    # Generate content for empty/short sections
                    generated = self._generate_section_from_evidence(
                        section_key=section_key,
                        section_title=section_title,
                        evidence_overview=evidence_overview,
                        re_extracted_data=re_extracted_data,
                        all_sections_text=all_sections_text,
                        query=query,
                    )
                    if generated and len(generated) > 50:
                        session.section_contents[section_key]["content"] = generated
                        print(f"[Evidence Review] Generated content for section {section_key}: {len(generated)} chars")

            print("[Evidence Review] Section enhancement complete")
            return session

        except Exception as e:
            print(f"[Evidence Review] Enhancement failed: {e}")
            return session

    def _build_evidence_overview(
        self,
        evidence_summaries: List[Dict[str, Any]],
        re_extracted_data: Dict[str, Dict[str, str]],
    ) -> str:
        """Build a comprehensive evidence overview string for LLM context."""
        parts = []

        # Key evidence items
        parts.append("=== Evidence Overview ===")
        for i, ev in enumerate(evidence_summaries[:30], 1):  # Top 30 evidence items
            parts.append(
                f"[{i}] {ev['title'][:80]} (section: {ev['section']}, relevance: {ev['relevance']:.1f})"
                f"\n    {ev['excerpt'][:200]}"
            )

        # Re-extracted precise data
        if re_extracted_data:
            parts.append("\n=== Re-extracted Key Data ===")
            for url, data in re_extracted_data.items():
                parts.append(f"\nSource: {data['title']}")
                parts.append(f"URL: {url}")
                parts.append(f"Data: {data['content'][:500]}")

        return "\n".join(parts)

    def _enhance_section_content(
        self,
        section_key: str,
        section_title: str,
        current_content: str,
        evidence_overview: str,
        re_extracted_data: Dict[str, Dict[str, str]],
        all_sections_text: str,
        query: str,
    ) -> Optional[str]:
        """Enhance existing section content with full evidence context."""
        try:
            lang = self.config.research.language
            if lang == "ja":
                prompt = f"""あなたはリサーチレポートのセクション内容を強化するアシスタントです。

【リサーチテーマ】
{query}

【現在のセクション】
セクション {section_key}: {section_title}

【現在の内容】
{current_content[:3000]}

【全エビデンスからの重要データ】
{evidence_overview[:4000]}

【他セクションの概要（重複回避用）】
{all_sections_text[:1500]}

【指示】
上記のエビデンスデータを活用して、このセクションの内容を強化してください。

重要ルール:
1. 具体的な数字、統計データ、年月日は正確に引用すること（要約せず直接使用）
2. 重要な出来事や数値データは具体的に記載すること
3. エビデンスに含まれる特徴的なデータポイントを見逃さないこと
4. 他のセクションと重複しないようにすること
5. 元の内容の構造を維持しつつ、情報を充実させること
6. 推測や仮定は明示すること

強化した本文のみを出力してください（JSON不要）:"""
            else:
                prompt = f"""You are enhancing a research report section with comprehensive evidence.

[RESEARCH TOPIC]
{query}

[CURRENT SECTION]
Section {section_key}: {section_title}

[CURRENT CONTENT]
{current_content[:3000]}

[KEY DATA FROM ALL EVIDENCE]
{evidence_overview[:4000]}

[OTHER SECTIONS OVERVIEW (for avoiding duplication)]
{all_sections_text[:1500]}

[INSTRUCTIONS]
Enhance this section using the evidence data above.

Key rules:
1. Use exact numbers, statistics, dates from evidence (don't summarize - use directly)
2. Include specific events and data points
3. Don't miss distinctive data points from evidence
4. Avoid duplication with other sections
5. Maintain original structure while enriching content
6. Mark speculation explicitly

Output only the enhanced text (no JSON):"""

            response = self.llm_client.generate(prompt)
            if response and response.content and len(response.content.strip()) > 50:
                return response.content.strip()
        except Exception as e:
            print(f"[Evidence Review] Section enhancement failed for {section_key}: {e}")
        return None

    def _generate_section_from_evidence(
        self,
        section_key: str,
        section_title: str,
        evidence_overview: str,
        re_extracted_data: Dict[str, Dict[str, str]],
        all_sections_text: str,
        query: str,
    ) -> Optional[str]:
        """Generate section content from full evidence when content is missing."""
        try:
            lang = self.config.research.language
            if lang == "ja":
                prompt = f"""あなたはリサーチレポートのセクションを執筆するアシスタントです。

【リサーチテーマ】
{query}

【執筆するセクション】
セクション {section_key}: {section_title}

【利用可能なエビデンス（全体）】
{evidence_overview[:5000]}

【他セクションの概要（重複回避用）】
{all_sections_text[:1500]}

【指示】
上記のエビデンスに基づいて、このセクションの詳細な内容を執筆してください。

重要ルール:
1. エビデンスに含まれる具体的な数字、統計データ、年月日を正確に使用すること
2. 重要なデータは要約せず直接引用すること
3. 特徴的な数字や出来事を網羅的に取り込むこと
4. 他のセクションと重複する内容は避けること
5. 論理的な構成で記述すること
6. エビデンスにない情報は含めないこと

本文のみを出力してください（JSON不要、見出し不要）:"""
            else:
                prompt = f"""You are writing a research report section from evidence.

[RESEARCH TOPIC]
{query}

[SECTION TO WRITE]
Section {section_key}: {section_title}

[AVAILABLE EVIDENCE (ALL)]
{evidence_overview[:5000]}

[OTHER SECTIONS (avoid duplication)]
{all_sections_text[:1500]}

[INSTRUCTIONS]
Write detailed content for this section based on the evidence above.

Key rules:
1. Use exact numbers, statistics, dates from evidence
2. Don't summarize key data - use it directly
3. Include distinctive data points comprehensively
4. Avoid content already covered in other sections
5. Write with logical structure
6. Don't include information not in the evidence

Output only the text (no JSON, no heading):"""

            response = self.llm_client.generate(prompt)
            if response and response.content and len(response.content.strip()) > 50:
                return response.content.strip()
        except Exception as e:
            print(f"[Evidence Review] Section generation failed for {section_key}: {e}")
        return None

    def _generate_figure_collection(
        self,
        session: ResearchSession,
        evidence_locker: EvidenceLocker,
    ):
        """ALL semantic figure work — BEFORE final verification.

        Numerical extraction, chart analysis, chart rendering and every
        LLM-generated title/caption/insight happen here, so the figure
        semantics can join the finalization body and be VERIFIED before
        the freeze. Returns (collection, generator) or (None, None).

        Uses intelligent chart analysis when enabled:
        1. Extract numerical data from evidence
        2. Calculate derived metrics (CAGR, growth rates)
        3. Analyze data for chart opportunities
        4. Generate charts with insights
        """
        import traceback

        figures_dir = self.config.report.output_dir / "reports" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Get proxy settings
        proxies = None
        if self.config.proxy.is_configured():
            proxies = self.config.proxy.get_proxies_dict()

        # Step 1-3: Extract numerical data and analyze for charts
        numerical_store = None
        chart_recommendations = []
        demoted_tables = []
        quality_rejection_summary = ""

        if self.config.report.numerical_extraction:
            try:
                numerical_store = self._extract_numerical_data(
                    session=session,
                    evidence_locker=evidence_locker,
                )

                # Step 2: Calculate derived metrics if enabled
                if self.config.report.derived_metrics and numerical_store:
                    calculator = DerivedMetricsCalculator(numerical_store)
                    derived = calculator.calculate_all()
                    numerical_store.add_many(derived)

                    # Fill missing data if enabled
                    if self.config.report.derived_fill_missing:
                        for series in numerical_store.get_time_series():
                            interpolated = calculator.interpolate_missing_years(series)
                            numerical_store.add_many(interpolated)

                # Step 3: Analyze for intelligent charts if enabled
                if self.config.report.intelligent_charts and numerical_store:
                    analyzer = ChartAnalyzer(
                        llm_client=self.llm_client if self.config.report.chart_insights else None,
                        max_workers=min(self.config.report.figure_max_workers,
                                        self.config.research.parallel_max_workers),
                        concurrency_limiter=getattr(self, "run_limits", None),
                        language=self.config.research.language,
                        min_confidence=self.config.report.numerical_min_confidence,
                        use_llm_analysis=self.config.report.chart_insights,
                        fill_missing_data=False,  # Already done above
                        max_charts_per_section=self.config.report.chart_max_per_section,
                    )

                    research_topic = session.query if session else ""
                    chart_recommendations = analyzer.analyze(
                        store=numerical_store,
                        research_topic=research_topic,
                    )
                    # Rejected-but-tabular candidates become tables instead;
                    # the rejection breakdown feeds the zero-figure warning
                    demoted_tables = list(analyzer.demoted_table_candidates)
                    quality_rejection_summary = (
                        analyzer.quality_gate.rejection_summary())
                    if quality_rejection_summary:
                        print(f"[AutoFigures] {quality_rejection_summary}")

                # Save numerical data store
                if numerical_store:
                    store_path = figures_dir / "numerical_data.json"
                    numerical_store.save_to_json(store_path)

            except Exception as e:
                print(f"[AutoFigures] Numerical data extraction/analysis failed: {e}")
                traceback.print_exc()
                ResearchWarnings.get_instance().add(
                    ResearchWarnings.MEDIUM,
                    "AutoFigures",
                    f"Numerical data extraction/analysis failed. "
                    f"Intelligent charts will not be generated; "
                    f"only basic figure/table extraction may still work. Error: {e}",
                )
                # Continue without numerical data — figure/table extraction can still work

        # Step 4: Create figure generator
        generator = FigureTableGenerator(
            llm_client=self.llm_client,
            output_dir=figures_dir,
            language=self.config.research.language,
            max_images_per_section=self.config.report.auto_figures_max_images,
            proxies=proxies,
            verify_ssl=self.config.proxy.verify_ssl,
            chart_library=self.config.report.chart_library,
            max_workers=min(self.config.report.figure_max_workers,
                                        self.config.research.parallel_max_workers),
                        concurrency_limiter=getattr(self, "run_limits", None),
        )

        # Step 5: Generate figures/tables/charts
        try:
            if chart_recommendations or demoted_tables:
                collection = generator.generate_from_recommendations(
                    session=session,
                    evidence_locker=evidence_locker,
                    recommendations=chart_recommendations,
                    include_images=self.config.report.auto_figures_include_images,
                    include_tables=self.config.report.auto_figures_include_tables,
                    demoted_tables=demoted_tables,
                )
            else:
                # Fallback to standard generation
                collection = generator.generate_figures_and_tables(
                    session=session,
                    evidence_locker=evidence_locker,
                    include_images=self.config.report.auto_figures_include_images,
                    include_tables=self.config.report.auto_figures_include_tables,
                    include_charts=self.config.report.auto_figures_include_charts,
                )
        except Exception as e:
            print(f"[AutoFigures] Figure/table generation failed: {e}")
            traceback.print_exc()
            ResearchWarnings.get_instance().add(
                ResearchWarnings.MEDIUM,
                "AutoFigures",
                f"Figure/table/chart generation failed. "
                f"Report will not contain any auto-generated visual elements. Error: {e}",
            )
            return None, None

        # live report: charts/figures become visible as soon as generated
        self._emit_live_figures(getattr(self, "live_sink", None), collection)

        # Guarantee layer: sections whose table extraction came up empty
        # still get a 主要数値一覧 table synthesized from the numerical store
        if self.config.report.auto_figures_include_tables and numerical_store:
            try:
                generator.add_numeric_summary_tables(collection, numerical_store)
            except Exception as e:
                print(f"[AutoFigures] numeric summary tables failed: {e}")

        # Report extraction results
        n_figures = len(collection.figures)
        n_tables = len(collection.tables)
        n_charts = len(collection.charts)
        total = n_figures + n_tables + n_charts

        # Table-zero visibility: tables silently absent (while charts exist)
        # was indistinguishable from "no tables wanted"
        if (self.config.report.auto_figures_include_tables
                and n_tables == 0 and total > 0):
            n_points = len(numerical_store.data_points) if numerical_store else 0
            ResearchWarnings.get_instance().add(
                ResearchWarnings.LOW,
                "AutoFigures",
                f"表は0件でした（図・チャートは{total}件生成）。HTML表の取得・"
                f"LLM表抽出・数値一覧の全てで表になるデータが見つかりません"
                f"でした（数値データ点: {n_points}）。",
            )

        print(f"[AutoFigures] Extraction complete: "
              f"{n_figures} figure(s), {n_tables} table(s), {n_charts} chart(s)")

        if total == 0:
            print("[AutoFigures] No figures, tables, or charts were extracted. "
                  "Skipping insertion into report.")
            n_points = len(numerical_store.data_points) if numerical_store else 0
            detail = f"品質検定の内訳 — {quality_rejection_summary}。" \
                if quality_rejection_summary else \
                "収集ソースに統計・数値情報が少ない可能性があります。"
            ResearchWarnings.get_instance().add(
                ResearchWarnings.MEDIUM,
                "AutoFigures",
                f"図表の自動生成は実行されましたが、0件でした"
                f"（抽出できた数値データ: {n_points}点、チャート推奨: "
                f"{len(chart_recommendations)}件）。{detail}"
                f"意味のない図（同値の羅列・年同士のプロット等）は品質検定で"
                f"自動的に除外されます。",
            )
            return None, None

        return collection, generator

    def _insert_figures_into_report(
        self,
        report_path: Path,
        collection,
        generator,
    ) -> Optional[Path]:
        """DETERMINISTIC insertion of the frozen figure collection.

        Runs AFTER the freeze: no LLM is involved — the images, captions
        and table cells were generated and verified before finalization;
        this step only places them into the rendered file.
        """
        import traceback

        if collection is None or generator is None:
            return None
        figures_dir = generator.output_dir

        # Step 6: Read the report content (text formats only — .docx/.pdf are
        # binary and their insertion paths below don't need the raw text)
        content = ""
        if report_path.suffix.lower() in ('.md', '.txt', '.html'):
            try:
                content = None
                encodings = ['utf-8', 'utf-8-sig', 'cp932', 'shift_jis', 'latin-1']
                for encoding in encodings:
                    try:
                        with open(report_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                if content is None:
                    with open(report_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
            except Exception as e:
                print(f"[AutoFigures] Failed to read report file {report_path}: {e}")
                return None

        # Step 7: Insert figures/tables into the report
        try:
            suffix = report_path.suffix.lower()

            if suffix == '.md':
                updated_content = generator.add_figures_to_markdown(content, collection)
                updated_path = report_path.parent / f"{report_path.stem}_with_figures.md"
                with open(updated_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
            elif suffix == '.docx':
                updated_path = generator.add_figures_to_docx(
                    docx_path=report_path,
                    collection=collection,
                )
                if not updated_path:
                    print(f"[AutoFigures] DOCX insertion failed; the report "
                          f"keeps its original content without figures.")
                    ResearchWarnings.get_instance().add(
                        ResearchWarnings.MEDIUM,
                        "AutoFigures",
                        f"{total}件の図表を生成しましたが、DOCXへの挿入に失敗"
                        f"しました。図表ファイル自体は {figures_dir} に保存"
                        f"されています。",
                    )
                    return None
            elif suffix in ('.pdf', '.html'):
                updated_path = generator.add_figures_to_pdf(
                    markdown_content=content,
                    collection=collection,
                    output_path=report_path.parent / f"{report_path.stem}_with_figures.pdf",
                )
                if not updated_path:
                    updated_path = report_path
            else:
                # Fallback: treat as markdown
                updated_content = generator.add_figures_to_markdown(content, collection)
                updated_path = report_path.parent / f"{report_path.stem}_with_figures.md"
                with open(updated_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)

            print(f"[AutoFigures] Report with figures saved to: {updated_path}")

        except Exception as e:
            print(f"[AutoFigures] Failed to insert figures into report: {e}")
            traceback.print_exc()
            return None

        # Export collection metadata
        try:
            collection_path = figures_dir / "figures_tables.json"
            generator.export_collection(collection, collection_path)
        except Exception as e:
            print(f"[AutoFigures] Failed to export collection metadata: {e}")
            # Non-fatal — the report was already saved

        return updated_path

    def _extract_numerical_data(
        self,
        session: ResearchSession,
        evidence_locker: EvidenceLocker,
    ) -> NumericalDataStore:
        """
        Extract numerical data from evidence for intelligent chart generation.

        Args:
            session: Research session with content
            evidence_locker: Evidence locker with sources

        Returns:
            NumericalDataStore with extracted data points
        """
        store = NumericalDataStore(
            research_topic=session.query if session else ""
        )

        # Numerical extraction is an evaluation-type task; route it to the
        # 'evaluation' stage LLM when configured (usually a cheaper/faster
        # model) instead of the default writing-grade model
        extraction_llm = self.stage_llm_clients.get("evaluation", self.llm_client)
        extractor = NumericalDataExtractor(
            llm_client=extraction_llm if self.config.report.numerical_llm_extraction else None,
            language=self.config.research.language,
            min_confidence=self.config.report.numerical_min_confidence,
            use_llm=self.config.report.numerical_llm_extraction,
            enable_unit_conversion=self.config.report.enable_unit_conversion,
            enable_pint=self.config.report.enable_pint,
        )

        # Extract from each evidence item.
        #
        # - Use the FULL extracted text when available. content_excerpt is a
        #   500-char summary snippet; feeding only that starved the extractor
        #   of numbers and typically yielded zero chartable data.
        # - Skip items whose text contains no digits (nothing to extract).
        # - Cap the number of items and run extractions in parallel: this was
        #   previously one sequential LLM call per evidence item, which alone
        #   could take hours on a large run.
        import concurrent.futures
        import re as _re

        MAX_NUMERIC_EVIDENCE = 80
        EXTRACT_WORKERS = max(1, min(
            self.config.report.figure_max_workers,
            self.config.research.parallel_max_workers))

        candidates = []
        for evidence in evidence_locker.get_all_evidence():
            text = (getattr(evidence, "extracted_text", "") or ""
                    ) or (evidence.content_excerpt or "")
            if not text or not _re.search(r"\d", text):
                continue
            candidates.append((evidence, text))

        skipped_no_digits = (
            len(evidence_locker.get_all_evidence()) - len(candidates))
        if len(candidates) > MAX_NUMERIC_EVIDENCE:
            # Keep the most relevant items when over the cap
            candidates.sort(
                key=lambda ct: getattr(ct[0], "relevance_score", 0.0) or 0.0,
                reverse=True,
            )
            print(f"[AutoFigures] capping numerical extraction to "
                  f"{MAX_NUMERIC_EVIDENCE} of {len(candidates)} evidence items")
            candidates = candidates[:MAX_NUMERIC_EVIDENCE]

        print(f"[AutoFigures] numerical extraction: {len(candidates)} evidence "
              f"items ({skipped_no_digits} skipped: no numbers), "
              f"{EXTRACT_WORKERS} parallel workers")

        def _section_for(evidence) -> str:
            if session and session.section_contents:
                for sec_id, sec_data in session.section_contents.items():
                    if sec_id.startswith("_"):
                        continue
                    if evidence.url in sec_data.get("sources", []):
                        return sec_id
            return ""

        def _extract_one(item):
            evidence, text = item
            source_reliability = 0.7
            if hasattr(evidence, 'quality_score') and evidence.quality_score:
                source_reliability = evidence.quality_score
            try:
                return extractor.extract_from_content(
                    content=text,
                    source_url=evidence.url,
                    source_title=evidence.title,
                    evidence_id=evidence.id,
                    section_id=_section_for(evidence),
                    source_reliability=source_reliability,
                    research_topic=session.query if session else "",
                )
            except Exception as e:
                print(f"[AutoFigures] numerical extraction failed for "
                      f"{evidence.url[:60]}: {e}")
                return []

        if candidates:
            workers = min(EXTRACT_WORKERS, len(candidates))
            if workers > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                    for i, data_points in enumerate(ex.map(_extract_one, candidates), 1):
                        store.add_many(data_points)
                        if i % 10 == 0 or i == len(candidates):
                            print(f"[AutoFigures] numerical extraction "
                                  f"{i}/{len(candidates)} done "
                                  f"({len(store.data_points)} data points)")
            else:
                for item in candidates:
                    store.add_many(_extract_one(item))

        # Also extract from section contents (may have synthesized data)
        if session and session.section_contents:
            for section_id, section_data in session.section_contents.items():
                if section_id.startswith("_"):
                    continue

                content = section_data.get("content", "")
                if not content:
                    continue

                data_points = extractor.extract_from_content(
                    content=content,
                    section_id=section_id,
                    source_reliability=0.8,  # Higher for synthesized content
                    research_topic=session.query if session else "",
                )

                store.add_many(data_points)

        return store

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
            # Use evidence id as the key and content_excerpt as the source text
            if evidence.id and evidence.content_excerpt:
                source_texts[evidence.id] = evidence.content_excerpt

        deep_think_results = {}
        targets = [
            (sid, sdata.get("content", ""))
            for sid, sdata in session.section_contents.items()
            if not sid.startswith("_") and sdata.get("content")
        ]

        # Handle empty session gracefully
        if not targets:
            if progress_callback:
                progress_callback("DeepThink: No sections to process", 90)
            return session, deep_think_results

        # Sections run CONCURRENTLY: DeepThink was fully sequential
        # (sections one by one, each with several serial LLM calls),
        # which dominated the runtime. The processor's validator/metrics
        # are stateful, so every worker gets its OWN processor instance.
        from .utils.concurrency import effective_workers
        max_workers = effective_workers(
            self.config.research.parallel_max_workers,
            self.config.deep_think.max_workers, len(targets))
        print(f"[DeepThink] processing {len(targets)} sections with "
              f"{max_workers} parallel worker(s)")

        def _process_one(section_num, content):
            if max_workers > 1:
                processor = DeepThinkProcessor(
                    llm_client=self.llm_client,
                    config=self._thinking_config,
                    language=self.config.research.language,
                )
            else:
                processor = self.deep_think_processor
            return processor.process(content=content,
                                     source_texts=source_texts)

        import concurrent.futures
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_one, sid, content): (sid, content)
                for sid, content in targets
            }
            for future in concurrent.futures.as_completed(futures):
                section_num, content = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"[DeepThink] section {section_num} failed: {e}")
                    ResearchWarnings.get_instance().add(
                        ResearchWarnings.MEDIUM, "DeepThink",
                        f"Section {section_num} DeepThink processing "
                        f"failed; the section keeps its original text. "
                        f"Error: {e}",
                    )
                    continue

                # Keep the section's full prose and APPEND DeepThink's
                # synthesized conclusion. Replacing the content wholesale
                # destroyed the section text: processed_content is a short
                # conclusion-only blob (produced by 「最終結論:」-style
                # prompts), which downstream report generation then
                # rendered as fragmentary 結論-labeled chapters.
                conclusion = (result.processed_content or "").strip()
                if conclusion and conclusion not in content:
                    session.section_contents[section_num]["content"] = (
                        content.rstrip() + "\n\n" + conclusion
                    )
                session.section_contents[section_num]["deep_think_conclusion"] = conclusion

                # Store result metrics
                deep_think_results[section_num] = {
                    "is_valid": result.is_valid,
                    "confidence": result.overall_confidence,
                    "metrics": result.metrics_summary,
                    "consistency": result.consistency_result.to_dict() if result.consistency_result else None,
                }

                # Add DeepThink info to section data
                session.section_contents[section_num]["deep_think"] = deep_think_results[section_num]

                completed += 1
                if progress_callback:
                    progress = 85 + (completed / len(targets) * 5)
                    progress_callback(
                        f"DeepThink: {completed}/{len(targets)} sections done",
                        progress)

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
    parallel_max_workers: int = 8,
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
    # Content filtering parameters
    content_filter_mode: str = "moderate",
    custom_blocked_domains: List[str] = None,
    custom_whitelisted_domains: List[str] = None,
    # Fast crawl mode parameters
    crawl_mode: str = "standard",
    fast_crawl_workers: int = 10,
    fast_crawl_batch_size: int = 5,
    # AI crawl mode parameters
    ai_crawl_max_total_pages: int = 15,
    ai_crawl_max_depth: int = 3,
    ai_crawl_site_depth: int = 2,
    ai_crawl_max_llm_calls: int = 25,
    ai_crawl_max_pages_per_domain: int = 5,
    ai_crawl_politeness_delay: float = 1.0,
    # Information source mode
    source_mode: str = "web",
    # Evidence importance / gap-fill parameters
    importance_threshold: float = 0.6,
    min_high_importance_sources: int = 2,
    max_gap_fill_rounds: int = 1,
    # Auto figure/table generation parameters
    auto_figures: bool = False,
    auto_figures_include_images: bool = True,
    auto_figures_include_tables: bool = True,
    auto_figures_include_charts: bool = True,
    auto_figures_max_images: int = 2,
    # Format strictness
    strict_format: bool = False,
    # Fermi estimation parameters
    fermi_estimation: bool = False,
    fermi_target_metrics: List[str] = None,
    fermi_auto_detect: bool = True,
    fermi_max_tree_depth: int = 4,
    fermi_max_leaf_nodes: int = 10,
    fermi_monte_carlo: int = 1000,
    fermi_validate: bool = True,
    fermi_include_sensitivity: bool = True,
    fermi_enable_sub_decomposition: bool = True,
    fermi_sub_decomposition_max_iterations: int = 3,
    fermi_sub_decomposition_confidence_threshold: float = 0.65,
    fermi_sub_decomposition_min_sensitivity_pct: float = 10.0,
    # V2 report generation parameters
    report_generator_version: str = "v1",
    v2_writing_style: str = "business",
    v2_target_audience: str = "business",
    v2_technical_level: int = 3,
    v2_enable_consistency_check: bool = True,
    v2_enable_two_phase: bool = True,
    v2_include_glossary: bool = True,
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
        parallel_max_workers: App-wide limit on simultaneous LLM/network
            operations (default 8, allowed 1-16; invalid values raise)
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
            (Note: This operates at the research/synthesis layer. When using V2 reports
            with v2_enable_two_phase=True, both can be active simultaneously —
            enhanced_synthesis improves information gathering quality, while two_phase
            improves report writing quality. They are complementary, not redundant.)
        max_queries_per_iteration: Max queries to execute per research iteration (default: 3)
        max_pages_per_query: Max pages to process per search query (default: 3)
        content_filter_mode: Content filter strictness ('strict', 'moderate', 'minimal', 'none')
        custom_blocked_domains: List of domains to block (ads, spam, etc.)
        custom_whitelisted_domains: List of domains to always allow
        crawl_mode: Crawl mode ('standard', 'fast_batch', 'fast_parallel',
            'aicrawl', 'ai_crawl_selenium')
        fast_crawl_workers: Max parallel workers for fast crawl mode
        fast_crawl_batch_size: Pages per batch in batch evaluation mode
        ai_crawl_max_total_pages: aicrawl - max pages fetched per section
        ai_crawl_max_depth: aicrawl - max link depth from search-result seeds
        ai_crawl_site_depth: aicrawl - max layers followed within one site
        ai_crawl_max_llm_calls: aicrawl - LLM decision call budget per section
        ai_crawl_max_pages_per_domain: aicrawl - max pages fetched per domain
        ai_crawl_politeness_delay: aicrawl - min seconds between same-domain fetches
        source_mode: Information sources: 'web' (default), 'local' (local
            documents only; requires additional_documents), 'hybrid' (both)
        importance_threshold: min importance score to count as high-importance
        min_high_importance_sources: below this, gap-fill re-search triggers
        max_gap_fill_rounds: max gap-fill re-search rounds per section
        auto_figures: Auto-generate figures/tables and embed in report
        auto_figures_include_images: Include images from web sources
        auto_figures_include_tables: Include extracted tables
        auto_figures_include_charts: Include generated charts
        auto_figures_max_images: Max images per section
        strict_format: If True, raise error instead of falling back to markdown
                      when DOCX generation fails (default: False)
        fermi_estimation: Enable Fermi estimation for quantitative metrics
        fermi_target_metrics: List of target metrics to estimate (empty = auto-detect)
        fermi_auto_detect: Auto-detect target metrics from research content
        fermi_max_tree_depth: Max depth for decomposition tree (1-6)
        fermi_max_leaf_nodes: Max leaf nodes in decomposition tree (2-20)
        fermi_monte_carlo: Number of Monte Carlo simulation iterations
        fermi_validate: Validate estimation results with LLM
        fermi_include_sensitivity: Include sensitivity analysis
        fermi_enable_sub_decomposition: Enable sub-decomposition for low-confidence leaves
        fermi_sub_decomposition_max_iterations: Max sub-decomposition iterations (0-10)
        fermi_sub_decomposition_confidence_threshold: Confidence threshold for sub-decomposition
        fermi_sub_decomposition_min_sensitivity_pct: Min sensitivity % to trigger sub-decomposition
        report_generator_version: Report generator version ('v1' or 'v2')
        v2_writing_style: V2 writing style ('formal', 'business', 'technical', 'executive', 'casual')
        v2_target_audience: V2 target audience ('expert', 'business', 'engineer', 'general', 'student')
        v2_technical_level: V2 technical level (1-5, 5 = most technical)
        v2_enable_consistency_check: V2 cross-chapter consistency check
        v2_enable_two_phase: V2 two-phase generation (draft + refinement)
        v2_include_glossary: V2 append glossary at end of report
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

        # With fast crawl mode (batch evaluation)
        result = run_research(
            "AI trends 2024",
            crawl_mode="fast_batch",  # Parallel crawl + batch LLM
            fast_crawl_workers=15,
            fast_crawl_batch_size=5,
        )

        # With fast crawl mode (parallel evaluation)
        result = run_research(
            "Market analysis",
            crawl_mode="fast_parallel",  # Parallel crawl + parallel LLM
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
        parallel_max_workers=parallel_max_workers,
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
        content_filter_mode=content_filter_mode,
        custom_blocked_domains=custom_blocked_domains,
        custom_whitelisted_domains=custom_whitelisted_domains,
        crawl_mode=crawl_mode,
        fast_crawl_workers=fast_crawl_workers,
        fast_crawl_batch_size=fast_crawl_batch_size,
        ai_crawl_max_total_pages=ai_crawl_max_total_pages,
        ai_crawl_max_depth=ai_crawl_max_depth,
        ai_crawl_site_depth=ai_crawl_site_depth,
        ai_crawl_max_llm_calls=ai_crawl_max_llm_calls,
        ai_crawl_max_pages_per_domain=ai_crawl_max_pages_per_domain,
        ai_crawl_politeness_delay=ai_crawl_politeness_delay,
        source_mode=source_mode,
        importance_threshold=importance_threshold,
        min_high_importance_sources=min_high_importance_sources,
        max_gap_fill_rounds=max_gap_fill_rounds,
        auto_figures=auto_figures,
        auto_figures_include_images=auto_figures_include_images,
        auto_figures_include_tables=auto_figures_include_tables,
        auto_figures_include_charts=auto_figures_include_charts,
        auto_figures_max_images=auto_figures_max_images,
        strict_format=strict_format,
        fermi_estimation=fermi_estimation,
        fermi_target_metrics=fermi_target_metrics,
        fermi_auto_detect=fermi_auto_detect,
        fermi_max_tree_depth=fermi_max_tree_depth,
        fermi_max_leaf_nodes=fermi_max_leaf_nodes,
        fermi_monte_carlo=fermi_monte_carlo,
        fermi_validate=fermi_validate,
        fermi_include_sensitivity=fermi_include_sensitivity,
        fermi_enable_sub_decomposition=fermi_enable_sub_decomposition,
        fermi_sub_decomposition_max_iterations=fermi_sub_decomposition_max_iterations,
        fermi_sub_decomposition_confidence_threshold=fermi_sub_decomposition_confidence_threshold,
        fermi_sub_decomposition_min_sensitivity_pct=fermi_sub_decomposition_min_sensitivity_pct,
        report_generator_version=report_generator_version,
        v2_writing_style=v2_writing_style,
        v2_target_audience=v2_target_audience,
        v2_technical_level=v2_technical_level,
        v2_enable_consistency_check=v2_enable_consistency_check,
        v2_enable_two_phase=v2_enable_two_phase,
        v2_include_glossary=v2_include_glossary,
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

    # Display warning summary
    warning_count = result.get("warning_count", 0)
    if warning_count > 0:
        language = config.research.language if hasattr(config.research, 'language') else "en"
        if language == "ja":
            print(f"\n[注意] 処理中に {warning_count} 件のフォールバック警告が発生しました。"
                  f"レポート末尾の「処理中の警告・注意事項」セクションを確認してください。")
        else:
            print(f"\n[NOTICE] {warning_count} fallback warning(s) occurred during processing. "
                  f"Check the 'Processing Warnings' section at the end of the report.")

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
    # Format strictness
    strict_format: bool = False,
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
        strict_format=strict_format,
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
            e.id: e.content_excerpt
            for e in evidence_locker.get_all_evidence()
            if e.id and e.content_excerpt
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

    # Evidence export happens AFTER finalization (below), so exports
    # include everything referenced by the finalized report.
    evidence_json = None
    evidence_csv = None

    # Verification: manual mode runs the SAME finalization pipeline as
    # run() (common chapter form -> verify -> decide -> act -> freeze).
    # There is no search client, so unresolved evidence problems end in
    # FINALIZE_WITH_LIMITATIONS instead of research rounds.
    verification_html = None
    verification_result = None

    def _finalize_manual(chapters, chapter_citation_callback=None,
                         figure_collection=None):
        from .report.finalization_runner import FinalizationRunner
        from .report.semantic_manifest import figure_semantics_markdown
        runner = FinalizationRunner(
            evidence_locker=evidence_locker,
            session_contents=session.section_contents,
            research_plan=session.research_plan,
            query=topic,
            requirements=requirements,
            language=config.research.language,
            llm_client=llm_client,
            search_client=None,
            report_config=config.report,
            research_config=config.research,
            output_dir=config.report.output_dir,
            session_id=session.session_id,
            chapter_citation_callback=chapter_citation_callback,
        )
        if figure_collection is not None:
            key = "付録D" if config.research.language == "ja" \
                else "Appendix D"
            fig_md = figure_semantics_markdown(
                figure_collection, language=config.research.language,
                section_id=key)
            if fig_md.strip():
                chapters[key] = fig_md
                runner.render_only_sections.add(key)
        return runner.run(chapters)

    # Generate report
    generator, version_tag = _create_report_generator(
        config=config,
        llm_client=llm_client,
        output_dir=config.report.output_dir / "reports",
    )

    # ALL semantic figure work happens BEFORE finalization (same as run())
    figure_collection = None
    if config.report.auto_figures:
        try:
            fig_generator = FigureTableGenerator(
                llm_client=llm_client,
                output_dir=config.report.output_dir / "reports" / "figures",
                language=config.research.language,
                proxies=config.proxy.get_proxies_dict(),
                verify_ssl=config.proxy.verify_ssl,
                max_workers=min(
                    config.report.figure_max_workers,
                    config.research.parallel_max_workers),
            )
            figure_collection = fig_generator.generate_figures_and_tables(
                session=session,
                evidence_locker=evidence_locker,
            )
        except Exception as e:
            if verbose:
                print(f"[AutoFigures] Failed: {e}. Continuing without figures.")
            figure_collection = None

    if version_tag == "v3":
        # V3: DOCX-native generation flow
        result = generator.generate_report(
            research_topic=topic,
            research_plan=session.research_plan,
            section_contents=session.section_contents,
        )

        if enable_verification:
            def _cb(sid, evidence):
                ch = result.chapters.get(sid)
                if ch is not None and evidence.url not in ch.citations:
                    ch.citations.append(evidence.url)
            chapters = {sid: ch.content
                        for sid, ch in result.chapters.items()}
            outcome = _finalize_manual(chapters, _cb,
                                       figure_collection=figure_collection)
            for sid, text in outcome["chapters"].items():
                if sid in result.chapters:
                    result.chapters[sid].content = text
                    result.chapters[sid].word_count = len(text)
            verification_result = outcome["verdict"]
            verification_html = outcome["html_path"]

        # Build warnings text
        warnings_text = ""
        warnings_collector = ResearchWarnings.get_instance()
        if warnings_collector.has_warnings():
            warnings_text = warnings_collector.to_report_section(
                config.research.language
            )

        output_dir = config.report.output_dir / "reports"
        report_path = generator.generate_and_save(
            result=result,
            output_dir=output_dir,
            filename=f"report_{session.session_id}",
            evidence_locker=evidence_locker,
            figure_collection=figure_collection,
            include_glossary=config.report.v2_include_glossary,
            warnings_text=warnings_text,
        )
    elif version_tag == "v2":
        # V2: Use new generation flow with consistency features
        result = generator.generate_report(
            research_topic=topic,
            research_plan=session.research_plan,
            section_contents=session.section_contents,
        )

        if enable_verification:
            def _cb(sid, evidence):
                ch = result.chapters.get(sid)
                if ch is not None and evidence.url not in ch.citations:
                    ch.citations.append(evidence.url)
            chapters = {sid: ch.content
                        for sid, ch in result.chapters.items()}
            outcome = _finalize_manual(chapters, _cb,
                                       figure_collection=figure_collection)
            for sid, text in outcome["chapters"].items():
                if sid in result.chapters:
                    result.chapters[sid].content = text
                    result.chapters[sid].word_count = len(text)
            verification_result = outcome["verdict"]
            verification_html = outcome["html_path"]

        # Generate final document (deterministic rendering of the
        # frozen chapters)
        final_doc = generator.generate_final_document(
            result,
            include_glossary=config.report.v2_include_glossary,
            evidence_locker=evidence_locker,
        )
        # Save to file in the configured format
        output_dir = config.report.output_dir / "reports"
        report_path = generator.save_report(
            markdown_content=final_doc,
            output_dir=output_dir,
            filename=f"report_{session.session_id}",
            format=config.report.format,
            strict_format=config.report.strict_format,
        )
    else:
        # V1: common finalization on the session sections, then the
        # deterministic V1 render (no post-verification length edits)
        if enable_verification:
            chapters = {
                sid: sdata.get("content", "")
                for sid, sdata in session.section_contents.items()
                if not sid.startswith("_") and sdata.get("content")
            }
            if chapters:
                outcome = _finalize_manual(
                    chapters, figure_collection=figure_collection)
                for sid, text in outcome["chapters"].items():
                    if sid in session.section_contents:
                        session.section_contents[sid]["content"] = text
                verification_result = outcome["verdict"]
                verification_html = outcome["html_path"]

        report_path = generator.generate_report(
            session=session,
            evidence_locker=evidence_locker,
            format=config.report.format,
            verification_result=None,
            target_pages=None if enable_verification else target_pages,
            target_characters=(None if enable_verification
                               else target_characters),
        )

    # Export evidence AFTER finalization: exports and the references
    # registry include everything the finalized report cites
    evidence_json = evidence_locker.export_to_json()
    evidence_csv = evidence_locker.export_to_csv()

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
    "ReportFormatError",
    "ManualTableOfContents",
]
