"""
Main module for Deep Research Tool.

This module provides the main interface for conducting automated research.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

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
        Tuple of (generator, is_v2)
    """
    output_dir = output_dir or config.report.output_dir / "reports"

    if config.report.generator_version == ReportGeneratorVersion.V2:
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
            language=config.research.language,
        )
        return generator, True
    else:
        # Create V1 generator
        generator = ReportGenerator(
            output_dir=output_dir,
            include_toc=config.report.include_toc,
            include_citations=config.report.include_citations,
            include_images=config.report.include_images,
            language=config.research.language,
        )
        return generator, False


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
        self._setup_logging()

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
            kwargs["implicit_wait"] = self.config.search.implicit_wait
        elif self.config.search.method == SearchMethod.DUCKDUCKGO:
            kwargs["region"] = self.config.search.region
            kwargs["safe_search"] = self.config.search.safe_search

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

        # Initialize researcher with content filter
        content_filter = self._create_content_filter()

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
            fast_crawl_batch_size=self.config.research.fast_crawl_batch_size,
            multilingual_config=self.config.multilingual if self.config.multilingual.enabled else None,
            max_content_length=self.config.research.max_content_length,
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

        # Export evidence (respecting save_evidence and evidence_format settings)
        evidence_json = None
        evidence_csv = None
        if self.config.research.save_evidence:
            evidence_format = self.config.research.evidence_format
            if evidence_format in ("json", "both"):
                evidence_json = evidence_locker.export_to_json()
            if evidence_format in ("csv", "both"):
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

        # Enhance section contents with full evidence review before report generation
        if progress_callback:
            progress_callback("Reviewing all evidence for report enhancement...", 92)

        session = self._enhance_sections_with_full_evidence(
            session=session,
            evidence_locker=evidence_locker,
            query=query,
        )

        # Generate report
        if progress_callback:
            progress_callback("Generating report...", 95)

        generator, is_v2 = _create_report_generator(
            config=self.config,
            llm_client=self.llm_client,
            output_dir=self.config.report.output_dir / "reports",
        )
        self.report_generator = generator

        if is_v2:
            # V2: Use new generation flow
            result = generator.generate_report(
                research_topic=query,
                research_plan=session.research_plan,
                section_contents=session.section_contents,
            )
            # Generate final document
            final_doc = generator.generate_final_document(
                result,
                include_glossary=self.config.report.v2_include_glossary,
            )
            # Save to file in the configured format
            output_dir = self.config.report.output_dir / "reports"
            report_path = generator.save_report(
                markdown_content=final_doc,
                output_dir=output_dir,
                filename=f"report_{session.session_id}",
                format=self.config.report.format,
                strict_format=self.config.report.strict_format,
            )
        else:
            # V1: Original generation flow
            report_path = generator.generate_report(
                session=session,
                evidence_locker=evidence_locker,
                format=self.config.report.format,
                verification_result=verification_result,
                target_pages=self.config.report.target_pages,
                target_characters=self.config.report.target_characters,
            )

        # Auto figure/table generation (if enabled)
        figures_report_path = None
        if self.config.report.auto_figures:
            if progress_callback:
                progress_callback("Generating figures and tables...", 97)

            figures_report_path = self._auto_generate_figures(
                report_path=Path(report_path),
                session=session,
                evidence_locker=evidence_locker,
            )

        # Clean up search client if selenium
        if hasattr(self.search_client, 'close'):
            self.search_client.close()

        # Get token usage statistics
        token_stats = get_token_stats()

        return {
            "session_id": session.session_id,
            "report_path": str(figures_report_path or report_path),
            "report_path_original": str(report_path),
            "figures_report_path": str(figures_report_path) if figures_report_path else None,
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

    def _auto_generate_figures(
        self,
        report_path: Path,
        session: ResearchSession,
        evidence_locker: EvidenceLocker,
    ) -> Optional[Path]:
        """
        Auto-generate figures and tables and embed them into the report.

        Uses intelligent chart analysis when enabled:
        1. Extract numerical data from evidence
        2. Calculate derived metrics (CAGR, growth rates)
        3. Analyze data for chart opportunities
        4. Generate charts with insights

        Args:
            report_path: Path to the generated report file
            session: Research session with content
            evidence_locker: Evidence locker with sources

        Returns:
            Path to the updated report with figures, or None if failed
        """
        import traceback

        figures_dir = report_path.parent / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Get proxy settings
        proxies = None
        if self.config.proxy.is_configured():
            proxies = self.config.proxy.get_proxies_dict()

        # Step 1-3: Extract numerical data and analyze for charts
        numerical_store = None
        chart_recommendations = []

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

                # Save numerical data store
                if numerical_store:
                    store_path = figures_dir / "numerical_data.json"
                    numerical_store.save_to_json(store_path)

            except Exception as e:
                print(f"[AutoFigures] Numerical data extraction/analysis failed: {e}")
                traceback.print_exc()
                # Continue without numerical data — figure/table extraction can still work

        # Step 4: Create figure generator
        generator = FigureTableGenerator(
            llm_client=self.llm_client,
            output_dir=figures_dir,
            language=self.config.research.language,
            max_images_per_section=self.config.report.auto_figures_max_images,
            proxies=proxies,
            verify_ssl=self.config.proxy.verify_ssl,
        )

        # Step 5: Generate figures/tables/charts
        try:
            if chart_recommendations:
                collection = generator.generate_from_recommendations(
                    session=session,
                    evidence_locker=evidence_locker,
                    recommendations=chart_recommendations,
                    include_images=self.config.report.auto_figures_include_images,
                    include_tables=self.config.report.auto_figures_include_tables,
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
            return None

        # Report extraction results
        n_figures = len(collection.figures)
        n_tables = len(collection.tables)
        n_charts = len(collection.charts)
        total = n_figures + n_tables + n_charts

        print(f"[AutoFigures] Extraction complete: "
              f"{n_figures} figure(s), {n_tables} table(s), {n_charts} chart(s)")

        if total == 0:
            print("[AutoFigures] No figures, tables, or charts were extracted. "
                  "Skipping insertion into report.")
            return None

        # Step 6: Read the report content
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
                    updated_path = report_path
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

        extractor = NumericalDataExtractor(
            llm_client=self.llm_client if self.config.report.numerical_llm_extraction else None,
            language=self.config.research.language,
            min_confidence=self.config.report.numerical_min_confidence,
            use_llm=self.config.report.numerical_llm_extraction,
            enable_unit_conversion=self.config.report.enable_unit_conversion,
            enable_pint=self.config.report.enable_pint,
        )

        # Extract from each evidence item
        for evidence in evidence_locker.get_all_evidence():
            if not evidence.content_excerpt:
                continue

            # Get section ID from evidence if available
            section_id = ""
            if session and session.section_contents:
                # Try to match evidence to section
                for sec_id, sec_data in session.section_contents.items():
                    if sec_id.startswith("_"):
                        continue
                    sources = sec_data.get("sources", [])
                    if evidence.url in [s.get("url", "") for s in sources]:
                        section_id = sec_id
                        break

            # Determine source reliability from verification if available
            source_reliability = 0.7  # Default
            if hasattr(evidence, 'quality_score') and evidence.quality_score:
                source_reliability = evidence.quality_score

            data_points = extractor.extract_from_content(
                content=evidence.content_excerpt,
                source_url=evidence.url,
                source_title=evidence.title,
                evidence_id=evidence.id,
                section_id=section_id,
                source_reliability=source_reliability,
                research_topic=session.query if session else "",
            )

            store.add_many(data_points)

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
    # Content filtering parameters
    content_filter_mode: str = "moderate",
    custom_blocked_domains: List[str] = None,
    custom_whitelisted_domains: List[str] = None,
    # Fast crawl mode parameters
    crawl_mode: str = "standard",
    fast_crawl_workers: int = 10,
    fast_crawl_batch_size: int = 5,
    # Auto figure/table generation parameters
    auto_figures: bool = False,
    auto_figures_include_images: bool = True,
    auto_figures_include_tables: bool = True,
    auto_figures_include_charts: bool = True,
    auto_figures_max_images: int = 2,
    # Format strictness
    strict_format: bool = False,
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
        content_filter_mode: Content filter strictness ('strict', 'moderate', 'minimal', 'none')
        custom_blocked_domains: List of domains to block (ads, spam, etc.)
        custom_whitelisted_domains: List of domains to always allow
        crawl_mode: Crawl mode for performance ('standard', 'fast_batch', 'fast_parallel')
        fast_crawl_workers: Max parallel workers for fast crawl mode
        fast_crawl_batch_size: Pages per batch in batch evaluation mode
        auto_figures: Auto-generate figures/tables and embed in report
        auto_figures_include_images: Include images from web sources
        auto_figures_include_tables: Include extracted tables
        auto_figures_include_charts: Include generated charts
        auto_figures_max_images: Max images per section
        strict_format: If True, raise error instead of falling back to markdown
                      when DOCX generation fails (default: False)
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
        auto_figures=auto_figures,
        auto_figures_include_images=auto_figures_include_images,
        auto_figures_include_tables=auto_figures_include_tables,
        auto_figures_include_charts=auto_figures_include_charts,
        auto_figures_max_images=auto_figures_max_images,
        strict_format=strict_format,
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
    generator, is_v2 = _create_report_generator(
        config=config,
        llm_client=llm_client,
        output_dir=config.report.output_dir / "reports",
    )

    if is_v2:
        # V2: Use new generation flow with consistency features
        result = generator.generate_report(
            research_topic=topic,
            research_plan=session.research_plan,
            section_contents=session.section_contents,
        )
        # Generate final document
        final_doc = generator.generate_final_document(
            result,
            include_glossary=config.report.v2_include_glossary,
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
        # V1: Original generation flow
        report_path = generator.generate_report(
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
    "ReportFormatError",
    "ManualTableOfContents",
]
