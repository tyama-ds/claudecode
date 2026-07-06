"""
Configuration for Patent Research Tool.

Reuses APIConfig and ProxyConfig from deep_research_tool,
adds patent-specific configuration classes.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict

from deep_research_tool.config import (
    APIConfig,
    ProxyConfig,
    LLMProvider,
    ReportFormat,
    SearchMethod,
)


@dataclass
class PatentSearchConfig:
    """Configuration for patent database searches (Layer 1)."""

    # Patent databases to search (all enabled by default for parallel search)
    enable_google_patents: bool = True
    enable_jplatpat: bool = True
    enable_espacenet: bool = True

    # Patent search parameters
    patent_jurisdictions: List[str] = field(
        default_factory=lambda: ["JP", "US", "EP"]
    )
    ipc_codes: List[str] = field(default_factory=list)
    cpc_codes: List[str] = field(default_factory=list)
    date_range_start: Optional[str] = None  # e.g., "2020-01-01"
    date_range_end: Optional[str] = None
    patent_status: str = "all"  # "all", "granted", "pending", "expired"
    max_patents_per_query: int = 20

    # Patent family tracking
    track_families: bool = True
    family_jurisdictions: List[str] = field(
        default_factory=lambda: ["JP", "US", "EP", "WO"]
    )

    # Espacenet OPS API settings
    ops_consumer_key: Optional[str] = None
    ops_consumer_secret: Optional[str] = None

    # Search method for scraping-based clients
    search_method: SearchMethod = SearchMethod.DUCKDUCKGO

    def __post_init__(self):
        """Load OPS credentials from environment if not provided."""
        if self.ops_consumer_key is None:
            self.ops_consumer_key = os.getenv("OPS_CONSUMER_KEY")
        if self.ops_consumer_secret is None:
            self.ops_consumer_secret = os.getenv("OPS_CONSUMER_SECRET")


@dataclass
class AuxiliarySearchConfig:
    """Configuration for secondary/tertiary search layers."""

    # Layer 2: Academic/technical paper search
    enable_academic_search: bool = True
    academic_sources: List[str] = field(
        default_factory=lambda: ["cinii", "jstage", "google_scholar"]
    )
    max_papers_per_trigger: int = 5
    technical_term_threshold: float = 0.6  # Confidence for triggering

    # Layer 2: Examination documents
    enable_examination_search: bool = True
    max_examination_docs: int = 5

    # Layer 3: Business evidence search
    enable_business_search: bool = True
    business_term_threshold: float = 0.5
    business_keywords: List[str] = field(
        default_factory=lambda: [
            "市場規模", "売上", "revenue", "market size",
            "シェア", "市場シェア", "market share",
            "成長率", "growth rate", "CAGR",
        ]
    )
    max_business_sources: int = 5


@dataclass
class PatentReportConfig:
    """Configuration for patent research reports."""

    format: ReportFormat = ReportFormat.MARKDOWN
    output_dir: Path = field(
        default_factory=lambda: Path("./output/patent_research")
    )

    # Report sections toggles
    generate_claim_chart: bool = True
    generate_landscape: bool = True
    generate_prior_art_summary: bool = True
    generate_citation_network: bool = False

    # Content options
    include_original_claims: bool = True
    include_ipc_analysis: bool = True
    include_auxiliary_evidence: bool = True
    include_examination_history: bool = True

    # Claim chart settings
    claim_chart_max_patents: int = 10
    claim_chart_detail_level: str = "detailed"  # "summary", "detailed", "comprehensive"

    # Output length control
    target_pages: Optional[int] = None
    target_characters: Optional[int] = None


@dataclass
class PatentResearchConfig:
    """Main configuration for Patent Research Tool."""

    # Reused from deep_research_tool
    api: APIConfig = field(default_factory=APIConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)

    # Patent-specific configs
    patent_search: PatentSearchConfig = field(default_factory=PatentSearchConfig)
    auxiliary: AuxiliarySearchConfig = field(default_factory=AuxiliarySearchConfig)
    report: PatentReportConfig = field(default_factory=PatentReportConfig)

    # General settings
    language: str = "ja"
    verbose: bool = False
    log_file: Optional[Path] = None

    # Research iteration settings
    min_iterations: int = 2
    max_iterations: int = 5
    max_queries_per_iteration: int = 3

    def __post_init__(self):
        """Ensure output directory exists."""
        self.report.output_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Check API key
        if self.api.provider == LLMProvider.OPENAI and not self.api.openai_api_key:
            errors.append("OpenAI API key not configured")
        if self.api.provider == LLMProvider.ANTHROPIC and not self.api.anthropic_api_key:
            errors.append("Anthropic API key not configured")

        # Check Espacenet OPS credentials if enabled
        if self.patent_search.enable_espacenet:
            if not self.patent_search.ops_consumer_key:
                errors.append(
                    "Espacenet OPS consumer key not configured "
                    "(set OPS_CONSUMER_KEY env var or disable espacenet)"
                )

        return errors


def create_patent_config(
    provider: str = "openai",
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    model: Optional[str] = None,
    language: str = "ja",
    output_format: str = "markdown",
    output_dir: str = "./output/patent_research",
    verbose: bool = False,
    log_file: Optional[str] = None,
    # Patent search settings
    enable_google_patents: bool = True,
    enable_jplatpat: bool = True,
    enable_espacenet: bool = True,
    patent_jurisdictions: Optional[List[str]] = None,
    ipc_codes: Optional[List[str]] = None,
    cpc_codes: Optional[List[str]] = None,
    date_range_start: Optional[str] = None,
    date_range_end: Optional[str] = None,
    max_patents_per_query: int = 20,
    track_families: bool = True,
    ops_consumer_key: Optional[str] = None,
    ops_consumer_secret: Optional[str] = None,
    # Auxiliary search settings
    enable_academic_search: bool = True,
    academic_sources: Optional[List[str]] = None,
    enable_examination_search: bool = True,
    enable_business_search: bool = True,
    # Report settings
    generate_claim_chart: bool = True,
    generate_landscape: bool = True,
    generate_prior_art_summary: bool = True,
    include_original_claims: bool = True,
    include_auxiliary_evidence: bool = True,
    claim_chart_detail_level: str = "detailed",
    target_pages: Optional[int] = None,
    target_characters: Optional[int] = None,
    # Research settings
    min_iterations: int = 2,
    max_iterations: int = 5,
    # Proxy settings
    http_proxy: Optional[str] = None,
    https_proxy: Optional[str] = None,
    verify_ssl: bool = True,
    **kwargs,
) -> PatentResearchConfig:
    """
    Factory function to create a PatentResearchConfig.

    Args:
        provider: LLM provider ('openai' or 'anthropic')
        openai_api_key: OpenAI API key
        anthropic_api_key: Anthropic API key
        model: Model name
        language: Output language (default: 'ja')
        output_format: Report format ('markdown', 'docx', 'pdf', 'html')
        output_dir: Output directory path
        verbose: Enable verbose logging
        log_file: Path to log file
        enable_google_patents: Enable Google Patents search
        enable_jplatpat: Enable J-PlatPat search
        enable_espacenet: Enable Espacenet search
        patent_jurisdictions: Jurisdictions to search (e.g., ["JP", "US", "EP"])
        ipc_codes: IPC codes to focus on
        cpc_codes: CPC codes to focus on
        date_range_start: Start date for patent search
        date_range_end: End date for patent search
        max_patents_per_query: Maximum patents per query
        track_families: Enable patent family tracking
        ops_consumer_key: Espacenet OPS API consumer key
        ops_consumer_secret: Espacenet OPS API consumer secret
        enable_academic_search: Enable academic paper search
        academic_sources: Academic sources to search
        enable_examination_search: Enable examination document search
        enable_business_search: Enable business evidence search
        generate_claim_chart: Generate claim chart in report
        generate_landscape: Generate technology landscape in report
        generate_prior_art_summary: Generate prior art summary in report
        include_original_claims: Include original claims in report
        include_auxiliary_evidence: Include auxiliary evidence in report
        claim_chart_detail_level: Detail level for claim chart
        target_pages: Target page count for report
        target_characters: Target character count for report
        min_iterations: Minimum research iterations
        max_iterations: Maximum research iterations
        http_proxy: HTTP proxy URL
        https_proxy: HTTPS proxy URL
        verify_ssl: Verify SSL certificates

    Returns:
        Configured PatentResearchConfig object
    """
    api_config = APIConfig(
        provider=LLMProvider(provider),
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
    )
    if model:
        if provider == "openai":
            api_config.openai_model = model
        else:
            api_config.anthropic_model = model

    proxy_config = ProxyConfig(
        http_proxy=http_proxy,
        https_proxy=https_proxy,
        verify_ssl=verify_ssl,
    )

    patent_search_config = PatentSearchConfig(
        enable_google_patents=enable_google_patents,
        enable_jplatpat=enable_jplatpat,
        enable_espacenet=enable_espacenet,
        patent_jurisdictions=patent_jurisdictions or ["JP", "US", "EP"],
        ipc_codes=ipc_codes or [],
        cpc_codes=cpc_codes or [],
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        max_patents_per_query=max_patents_per_query,
        track_families=track_families,
        ops_consumer_key=ops_consumer_key,
        ops_consumer_secret=ops_consumer_secret,
    )

    auxiliary_config = AuxiliarySearchConfig(
        enable_academic_search=enable_academic_search,
        academic_sources=academic_sources or ["cinii", "jstage", "google_scholar"],
        enable_examination_search=enable_examination_search,
        enable_business_search=enable_business_search,
    )

    report_config = PatentReportConfig(
        format=ReportFormat(output_format),
        output_dir=Path(output_dir),
        generate_claim_chart=generate_claim_chart,
        generate_landscape=generate_landscape,
        generate_prior_art_summary=generate_prior_art_summary,
        include_original_claims=include_original_claims,
        include_auxiliary_evidence=include_auxiliary_evidence,
        claim_chart_detail_level=claim_chart_detail_level,
        target_pages=target_pages,
        target_characters=target_characters,
    )

    return PatentResearchConfig(
        api=api_config,
        proxy=proxy_config,
        patent_search=patent_search_config,
        auxiliary=auxiliary_config,
        report=report_config,
        language=language,
        verbose=verbose,
        log_file=Path(log_file) if log_file else None,
        min_iterations=min_iterations,
        max_iterations=max_iterations,
    )
