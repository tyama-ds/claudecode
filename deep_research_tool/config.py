"""
Configuration management for Deep Research Tool.

Supports both environment variables and direct parameter passing.
"""

import os
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass, field
from pathlib import Path


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class SearchMethod(str, Enum):
    """Supported web search methods."""
    DUCKDUCKGO = "duckduckgo"
    SELENIUM = "selenium"


class ReportFormat(str, Enum):
    """Supported report output formats."""
    DOCX = "docx"
    PDF = "pdf"
    MARKDOWN = "markdown"


@dataclass
class APIConfig:
    """Configuration for LLM API access."""

    provider: LLMProvider = LLMProvider.OPENAI

    # API Keys - can be set directly or via environment variables
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Model settings
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # API parameters
    temperature: float = 0.7
    max_tokens: int = 4096

    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    def get_active_api_key(self) -> Optional[str]:
        """Get the API key for the active provider."""
        if self.provider == LLMProvider.OPENAI:
            return self.openai_api_key
        return self.anthropic_api_key

    def get_active_model(self) -> str:
        """Get the model name for the active provider."""
        if self.provider == LLMProvider.OPENAI:
            return self.openai_model
        return self.anthropic_model


@dataclass
class SearchConfig:
    """Configuration for web search."""

    method: SearchMethod = SearchMethod.DUCKDUCKGO

    # DuckDuckGo settings
    max_results: int = 10
    region: str = "wt-wt"  # Worldwide
    safe_search: str = "moderate"

    # Selenium settings
    headless: bool = True
    browser: str = "chrome"
    page_load_timeout: int = 30
    implicit_wait: int = 10

    # Content extraction settings
    extract_images: bool = True
    max_images_per_page: int = 5


@dataclass
class ResearchConfig:
    """Configuration for research process."""

    # Research loop settings
    min_iterations: int = 3
    max_iterations: int = 10

    # Content settings
    max_content_length: int = 50000
    language: str = "ja"  # Default to Japanese

    # Evidence settings
    save_evidence: bool = True
    evidence_format: str = "json"

    # Extended mode settings (deep site crawling)
    extended_mode: bool = False
    crawl_max_pages: int = 10  # Max pages to crawl per site
    crawl_max_depth: int = 2   # Max link depth from seed URL
    crawl_max_sites: int = 3   # Max sites to crawl per search
    crawl_relevance_threshold: float = 0.3  # Min relevance to include page


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    format: ReportFormat = ReportFormat.MARKDOWN

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    include_images: bool = True
    include_citations: bool = True
    include_toc: bool = True

    # Output length control
    target_pages: Optional[int] = None  # Target page count (approximate)
    target_characters: Optional[int] = None  # Target character count

    # Evidence locker settings
    export_evidence_csv: bool = True
    export_evidence_json: bool = True


@dataclass
class Config:
    """Main configuration class combining all settings."""

    api: APIConfig = field(default_factory=APIConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    # Additional document settings
    additional_documents: List[Path] = field(default_factory=list)
    process_additional_documents: bool = False

    # Verification settings
    enable_verification: bool = True
    verification_strictness: str = "medium"  # low, medium, high

    # Logging
    verbose: bool = False
    log_file: Optional[Path] = None

    def __post_init__(self):
        """Ensure output directory exists."""
        self.report.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        config = cls()

        # Override with environment variables if present
        if os.getenv("LLM_PROVIDER"):
            config.api.provider = LLMProvider(os.getenv("LLM_PROVIDER"))

        if os.getenv("SEARCH_METHOD"):
            config.search.method = SearchMethod(os.getenv("SEARCH_METHOD"))

        if os.getenv("REPORT_FORMAT"):
            config.report.format = ReportFormat(os.getenv("REPORT_FORMAT"))

        if os.getenv("RESEARCH_ITERATIONS"):
            config.research.min_iterations = int(os.getenv("RESEARCH_ITERATIONS"))

        if os.getenv("OUTPUT_DIR"):
            config.report.output_dir = Path(os.getenv("OUTPUT_DIR"))

        return config

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Check API key availability
        if self.api.provider == LLMProvider.OPENAI and not self.api.openai_api_key:
            errors.append("OpenAI API key not configured")

        if self.api.provider == LLMProvider.ANTHROPIC and not self.api.anthropic_api_key:
            errors.append("Anthropic API key not configured")

        # Validate research iterations
        if self.research.min_iterations < 1:
            errors.append("Minimum research iterations must be at least 1")

        if self.research.max_iterations < self.research.min_iterations:
            errors.append("Maximum iterations must be >= minimum iterations")

        # Validate additional documents
        for doc in self.additional_documents:
            if not doc.exists():
                errors.append(f"Additional document not found: {doc}")

        return errors


def create_config(
    provider: str = "openai",
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    model: Optional[str] = None,
    search_method: str = "duckduckgo",
    research_iterations: int = 3,
    output_format: str = "markdown",
    output_dir: str = "./output",
    additional_documents: Optional[List[str]] = None,
    enable_verification: bool = True,
    verbose: bool = False,
    target_pages: Optional[int] = None,
    target_characters: Optional[int] = None,
    extended_mode: bool = False,
    crawl_max_pages: int = 10,
    crawl_max_depth: int = 2,
    crawl_max_sites: int = 3,
    **kwargs
) -> Config:
    """
    Factory function to create a Config object with common settings.

    Args:
        provider: LLM provider ('openai' or 'anthropic')
        openai_api_key: OpenAI API key (optional, uses env var if not provided)
        anthropic_api_key: Anthropic API key (optional, uses env var if not provided)
        model: Model name to use (optional, uses default for provider)
        search_method: Web search method ('duckduckgo' or 'selenium')
        research_iterations: Number of research iterations
        output_format: Report format ('docx', 'pdf', or 'markdown')
        output_dir: Output directory path
        additional_documents: List of additional document paths
        enable_verification: Enable hallucination verification
        verbose: Enable verbose logging
        target_pages: Target page count for output (approximate)
        target_characters: Target character count for output
        extended_mode: Enable extended mode (deep site crawling)
        crawl_max_pages: Max pages to crawl per site in extended mode
        crawl_max_depth: Max link depth from seed URL in extended mode
        crawl_max_sites: Max sites to crawl per search in extended mode
        **kwargs: Additional keyword arguments

    Returns:
        Configured Config object
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

    search_config = SearchConfig(
        method=SearchMethod(search_method),
        headless=kwargs.get("headless", True),
        max_results=kwargs.get("max_results", 10),
    )

    research_config = ResearchConfig(
        min_iterations=research_iterations,
        max_iterations=kwargs.get("max_iterations", research_iterations + 5),
        language=kwargs.get("language", "ja"),
        extended_mode=extended_mode,
        crawl_max_pages=crawl_max_pages,
        crawl_max_depth=crawl_max_depth,
        crawl_max_sites=crawl_max_sites,
    )

    report_config = ReportConfig(
        format=ReportFormat(output_format),
        output_dir=Path(output_dir),
        target_pages=target_pages,
        target_characters=target_characters,
    )

    docs = []
    if additional_documents:
        docs = [Path(doc) for doc in additional_documents]

    return Config(
        api=api_config,
        search=search_config,
        research=research_config,
        report=report_config,
        additional_documents=docs,
        process_additional_documents=bool(additional_documents),
        enable_verification=enable_verification,
        verbose=verbose,
    )
