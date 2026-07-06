"""
Configuration management for Deep Research Tool.

Supports both environment variables and direct parameter passing.
"""

import os
from enum import Enum
from typing import Any, Optional, List, Dict
from dataclasses import dataclass, field
from pathlib import Path


# Language to region mapping for multilingual search
LANGUAGE_REGION_MAP: Dict[str, Dict[str, str]] = {
    "ja": {"region": "jp-jp", "name": "Japanese", "native": "日本語"},
    "en": {"region": "us-en", "name": "English", "native": "English"},
    "zh": {"region": "cn-zh", "name": "Chinese", "native": "中文"},
    "ko": {"region": "kr-kr", "name": "Korean", "native": "한국어"},
    "de": {"region": "de-de", "name": "German", "native": "Deutsch"},
    "fr": {"region": "fr-fr", "name": "French", "native": "Français"},
    "es": {"region": "es-es", "name": "Spanish", "native": "Español"},
    "pt": {"region": "br-pt", "name": "Portuguese", "native": "Português"},
    "ru": {"region": "ru-ru", "name": "Russian", "native": "Русский"},
    "it": {"region": "it-it", "name": "Italian", "native": "Italiano"},
    "nl": {"region": "nl-nl", "name": "Dutch", "native": "Nederlands"},
    "pl": {"region": "pl-pl", "name": "Polish", "native": "Polski"},
    "ar": {"region": "xa-ar", "name": "Arabic", "native": "العربية"},
    "hi": {"region": "in-en", "name": "Hindi", "native": "हिन्दी"},
    "th": {"region": "th-th", "name": "Thai", "native": "ไทย"},
    "vi": {"region": "vn-vi", "name": "Vietnamese", "native": "Tiếng Việt"},
}


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class LocalLLMBackend(str, Enum):
    """Supported local LLM backends."""
    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENAI_COMPATIBLE = "openai_compatible"


class SearchMethod(str, Enum):
    """Supported web search methods."""
    DUCKDUCKGO = "duckduckgo"
    SELENIUM = "selenium"


class ReportFormat(str, Enum):
    """Supported report output formats."""
    DOCX = "docx"
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"


class ReportGeneratorVersion(str, Enum):
    """Report generator version selection."""
    V1 = "v1"  # Original report generator
    V2 = "v2"  # Enhanced with consistency features
    V3 = "v3"  # DOCX-native generator (python-docx direct API)


@dataclass
class ProxyConfig:
    """Configuration for HTTP proxy settings."""

    # Proxy URL (e.g., "http://proxy.example.com:8080")
    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None

    # Authentication (if required)
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None

    # SSL verification
    verify_ssl: bool = True

    def __post_init__(self):
        """Load proxy settings from environment if not provided."""
        if self.http_proxy is None:
            self.http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        if self.https_proxy is None:
            self.https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")

    def get_proxies_dict(self) -> Optional[dict]:
        """Get proxies dict for requests library."""
        if not self.http_proxy and not self.https_proxy:
            return None

        proxies = {}
        if self.http_proxy:
            proxies["http"] = self._add_auth_to_url(self.http_proxy)
        if self.https_proxy:
            proxies["https"] = self._add_auth_to_url(self.https_proxy)

        return proxies if proxies else None

    def _add_auth_to_url(self, url: str) -> str:
        """Add authentication to proxy URL if credentials provided."""
        if not self.proxy_username or not self.proxy_password:
            return url

        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url)
        netloc = f"{self.proxy_username}:{self.proxy_password}@{parsed.hostname}"
        if parsed.port:
            netloc += f":{parsed.port}"

        return urlunparse((parsed.scheme, netloc, parsed.path,
                          parsed.params, parsed.query, parsed.fragment))

    def is_configured(self) -> bool:
        """Check if any proxy is configured."""
        return bool(self.http_proxy or self.https_proxy)


@dataclass
class APIConfig:
    """Configuration for LLM API access."""

    provider: LLMProvider = LLMProvider.OPENAI

    # API Keys - can be set directly or via environment variables
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Custom API endpoints (None = official endpoint). Needed when requests
    # must go through a corporate API gateway / OpenAI-compatible server.
    openai_base_url: Optional[str] = None  # e.g., "https://gateway.example.com/v1"
    anthropic_base_url: Optional[str] = None

    # Model settings (updated with GPT-5 series)
    openai_model: str = "gpt-5-mini"
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Local LLM settings
    local_model: str = "llama3.1:8b"
    local_backend: LocalLLMBackend = LocalLLMBackend.OLLAMA
    local_base_url: Optional[str] = None  # e.g., "http://localhost:11434"
    local_api_key: Optional[str] = None  # Optional auth for local servers

    # API parameters
    temperature: float = 0.7
    max_tokens: int = 4096
    max_tokens_limit: int = 200_000  # Upper bound for auto-retry on truncation

    # Per-stage LLM overrides. Keys: planning / crawling / evaluation / writing.
    # Values: {"provider": ..., "model": ..., "api_key": optional,
    #          "base_url": optional, "backend": optional}
    stage_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Available models for reference
    OPENAI_MODELS: tuple = (
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-thinking",
        "gpt-5-thinking-mini",
        "gpt-5-nano",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
    )
    ANTHROPIC_MODELS: tuple = (
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    )
    LOCAL_MODELS: tuple = (
        # Llama models (Ollama)
        "llama3.1:8b",
        "llama3.1:70b",
        "llama3.2:3b",
        "llama2:7b",
        "codellama:7b",
        # GPT-OSS models (vLLM)
        "gpt-oss-20b",
        "gpt-oss-120b",
        # Other local models
        "mistral:7b",
        "mixtral:8x7b",
        "phi3:mini",
        "qwen2:7b",
        "gemma2:9b",
    )

    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.openai_base_url is None:
            self.openai_base_url = os.getenv("OPENAI_BASE_URL")
        if self.anthropic_base_url is None:
            self.anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL")
        if self.local_base_url is None:
            self.local_base_url = os.getenv("LOCAL_LLM_BASE_URL")
        if self.local_api_key is None:
            self.local_api_key = os.getenv("LOCAL_LLM_API_KEY")

    def get_active_api_key(self) -> Optional[str]:
        """Get the API key for the active provider."""
        if self.provider == LLMProvider.OPENAI:
            return self.openai_api_key
        elif self.provider == LLMProvider.LOCAL:
            return self.local_api_key
        return self.anthropic_api_key

    def get_active_model(self) -> str:
        """Get the model name for the active provider."""
        if self.provider == LLMProvider.OPENAI:
            return self.openai_model
        elif self.provider == LLMProvider.LOCAL:
            return self.local_model
        return self.anthropic_model

    def get_active_base_url(self) -> Optional[str]:
        """Get the custom API endpoint for the active provider (None = official)."""
        if self.provider == LLMProvider.OPENAI:
            return self.openai_base_url
        elif self.provider == LLMProvider.LOCAL:
            return self.local_base_url
        return self.anthropic_base_url

    def get_base_url_for(self, provider: str) -> Optional[str]:
        """Get the configured base URL for a provider name string."""
        return {
            "openai": self.openai_base_url,
            "anthropic": self.anthropic_base_url,
            "local": self.local_base_url,
        }.get(provider)


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
    browser: str = "chrome"  # chrome / edge / firefox
    page_load_timeout: int = 30
    implicit_wait: int = 10
    # Path to a local WebDriver executable (chromedriver / msedgedriver /
    # geckodriver). When None, webdriver-manager tries to download one, which
    # fails in offline / proxy-restricted environments. Falls back to the
    # SELENIUM_DRIVER_PATH environment variable.
    driver_path: Optional[str] = None

    def __post_init__(self):
        if self.driver_path is None:
            self.driver_path = os.getenv("SELENIUM_DRIVER_PATH")

    # Content extraction settings
    extract_images: bool = True
    max_images_per_page: int = 5

    # Query simplification retry settings
    query_simplify_min_results: int = 3      # Trigger simplification when results <= this
    query_simplify_max_retries: int = 3      # Max simplification levels (1-3)


@dataclass
class MultilingualSearchConfig:
    """Configuration for multilingual search mode."""

    # Main toggle
    enabled: bool = False

    # Languages to search in (ISO 639-1 codes)
    search_languages: List[str] = field(default_factory=lambda: ["ja", "en"])

    # Results per language
    results_per_language: int = 10

    # Query translation method: "llm" (use LLM), "none" (search as-is)
    query_translation: str = "llm"

    # Whether to translate extracted content to output language
    translate_results: bool = True

    # Language weights for relevance scoring (higher = more important)
    language_weights: Dict[str, float] = field(default_factory=lambda: {
        "ja": 1.0, "en": 1.0, "zh": 0.9, "ko": 0.9,
        "de": 0.8, "fr": 0.8, "es": 0.8, "ru": 0.8
    })

    # Deduplication threshold (0.0-1.0, higher = stricter)
    dedup_threshold: float = 0.85

    # Maximum concurrent language searches
    max_concurrent_searches: int = 3

    # Include language statistics in report
    include_language_stats: bool = True

    def get_language_weight(self, lang: str) -> float:
        """Get weight for a language, defaulting to 0.5 for unknown."""
        return self.language_weights.get(lang, 0.5)

    def get_region_for_language(self, lang: str) -> str:
        """Get search region for a language code."""
        if lang in LANGUAGE_REGION_MAP:
            return LANGUAGE_REGION_MAP[lang]["region"]
        return "wt-wt"  # Default to worldwide


@dataclass
class DeepThinkConfig:
    """Configuration for DeepThink reasoning enhancement."""

    # Main toggle
    enabled: bool = False

    # Reasoning depth (0.0 = conservative, 1.0 = exploratory)
    level: float = 0.5

    # Number of reasoning iterations
    reasoning_iterations: int = 3

    # Consistency checking
    consistency_threshold: float = 0.3
    consistency_mode: str = "warn"  # warn, revise, strict

    # Quality thresholds
    fidelity_threshold: float = 0.7

    # Hidden parameters (internal use)
    _expansion_tolerance: float = 0.2
    _deviation_weights: tuple = (0.4, 0.4, 0.2)  # semantic, logical, contradiction


@dataclass
class FermiEstimationConfig:
    """Configuration for Fermi estimation module."""

    # Main toggle
    enabled: bool = False

    # Target metrics (empty = auto-detect)
    target_metrics: List[str] = field(default_factory=list)
    auto_detect_targets: bool = True

    # Decomposition settings
    max_tree_depth: int = 4
    max_leaf_nodes: int = 10

    # Calculation settings
    monte_carlo_iterations: int = 1000

    # Validation settings
    validate_with_llm: bool = True
    min_confidence_threshold: float = 0.3

    # Output settings
    write_to_data_store: bool = True
    include_sensitivity: bool = True

    # Sub-decomposition settings
    enable_sub_decomposition: bool = True
    sub_decomposition_confidence_threshold: float = 0.65
    sub_decomposition_max_iterations: int = 3
    sub_decomposition_min_sensitivity_pct: float = 10.0


class ContentFilterMode(str, Enum):
    """Content filter strictness modes."""
    STRICT = "strict"      # Aggressive filtering (removes most ads/spam)
    MODERATE = "moderate"  # Balanced filtering (default)
    MINIMAL = "minimal"    # Light filtering (only obvious ads)
    NONE = "none"          # No filtering


class ResearchSourceMode(str, Enum):
    """Which information sources the research uses."""
    WEB = "web"        # Web search/crawl only (default)
    LOCAL = "local"    # Local documents only (no web access)
    HYBRID = "hybrid"  # Local documents + web


class CrawlMode(str, Enum):
    """Crawl and evaluation mode for performance optimization."""
    STANDARD = "standard"          # Original sequential mode
    FAST_BATCH = "fast_batch"      # Fast parallel crawl + batch LLM evaluation
    FAST_PARALLEL = "fast_parallel"  # Fast parallel crawl + parallel LLM evaluation
    AI_CRAWL = "aicrawl"           # LLM-driven crawl (LLM decides which links to follow)
    AI_CRAWL_SELENIUM = "ai_crawl_selenium"  # LLM-driven crawl fetching pages via Selenium browser


@dataclass
class ResearchConfig:
    """Configuration for research process."""

    # Research loop settings
    min_iterations: int = 3
    max_iterations: int = 10

    # Search depth settings
    max_queries_per_iteration: int = 3  # Max queries to execute per iteration
    max_pages_per_query: int = 3        # Max pages to process per query

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

    # Enhanced synthesis settings
    use_enhanced_synthesis: bool = True  # Use multi-pass content generation

    # Content filtering settings (ads, spam, low-quality content)
    content_filter_mode: ContentFilterMode = ContentFilterMode.MODERATE
    custom_blocked_domains: List[str] = field(default_factory=list)
    custom_whitelisted_domains: List[str] = field(default_factory=list)

    # Fast crawl mode settings
    crawl_mode: CrawlMode = CrawlMode.STANDARD  # standard, fast_batch, fast_parallel, aicrawl
    fast_crawl_workers: int = 10   # Max parallel workers for fetching
    fast_crawl_batch_size: int = 5  # Pages per batch in batch evaluation mode

    # AI crawl mode settings (LLM-driven crawling)
    ai_crawl_max_total_pages: int = 15   # Fetch budget per section
    ai_crawl_max_depth: int = 3          # Max link depth from search-result seeds
    ai_crawl_site_depth: int = 2         # Max layers followed within one site (domain)
    ai_crawl_max_llm_calls: int = 25     # LLM decision call budget per section
    ai_crawl_max_pages_per_domain: int = 5  # Cap of fetched pages per domain
    ai_crawl_politeness_delay: float = 1.0  # Min seconds between same-domain fetches

    # Information source mode (web / local documents / hybrid)
    source_mode: ResearchSourceMode = ResearchSourceMode.WEB

    # Evidence importance / gap-fill settings
    importance_threshold: float = 0.6      # Min score to count as high-importance
    min_high_importance_sources: int = 2   # Below this, gap-fill re-search triggers
    max_gap_fill_rounds: int = 1           # Max re-search rounds per section


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    format: ReportFormat = ReportFormat.MARKDOWN

    # Format strictness: if True, never fall back to markdown when DOCX/PDF/HTML fails.
    # Raises ReportFormatError instead.
    strict_format: bool = False

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

    # Report generator version
    generator_version: ReportGeneratorVersion = ReportGeneratorVersion.V1

    # V2-specific settings (consistency features)
    v2_writing_style: str = "business"  # formal, business, technical, executive, casual
    v2_target_audience: str = "business"  # expert, business, engineer, general, student
    v2_technical_level: int = 3  # 1-5 (1=basic, 5=advanced)
    v2_enable_consistency_check: bool = True
    v2_enable_two_phase: bool = True
    v2_include_glossary: bool = True
    v2_enable_polish: bool = True  # Final naturalness polish pass over all chapters

    # Chart rendering library ("matplotlib" or "seaborn")
    chart_library: str = "matplotlib"

    # Auto figure/table generation settings
    auto_figures: bool = False  # Auto-generate figures/tables during run_research
    auto_figures_include_images: bool = True  # Include images from web sources
    auto_figures_include_tables: bool = True  # Include extracted tables
    auto_figures_include_charts: bool = True  # Include generated charts
    auto_figures_max_images: int = 2  # Max images per section

    # Numerical data extraction settings (for intelligent chart generation)
    numerical_extraction: bool = True  # Extract numerical data during research
    numerical_llm_extraction: bool = True  # Use LLM for extraction (vs pattern-only)
    numerical_min_confidence: float = 0.5  # Minimum confidence for data points

    # Unit conversion settings
    enable_unit_conversion: bool = True  # Enable SI prefix normalization and unit conversion
    enable_pint: bool = True  # Enable pint for dimensional analysis (requires: pip install pint)

    # Derived metrics settings
    derived_metrics: bool = True  # Calculate derived metrics (CAGR, growth rates)
    derived_fill_missing: bool = True  # Fill missing data points (interpolation)

    # Intelligent chart analysis settings
    intelligent_charts: bool = True  # Use ChartAnalyzer for smart chart recommendations
    chart_insights: bool = True  # Generate insights for charts
    chart_max_per_section: int = 3  # Maximum charts per section


@dataclass
class Config:
    """Main configuration class combining all settings."""

    api: APIConfig = field(default_factory=APIConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    deep_think: DeepThinkConfig = field(default_factory=DeepThinkConfig)
    fermi_estimation: FermiEstimationConfig = field(default_factory=FermiEstimationConfig)
    multilingual: MultilingualSearchConfig = field(default_factory=MultilingualSearchConfig)

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
    openai_base_url: Optional[str] = None,
    anthropic_base_url: Optional[str] = None,
    local_base_url: Optional[str] = None,
    local_backend: str = "ollama",
    model: Optional[str] = None,
    search_method: str = "duckduckgo",
    search_region: str = "wt-wt",
    safe_search: str = "moderate",
    implicit_wait: int = 10,
    driver_path: Optional[str] = None,
    research_iterations: int = 3,
    output_format: str = "markdown",
    output_dir: str = "./output",
    additional_documents: Optional[List[str]] = None,
    enable_verification: bool = True,
    verbose: bool = False,
    log_file: Optional[str] = None,
    target_pages: Optional[int] = None,
    target_characters: Optional[int] = None,
    extended_mode: bool = False,
    crawl_max_pages: int = 10,
    crawl_max_depth: int = 2,
    crawl_max_sites: int = 3,
    http_proxy: Optional[str] = None,
    https_proxy: Optional[str] = None,
    proxy_username: Optional[str] = None,
    proxy_password: Optional[str] = None,
    verify_ssl: bool = True,
    # DeepThink parameters
    deep_think: bool = False,
    deep_think_level: float = 0.5,
    reasoning_iterations: int = 3,
    consistency_threshold: float = 0.3,
    consistency_mode: str = "warn",
    fidelity_threshold: float = 0.7,
    # Multilingual search parameters
    multilingual: bool = False,
    search_languages: Optional[List[str]] = None,
    results_per_language: int = 10,
    query_translation: str = "llm",
    translate_results: bool = True,
    # Enhanced synthesis
    use_enhanced_synthesis: bool = True,
    # Research content settings
    max_content_length: int = 50000,
    save_evidence: bool = True,
    evidence_format: str = "json",
    # Search depth parameters
    max_queries_per_iteration: int = 3,
    max_pages_per_query: int = 3,
    # Content filtering parameters
    content_filter_mode: str = "moderate",
    custom_blocked_domains: Optional[List[str]] = None,
    custom_whitelisted_domains: Optional[List[str]] = None,
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
    # Report generator version parameters
    report_generator_version: str = "v1",
    v2_writing_style: str = "business",
    v2_target_audience: str = "business",
    v2_technical_level: int = 3,
    v2_enable_consistency_check: bool = True,
    v2_enable_two_phase: bool = True,
    v2_include_glossary: bool = True,
    v2_enable_polish: bool = True,
    # Chart library parameter
    chart_library: str = "matplotlib",
    # Per-stage LLM overrides
    stage_llm: Optional[Dict[str, Dict[str, Any]]] = None,
    # Auto figure/table generation parameters
    auto_figures: bool = False,
    auto_figures_include_images: bool = True,
    auto_figures_include_tables: bool = True,
    auto_figures_include_charts: bool = True,
    auto_figures_max_images: int = 2,
    # Numerical data extraction parameters
    numerical_extraction: bool = True,
    numerical_llm_extraction: bool = True,
    numerical_min_confidence: float = 0.5,
    # Unit conversion parameters
    enable_unit_conversion: bool = True,
    enable_pint: bool = True,
    # Derived metrics parameters
    derived_metrics: bool = True,
    derived_fill_missing: bool = True,
    # Intelligent chart analysis parameters
    intelligent_charts: bool = True,
    chart_insights: bool = True,
    chart_max_per_section: int = 3,
    # Format strictness
    strict_format: bool = False,
    # Fermi estimation parameters
    fermi_estimation: bool = False,
    fermi_target_metrics: Optional[List[str]] = None,
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
    **kwargs
) -> Config:
    """
    Factory function to create a Config object with common settings.

    Args:
        provider: LLM provider ('openai' or 'anthropic')
        openai_api_key: OpenAI API key (optional, uses env var if not provided)
        anthropic_api_key: Anthropic API key (optional, uses env var if not provided)
        openai_base_url: Custom OpenAI API endpoint (optional; falls back to
            OPENAI_BASE_URL env var, then the official endpoint). Use for
            corporate gateways / OpenAI-compatible APIs
        anthropic_base_url: Custom Anthropic API endpoint (optional; falls back
            to ANTHROPIC_BASE_URL env var, then the official endpoint)
        local_base_url: Local LLM server URL for provider='local' (optional;
            falls back to LOCAL_LLM_BASE_URL env var, then the backend default,
            e.g. http://localhost:11434 for Ollama)
        local_backend: Local LLM backend type ('ollama', 'vllm', or
            'openai_compatible')
        model: Model name to use (optional, uses default for provider)
        search_method: Web search method ('duckduckgo' or 'selenium')
        browser: Selenium browser via kwargs ('chrome', 'edge', or 'firefox';
            used by selenium search and the ai_crawl_selenium crawl mode)
        search_region: DuckDuckGo search region (e.g., 'wt-wt' worldwide, 'jp-jp' Japan)
        safe_search: DuckDuckGo safe search level ('off', 'moderate', 'strict')
        implicit_wait: Selenium implicit wait time in seconds
        driver_path: Path to a local WebDriver executable (chromedriver /
            msedgedriver / geckodriver). Required in offline or
            proxy-restricted environments where webdriver-manager cannot
            download drivers. Falls back to SELENIUM_DRIVER_PATH env var
        research_iterations: Number of research iterations
        output_format: Report format ('docx', 'pdf', or 'markdown')
        output_dir: Output directory path
        additional_documents: List of additional document paths
        enable_verification: Enable hallucination verification
        verbose: Enable verbose logging
        log_file: Path to log file (optional, logs to file if specified)
        target_pages: Target page count for output (approximate)
        target_characters: Target character count for output
        extended_mode: Enable extended mode (deep site crawling)
        crawl_max_pages: Max pages to crawl per site in extended mode
        crawl_max_depth: Max link depth from seed URL in extended mode
        crawl_max_sites: Max sites to crawl per search in extended mode
        http_proxy: HTTP proxy URL (e.g., "http://proxy.example.com:8080")
        https_proxy: HTTPS proxy URL
        proxy_username: Proxy authentication username
        proxy_password: Proxy authentication password
        verify_ssl: Verify SSL certificates (set False for self-signed certs)
        deep_think: Enable DeepThink reasoning enhancement
        deep_think_level: Reasoning depth level (0.0-1.0, 0=conservative, 1=exploratory)
        reasoning_iterations: Number of reasoning iterations for DeepThink
        consistency_threshold: Threshold for consistency check (0.0-1.0)
        consistency_mode: How to handle consistency issues ('warn', 'revise', 'strict')
        fidelity_threshold: Minimum source fidelity score (0.0-1.0)
        multilingual: Enable multilingual search mode
        search_languages: List of language codes to search (e.g., ['ja', 'en', 'zh'])
        results_per_language: Number of results per language
        query_translation: Query translation method ('llm' or 'none')
        translate_results: Whether to translate results to output language
        use_enhanced_synthesis: Use multi-pass content generation for better quality
        max_content_length: Maximum content length for extraction truncation (default: 50000)
        save_evidence: Whether to save evidence exports (default: True)
        evidence_format: Evidence export format: 'json', 'csv', or 'both' (default: 'json')
        max_queries_per_iteration: Max queries to execute per research iteration (default: 3)
        max_pages_per_query: Max pages to process per search query (default: 3)
        content_filter_mode: Content filter strictness ('strict', 'moderate', 'minimal', 'none')
        custom_blocked_domains: List of domains to block in addition to defaults
        custom_whitelisted_domains: List of domains to always allow
        crawl_mode: Crawl mode ('standard', 'fast_batch', 'fast_parallel',
            'aicrawl', 'ai_crawl_selenium')
        fast_crawl_workers: Max parallel workers for fast crawl mode
        fast_crawl_batch_size: Pages per batch in batch evaluation mode
        ai_crawl_max_total_pages: aicrawl mode - max pages fetched per section
        ai_crawl_max_depth: aicrawl mode - max link depth from search-result seeds
        ai_crawl_site_depth: aicrawl mode - max layers followed within one site
        ai_crawl_max_llm_calls: aicrawl mode - LLM decision call budget per section
        ai_crawl_max_pages_per_domain: aicrawl mode - max pages fetched per domain
        ai_crawl_politeness_delay: aicrawl mode - min seconds between same-domain fetches
        source_mode: Information sources to use: 'web' (web only, default),
            'local' (local documents only; requires additional_documents),
            'hybrid' (local documents + web)
        importance_threshold: min importance score (0-1) to count a source as
            high-importance for the research purpose
        min_high_importance_sources: sections with fewer high-importance sources
            trigger gap-fill re-search
        max_gap_fill_rounds: max gap-fill re-search rounds per section
        report_generator_version: Report generator version ('v1' or 'v2')
        v2_writing_style: V2 writing style ('formal', 'business', 'technical', 'executive', 'casual')
        v2_target_audience: V2 target audience ('expert', 'business', 'engineer', 'general', 'student')
        v2_technical_level: V2 technical level (1-5, 1=basic, 5=advanced)
        v2_enable_consistency_check: Enable V2 consistency checking
        v2_enable_two_phase: Enable V2 two-phase generation (draft + refinement)
        v2_include_glossary: Include glossary section in V2 reports
        v2_enable_polish: Enable final naturalness polish pass over all V2 chapters
        chart_library: Chart rendering library ('matplotlib' or 'seaborn';
            falls back to matplotlib when seaborn is not installed)
        stage_llm: Per-stage LLM overrides. Keys: 'planning' (research plan /
            queries), 'crawling' (crawl decisions), 'evaluation' (importance /
            quality / consistency scoring), 'writing' (prose generation).
            Values: {'provider': ..., 'model': ..., 'api_key': optional}.
            Stages without an override use the default provider/model.
            Example: {"planning": {"provider": "openai", "model": "gpt-5-mini"},
                      "writing": {"provider": "anthropic",
                                  "model": "claude-3-5-sonnet-20241022"}}
        auto_figures: Auto-generate figures/tables during run_research
        auto_figures_include_images: Include images from web sources in auto figures
        auto_figures_include_tables: Include extracted tables in auto figures
        auto_figures_include_charts: Include generated charts in auto figures
        auto_figures_max_images: Maximum images per section in auto figures
        numerical_extraction: Extract numerical data during research for intelligent charts
        numerical_llm_extraction: Use LLM for numerical extraction (vs pattern-only)
        numerical_min_confidence: Minimum confidence threshold for numerical data points
        enable_unit_conversion: Enable SI prefix normalization and unit conversion tables
        enable_pint: Enable pint library for dimensional analysis (requires: pip install pint)
        derived_metrics: Calculate derived metrics (CAGR, growth rates, etc.)
        derived_fill_missing: Fill missing data points via interpolation
        intelligent_charts: Use ChartAnalyzer for smart chart recommendations
        chart_insights: Generate insights and messages for charts
        chart_max_per_section: Maximum charts per section
        **kwargs: Additional keyword arguments

    Returns:
        Configured Config object
    """
    if stage_llm:
        from .api.stage_router import LLM_STAGES
        unknown_stages = set(stage_llm) - set(LLM_STAGES)
        if unknown_stages:
            raise ValueError(
                f"Unknown LLM stages: {sorted(unknown_stages)}. "
                f"Valid stages: {list(LLM_STAGES)}"
            )

    api_config = APIConfig(
        provider=LLMProvider(provider),
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        openai_base_url=openai_base_url,
        anthropic_base_url=anthropic_base_url,
        local_base_url=local_base_url,
        local_backend=LocalLLMBackend(local_backend),
        stage_overrides=dict(stage_llm) if stage_llm else {},
    )

    if model:
        if provider == "openai":
            api_config.openai_model = model
        elif provider == "local":
            api_config.local_model = model
        else:
            api_config.anthropic_model = model

    search_config = SearchConfig(
        method=SearchMethod(search_method),
        headless=kwargs.get("headless", True),
        max_results=kwargs.get("max_results", 10),
        browser=kwargs.get("browser", "chrome"),
        region=search_region,
        safe_search=safe_search,
        implicit_wait=implicit_wait,
        driver_path=driver_path,
    )

    research_config = ResearchConfig(
        min_iterations=research_iterations,
        max_iterations=kwargs.get("max_iterations", research_iterations + 5),
        max_queries_per_iteration=max_queries_per_iteration,
        max_pages_per_query=max_pages_per_query,
        max_content_length=max_content_length,
        language=kwargs.get("language", "ja"),
        save_evidence=save_evidence,
        evidence_format=evidence_format,
        extended_mode=extended_mode,
        crawl_max_pages=crawl_max_pages,
        crawl_max_depth=crawl_max_depth,
        crawl_max_sites=crawl_max_sites,
        use_enhanced_synthesis=use_enhanced_synthesis,
        content_filter_mode=ContentFilterMode(content_filter_mode),
        custom_blocked_domains=custom_blocked_domains if custom_blocked_domains else [],
        custom_whitelisted_domains=custom_whitelisted_domains if custom_whitelisted_domains else [],
        crawl_mode=CrawlMode(crawl_mode),
        fast_crawl_workers=fast_crawl_workers,
        fast_crawl_batch_size=fast_crawl_batch_size,
        ai_crawl_max_total_pages=ai_crawl_max_total_pages,
        ai_crawl_max_depth=ai_crawl_max_depth,
        ai_crawl_site_depth=ai_crawl_site_depth,
        ai_crawl_max_llm_calls=ai_crawl_max_llm_calls,
        ai_crawl_max_pages_per_domain=ai_crawl_max_pages_per_domain,
        ai_crawl_politeness_delay=ai_crawl_politeness_delay,
        source_mode=ResearchSourceMode(source_mode),
        importance_threshold=importance_threshold,
        min_high_importance_sources=min_high_importance_sources,
        max_gap_fill_rounds=max_gap_fill_rounds,
    )

    report_config = ReportConfig(
        format=ReportFormat(output_format),
        strict_format=strict_format,
        output_dir=Path(output_dir),
        target_pages=target_pages,
        target_characters=target_characters,
        generator_version=ReportGeneratorVersion(report_generator_version),
        v2_writing_style=v2_writing_style,
        v2_target_audience=v2_target_audience,
        v2_technical_level=v2_technical_level,
        v2_enable_consistency_check=v2_enable_consistency_check,
        v2_enable_two_phase=v2_enable_two_phase,
        v2_include_glossary=v2_include_glossary,
        v2_enable_polish=v2_enable_polish,
        chart_library=chart_library,
        auto_figures=auto_figures,
        auto_figures_include_images=auto_figures_include_images,
        auto_figures_include_tables=auto_figures_include_tables,
        auto_figures_include_charts=auto_figures_include_charts,
        auto_figures_max_images=auto_figures_max_images,
        numerical_extraction=numerical_extraction,
        numerical_llm_extraction=numerical_llm_extraction,
        numerical_min_confidence=numerical_min_confidence,
        enable_unit_conversion=enable_unit_conversion,
        enable_pint=enable_pint,
        derived_metrics=derived_metrics,
        derived_fill_missing=derived_fill_missing,
        intelligent_charts=intelligent_charts,
        chart_insights=chart_insights,
        chart_max_per_section=chart_max_per_section,
    )

    proxy_config = ProxyConfig(
        http_proxy=http_proxy,
        https_proxy=https_proxy,
        proxy_username=proxy_username,
        proxy_password=proxy_password,
        verify_ssl=verify_ssl,
    )

    deep_think_config = DeepThinkConfig(
        enabled=deep_think,
        level=deep_think_level,
        reasoning_iterations=reasoning_iterations,
        consistency_threshold=consistency_threshold,
        consistency_mode=consistency_mode,
        fidelity_threshold=fidelity_threshold,
        _expansion_tolerance=kwargs.get("_expansion_tolerance", 0.2),
        _deviation_weights=kwargs.get("_deviation_weights", (0.4, 0.4, 0.2)),
    )

    multilingual_config = MultilingualSearchConfig(
        enabled=multilingual,
        search_languages=search_languages if search_languages else ["ja", "en"],
        results_per_language=results_per_language,
        query_translation=query_translation,
        translate_results=translate_results,
        dedup_threshold=kwargs.get("dedup_threshold", 0.85),
        max_concurrent_searches=kwargs.get("max_concurrent_searches", 3),
        include_language_stats=kwargs.get("include_language_stats", True),
    )

    docs = []
    if additional_documents:
        docs = [Path(doc) for doc in additional_documents]

    # Validate kwargs: warn about unrecognized keys to catch typos
    _KNOWN_KWARGS = {
        "headless", "max_results", "max_iterations", "language", "browser",
        "_expansion_tolerance", "_deviation_weights",
        "dedup_threshold", "max_concurrent_searches", "include_language_stats",
    }
    _unknown_kwargs = set(kwargs.keys()) - _KNOWN_KWARGS
    if _unknown_kwargs:
        import warnings
        warnings.warn(
            f"create_config() received unrecognized keyword arguments: "
            f"{sorted(_unknown_kwargs)}. These will be ignored. "
            f"Check for typos or use explicit parameters instead.",
            stacklevel=2,
        )

    fermi_config = FermiEstimationConfig(
        enabled=fermi_estimation,
        target_metrics=fermi_target_metrics or [],
        auto_detect_targets=fermi_auto_detect,
        max_tree_depth=fermi_max_tree_depth,
        max_leaf_nodes=fermi_max_leaf_nodes,
        monte_carlo_iterations=fermi_monte_carlo,
        validate_with_llm=fermi_validate,
        include_sensitivity=fermi_include_sensitivity,
        enable_sub_decomposition=fermi_enable_sub_decomposition,
        sub_decomposition_max_iterations=fermi_sub_decomposition_max_iterations,
        sub_decomposition_confidence_threshold=fermi_sub_decomposition_confidence_threshold,
        sub_decomposition_min_sensitivity_pct=fermi_sub_decomposition_min_sensitivity_pct,
    )

    return Config(
        api=api_config,
        search=search_config,
        research=research_config,
        report=report_config,
        proxy=proxy_config,
        deep_think=deep_think_config,
        fermi_estimation=fermi_config,
        multilingual=multilingual_config,
        additional_documents=docs,
        process_additional_documents=bool(additional_documents),
        enable_verification=enable_verification,
        verbose=verbose,
        log_file=Path(log_file) if log_file else None,
    )
