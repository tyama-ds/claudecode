"""
Configuration management for Deep Research Tool.

Supports both environment variables and direct parameter passing.
"""

import os
from enum import Enum
from typing import Optional, List, Dict
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


class ContentFilterMode(str, Enum):
    """Content filter strictness modes."""
    STRICT = "strict"      # Aggressive filtering (removes most ads/spam)
    MODERATE = "moderate"  # Balanced filtering (default)
    MINIMAL = "minimal"    # Light filtering (only obvious ads)
    NONE = "none"          # No filtering


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
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    deep_think: DeepThinkConfig = field(default_factory=DeepThinkConfig)
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
    # Search depth parameters
    max_queries_per_iteration: int = 3,
    max_pages_per_query: int = 3,
    # Content filtering parameters
    content_filter_mode: str = "moderate",
    custom_blocked_domains: Optional[List[str]] = None,
    custom_whitelisted_domains: Optional[List[str]] = None,
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
        max_queries_per_iteration: Max queries to execute per research iteration (default: 3)
        max_pages_per_query: Max pages to process per search query (default: 3)
        content_filter_mode: Content filter strictness ('strict', 'moderate', 'minimal', 'none')
        custom_blocked_domains: List of domains to block in addition to defaults
        custom_whitelisted_domains: List of domains to always allow
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
        max_queries_per_iteration=max_queries_per_iteration,
        max_pages_per_query=max_pages_per_query,
        language=kwargs.get("language", "ja"),
        extended_mode=extended_mode,
        crawl_max_pages=crawl_max_pages,
        crawl_max_depth=crawl_max_depth,
        crawl_max_sites=crawl_max_sites,
        use_enhanced_synthesis=use_enhanced_synthesis,
        content_filter_mode=ContentFilterMode(content_filter_mode),
        custom_blocked_domains=custom_blocked_domains if custom_blocked_domains else [],
        custom_whitelisted_domains=custom_whitelisted_domains if custom_whitelisted_domains else [],
    )

    report_config = ReportConfig(
        format=ReportFormat(output_format),
        output_dir=Path(output_dir),
        target_pages=target_pages,
        target_characters=target_characters,
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

    return Config(
        api=api_config,
        search=search_config,
        research=research_config,
        report=report_config,
        proxy=proxy_config,
        deep_think=deep_think_config,
        multilingual=multilingual_config,
        additional_documents=docs,
        process_additional_documents=bool(additional_documents),
        enable_verification=enable_verification,
        verbose=verbose,
    )
