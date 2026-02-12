"""
Main FactChecker class that integrates all components.

Provides a simple interface for sentence-level fact-checking:
1. Split text into sentences
2. Extract claims from each sentence
3. Search and crawl web for evidence
4. Verify claims using separate LLM session
5. Generate comprehensive report with corrections

Supports multiple LLM providers (OpenAI, Anthropic) and search engines
(Google, DuckDuckGo, Bing).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable

from .sentence_splitter import SentenceSplitter, Sentence, SplitMethod
from .claim_extractor import ClaimExtractor, Claim
from .web_crawler import RecursiveWebCrawler, CrawlResult, AdFilter, SearchEngine
from .fact_verifier import (
    FactVerifier,
    SentenceVerificationResult,
    FactCheckReport,
    VerificationLabel,
)


@dataclass
class FactCheckerConfig:
    """
    Configuration for FactChecker.

    Supports both environment variables and direct parameter passing.
    Environment variables take precedence if parameter is not explicitly set.
    """

    # LLM Configuration for claim extraction
    extraction_llm_provider: str = "openai"
    extraction_llm_api_key: Optional[str] = None
    extraction_llm_model: Optional[str] = None

    # LLM Configuration for verification (separate session)
    verification_llm_provider: str = "openai"
    verification_llm_api_key: Optional[str] = None
    verification_llm_model: Optional[str] = None

    # Common LLM settings
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096

    # Sentence splitting settings
    split_method: SplitMethod = SplitMethod.RULE_BASED
    min_sentence_length: int = 5

    # Claim extraction settings
    use_pattern_matching: bool = True
    extract_search_queries: bool = True

    # Web search settings
    search_engine: str = SearchEngine.GOOGLE
    max_search_results: int = 5
    max_crawl_depth: int = 1
    max_pages_per_depth: int = 2

    # Browser settings
    headless: bool = True
    browser: str = "chrome"
    page_load_timeout: int = 30

    # Ad filtering
    strict_ad_filtering: bool = False

    # Output settings
    output_language: str = "ja"
    output_dir: Path = field(default_factory=lambda: Path("./output"))

    # Progress callback
    progress_callback: Optional[Callable[[str, float], None]] = None

    def __post_init__(self):
        """Load from environment variables if not explicitly set."""
        # Extraction LLM API keys
        if self.extraction_llm_api_key is None:
            if self.extraction_llm_provider == "openai":
                self.extraction_llm_api_key = os.getenv("OPENAI_API_KEY")
            elif self.extraction_llm_provider == "anthropic":
                self.extraction_llm_api_key = os.getenv("ANTHROPIC_API_KEY")

        # Verification LLM API keys
        if self.verification_llm_api_key is None:
            if self.verification_llm_provider == "openai":
                self.verification_llm_api_key = os.getenv("OPENAI_API_KEY")
            elif self.verification_llm_provider == "anthropic":
                self.verification_llm_api_key = os.getenv("ANTHROPIC_API_KEY")

        # Override from environment if specific env vars are set
        if os.getenv("FACT_CHECKER_EXTRACTION_LLM"):
            self.extraction_llm_provider = os.getenv("FACT_CHECKER_EXTRACTION_LLM")
        if os.getenv("FACT_CHECKER_VERIFICATION_LLM"):
            self.verification_llm_provider = os.getenv("FACT_CHECKER_VERIFICATION_LLM")
        if os.getenv("FACT_CHECKER_SEARCH_ENGINE"):
            self.search_engine = os.getenv("FACT_CHECKER_SEARCH_ENGINE")
        if os.getenv("FACT_CHECKER_LANGUAGE"):
            self.output_language = os.getenv("FACT_CHECKER_LANGUAGE")

        # Create output directory
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Check extraction LLM API key
        if self.extraction_llm_provider == "openai" and not self.extraction_llm_api_key:
            errors.append(
                "OpenAI API key for extraction not set. "
                "Set OPENAI_API_KEY environment variable or pass extraction_llm_api_key."
            )
        elif self.extraction_llm_provider == "anthropic" and not self.extraction_llm_api_key:
            errors.append(
                "Anthropic API key for extraction not set. "
                "Set ANTHROPIC_API_KEY environment variable or pass extraction_llm_api_key."
            )

        # Check verification LLM API key
        if self.verification_llm_provider == "openai" and not self.verification_llm_api_key:
            errors.append(
                "OpenAI API key for verification not set. "
                "Set OPENAI_API_KEY environment variable or pass verification_llm_api_key."
            )
        elif self.verification_llm_provider == "anthropic" and not self.verification_llm_api_key:
            errors.append(
                "Anthropic API key for verification not set. "
                "Set ANTHROPIC_API_KEY environment variable or pass verification_llm_api_key."
            )

        return errors

    @classmethod
    def from_env(cls) -> "FactCheckerConfig":
        """Create configuration from environment variables."""
        return cls(
            extraction_llm_provider=os.getenv("FACT_CHECKER_EXTRACTION_LLM", "openai"),
            verification_llm_provider=os.getenv("FACT_CHECKER_VERIFICATION_LLM", "openai"),
            search_engine=os.getenv("FACT_CHECKER_SEARCH_ENGINE", SearchEngine.GOOGLE),
            output_language=os.getenv("FACT_CHECKER_LANGUAGE", "ja"),
        )


class FactChecker:
    """
    Main fact-checking interface.

    Provides sentence-level fact-checking with:
    - Text splitting into sentences
    - Claim extraction
    - Web-based evidence gathering
    - LLM-based verification
    - Comprehensive reporting
    """

    def __init__(self, config: Optional[FactCheckerConfig] = None):
        """
        Initialize FactChecker.

        Args:
            config: Configuration object. If None, creates default config.
        """
        self.config = config or FactCheckerConfig()

        # Validate configuration
        errors = self.config.validate()
        if errors:
            print("Configuration warnings:")
            for error in errors:
                print(f"  - {error}")

        # Initialize components (lazy initialization)
        self._sentence_splitter = None
        self._claim_extractor = None
        self._web_crawler = None
        self._fact_verifier = None
        self._extraction_llm = None

    def _get_extraction_llm(self):
        """Get or create extraction LLM client."""
        if self._extraction_llm is None:
            self._extraction_llm = self._create_llm_client(
                self.config.extraction_llm_provider,
                self.config.extraction_llm_api_key,
                self.config.extraction_llm_model,
            )
        return self._extraction_llm

    def _create_llm_client(
        self,
        provider: str,
        api_key: Optional[str],
        model: Optional[str],
    ):
        """Create an LLM client."""
        if provider == "openai":
            from ..api.openai_client import OpenAIClient
            return OpenAIClient(
                api_key=api_key,
                model=model or "gpt-4o-mini",
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
        elif provider == "anthropic":
            from ..api.anthropic_client import AnthropicClient
            return AnthropicClient(
                api_key=api_key,
                model=model or "claude-3-5-sonnet-20241022",
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _get_sentence_splitter(self) -> SentenceSplitter:
        """Get or create sentence splitter."""
        if self._sentence_splitter is None:
            llm_client = None
            if self.config.split_method in [SplitMethod.LLM, SplitMethod.HYBRID]:
                llm_client = self._get_extraction_llm()

            self._sentence_splitter = SentenceSplitter(
                method=self.config.split_method,
                min_sentence_length=self.config.min_sentence_length,
                llm_client=llm_client,
            )
        return self._sentence_splitter

    def _get_claim_extractor(self) -> ClaimExtractor:
        """Get or create claim extractor."""
        if self._claim_extractor is None:
            self._claim_extractor = ClaimExtractor(
                llm_client=self._get_extraction_llm(),
                use_patterns=self.config.use_pattern_matching,
                extract_search_queries=self.config.extract_search_queries,
                language=self.config.output_language,
            )
        return self._claim_extractor

    def _get_web_crawler(self) -> RecursiveWebCrawler:
        """Get or create web crawler."""
        if self._web_crawler is None:
            ad_filter = AdFilter(strict_mode=self.config.strict_ad_filtering)

            self._web_crawler = RecursiveWebCrawler(
                headless=self.config.headless,
                browser=self.config.browser,
                search_engine=self.config.search_engine,
                max_search_results=self.config.max_search_results,
                max_crawl_depth=self.config.max_crawl_depth,
                max_pages_per_depth=self.config.max_pages_per_depth,
                page_load_timeout=self.config.page_load_timeout,
                ad_filter=ad_filter,
            )
        return self._web_crawler

    def _get_fact_verifier(self) -> FactVerifier:
        """Get or create fact verifier."""
        if self._fact_verifier is None:
            self._fact_verifier = FactVerifier(
                llm_provider=self.config.verification_llm_provider,
                api_key=self.config.verification_llm_api_key,
                model=self.config.verification_llm_model,
                temperature=self.config.llm_temperature,
                language=self.config.output_language,
            )
        return self._fact_verifier

    def _report_progress(self, message: str, progress: float):
        """Report progress if callback is set."""
        if self.config.progress_callback:
            self.config.progress_callback(message, progress)
        else:
            print(f"[{progress:.0%}] {message}")

    def check_text(
        self,
        text: str,
        document_title: str = "Fact Check",
        skip_opinions: bool = True,
    ) -> FactCheckReport:
        """
        Perform fact-checking on a text.

        Args:
            text: The text to fact-check
            document_title: Title for the report
            skip_opinions: Skip non-verifiable claims (opinions)

        Returns:
            FactCheckReport with detailed results
        """
        self._report_progress("Starting fact-check...", 0.0)

        # Step 1: Split into sentences
        self._report_progress("Splitting text into sentences...", 0.1)
        splitter = self._get_sentence_splitter()
        sentences = splitter.split(text)
        self._report_progress(f"Found {len(sentences)} sentences", 0.15)

        # Step 2: Extract claims
        self._report_progress("Extracting claims from sentences...", 0.2)
        extractor = self._get_claim_extractor()
        claims = extractor.extract_claims(sentences)
        self._report_progress(f"Extracted {len(claims)} claims", 0.3)

        # Filter out non-verifiable claims if requested
        if skip_opinions:
            verifiable_claims = [c for c in claims if c.is_verifiable]
            self._report_progress(
                f"Found {len(verifiable_claims)} verifiable claims "
                f"(skipped {len(claims) - len(verifiable_claims)} opinions)",
                0.35
            )
        else:
            verifiable_claims = claims

        # Step 3: Gather evidence for each claim
        self._report_progress("Gathering evidence from web...", 0.4)
        crawler = self._get_web_crawler()
        evidence_map: Dict[int, List[CrawlResult]] = {}

        for i, claim in enumerate(verifiable_claims):
            progress = 0.4 + (0.3 * (i / len(verifiable_claims)))
            self._report_progress(f"Searching evidence for claim {i+1}/{len(verifiable_claims)}...", progress)

            # Get search queries for this claim
            queries = claim.search_queries or [claim.text[:100]]

            # Search and crawl for each query
            all_evidence = []
            for query in queries[:2]:  # Limit to 2 queries per claim
                try:
                    results = crawler.search_and_crawl(query, recursive=True)
                    all_evidence.extend(results)
                except Exception as e:
                    print(f"Warning: Search failed for query '{query}': {e}")

            # Also fetch reference URLs if present
            if claim.source_sentence.has_reference:
                for ref_url in claim.source_sentence.reference_urls[:2]:
                    try:
                        result = crawler.fetch_reference_url(ref_url)
                        all_evidence.append(result)
                    except Exception as e:
                        print(f"Warning: Failed to fetch reference {ref_url}: {e}")

            evidence_map[i] = all_evidence[:10]  # Limit to 10 evidence items per claim

        # Step 4: Verify claims
        self._report_progress("Verifying claims with LLM...", 0.7)
        verifier = self._get_fact_verifier()
        verification_results = []

        for i, claim in enumerate(verifiable_claims):
            progress = 0.7 + (0.2 * (i / len(verifiable_claims)))
            self._report_progress(f"Verifying claim {i+1}/{len(verifiable_claims)}...", progress)

            evidence = evidence_map.get(i, [])
            result = verifier.verify_claim(claim, evidence)
            verification_results.append(result)

        # Step 5: Generate report
        self._report_progress("Generating report...", 0.95)
        report = verifier.generate_report(
            results=verification_results,
            document_title=document_title,
            total_sentences=len(sentences),
        )

        self._report_progress("Fact-check complete!", 1.0)

        return report

    def check_file(
        self,
        file_path: str,
        document_title: Optional[str] = None,
        skip_opinions: bool = True,
    ) -> FactCheckReport:
        """
        Perform fact-checking on a file.

        Args:
            file_path: Path to the file to check
            document_title: Title for the report (defaults to filename)
            skip_opinions: Skip non-verifiable claims

        Returns:
            FactCheckReport with detailed results
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content
        if path.suffix.lower() == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif path.suffix.lower() == '.md':
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif path.suffix.lower() == '.pdf':
            from ..utils.document_reader import DocumentReader
            reader = DocumentReader()
            doc = reader.read(str(path))
            text = doc.full_text
        elif path.suffix.lower() == '.docx':
            from ..utils.document_reader import DocumentReader
            reader = DocumentReader()
            doc = reader.read(str(path))
            text = doc.full_text
        else:
            # Try reading as plain text
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()

        title = document_title or path.stem
        return self.check_text(text, document_title=title, skip_opinions=skip_opinions)

    def export_report(
        self,
        report: FactCheckReport,
        output_path: Optional[str] = None,
        format: str = "html",
    ) -> str:
        """
        Export report to file.

        Args:
            report: The FactCheckReport to export
            output_path: Output file path (auto-generated if None)
            format: Output format ("html", "json")

        Returns:
            Path to the exported file
        """
        if output_path is None:
            timestamp = report.created_at.replace(":", "-").replace("T", "_")[:19]
            filename = f"fact_check_{timestamp}.{format}"
            output_path = str(self.config.output_dir / filename)

        verifier = self._get_fact_verifier()

        if format == "html":
            verifier.export_report_html(report, output_path)
        elif format == "json":
            verifier.export_report_json(report, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return output_path

    def close(self):
        """Close all resources."""
        if self._web_crawler:
            self._web_crawler.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_fact_checker(
    extraction_provider: str = "openai",
    extraction_api_key: Optional[str] = None,
    extraction_model: Optional[str] = None,
    verification_provider: str = "openai",
    verification_api_key: Optional[str] = None,
    verification_model: Optional[str] = None,
    search_engine: str = SearchEngine.GOOGLE,
    language: str = "ja",
    headless: bool = True,
    **kwargs,
) -> FactChecker:
    """
    Factory function to create a FactChecker with common settings.

    Args:
        extraction_provider: LLM provider for claim extraction ("openai" or "anthropic")
        extraction_api_key: API key for extraction LLM
        extraction_model: Model name for extraction
        verification_provider: LLM provider for verification ("openai" or "anthropic")
        verification_api_key: API key for verification LLM
        verification_model: Model name for verification
        search_engine: Search engine to use ("google", "duckduckgo", "bing")
        language: Output language ("ja" or "en")
        headless: Run browser in headless mode
        **kwargs: Additional configuration options

    Returns:
        Configured FactChecker instance
    """
    config = FactCheckerConfig(
        extraction_llm_provider=extraction_provider,
        extraction_llm_api_key=extraction_api_key,
        extraction_llm_model=extraction_model,
        verification_llm_provider=verification_provider,
        verification_llm_api_key=verification_api_key,
        verification_llm_model=verification_model,
        search_engine=search_engine,
        output_language=language,
        headless=headless,
        **kwargs,
    )

    return FactChecker(config=config)


# Convenience function for quick fact-checking
def quick_fact_check(
    text: str,
    api_key: Optional[str] = None,
    provider: str = "openai",
    search_engine: str = SearchEngine.GOOGLE,
    language: str = "ja",
) -> FactCheckReport:
    """
    Quick fact-check a text with minimal configuration.

    Args:
        text: Text to fact-check
        api_key: LLM API key (uses environment variable if not provided)
        provider: LLM provider ("openai" or "anthropic")
        search_engine: Search engine to use
        language: Output language

    Returns:
        FactCheckReport with results
    """
    config = FactCheckerConfig(
        extraction_llm_provider=provider,
        extraction_llm_api_key=api_key,
        verification_llm_provider=provider,
        verification_llm_api_key=api_key,
        search_engine=search_engine,
        output_language=language,
    )

    with FactChecker(config=config) as checker:
        return checker.check_text(text)
