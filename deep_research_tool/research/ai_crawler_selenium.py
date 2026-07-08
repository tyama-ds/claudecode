"""
AI Crawler (Selenium variant) - LLM-driven crawling with browser-based fetching.

Same crawl logic as AICrawler, but pages are fetched through the existing
SeleniumBrowser client so JavaScript-rendered sites can be read. Search
seeding still goes through the regular search client; only page fetching
uses the browser.
"""

from typing import Callable, List, Optional

from ..evidence.content_filter import ContentFilter
from ..utils.helpers import ResearchWarnings
from .ai_crawler import AICrawler
from .fast_crawler import CrawlResult


class AICrawlerSelenium(AICrawler):
    """
    LLM-driven crawler that fetches pages with a Selenium browser.

    Use for research targets where the plain requests-based fetch misses
    content (JavaScript-rendered pages, lazy-loaded articles). The browser
    is created lazily on first fetch and must be released with close()
    after the research run.
    """

    def __init__(
        self,
        search_client,
        llm_client,
        content_filter: ContentFilter = None,
        max_total_pages: int = 15,
        max_depth: int = 3,
        max_site_depth: int = 2,
        max_llm_calls: int = 25,
        max_pages_per_domain: int = 5,
        politeness_delay: float = 1.0,
        max_link_candidates: int = AICrawler.DEFAULT_MAX_LINK_CANDIDATES,
        max_reseed_rounds: int = 2,
        language: str = "ja",
        selenium_client=None,
        headless: bool = True,
        browser: str = "chrome",
        page_load_timeout: int = 30,
        proxies: dict = None,
        verify_ssl: bool = True,
        driver_path: str = None,
    ):
        """
        Initialize AICrawlerSelenium.

        Args:
            search_client: Web search client (used for search seeding only)
            llm_client: LLM client for crawl decisions
            content_filter: Content filter for ads/spam removal
            max_total_pages: Fetch budget per crawl_and_evaluate call
            max_depth: Maximum link depth from search-result seeds
            max_site_depth: Maximum layers followed within one site (domain)
            max_llm_calls: Budget of LLM decision calls per crawl
            max_pages_per_domain: Cap of fetched pages per domain
            politeness_delay: Minimum seconds between fetches to the same domain
            max_link_candidates: Max links per page offered to the LLM
            max_reseed_rounds: Max re-seeding rounds from suggested queries
            language: Language for decision prompts
            selenium_client: Pre-built Selenium client (created lazily if None)
            headless: Run the browser headless
            browser: Browser type ("chrome", "edge", or "firefox")
            page_load_timeout: Page load timeout in seconds
            proxies: Proxy dict like {"https": "http://proxy:8080"} passed
                to the browser launch
            verify_ssl: When False the browser ignores certificate errors
            driver_path: Path to a local WebDriver executable (chromedriver /
                msedgedriver / geckodriver); None lets webdriver-manager /
                Selenium Manager resolve one
        """
        super().__init__(
            search_client=search_client,
            llm_client=llm_client,
            content_filter=content_filter,
            max_total_pages=max_total_pages,
            max_depth=max_depth,
            max_site_depth=max_site_depth,
            max_llm_calls=max_llm_calls,
            max_pages_per_domain=max_pages_per_domain,
            politeness_delay=politeness_delay,
            max_link_candidates=max_link_candidates,
            max_reseed_rounds=max_reseed_rounds,
            language=language,
        )
        self._selenium_client = selenium_client
        self._headless = headless
        self._browser = browser
        self._page_load_timeout = page_load_timeout
        self._proxies = proxies
        self._verify_ssl = verify_ssl
        self._driver_path = driver_path

    def _get_selenium_client(self):
        """Get or lazily create the Selenium browser client."""
        if self._selenium_client is None:
            from ..search.selenium_browser import SeleniumBrowser
            self._selenium_client = SeleniumBrowser(
                headless=self._headless,
                browser=self._browser,
                timeout=self._page_load_timeout,
                proxies=self._proxies,
                verify_ssl=self._verify_ssl,
                driver_path=self._driver_path,
            )
        return self._selenium_client

    def _start_browser(self) -> Optional[str]:
        """Eagerly launch the browser so failures surface once and clearly.

        Returns None on success, or an error message string on failure.
        Without this, a browser that never launches would make every page
        fetch fail silently and the crawl would just return zero pages.
        """
        try:
            self._get_selenium_client()._get_driver()
            return None
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    def crawl_and_evaluate(
        self,
        queries: List[str],
        section_context: str,
        research_topic: str = "",
        max_pages_per_query: int = 3,
        min_relevance_score: float = 0.2,
        progress_callback: Callable[[str, int, int], None] = None,
    ) -> CrawlResult:
        """Validate the browser can launch, then run the normal AI crawl.

        If the browser cannot start (missing/mismatched WebDriver, blocked
        auto-download behind a proxy, etc.) a CRITICAL ResearchWarning with
        actionable guidance is recorded and an empty result is returned,
        instead of silently producing no pages.
        """
        err = self._start_browser()
        if err is not None:
            hint = (
                f"Selenium browser ({self._browser}) failed to launch, so the "
                f"'ai_crawl_selenium' crawl collected nothing. Set the WebDriver "
                f"path (driver_path / SELENIUM_DRIVER_PATH) to a local "
                f"{self._browser} driver matching the installed browser version, "
                f"or switch crawl_mode to 'aicrawl' (no browser required). "
                f"Underlying error: {err}"
            )
            print(f"[AICrawlerSelenium] {hint}")
            ResearchWarnings.get_instance().add(
                ResearchWarnings.CRITICAL, "AICrawlerSelenium", hint,
            )
            return CrawlResult(
                pages=[],
                total_fetch_time=0.0,
                total_eval_time=0.0,
                pages_fetched=0,
                pages_filtered=0,
                pages_evaluated=0,
                errors=[hint],
            )
        return super().crawl_and_evaluate(
            queries=queries,
            section_context=section_context,
            research_topic=research_topic,
            max_pages_per_query=max_pages_per_query,
            min_relevance_score=min_relevance_score,
            progress_callback=progress_callback,
        )

    def _fetch_page(self, url: str):
        """Fetch a page with the Selenium browser (JS-rendered content)."""
        return self._get_selenium_client().get_page_content(url)

    def close(self) -> None:
        """Release the browser driver."""
        if self._selenium_client is not None:
            try:
                self._selenium_client.close()
            except Exception:
                pass
