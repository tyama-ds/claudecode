"""
AI Crawler (Selenium variant) - LLM-driven crawling with browser-based fetching.

Same crawl logic as AICrawler, but pages are fetched through the existing
SeleniumBrowser client so JavaScript-rendered sites can be read. Search
seeding still goes through the regular search client; only page fetching
uses the browser.
"""

from typing import Optional

from ..evidence.content_filter import ContentFilter
from .ai_crawler import AICrawler


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
            browser: Browser type ("chrome" or "firefox")
            page_load_timeout: Page load timeout in seconds
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

    def _get_selenium_client(self):
        """Get or lazily create the Selenium browser client."""
        if self._selenium_client is None:
            from ..search.selenium_browser import SeleniumBrowser
            self._selenium_client = SeleniumBrowser(
                headless=self._headless,
                browser=self._browser,
                timeout=self._page_load_timeout,
            )
        return self._selenium_client

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
