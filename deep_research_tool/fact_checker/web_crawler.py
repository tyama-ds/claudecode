"""
Recursive web crawler for fact-checking evidence gathering.

Features:
- Multi-engine search (Google, DuckDuckGo, Bing) via Selenium
- Recursive page crawling with depth control
- Advanced ad filtering
- Clean text extraction
- Reference URL fetching
"""

import re
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Any
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup


class SearchEngine:
    """Supported search engines."""
    GOOGLE = "google"
    DUCKDUCKGO = "duckduckgo"
    BING = "bing"


@dataclass
class CrawlResult:
    """Result from crawling a single page."""
    url: str
    title: str
    text_content: str
    clean_text: str  # Text without ads/navigation
    links: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_ad_page: bool = False
    crawl_depth: int = 0
    content_hash: str = ""
    word_count: int = 0

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.clean_text.encode()).hexdigest()
        if not self.word_count:
            self.word_count = len(self.clean_text.split())


@dataclass
class SearchResultItem:
    """A single search result."""
    title: str
    url: str
    snippet: str
    source_engine: str


class AdFilter:
    """
    Filter for detecting and removing advertisements.

    Uses multiple strategies:
    - Domain-based blocking (known ad networks)
    - CSS class/ID-based detection
    - Content pattern detection
    - Structural analysis
    """

    # Known ad network domains
    AD_DOMAINS = {
        'doubleclick.net', 'googlesyndication.com', 'googleadservices.com',
        'google-analytics.com', 'googletagmanager.com', 'facebook.com/tr',
        'amazon-adsystem.com', 'advertising.com', 'adnxs.com', 'adsrvr.org',
        'criteo.com', 'outbrain.com', 'taboola.com', 'zedo.com', 'adroll.com',
        'adcolony.com', 'applovin.com', 'unity3d.com', 'chartboost.com',
        'a8.net', 'accesstrade.net', 'valuecommerce.com', 'felmat.net',
        'i-mobile.co.jp', 'nend.net', 'microad.jp',
    }

    # CSS selectors commonly used for ads
    AD_SELECTORS = [
        # Common ad containers
        '[class*="ad-"]', '[class*="-ad"]', '[class*="_ad"]',
        '[class*="advertisement"]', '[class*="sponsored"]',
        '[class*="promo"]', '[class*="banner"]',
        '[id*="ad-"]', '[id*="-ad"]', '[id*="_ad"]',
        '[id*="advertisement"]', '[id*="sponsored"]',
        # Google ads
        '.adsbygoogle', 'ins.adsbygoogle',
        # Common ad frameworks
        '[data-ad]', '[data-ad-unit]', '[data-advertisement]',
        '[data-google-query-id]',
        # Social plugins (often promotional)
        '.fb-like', '.twitter-share',
        # Popup/overlay
        '[class*="popup"]', '[class*="modal"]', '[class*="overlay"]',
        # Sidebar promotions
        '[class*="sidebar-ad"]', '[class*="widget-ad"]',
        # Japanese ad frameworks
        '[class*="pr-"]', '[class*="koukoku"]', '[class*="広告"]',
    ]

    # Navigation/footer selectors to exclude
    NAVIGATION_SELECTORS = [
        'nav', 'header', 'footer', 'aside',
        '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]',
        '.nav', '.navigation', '.menu', '.header', '.footer', '.sidebar',
        '#nav', '#navigation', '#menu', '#header', '#footer', '#sidebar',
        '.breadcrumb', '.pagination',
    ]

    # Patterns indicating ad content
    AD_TEXT_PATTERNS = [
        r'\b(?:sponsored|advertisement|広告|PR|プロモーション)\b',
        r'\b(?:click here|buy now|limited offer|今すぐ購入)\b',
        r'\b(?:free trial|無料トライアル)\b',
    ]

    def __init__(self, strict_mode: bool = False):
        """
        Initialize ad filter.

        Args:
            strict_mode: If True, apply more aggressive filtering
        """
        self.strict_mode = strict_mode

    def is_ad_url(self, url: str) -> bool:
        """Check if URL is from a known ad network."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        for ad_domain in self.AD_DOMAINS:
            if ad_domain in domain:
                return True

        return False

    def filter_soup(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove ad elements from BeautifulSoup object."""
        # Remove elements matching ad selectors
        for selector in self.AD_SELECTORS:
            try:
                for element in soup.select(selector):
                    element.decompose()
            except Exception:
                continue

        # Remove navigation elements
        for selector in self.NAVIGATION_SELECTORS:
            try:
                for element in soup.select(selector):
                    element.decompose()
            except Exception:
                continue

        # Remove script and style tags
        for tag in ['script', 'style', 'noscript', 'iframe']:
            for element in soup.find_all(tag):
                element.decompose()

        # Remove elements with ad-related text content
        if self.strict_mode:
            for pattern in self.AD_TEXT_PATTERNS:
                for element in soup.find_all(string=re.compile(pattern, re.IGNORECASE)):
                    if element.parent:
                        element.parent.decompose()

        return soup

    def clean_text(self, text: str) -> str:
        """Clean text by removing ad-related content."""
        # Remove lines that look like ads
        lines = text.split('\n')
        clean_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip short lines that are likely navigation/buttons
            if len(line) < 10 and not re.search(r'[.。!！?？]', line):
                continue

            # Skip lines matching ad patterns
            is_ad = False
            for pattern in self.AD_TEXT_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    is_ad = True
                    break

            if not is_ad:
                clean_lines.append(line)

        return '\n'.join(clean_lines)


class RecursiveWebCrawler:
    """
    Recursive web crawler with search engine integration.

    Supports:
    - Multiple search engines (Google, DuckDuckGo, Bing)
    - Recursive page crawling with configurable depth
    - Ad filtering
    - Content deduplication
    """

    def __init__(
        self,
        headless: bool = True,
        browser: str = "chrome",
        search_engine: str = SearchEngine.GOOGLE,
        max_search_results: int = 10,
        max_crawl_depth: int = 1,
        max_pages_per_depth: int = 3,
        page_load_timeout: int = 30,
        implicit_wait: int = 10,
        ad_filter: Optional[AdFilter] = None,
        respect_robots: bool = True,
    ):
        """
        Initialize the recursive web crawler.

        Args:
            headless: Run browser in headless mode
            browser: Browser type (chrome, firefox)
            search_engine: Default search engine
            max_search_results: Maximum search results to fetch
            max_crawl_depth: Maximum depth for recursive crawling
            max_pages_per_depth: Maximum pages to crawl per depth level
            page_load_timeout: Page load timeout in seconds
            implicit_wait: Implicit wait time for elements
            ad_filter: Ad filter instance
            respect_robots: Respect robots.txt (not fully implemented)
        """
        self.headless = headless
        self.browser = browser
        self.search_engine = search_engine
        self.max_search_results = max_search_results
        self.max_crawl_depth = max_crawl_depth
        self.max_pages_per_depth = max_pages_per_depth
        self.page_load_timeout = page_load_timeout
        self.implicit_wait = implicit_wait
        self.ad_filter = ad_filter or AdFilter()
        self.respect_robots = respect_robots

        self._driver = None
        self._visited_urls: Set[str] = set()
        self._content_hashes: Set[str] = set()

    def _get_driver(self):
        """Get or create Selenium WebDriver."""
        if self._driver is None:
            self._driver = self._create_driver()
        return self._driver

    def _create_driver(self):
        """Create a new Selenium WebDriver."""
        try:
            if self.browser.lower() == "chrome":
                return self._create_chrome_driver()
            elif self.browser.lower() == "firefox":
                return self._create_firefox_driver()
            else:
                raise ValueError(f"Unsupported browser: {self.browser}")
        except ImportError as e:
            raise ImportError(
                "Selenium packages not installed. Install with: "
                "pip install selenium webdriver-manager"
            ) from e

    def _create_chrome_driver(self):
        """Create Chrome WebDriver."""
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager

        options = Options()
        if self.headless:
            options.add_argument("--headless=new")

        # Stability options
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-popup-blocking")

        # User agent
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        # Disable automation flags
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        # Block images and ads for faster loading (optional)
        prefs = {
            "profile.managed_default_content_settings.images": 2,  # Disable images
        }
        options.add_experimental_option("prefs", prefs)

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        driver.set_page_load_timeout(self.page_load_timeout)
        driver.implicitly_wait(self.implicit_wait)

        return driver

    def _create_firefox_driver(self):
        """Create Firefox WebDriver."""
        from selenium import webdriver
        from selenium.webdriver.firefox.options import Options
        from selenium.webdriver.firefox.service import Service
        from webdriver_manager.firefox import GeckoDriverManager

        options = Options()
        if self.headless:
            options.add_argument("--headless")

        service = Service(GeckoDriverManager().install())
        driver = webdriver.Firefox(service=service, options=options)

        driver.set_page_load_timeout(self.page_load_timeout)
        driver.implicitly_wait(self.implicit_wait)

        return driver

    def search(
        self,
        query: str,
        engine: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[SearchResultItem]:
        """
        Perform a web search.

        Args:
            query: Search query
            engine: Search engine (overrides default)
            max_results: Maximum results (overrides default)

        Returns:
            List of search results
        """
        engine = engine or self.search_engine
        max_results = max_results or self.max_search_results

        if engine == SearchEngine.GOOGLE:
            return self._search_google(query, max_results)
        elif engine == SearchEngine.DUCKDUCKGO:
            return self._search_duckduckgo(query, max_results)
        elif engine == SearchEngine.BING:
            return self._search_bing(query, max_results)
        else:
            raise ValueError(f"Unsupported search engine: {engine}")

    def _search_google(self, query: str, max_results: int) -> List[SearchResultItem]:
        """Search using Google."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        driver = self._get_driver()
        results = []

        try:
            driver.get("https://www.google.com/")

            # Handle consent if present
            try:
                consent_buttons = driver.find_elements(By.CSS_SELECTOR, "[id='L2AGLb'], [id='W0wltc']")
                for btn in consent_buttons:
                    if btn.is_displayed():
                        btn.click()
                        time.sleep(1)
                        break
            except Exception:
                pass

            # Find search box and enter query
            search_box = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.NAME, "q"))
            )
            search_box.clear()
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)

            # Wait for results
            time.sleep(2)

            # Extract results
            result_elements = driver.find_elements(By.CSS_SELECTOR, "div.g")

            for element in result_elements[:max_results]:
                try:
                    title_elem = element.find_element(By.CSS_SELECTOR, "h3")
                    title = title_elem.text

                    link_elem = element.find_element(By.CSS_SELECTOR, "a")
                    url = link_elem.get_attribute("href")

                    # Skip Google URLs
                    if not url or "google.com" in url:
                        continue

                    # Skip ad URLs
                    if self.ad_filter.is_ad_url(url):
                        continue

                    try:
                        snippet_elem = element.find_element(
                            By.CSS_SELECTOR, "[data-sncf], .VwiC3b, .IsZvec"
                        )
                        snippet = snippet_elem.text
                    except Exception:
                        snippet = ""

                    results.append(SearchResultItem(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source_engine="google",
                    ))
                except Exception:
                    continue

        except Exception as e:
            print(f"Google search error: {e}")

        return results

    def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResultItem]:
        """Search using DuckDuckGo."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        driver = self._get_driver()
        results = []

        try:
            driver.get("https://duckduckgo.com/")

            # Find search box and enter query
            search_box = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.NAME, "q"))
            )
            search_box.clear()
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)

            # Wait for results
            time.sleep(2)

            # Extract results
            result_selectors = [
                "[data-result='web-result']",
                ".result",
                "article[data-testid='result']",
            ]

            for selector in result_selectors:
                result_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if result_elements:
                    break

            for element in result_elements[:max_results]:
                try:
                    title_elem = element.find_element(
                        By.CSS_SELECTOR, "h2 a, .result__title a, [data-testid='result-title-a']"
                    )
                    title = title_elem.text
                    url = title_elem.get_attribute("href")

                    if not url or self.ad_filter.is_ad_url(url):
                        continue

                    snippet_elem = element.find_element(
                        By.CSS_SELECTOR, ".result__snippet, [data-result='snippet'], [data-testid='result-snippet']"
                    )
                    snippet = snippet_elem.text if snippet_elem else ""

                    results.append(SearchResultItem(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source_engine="duckduckgo",
                    ))
                except Exception:
                    continue

        except Exception as e:
            print(f"DuckDuckGo search error: {e}")

        return results

    def _search_bing(self, query: str, max_results: int) -> List[SearchResultItem]:
        """Search using Bing."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        driver = self._get_driver()
        results = []

        try:
            driver.get("https://www.bing.com/")

            # Find search box and enter query
            search_box = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.NAME, "q"))
            )
            search_box.clear()
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)

            # Wait for results
            time.sleep(2)

            # Extract results
            result_elements = driver.find_elements(By.CSS_SELECTOR, "#b_results > li.b_algo")

            for element in result_elements[:max_results]:
                try:
                    title_elem = element.find_element(By.CSS_SELECTOR, "h2 a")
                    title = title_elem.text
                    url = title_elem.get_attribute("href")

                    if not url or self.ad_filter.is_ad_url(url):
                        continue

                    try:
                        snippet_elem = element.find_element(By.CSS_SELECTOR, ".b_caption p")
                        snippet = snippet_elem.text
                    except Exception:
                        snippet = ""

                    results.append(SearchResultItem(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source_engine="bing",
                    ))
                except Exception:
                    continue

        except Exception as e:
            print(f"Bing search error: {e}")

        return results

    def crawl_url(self, url: str, depth: int = 0) -> CrawlResult:
        """
        Crawl a single URL.

        Args:
            url: URL to crawl
            depth: Current crawl depth

        Returns:
            CrawlResult with extracted content
        """
        driver = self._get_driver()

        try:
            driver.get(url)
            time.sleep(2)  # Wait for dynamic content

            # Get page source and parse
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, "lxml")

            # Get title
            title = driver.title or soup.title.text if soup.title else ""

            # Filter ads
            clean_soup = self.ad_filter.filter_soup(soup)

            # Extract main content
            text_content = soup.get_text(separator="\n", strip=True)
            clean_text = self._extract_main_content(clean_soup)
            clean_text = self.ad_filter.clean_text(clean_text)

            # Extract links for recursive crawling
            links = self._extract_links(clean_soup, url)

            # Extract metadata
            metadata = self._extract_metadata(clean_soup)

            return CrawlResult(
                url=url,
                title=title,
                text_content=text_content,
                clean_text=clean_text,
                links=links,
                metadata=metadata,
                is_ad_page=False,
                crawl_depth=depth,
            )

        except Exception as e:
            return CrawlResult(
                url=url,
                title="Error",
                text_content="",
                clean_text=f"Failed to crawl: {str(e)}",
                is_ad_page=False,
                crawl_depth=depth,
                metadata={"error": str(e)},
            )

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from page."""
        # Try to find main content area
        main_content = None
        for selector in ["main", "article", '[role="main"]', ".content", "#content",
                         ".post-content", ".article-content", ".entry-content"]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.body or soup

        # Get text content
        text = main_content.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract links from page."""
        links = []
        seen = set()

        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue

            # Make absolute URL
            if not href.startswith(("http://", "https://")):
                href = urljoin(base_url, href)

            # Skip already seen
            if href in seen:
                continue
            seen.add(href)

            # Skip ad URLs
            if self.ad_filter.is_ad_url(href):
                continue

            # Skip non-content URLs
            if self._is_non_content_url(href):
                continue

            links.append(href)

        return links[:50]  # Limit to 50 links

    def _is_non_content_url(self, url: str) -> bool:
        """Check if URL is likely not content (social, login, etc.)."""
        non_content_patterns = [
            r'login', r'signin', r'signup', r'register',
            r'cart', r'checkout', r'payment',
            r'facebook\.com', r'twitter\.com', r'instagram\.com',
            r'linkedin\.com', r'youtube\.com',
            r'\.pdf$', r'\.zip$', r'\.exe$',
            r'mailto:', r'tel:',
        ]

        url_lower = url.lower()
        for pattern in non_content_patterns:
            if re.search(pattern, url_lower):
                return True

        return False

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from page."""
        metadata = {}

        for meta in soup.find_all("meta"):
            name = meta.get("name", meta.get("property", ""))
            content = meta.get("content", "")
            if name and content:
                metadata[name] = content

        return metadata

    def crawl_recursive(
        self,
        start_url: str,
        max_depth: Optional[int] = None,
    ) -> List[CrawlResult]:
        """
        Recursively crawl starting from a URL.

        Args:
            start_url: Starting URL
            max_depth: Maximum depth (overrides default)

        Returns:
            List of CrawlResults from all crawled pages
        """
        max_depth = max_depth if max_depth is not None else self.max_crawl_depth
        results = []
        to_crawl = [(start_url, 0)]  # (url, depth)

        while to_crawl:
            url, depth = to_crawl.pop(0)

            # Skip if already visited
            if url in self._visited_urls:
                continue
            self._visited_urls.add(url)

            # Skip if too deep
            if depth > max_depth:
                continue

            # Crawl the page
            result = self.crawl_url(url, depth)

            # Skip if content already seen (deduplication)
            if result.content_hash in self._content_hashes:
                continue
            self._content_hashes.add(result.content_hash)

            results.append(result)

            # Add links for next level crawling
            if depth < max_depth:
                new_links = result.links[:self.max_pages_per_depth]
                for link in new_links:
                    if link not in self._visited_urls:
                        to_crawl.append((link, depth + 1))

        return results

    def search_and_crawl(
        self,
        query: str,
        engine: Optional[str] = None,
        recursive: bool = True,
    ) -> List[CrawlResult]:
        """
        Search and crawl results.

        Args:
            query: Search query
            engine: Search engine to use
            recursive: Whether to crawl linked pages

        Returns:
            List of CrawlResults
        """
        # Clear visited URLs for new search
        self._visited_urls.clear()
        self._content_hashes.clear()

        # Perform search
        search_results = self.search(query, engine)

        # Crawl each result
        all_results = []
        for sr in search_results:
            if sr.url in self._visited_urls:
                continue

            if recursive:
                results = self.crawl_recursive(sr.url)
            else:
                result = self.crawl_url(sr.url)
                results = [result]

            all_results.extend(results)

        return all_results

    def fetch_reference_url(self, url: str) -> CrawlResult:
        """
        Fetch a specific reference URL (from citations).

        Args:
            url: Reference URL

        Returns:
            CrawlResult with content
        """
        return self.crawl_url(url, depth=0)

    def close(self):
        """Close the browser and clean up resources."""
        if self._driver:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
