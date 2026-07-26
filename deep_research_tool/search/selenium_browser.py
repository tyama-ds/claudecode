"""
Selenium browser-based web search and content extraction.
"""

import os
import threading
import time
from typing import List, Optional
from pathlib import Path
from bs4 import BeautifulSoup

from .base import BaseSearchClient, SearchResult, PageContent


# DuckDuckGo kl language part -> Google hl code (locale search support)
_KL_LANG_TO_GOOGLE_HL = {
    "jp": "ja", "kr": "ko", "tzh": "zh-TW", "zh": "zh-CN",
    "pt": "pt-BR", "en": "en", "de": "de", "fr": "fr", "es": "es",
    "it": "it", "nl": "nl", "pl": "pl", "ru": "ru", "tr": "tr",
    "th": "th", "vi": "vi", "id": "id", "ms": "ms", "ar": "ar",
}


def ddg_search_url(query: str, region: str) -> str:
    """Direct DuckDuckGo results URL with a locale (kl) parameter."""
    from urllib.parse import quote_plus
    return f"https://duckduckgo.com/?q={quote_plus(query)}&kl={region}"


def google_search_url(query: str, region: str) -> str:
    """Direct Google results URL with gl (country) / hl (UI language)
    derived from a DuckDuckGo-style region code like 'de-de' or 'jp-jp'."""
    from urllib.parse import quote_plus
    parts = (region or "").split("-", 1)
    cc = parts[0] if parts else ""
    lang = parts[1] if len(parts) > 1 else ""
    hl = _KL_LANG_TO_GOOGLE_HL.get(lang, lang or "en")
    params = f"q={quote_plus(query)}&hl={hl}"
    if cc and cc not in ("wt", "xa"):    # wt-wt: worldwide, xa-*: no country
        params += f"&gl={cc}"
    return f"https://www.google.com/search?{params}"


class SeleniumBrowser(BaseSearchClient):
    """Selenium-based browser for dynamic content extraction."""

    def __init__(
        self,
        max_results: int = 10,
        timeout: int = 30,
        headless: bool = True,
        browser: str = "chrome",
        extract_images: bool = True,
        max_images: int = 5,
        implicit_wait: int = 10,
        proxies: dict = None,
        verify_ssl: bool = True,
        driver_path: str = None,
    ):
        """
        Initialize Selenium browser client.

        Args:
            max_results: Maximum number of search results
            timeout: Page load timeout in seconds
            headless: Run browser in headless mode
            browser: Browser type (chrome, edge, firefox)
            extract_images: Whether to extract images from pages
            max_images: Maximum number of images to extract per page
            implicit_wait: Implicit wait time for elements
            proxies: Proxy dict like {"https": "http://proxy:8080"}. The
                browser is launched with this proxy. Note: URL-embedded
                credentials are not supported by Chromium's --proxy-server;
                use an unauthenticated proxy URL for the browser.
            verify_ssl: When False, the browser ignores certificate errors
                (for self-signed proxy certificates)
            driver_path: Path to a local WebDriver executable (chromedriver /
                msedgedriver / geckodriver). When None, falls back to the
                SELENIUM_DRIVER_PATH env var, then webdriver-manager download,
                then Selenium Manager. Set this explicitly in offline or
                proxy-restricted environments.
        """
        super().__init__(
            max_results=max_results,
            timeout=timeout,
            extract_images=extract_images,
            max_images=max_images,
        )
        self.headless = headless
        self.browser = browser
        self.implicit_wait = implicit_wait
        self.proxies = proxies
        self.verify_ssl = verify_ssl
        self.driver_path = driver_path or os.getenv("SELENIUM_DRIVER_PATH")
        self._driver = None
        # A Selenium WebDriver is NOT thread-safe: one browser session
        # holds one navigation state, so concurrent search() /
        # get_page_content() calls from parallel workers would corrupt
        # each other. All driver access is serialized on this lock —
        # parallel callers (multilingual search, research rounds) simply
        # queue on it instead of sharing the driver mid-navigation.
        self._driver_lock = threading.RLock()

    def _proxy_url(self) -> str:
        """Proxy URL for the browser (credentials stripped if present)."""
        if not self.proxies:
            return ""
        url = self.proxies.get("https") or self.proxies.get("http") or ""
        if "@" in url:
            # Chromium/--proxy-server does not accept embedded credentials
            from urllib.parse import urlparse
            parsed = urlparse(url)
            host = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port else ""
            url = f"{parsed.scheme}://{host}{port}"
        return url

    def _chromium_proxy_args(self) -> list:
        """Chromium (Chrome/Edge) launch arguments for proxy/SSL settings."""
        args = []
        proxy_url = self._proxy_url()
        if proxy_url:
            args.append(f"--proxy-server={proxy_url}")
        if not self.verify_ssl:
            args.append("--ignore-certificate-errors")
        return args

    def _get_driver(self):
        """Get or create Selenium WebDriver."""
        if self._driver is None:
            self._driver = self._create_driver()
        return self._driver

    def _make_service(self, service_cls, manager_factory):
        """Build a WebDriver Service.

        Priority: explicit driver_path > webdriver-manager download >
        None (Selenium Manager, built into Selenium 4.6+, resolves the
        driver itself when no Service is passed).
        """
        if self.driver_path:
            return service_cls(executable_path=self.driver_path)
        try:
            return service_cls(manager_factory())
        except Exception as e:
            print(f"[SeleniumBrowser] webdriver-manager could not provide a "
                  f"driver ({e}); falling back to Selenium Manager. If this "
                  f"fails too, set driver_path (or SELENIUM_DRIVER_PATH) to a "
                  f"local WebDriver executable.")
            return None

    @staticmethod
    def _apply_anti_automation_options(options) -> None:
        """Best-effort anti-automation flags for Chromium (Chrome/Edge).

        These experimental options are cosmetic (hide the "automation"
        banner) but on some Selenium / Edge / Chrome version combinations
        they raise InvalidArgumentException and abort the whole launch.
        Applying them best-effort keeps the browser starting no matter what.
        """
        for key, value in (
            ("excludeSwitches", ["enable-automation"]),
            ("useAutomationExtension", False),
        ):
            try:
                options.add_experimental_option(key, value)
            except Exception as e:
                print(f"[SeleniumBrowser] skipping experimental option "
                      f"{key!r} ({e})")

    def _create_driver(self):
        """Create a new Selenium WebDriver."""
        try:
            if self.browser.lower() == "chrome":
                return self._create_chrome_driver()
            elif self.browser.lower() == "edge":
                return self._create_edge_driver()
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

        # Common options for stability
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        # Disable automation flags (best-effort; never abort launch)
        self._apply_anti_automation_options(options)

        # Proxy / SSL settings
        for arg in self._chromium_proxy_args():
            options.add_argument(arg)
        if not self.verify_ssl:
            options.set_capability("acceptInsecureCerts", True)

        service = self._make_service(Service, lambda: ChromeDriverManager().install())
        if service:
            driver = webdriver.Chrome(service=service, options=options)
        else:
            driver = webdriver.Chrome(options=options)

        driver.set_page_load_timeout(self.timeout)
        driver.implicitly_wait(self.implicit_wait)

        return driver

    def _create_edge_driver(self):
        """Create Microsoft Edge WebDriver (Chromium-based)."""
        from selenium import webdriver
        from selenium.webdriver.edge.options import Options
        from selenium.webdriver.edge.service import Service
        from webdriver_manager.microsoft import EdgeChromiumDriverManager

        options = Options()
        if self.headless:
            options.add_argument("--headless=new")

        # Common options for stability (Edge is Chromium-based)
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 "
            "Safari/537.36 Edg/120.0.0.0"
        )

        # Disable automation flags (best-effort; never abort launch)
        self._apply_anti_automation_options(options)

        # Proxy / SSL settings
        for arg in self._chromium_proxy_args():
            options.add_argument(arg)
        if not self.verify_ssl:
            options.set_capability("acceptInsecureCerts", True)

        service = self._make_service(Service, lambda: EdgeChromiumDriverManager().install())
        if service:
            driver = webdriver.Edge(service=service, options=options)
        else:
            driver = webdriver.Edge(options=options)

        driver.set_page_load_timeout(self.timeout)
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

        # Proxy settings via Firefox preferences
        proxy_url = self._proxy_url()
        if proxy_url:
            from urllib.parse import urlparse
            parsed = urlparse(proxy_url)
            host = parsed.hostname or ""
            port = parsed.port or 8080
            options.set_preference("network.proxy.type", 1)
            options.set_preference("network.proxy.http", host)
            options.set_preference("network.proxy.http_port", port)
            options.set_preference("network.proxy.ssl", host)
            options.set_preference("network.proxy.ssl_port", port)
        if not self.verify_ssl:
            options.set_capability("acceptInsecureCerts", True)

        service = self._make_service(Service, lambda: GeckoDriverManager().install())
        if service:
            driver = webdriver.Firefox(service=service, options=options)
        else:
            driver = webdriver.Firefox(options=options)

        driver.set_page_load_timeout(self.timeout)
        driver.implicitly_wait(self.implicit_wait)

        return driver

    def search(
        self,
        query: str,
        search_engine: str = "duckduckgo",
        max_results: Optional[int] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform a web search using the browser.

        Args:
            query: The search query
            search_engine: Search engine to use (duckduckgo, google)
            max_results: Override default max results
            **kwargs: Additional search parameters

        Returns:
            List of search results
        """
        # ORDER MATTERS: the run/process LEAF PERMIT is taken FIRST,
        # then the driver lock — a thread queuing for a permit must
        # never sit on the driver lock, and browser work counts against
        # the app-wide concurrency limit like any other network I/O.
        with self._leaf_permit():
            with self._driver_lock:
                driver = self._get_driver()
                max_results = max_results or self.max_results
                region = kwargs.get("region") or None
                if region in ("wt-wt", ""):
                    region = None
                label = f" [{region}]" if region else ""
                print(f"[Selenium/{search_engine}] Searching{label}: "
                      f"{query}")

                if search_engine.lower() == "duckduckgo":
                    return self._search_duckduckgo(
                        driver, query, max_results, region=region)
                elif search_engine.lower() == "google":
                    return self._search_google(
                        driver, query, max_results, region=region)
                raise ValueError(
                    f"Unsupported search engine: {search_engine}")

    def _search_duckduckgo(
        self,
        driver,
        query: str,
        max_results: int,
        region: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search using DuckDuckGo (locale search via the kl parameter)."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
            if region:
                # locale search: navigate straight to region-scoped results
                driver.get(ddg_search_url(query, region))
            else:
                # Navigate to DuckDuckGo
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
            results = []
            result_elements = driver.find_elements(
                By.CSS_SELECTOR, "[data-result='web-result'], .result"
            )

            for element in result_elements[:max_results]:
                try:
                    title_elem = element.find_element(By.CSS_SELECTOR, "h2 a, .result__title a")
                    title = title_elem.text
                    url = title_elem.get_attribute("href")

                    snippet_elem = element.find_element(
                        By.CSS_SELECTOR, ".result__snippet, [data-result='snippet']"
                    )
                    snippet = snippet_elem.text if snippet_elem else ""

                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        metadata={
                            "source": "duckduckgo_selenium",
                            "query": query,
                        }
                    ))
                except Exception:
                    continue

            return results

        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []

    def _search_google(
        self,
        driver,
        query: str,
        max_results: int,
        region: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search using Google (locale search via gl / hl parameters)."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
            if region:
                # locale search: results scoped to the region's country
                # (gl) and language (hl)
                driver.get(google_search_url(query, region))
                try:
                    consent_button = driver.find_element(By.ID, "L2AGLb")
                    consent_button.click()
                    time.sleep(1)
                except Exception:
                    pass
            else:
                # Navigate to Google
                driver.get("https://www.google.com/")

                # Handle consent if present
                try:
                    consent_button = driver.find_element(By.ID, "L2AGLb")
                    consent_button.click()
                    time.sleep(1)
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
            results = []
            result_elements = driver.find_elements(By.CSS_SELECTOR, "div.g")

            for element in result_elements[:max_results]:
                try:
                    title_elem = element.find_element(By.CSS_SELECTOR, "h3")
                    title = title_elem.text

                    link_elem = element.find_element(By.CSS_SELECTOR, "a")
                    url = link_elem.get_attribute("href")

                    try:
                        snippet_elem = element.find_element(
                            By.CSS_SELECTOR, "[data-sncf], .VwiC3b"
                        )
                        snippet = snippet_elem.text
                    except Exception:
                        snippet = ""

                    if url and not url.startswith("https://www.google.com"):
                        results.append(SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            metadata={
                                "source": "google_selenium",
                                "query": query,
                            }
                        ))
                except Exception:
                    continue

            return results

        except Exception as e:
            print(f"Google search error: {e}")
            return []

    def get_page_content(
        self,
        url: str,
        wait_for_dynamic: bool = True,
        scroll_to_load: bool = True,
        extract_images: Optional[bool] = None,
        **kwargs
    ) -> PageContent:
        """Thread-safe wrapper: one navigation at a time per driver.

        Leaf permit FIRST (browser fetches honor the app-wide limit),
        driver lock second (permit-waiting threads never hold it)."""
        with self._leaf_permit():
            with self._driver_lock:
                return self._get_page_content_impl(
                    url, wait_for_dynamic=wait_for_dynamic,
                    scroll_to_load=scroll_to_load,
                    extract_images=extract_images, **kwargs)

    def _get_page_content_impl(
        self,
        url: str,
        wait_for_dynamic: bool = True,
        scroll_to_load: bool = True,
        extract_images: Optional[bool] = None,
        **kwargs
    ) -> PageContent:
        """
        Extract content from a URL using Selenium.

        Args:
            url: The URL to fetch
            wait_for_dynamic: Wait for dynamic content to load
            scroll_to_load: Scroll page to trigger lazy loading
            extract_images: Override default image extraction setting
            **kwargs: Additional parameters

        Returns:
            Extracted page content
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        driver = self._get_driver()
        extract_images = extract_images if extract_images is not None else self.extract_images

        try:
            # Navigate to URL
            driver.get(url)

            # Wait for page to load
            if wait_for_dynamic:
                WebDriverWait(driver, self.timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                time.sleep(2)  # Additional wait for JavaScript

            # Scroll to load lazy content
            if scroll_to_load:
                self._scroll_page(driver)

            # Get page source
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, "lxml")

            # Extract title
            title = driver.title or ""

            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()

            # Extract main text content
            text_content = self._extract_main_content(soup)

            # Extract images if enabled
            images = []
            if extract_images:
                images = self._extract_images(soup, url)

            # Extract links
            links = self._extract_links(soup, url)

            # Extract metadata
            metadata = self._extract_metadata(soup)
            metadata["dynamic_load"] = wait_for_dynamic

            return PageContent(
                url=url,
                title=title,
                text_content=text_content,
                html_content=str(soup),
                images=images[:self.max_images],
                links=links,
                metadata=metadata,
            )

        except Exception as e:
            return PageContent(
                url=url,
                title="Error",
                text_content=f"Failed to extract content: {str(e)}",
                metadata={"error": str(e)},
            )

    def _scroll_page(self, driver, scroll_pause: float = 0.5, max_scrolls: int = 5):
        """Scroll page to trigger lazy loading."""
        last_height = driver.execute_script("return document.body.scrollHeight")

        for _ in range(max_scrolls):
            # Scroll down
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause)

            # Check if we've reached the bottom
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # Scroll back to top
        driver.execute_script("window.scrollTo(0, 0);")

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from page."""
        # Try to find main content area
        main_content = None
        for selector in ["main", "article", '[role="main"]', ".content", "#content"]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.body or soup

        # Get text content
        text = main_content.get_text(separator="\n", strip=True)
        return self._clean_text(text)

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[dict]:
        """Extract images from page."""
        from urllib.parse import urljoin

        images = []
        for img in soup.find_all("img"):
            # Check multiple src attributes
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if not src:
                continue

            # Make absolute URL
            if not src.startswith(("http://", "https://", "data:")):
                src = urljoin(base_url, src)

            # Skip data URLs for storage
            if src.startswith("data:"):
                continue

            # Check if valid image
            if not self._is_valid_image_url(src):
                continue

            images.append({
                "src": src,
                "alt": img.get("alt", ""),
                "title": img.get("title", ""),
            })

        return images

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[dict]:
        """Extract links from page."""
        from urllib.parse import urljoin

        links = []
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue

            # Make absolute URL
            if not href.startswith(("http://", "https://")):
                href = urljoin(base_url, href)

            links.append({
                "url": href,
                "text": a.get_text(strip=True)[:100],
            })

        return links[:50]  # Limit to 50 links

    def _extract_metadata(self, soup: BeautifulSoup) -> dict:
        """Extract metadata from page."""
        metadata = {}

        # Extract meta tags
        for meta in soup.find_all("meta"):
            name = meta.get("name", meta.get("property", ""))
            content = meta.get("content", "")
            if name and content:
                metadata[name] = content

        return metadata

    def take_screenshot(self, save_path: Path) -> bool:
        """
        Take a screenshot of the current page.

        Args:
            save_path: Path to save the screenshot

        Returns:
            True if successful
        """
        try:
            with self._leaf_permit():
                with self._driver_lock:
                    driver = self._get_driver()
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    driver.save_screenshot(str(save_path))
            return True
        except Exception:
            return False

    def execute_javascript(self, script: str) -> any:
        """
        Execute JavaScript on the current page.

        Args:
            script: JavaScript code to execute

        Returns:
            Result of the script execution
        """
        with self._leaf_permit():
            with self._driver_lock:
                driver = self._get_driver()
                return driver.execute_script(script)

    def close(self):
        """Close the browser and clean up resources."""
        with self._driver_lock:
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
