"""
Selenium browser-based web search and content extraction.
"""

import time
from typing import List, Optional
from pathlib import Path
from bs4 import BeautifulSoup

from .base import BaseSearchClient, SearchResult, PageContent


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
    ):
        """
        Initialize Selenium browser client.

        Args:
            max_results: Maximum number of search results
            timeout: Page load timeout in seconds
            headless: Run browser in headless mode
            browser: Browser type (chrome, firefox)
            extract_images: Whether to extract images from pages
            max_images: Maximum number of images to extract per page
            implicit_wait: Implicit wait time for elements
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
        self._driver = None

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

        # Common options for stability
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        # Disable automation flags
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

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

        service = Service(GeckoDriverManager().install())
        driver = webdriver.Firefox(service=service, options=options)

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
        driver = self._get_driver()
        max_results = max_results or self.max_results
        print(f"[Selenium/{search_engine}] Searching: {query}")

        if search_engine.lower() == "duckduckgo":
            return self._search_duckduckgo(driver, query, max_results)
        elif search_engine.lower() == "google":
            return self._search_google(driver, query, max_results)
        else:
            raise ValueError(f"Unsupported search engine: {search_engine}")

    def _search_duckduckgo(
        self,
        driver,
        query: str,
        max_results: int
    ) -> List[SearchResult]:
        """Search using DuckDuckGo."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
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
        max_results: int
    ) -> List[SearchResult]:
        """Search using Google."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
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
        driver = self._get_driver()
        return driver.execute_script(script)

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
