"""
DuckDuckGo search client implementation.
"""

import time
from typing import List, Optional
import requests
from bs4 import BeautifulSoup

from .base import BaseSearchClient, SearchResult, PageContent


class DuckDuckGoSearch(BaseSearchClient):
    """DuckDuckGo search client using duckduckgo-search library."""

    def __init__(
        self,
        max_results: int = 10,
        timeout: int = 30,
        region: str = "wt-wt",
        safe_search: str = "moderate",
        extract_images: bool = True,
        max_images: int = 5,
        proxies: dict = None,
        verify_ssl: bool = True,
    ):
        """
        Initialize DuckDuckGo search client.

        Args:
            max_results: Maximum number of search results
            timeout: Request timeout in seconds
            region: Search region (default: worldwide)
            safe_search: Safe search level (off, moderate, strict)
            extract_images: Whether to extract images from pages
            max_images: Maximum number of images to extract per page
            proxies: Proxy settings dict (e.g., {"http": "http://proxy:8080", "https": "http://proxy:8080"})
            verify_ssl: Verify SSL certificates
        """
        super().__init__(
            max_results=max_results,
            timeout=timeout,
            extract_images=extract_images,
            max_images=max_images,
        )
        self.region = region
        self.safe_search = safe_search
        self.proxies = proxies
        self.verify_ssl = verify_ssl
        self._ddgs = None

    def _get_ddgs(self):
        """Get or create DuckDuckGo search instance."""
        if self._ddgs is None:
            try:
                from duckduckgo_search import DDGS

                # DDGS supports proxy via proxies parameter
                if self.proxies:
                    proxy_url = self.proxies.get("https") or self.proxies.get("http")
                    self._ddgs = DDGS(proxy=proxy_url)
                else:
                    self._ddgs = DDGS()

            except ImportError:
                raise ImportError(
                    "duckduckgo-search package not installed. "
                    "Install with: pip install duckduckgo-search"
                )
        return self._ddgs

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform a DuckDuckGo web search.

        Args:
            query: The search query
            max_results: Override default max results
            **kwargs: Additional search parameters

        Returns:
            List of search results
        """
        ddgs = self._get_ddgs()
        max_results = max_results or self.max_results

        try:
            results = ddgs.text(
                query,
                region=kwargs.get("region", self.region),
                safesearch=kwargs.get("safe_search", self.safe_search),
                max_results=max_results,
            )

            search_results = []
            for result in results:
                search_results.append(SearchResult(
                    title=result.get("title", ""),
                    url=result.get("href", result.get("link", "")),
                    snippet=result.get("body", result.get("snippet", "")),
                    metadata={
                        "source": "duckduckgo",
                        "query": query,
                    }
                ))

            return search_results

        except Exception as e:
            # Return empty results on error, log the error
            print(f"DuckDuckGo search error: {e}")
            return []

    def search_news(
        self,
        query: str,
        max_results: Optional[int] = None,
        timelimit: Optional[str] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search DuckDuckGo news.

        Args:
            query: The search query
            max_results: Override default max results
            timelimit: Time limit (d: day, w: week, m: month)
            **kwargs: Additional search parameters

        Returns:
            List of news search results
        """
        ddgs = self._get_ddgs()
        max_results = max_results or self.max_results

        try:
            results = ddgs.news(
                query,
                region=kwargs.get("region", self.region),
                safesearch=kwargs.get("safe_search", self.safe_search),
                timelimit=timelimit,
                max_results=max_results,
            )

            search_results = []
            for result in results:
                search_results.append(SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", result.get("link", "")),
                    snippet=result.get("body", result.get("excerpt", "")),
                    metadata={
                        "source": "duckduckgo_news",
                        "query": query,
                        "date": result.get("date", ""),
                        "publisher": result.get("source", ""),
                    }
                ))

            return search_results

        except Exception as e:
            print(f"DuckDuckGo news search error: {e}")
            return []

    def search_images(
        self,
        query: str,
        max_results: Optional[int] = None,
        **kwargs
    ) -> List[dict]:
        """
        Search for images on DuckDuckGo.

        Args:
            query: The search query
            max_results: Override default max results
            **kwargs: Additional search parameters

        Returns:
            List of image results
        """
        ddgs = self._get_ddgs()
        max_results = max_results or self.max_images

        try:
            results = ddgs.images(
                query,
                region=kwargs.get("region", self.region),
                safesearch=kwargs.get("safe_search", self.safe_search),
                max_results=max_results,
            )

            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("image", ""),
                    "thumbnail": r.get("thumbnail", ""),
                    "source": r.get("url", ""),
                }
                for r in results
            ]

        except Exception as e:
            print(f"DuckDuckGo image search error: {e}")
            return []

    def get_page_content(
        self,
        url: str,
        extract_images: Optional[bool] = None,
        **kwargs
    ) -> PageContent:
        """
        Extract content from a URL using requests and BeautifulSoup.

        Args:
            url: The URL to fetch
            extract_images: Override default image extraction setting
            **kwargs: Additional parameters

        Returns:
            Extracted page content
        """
        extract_images = extract_images if extract_images is not None else self.extract_images

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=True,
                proxies=self.proxies,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            response.encoding = response.apparent_encoding

            soup = BeautifulSoup(response.text, "lxml")

            # Extract title
            title = ""
            if soup.title:
                title = soup.title.string or ""

            # Remove script and style elements
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
            src = img.get("src", "")
            if not src:
                continue

            # Make absolute URL
            if not src.startswith(("http://", "https://")):
                src = urljoin(base_url, src)

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
            if not href or href.startswith("#"):
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

    def close(self):
        """Clean up resources."""
        self._ddgs = None
