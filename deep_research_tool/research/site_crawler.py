"""
Site Crawler for Extended Research Mode.

This module provides functionality to crawl websites starting from a seed URL,
discovering and extracting relevant content from related pages within the same domain.
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Callable, Tuple
from urllib.parse import urlparse, urljoin, urldefrag
from collections import deque


@dataclass
class CrawledPage:
    """Represents a crawled page."""
    url: str
    title: str
    content: str
    links: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    depth: int = 0
    images: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "links_count": len(self.links),
            "relevance_score": self.relevance_score,
            "depth": self.depth,
            "images_count": len(self.images),
        }


@dataclass
class CrawlResult:
    """Result of a site crawl operation."""
    seed_url: str
    root_domain: str
    pages_crawled: int
    pages_relevant: int
    crawled_pages: List[CrawledPage] = field(default_factory=list)
    discovered_topics: List[str] = field(default_factory=list)
    suggested_queries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "seed_url": self.seed_url,
            "root_domain": self.root_domain,
            "pages_crawled": self.pages_crawled,
            "pages_relevant": self.pages_relevant,
            "crawled_pages": [p.to_dict() for p in self.crawled_pages],
            "discovered_topics": self.discovered_topics,
            "suggested_queries": self.suggested_queries,
        }


class SiteCrawler:
    """
    Crawls websites to discover and extract relevant content.

    Used in Extended Mode to perform deep research by exploring
    entire websites rather than just individual pages.
    """

    # Absolute maximum pages to prevent infinite loops
    GLOBAL_MAX_PAGES = 50

    def __init__(
        self,
        search_client,
        llm_client=None,
        max_pages: int = 10,
        max_depth: int = 2,
        relevance_threshold: float = 0.3,
        delay_between_requests: float = 0.5,
        language: str = "ja",
    ):
        """
        Initialize SiteCrawler.

        Args:
            search_client: Client for fetching web pages
            llm_client: Optional LLM client for relevance scoring
            max_pages: Maximum pages to crawl per site
            max_depth: Maximum link depth from seed URL
            relevance_threshold: Minimum relevance score to include page
            delay_between_requests: Delay between requests in seconds
            language: Target language
        """
        self.search = search_client
        self.llm = llm_client
        self.max_pages = min(max_pages, self.GLOBAL_MAX_PAGES)
        self.max_depth = max_depth
        self.relevance_threshold = relevance_threshold
        self.delay = delay_between_requests
        self.language = language
        # Track total pages crawled across all sites
        self._total_pages_crawled = 0

    def reset_page_counter(self) -> None:
        """Reset the global page counter. Call at start of new research session."""
        self._total_pages_crawled = 0

    def get_remaining_pages(self) -> int:
        """Get remaining pages available before hitting global limit."""
        return max(0, self.GLOBAL_MAX_PAGES - self._total_pages_crawled)

    def get_root_domain(self, url: str) -> str:
        """Extract root domain from URL."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def get_domain(self, url: str) -> str:
        """Extract domain (netloc) from URL."""
        return urlparse(url).netloc

    def is_same_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs belong to the same domain."""
        return self.get_domain(url1) == self.get_domain(url2)

    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and trailing slashes."""
        url, _ = urldefrag(url)
        return url.rstrip("/")

    def is_document_url(self, url: str) -> bool:
        """Check if URL is a downloadable document (PDF, XLSX, DOCX, CSV)."""
        doc_extensions = [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".ppt", ".pptx"]
        lower_url = url.lower()
        return any(lower_url.endswith(ext) for ext in doc_extensions)

    def is_valid_page_url(self, url: str) -> bool:
        """Check if URL is a valid page URL (not a file download, etc.).

        Note: PDF/DOCX/XLSX documents are now handled separately via
        is_document_url() and are not skipped outright.
        """
        # Skip binary/media files (but NOT PDF/DOCX/XLSX/CSV - those are now handled)
        skip_extensions = [
            ".zip", ".rar", ".tar", ".gz", ".exe", ".dmg",
            ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
            ".mp3", ".mp4", ".avi", ".mov", ".wmv",
            ".css", ".js", ".json", ".xml",
        ]
        lower_url = url.lower()
        for ext in skip_extensions:
            if lower_url.endswith(ext):
                return False

        # Skip common non-content paths
        skip_patterns = [
            "/login", "/signup", "/register", "/cart", "/checkout",
            "/account", "/admin", "/wp-admin", "/api/",
            "/search?", "/tag/", "/category/", "/author/",
            "/page/", "/feed/", "/rss",
        ]
        for pattern in skip_patterns:
            if pattern in lower_url:
                return False

        return True

    def extract_links(self, base_url: str, html_content: str) -> Tuple[List[str], List[str]]:
        """Extract links from HTML content.

        Returns:
            Tuple of (page_links, document_links) where document_links
            are URLs to PDF/XLSX/DOCX/CSV files.
        """
        page_links = []
        document_links = []

        # Simple regex to find href attributes
        href_pattern = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
        matches = href_pattern.findall(html_content)

        root_domain = self.get_root_domain(base_url)

        for href in matches:
            # Skip anchors, javascript, mailto, etc.
            if href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            # Convert relative URLs to absolute
            if href.startswith("/"):
                full_url = urljoin(root_domain, href)
            elif href.startswith("http"):
                full_url = href
            else:
                full_url = urljoin(base_url, href)

            normalized = self.normalize_url(full_url)

            # Categorize: document link vs page link
            if self.is_document_url(normalized):
                if normalized not in document_links:
                    document_links.append(normalized)
            elif self.is_same_domain(full_url, base_url):
                if self.is_valid_page_url(normalized) and normalized not in page_links:
                    page_links.append(normalized)

        return page_links, document_links

    def score_relevance_simple(
        self,
        content: str,
        title: str,
        research_topic: str,
        keywords: List[str],
    ) -> float:
        """
        Score page relevance using simple keyword matching.

        Args:
            content: Page content
            title: Page title
            research_topic: Main research topic
            keywords: Related keywords

        Returns:
            Relevance score between 0 and 1
        """
        text = f"{title} {content}".lower()
        topic_lower = research_topic.lower()

        score = 0.0

        # Check topic in title (high weight)
        if topic_lower in title.lower():
            score += 0.4

        # Check topic in content
        topic_words = topic_lower.split()
        topic_matches = sum(1 for word in topic_words if word in text)
        score += (topic_matches / max(len(topic_words), 1)) * 0.3

        # Check keywords
        keyword_matches = sum(1 for kw in keywords if kw.lower() in text)
        score += (keyword_matches / max(len(keywords), 1)) * 0.3

        return min(score, 1.0)

    def score_relevance_llm(
        self,
        content: str,
        title: str,
        research_topic: str,
        section_context: str = "",
    ) -> float:
        """
        Score page relevance using LLM.

        Args:
            content: Page content
            title: Page title
            research_topic: Main research topic
            section_context: Current section context

        Returns:
            Relevance score between 0 and 1
        """
        if not self.llm:
            return 0.5

        prompt = f"""Rate the relevance of this webpage to the research topic.

Research Topic: {research_topic}
{f'Section Context: {section_context}' if section_context else ''}

Page Title: {title}
Page Content (excerpt):
{content[:1500]}...

Rate the relevance on a scale of 0 to 1:
- 0.0-0.2: Not relevant
- 0.3-0.5: Somewhat relevant
- 0.6-0.8: Relevant
- 0.9-1.0: Highly relevant

Return only a number (e.g., 0.7):"""

        try:
            response = self.llm.generate(prompt)
            score_text = response.content.strip()
            # Extract first number found
            match = re.search(r"(\d+\.?\d*)", score_text)
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)
        except Exception:
            pass

        return 0.5

    def discover_topics_and_queries(
        self,
        crawled_pages: List[CrawledPage],
        research_topic: str,
        existing_content: str = "",
    ) -> Dict[str, List[str]]:
        """
        Analyze crawled pages to discover new topics and generate follow-up queries.

        Args:
            crawled_pages: List of crawled pages
            research_topic: Main research topic
            existing_content: Already collected content

        Returns:
            Dictionary with 'topics' and 'queries' lists
        """
        if not self.llm or not crawled_pages:
            return {"topics": [], "queries": []}

        # Combine page summaries
        page_summaries = []
        for page in crawled_pages[:5]:  # Limit to top 5 pages
            summary = f"- {page.title}: {page.content[:300]}..."
            page_summaries.append(summary)

        prompt = f"""Analyze these crawled pages related to "{research_topic}" and identify:
1. New sub-topics or aspects discovered that weren't in the existing content
2. Follow-up search queries to investigate these new aspects

Crawled Pages:
{chr(10).join(page_summaries)}

Existing Content Summary:
{existing_content[:500]}...

Return as JSON:
{{
    "discovered_topics": ["topic1", "topic2", ...],
    "suggested_queries": ["query1", "query2", ...]
}}

Focus on specific, actionable topics and queries. Limit to 5 each."""

        try:
            response = self.llm.generate(prompt)
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                import json
                result = json.loads(content[start:end])
                return {
                    "topics": result.get("discovered_topics", [])[:5],
                    "queries": result.get("suggested_queries", [])[:5],
                }
        except Exception:
            pass

        return {"topics": [], "queries": []}

    def crawl_site(
        self,
        seed_url: str,
        research_topic: str,
        keywords: List[str] = None,
        section_context: str = "",
        existing_content: str = "",
        progress_callback: Callable[[str, int, int], None] = None,
    ) -> CrawlResult:
        """
        Crawl a website starting from a seed URL.

        Args:
            seed_url: Starting URL for crawl
            research_topic: Main research topic
            keywords: Related keywords for relevance scoring
            section_context: Current section context
            existing_content: Already collected content
            progress_callback: Callback for progress (message, current, total)

        Returns:
            CrawlResult with crawled pages and discovered information
        """
        keywords = keywords or []
        root_domain = self.get_root_domain(seed_url)

        # BFS crawl
        visited: Set[str] = set()
        queue: deque = deque()
        queue.append((self.normalize_url(seed_url), 0))  # (url, depth)

        crawled_pages: List[CrawledPage] = []
        pages_crawled = 0

        while queue and pages_crawled < self.max_pages:
            # Check global limit to prevent infinite loops
            if self._total_pages_crawled >= self.GLOBAL_MAX_PAGES:
                print(f"Global page limit ({self.GLOBAL_MAX_PAGES}) reached. Stopping crawl.")
                break

            current_url, depth = queue.popleft()

            if current_url in visited:
                continue
            if depth > self.max_depth:
                continue

            visited.add(current_url)

            if progress_callback:
                progress_callback(
                    f"Crawling: {current_url[:50]}...",
                    pages_crawled + 1,
                    self.max_pages
                )

            try:
                # Fetch page (handle both regular pages and document URLs)
                is_doc = self.is_document_url(current_url)
                page = self.search.get_page_content(current_url)
                pages_crawled += 1
                self._total_pages_crawled += 1

                # Extract links for further crawling (only for HTML pages)
                page_links = []
                doc_links = []
                if not is_doc and page.html_content:
                    page_links, doc_links = self.extract_links(current_url, page.html_content)

                # Score relevance
                if self.llm:
                    relevance = self.score_relevance_llm(
                        page.text_content,
                        page.title or "",
                        research_topic,
                        section_context,
                    )
                else:
                    relevance = self.score_relevance_simple(
                        page.text_content,
                        page.title or "",
                        research_topic,
                        keywords,
                    )

                # Create crawled page
                crawled_page = CrawledPage(
                    url=current_url,
                    title=page.title or "",
                    content=page.text_content,
                    links=page_links,
                    relevance_score=relevance,
                    depth=depth,
                    images=[{"url": img.get("src", ""), "alt": img.get("alt", "")}
                            for img in (page.images or [])[:5]],
                )

                # Include if relevant
                if relevance >= self.relevance_threshold:
                    crawled_pages.append(crawled_page)

                # Process discovered document links (PDF/XLSX/DOCX/CSV)
                for doc_url in doc_links[:3]:  # Limit document downloads per page
                    if doc_url not in visited and self._total_pages_crawled < self.GLOBAL_MAX_PAGES:
                        visited.add(doc_url)
                        try:
                            doc_page = self.search.get_page_content(doc_url)
                            self._total_pages_crawled += 1
                            if doc_page.text_content and len(doc_page.text_content) > 50:
                                doc_relevance = self.score_relevance_simple(
                                    doc_page.text_content,
                                    doc_page.title or "",
                                    research_topic,
                                    keywords,
                                )
                                doc_crawled = CrawledPage(
                                    url=doc_url,
                                    title=doc_page.title or "",
                                    content=doc_page.text_content,
                                    links=[],
                                    relevance_score=doc_relevance,
                                    depth=depth + 1,
                                )
                                if doc_relevance >= self.relevance_threshold:
                                    crawled_pages.append(doc_crawled)
                                    print(f"[SiteCrawler] Extracted document: {doc_url[:60]} (relevance: {doc_relevance:.2f})")
                        except Exception as doc_e:
                            print(f"[SiteCrawler] Failed to extract document {doc_url[:60]}: {doc_e}")

                # Add page links to queue (prioritize links with topic keywords in URL)
                for link in page_links:
                    if link not in visited:
                        # Prioritize URLs containing topic keywords
                        priority_boost = any(
                            kw.lower() in link.lower()
                            for kw in keywords + research_topic.split()
                        )
                        if priority_boost:
                            queue.appendleft((link, depth + 1))
                        else:
                            queue.append((link, depth + 1))

                # Delay between requests
                time.sleep(self.delay)

            except Exception as e:
                print(f"Error crawling {current_url}: {e}")
                continue

        # Sort by relevance
        crawled_pages.sort(key=lambda p: p.relevance_score, reverse=True)

        # Discover new topics and queries
        discoveries = self.discover_topics_and_queries(
            crawled_pages,
            research_topic,
            existing_content,
        )

        return CrawlResult(
            seed_url=seed_url,
            root_domain=root_domain,
            pages_crawled=pages_crawled,
            pages_relevant=len(crawled_pages),
            crawled_pages=crawled_pages,
            discovered_topics=discoveries.get("topics", []),
            suggested_queries=discoveries.get("queries", []),
        )

    def crawl_multiple_sites(
        self,
        seed_urls: List[str],
        research_topic: str,
        keywords: List[str] = None,
        section_context: str = "",
        existing_content: str = "",
        max_sites: int = 3,
        progress_callback: Callable[[str, int, int], None] = None,
    ) -> List[CrawlResult]:
        """
        Crawl multiple websites.

        Args:
            seed_urls: List of seed URLs
            research_topic: Main research topic
            keywords: Related keywords
            section_context: Current section context
            existing_content: Already collected content
            max_sites: Maximum number of sites to crawl
            progress_callback: Progress callback

        Returns:
            List of CrawlResults
        """
        results = []
        crawled_domains = set()

        for url in seed_urls:
            if len(results) >= max_sites:
                break

            # Check global limit before starting new site
            if self._total_pages_crawled >= self.GLOBAL_MAX_PAGES:
                print(f"Global page limit ({self.GLOBAL_MAX_PAGES}) reached. Stopping crawl.")
                break

            domain = self.get_domain(url)
            if domain in crawled_domains:
                continue

            crawled_domains.add(domain)

            if progress_callback:
                progress_callback(
                    f"Starting crawl of {domain} (Total pages: {self._total_pages_crawled}/{self.GLOBAL_MAX_PAGES})",
                    len(results) + 1,
                    max_sites
                )

            result = self.crawl_site(
                seed_url=url,
                research_topic=research_topic,
                keywords=keywords,
                section_context=section_context,
                existing_content=existing_content,
            )

            if result.pages_relevant > 0:
                results.append(result)

        return results


def extract_keywords_from_topic(topic: str) -> List[str]:
    """
    Extract keywords from a research topic.

    Args:
        topic: Research topic string

    Returns:
        List of keywords
    """
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "of", "to", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after",
        "above", "below", "between", "under", "again", "further",
        "then", "once", "and", "or", "but", "if", "than", "too",
        "very", "just", "only", "own", "same", "so", "more", "most",
        "other", "some", "such", "no", "nor", "not", "about", "what",
        # Japanese particles and common words
        "の", "に", "は", "を", "が", "と", "で", "も", "や", "から",
        "まで", "より", "など", "について", "における", "として",
    }

    # Split and filter
    words = re.split(r'[\s、。，．・]+', topic)
    keywords = [
        word for word in words
        if len(word) > 1 and word.lower() not in stop_words
    ]

    return keywords
