"""
Tests for site crawler (Extended Mode).
"""

import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any

from deep_research_tool2.research.site_crawler import (
    SiteCrawler,
    CrawledPage,
    CrawlResult,
    extract_keywords_from_topic,
)


@dataclass
class MockPage:
    """Mock page for testing."""
    title: str
    text_content: str
    html_content: str
    images: List[Dict[str, str]]


class TestSiteCrawler:
    """Tests for SiteCrawler class."""

    @pytest.fixture
    def mock_search_client(self):
        """Create mock search client."""
        client = Mock()
        client.get_page_content = Mock(return_value=MockPage(
            title="Test Page",
            text_content="This is test content about AI and machine learning.",
            html_content='<html><body><a href="/page2">Link</a></body></html>',
            images=[{"src": "test.jpg", "alt": "test"}],
        ))
        return client

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = Mock()
        response = Mock()
        response.content = "0.7"
        client.generate = Mock(return_value=response)
        return client

    @pytest.fixture
    def crawler(self, mock_search_client, mock_llm_client):
        """Create SiteCrawler instance."""
        return SiteCrawler(
            search_client=mock_search_client,
            llm_client=mock_llm_client,
            max_pages=5,
            max_depth=2,
            relevance_threshold=0.3,
            delay_between_requests=0,  # No delay in tests
        )

    def test_init(self, crawler):
        """Test crawler initialization."""
        assert crawler.max_pages == 5
        assert crawler.max_depth == 2
        assert crawler.relevance_threshold == 0.3

    def test_global_max_pages_limit(self, mock_search_client, mock_llm_client):
        """Test that global max pages limit is enforced."""
        # Try to create crawler with max_pages > GLOBAL_MAX_PAGES
        crawler = SiteCrawler(
            search_client=mock_search_client,
            llm_client=mock_llm_client,
            max_pages=100,  # Exceeds GLOBAL_MAX_PAGES (50)
        )
        assert crawler.max_pages == 50  # Should be capped at 50

    def test_reset_page_counter(self, crawler):
        """Test page counter reset."""
        crawler._total_pages_crawled = 25
        crawler.reset_page_counter()
        assert crawler._total_pages_crawled == 0

    def test_get_remaining_pages(self, crawler):
        """Test remaining pages calculation."""
        crawler._total_pages_crawled = 30
        remaining = crawler.get_remaining_pages()
        assert remaining == 20  # 50 - 30

    def test_get_root_domain(self, crawler):
        """Test root domain extraction."""
        url = "https://example.com/path/to/page"
        assert crawler.get_root_domain(url) == "https://example.com"

    def test_get_domain(self, crawler):
        """Test domain extraction."""
        url = "https://www.example.com/page"
        assert crawler.get_domain(url) == "www.example.com"

    def test_is_same_domain(self, crawler):
        """Test same domain check."""
        url1 = "https://example.com/page1"
        url2 = "https://example.com/page2"
        url3 = "https://other.com/page"

        assert crawler.is_same_domain(url1, url2) is True
        assert crawler.is_same_domain(url1, url3) is False

    def test_normalize_url(self, crawler):
        """Test URL normalization."""
        url = "https://example.com/page#section/"
        normalized = crawler.normalize_url(url)
        assert normalized == "https://example.com/page"

    def test_is_valid_page_url(self, crawler):
        """Test URL validation."""
        # Valid URLs
        assert crawler.is_valid_page_url("https://example.com/article") is True
        assert crawler.is_valid_page_url("https://example.com/news/tech") is True

        # Invalid URLs (files)
        assert crawler.is_valid_page_url("https://example.com/file.pdf") is False
        assert crawler.is_valid_page_url("https://example.com/image.jpg") is False
        assert crawler.is_valid_page_url("https://example.com/script.js") is False

        # Invalid URLs (non-content paths)
        assert crawler.is_valid_page_url("https://example.com/login") is False
        assert crawler.is_valid_page_url("https://example.com/api/data") is False
        assert crawler.is_valid_page_url("https://example.com/wp-admin/") is False

    def test_extract_links(self, crawler):
        """Test link extraction from HTML."""
        base_url = "https://example.com/page"
        html = '''
        <html>
        <body>
            <a href="/about">About</a>
            <a href="https://example.com/contact">Contact</a>
            <a href="https://other.com/external">External</a>
            <a href="#section">Anchor</a>
            <a href="javascript:void(0)">JS</a>
        </body>
        </html>
        '''

        links = crawler.extract_links(base_url, html)

        # Should include same-domain links
        assert "https://example.com/about" in links
        assert "https://example.com/contact" in links

        # Should exclude external links
        assert not any("other.com" in link for link in links)

        # Should exclude anchors and javascript
        assert "#section" not in links
        assert "javascript:" not in str(links)

    def test_score_relevance_simple(self, crawler):
        """Test simple relevance scoring."""
        content = "This article discusses artificial intelligence and machine learning applications."
        title = "AI Technology Overview"
        topic = "artificial intelligence"
        keywords = ["AI", "machine learning", "technology"]

        score = crawler.score_relevance_simple(content, title, topic, keywords)

        # Score should be between 0 and 1
        assert 0 <= score <= 1
        # Should have high relevance given topic match
        assert score > 0.3

    def test_score_relevance_simple_no_match(self, crawler):
        """Test simple relevance scoring with no match."""
        content = "This is about cooking recipes and food preparation."
        title = "Cooking Tips"
        topic = "quantum physics"
        keywords = ["physics", "quantum", "particles"]

        score = crawler.score_relevance_simple(content, title, topic, keywords)

        # Score should be low
        assert score < 0.3

    def test_crawl_site_basic(self, crawler, mock_search_client):
        """Test basic site crawl."""
        result = crawler.crawl_site(
            seed_url="https://example.com",
            research_topic="AI technology",
            keywords=["AI", "technology"],
        )

        assert isinstance(result, CrawlResult)
        assert result.seed_url == "https://example.com"
        assert result.pages_crawled >= 1

    def test_crawl_site_respects_max_pages(self, mock_search_client, mock_llm_client):
        """Test that crawl respects max_pages limit."""
        # Create mock that returns HTML with many links
        html_with_links = '<html><body>' + ''.join(
            f'<a href="/page{i}">Page {i}</a>' for i in range(100)
        ) + '</body></html>'

        mock_search_client.get_page_content = Mock(return_value=MockPage(
            title="Test",
            text_content="Test content about AI",
            html_content=html_with_links,
            images=[],
        ))

        crawler = SiteCrawler(
            search_client=mock_search_client,
            llm_client=mock_llm_client,
            max_pages=3,
            max_depth=5,
            delay_between_requests=0,
        )

        result = crawler.crawl_site(
            seed_url="https://example.com",
            research_topic="AI",
        )

        # Should not exceed max_pages
        assert result.pages_crawled <= 3

    def test_crawl_site_respects_max_depth(self, mock_search_client, mock_llm_client):
        """Test that crawl respects max_depth limit."""
        # Create mock that returns HTML with deep links
        def create_page(url):
            depth = url.count('/') - 2  # Estimate depth from URL structure
            if depth < 3:
                html = f'<a href="{url}/subpage">Deeper</a>'
            else:
                html = '<p>No more links</p>'
            return MockPage(
                title=f"Page at depth {depth}",
                text_content="Test content",
                html_content=html,
                images=[],
            )

        mock_search_client.get_page_content = Mock(side_effect=create_page)

        crawler = SiteCrawler(
            search_client=mock_search_client,
            llm_client=mock_llm_client,
            max_pages=100,
            max_depth=1,  # Only depth 0 and 1
            delay_between_requests=0,
        )

        result = crawler.crawl_site(
            seed_url="https://example.com",
            research_topic="AI",
        )

        # Should stop at max_depth
        for page in result.crawled_pages:
            assert page.depth <= 1

    def test_crawl_site_handles_errors(self, mock_search_client, mock_llm_client):
        """Test that crawl handles page fetch errors gracefully."""
        call_count = [0]

        def sometimes_fail(url):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise Exception("Network error")
            return MockPage(
                title="Test",
                text_content="Test content",
                html_content='<a href="/next">Next</a>',
                images=[],
            )

        mock_search_client.get_page_content = Mock(side_effect=sometimes_fail)

        crawler = SiteCrawler(
            search_client=mock_search_client,
            llm_client=mock_llm_client,
            max_pages=5,
            delay_between_requests=0,
        )

        # Should not raise exception
        result = crawler.crawl_site(
            seed_url="https://example.com",
            research_topic="test",
        )

        assert isinstance(result, CrawlResult)

    def test_crawl_multiple_sites(self, crawler):
        """Test crawling multiple sites."""
        seed_urls = [
            "https://site1.com",
            "https://site2.com",
            "https://site3.com",
        ]

        results = crawler.crawl_multiple_sites(
            seed_urls=seed_urls,
            research_topic="AI",
            max_sites=2,
        )

        # Should respect max_sites
        assert len(results) <= 2

    def test_crawl_multiple_sites_skips_same_domain(self, crawler):
        """Test that duplicate domains are skipped."""
        seed_urls = [
            "https://example.com/page1",
            "https://example.com/page2",  # Same domain
            "https://other.com",
        ]

        results = crawler.crawl_multiple_sites(
            seed_urls=seed_urls,
            research_topic="AI",
            max_sites=3,
        )

        # Should have crawled at most 2 unique domains
        domains = set()
        for result in results:
            domains.add(crawler.get_domain(result.seed_url))
        assert len(domains) <= 2

    def test_global_limit_across_sites(self, mock_search_client, mock_llm_client):
        """Test that global page limit is enforced across multiple sites."""
        # Create crawler with low global limit scenario
        crawler = SiteCrawler(
            search_client=mock_search_client,
            llm_client=mock_llm_client,
            max_pages=10,
            delay_between_requests=0,
        )

        # Manually set high page count
        crawler._total_pages_crawled = 45  # Close to global limit of 50

        seed_urls = [
            "https://site1.com",
            "https://site2.com",
        ]

        results = crawler.crawl_multiple_sites(
            seed_urls=seed_urls,
            research_topic="test",
            max_sites=10,
        )

        # Should stop due to global limit
        assert crawler._total_pages_crawled <= 50


class TestCrawledPage:
    """Tests for CrawledPage dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        page = CrawledPage(
            url="https://example.com",
            title="Test Page",
            content="Test content " * 200,  # Long content
            links=["https://example.com/page1", "https://example.com/page2"],
            relevance_score=0.8,
            depth=1,
            images=[{"url": "test.jpg", "alt": "test"}],
        )

        result = page.to_dict()

        assert result["url"] == "https://example.com"
        assert result["title"] == "Test Page"
        # Content should be truncated
        assert len(result["content"]) <= 503  # 500 + "..."
        assert result["links_count"] == 2
        assert result["relevance_score"] == 0.8
        assert result["depth"] == 1
        assert result["images_count"] == 1


class TestCrawlResult:
    """Tests for CrawlResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = CrawlResult(
            seed_url="https://example.com",
            root_domain="https://example.com",
            pages_crawled=5,
            pages_relevant=3,
            crawled_pages=[
                CrawledPage(
                    url="https://example.com/page1",
                    title="Page 1",
                    content="Content 1",
                    relevance_score=0.8,
                    depth=0,
                ),
            ],
            discovered_topics=["topic1", "topic2"],
            suggested_queries=["query1"],
        )

        data = result.to_dict()

        assert data["seed_url"] == "https://example.com"
        assert data["pages_crawled"] == 5
        assert data["pages_relevant"] == 3
        assert len(data["crawled_pages"]) == 1
        assert data["discovered_topics"] == ["topic1", "topic2"]


class TestExtractKeywords:
    """Tests for keyword extraction function."""

    def test_basic_extraction(self):
        """Test basic keyword extraction."""
        topic = "artificial intelligence in healthcare"
        keywords = extract_keywords_from_topic(topic)

        assert "artificial" in keywords
        assert "intelligence" in keywords
        assert "healthcare" in keywords
        # Stop words should be removed
        assert "in" not in keywords

    def test_japanese_extraction(self):
        """Test Japanese keyword extraction."""
        topic = "人工知能における医療への応用"
        keywords = extract_keywords_from_topic(topic)

        # Should extract meaningful keywords
        assert len(keywords) > 0
        # Stop words should be removed
        assert "における" not in keywords
        assert "への" not in keywords

    def test_empty_input(self):
        """Test with empty input."""
        keywords = extract_keywords_from_topic("")
        assert keywords == []

    def test_only_stop_words(self):
        """Test with only stop words."""
        topic = "the a an is are"
        keywords = extract_keywords_from_topic(topic)
        assert len(keywords) == 0


class TestProgressCallback:
    """Tests for progress callback functionality."""

    @pytest.fixture
    def mock_search_client(self):
        """Create mock search client."""
        client = Mock()
        client.get_page_content = Mock(return_value=MockPage(
            title="Test Page",
            text_content="This is test content about AI and machine learning.",
            html_content='<html><body><a href="/page2">Link</a></body></html>',
            images=[{"src": "test.jpg", "alt": "test"}],
        ))
        return client

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = Mock()
        response = Mock()
        response.content = "0.7"
        client.generate = Mock(return_value=response)
        return client

    def test_crawl_with_progress_callback(self, mock_search_client, mock_llm_client):
        """Test that progress callback is called during crawl."""
        crawler = SiteCrawler(
            search_client=mock_search_client,
            llm_client=mock_llm_client,
            max_pages=3,
            delay_between_requests=0,
        )

        progress_calls = []

        def callback(message, current, total):
            progress_calls.append((message, current, total))

        crawler.crawl_site(
            seed_url="https://example.com",
            research_topic="test",
            progress_callback=callback,
        )

        # Should have called progress callback
        assert len(progress_calls) > 0

    def test_multiple_sites_with_progress(self, mock_search_client, mock_llm_client):
        """Test progress callback for multiple sites."""
        crawler = SiteCrawler(
            search_client=mock_search_client,
            llm_client=mock_llm_client,
            max_pages=2,
            delay_between_requests=0,
        )

        progress_calls = []

        def callback(message, current, total):
            progress_calls.append((message, current, total))

        crawler.crawl_multiple_sites(
            seed_urls=["https://site1.com", "https://site2.com"],
            research_topic="test",
            max_sites=2,
            progress_callback=callback,
        )

        # Should have site-level progress calls
        assert any("site1.com" in call[0] or "site2.com" in call[0]
                   for call in progress_calls)
