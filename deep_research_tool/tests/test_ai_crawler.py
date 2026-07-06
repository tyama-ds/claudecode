"""
Tests for AICrawler (aicrawl mode - LLM-driven crawling).
"""

import json

import pytest
from unittest.mock import Mock
from dataclasses import dataclass, field
from typing import List, Dict

from deep_research_tool.research.ai_crawler import (
    AICrawler,
    AICrawlDecision,
    LinkCandidate,
)
from deep_research_tool.research.fast_crawler import CrawlResult, EvaluatedPage


@dataclass
class MockPage:
    """Mock page for testing."""
    title: str
    text_content: str
    html_content: str = ""
    links: List[Dict[str, str]] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)


def make_search_result(url, title="Result", snippet="snippet"):
    result = Mock()
    result.url = url
    result.title = title
    result.snippet = snippet
    return result


def decision_json(relevance=0.8, processed="要約テキスト", key_points=None,
                  follow_links=None, suggested_queries=None):
    return json.dumps({
        "relevance_score": relevance,
        "processed_content": processed,
        "key_points": key_points or ["ポイント1"],
        "follow_links": follow_links or [],
        "suggested_queries": suggested_queries or [],
    }, ensure_ascii=False)


def make_llm(responses):
    """Mock LLM client returning the given contents in order (last repeats)."""
    client = Mock()
    mocks = [Mock(content=r) for r in responses]

    def generate(prompt, **kwargs):
        if len(mocks) > 1:
            return mocks.pop(0)
        return mocks[0]

    client.generate = Mock(side_effect=generate)
    return client


def make_passthrough_filter():
    """Content filter mock that lets everything through."""
    f = Mock()
    f.filter_content = Mock(
        return_value=Mock(should_include=True, reason="", quality_score=1.0)
    )
    return f


def make_crawler(search_client, llm_client, **kwargs):
    defaults = dict(
        max_total_pages=10,
        max_depth=3,
        max_llm_calls=25,
        max_pages_per_domain=5,
        politeness_delay=0,
        content_filter=make_passthrough_filter(),
    )
    defaults.update(kwargs)
    return AICrawler(search_client=search_client, llm_client=llm_client, **defaults)


class TestAICrawler:
    """Tests for AICrawler."""

    def test_seeds_from_search_and_dedups(self):
        """Frontier is seeded from search results, duplicate URLs fetched once."""
        search = Mock()
        search.search = Mock(return_value=[
            make_search_result("https://example.com/a"),
            make_search_result("https://example.com/a/"),  # dup after normalization
            make_search_result("https://example.com/b"),
        ])
        search.get_page_content = Mock(return_value=MockPage(
            title="Page", text_content="AI research content",
        ))
        llm = make_llm([decision_json()])
        crawler = make_crawler(search, llm)

        result = crawler.crawl_and_evaluate(
            queries=["AI research"],
            section_context="1. Overview",
            research_topic="AI research",
        )

        assert isinstance(result, CrawlResult)
        assert search.get_page_content.call_count == 2  # a and b, dup skipped
        assert result.pages_fetched == 2
        assert all(isinstance(p, EvaluatedPage) for p in result.pages)
        # Fields the Researcher reads downstream
        page = result.pages[0]
        assert page.url.startswith("https://example.com/")
        assert page.processed_content == "要約テキスト"
        assert page.key_points == ["ポイント1"]
        assert page.relevance_score == 0.8
        assert page.metadata["decided_by"] == "llm"
        assert page.metadata["query"] == "AI research"

    def test_follows_only_llm_selected_links(self):
        """Only links the LLM selects are followed."""
        search = Mock()
        search.search = Mock(return_value=[make_search_result("https://example.com/seed")])
        pages = {
            "https://example.com/seed": MockPage(
                title="Seed", text_content="seed content",
                links=[
                    {"url": "https://example.com/good", "text": "Good link"},
                    {"url": "https://example.com/bad", "text": "Bad link"},
                ],
            ),
            "https://example.com/good": MockPage(
                title="Good", text_content="good content", links=[],
            ),
        }
        search.get_page_content = Mock(side_effect=lambda url: pages[url])
        llm = make_llm([
            decision_json(follow_links=[{"index": 1, "priority": 0.9, "reason": "relevant"}]),
            decision_json(),
        ])
        crawler = make_crawler(search, llm)

        result = crawler.crawl_and_evaluate(
            queries=["q"], section_context="ctx", research_topic="topic",
        )

        fetched = [c.args[0] for c in search.get_page_content.call_args_list]
        assert fetched == ["https://example.com/seed", "https://example.com/good"]
        assert "https://example.com/bad" not in fetched
        # The followed page carries its depth in metadata
        good = [p for p in result.pages if p.url.endswith("/good")][0]
        assert good.metadata["depth"] == 1

    def test_max_total_pages_respected(self):
        """Crawl stops at max_total_pages even if LLM keeps proposing links."""
        search = Mock()
        search.search = Mock(return_value=[make_search_result("https://example.com/0")])
        counter = {"n": 0}

        def get_page(url):
            counter["n"] += 1
            return MockPage(
                title=f"Page {counter['n']}", text_content="content",
                links=[{"url": f"https://site{counter['n']}.com/next", "text": "next"}],
            )

        search.get_page_content = Mock(side_effect=get_page)
        llm = make_llm([
            decision_json(follow_links=[{"index": 1, "priority": 0.9, "reason": "r"}]),
        ])
        crawler = make_crawler(search, llm, max_total_pages=3)

        result = crawler.crawl_and_evaluate(
            queries=["q"], section_context="ctx", research_topic="topic",
        )

        assert search.get_page_content.call_count == 3
        assert result.pages_fetched == 3

    def test_max_depth_respected(self):
        """Links beyond max_depth are not fetched."""
        search = Mock()
        search.search = Mock(return_value=[make_search_result("https://a.com/d0")])
        pages = {
            "https://a.com/d0": MockPage(
                title="d0", text_content="c",
                links=[{"url": "https://b.com/d1", "text": "l"}],
            ),
            "https://b.com/d1": MockPage(
                title="d1", text_content="c",
                links=[{"url": "https://c.com/d2", "text": "l"}],
            ),
            "https://c.com/d2": MockPage(
                title="d2", text_content="c",
                links=[{"url": "https://d.com/d3", "text": "l"}],
            ),
        }
        search.get_page_content = Mock(side_effect=lambda url: pages[url])
        llm = make_llm([
            decision_json(follow_links=[{"index": 1, "priority": 0.9, "reason": "r"}]),
        ])
        crawler = make_crawler(search, llm, max_depth=1)

        crawler.crawl_and_evaluate(
            queries=["q"], section_context="ctx", research_topic="topic",
        )

        fetched = [c.args[0] for c in search.get_page_content.call_args_list]
        assert "https://a.com/d0" in fetched
        assert "https://b.com/d1" in fetched
        assert "https://c.com/d2" not in fetched  # depth 2 > max_depth 1

    def test_per_domain_page_cap(self):
        """No more than max_pages_per_domain pages fetched from one domain."""
        search = Mock()
        search.search = Mock(return_value=[make_search_result("https://one.com/p0")])
        counter = {"n": 0}

        def get_page(url):
            counter["n"] += 1
            return MockPage(
                title=url, text_content="c",
                links=[{"url": f"https://one.com/p{counter['n']}", "text": "l"}],
            )

        search.get_page_content = Mock(side_effect=get_page)
        llm = make_llm([
            decision_json(follow_links=[{"index": 1, "priority": 0.9, "reason": "r"}]),
        ])
        crawler = make_crawler(search, llm, max_pages_per_domain=2, max_total_pages=10)

        result = crawler.crawl_and_evaluate(
            queries=["q"], section_context="ctx", research_topic="topic",
        )

        assert result.pages_fetched == 2

    def test_malformed_llm_response_falls_back_to_keywords(self):
        """Malformed LLM output triggers keyword fallback scoring."""
        search = Mock()
        search.search = Mock(return_value=[make_search_result("https://example.com/a")])
        search.get_page_content = Mock(return_value=MockPage(
            title="quantum computing overview",
            text_content="quantum computing is advancing rapidly",
            links=[{"url": "https://example.com/quantum-details", "text": "quantum details"}],
        ))
        llm = make_llm(["this is not valid json at all"])
        crawler = make_crawler(search, llm, max_total_pages=2)

        result = crawler.crawl_and_evaluate(
            queries=["quantum computing"],
            section_context="ctx",
            research_topic="quantum computing",
            min_relevance_score=0.1,
        )

        assert result.pages_fetched >= 1
        assert any(p.metadata["decided_by"] == "fallback" for p in result.pages)
        # Keyword-matching link was enqueued by the fallback and fetched
        fetched = [c.args[0] for c in search.get_page_content.call_args_list]
        assert "https://example.com/quantum-details" in fetched

    def test_dead_end_reseed_from_suggested_queries(self):
        """When frontier drains with budget left, suggested queries re-seed."""
        search = Mock()
        search.search = Mock(side_effect=[
            [make_search_result("https://example.com/a")],
            [make_search_result("https://other.com/b")],  # re-seed round
        ])
        search.get_page_content = Mock(return_value=MockPage(
            title="Page", text_content="content", links=[],
        ))
        llm = make_llm([
            decision_json(follow_links=[], suggested_queries=["deeper query"]),
            decision_json(),
        ])
        crawler = make_crawler(search, llm, max_total_pages=5)

        crawler.crawl_and_evaluate(
            queries=["q"], section_context="ctx", research_topic="topic",
        )

        assert search.search.call_count == 2
        second_query = search.search.call_args_list[1].args[0]
        assert second_query == "deeper query"
        assert search.get_page_content.call_count == 2

    def test_max_llm_calls_exhaustion_uses_fallback(self):
        """After max_llm_calls, remaining pages are scored by fallback."""
        search = Mock()
        search.search = Mock(return_value=[
            make_search_result("https://a.com/1", title="topic page"),
            make_search_result("https://b.com/2", title="topic page"),
        ])
        search.get_page_content = Mock(return_value=MockPage(
            title="topic page", text_content="topic content here", links=[],
        ))
        llm = make_llm([decision_json()])
        crawler = make_crawler(search, llm, max_llm_calls=1)

        result = crawler.crawl_and_evaluate(
            queries=["topic"], section_context="ctx", research_topic="topic",
            min_relevance_score=0.0,
        )

        assert llm.generate.call_count == 1
        assert result.pages_fetched == 2
        decided_by = sorted(p.metadata["decided_by"] for p in result.pages)
        assert decided_by == ["fallback", "llm"]

    def test_fetch_error_recorded_and_crawl_continues(self):
        """A fetch error is recorded in errors and does not abort the crawl."""
        search = Mock()
        search.search = Mock(return_value=[
            make_search_result("https://bad.com/x"),
            make_search_result("https://good.com/y"),
        ])

        def get_page(url):
            if "bad.com" in url:
                raise ConnectionError("boom")
            return MockPage(title="Good", text_content="content", links=[])

        search.get_page_content = Mock(side_effect=get_page)
        llm = make_llm([decision_json()])
        crawler = make_crawler(search, llm)

        result = crawler.crawl_and_evaluate(
            queries=["q"], section_context="ctx", research_topic="topic",
        )

        assert result.pages_fetched == 1
        assert len(result.errors) == 1
        assert "bad.com" in result.errors[0]

    def test_decision_prompt_language(self):
        """Japanese prompts by default; English when language='en'."""
        search = Mock()
        llm = make_llm([decision_json()])
        crawler_ja = make_crawler(search, llm)
        prompt_ja = crawler_ja._build_decision_prompt(
            "https://x.com", "t", "c", [LinkCandidate("https://x.com/1", "a")],
            "topic", ["kw"], "ctx", 5,
        )
        assert "調査テーマ" in prompt_ja

        crawler_en = make_crawler(search, llm, language="en")
        prompt_en = crawler_en._build_decision_prompt(
            "https://x.com", "t", "c", [LinkCandidate("https://x.com/1", "a")],
            "topic", ["kw"], "ctx", 5,
        )
        assert "Research topic" in prompt_en


class TestSiteDepth:
    """Tests for the per-site layer limit (max_site_depth)."""

    def _chain_pages(self, domain_urls):
        """Build pages where each links to the next URL in the list."""
        pages = {}
        for i, url in enumerate(domain_urls):
            next_links = (
                [{"url": domain_urls[i + 1], "text": "next"}]
                if i + 1 < len(domain_urls) else []
            )
            pages[url] = MockPage(title=url, text_content="content", links=next_links)
        return pages

    def test_same_site_chain_stops_at_max_site_depth(self):
        """Following layers within one site stops at max_site_depth."""
        chain = [
            "https://one.com/l0",
            "https://one.com/l1",
            "https://one.com/l2",
            "https://one.com/l3",
        ]
        search = Mock()
        search.search = Mock(return_value=[make_search_result(chain[0])])
        pages = self._chain_pages(chain)
        search.get_page_content = Mock(side_effect=lambda url: pages[url])
        llm = make_llm([
            decision_json(follow_links=[{"index": 1, "priority": 0.9, "reason": "r"}]),
        ])
        crawler = make_crawler(
            search, llm, max_site_depth=1, max_depth=10, max_pages_per_domain=10,
        )

        crawler.crawl_and_evaluate(
            queries=["q"], section_context="ctx", research_topic="topic",
        )

        fetched = [c.args[0] for c in search.get_page_content.call_args_list]
        assert chain[0] in fetched  # site layer 0
        assert chain[1] in fetched  # site layer 1
        assert chain[2] not in fetched  # site layer 2 > max_site_depth 1

    def test_site_depth_resets_across_domains(self):
        """Crossing to another domain resets the site-layer counter."""
        search = Mock()
        search.search = Mock(return_value=[make_search_result("https://one.com/a")])
        pages = {
            "https://one.com/a": MockPage(
                title="a", text_content="c",
                links=[{"url": "https://one.com/b", "text": "same"}],
            ),
            "https://one.com/b": MockPage(
                title="b", text_content="c",
                links=[{"url": "https://two.com/x", "text": "cross"}],
            ),
            # two.com/x is reached at site layer 1 (reset), so its same-site
            # child is layer 2 and must be skipped with max_site_depth=1
            "https://two.com/x": MockPage(
                title="x", text_content="c",
                links=[{"url": "https://two.com/y", "text": "same"}],
            ),
        }
        search.get_page_content = Mock(side_effect=lambda url: pages[url])
        llm = make_llm([
            decision_json(follow_links=[{"index": 1, "priority": 0.9, "reason": "r"}]),
        ])
        crawler = make_crawler(
            search, llm, max_site_depth=1, max_depth=10, max_pages_per_domain=10,
        )

        crawler.crawl_and_evaluate(
            queries=["q"], section_context="ctx", research_topic="topic",
        )

        fetched = [c.args[0] for c in search.get_page_content.call_args_list]
        assert "https://two.com/x" in fetched  # cross-domain: counter reset
        assert "https://two.com/y" not in fetched  # layer 2 within two.com


class TestAICrawlerSelenium:
    """Tests for the Selenium-fetching variant."""

    def test_fetches_pages_via_selenium_client(self):
        from deep_research_tool.research.ai_crawler_selenium import AICrawlerSelenium

        search = Mock()
        search.search = Mock(return_value=[make_search_result("https://example.com/a")])
        # search client's fetch must NOT be used
        search.get_page_content = Mock(side_effect=AssertionError("should not be called"))

        selenium_client = Mock()
        selenium_client.get_page_content = Mock(return_value=MockPage(
            title="JS Page", text_content="rendered content", links=[],
        ))

        llm = make_llm([decision_json()])
        crawler = AICrawlerSelenium(
            search_client=search,
            llm_client=llm,
            selenium_client=selenium_client,
            politeness_delay=0,
            content_filter=make_passthrough_filter(),
        )

        result = crawler.crawl_and_evaluate(
            queries=["q"], section_context="ctx", research_topic="topic",
        )

        selenium_client.get_page_content.assert_called_once_with("https://example.com/a")
        search.get_page_content.assert_not_called()
        assert result.pages_fetched == 1
        assert result.pages[0].title == "JS Page"

    def test_close_releases_driver(self):
        from deep_research_tool.research.ai_crawler_selenium import AICrawlerSelenium

        selenium_client = Mock()
        crawler = AICrawlerSelenium(
            search_client=Mock(),
            llm_client=Mock(),
            selenium_client=selenium_client,
        )
        crawler.close()
        selenium_client.close.assert_called_once()

    def test_accepts_site_depth_parameter(self):
        from deep_research_tool.research.ai_crawler_selenium import AICrawlerSelenium

        crawler = AICrawlerSelenium(
            search_client=Mock(),
            llm_client=Mock(),
            selenium_client=Mock(),
            max_site_depth=4,
        )
        assert crawler.max_site_depth == 4


class TestFallbackDecision:
    """Tests for the keyword fallback decision."""

    def test_fallback_scores_and_selects_keyword_links(self):
        crawler = make_crawler(Mock(), Mock())
        candidates = [
            LinkCandidate("https://x.com/quantum", "quantum info"),
            LinkCandidate("https://x.com/unrelated", "cookies policy"),
        ]
        decision = crawler._fallback_decision(
            title="quantum computing",
            content="quantum computing content",
            candidates=candidates,
            research_topic="quantum computing",
            keywords=["quantum", "computing"],
        )
        assert decision.used_fallback is True
        assert decision.relevance_score > 0
        followed_urls = [link.url for link, _ in decision.follow_links]
        assert "https://x.com/quantum" in followed_urls
        assert "https://x.com/unrelated" not in followed_urls
