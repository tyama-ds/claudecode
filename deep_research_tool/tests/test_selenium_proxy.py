"""
Tests for Selenium browser proxy support and Edge browser selection.
"""

import pytest
from unittest.mock import Mock, patch

from deep_research_tool.config import CrawlMode, create_config
from deep_research_tool.research.ai_crawler_selenium import AICrawlerSelenium
from deep_research_tool.research.researcher import Researcher
from deep_research_tool.search.selenium_browser import SeleniumBrowser

PROXIES = {"https": "http://proxy.example.com:8080", "http": "http://proxy.example.com:8080"}


class TestSeleniumBrowserProxy:
    def test_accepts_proxies_kwarg(self):
        """Regression: SeleniumBrowser used to raise TypeError on proxies."""
        client = SeleniumBrowser(proxies=PROXIES, verify_ssl=False)
        assert client.proxies == PROXIES
        assert client.verify_ssl is False

    def test_proxy_url_selection(self):
        client = SeleniumBrowser(proxies=PROXIES)
        assert client._proxy_url() == "http://proxy.example.com:8080"

    def test_proxy_url_strips_credentials(self):
        client = SeleniumBrowser(
            proxies={"https": "http://user:pass@proxy.example.com:8080"},
        )
        assert client._proxy_url() == "http://proxy.example.com:8080"

    def test_no_proxy(self):
        client = SeleniumBrowser()
        assert client._proxy_url() == ""
        assert client._chromium_proxy_args() == []

    def test_chromium_args_with_proxy(self):
        client = SeleniumBrowser(proxies=PROXIES, verify_ssl=False)
        args = client._chromium_proxy_args()
        assert "--proxy-server=http://proxy.example.com:8080" in args
        assert "--ignore-certificate-errors" in args

    def test_chromium_args_verify_ssl_on(self):
        client = SeleniumBrowser(proxies=PROXIES, verify_ssl=True)
        args = client._chromium_proxy_args()
        assert args == ["--proxy-server=http://proxy.example.com:8080"]


class TestEdgeBrowser:
    def test_edge_accepted(self):
        client = SeleniumBrowser(browser="edge")
        assert client.browser == "edge"

    def test_create_driver_routes_to_edge(self):
        client = SeleniumBrowser(browser="edge")
        with patch.object(client, "_create_edge_driver", return_value="edge-driver") as m:
            assert client._create_driver() == "edge-driver"
            m.assert_called_once()

    def test_unsupported_browser_raises(self):
        client = SeleniumBrowser(browser="safari")
        with pytest.raises(ValueError, match="Unsupported browser"):
            client._create_driver()

    def test_config_browser_edge(self):
        config = create_config(browser="edge")
        assert config.search.browser == "edge"

    def test_config_browser_default_chrome(self):
        assert create_config().search.browser == "chrome"


class TestAICrawlerSeleniumProxy:
    def test_proxy_and_browser_passed_to_selenium_client(self):
        crawler = AICrawlerSelenium(
            search_client=Mock(),
            llm_client=Mock(),
            browser="edge",
            proxies=PROXIES,
            verify_ssl=False,
        )
        with patch(
            "deep_research_tool.search.selenium_browser.SeleniumBrowser"
        ) as MockBrowser:
            crawler._get_selenium_client()
            kwargs = MockBrowser.call_args.kwargs
            assert kwargs["browser"] == "edge"
            assert kwargs["proxies"] == PROXIES
            assert kwargs["verify_ssl"] is False

    def test_researcher_wires_selenium_settings(self, tmp_path):
        r = Researcher(
            llm_client=Mock(),
            search_client=Mock(),
            output_dir=tmp_path,
            crawl_mode=CrawlMode.AI_CRAWL_SELENIUM,
            selenium_browser="edge",
            selenium_proxies=PROXIES,
            selenium_verify_ssl=False,
        )
        assert isinstance(r.ai_crawler, AICrawlerSelenium)
        assert r.ai_crawler._browser == "edge"
        assert r.ai_crawler._proxies == PROXIES
        assert r.ai_crawler._verify_ssl is False
