"""
Tests for locale (region-first) search: searching AS the target region
to collect genuinely local information.

Covers: config wiring/validation, per-region locale (kl) search, query
LOCALIZATION (not mere translation), local-source (country TLD) boost,
region tagging of results, the Selenium gl/hl URL builders, and the
regression fix for the translation .strip() bug.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from deep_research_tool.config import (
    REGION_LOCALE_MAP,
    MultilingualSearchConfig,
    create_config,
)
from deep_research_tool.search.multilingual import (
    MultilingualSearcher,
    TranslatedQuery,
)
from deep_research_tool.search.selenium_browser import (
    ddg_search_url,
    google_search_url,
)


class ObjLLM:
    """LLM whose generate() returns an OBJECT with .content (the real
    client shape — the old code crashed on .strip() and silently fell
    back to the untranslated query)."""

    def __init__(self, text="localized query"):
        self.text = text
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.text)


class RecordingSearch:
    """Search client recording the region (kl) each call used."""

    def __init__(self, results_by_region=None):
        self.calls = []      # (query, region)
        self.results_by_region = results_by_region or {}

    def search(self, query, max_results=10, region="wt-wt", **kwargs):
        self.calls.append((query, region))
        urls = self.results_by_region.get(region, [])
        return [SimpleNamespace(url=u, title=f"t:{u}", snippet="s")
                for u in urls]


class TestRegionConfig(unittest.TestCase):

    def test_search_regions_enable_multilingual_machinery(self):
        config = create_config(provider="openai", openai_api_key="sk-test",
                               search_regions=["DE", "th"])
        self.assertTrue(config.multilingual.enabled)     # even w/o toggle
        self.assertEqual(config.multilingual.search_regions, ["de", "th"])
        self.assertEqual(config.validate(), [])

    def test_unknown_region_rejected(self):
        config = create_config(provider="openai", openai_api_key="sk-test",
                               search_regions=["de", "atlantis"])
        errors = config.validate()
        self.assertTrue(any("atlantis" in e for e in errors))

    def test_region_map_entries_are_complete(self):
        for code, info in REGION_LOCALE_MAP.items():
            for key in ("kl", "language", "lang_name", "name", "tld"):
                self.assertIn(key, info, f"{code} missing {key}")
            self.assertTrue(info["tld"].startswith("."))


class TestRegionSearch(unittest.TestCase):

    def _searcher(self, regions, search_client, llm=None, **cfg):
        config = MultilingualSearchConfig(
            enabled=True, search_regions=regions,
            results_per_language=5, **cfg)
        return MultilingualSearcher(config=config,
                                    search_client=search_client,
                                    llm_client=llm)

    def test_each_region_searches_with_its_locale(self):
        client = RecordingSearch(results_by_region={
            "de-de": ["https://example.de/a"],
            "th-th": ["https://example.co.th/b"],
        })
        searcher = self._searcher(["de", "th"], client, llm=ObjLLM("q"))
        results, stats = searcher.search_parallel("EV補助金")

        regions_used = {r for _, r in client.calls}
        self.assertEqual(regions_used, {"de-de", "th-th"})   # DDG kl codes
        self.assertEqual(stats.results_by_region,
                         {"de": 1, "th": 1})
        self.assertEqual({r.region for r in results}, {"de", "th"})

    def test_query_is_localized_not_just_translated(self):
        llm = ObjLLM("Umweltbonus BAFA Förderung Elektroauto")
        client = RecordingSearch(results_by_region={
            "de-de": ["https://bafa.de/x"]})
        searcher = self._searcher(["de"], client, llm=llm)
        searcher.search_parallel("EV補助金")

        # the localization prompt asks for LOCAL institution/program
        # names in the region, not a literal translation
        prompt = llm.prompts[0]
        self.assertIn("Germany", prompt)
        self.assertIn("A LOCAL", prompt)
        self.assertIn("institutions", prompt)
        # and the localized query is what actually got searched
        self.assertEqual(client.calls[0][0],
                         "Umweltbonus BAFA Förderung Elektroauto")

    def test_local_tld_sources_are_boosted(self):
        client = RecordingSearch(results_by_region={
            "de-de": ["https://global.example.com/1",
                      "https://ministerium.example.de/2"],
        })
        searcher = self._searcher(["de"], client, llm=ObjLLM("q"),
                                  prefer_local_sources=True,
                                  local_source_boost=0.3)
        results, _ = searcher.search_parallel("query")
        by_url = {r.url: r.relevance_score for r in results}
        self.assertGreater(by_url["https://ministerium.example.de/2"],
                           by_url["https://global.example.com/1"])
        # boosted local source ranks first
        self.assertTrue(results[0].url.endswith("/2"))

    def test_localization_failure_falls_back_to_original_query(self):
        failing = MagicMock()
        failing.generate.side_effect = RuntimeError("llm down")
        client = RecordingSearch(results_by_region={"th-th": []})
        searcher = self._searcher(["th"], client, llm=failing)
        searcher.search_parallel("元のクエリ")
        self.assertEqual(client.calls[0][0], "元のクエリ")
        self.assertEqual(searcher.stats.translation_errors, 1)

    def test_language_mode_still_works_without_regions(self):
        client = RecordingSearch(results_by_region={
            "jp-jp": ["https://a.jp/1"], "us-en": ["https://b.com/2"]})
        config = MultilingualSearchConfig(
            enabled=True, search_languages=["ja", "en"],
            results_per_language=5)
        searcher = MultilingualSearcher(config=config,
                                        search_client=client,
                                        llm_client=ObjLLM("translated"))
        results, stats = searcher.search_parallel("query")
        self.assertEqual({r for _, r in client.calls}, {"jp-jp", "us-en"})
        self.assertEqual(stats.results_by_region, {})    # language mode
        self.assertEqual({r.region for r in results}, {""})

    def test_translation_handles_llm_response_objects(self):
        """Regression: .generate() returns an object with .content; the
        old .strip() call raised and every query fell back untranslated."""
        searcher = MultilingualSearcher(
            config=MultilingualSearchConfig(enabled=True,
                                            search_languages=["en"]),
            search_client=RecordingSearch(),
            llm_client=ObjLLM("carbon fiber market size"))
        tq = searcher.translate_query("炭素繊維の市場規模", "en")
        self.assertEqual(tq.translated_query, "carbon fiber market size")
        self.assertGreaterEqual(tq.confidence, 0.9)
        self.assertEqual(searcher.stats.translation_errors, 0)


class TestSeleniumLocaleUrls(unittest.TestCase):

    def test_ddg_url_carries_kl(self):
        url = ddg_search_url("EV 補助金", "de-de")
        self.assertIn("kl=de-de", url)
        self.assertIn("duckduckgo.com", url)
        self.assertNotIn(" ", url)

    def test_google_url_gl_hl_mapping(self):
        self.assertIn("gl=de", google_search_url("q", "de-de"))
        self.assertIn("hl=de", google_search_url("q", "de-de"))
        # jp-jp: DDG language code jp maps to Google hl=ja
        url = google_search_url("q", "jp-jp")
        self.assertIn("gl=jp", url)
        self.assertIn("hl=ja", url)
        # tw-tzh -> traditional Chinese
        self.assertIn("hl=zh-TW", google_search_url("q", "tw-tzh"))
        # xa-ar (pan-Arabic): language only, no country
        url = google_search_url("q", "xa-ar")
        self.assertIn("hl=ar", url)
        self.assertNotIn("gl=", url)


class TestServerParam(unittest.TestCase):

    def test_search_regions_passthrough(self):
        from deep_research_tool.webui.server import build_config_kwargs
        kwargs = build_config_kwargs({"search_regions": ["de", "th"]})
        self.assertEqual(kwargs["search_regions"], ["de", "th"])


if __name__ == "__main__":
    unittest.main()
