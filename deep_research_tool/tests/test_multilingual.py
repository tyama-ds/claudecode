"""Tests for multilingual search module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from deep_research_tool.config import (
    MultilingualSearchConfig,
    LANGUAGE_REGION_MAP,
    create_config,
)
from deep_research_tool.search.multilingual import (
    MultilingualSearcher,
    MultilingualSearchResult,
    MultilingualSearchStats,
    TranslatedQuery,
    create_multilingual_searcher,
)


class TestLanguageRegionMap:
    """Tests for language region mapping."""

    def test_japanese_mapping(self):
        assert "ja" in LANGUAGE_REGION_MAP
        assert LANGUAGE_REGION_MAP["ja"]["region"] == "jp-jp"
        assert LANGUAGE_REGION_MAP["ja"]["name"] == "Japanese"

    def test_english_mapping(self):
        assert "en" in LANGUAGE_REGION_MAP
        assert LANGUAGE_REGION_MAP["en"]["region"] == "us-en"
        assert LANGUAGE_REGION_MAP["en"]["name"] == "English"

    def test_chinese_mapping(self):
        assert "zh" in LANGUAGE_REGION_MAP
        assert LANGUAGE_REGION_MAP["zh"]["region"] == "cn-zh"
        assert LANGUAGE_REGION_MAP["zh"]["name"] == "Chinese"

    def test_all_languages_have_required_fields(self):
        for code, info in LANGUAGE_REGION_MAP.items():
            assert "region" in info
            assert "name" in info
            assert "native" in info


class TestMultilingualSearchConfig:
    """Tests for MultilingualSearchConfig."""

    def test_default_values(self):
        config = MultilingualSearchConfig()
        assert config.enabled is False
        assert config.search_languages == ["ja", "en"]
        assert config.results_per_language == 10
        assert config.query_translation == "llm"
        assert config.translate_results is True
        assert config.dedup_threshold == 0.85

    def test_custom_languages(self):
        config = MultilingualSearchConfig(
            enabled=True,
            search_languages=["ja", "en", "zh", "de"]
        )
        assert config.enabled is True
        assert len(config.search_languages) == 4
        assert "zh" in config.search_languages

    def test_get_language_weight(self):
        config = MultilingualSearchConfig()
        assert config.get_language_weight("ja") == 1.0
        assert config.get_language_weight("en") == 1.0
        assert config.get_language_weight("zh") == 0.9
        assert config.get_language_weight("unknown") == 0.5  # default

    def test_get_region_for_language(self):
        config = MultilingualSearchConfig()
        assert config.get_region_for_language("ja") == "jp-jp"
        assert config.get_region_for_language("en") == "us-en"
        assert config.get_region_for_language("unknown") == "wt-wt"


class TestTranslatedQuery:
    """Tests for TranslatedQuery dataclass."""

    def test_creation(self):
        query = TranslatedQuery(
            original_query="量子コンピュータ",
            translated_query="quantum computer",
            target_language="en",
            confidence=0.9
        )
        assert query.original_query == "量子コンピュータ"
        assert query.translated_query == "quantum computer"
        assert query.target_language == "en"
        assert query.confidence == 0.9


class TestMultilingualSearchResult:
    """Tests for MultilingualSearchResult dataclass."""

    def test_creation(self):
        result = MultilingualSearchResult(
            url="https://example.com",
            title="Test Title",
            snippet="Test snippet",
            source_language="en",
            search_query="test query"
        )
        assert result.url == "https://example.com"
        assert result.source_language == "en"
        assert result.is_translated is False

    def test_content_hash(self):
        result1 = MultilingualSearchResult(
            url="https://example.com/page1",
            title="Title 1",
            snippet="Snippet 1",
            source_language="en",
            search_query="query"
        )
        result2 = MultilingualSearchResult(
            url="https://example.com/page1",
            title="Different Title",
            snippet="Different Snippet",
            source_language="ja",
            search_query="query"
        )
        # Same URL should produce same hash
        assert result1.get_content_hash() == result2.get_content_hash()


class TestMultilingualSearchStats:
    """Tests for MultilingualSearchStats dataclass."""

    def test_default_values(self):
        stats = MultilingualSearchStats()
        assert stats.total_results == 0
        assert stats.results_by_language == {}
        assert stats.duplicates_removed == 0

    def test_to_dict(self):
        stats = MultilingualSearchStats(
            total_results=100,
            results_by_language={"ja": 50, "en": 50},
            duplicates_removed=10
        )
        data = stats.to_dict()
        assert data["total_results"] == 100
        assert data["results_by_language"]["ja"] == 50

    def test_get_language_distribution(self):
        stats = MultilingualSearchStats(
            total_results=100,
            results_by_language={"ja": 60, "en": 40}
        )
        distribution = stats.get_language_distribution()
        assert len(distribution) == 2
        # Sorted by count descending
        assert distribution[0][0] == "Japanese"
        assert distribution[0][1] == 60
        assert distribution[0][2] == 60.0  # percentage


class TestMultilingualSearcher:
    """Tests for MultilingualSearcher class."""

    @pytest.fixture
    def mock_search_client(self):
        """Create a mock search client."""
        client = Mock()

        @dataclass
        class MockResult:
            url: str
            title: str
            snippet: str

        client.search.return_value = [
            MockResult("https://example.com/1", "Result 1", "Snippet 1"),
            MockResult("https://example.com/2", "Result 2", "Snippet 2"),
        ]
        return client

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return MultilingualSearchConfig(
            enabled=True,
            search_languages=["ja", "en"],
            results_per_language=5,
            query_translation="none",  # Don't translate for tests
        )

    def test_searcher_creation(self, config, mock_search_client):
        searcher = MultilingualSearcher(
            config=config,
            search_client=mock_search_client
        )
        assert searcher.config == config
        assert searcher.search_client == mock_search_client

    def test_translate_queries_no_translation(self, config, mock_search_client):
        config.query_translation = "none"
        searcher = MultilingualSearcher(config=config, search_client=mock_search_client)

        queries = searcher.translate_queries("test query")
        assert len(queries) == 2  # ja and en
        for q in queries:
            assert q.translated_query == "test query"  # No translation

    def test_search_single_language(self, config, mock_search_client):
        searcher = MultilingualSearcher(config=config, search_client=mock_search_client)

        query = TranslatedQuery(
            original_query="test",
            translated_query="test",
            target_language="en"
        )
        results = searcher.search_single_language(query)

        assert len(results) == 2
        assert results[0].source_language == "en"
        mock_search_client.search.assert_called_once()

    def test_deduplicate_results(self, config, mock_search_client):
        searcher = MultilingualSearcher(config=config, search_client=mock_search_client)

        results = [
            MultilingualSearchResult(
                url="https://example.com/1",
                title="Same Title",
                snippet="Snippet",
                source_language="ja",
                search_query="query"
            ),
            MultilingualSearchResult(
                url="https://example.com/1",  # Duplicate URL
                title="Same Title",
                snippet="Different snippet",
                source_language="en",
                search_query="query"
            ),
            MultilingualSearchResult(
                url="https://example.com/2",
                title="Different Title",
                snippet="Snippet",
                source_language="en",
                search_query="query"
            ),
        ]

        deduplicated = searcher._deduplicate_results(results)
        assert len(deduplicated) == 2  # One duplicate removed

    def test_score_results(self, config, mock_search_client):
        searcher = MultilingualSearcher(config=config, search_client=mock_search_client)

        results = [
            MultilingualSearchResult(
                url="https://example.com/1",
                title="Title 1",
                snippet="Snippet 1",
                source_language="ja",
                search_query="query"
            ),
            MultilingualSearchResult(
                url="https://example.com/2",
                title="Title 2",
                snippet="Snippet 2",
                source_language="en",
                search_query="query"
            ),
        ]

        scored = searcher._score_results(results)
        assert all(r.relevance_score > 0 for r in scored)


class TestCreateMultilingualSearcher:
    """Tests for create_multilingual_searcher factory function."""

    def test_factory_function(self):
        config = MultilingualSearchConfig(enabled=True)
        search_client = Mock()

        searcher = create_multilingual_searcher(
            config=config,
            search_client=search_client
        )

        assert isinstance(searcher, MultilingualSearcher)
        assert searcher.config == config


class TestConfigIntegration:
    """Tests for multilingual config integration with create_config."""

    def test_create_config_with_multilingual(self):
        config = create_config(
            provider="openai",
            multilingual=True,
            search_languages=["ja", "en", "zh"],
            results_per_language=15,
            translate_results=True,
        )

        assert config.multilingual.enabled is True
        assert config.multilingual.search_languages == ["ja", "en", "zh"]
        assert config.multilingual.results_per_language == 15
        assert config.multilingual.translate_results is True

    def test_create_config_default_multilingual(self):
        config = create_config(provider="openai")

        assert config.multilingual.enabled is False
        assert config.multilingual.search_languages == ["ja", "en"]

    def test_gpt5_default_model(self):
        config = create_config(provider="openai")
        assert config.api.openai_model == "gpt-5-mini"


class TestEvidenceLanguageFields:
    """Tests for Evidence language fields."""

    def test_evidence_has_language_fields(self):
        from deep_research_tool.evidence.locker import Evidence

        evidence = Evidence(
            url="https://example.com",
            title="Test Title",
            content_excerpt="Test content",
            source_language="ja",
            original_title="テストタイトル",
            original_content="テスト内容",
            translated_title="Test Title",
            translated_content="Test content",
            translation_confidence=0.9,
            is_translated=True,
        )

        assert evidence.source_language == "ja"
        assert evidence.original_title == "テストタイトル"
        assert evidence.is_translated is True
        assert evidence.translation_confidence == 0.9

    def test_evidence_to_dict_includes_language_fields(self):
        from deep_research_tool.evidence.locker import Evidence

        evidence = Evidence(
            url="https://example.com",
            title="Test",
            source_language="en",
            is_translated=False,
        )

        data = evidence.to_dict()
        assert "source_language" in data
        assert "is_translated" in data
        assert "translation_confidence" in data
