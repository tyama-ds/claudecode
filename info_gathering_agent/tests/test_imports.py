"""
Test that all public imports work correctly.
"""

import pytest


class TestPackageImports:
    """Test top-level package imports."""

    def test_import_package(self):
        import info_gathering_agent
        assert hasattr(info_gathering_agent, "InfoGatheringAgent")
        assert hasattr(info_gathering_agent, "Config")
        assert hasattr(info_gathering_agent, "create_config")
        assert hasattr(info_gathering_agent, "GatheringResult")
        assert hasattr(info_gathering_agent, "run_gathering")

    def test_import_main_classes(self):
        from info_gathering_agent import InfoGatheringAgent, GatheringResult
        assert InfoGatheringAgent is not None
        assert GatheringResult is not None

    def test_import_config(self):
        from info_gathering_agent import Config, create_config
        assert Config is not None
        assert create_config is not None

    def test_import_run_gathering(self):
        from info_gathering_agent import run_gathering
        assert callable(run_gathering)


class TestConfigImports:
    """Test config module imports."""

    def test_import_enums(self):
        from info_gathering_agent.config import (
            LLMProvider,
            LocalLLMBackend,
            SearchMethod,
            ContentFilterMode,
            CrawlMode,
        )
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.LOCAL.value == "local"
        assert SearchMethod.DUCKDUCKGO.value == "duckduckgo"
        assert ContentFilterMode.MODERATE.value == "moderate"
        assert CrawlMode.STANDARD.value == "standard"

    def test_import_dataclasses(self):
        from info_gathering_agent.config import (
            APIConfig,
            SearchConfig,
            ResearchConfig,
            ProxyConfig,
            MultilingualSearchConfig,
            Config,
        )
        assert APIConfig is not None
        assert SearchConfig is not None
        assert ResearchConfig is not None
        assert ProxyConfig is not None
        assert MultilingualSearchConfig is not None
        assert Config is not None

    def test_no_report_imports(self):
        """Verify that report-related classes are NOT present."""
        import info_gathering_agent.config as config_mod
        assert not hasattr(config_mod, "ReportFormat")
        assert not hasattr(config_mod, "ReportConfig")
        assert not hasattr(config_mod, "DeepThinkConfig")


class TestResearchImports:
    """Test research module imports."""

    def test_import_gatherer(self):
        from info_gathering_agent.research.gatherer import (
            Gatherer,
            GatheringSession,
            GatheringState,
            GatheringIteration,
        )
        assert Gatherer is not None
        assert GatheringSession is not None
        assert GatheringState is not None
        assert GatheringIteration is not None

    def test_import_query_generator(self):
        from info_gathering_agent.research.query_generator import (
            QueryGenerator,
            ResearchPlan,
            TableOfContents,
        )
        assert QueryGenerator is not None
        assert ResearchPlan is not None
        assert TableOfContents is not None

    def test_import_content_extractor(self):
        from info_gathering_agent.research.content_extractor import (
            ContentExtractor,
            ExtractedContent,
        )
        assert ContentExtractor is not None
        assert ExtractedContent is not None

    def test_import_research_module(self):
        from info_gathering_agent.research import (
            Gatherer,
            GatheringSession,
            GatheringState,
            QueryGenerator,
            ResearchPlan,
            ContentExtractor,
            ExtractedContent,
        )
        assert Gatherer is not None


class TestApiImports:
    """Test API module imports."""

    def test_import_get_client(self):
        from info_gathering_agent.api import get_client
        assert callable(get_client)

    def test_import_base(self):
        from info_gathering_agent.api.base import (
            BaseLLMClient,
            Message,
            TokenUsage,
            get_token_stats,
            reset_token_stats,
        )
        assert BaseLLMClient is not None
        assert Message is not None
        assert TokenUsage is not None
        assert callable(get_token_stats)
        assert callable(reset_token_stats)


class TestEvidenceImports:
    """Test evidence module imports."""

    def test_import_evidence_locker(self):
        from info_gathering_agent.evidence.locker import (
            EvidenceLocker,
            Evidence,
            EvidenceType,
        )
        assert EvidenceLocker is not None
        assert Evidence is not None
        assert EvidenceType is not None

    def test_import_content_filter(self):
        from info_gathering_agent.evidence.content_filter import (
            ContentFilter,
            ContentFilterConfig,
            create_moderate_filter,
        )
        assert ContentFilter is not None
        assert ContentFilterConfig is not None
        assert callable(create_moderate_filter)

    def test_import_evidence_package(self):
        from info_gathering_agent.evidence import (
            EvidenceLocker,
            Evidence,
            EvidenceType,
            ContentFilter,
        )
        assert EvidenceLocker is not None


class TestSearchImports:
    """Test search module imports."""

    def test_import_search_base(self):
        from info_gathering_agent.search.base import (
            BaseSearchClient,
            SearchResult,
            PageContent,
        )
        assert BaseSearchClient is not None
        assert SearchResult is not None
        assert PageContent is not None

    def test_import_get_search_client(self):
        from info_gathering_agent.search import get_search_client
        assert callable(get_search_client)


class TestUtilsImports:
    """Test utils module imports."""

    def test_import_helpers(self):
        from info_gathering_agent.utils.helpers import (
            setup_logging,
            format_timestamp,
            truncate_text,
            chunk_text,
        )
        assert callable(setup_logging)
        assert callable(format_timestamp)
        assert callable(truncate_text)
        assert callable(chunk_text)

    def test_import_utils_package(self):
        from info_gathering_agent.utils import (
            setup_logging,
            format_timestamp,
            truncate_text,
            chunk_text,
        )
        assert callable(setup_logging)
