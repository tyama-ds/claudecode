"""
Tests for the simplified configuration module.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch


class TestConfig:
    """Test Config dataclass."""

    def test_default_config(self, tmp_path):
        from info_gathering_agent.config import Config

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = Config(output_dir=tmp_path / "output")
            assert config.api.provider.value == "openai"
            assert config.search.method.value == "duckduckgo"
            assert config.research.min_iterations == 3
            assert config.output_dir == tmp_path / "output"
            assert config.export_evidence_json is True
            assert config.export_evidence_csv is True
            assert config.verbose is False

    def test_no_report_fields(self):
        from info_gathering_agent.config import Config

        config_fields = {f.name for f in Config.__dataclass_fields__.values()}
        assert "report" not in config_fields
        assert "deep_think" not in config_fields
        assert "enable_verification" not in config_fields
        assert "verification_strictness" not in config_fields

    def test_has_output_dir(self, tmp_path):
        from info_gathering_agent.config import Config

        config = Config(output_dir=tmp_path / "test_output")
        assert config.output_dir == tmp_path / "test_output"

    def test_validate_openai_no_key(self, tmp_path):
        from info_gathering_agent.config import Config, APIConfig, LLMProvider

        with patch.dict(os.environ, {}, clear=True):
            config = Config(
                api=APIConfig(provider=LLMProvider.OPENAI, openai_api_key=None),
                output_dir=tmp_path / "output",
            )
            # Force clear the key that __post_init__ might have loaded
            config.api.openai_api_key = None
            errors = config.validate()
            assert any("OpenAI API key" in e for e in errors)

    def test_validate_anthropic_no_key(self, tmp_path):
        from info_gathering_agent.config import Config, APIConfig, LLMProvider

        with patch.dict(os.environ, {}, clear=True):
            config = Config(
                api=APIConfig(provider=LLMProvider.ANTHROPIC, anthropic_api_key=None),
                output_dir=tmp_path / "output",
            )
            config.api.anthropic_api_key = None
            errors = config.validate()
            assert any("Anthropic API key" in e for e in errors)

    def test_validate_iterations(self, tmp_path):
        from info_gathering_agent.config import Config, ResearchConfig

        config = Config(
            research=ResearchConfig(min_iterations=0),
            output_dir=tmp_path / "output",
        )
        errors = config.validate()
        assert any("at least 1" in e for e in errors)

    def test_validate_max_lt_min(self, tmp_path):
        from info_gathering_agent.config import Config, ResearchConfig

        config = Config(
            research=ResearchConfig(min_iterations=5, max_iterations=2),
            output_dir=tmp_path / "output",
        )
        errors = config.validate()
        assert any("Maximum iterations" in e for e in errors)

    def test_from_env(self, tmp_path):
        from info_gathering_agent.config import Config

        with patch.dict(os.environ, {
            "LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "test-key",
            "RESEARCH_ITERATIONS": "7",
            "OUTPUT_DIR": str(tmp_path / "env_output"),
        }):
            config = Config.from_env()
            assert config.api.provider.value == "anthropic"
            assert config.research.min_iterations == 7
            assert config.output_dir == Path(str(tmp_path / "env_output"))


class TestCreateConfig:
    """Test create_config factory function."""

    def test_basic_creation(self, tmp_path):
        from info_gathering_agent.config import create_config

        config = create_config(
            provider="openai",
            openai_api_key="test-key",
            output_dir=str(tmp_path / "output"),
        )
        assert config.api.provider.value == "openai"
        assert config.api.openai_api_key == "test-key"

    def test_anthropic_provider(self, tmp_path):
        from info_gathering_agent.config import create_config

        config = create_config(
            provider="anthropic",
            anthropic_api_key="test-key",
            model="claude-3-opus-20240229",
            output_dir=str(tmp_path / "output"),
        )
        assert config.api.provider.value == "anthropic"
        assert config.api.anthropic_model == "claude-3-opus-20240229"

    def test_research_iterations(self, tmp_path):
        from info_gathering_agent.config import create_config

        config = create_config(
            research_iterations=7,
            output_dir=str(tmp_path / "output"),
        )
        assert config.research.min_iterations == 7

    def test_extended_mode(self, tmp_path):
        from info_gathering_agent.config import create_config

        config = create_config(
            extended_mode=True,
            crawl_max_pages=20,
            crawl_max_depth=3,
            output_dir=str(tmp_path / "output"),
        )
        assert config.research.extended_mode is True
        assert config.research.crawl_max_pages == 20
        assert config.research.crawl_max_depth == 3

    def test_content_filter(self, tmp_path):
        from info_gathering_agent.config import create_config

        config = create_config(
            content_filter_mode="strict",
            custom_blocked_domains=["spam.com"],
            output_dir=str(tmp_path / "output"),
        )
        assert config.research.content_filter_mode.value == "strict"
        assert "spam.com" in config.research.custom_blocked_domains

    def test_fast_crawl_mode(self, tmp_path):
        from info_gathering_agent.config import create_config

        config = create_config(
            crawl_mode="fast_batch",
            fast_crawl_workers=15,
            output_dir=str(tmp_path / "output"),
        )
        assert config.research.crawl_mode.value == "fast_batch"
        assert config.research.fast_crawl_workers == 15

    def test_multilingual(self, tmp_path):
        from info_gathering_agent.config import create_config

        config = create_config(
            multilingual=True,
            search_languages=["ja", "en", "zh"],
            output_dir=str(tmp_path / "output"),
        )
        assert config.multilingual.enabled is True
        assert config.multilingual.search_languages == ["ja", "en", "zh"]

    def test_evidence_export_flags(self, tmp_path):
        from info_gathering_agent.config import create_config

        config = create_config(
            export_evidence_csv=False,
            export_evidence_json=True,
            output_dir=str(tmp_path / "output"),
        )
        assert config.export_evidence_json is True
        assert config.export_evidence_csv is False

    def test_no_report_params(self):
        """Verify create_config doesn't accept report-related params."""
        from info_gathering_agent.config import create_config
        import inspect

        sig = inspect.signature(create_config)
        param_names = set(sig.parameters.keys())

        assert "output_format" not in param_names
        assert "target_pages" not in param_names
        assert "target_characters" not in param_names
        assert "enable_verification" not in param_names
        assert "deep_think" not in param_names
        assert "deep_think_level" not in param_names


class TestEnums:
    """Test enum values."""

    def test_llm_provider_values(self):
        from info_gathering_agent.config import LLMProvider

        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.LOCAL.value == "local"

    def test_search_method_values(self):
        from info_gathering_agent.config import SearchMethod

        assert SearchMethod.DUCKDUCKGO.value == "duckduckgo"
        assert SearchMethod.SELENIUM.value == "selenium"

    def test_content_filter_mode_values(self):
        from info_gathering_agent.config import ContentFilterMode

        assert ContentFilterMode.STRICT.value == "strict"
        assert ContentFilterMode.MODERATE.value == "moderate"
        assert ContentFilterMode.MINIMAL.value == "minimal"
        assert ContentFilterMode.NONE.value == "none"

    def test_crawl_mode_values(self):
        from info_gathering_agent.config import CrawlMode

        assert CrawlMode.STANDARD.value == "standard"
        assert CrawlMode.FAST_BATCH.value == "fast_batch"
        assert CrawlMode.FAST_PARALLEL.value == "fast_parallel"
