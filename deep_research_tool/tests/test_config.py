"""
Tests for configuration module.
"""

import os
import pytest
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deep_research_tool.config import (
    Config,
    APIConfig,
    SearchConfig,
    ResearchConfig,
    ReportConfig,
    LLMProvider,
    SearchMethod,
    ReportFormat,
    create_config,
)


class TestAPIConfig:
    """Tests for APIConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = APIConfig()
        assert config.provider == LLMProvider.OPENAI
        assert config.openai_model == "gpt-4o-mini"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096

    def test_get_active_model_openai(self):
        """Test getting active model for OpenAI."""
        config = APIConfig(provider=LLMProvider.OPENAI)
        assert config.get_active_model() == "gpt-4o-mini"

    def test_get_active_model_anthropic(self):
        """Test getting active model for Anthropic."""
        config = APIConfig(provider=LLMProvider.ANTHROPIC)
        assert config.get_active_model() == "claude-3-5-sonnet-20241022"


class TestSearchConfig:
    """Tests for SearchConfig."""

    def test_default_values(self):
        """Test default search configuration."""
        config = SearchConfig()
        assert config.method == SearchMethod.DUCKDUCKGO
        assert config.max_results == 10
        assert config.headless is True


class TestResearchConfig:
    """Tests for ResearchConfig."""

    def test_default_iterations(self):
        """Test default iteration settings."""
        config = ResearchConfig()
        assert config.min_iterations == 3
        assert config.max_iterations == 10


class TestConfig:
    """Tests for main Config class."""

    def test_default_config(self):
        """Test creating default configuration."""
        config = Config()
        assert config.api.provider == LLMProvider.OPENAI
        assert config.search.method == SearchMethod.DUCKDUCKGO
        assert config.enable_verification is True

    def test_validate_missing_api_key(self):
        """Test validation catches missing API key."""
        config = Config()
        config.api.openai_api_key = None
        errors = config.validate()
        assert len(errors) > 0
        assert "OpenAI API key" in errors[0]

    def test_validate_iteration_settings(self):
        """Test validation of iteration settings."""
        config = Config()
        config.api.openai_api_key = "test-key"
        config.research.min_iterations = 0
        errors = config.validate()
        assert any("at least 1" in e for e in errors)


class TestCreateConfig:
    """Tests for create_config factory function."""

    def test_create_basic_config(self):
        """Test creating basic configuration."""
        config = create_config(
            provider="openai",
            openai_api_key="test-key",
        )
        assert config.api.provider == LLMProvider.OPENAI
        assert config.api.openai_api_key == "test-key"

    def test_create_anthropic_config(self):
        """Test creating Anthropic configuration."""
        config = create_config(
            provider="anthropic",
            anthropic_api_key="test-key",
            model="claude-3-opus",
        )
        assert config.api.provider == LLMProvider.ANTHROPIC
        assert config.api.anthropic_model == "claude-3-opus"

    def test_create_with_documents(self):
        """Test creating configuration with additional documents."""
        config = create_config(
            provider="openai",
            openai_api_key="test-key",
            additional_documents=["doc1.pdf", "doc2.docx"],
        )
        assert len(config.additional_documents) == 2
        assert config.process_additional_documents is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
