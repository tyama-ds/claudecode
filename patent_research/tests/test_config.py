"""Tests for patent research configuration."""

import pytest
from patent_research.config import (
    PatentResearchConfig,
    PatentSearchConfig,
    AuxiliarySearchConfig,
    PatentReportConfig,
    create_patent_config,
)


class TestPatentSearchConfig:
    def test_default_config(self):
        config = PatentSearchConfig()
        assert config.enable_google_patents is True
        assert config.enable_jplatpat is True
        assert config.enable_espacenet is True
        assert config.patent_jurisdictions == ["JP", "US", "EP"]
        assert config.max_patents_per_query == 20
        assert config.track_families is True

    def test_custom_jurisdictions(self):
        config = PatentSearchConfig(patent_jurisdictions=["JP", "CN"])
        assert config.patent_jurisdictions == ["JP", "CN"]


class TestAuxiliarySearchConfig:
    def test_default_config(self):
        config = AuxiliarySearchConfig()
        assert config.enable_academic_search is True
        assert "cinii" in config.academic_sources
        assert "jstage" in config.academic_sources
        assert "google_scholar" in config.academic_sources
        assert config.enable_examination_search is True
        assert config.enable_business_search is True

    def test_threshold_defaults(self):
        config = AuxiliarySearchConfig()
        assert config.technical_term_threshold == 0.6
        assert config.business_term_threshold == 0.5


class TestPatentResearchConfig:
    def test_default_config(self):
        config = PatentResearchConfig()
        assert config.language == "ja"
        assert config.verbose is False
        assert config.min_iterations == 2
        assert config.max_iterations == 5

    def test_validate_no_api_key(self):
        config = PatentResearchConfig()
        config.api.openai_api_key = None
        errors = config.validate()
        assert any("API key" in e for e in errors)


class TestCreatePatentConfig:
    def test_basic_creation(self):
        config = create_patent_config(
            provider="openai",
            language="ja",
        )
        assert config.language == "ja"
        assert config.patent_search.enable_google_patents is True

    def test_custom_ipc(self):
        config = create_patent_config(
            ipc_codes=["H01L21/00", "G06F3/041"],
        )
        assert "H01L21/00" in config.patent_search.ipc_codes

    def test_disable_sources(self):
        config = create_patent_config(
            enable_google_patents=False,
            enable_jplatpat=True,
            enable_espacenet=False,
        )
        assert config.patent_search.enable_google_patents is False
        assert config.patent_search.enable_jplatpat is True
        assert config.patent_search.enable_espacenet is False
