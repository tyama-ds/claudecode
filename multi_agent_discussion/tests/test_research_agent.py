"""Tests for research participant agent and search mixin."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from multi_agent_discussion.config import (
    AgentConfig,
    AgentRole,
    LLMConfig,
    LLMProvider,
    Config,
    DiscussionConfig,
    create_config,
)
from multi_agent_discussion.agents.search_mixin import (
    AgentSearchConfig,
    SearchCapabilityMixin,
)
from multi_agent_discussion.agents.research_participant import (
    ResearchParticipantAgent,
)
from multi_agent_discussion.agents import create_agent


class TestAgentSearchConfig:
    """Tests for AgentSearchConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AgentSearchConfig()
        assert config.enabled is True
        assert config.search_method == "duckduckgo"
        assert config.max_queries_per_turn == 3
        assert config.max_results_per_query == 5
        assert config.max_content_length == 500
        assert config.extract_page_content is False
        assert config.region == "jp-jp"
        assert config.search_kwargs == {}

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AgentSearchConfig(
            enabled=False,
            search_method="selenium",
            max_queries_per_turn=2,
            max_results_per_query=10,
            region="en-us",
        )
        assert config.enabled is False
        assert config.search_method == "selenium"
        assert config.max_queries_per_turn == 2
        assert config.max_results_per_query == 10
        assert config.region == "en-us"


class TestAgentRole:
    """Tests for AgentRole enum with RESEARCH_PARTICIPANT."""

    def test_research_participant_role_exists(self):
        """Test that RESEARCH_PARTICIPANT role exists."""
        assert AgentRole.RESEARCH_PARTICIPANT == "research_participant"

    def test_all_roles(self):
        """Test all roles are defined."""
        assert AgentRole.MODERATOR == "moderator"
        assert AgentRole.PARTICIPANT == "participant"
        assert AgentRole.RESEARCH_PARTICIPANT == "research_participant"
        assert AgentRole.EVALUATOR == "evaluator"


class TestAgentConfigSearchConfig:
    """Tests for search_config field in AgentConfig."""

    def test_agent_config_with_search_config(self):
        """Test creating AgentConfig with search_config."""
        config = AgentConfig(
            name="test",
            role=AgentRole.RESEARCH_PARTICIPANT,
            persona="test persona",
            search_config={"region": "en-us", "max_queries_per_turn": 2},
        )
        assert config.search_config is not None
        assert config.search_config["region"] == "en-us"
        assert config.search_config["max_queries_per_turn"] == 2

    def test_agent_config_without_search_config(self):
        """Test creating AgentConfig without search_config."""
        config = AgentConfig(
            name="test",
            role=AgentRole.PARTICIPANT,
        )
        assert config.search_config is None

    def test_research_participant_default_prompt(self):
        """Test default system prompt for RESEARCH_PARTICIPANT."""
        config = AgentConfig(
            name="test",
            role=AgentRole.RESEARCH_PARTICIPANT,
        )
        assert "調査能力" in config.system_prompt
        assert "ウェブ検索" in config.system_prompt


class TestCreateConfigWithSearch:
    """Tests for create_config with enable_search."""

    def test_create_config_enable_search_false(self):
        """Test create_config with enable_search=False (default)."""
        config = create_config(topic="test", enable_search=False)
        participants = [
            a for a in config.agents
            if a.role in (AgentRole.PARTICIPANT, AgentRole.RESEARCH_PARTICIPANT)
        ]
        for p in participants:
            assert p.role == AgentRole.PARTICIPANT
            assert p.search_config is None

    def test_create_config_enable_search_true(self):
        """Test create_config with enable_search=True."""
        config = create_config(topic="test", enable_search=True)
        participants = [
            a for a in config.agents
            if a.role in (AgentRole.PARTICIPANT, AgentRole.RESEARCH_PARTICIPANT)
        ]
        for p in participants:
            assert p.role == AgentRole.RESEARCH_PARTICIPANT

    def test_create_config_with_search_config(self):
        """Test create_config with custom search_config."""
        search_cfg = {"region": "en-us", "max_queries_per_turn": 1}
        config = create_config(
            topic="test",
            enable_search=True,
            search_config=search_cfg,
        )
        participants = [
            a for a in config.agents
            if a.role == AgentRole.RESEARCH_PARTICIPANT
        ]
        for p in participants:
            assert p.search_config == search_cfg

    def test_create_config_custom_personas_with_search(self):
        """Test create_config with custom personas and search enabled."""
        personas = [
            {"name": "Expert 1", "persona": "Technical expert"},
            {"name": "Expert 2", "persona": "Business expert"},
        ]
        config = create_config(
            topic="test",
            participant_personas=personas,
            enable_search=True,
        )
        participants = [
            a for a in config.agents
            if a.role == AgentRole.RESEARCH_PARTICIPANT
        ]
        assert len(participants) == 2
        assert participants[0].name == "Expert 1"
        assert participants[1].name == "Expert 2"


class TestConfigValidation:
    """Tests for Config validation with research participants."""

    def test_validation_counts_research_participants(self):
        """Test that validation counts RESEARCH_PARTICIPANT as participants."""
        config = Config(
            llm=LLMConfig(provider=LLMProvider.OPENAI),
            discussion=DiscussionConfig(topic="test"),
            agents=[
                AgentConfig(name="mod", role=AgentRole.MODERATOR),
                AgentConfig(name="p1", role=AgentRole.RESEARCH_PARTICIPANT),
                AgentConfig(name="p2", role=AgentRole.RESEARCH_PARTICIPANT),
            ],
        )
        errors = config.validate()
        # Should not have "At least two participant agents required" error
        participant_errors = [e for e in errors if "participant" in e.lower()]
        assert len(participant_errors) == 0

    def test_validation_fails_with_one_research_participant(self):
        """Test that validation fails with only one research participant."""
        config = Config(
            llm=LLMConfig(provider=LLMProvider.OPENAI),
            discussion=DiscussionConfig(topic="test"),
            agents=[
                AgentConfig(name="mod", role=AgentRole.MODERATOR),
                AgentConfig(name="p1", role=AgentRole.RESEARCH_PARTICIPANT),
            ],
        )
        errors = config.validate()
        participant_errors = [e for e in errors if "participant" in e.lower()]
        assert len(participant_errors) > 0


class TestCreateAgentFactory:
    """Tests for create_agent factory with RESEARCH_PARTICIPANT."""

    def test_create_research_participant(self):
        """Test creating a research participant agent via factory."""
        config = AgentConfig(
            name="test",
            role=AgentRole.RESEARCH_PARTICIPANT,
            persona="test persona",
            search_config={"region": "en-us"},
        )
        llm_config = LLMConfig(provider=LLMProvider.OPENAI)

        agent = create_agent(config, llm_config)

        assert isinstance(agent, ResearchParticipantAgent)
        assert agent.name == "test"
        assert agent.search_config.region == "en-us"

    def test_create_research_participant_default_search_config(self):
        """Test creating research participant with default search config."""
        config = AgentConfig(
            name="test",
            role=AgentRole.RESEARCH_PARTICIPANT,
        )
        llm_config = LLMConfig(provider=LLMProvider.OPENAI)

        agent = create_agent(config, llm_config)

        assert isinstance(agent, ResearchParticipantAgent)
        assert agent.search_config is not None
        assert agent.search_config.enabled is True


class TestSearchCapabilityMixin:
    """Tests for SearchCapabilityMixin."""

    def test_search_history_tracking(self):
        """Test that search history is tracked."""
        config = AgentConfig(
            name="test",
            role=AgentRole.RESEARCH_PARTICIPANT,
            persona="test",
        )
        llm_config = LLMConfig(provider=LLMProvider.OPENAI)
        agent = ResearchParticipantAgent(config, llm_config)

        assert agent.get_search_history() == []
        agent.clear_search_history()
        assert agent.get_search_history() == []

    def test_format_search_results_empty(self):
        """Test formatting empty search results."""
        config = AgentConfig(
            name="test",
            role=AgentRole.RESEARCH_PARTICIPANT,
            persona="test",
        )
        llm_config = LLMConfig(provider=LLMProvider.OPENAI)
        agent = ResearchParticipantAgent(config, llm_config)

        result = agent.format_search_results([])
        assert result == "(検索結果なし)"

    def test_format_search_results_with_results(self):
        """Test formatting search results."""
        config = AgentConfig(
            name="test",
            role=AgentRole.RESEARCH_PARTICIPANT,
            persona="test",
        )
        llm_config = LLMConfig(provider=LLMProvider.OPENAI)
        agent = ResearchParticipantAgent(config, llm_config)

        # Create mock search results
        @dataclass
        class MockSearchResult:
            title: str
            url: str
            snippet: str
            content: str = ""

        results = [
            MockSearchResult(
                title="Test Title 1",
                url="https://example.com/1",
                snippet="This is a test snippet.",
            ),
            MockSearchResult(
                title="Test Title 2",
                url="https://example.com/2",
                snippet="Another test snippet.",
            ),
        ]

        formatted = agent.format_search_results(results)
        assert "Test Title 1" in formatted
        assert "https://example.com/1" in formatted
        assert "This is a test snippet" in formatted
        assert "Test Title 2" in formatted

    def test_format_search_results_deduplicates_urls(self):
        """Test that duplicate URLs are removed."""
        config = AgentConfig(
            name="test",
            role=AgentRole.RESEARCH_PARTICIPANT,
            persona="test",
        )
        llm_config = LLMConfig(provider=LLMProvider.OPENAI)
        agent = ResearchParticipantAgent(config, llm_config)

        @dataclass
        class MockSearchResult:
            title: str
            url: str
            snippet: str
            content: str = ""

        results = [
            MockSearchResult(title="Title 1", url="https://example.com", snippet="First"),
            MockSearchResult(title="Title 2", url="https://example.com", snippet="Duplicate"),
        ]

        formatted = agent.format_search_results(results)
        # Should only have one entry for the URL
        assert formatted.count("https://example.com") == 1

    def test_format_search_results_truncates_long_snippets(self):
        """Test that long snippets are truncated."""
        config = AgentConfig(
            name="test",
            role=AgentRole.RESEARCH_PARTICIPANT,
            persona="test",
            search_config={"max_content_length": 50},
        )
        llm_config = LLMConfig(provider=LLMProvider.OPENAI)
        search_cfg = AgentSearchConfig(max_content_length=50)
        agent = ResearchParticipantAgent(config, llm_config, search_config=search_cfg)

        @dataclass
        class MockSearchResult:
            title: str
            url: str
            snippet: str
            content: str = ""

        results = [
            MockSearchResult(
                title="Test",
                url="https://example.com",
                snippet="A" * 100,  # Long snippet
            ),
        ]

        formatted = agent.format_search_results(results)
        # Should be truncated with ...
        assert "..." in formatted

    def test_research_and_build_context_disabled(self):
        """Test research_and_build_context when search is disabled."""
        config = AgentConfig(
            name="test",
            role=AgentRole.RESEARCH_PARTICIPANT,
            persona="test",
        )
        llm_config = LLMConfig(provider=LLMProvider.OPENAI)
        search_cfg = AgentSearchConfig(enabled=False)
        agent = ResearchParticipantAgent(config, llm_config, search_config=search_cfg)

        context, metadata = agent.research_and_build_context("topic", [], None)
        assert context == ""
        assert metadata == {}


class TestResearchParticipantAgent:
    """Tests for ResearchParticipantAgent."""

    def test_inheritance(self):
        """Test that ResearchParticipantAgent inherits from correct classes."""
        from multi_agent_discussion.agents.participant import ParticipantAgent

        config = AgentConfig(
            name="test",
            role=AgentRole.RESEARCH_PARTICIPANT,
            persona="test",
        )
        llm_config = LLMConfig(provider=LLMProvider.OPENAI)
        agent = ResearchParticipantAgent(config, llm_config)

        assert isinstance(agent, ResearchParticipantAgent)
        assert isinstance(agent, SearchCapabilityMixin)
        assert isinstance(agent, ParticipantAgent)

    def test_has_search_config(self):
        """Test that agent has search_config attribute."""
        config = AgentConfig(
            name="test",
            role=AgentRole.RESEARCH_PARTICIPANT,
            persona="test",
        )
        llm_config = LLMConfig(provider=LLMProvider.OPENAI)
        agent = ResearchParticipantAgent(config, llm_config)

        assert hasattr(agent, "search_config")
        assert isinstance(agent.search_config, AgentSearchConfig)

    def test_custom_search_config(self):
        """Test agent with custom search config."""
        config = AgentConfig(
            name="test",
            role=AgentRole.RESEARCH_PARTICIPANT,
            persona="test",
        )
        llm_config = LLMConfig(provider=LLMProvider.OPENAI)
        custom_cfg = AgentSearchConfig(region="en-us", max_queries_per_turn=1)
        agent = ResearchParticipantAgent(config, llm_config, search_config=custom_cfg)

        assert agent.search_config.region == "en-us"
        assert agent.search_config.max_queries_per_turn == 1

    def test_has_response_methods(self):
        """Test that agent has all required response methods."""
        config = AgentConfig(
            name="test",
            role=AgentRole.RESEARCH_PARTICIPANT,
            persona="test",
        )
        llm_config = LLMConfig(provider=LLMProvider.OPENAI)
        agent = ResearchParticipantAgent(config, llm_config)

        assert hasattr(agent, "generate_response")
        assert hasattr(agent, "generate_initial_opinion")
        assert hasattr(agent, "generate_rebuttal")
        assert hasattr(agent, "generate_agreement")
        assert callable(agent.generate_response)
        assert callable(agent.generate_initial_opinion)


class TestImports:
    """Tests for module imports."""

    def test_import_from_main_package(self):
        """Test importing from main package."""
        from multi_agent_discussion import (
            ResearchParticipantAgent,
            SearchCapabilityMixin,
            AgentSearchConfig,
        )
        assert ResearchParticipantAgent is not None
        assert SearchCapabilityMixin is not None
        assert AgentSearchConfig is not None

    def test_import_from_agents_package(self):
        """Test importing from agents package."""
        from multi_agent_discussion.agents import (
            ResearchParticipantAgent,
            SearchCapabilityMixin,
            AgentSearchConfig,
        )
        assert ResearchParticipantAgent is not None
        assert SearchCapabilityMixin is not None
        assert AgentSearchConfig is not None

    def test_create_agent_in_all(self):
        """Test that create_agent is exported."""
        from multi_agent_discussion.agents import create_agent
        assert create_agent is not None
        assert callable(create_agent)


class TestSearchMixinWithoutDeepResearchTool:
    """Tests for SearchCapabilityMixin when deep_research_tool is not available."""

    def test_research_without_tool_returns_empty(self):
        """Test that research returns empty when tool is not available."""
        config = AgentConfig(
            name="test",
            role=AgentRole.RESEARCH_PARTICIPANT,
            persona="test",
        )
        llm_config = LLMConfig(provider=LLMProvider.OPENAI)
        agent = ResearchParticipantAgent(config, llm_config)

        # Temporarily disable the search tool availability flag
        import multi_agent_discussion.agents.search_mixin as search_module
        original_flag = search_module.HAS_SEARCH_TOOL
        search_module.HAS_SEARCH_TOOL = False

        try:
            context, metadata = agent.research_and_build_context("topic", [], None)
            assert context == ""
            assert "error" in metadata
        finally:
            search_module.HAS_SEARCH_TOOL = original_flag
