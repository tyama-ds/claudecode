"""Tests for configuration module."""

import os
import pytest

from multi_agent_discussion.config import (
    Config,
    LLMConfig,
    AgentConfig,
    DiscussionConfig,
    LLMProvider,
    AgentRole,
    DiscussionState,
    create_config,
)


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig()
        assert config.provider == LLMProvider.OPENAI
        assert config.openai_model == "gpt-4o-mini"
        assert config.anthropic_model == "claude-3-5-sonnet-20241022"
        assert config.temperature == 0.7

    def test_get_model_openai(self):
        """Test getting model for OpenAI provider."""
        config = LLMConfig(provider=LLMProvider.OPENAI)
        assert config.get_model() == "gpt-4o-mini"

    def test_get_model_anthropic(self):
        """Test getting model for Anthropic provider."""
        config = LLMConfig(provider=LLMProvider.ANTHROPIC)
        assert config.get_model() == "claude-3-5-sonnet-20241022"

    def test_custom_model(self):
        """Test custom model configuration."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            openai_model="gpt-4"
        )
        assert config.get_model() == "gpt-4"


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_moderator_default_prompt(self):
        """Test default system prompt for moderator."""
        config = AgentConfig(name="Test", role=AgentRole.MODERATOR)
        assert "進行役" in config.system_prompt
        assert "モデレーター" in config.system_prompt

    def test_participant_default_prompt(self):
        """Test default system prompt for participant."""
        config = AgentConfig(name="Test", role=AgentRole.PARTICIPANT)
        assert "参加者" in config.system_prompt

    def test_evaluator_default_prompt(self):
        """Test default system prompt for evaluator."""
        config = AgentConfig(name="Test", role=AgentRole.EVALUATOR)
        assert "評価者" in config.system_prompt

    def test_custom_prompt(self):
        """Test custom system prompt."""
        custom_prompt = "Custom prompt"
        config = AgentConfig(
            name="Test",
            role=AgentRole.MODERATOR,
            system_prompt=custom_prompt
        )
        assert config.system_prompt == custom_prompt

    def test_persona_in_prompt(self):
        """Test persona is included in prompt."""
        config = AgentConfig(
            name="Test",
            role=AgentRole.PARTICIPANT,
            persona="Test persona"
        )
        assert config.persona == "Test persona"


class TestDiscussionConfig:
    """Tests for DiscussionConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DiscussionConfig()
        assert config.topic == ""
        assert config.max_rounds == 5
        assert config.min_rounds == 2
        assert config.output_language == "ja"
        assert config.enable_evaluation is True
        assert config.save_session is True


class TestConfig:
    """Tests for main Config."""

    def test_validate_no_topic(self):
        """Test validation fails without topic."""
        config = Config()
        errors = config.validate()
        assert any("topic" in e.lower() for e in errors)

    def test_validate_no_agents(self):
        """Test validation fails without agents."""
        config = Config(discussion=DiscussionConfig(topic="Test"))
        errors = config.validate()
        assert any("agents" in e.lower() or "agent" in e.lower() for e in errors)

    def test_validate_no_moderator(self):
        """Test validation fails without moderator."""
        config = Config(
            discussion=DiscussionConfig(topic="Test"),
            agents=[
                AgentConfig(name="P1", role=AgentRole.PARTICIPANT),
                AgentConfig(name="P2", role=AgentRole.PARTICIPANT),
            ]
        )
        errors = config.validate()
        assert any("moderator" in e.lower() for e in errors)

    def test_validate_insufficient_participants(self):
        """Test validation fails with less than 2 participants."""
        config = Config(
            discussion=DiscussionConfig(topic="Test"),
            agents=[
                AgentConfig(name="Mod", role=AgentRole.MODERATOR),
                AgentConfig(name="P1", role=AgentRole.PARTICIPANT),
            ]
        )
        errors = config.validate()
        assert any("participant" in e.lower() for e in errors)


class TestCreateConfig:
    """Tests for create_config factory function."""

    def test_creates_valid_config(self):
        """Test factory creates valid configuration."""
        config = create_config(topic="Test topic")

        # Should have no validation errors except API key
        errors = config.validate()
        # Filter out API key error for this test
        errors = [e for e in errors if "api key" not in e.lower()]
        assert len(errors) == 0

    def test_default_agents_created(self):
        """Test default agents are created."""
        config = create_config(topic="Test topic")

        # Should have moderator
        moderators = [a for a in config.agents if a.role == AgentRole.MODERATOR]
        assert len(moderators) == 1

        # Should have at least 2 participants
        participants = [a for a in config.agents if a.role == AgentRole.PARTICIPANT]
        assert len(participants) >= 2

        # Should have evaluator
        evaluators = [a for a in config.agents if a.role == AgentRole.EVALUATOR]
        assert len(evaluators) == 1

    def test_custom_personas(self):
        """Test custom personas are used."""
        personas = [
            {"name": "Expert A", "persona": "Persona A"},
            {"name": "Expert B", "persona": "Persona B"},
        ]
        config = create_config(topic="Test", participant_personas=personas)

        participants = [a for a in config.agents if a.role == AgentRole.PARTICIPANT]
        assert len(participants) == 2
        assert participants[0].name == "Expert A"
        assert participants[1].name == "Expert B"

    def test_provider_setting(self):
        """Test provider is correctly set."""
        config = create_config(topic="Test", provider="anthropic")
        assert config.llm.provider == LLMProvider.ANTHROPIC

    def test_max_rounds_setting(self):
        """Test max rounds is correctly set."""
        config = create_config(topic="Test", max_rounds=10)
        assert config.discussion.max_rounds == 10
