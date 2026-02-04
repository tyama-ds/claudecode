"""Tests for module imports."""

import pytest


class TestImports:
    """Tests for verifying all imports work correctly."""

    def test_import_main_module(self):
        """Test importing main module."""
        import multi_agent_discussion
        assert multi_agent_discussion.__version__ == "0.1.0"

    def test_import_config(self):
        """Test importing config classes."""
        from multi_agent_discussion import (
            Config,
            LLMConfig,
            AgentConfig,
            DiscussionConfig,
            LLMProvider,
            AgentRole,
            DiscussionState,
            create_config,
        )
        assert Config is not None
        assert LLMConfig is not None
        assert AgentConfig is not None

    def test_import_agents(self):
        """Test importing agent classes."""
        from multi_agent_discussion import (
            BaseAgent,
            AgentResponse,
            ModeratorAgent,
            ParticipantAgent,
            EvaluatorAgent,
            EvaluationResult,
            create_agent,
        )
        assert BaseAgent is not None
        assert ModeratorAgent is not None
        assert ParticipantAgent is not None
        assert EvaluatorAgent is not None

    def test_import_conversation(self):
        """Test importing conversation classes."""
        from multi_agent_discussion import (
            Message,
            MessageType,
            Turn,
            Round,
            DiscussionSession,
        )
        assert Message is not None
        assert MessageType is not None
        assert DiscussionSession is not None

    def test_import_main(self):
        """Test importing main orchestrator."""
        from multi_agent_discussion import (
            MultiAgentDiscussion,
            run_discussion,
        )
        assert MultiAgentDiscussion is not None
        assert run_discussion is not None

    def test_import_agents_submodule(self):
        """Test importing from agents submodule."""
        from multi_agent_discussion.agents import (
            BaseAgent,
            AgentResponse,
            ModeratorAgent,
            ParticipantAgent,
            EvaluatorAgent,
            create_agent,
        )
        assert create_agent is not None

    def test_import_conversation_submodule(self):
        """Test importing from conversation submodule."""
        from multi_agent_discussion.conversation import (
            Message,
            MessageType,
            Turn,
            Round,
            DiscussionSession,
        )
        assert DiscussionSession is not None

    def test_create_agent_factory(self):
        """Test agent factory function."""
        from multi_agent_discussion import create_agent, AgentConfig, AgentRole, LLMConfig

        config = AgentConfig(name="Test", role=AgentRole.MODERATOR)
        llm_config = LLMConfig()

        agent = create_agent(config, llm_config)
        assert agent.name == "Test"
        assert agent.role == AgentRole.MODERATOR

    def test_create_agent_all_roles(self):
        """Test creating agents for all roles."""
        from multi_agent_discussion import (
            create_agent,
            AgentConfig,
            AgentRole,
            LLMConfig,
            ModeratorAgent,
            ParticipantAgent,
            EvaluatorAgent,
        )

        llm_config = LLMConfig()

        # Test each role
        roles_and_classes = [
            (AgentRole.MODERATOR, ModeratorAgent),
            (AgentRole.PARTICIPANT, ParticipantAgent),
            (AgentRole.EVALUATOR, EvaluatorAgent),
        ]

        for role, expected_class in roles_and_classes:
            config = AgentConfig(name="Test", role=role)
            agent = create_agent(config, llm_config)
            assert isinstance(agent, expected_class)
