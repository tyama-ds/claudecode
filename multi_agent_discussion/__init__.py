"""
Multi-Agent Discussion Tool

A framework for conducting AI-powered group discussions with multiple agents.
"""

from .config import (
    Config,
    LLMConfig,
    AgentConfig,
    DiscussionConfig,
    LLMProvider,
    AgentRole,
    DiscussionState,
    create_config,
)
from .main import MultiAgentDiscussion, run_discussion
from .agents import (
    BaseAgent,
    AgentResponse,
    ModeratorAgent,
    ParticipantAgent,
    ResearchParticipantAgent,
    SearchCapabilityMixin,
    AgentSearchConfig,
    EvaluatorAgent,
    EvaluationResult,
    create_agent,
)
from .conversation import (
    Message,
    MessageType,
    Turn,
    Round,
    DiscussionSession,
)

__version__ = "0.1.0"
__all__ = [
    # Config
    "Config",
    "LLMConfig",
    "AgentConfig",
    "DiscussionConfig",
    "LLMProvider",
    "AgentRole",
    "DiscussionState",
    "create_config",
    # Main
    "MultiAgentDiscussion",
    "run_discussion",
    # Agents
    "BaseAgent",
    "AgentResponse",
    "ModeratorAgent",
    "ParticipantAgent",
    "ResearchParticipantAgent",
    "SearchCapabilityMixin",
    "AgentSearchConfig",
    "EvaluatorAgent",
    "EvaluationResult",
    "create_agent",
    # Conversation
    "Message",
    "MessageType",
    "Turn",
    "Round",
    "DiscussionSession",
    # Orchestrator (available via multi_agent_discussion.orchestrator)
    "orchestrator",
]
