"""Agent modules for multi-agent discussion."""

from .base import BaseAgent, AgentResponse
from .moderator import ModeratorAgent
from .participant import ParticipantAgent
from .research_participant import ResearchParticipantAgent
from .search_mixin import SearchCapabilityMixin, AgentSearchConfig
from .evaluator import EvaluatorAgent, EvaluationResult

__all__ = [
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
]


def create_agent(config, llm_config):
    """
    Factory function to create an agent based on its role.

    Args:
        config: AgentConfig for the agent
        llm_config: LLMConfig for LLM access

    Returns:
        Appropriate agent instance
    """
    from ..config import AgentRole

    role_to_class = {
        AgentRole.MODERATOR: ModeratorAgent,
        AgentRole.PARTICIPANT: ParticipantAgent,
        AgentRole.RESEARCH_PARTICIPANT: ResearchParticipantAgent,
        AgentRole.EVALUATOR: EvaluatorAgent,
    }

    agent_class = role_to_class.get(config.role)
    if agent_class is None:
        raise ValueError(f"Unknown agent role: {config.role}")

    # For research participant, pass search_config
    if config.role == AgentRole.RESEARCH_PARTICIPANT:
        search_cfg = AgentSearchConfig(**(config.search_config or {}))
        return agent_class(config, llm_config, search_config=search_cfg)

    return agent_class(config, llm_config)
