"""Agent modules for multi-agent discussion."""

from .base import BaseAgent, AgentResponse
from .moderator import ModeratorAgent
from .participant import ParticipantAgent
from .evaluator import EvaluatorAgent, EvaluationResult

__all__ = [
    "BaseAgent",
    "AgentResponse",
    "ModeratorAgent",
    "ParticipantAgent",
    "EvaluatorAgent",
    "EvaluationResult",
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
        AgentRole.EVALUATOR: EvaluatorAgent,
    }

    agent_class = role_to_class.get(config.role)
    if agent_class is None:
        raise ValueError(f"Unknown agent role: {config.role}")

    return agent_class(config, llm_config)
