"""Configuration module for multi-agent discussion tool."""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class AgentRole(str, Enum):
    """Roles that agents can take in a discussion."""
    MODERATOR = "moderator"
    PARTICIPANT = "participant"
    EVALUATOR = "evaluator"


class DiscussionState(str, Enum):
    """States of a discussion session."""
    INITIALIZED = "initialized"
    OPENING = "opening"
    DISCUSSING = "discussing"
    CONCLUDING = "concluding"
    COMPLETED = "completed"


@dataclass
class LLMConfig:
    """Configuration for LLM API."""
    provider: LLMProvider = LLMProvider.OPENAI
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7
    max_tokens: int = 2048

    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    def get_api_key(self) -> Optional[str]:
        """Get the API key for the current provider."""
        if self.provider == LLMProvider.OPENAI:
            return self.openai_api_key
        elif self.provider == LLMProvider.ANTHROPIC:
            return self.anthropic_api_key
        return None

    def get_model(self) -> str:
        """Get the model name for the current provider."""
        if self.provider == LLMProvider.OPENAI:
            return self.openai_model
        elif self.provider == LLMProvider.ANTHROPIC:
            return self.anthropic_model
        return self.openai_model


@dataclass
class AgentConfig:
    """Configuration for an individual agent."""
    name: str
    role: AgentRole
    persona: str = ""
    system_prompt: str = ""
    llm_config: Optional[LLMConfig] = None

    def __post_init__(self):
        """Set default system prompt based on role if not provided."""
        if not self.system_prompt:
            self.system_prompt = self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt based on role."""
        prompts = {
            AgentRole.MODERATOR: (
                "あなたは議論の進行役（モデレーター）です。"
                "参加者の意見を整理し、議論を建設的な方向に導いてください。"
                "中立的な立場を保ち、全員が発言できるよう配慮してください。"
            ),
            AgentRole.PARTICIPANT: (
                "あなたは議論の参加者です。"
                "与えられたペルソナに基づいて、自分の意見や視点を述べてください。"
                "他の参加者の意見にも耳を傾け、建設的な議論に貢献してください。"
            ),
            AgentRole.EVALUATOR: (
                "あなたは議論の評価者です。"
                "議論の内容を客観的に分析し、論点の整理、"
                "各意見の妥当性、そして結論をまとめてください。"
            ),
        }
        return prompts.get(self.role, "")


@dataclass
class DiscussionConfig:
    """Configuration for the discussion session."""
    topic: str = ""
    max_rounds: int = 5
    min_rounds: int = 2
    output_language: str = "ja"
    enable_evaluation: bool = True
    save_session: bool = True
    session_dir: str = "./discussion_sessions"


@dataclass
class Config:
    """Main configuration for multi-agent discussion."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    discussion: DiscussionConfig = field(default_factory=DiscussionConfig)
    agents: List[AgentConfig] = field(default_factory=list)

    def validate(self) -> List[str]:
        """Validate the configuration and return list of errors."""
        errors = []

        # Check API key
        if not self.llm.get_api_key():
            errors.append(
                f"API key for {self.llm.provider.value} is not set. "
                "Please set the environment variable or provide it in config."
            )

        # Check topic
        if not self.discussion.topic:
            errors.append("Discussion topic is not set.")

        # Check agents
        if not self.agents:
            errors.append("No agents configured for the discussion.")

        # Check for moderator
        has_moderator = any(a.role == AgentRole.MODERATOR for a in self.agents)
        if not has_moderator:
            errors.append("At least one moderator agent is required.")

        # Check for participants
        participants = [a for a in self.agents if a.role == AgentRole.PARTICIPANT]
        if len(participants) < 2:
            errors.append("At least two participant agents are required for discussion.")

        return errors


def create_config(
    topic: str,
    provider: str = "openai",
    model: Optional[str] = None,
    participant_personas: Optional[List[dict]] = None,
    max_rounds: int = 5,
    output_language: str = "ja",
) -> Config:
    """
    Create a configuration with sensible defaults.

    Args:
        topic: The topic to discuss
        provider: LLM provider ("openai" or "anthropic")
        model: Model name (uses default if not specified)
        participant_personas: List of dicts with 'name' and 'persona' keys
        max_rounds: Maximum number of discussion rounds
        output_language: Output language code

    Returns:
        Configured Config object
    """
    # Create LLM config
    llm_config = LLMConfig(provider=LLMProvider(provider))
    if model:
        if provider == "openai":
            llm_config.openai_model = model
        else:
            llm_config.anthropic_model = model

    # Create discussion config
    discussion_config = DiscussionConfig(
        topic=topic,
        max_rounds=max_rounds,
        output_language=output_language,
    )

    # Create agents
    agents = []

    # Add moderator
    agents.append(AgentConfig(
        name="モデレーター",
        role=AgentRole.MODERATOR,
    ))

    # Add participants
    if participant_personas:
        for p in participant_personas:
            agents.append(AgentConfig(
                name=p.get("name", "参加者"),
                role=AgentRole.PARTICIPANT,
                persona=p.get("persona", ""),
            ))
    else:
        # Default participants with different perspectives
        default_personas = [
            {"name": "賛成派", "persona": "このトピックに対して肯定的な立場から意見を述べます。"},
            {"name": "反対派", "persona": "このトピックに対して批判的な立場から意見を述べます。"},
            {"name": "中立派", "persona": "このトピックに対して中立的な立場からバランスの取れた意見を述べます。"},
        ]
        for p in default_personas:
            agents.append(AgentConfig(
                name=p["name"],
                role=AgentRole.PARTICIPANT,
                persona=p["persona"],
            ))

    # Add evaluator
    agents.append(AgentConfig(
        name="評価者",
        role=AgentRole.EVALUATOR,
    ))

    return Config(
        llm=llm_config,
        discussion=discussion_config,
        agents=agents,
    )
