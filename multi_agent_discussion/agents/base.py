"""Base agent class for multi-agent discussion."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..conversation.message import Message
    from ..config import AgentConfig, LLMConfig


@dataclass
class AgentResponse:
    """Response from an agent."""
    agent_name: str
    content: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentResponse":
        """Create from dictionary."""
        return cls(
            agent_name=data["agent_name"],
            content=data["content"],
            metadata=data.get("metadata", {}),
        )


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, config: "AgentConfig", llm_config: "LLMConfig"):
        """
        Initialize the agent.

        Args:
            config: Agent configuration
            llm_config: LLM configuration
        """
        self.config = config
        self.llm_config = llm_config
        self.name = config.name
        self.role = config.role
        self._client = None

    @property
    def client(self):
        """Lazy initialization of LLM client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self):
        """Create LLM client based on configuration."""
        from ..config import LLMProvider

        provider = self.llm_config.provider
        api_key = self.llm_config.get_api_key()

        if provider == LLMProvider.OPENAI:
            try:
                from openai import OpenAI
                return OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package is required for OpenAI provider")

        elif provider == LLMProvider.ANTHROPIC:
            try:
                from anthropic import Anthropic
                return Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic package is required for Anthropic provider")

        elif provider == LLMProvider.GOOGLE:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                return genai
            except ImportError:
                raise ImportError("google-generativeai package is required for Google provider")

        elif provider == LLMProvider.OLLAMA:
            try:
                from openai import OpenAI
                # Ollama provides OpenAI-compatible API
                return OpenAI(
                    base_url=f"{self.llm_config.ollama_base_url}/v1",
                    api_key="ollama",  # Ollama doesn't require API key
                )
            except ImportError:
                raise ImportError("openai package is required for Ollama provider")

        elif provider == LLMProvider.XAI:
            try:
                from openai import OpenAI
                # xAI provides OpenAI-compatible API
                return OpenAI(
                    base_url="https://api.x.ai/v1",
                    api_key=api_key,
                )
            except ImportError:
                raise ImportError("openai package is required for xAI provider")

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _call_llm(self, messages: List[dict], system_prompt: str = None) -> str:
        """
        Call the LLM with the given messages.

        Args:
            messages: List of message dictionaries
            system_prompt: System prompt to use

        Returns:
            Response content from the LLM
        """
        from ..config import LLMProvider

        provider = self.llm_config.provider
        model = self.llm_config.get_model()

        if provider == LLMProvider.OPENAI:
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)

            response = self.client.chat.completions.create(
                model=model,
                messages=full_messages,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )
            return response.choices[0].message.content

        elif provider == LLMProvider.ANTHROPIC:
            response = self.client.messages.create(
                model=model,
                max_tokens=self.llm_config.max_tokens,
                system=system_prompt or "",
                messages=messages,
            )
            return response.content[0].text

        elif provider == LLMProvider.GOOGLE:
            # Google Gemini API
            genai_model = self.client.GenerativeModel(
                model_name=model,
                system_instruction=system_prompt,
            )
            # Convert messages to Gemini format
            chat_history = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                chat_history.append({"role": role, "parts": [msg["content"]]})

            chat = genai_model.start_chat(history=chat_history[:-1] if chat_history else [])
            last_message = chat_history[-1]["parts"][0] if chat_history else ""
            response = chat.send_message(last_message)
            return response.text

        elif provider in (LLMProvider.OLLAMA, LLMProvider.XAI):
            # Both use OpenAI-compatible API
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)

            response = self.client.chat.completions.create(
                model=model,
                messages=full_messages,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        base_prompt = self.config.system_prompt
        if self.config.persona:
            base_prompt += f"\n\nあなたのペルソナ: {self.config.persona}"
        return base_prompt

    @abstractmethod
    def generate_response(
        self,
        topic: str,
        conversation_history: List["Message"],
        context: Optional[str] = None,
    ) -> AgentResponse:
        """
        Generate a response based on the conversation.

        Args:
            topic: The discussion topic
            conversation_history: List of previous messages
            context: Additional context for the response

        Returns:
            AgentResponse with the generated content
        """
        pass

    def _format_history(self, history: List["Message"]) -> List[dict]:
        """Format conversation history for LLM input."""
        messages = []
        for msg in history:
            role = "assistant" if msg.agent_name == self.name else "user"
            messages.append({
                "role": role,
                "content": f"[{msg.agent_name}]: {msg.content}",
            })
        return messages

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, role={self.role})"
