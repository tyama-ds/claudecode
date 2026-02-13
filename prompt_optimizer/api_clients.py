"""
Lightweight API clients for OpenAI and Anthropic.

Kept simple and self-contained for the prompt optimizer module.
"""

import os
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    usage: dict


class OpenAIClient:
    """Minimal OpenAI API client."""

    MODELS = {
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4-turbo": "gpt-4-turbo",
        "o1": "o1",
        "o1-mini": "o1-mini",
        "o3-mini": "o3-mini",
    }

    # Models that do not support temperature parameter
    NO_TEMPERATURE_PREFIXES = ("o1", "o3", "o4")

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _supports_temperature(self, model: str) -> bool:
        model_lower = model.lower()
        for prefix in self.NO_TEMPERATURE_PREFIXES:
            if model_lower == prefix or model_lower.startswith(prefix + "-"):
                return False
        return True

    def chat(
        self,
        user_message: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        params: dict = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if self._supports_temperature(self.model):
            params["temperature"] = temperature

        response = client.chat.completions.create(**params)

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )


class AnthropicClient:
    """Minimal Anthropic API client."""

    MODELS = {
        "claude-sonnet-4-20250514": "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022": "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022",
        "claude-3-haiku-20240307": "claude-3-haiku-20240307",
    }

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def chat(
        self,
        user_message: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        client = self._get_client()

        params: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_message}],
            "max_tokens": max_tokens,
        }
        if system_prompt:
            params["system"] = system_prompt
        if temperature != 1.0:
            params["temperature"] = temperature

        response = client.messages.create(**params)

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        )


def get_available_models() -> dict:
    """Return available models grouped by provider."""
    return {
        "openai": [
            {"id": "gpt-4o", "name": "GPT-4o"},
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
            {"id": "o1", "name": "o1"},
            {"id": "o1-mini", "name": "o1 Mini"},
            {"id": "o3-mini", "name": "o3 Mini"},
        ],
        "anthropic": [
            {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4"},
            {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet"},
            {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku"},
            {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku"},
        ],
    }


def create_client(provider: str, model: str, api_key: str) -> OpenAIClient | AnthropicClient:
    """Factory function to create the appropriate client."""
    if provider == "openai":
        return OpenAIClient(api_key=api_key, model=model)
    elif provider == "anthropic":
        return AnthropicClient(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")
