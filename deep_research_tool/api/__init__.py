"""
API client modules for LLM providers.
"""

from .base import BaseLLMClient, LLMResponse
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient

__all__ = [
    "BaseLLMClient",
    "LLMResponse",
    "OpenAIClient",
    "AnthropicClient",
]


def get_client(provider: str, api_key: str = None, model: str = None):
    """
    Factory function to get the appropriate LLM client.

    Args:
        provider: 'openai' or 'anthropic'
        api_key: API key (optional, uses env var if not provided)
        model: Model name (optional, uses default if not provided)

    Returns:
        Configured LLM client instance
    """
    if provider.lower() == "openai":
        return OpenAIClient(api_key=api_key, model=model)
    elif provider.lower() == "anthropic":
        return AnthropicClient(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
