"""
API client modules for LLM providers.
"""

from .base import BaseLLMClient, LLMResponse
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .local_client import LocalLLMClient, LocalLLMBackend, KNOWN_MODELS

__all__ = [
    "BaseLLMClient",
    "LLMResponse",
    "OpenAIClient",
    "AnthropicClient",
    "LocalLLMClient",
    "LocalLLMBackend",
    "KNOWN_MODELS",
]


def get_client(
    provider: str,
    api_key: str = None,
    model: str = None,
    http_proxy: str = None,
    https_proxy: str = None,
    verify_ssl: bool = True,
    base_url: str = None,
    backend: str = None,
):
    """
    Factory function to get the appropriate LLM client.

    Args:
        provider: 'openai', 'anthropic', or 'local'
        api_key: API key (optional, uses env var if not provided)
        model: Model name (optional, uses default if not provided)
        http_proxy: HTTP proxy URL
        https_proxy: HTTPS proxy URL
        verify_ssl: Verify SSL certificates
        base_url: Base URL for local LLM server (only for provider='local')
        backend: Backend type for local LLM ('ollama', 'vllm', 'openai_compatible')

    Returns:
        Configured LLM client instance
    """
    if provider.lower() == "openai":
        return OpenAIClient(
            api_key=api_key,
            model=model,
            http_proxy=http_proxy,
            https_proxy=https_proxy,
            verify_ssl=verify_ssl,
        )
    elif provider.lower() == "anthropic":
        return AnthropicClient(
            api_key=api_key,
            model=model,
            http_proxy=http_proxy,
            https_proxy=https_proxy,
            verify_ssl=verify_ssl,
        )
    elif provider.lower() == "local":
        return LocalLLMClient(
            model=model,
            backend=backend,
            base_url=base_url,
            api_key=api_key,
            http_proxy=http_proxy,
            https_proxy=https_proxy,
            verify_ssl=verify_ssl,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'anthropic', or 'local'.")
