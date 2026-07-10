"""
Local LLM Client - Support for local LLM servers (Ollama, vLLM, etc.)

Supports:
- Ollama (for Llama models)
- vLLM (for gpt-oss-20b, gpt-oss-120b)
- Any OpenAI-compatible API endpoint
"""

import os
import json
from typing import Optional, List, Dict, Any

from .base import (
    BaseLLMClient,
    LLMResponse,
    Message,
    MessageRole,
    TokenUsage,
    get_token_stats,
)


class LocalLLMBackend:
    """Supported local LLM backends."""
    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENAI_COMPATIBLE = "openai_compatible"


# Default models for each backend
DEFAULT_MODELS = {
    LocalLLMBackend.OLLAMA: "llama3.1:8b",
    LocalLLMBackend.VLLM: "gpt-oss-20b",
    LocalLLMBackend.OPENAI_COMPATIBLE: "gpt-oss-20b",
}

# Default endpoints for each backend
DEFAULT_ENDPOINTS = {
    LocalLLMBackend.OLLAMA: "http://localhost:11434",
    LocalLLMBackend.VLLM: "http://localhost:8000",
    LocalLLMBackend.OPENAI_COMPATIBLE: "http://localhost:8000",
}

# Known model mappings
KNOWN_MODELS = {
    # Llama models (Ollama)
    "llama": "llama3.1:8b",
    "llama2": "llama2:7b",
    "llama3": "llama3.1:8b",
    "llama3.1": "llama3.1:8b",
    "llama3.1:8b": "llama3.1:8b",
    "llama3.1:70b": "llama3.1:70b",
    "llama3.2": "llama3.2:3b",
    "codellama": "codellama:7b",

    # GPT-OSS models (vLLM or OpenAI-compatible)
    "gpt-oss-20b": "gpt-oss-20b",
    "gpt-oss-120b": "gpt-oss-120b",

    # Other common local models
    "mistral": "mistral:7b",
    "mixtral": "mixtral:8x7b",
    "phi": "phi3:mini",
    "qwen": "qwen2:7b",
    "gemma": "gemma2:9b",
}


class LocalLLMClient(BaseLLMClient):
    """
    Client for local LLM servers.

    Supports Ollama, vLLM, and any OpenAI-compatible API endpoint.

    Example usage:
        # Using Ollama with Llama
        client = LocalLLMClient(
            model="llama3.1:8b",
            backend="ollama",
            base_url="http://localhost:11434",
        )

        # Using vLLM with gpt-oss-20b
        client = LocalLLMClient(
            model="gpt-oss-20b",
            backend="vllm",
            base_url="http://localhost:8000",
        )
    """

    def __init__(
        self,
        model: str = None,
        backend: str = None,
        base_url: str = None,
        api_key: str = None,  # Some local servers may require auth
        temperature: float = 0.7,
        max_tokens: int = 8192,
        max_tokens_limit: int = 200_000,
        timeout: int = 120,
        http_proxy: str = None,
        https_proxy: str = None,
        verify_ssl: bool = True,
    ):
        """
        Initialize LocalLLMClient.

        Args:
            model: Model name (e.g., "llama3.1:8b", "gpt-oss-20b")
            backend: Backend type ("ollama", "vllm", "openai_compatible")
            base_url: Base URL of the local LLM server
            api_key: Optional API key for authentication
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            max_tokens_limit: Upper bound for auto-retry on truncation (default: 200000)
            timeout: Request timeout in seconds
            http_proxy: HTTP proxy URL
            https_proxy: HTTPS proxy URL
            verify_ssl: Verify SSL certificates
        """
        # Detect backend from model name if not specified
        if backend is None:
            backend = self._detect_backend(model)

        self.backend = backend

        # Resolve model name
        if model:
            self.model = KNOWN_MODELS.get(model.lower(), model)
        else:
            self.model = DEFAULT_MODELS.get(backend, "llama3.1:8b")

        # Set base URL
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            self.base_url = os.getenv(
                "LOCAL_LLM_BASE_URL",
                DEFAULT_ENDPOINTS.get(backend, "http://localhost:8000")
            )

        self.api_key = api_key or os.getenv("LOCAL_LLM_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tokens_limit = max_tokens_limit
        self.timeout = timeout
        self._http_proxy = http_proxy
        self._https_proxy = https_proxy
        self._verify_ssl = verify_ssl

        self._client = None
        self._initialize_client()

    def _detect_backend(self, model: str) -> str:
        """Detect backend from model name."""
        if model is None:
            return LocalLLMBackend.OLLAMA

        model_lower = model.lower()

        # Llama models typically use Ollama
        if any(x in model_lower for x in ["llama", "mistral", "mixtral", "phi", "qwen", "gemma", "codellama"]):
            return LocalLLMBackend.OLLAMA

        # GPT-OSS models typically use vLLM
        if "gpt-oss" in model_lower:
            return LocalLLMBackend.VLLM

        # Default to OpenAI-compatible
        return LocalLLMBackend.OPENAI_COMPATIBLE

    def _initialize_client(self) -> None:
        """Initialize the HTTP client."""
        try:
            import requests
            self._session = requests.Session()

            # Set up proxies if provided
            if self._http_proxy or self._https_proxy:
                self._session.proxies = {}
                if self._http_proxy:
                    self._session.proxies["http"] = self._http_proxy
                if self._https_proxy:
                    self._session.proxies["https"] = self._https_proxy

            self._session.verify = self._verify_ssl

            # Set up headers
            self._session.headers.update({
                "Content-Type": "application/json",
            })
            if self.api_key:
                self._session.headers["Authorization"] = f"Bearer {self.api_key}"

        except ImportError:
            raise ImportError("requests library is required for LocalLLMClient")

    def _base_has_version_segment(self) -> bool:
        """Whether base_url already ends with a REST version segment (/v1, /v2...)."""
        import re
        return bool(re.search(r"/v\d+$", self.base_url))

    def _use_openai_routing(self) -> bool:
        """Whether to use the OpenAI-compatible /chat/completions endpoint.

        vLLM / openai_compatible always use it. Ollama normally uses its
        native /api/* endpoints, but when the user points base_url at a
        versioned REST path (e.g. http://host/v1) they mean an
        OpenAI-compatible server (this also matches Ollama's own OpenAI
        compatibility layer), so route accordingly.
        """
        if self.backend != LocalLLMBackend.OLLAMA:
            return True
        return self._base_has_version_segment()

    def _get_api_url(self, endpoint: str) -> str:
        """Get the full API URL based on backend type.

        Handles a base_url that already includes the version segment so we
        never emit a doubled path like `/v1/v1/chat/completions`.
        """
        import re
        base = self.base_url.rstrip("/")

        if self._use_openai_routing():
            # OpenAI-compatible: <base>/v1/chat/completions, but if the base
            # already ends with /v1 (or /vN) just append chat/completions so
            # http://host/v1 -> http://host/v1/chat/completions (not /v1/v1/...)
            if self._base_has_version_segment():
                return f"{base}/chat/completions"
            return f"{base}/v1/chat/completions"

        # Ollama native API uses /api/chat and /api/generate. Strip an
        # accidental trailing /vN so http://host/v1 still resolves correctly.
        base = re.sub(r"/v\d+$", "", base)
        if endpoint == "chat":
            return f"{base}/api/chat"
        return f"{base}/api/generate"

    def chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            LLMResponse with the model's response
        """
        # Build message list
        api_messages = []

        if system_prompt:
            api_messages.append({
                "role": "system",
                "content": system_prompt,
            })

        for msg in messages:
            api_messages.append({
                "role": msg.role.value if isinstance(msg.role, MessageRole) else msg.role,
                "content": msg.content,
            })

        return self._send_chat_request(api_messages, **kwargs)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from a single prompt.

        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            LLMResponse with the model's response
        """
        messages = [Message(role=MessageRole.USER, content=prompt)]
        return self.chat(messages, system_prompt=system_prompt, **kwargs)

    def _send_chat_request(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Send chat request to the local LLM server."""

        # Route by the same rule used for URL building so the payload format
        # (Ollama-native vs OpenAI-compatible) always matches the endpoint.
        if self._use_openai_routing():
            return self._send_openai_compatible_request(messages, **kwargs)
        else:
            return self._send_ollama_request(messages, **kwargs)

    def _send_ollama_request(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Send request to Ollama server."""
        url = self._get_api_url("chat")

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }

        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Extract response
            content = data.get("message", {}).get("content", "")

            # Ollama provides eval_count and prompt_eval_count
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            total_tokens = prompt_tokens + completion_tokens

            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

            # Record token usage
            token_usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=self.model,
            )
            get_token_stats().add_usage(token_usage)

            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                finish_reason=data.get("done_reason", "stop"),
                raw_response=data,
            )

        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def _send_openai_compatible_request(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Send request to OpenAI-compatible server (vLLM, etc.)."""
        url = self._get_api_url("chat")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False,
        }

        # Add optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]

        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Extract response (OpenAI format)
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")

            # Extract usage
            usage_data = data.get("usage", {})
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            completion_tokens = usage_data.get("completion_tokens", 0)
            total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens)

            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

            # Record token usage
            token_usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=data.get("model", self.model),
            )
            get_token_stats().add_usage(token_usage)

            return LLMResponse(
                content=content,
                model=data.get("model", self.model),
                usage=usage,
                finish_reason=choice.get("finish_reason", "stop"),
                raw_response=data,
            )

        except Exception as e:
            raise RuntimeError(f"Local LLM API error: {e}")

    def _models_url(self) -> str:
        """Build the model-list URL, mirroring _get_api_url version handling."""
        import re
        base = self.base_url.rstrip("/")
        if self._use_openai_routing():
            if self._base_has_version_segment():
                return f"{base}/models"
            return f"{base}/v1/models"
        base = re.sub(r"/v\d+$", "", base)
        return f"{base}/api/tags"

    def list_models(self) -> List[str]:
        """
        List available models on the local server.

        Returns:
            List of model names
        """
        url = self._models_url()
        openai_style = self._use_openai_routing()

        try:
            response = self._session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if openai_style:
                models = [m.get("id", "") for m in data.get("data", [])]
            else:
                models = [m.get("name", "") for m in data.get("models", [])]

            return models

        except Exception as e:
            print(f"Failed to list models: {e}")
            return []

    def is_available(self) -> bool:
        """
        Check if the local LLM server is available.

        Returns:
            True if server is reachable
        """
        try:
            response = self._session.get(self._models_url(), timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def generate_with_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate JSON response (for structured output).

        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            LLMResponse with JSON content
        """
        # Add JSON instruction to system prompt
        json_system = (system_prompt or "") + "\n\nRespond ONLY with valid JSON, no other text."

        response = self.generate(prompt, system_prompt=json_system, **kwargs)

        # Try to extract JSON from response
        content = response.content.strip()

        # Handle markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (code block markers)
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        # Validate JSON
        try:
            json.loads(content)
            response.content = content
        except json.JSONDecodeError:
            # Try to find JSON in the response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                response.content = content[start:end]

        return response
