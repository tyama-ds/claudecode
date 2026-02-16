"""
OpenAI API client implementation.
"""

import os
import json
from typing import Optional, List

from .base import BaseLLMClient, Message, MessageRole, LLMResponse, TokenUsage, get_token_stats


class OpenAIClient(BaseLLMClient):
    """OpenAI API client for GPT models."""

    DEFAULT_MODEL = "gpt-5-mini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        http_proxy: Optional[str] = None,
        https_proxy: Optional[str] = None,
        verify_ssl: bool = True,
        timeout: int = 120,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: Model name (default: gpt-4o-mini)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            http_proxy: HTTP proxy URL (e.g., "http://proxy:8080")
            https_proxy: HTTPS proxy URL
            verify_ssl: Verify SSL certificates
            timeout: Request timeout in seconds (default: 120)
        """
        super().__init__(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model=model or self.DEFAULT_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._http_proxy = http_proxy
        self._https_proxy = https_proxy
        self._verify_ssl = verify_ssl
        self._timeout = timeout
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI

            # Configure HTTP client with proxy if specified
            http_client = None
            if self._http_proxy or self._https_proxy:
                try:
                    import httpx
                    proxy_url = self._https_proxy or self._http_proxy
                    http_client = httpx.Client(
                        proxy=proxy_url,
                        verify=self._verify_ssl,
                        timeout=httpx.Timeout(self._timeout),
                    )
                except ImportError:
                    print("Warning: httpx not installed, proxy settings ignored")

            if http_client:
                self._client = OpenAI(api_key=self.api_key, http_client=http_client)
            else:
                self._client = OpenAI(api_key=self.api_key, timeout=self._timeout)

        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

    def _requires_max_completion_tokens(self, model: str) -> bool:
        """Check if model requires max_completion_tokens instead of max_tokens.

        As of late 2024, most OpenAI models prefer max_completion_tokens.
        Only very old models (gpt-3.5-turbo, gpt-4 without suffix) use max_tokens.
        """
        model_lower = model.lower()

        # Old models that still use max_tokens
        legacy_models = ['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4-0314', 'gpt-4-0613']

        # Check if it's a legacy model
        for legacy in legacy_models:
            if model_lower == legacy or model_lower.startswith(legacy + '-'):
                return False

        # All other models (gpt-4o, o1, o3, etc.) use max_completion_tokens
        return True

    def _supports_temperature(self, model: str) -> bool:
        """Check if model supports custom temperature parameter.

        Some models (o1, o3, gpt-5 series) only support temperature=1 (default).
        This includes reasoning models and newer generation models.
        """
        model_lower = model.lower()

        # Model series prefixes that do NOT support custom temperature
        # These are typically reasoning models or newer generation models
        no_temperature_prefixes = [
            'o1',      # o1, o1-mini, o1-preview, o1-pro, o1.5, etc.
            'o3',      # o3, o3-mini, o3.1, etc.
            'o4',      # Future o4 series (anticipated)
            'gpt-5',   # gpt-5, gpt-5-mini, gpt-5.2, gpt-5.2-turbo, etc.
            'gpt-6',   # Future gpt-6 series (anticipated)
        ]

        for prefix in no_temperature_prefixes:
            # Match: exact, prefix-, prefix., prefixmini (e.g., gpt-5, gpt-5-mini, gpt-5.2, gpt-5mini)
            if (model_lower == prefix or
                model_lower.startswith(prefix + '-') or
                model_lower.startswith(prefix + '.') or
                model_lower.startswith(prefix + 'mini')):
                return False

        return True

    def chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send a chat completion request to OpenAI.

        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the API call

        Returns:
            LLMResponse with the model's response
        """
        api_messages = []

        # Add system prompt if provided
        if system_prompt:
            api_messages.append({
                "role": "system",
                "content": system_prompt
            })

        # Convert messages to API format
        for msg in messages:
            api_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })

        # Determine the correct token limit parameter based on model
        model = kwargs.get("model", self.model)
        token_limit = kwargs.get("max_tokens", self.max_tokens)

        # Build API call parameters
        api_params = {
            "model": model,
            "messages": api_messages,
        }

        # Use correct parameter name based on model
        if self._requires_max_completion_tokens(model):
            api_params["max_completion_tokens"] = token_limit
        else:
            api_params["max_tokens"] = token_limit

        # Only add temperature if the model supports it (o1, o3, gpt-5 series don't)
        if self._supports_temperature(model):
            api_params["temperature"] = kwargs.get("temperature", self.temperature)

        # Add any additional kwargs (excluding already handled ones)
        excluded_keys = {"model", "temperature", "max_tokens", "max_completion_tokens"}
        for k, v in kwargs.items():
            if k not in excluded_keys:
                api_params[k] = v

        # Debug: Print API params to verify correct parameter is used
        supports_temp = self._supports_temperature(model)
        print(f"[DEBUG OpenAI] Model: {model}, max_completion_tokens: {token_limit}, temperature_supported: {supports_temp}")

        # Make API call
        response = self._client.chat.completions.create(**api_params)

        # Extract response - handle missing/empty choices
        if not response.choices:
            print(f"[WARNING OpenAI] API returned no choices. Model: {model}")
            return LLMResponse(
                content="",
                model=response.model if response else model,
                usage={},
                finish_reason="error",
                raw_response=response,
            )

        choice = response.choices[0]

        # Handle None content - OpenAI can return content=None in several cases:
        # - Content filter triggered (finish_reason="content_filter")
        # - Reasoning models (o1/o3) with certain configs
        # - Token limit exhausted before any output (finish_reason="length")
        content = choice.message.content
        if content is None:
            # Check for refusal (newer OpenAI models)
            refusal = getattr(choice.message, "refusal", None)
            if refusal:
                print(f"[WARNING OpenAI] Model refused request: {refusal}")
                content = ""
            else:
                print(f"[WARNING OpenAI] Response content is None. "
                      f"finish_reason={choice.finish_reason}, model={response.model}")
                content = ""

        if choice.finish_reason == "length":
            print(f"[WARNING OpenAI] Response truncated (finish_reason=length). "
                  f"Output may be incomplete. Consider increasing max_tokens.")
        elif choice.finish_reason == "content_filter":
            print(f"[WARNING OpenAI] Content was filtered by OpenAI safety system.")

        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        # Record token usage to global tracker
        if response.usage:
            token_usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                model=response.model,
            )
            get_token_stats().add_usage(token_usage)

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            finish_reason=choice.finish_reason,
            raw_response=response,
        )

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

    def generate_with_json_output(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> dict:
        """
        Generate JSON output from prompt.

        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Parsed JSON response
        """
        # Add JSON instruction to system prompt
        json_system = (system_prompt or "") + "\n\nAlways respond with valid JSON only."

        response = self.generate(
            prompt,
            system_prompt=json_system,
            response_format={"type": "json_object"},
            **kwargs
        )

        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
            raise ValueError(f"Failed to parse JSON from response: {content}")

    def analyze_research_query(self, query: str, context: str = "") -> LLMResponse:
        """
        Analyze a research query using JSON mode.

        Args:
            query: The research query
            context: Additional context

        Returns:
            LLMResponse with structured analysis
        """
        system_prompt = """You are a research analyst assistant. Analyze the research query and provide a structured research plan.
Always respond in valid JSON format."""

        prompt = f"""Research Query: {query}

Additional Context: {context if context else "None provided"}

Analyze this research query and provide a JSON response with:
{{
    "title": "Suggested report title",
    "summary": "Brief summary of the research scope",
    "table_of_contents": [
        {{"section": "1", "title": "Section title", "description": "Brief description", "subsections": []}}
    ],
    "search_queries": ["Prioritized list of search queries"],
    "key_terms": ["Important terms to investigate"],
    "suggested_sources": ["Types of sources to explore"],
    "estimated_complexity": "low/medium/high"
}}"""

        try:
            result = self.generate_with_json_output(prompt, system_prompt=system_prompt)
            return LLMResponse(
                content=json.dumps(result, ensure_ascii=False, indent=2),
                model=self.model,
                usage={},
            )
        except Exception:
            # Fall back to regular generation
            return self.generate(prompt, system_prompt=system_prompt)
