"""
Anthropic API client implementation.
"""

import os
import json
from typing import Optional, List

from .base import BaseLLMClient, Message, MessageRole, LLMResponse, TokenUsage, get_token_stats


class AnthropicClient(BaseLLMClient):
    """Anthropic API client for Claude models."""

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

    # Available models mapping
    MODELS = {
        "opus": "claude-3-opus-20240229",
        "opus4": "claude-sonnet-4-20250514",  # Placeholder for future models
        "opus4.5": "claude-opus-4-5-20251101",
        "sonnet": "claude-3-5-sonnet-20241022",
        "sonnet4": "claude-sonnet-4-20250514",
        "haiku": "claude-3-haiku-20240307",
        "haiku4": "claude-3-5-haiku-20241022",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        http_proxy: Optional[str] = None,
        https_proxy: Optional[str] = None,
        verify_ssl: bool = True,
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            model: Model name or alias (default: claude-3-5-sonnet)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            http_proxy: HTTP proxy URL (e.g., "http://proxy:8080")
            https_proxy: HTTPS proxy URL
            verify_ssl: Verify SSL certificates
        """
        # Resolve model alias if provided
        resolved_model = self._resolve_model(model) if model else self.DEFAULT_MODEL

        super().__init__(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._http_proxy = http_proxy
        self._https_proxy = https_proxy
        self._verify_ssl = verify_ssl
        self._initialize_client()

    def _resolve_model(self, model: str) -> str:
        """Resolve model alias to full model name."""
        return self.MODELS.get(model.lower(), model)

    def _initialize_client(self) -> None:
        """Initialize the Anthropic client."""
        try:
            import anthropic

            # Configure HTTP client with proxy if specified
            http_client = None
            if self._http_proxy or self._https_proxy:
                try:
                    import httpx
                    proxy_url = self._https_proxy or self._http_proxy
                    http_client = httpx.Client(
                        proxy=proxy_url,
                        verify=self._verify_ssl,
                    )
                except ImportError:
                    print("Warning: httpx not installed, proxy settings ignored")

            if http_client:
                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    http_client=http_client,
                )
            else:
                self._client = anthropic.Anthropic(api_key=self.api_key)

        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )

    def chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send a chat completion request to Anthropic.

        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the API call

        Returns:
            LLMResponse with the model's response
        """
        # Convert messages to API format
        api_messages = []
        for msg in messages:
            # Anthropic uses 'user' and 'assistant' roles
            role = msg.role.value
            if role == "system":
                # System messages are handled separately in Anthropic
                continue
            api_messages.append({
                "role": role,
                "content": msg.content
            })

        # Build API call parameters
        api_params = {
            "model": kwargs.get("model", self.model),
            "messages": api_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        # Add system prompt if provided
        if system_prompt:
            api_params["system"] = system_prompt

        # Add temperature if not using default
        temp = kwargs.get("temperature", self.temperature)
        if temp != 1.0:  # Anthropic default is 1.0
            api_params["temperature"] = temp

        # Make API call
        response = self._client.messages.create(**api_params)

        # Extract response
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        # Record token usage to global tracker
        token_usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            model=response.model,
        )
        get_token_stats().add_usage(token_usage)

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            finish_reason=response.stop_reason,
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
        # Add JSON instruction to prompt
        json_prompt = prompt + "\n\nRespond with valid JSON only. No other text."

        # Enhance system prompt for JSON
        json_system = (system_prompt or "") + """

IMPORTANT: You must respond with valid JSON only. Do not include any text outside the JSON object.
Start your response with { and end with }."""

        response = self.generate(json_prompt, system_prompt=json_system, **kwargs)

        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
            raise ValueError(f"Failed to parse JSON from response: {content[:500]}")

    def analyze_research_query(self, query: str, context: str = "") -> LLMResponse:
        """
        Analyze a research query with structured output.

        Args:
            query: The research query
            context: Additional context

        Returns:
            LLMResponse with structured analysis
        """
        system_prompt = """You are a research analyst assistant with expertise in creating comprehensive research plans.
Your task is to analyze research queries and create detailed, actionable research plans.
You excel at breaking down complex topics into manageable sections and identifying the best search strategies."""

        prompt = f"""Research Query: {query}

Additional Context: {context if context else "None provided"}

Please analyze this research query and provide a comprehensive research plan. Return your response as a JSON object with this structure:

{{
    "title": "Suggested title for the research report",
    "summary": "Brief summary of the research scope and objectives",
    "table_of_contents": [
        {{
            "section": "1",
            "title": "Section title",
            "description": "Brief description of what this section will cover",
            "subsections": [
                {{"section": "1.1", "title": "Subsection title"}}
            ]
        }}
    ],
    "search_queries": [
        "Prioritized list of specific search queries to execute"
    ],
    "key_terms": [
        "Important terms and concepts to investigate"
    ],
    "suggested_sources": [
        "Types of sources to explore (e.g., academic papers, news, official documents)"
    ],
    "methodology_notes": "Any specific methodological considerations",
    "estimated_complexity": "low/medium/high"
}}

Ensure the table of contents is comprehensive and covers all aspects of the research topic.
Generate at least 10 relevant search queries prioritized by importance."""

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

    def extended_thinking(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        budget_tokens: int = 10000,
        **kwargs
    ) -> LLMResponse:
        """
        Use extended thinking for complex analysis (Claude 3.5+ feature).

        Note: This is a placeholder for future extended thinking API support.

        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt
            budget_tokens: Token budget for thinking
            **kwargs: Additional parameters

        Returns:
            LLMResponse with the model's response
        """
        # For now, use standard generation with explicit thinking instructions
        thinking_prompt = f"""Please think through this carefully step by step before providing your final answer.

{prompt}

First, outline your thinking process, then provide your final answer."""

        return self.generate(thinking_prompt, system_prompt=system_prompt, **kwargs)
