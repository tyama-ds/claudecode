"""
OpenAI API client implementation.
"""

import os
import json
from typing import Optional, List

from .base import BaseLLMClient, Message, MessageRole, LLMResponse


class OpenAIClient(BaseLLMClient):
    """OpenAI API client for GPT models."""

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: Model name (default: gpt-4o-mini)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        super().__init__(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model=model or self.DEFAULT_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

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

        # Make API call
        response = self._client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=api_messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            **{k: v for k, v in kwargs.items()
               if k not in ["model", "temperature", "max_tokens"]}
        )

        # Extract response
        choice = response.choices[0]
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return LLMResponse(
            content=choice.message.content,
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
