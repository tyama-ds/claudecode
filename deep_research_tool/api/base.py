"""
Base class for LLM API clients.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from datetime import datetime


class MessageRole(str, Enum):
    """Message roles for chat completion."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in a conversation."""
    role: MessageRole
    content: str


@dataclass
class TokenUsage:
    """Token usage for a single API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "timestamp": self.timestamp,
        }


@dataclass
class TokenUsageStats:
    """Aggregated token usage statistics."""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_calls: int = 0
    calls_by_model: Dict[str, int] = field(default_factory=dict)
    tokens_by_model: Dict[str, int] = field(default_factory=dict)
    history: List[TokenUsage] = field(default_factory=list)

    def add_usage(self, usage: TokenUsage) -> None:
        """Add a usage record to the statistics."""
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.total_calls += 1

        # Track by model
        model = usage.model or "unknown"
        self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1
        self.tokens_by_model[model] = self.tokens_by_model.get(model, 0) + usage.total_tokens

        # Keep history
        self.history.append(usage)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
            "calls_by_model": self.calls_by_model,
            "tokens_by_model": self.tokens_by_model,
        }

    def get_summary(self, language: str = "en") -> str:
        """Get a human-readable summary of token usage."""
        if language == "ja":
            lines = [
                "=== トークン使用量 ===",
                f"総トークン数: {self.total_tokens:,}",
                f"  - 入力: {self.total_prompt_tokens:,}",
                f"  - 出力: {self.total_completion_tokens:,}",
                f"API呼び出し回数: {self.total_calls}",
            ]
            if self.tokens_by_model:
                lines.append("モデル別:")
                for model, tokens in sorted(self.tokens_by_model.items(),
                                            key=lambda x: x[1], reverse=True):
                    calls = self.calls_by_model.get(model, 0)
                    lines.append(f"  {model}: {tokens:,} tokens ({calls} calls)")
        else:
            lines = [
                "=== Token Usage ===",
                f"Total tokens: {self.total_tokens:,}",
                f"  - Input: {self.total_prompt_tokens:,}",
                f"  - Output: {self.total_completion_tokens:,}",
                f"API calls: {self.total_calls}",
            ]
            if self.tokens_by_model:
                lines.append("By model:")
                for model, tokens in sorted(self.tokens_by_model.items(),
                                            key=lambda x: x[1], reverse=True):
                    calls = self.calls_by_model.get(model, 0)
                    lines.append(f"  {model}: {tokens:,} tokens ({calls} calls)")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_calls = 0
        self.calls_by_model.clear()
        self.tokens_by_model.clear()
        self.history.clear()


# Global token usage tracker
_global_token_stats = TokenUsageStats()


def get_token_stats() -> TokenUsageStats:
    """Get the global token usage statistics."""
    return _global_token_stats


def reset_token_stats() -> None:
    """Reset the global token usage statistics."""
    _global_token_stats.reset()


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.usage.get("total_tokens", 0)


class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        max_tokens_limit: int = 200_000,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for the provider
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response (default: 8192)
            max_tokens_limit: Upper bound for auto-retry on truncation (default: 200000)
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tokens_limit = max_tokens_limit
        self._client = None

    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the underlying API client."""
        pass

    @abstractmethod
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
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with the model's response
        """
        pass

    @abstractmethod
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
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with the model's response
        """
        pass

    def analyze_research_query(self, query: str, context: str = "") -> LLMResponse:
        """
        Analyze a research query and generate structured research plan.

        Args:
            query: The research query/topic
            context: Additional context or requirements

        Returns:
            LLMResponse with analysis and plan
        """
        system_prompt = """You are a research analyst assistant. Your task is to analyze research queries
and create structured research plans. For each query, you should:
1. Identify the main research topic and subtopics
2. Generate a table of contents for the final report
3. Create a list of search queries to gather information
4. Identify key terms and concepts to investigate

Respond in a structured JSON format."""

        prompt = f"""Research Query: {query}

Additional Context: {context if context else "None provided"}

Please analyze this research query and provide:
1. A structured table of contents for the research report
2. A list of search queries (in order of priority)
3. Key terms and concepts to investigate
4. Potential sources of information to explore

Respond in JSON format with the following structure:
{{
    "title": "Report title",
    "table_of_contents": [
        {{"section": "1", "title": "Section title", "subsections": []}},
        ...
    ],
    "search_queries": ["query1", "query2", ...],
    "key_terms": ["term1", "term2", ...],
    "suggested_sources": ["source1", "source2", ...]
}}"""

        return self.generate(prompt, system_prompt=system_prompt)

    def synthesize_information(
        self,
        section_title: str,
        gathered_info: List[Dict[str, Any]],
        requirements: str = ""
    ) -> LLMResponse:
        """
        Synthesize gathered information into coherent content.

        Args:
            section_title: Title of the section being written
            gathered_info: List of information items with sources
            requirements: Original research requirements

        Returns:
            LLMResponse with synthesized content
        """
        system_prompt = """You are a research writer. Your task is to synthesize information from multiple sources
into coherent, well-structured content. Always:
1. Clearly distinguish between factual information (with citations) and analysis/interpretation
2. Use [SOURCE: X] notation for direct citations
3. Use [ANALYSIS] notation for your interpretations or inferences
4. Maintain academic writing standards"""

        info_text = "\n\n".join([
            f"Source {i+1} ({item.get('url', 'Unknown source')}):\n{item.get('content', '')}"
            for i, item in enumerate(gathered_info)
        ])

        prompt = f"""Section Title: {section_title}

Research Requirements: {requirements if requirements else "General research"}

Gathered Information:
{info_text}

Please synthesize this information into a coherent section for the research report.
- Clearly cite sources using [SOURCE: N] notation
- Mark your own analysis/interpretation with [ANALYSIS] notation
- Identify any gaps in the information that need further research"""

        return self.generate(prompt, system_prompt=system_prompt)

    def generate_search_queries(
        self,
        topic: str,
        existing_info: str = "",
        gaps: List[str] = None
    ) -> LLMResponse:
        """
        Generate additional search queries based on research progress.

        Args:
            topic: The research topic
            existing_info: Summary of already gathered information
            gaps: Identified information gaps

        Returns:
            LLMResponse with new search queries
        """
        system_prompt = """You are a research assistant specializing in query generation.
Generate specific, targeted search queries that will help fill information gaps
and deepen understanding of the research topic."""

        gaps_text = "\n".join(gaps) if gaps else "Not yet identified"

        prompt = f"""Research Topic: {topic}

Information Already Gathered:
{existing_info if existing_info else "Initial research phase"}

Identified Information Gaps:
{gaps_text}

Generate 5-10 specific search queries that will help:
1. Fill the identified gaps
2. Find supporting evidence
3. Explore different perspectives
4. Verify existing information

Return as a JSON array of search queries."""

        return self.generate(prompt, system_prompt=system_prompt)

    def verify_information(
        self,
        content: str,
        sources: List[Dict[str, Any]]
    ) -> LLMResponse:
        """
        Verify information for potential hallucinations.

        Args:
            content: The content to verify
            sources: Source information for reference

        Returns:
            LLMResponse with verification results
        """
        system_prompt = """You are a fact-checking assistant. Your task is to identify:
1. Claims that are well-supported by the provided sources
2. Claims that may be inaccurate or unsupported
3. Potential hallucinations or unsupported inferences
4. Areas requiring additional verification

Be thorough and skeptical. Rate confidence levels for each claim."""

        sources_text = "\n".join([
            f"Source {i+1}: {s.get('url', 'Unknown')}\n{s.get('content', '')[:500]}..."
            for i, s in enumerate(sources)
        ])

        prompt = f"""Content to Verify:
{content}

Available Sources:
{sources_text}

Please verify this content and provide:
1. A list of claims with confidence ratings (HIGH/MEDIUM/LOW/UNSUPPORTED)
2. Potential hallucinations or unsupported statements
3. Recommendations for additional verification

Return in JSON format:
{{
    "verified_claims": [{{"claim": "...", "confidence": "HIGH/MEDIUM/LOW", "source_reference": "..."}}],
    "potential_hallucinations": [{{"statement": "...", "reason": "..."}}],
    "verification_recommendations": ["..."]
}}"""

        return self.generate(prompt, system_prompt=system_prompt)
