"""
Base class for LLM API clients.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


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
        max_tokens: int = 4096,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for the provider
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
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
