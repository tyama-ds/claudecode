"""Search capability mixin for agents that can gather information."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

try:
    from deep_research_tool.search import get_search_client, SearchResult
    HAS_SEARCH_TOOL = True
except ImportError:
    HAS_SEARCH_TOOL = False
    SearchResult = None


@dataclass
class AgentSearchConfig:
    """Configuration for agent search capabilities."""
    enabled: bool = True
    search_method: str = "duckduckgo"  # "duckduckgo" or "selenium"
    max_queries_per_turn: int = 3
    max_results_per_query: int = 5
    max_content_length: int = 500  # Per result snippet
    extract_page_content: bool = False  # Whether to fetch full pages
    region: str = "jp-jp"  # Default to Japan for Japanese discussions
    search_kwargs: Dict[str, Any] = field(default_factory=dict)


class SearchCapabilityMixin:
    """
    Mixin that adds web search capabilities to any agent.

    Usage:
        class ResearchParticipantAgent(SearchCapabilityMixin, ParticipantAgent):
            ...

    Requires the host class to have:
        - self.config (AgentConfig)
        - self.llm_config (LLMConfig)
        - self._call_llm(messages, system_prompt)
        - self.name (str)
        - self.get_system_prompt()
    """

    def __init__(self, *args, search_config: Optional[AgentSearchConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_config = search_config or AgentSearchConfig()
        self._search_client = None
        self._search_history: List[Dict[str, Any]] = []  # Track all searches

    @property
    def search_client(self):
        """Lazy initialization of search client."""
        if self._search_client is None:
            if not HAS_SEARCH_TOOL:
                raise ImportError(
                    "deep_research_tool is required for search capabilities. "
                    "Please install it or set search_config.enabled=False."
                )
            self._search_client = get_search_client(
                method=self.search_config.search_method,
                max_results=self.search_config.max_results_per_query,
                **self.search_config.search_kwargs,
            )
        return self._search_client

    def generate_search_queries(
        self,
        topic: str,
        conversation_history: List,
        context: Optional[str] = None,
    ) -> List[str]:
        """
        Use the LLM to generate targeted search queries based on persona and discussion.

        Args:
            topic: Discussion topic
            conversation_history: List of Message objects
            context: Optional additional context

        Returns:
            List of search query strings (1-3 queries)
        """
        # Format recent conversation history
        history_text = "\n".join([
            f"[{msg.agent_name}]: {msg.content[:200]}..."
            if len(msg.content) > 200 else f"[{msg.agent_name}]: {msg.content}"
            for msg in conversation_history[-5:]
        ]) if conversation_history else "(議論はまだ始まっていません)"

        # Get persona from config
        persona = getattr(self.config, 'persona', '') or "議論の参加者"

        prompt = f"""あなたは「{persona}」の専門家です。
以下の議論トピックと最近の議論内容を踏まえて、あなたの専門分野の視点から
情報を収集するための検索クエリを1〜{self.search_config.max_queries_per_turn}個生成してください。

トピック: {topic}

最近の議論:
{history_text}

以下の形式で検索クエリのみを出力してください（1行1クエリ）:
QUERY: <検索クエリ>

注意:
- 日本語で検索クエリを生成してください
- あなたの専門分野（{persona}）に関連する情報を検索してください
- 具体的で検索に適したクエリにしてください
"""
        if context:
            prompt += f"\n追加の指示: {context}"

        messages = [{"role": "user", "content": prompt}]
        response = self._call_llm(messages, self.get_system_prompt())

        # Parse queries from response
        queries = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("QUERY:"):
                query = line[6:].strip()
                if query:
                    queries.append(query)

        # Fallback: if no QUERY: prefix found, treat each non-empty line as a query
        if not queries:
            for line in response.strip().split("\n"):
                line = line.strip()
                # Skip lines that look like explanations
                if line and not line.startswith(("-", "・", "※", "注")):
                    queries.append(line)

        return queries[:self.search_config.max_queries_per_turn]

    def execute_searches(self, queries: List[str]) -> List:
        """
        Execute search queries and return combined results.

        Args:
            queries: List of search query strings

        Returns:
            List of SearchResult objects
        """
        all_results = []

        for query in queries:
            try:
                if self.search_config.extract_page_content:
                    results = self.search_client.search_and_extract(
                        query,
                        max_pages=2,
                    )
                else:
                    results = self.search_client.search(query)

                all_results.extend(results)

                # Track for metadata
                self._search_history.append({
                    "query": query,
                    "result_count": len(results),
                    "urls": [r.url for r in results[:5]],  # Limit URLs in history
                })
            except Exception as e:
                self._search_history.append({
                    "query": query,
                    "error": str(e),
                })

        return all_results

    def format_search_results(self, results: List) -> str:
        """
        Format search results into a text block for prompt injection.

        Args:
            results: List of SearchResult objects

        Returns:
            Formatted string with search results
        """
        if not results:
            return "(検索結果なし)"

        formatted = []
        seen_urls = set()

        for result in results:
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)

            snippet = result.snippet or result.content
            if snippet and len(snippet) > self.search_config.max_content_length:
                snippet = snippet[:self.search_config.max_content_length] + "..."

            entry = f"- **{result.title}**\n  URL: {result.url}\n  {snippet}"
            formatted.append(entry)

            # Limit total formatted results
            if len(formatted) >= self.search_config.max_results_per_query * 2:
                break

        return "\n\n".join(formatted)

    def research_and_build_context(
        self,
        topic: str,
        conversation_history: List,
        context: Optional[str] = None,
    ) -> tuple:
        """
        Full research cycle: generate queries -> search -> format results.

        Args:
            topic: Discussion topic
            conversation_history: List of Message objects
            context: Optional additional context

        Returns:
            Tuple of (formatted_search_context, search_metadata)
        """
        if not self.search_config.enabled:
            return "", {}

        if not HAS_SEARCH_TOOL:
            return "", {"error": "deep_research_tool not available"}

        try:
            queries = self.generate_search_queries(topic, conversation_history, context)
            results = self.execute_searches(queries)
            formatted = self.format_search_results(results)

            metadata = {
                "queries": queries,
                "result_count": len(results),
                "sources": [
                    {"title": r.title, "url": r.url}
                    for r in results[:10]  # Limit sources in metadata
                ],
            }

            return formatted, metadata

        except Exception as e:
            return "", {"error": str(e)}

    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get the history of all searches performed by this agent."""
        return self._search_history.copy()

    def clear_search_history(self):
        """Clear the search history."""
        self._search_history = []
