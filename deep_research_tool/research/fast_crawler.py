"""
Fast Crawler - Parallel content fetching and batch/parallel relevance evaluation.

This module provides optimized crawling by:
1. Phase 1: Parallel HTTP fetching (no LLM calls)
2. Phase 2: Batch or parallel LLM relevance evaluation
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Tuple
from urllib.parse import urlparse

from ..evidence.content_filter import ContentFilter, create_moderate_filter


class EvaluationMode(str, Enum):
    """Mode for relevance evaluation."""
    BATCH = "batch"          # Single LLM call evaluates multiple pages
    PARALLEL = "parallel"    # Multiple parallel LLM calls
    SEQUENTIAL = "sequential"  # Original sequential mode (fallback)


@dataclass
class CrawledPage:
    """Represents a crawled page before relevance evaluation."""
    url: str
    title: str
    snippet: str
    content: str
    fetch_time: float = 0.0
    error: Optional[str] = None
    filtered: bool = False
    filter_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluatedPage(CrawledPage):
    """Crawled page with relevance evaluation results."""
    relevance_score: float = 0.0
    processed_content: str = ""
    key_points: List[str] = field(default_factory=list)
    quotes: List[str] = field(default_factory=list)
    evaluation_time: float = 0.0


@dataclass
class CrawlResult:
    """Result of a fast crawl operation."""
    pages: List[EvaluatedPage]
    total_fetch_time: float
    total_eval_time: float
    pages_fetched: int
    pages_filtered: int
    pages_evaluated: int
    errors: List[str] = field(default_factory=list)


class FastCrawler:
    """
    Fast parallel crawler with batch/parallel relevance evaluation.

    Usage:
        crawler = FastCrawler(
            search_client=search_client,
            llm_client=llm_client,
            evaluation_mode=EvaluationMode.BATCH,
        )

        result = crawler.crawl_and_evaluate(
            queries=["query1", "query2"],
            section_context="1. Introduction",
            max_pages_per_query=3,
        )
    """

    def __init__(
        self,
        search_client,
        llm_client,
        evaluation_mode: EvaluationMode = EvaluationMode.BATCH,
        content_filter: ContentFilter = None,
        max_workers: int = 10,
        fetch_timeout: int = 15,
        batch_size: int = 5,
        language: str = "ja",
    ):
        """
        Initialize FastCrawler.

        Args:
            search_client: Web search client
            llm_client: LLM API client for relevance evaluation
            evaluation_mode: How to evaluate relevance (batch, parallel, sequential)
            content_filter: Content filter for ads/spam removal
            max_workers: Max parallel workers for fetching
            fetch_timeout: Timeout for page fetching (seconds)
            batch_size: Pages per batch in batch evaluation mode
            language: Language for evaluation prompts
        """
        self.search = search_client
        self.llm = llm_client
        self.evaluation_mode = evaluation_mode
        self.content_filter = content_filter or create_moderate_filter()
        self.max_workers = max_workers
        self.fetch_timeout = fetch_timeout
        self.batch_size = batch_size
        self.language = language

    def _extract_research_context(self, research_topic: str) -> dict:
        """
        Extract key aspects from the research topic for evaluation context.

        Args:
            research_topic: Original research query/topic

        Returns:
            Dict with extracted context elements
        """
        if not research_topic:
            return {"topic": "", "keywords": [], "focus_areas": ""}

        # For short queries, use as-is
        if len(research_topic) < 100:
            # Extract potential keywords (simple approach)
            import re
            # Remove common particles and split
            topic_clean = re.sub(r'[、。・「」『』（）\[\]【】]', ' ', research_topic)
            words = [w.strip() for w in topic_clean.split() if len(w.strip()) > 1]
            # Filter out very common words
            stop_words = {'の', 'を', 'に', 'は', 'が', 'と', 'で', 'する', 'ある', 'について', 'に関する',
                         'the', 'a', 'an', 'of', 'in', 'to', 'and', 'for', 'on', 'about', 'with'}
            keywords = [w for w in words if w.lower() not in stop_words][:10]

            return {
                "topic": research_topic,
                "keywords": keywords,
                "focus_areas": research_topic,
            }

        # For longer queries, use LLM to extract key aspects
        if self.language == "ja":
            extract_prompt = f"""以下の調査テーマから、キーワードと調査の焦点を抽出してください。

調査テーマ: {research_topic}

以下のJSON形式で回答:
{{"keywords": ["キーワード1", "キーワード2", ...], "focus_areas": "調査の主な焦点を1-2文で"}}

JSONのみ出力:"""
        else:
            extract_prompt = f"""Extract keywords and focus areas from this research topic.

Research Topic: {research_topic}

Respond in JSON format:
{{"keywords": ["keyword1", "keyword2", ...], "focus_areas": "Main focus in 1-2 sentences"}}

Output only JSON:"""

        try:
            response = self.llm.generate(extract_prompt)
            import json
            content = response.content.strip()
            if "```" in content:
                content = content.split("```")[1].split("```")[0]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content)
            return {
                "topic": research_topic,
                "keywords": data.get("keywords", [])[:10],
                "focus_areas": data.get("focus_areas", research_topic[:200]),
            }
        except Exception:
            # Fallback to simple extraction
            return {
                "topic": research_topic,
                "keywords": [],
                "focus_areas": research_topic[:200],
            }

    def crawl_and_evaluate(
        self,
        queries: List[str],
        section_context: str,
        research_topic: str = "",
        max_pages_per_query: int = 3,
        min_relevance_score: float = 0.3,
        progress_callback: Callable[[str, int, int], None] = None,
    ) -> CrawlResult:
        """
        Crawl pages for all queries and evaluate relevance.

        Args:
            queries: List of search queries
            section_context: Section context for relevance evaluation
            research_topic: Original research topic/purpose for context-aware evaluation
            max_pages_per_query: Max pages to fetch per query
            min_relevance_score: Minimum relevance score to include
            progress_callback: Callback(message, current, total)

        Returns:
            CrawlResult with evaluated pages
        """
        # Phase 1: Fast parallel fetching
        if progress_callback:
            progress_callback("Phase 1: Searching and fetching pages...", 0, 100)

        fetch_start = time.time()
        crawled_pages = self._parallel_fetch(
            queries=queries,
            max_pages_per_query=max_pages_per_query,
            progress_callback=progress_callback,
        )
        fetch_time = time.time() - fetch_start

        # Apply content filter
        filtered_pages = []
        filtered_count = 0
        for page in crawled_pages:
            if page.error:
                filtered_count += 1
                continue

            if self.content_filter:
                filter_result = self.content_filter.filter_content(
                    url=page.url,
                    title=page.title,
                    content=page.content,
                )
                if not filter_result.should_include:
                    page.filtered = True
                    page.filter_reason = filter_result.reason
                    filtered_count += 1
                    continue

            filtered_pages.append(page)

        if progress_callback:
            progress_callback(
                f"Phase 1 complete: {len(filtered_pages)} pages after filtering",
                50, 100
            )

        # Phase 2: Relevance evaluation
        if progress_callback:
            progress_callback("Phase 2: Evaluating relevance...", 50, 100)

        eval_start = time.time()

        # Extract research context for better evaluation
        research_context = self._extract_research_context(research_topic)

        if self.evaluation_mode == EvaluationMode.BATCH:
            evaluated_pages = self._batch_evaluate(
                pages=filtered_pages,
                section_context=section_context,
                research_context=research_context,
                progress_callback=progress_callback,
            )
        elif self.evaluation_mode == EvaluationMode.PARALLEL:
            evaluated_pages = self._parallel_evaluate(
                pages=filtered_pages,
                section_context=section_context,
                research_context=research_context,
                progress_callback=progress_callback,
            )
        else:  # SEQUENTIAL
            evaluated_pages = self._sequential_evaluate(
                pages=filtered_pages,
                section_context=section_context,
                research_context=research_context,
                progress_callback=progress_callback,
            )

        eval_time = time.time() - eval_start

        # Filter by relevance score
        relevant_pages = [
            p for p in evaluated_pages
            if p.relevance_score >= min_relevance_score
        ]

        if progress_callback:
            progress_callback(
                f"Complete: {len(relevant_pages)} relevant pages found",
                100, 100
            )

        return CrawlResult(
            pages=relevant_pages,
            total_fetch_time=fetch_time,
            total_eval_time=eval_time,
            pages_fetched=len(crawled_pages),
            pages_filtered=filtered_count,
            pages_evaluated=len(evaluated_pages),
            errors=[p.error for p in crawled_pages if p.error],
        )

    def _parallel_fetch(
        self,
        queries: List[str],
        max_pages_per_query: int,
        progress_callback: Callable = None,
    ) -> List[CrawledPage]:
        """
        Fetch pages for all queries in parallel.

        Args:
            queries: Search queries
            max_pages_per_query: Max results per query
            progress_callback: Progress callback

        Returns:
            List of crawled pages
        """
        # First, execute all searches to get URLs
        all_results = []
        for qi, query in enumerate(queries, 1):
            print(f"[FastCrawler] ({qi}/{len(queries)}) Query: {query}")
            try:
                results = self.search.search(query, max_results=max_pages_per_query)
                for result in results[:max_pages_per_query]:
                    all_results.append({
                        "url": result.url,
                        "title": result.title,
                        "snippet": result.snippet,
                        "query": query,
                    })
            except Exception as e:
                print(f"[FastCrawler] Search error for '{query}': {e}")

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result["url"] not in seen_urls:
                seen_urls.add(result["url"])
                unique_results.append(result)

        if progress_callback:
            progress_callback(
                f"Found {len(unique_results)} unique URLs",
                10, 100
            )

        # Parallel fetch all pages
        crawled_pages = []
        total = len(unique_results)

        def fetch_page(result: Dict) -> CrawledPage:
            """Fetch a single page."""
            start = time.time()
            try:
                page = self.search.get_page_content(result["url"])
                return CrawledPage(
                    url=result["url"],
                    title=result["title"],
                    snippet=result["snippet"],
                    content=page.text_content,
                    fetch_time=time.time() - start,
                    metadata={"query": result["query"]},
                )
            except Exception as e:
                return CrawledPage(
                    url=result["url"],
                    title=result["title"],
                    snippet=result["snippet"],
                    content="",
                    fetch_time=time.time() - start,
                    error=str(e),
                )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(fetch_page, result): result
                for result in unique_results
            }

            completed = 0
            for future in as_completed(futures):
                crawled_pages.append(future.result())
                completed += 1
                if progress_callback and completed % 5 == 0:
                    progress_callback(
                        f"Fetched {completed}/{total} pages",
                        10 + int(40 * completed / total), 100
                    )

        return crawled_pages

    def _batch_evaluate(
        self,
        pages: List[CrawledPage],
        section_context: str,
        research_context: dict = None,
        progress_callback: Callable = None,
    ) -> List[EvaluatedPage]:
        """
        Evaluate relevance using batch prompts (multiple pages per LLM call).

        Args:
            pages: Crawled pages to evaluate
            section_context: Context for evaluation
            research_context: Extracted research context dict with topic, keywords, focus_areas
            progress_callback: Progress callback

        Returns:
            List of evaluated pages
        """
        research_context = research_context or {"topic": "", "keywords": [], "focus_areas": ""}
        evaluated_pages = []
        total_batches = (len(pages) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(0, len(pages), self.batch_size):
            batch = pages[batch_idx:batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size + 1

            if progress_callback:
                progress_callback(
                    f"Evaluating batch {batch_num}/{total_batches}",
                    50 + int(50 * batch_num / total_batches), 100
                )

            start = time.time()
            try:
                batch_results = self._evaluate_batch(batch, section_context, research_context)
                eval_time = time.time() - start

                for page, result in zip(batch, batch_results):
                    evaluated_pages.append(EvaluatedPage(
                        url=page.url,
                        title=page.title,
                        snippet=page.snippet,
                        content=page.content,
                        fetch_time=page.fetch_time,
                        metadata=page.metadata,
                        relevance_score=result.get("relevance_score", 0.0),
                        processed_content=result.get("processed_content", ""),
                        key_points=result.get("key_points", []),
                        evaluation_time=eval_time / len(batch),
                    ))
            except Exception as e:
                print(f"[FastCrawler] Batch evaluation error: {e}")
                # Fallback: give low scores to failed batch
                for page in batch:
                    evaluated_pages.append(EvaluatedPage(
                        url=page.url,
                        title=page.title,
                        snippet=page.snippet,
                        content=page.content,
                        fetch_time=page.fetch_time,
                        metadata=page.metadata,
                        relevance_score=0.1,
                        error=str(e),
                    ))

        return evaluated_pages

    def _evaluate_batch(
        self,
        pages: List[CrawledPage],
        section_context: str,
        research_context: dict = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of pages with a single LLM call.

        Args:
            pages: Pages to evaluate
            section_context: Context for evaluation
            research_context: Extracted research context dict

        Returns:
            List of evaluation results
        """
        research_context = research_context or {"topic": "", "keywords": [], "focus_areas": ""}

        # Build batch prompt
        pages_text = []
        for i, page in enumerate(pages, 1):
            content_preview = page.content[:1500] if page.content else page.snippet
            pages_text.append(f"""
=== PAGE {i} ===
URL: {page.url}
Title: {page.title}
Content Preview:
{content_preview}
""")

        # Build enhanced context strings with keywords
        research_topic = research_context.get("topic", "")
        keywords = research_context.get("keywords", [])
        focus_areas = research_context.get("focus_areas", "")

        keywords_str = ", ".join(keywords) if keywords else ""

        if self.language == "ja":
            context_block = f"""【調査コンテキスト】
調査テーマ: {research_topic}
{f'重要キーワード: {keywords_str}' if keywords_str else ''}
{f'調査の焦点: {focus_areas}' if focus_areas and focus_areas != research_topic else ''}
現在のセクション: {section_context}
""" if research_topic else f"現在のセクション: {section_context}"

            prompt = f"""以下の{len(pages)}ページについて、調査テーマとの関連性を評価してください。

{context_block}

【重要な評価指針】
1. 調査テーマ「{research_topic}」に直接関連する情報かどうかを最優先で判断
2. 重要キーワード（{keywords_str}）を含むページを高く評価
3. セクション「{section_context}」の内容に貢献する情報かを確認
4. 一般的・無関係な情報（広告、他トピック、ニュース一般等）は低スコア

各ページについて、以下の形式でJSON配列として回答してください:
[
  {{"page": 1, "relevance_score": 0.0-1.0, "key_points": ["要点1", "要点2"], "processed_content": "関連する内容の要約（200-500文字）"}},
  ...
]

評価基準:
- 0.8-1.0: 調査テーマに直接関連し、重要キーワードを含む具体的情報
- 0.6-0.7: 調査テーマに関連するが、やや周辺的な情報
- 0.3-0.5: 部分的に関連、または間接的に有用
- 0.1-0.2: わずかに関連、情報価値が低い
- 0.0: 調査テーマと無関係

{"".join(pages_text)}

JSON配列のみを出力してください:"""
        else:
            context_block = f"""[Research Context]
Research Topic: {research_topic}
{f'Key Keywords: {keywords_str}' if keywords_str else ''}
{f'Research Focus: {focus_areas}' if focus_areas and focus_areas != research_topic else ''}
Current Section: {section_context}
""" if research_topic else f"Current Section: {section_context}"

            prompt = f"""Evaluate the relevance of the following {len(pages)} pages to the research topic.

{context_block}

IMPORTANT EVALUATION GUIDELINES:
1. Prioritize whether the information directly relates to the research topic "{research_topic}"
2. Rate pages containing key keywords ({keywords_str}) higher
3. Check if the information contributes to the section "{section_context}"
4. Give low scores to generic/unrelated content (ads, off-topic, general news)

For each page, respond with a JSON array in this format:
[
  {{"page": 1, "relevance_score": 0.0-1.0, "key_points": ["point1", "point2"], "processed_content": "Summary of relevant content (200-500 chars)"}},
  ...
]

Scoring criteria:
- 0.8-1.0: Directly related to research topic with key keywords and specific information
- 0.6-0.7: Related to research topic but somewhat peripheral
- 0.3-0.5: Partially related or indirectly useful
- 0.1-0.2: Slightly related, low information value
- 0.0: Not related to research topic

{"".join(pages_text)}

Output only the JSON array:"""

        response = self.llm.generate(prompt)

        # Parse response
        import json
        try:
            # Try to extract JSON from response
            content = response.content.strip()
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            results = json.loads(content)

            # Ensure we have results for all pages
            if len(results) < len(pages):
                # Pad with low-score defaults
                for i in range(len(results), len(pages)):
                    results.append({
                        "page": i + 1,
                        "relevance_score": 0.1,
                        "key_points": [],
                        "processed_content": "",
                    })

            return results

        except json.JSONDecodeError:
            # Fallback: give moderate scores
            return [
                {
                    "page": i + 1,
                    "relevance_score": 0.5,
                    "key_points": [],
                    "processed_content": page.content[:500] if page.content else "",
                }
                for i, page in enumerate(pages)
            ]

    def _parallel_evaluate(
        self,
        pages: List[CrawledPage],
        section_context: str,
        research_context: dict = None,
        progress_callback: Callable = None,
    ) -> List[EvaluatedPage]:
        """
        Evaluate relevance using parallel LLM calls.

        Args:
            pages: Crawled pages to evaluate
            section_context: Context for evaluation
            research_context: Extracted research context dict
            progress_callback: Progress callback

        Returns:
            List of evaluated pages
        """
        research_context = research_context or {"topic": "", "keywords": [], "focus_areas": ""}
        evaluated_pages = []
        total = len(pages)

        def evaluate_single(page: CrawledPage) -> EvaluatedPage:
            """Evaluate a single page."""
            start = time.time()
            try:
                result = self._evaluate_single_page(page, section_context, research_context)
                return EvaluatedPage(
                    url=page.url,
                    title=page.title,
                    snippet=page.snippet,
                    content=page.content,
                    fetch_time=page.fetch_time,
                    metadata=page.metadata,
                    relevance_score=result.get("relevance_score", 0.0),
                    processed_content=result.get("processed_content", ""),
                    key_points=result.get("key_points", []),
                    evaluation_time=time.time() - start,
                )
            except Exception as e:
                return EvaluatedPage(
                    url=page.url,
                    title=page.title,
                    snippet=page.snippet,
                    content=page.content,
                    fetch_time=page.fetch_time,
                    metadata=page.metadata,
                    relevance_score=0.1,
                    error=str(e),
                    evaluation_time=time.time() - start,
                )

        # Use ThreadPoolExecutor for parallel LLM calls
        # Note: For true async, would need async LLM client
        with ThreadPoolExecutor(max_workers=min(5, len(pages))) as executor:
            futures = {executor.submit(evaluate_single, page): page for page in pages}

            completed = 0
            for future in as_completed(futures):
                evaluated_pages.append(future.result())
                completed += 1
                if progress_callback and completed % 3 == 0:
                    progress_callback(
                        f"Evaluated {completed}/{total} pages",
                        50 + int(50 * completed / total), 100
                    )

        return evaluated_pages

    def _evaluate_single_page(
        self,
        page: CrawledPage,
        section_context: str,
        research_context: dict = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single page's relevance.

        Args:
            page: Page to evaluate
            section_context: Context for evaluation
            research_context: Extracted research context dict

        Returns:
            Evaluation result dict
        """
        research_context = research_context or {"topic": "", "keywords": [], "focus_areas": ""}
        content_preview = page.content[:2000] if page.content else page.snippet

        # Build enhanced context
        research_topic = research_context.get("topic", "")
        keywords = research_context.get("keywords", [])
        keywords_str = ", ".join(keywords) if keywords else ""

        if self.language == "ja":
            context_block = f"""調査テーマ: {research_topic}
{f'重要キーワード: {keywords_str}' if keywords_str else ''}
現在のセクション: {section_context}""" if research_topic else f"現在のセクション: {section_context}"

            prompt = f"""以下のページが調査テーマにどの程度関連するか評価してください。

{context_block}

【評価指針】
- 調査テーマ「{research_topic}」に直接関連するか
- 重要キーワード（{keywords_str}）を含むか
- セクションの内容に貢献するか

URL: {page.url}
タイトル: {page.title}
内容:
{content_preview}

以下の形式でJSONで回答してください:
{{"relevance_score": 0.0-1.0, "key_points": ["要点1", "要点2"], "processed_content": "関連する内容の要約"}}

JSONのみを出力:"""
        else:
            context_block = f"""Research Topic: {research_topic}
{f'Key Keywords: {keywords_str}' if keywords_str else ''}
Current Section: {section_context}""" if research_topic else f"Current Section: {section_context}"

            prompt = f"""Evaluate how relevant this page is to the research topic.

{context_block}

EVALUATION GUIDELINES:
- Does it directly relate to "{research_topic}"?
- Does it contain key keywords ({keywords_str})?
- Does it contribute to the section content?

URL: {page.url}
Title: {page.title}
Content:
{content_preview}

Respond in JSON format:
{{"relevance_score": 0.0-1.0, "key_points": ["point1", "point2"], "processed_content": "Summary of relevant content"}}

Output only JSON:"""

        response = self.llm.generate(prompt)

        import json
        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "relevance_score": 0.5,
                "key_points": [],
                "processed_content": content_preview[:500],
            }

    def _sequential_evaluate(
        self,
        pages: List[CrawledPage],
        section_context: str,
        research_context: dict = None,
        progress_callback: Callable = None,
    ) -> List[EvaluatedPage]:
        """
        Evaluate relevance sequentially (original behavior).

        Args:
            pages: Crawled pages to evaluate
            section_context: Context for evaluation
            research_context: Extracted research context dict
            progress_callback: Progress callback

        Returns:
            List of evaluated pages
        """
        research_context = research_context or {"topic": "", "keywords": [], "focus_areas": ""}
        evaluated_pages = []
        total = len(pages)

        for i, page in enumerate(pages):
            if progress_callback:
                progress_callback(
                    f"Evaluating {i+1}/{total}",
                    50 + int(50 * (i + 1) / total), 100
                )

            start = time.time()
            try:
                result = self._evaluate_single_page(page, section_context, research_context)
                evaluated_pages.append(EvaluatedPage(
                    url=page.url,
                    title=page.title,
                    snippet=page.snippet,
                    content=page.content,
                    fetch_time=page.fetch_time,
                    metadata=page.metadata,
                    relevance_score=result.get("relevance_score", 0.0),
                    processed_content=result.get("processed_content", ""),
                    key_points=result.get("key_points", []),
                    evaluation_time=time.time() - start,
                ))
            except Exception as e:
                evaluated_pages.append(EvaluatedPage(
                    url=page.url,
                    title=page.title,
                    snippet=page.snippet,
                    content=page.content,
                    fetch_time=page.fetch_time,
                    metadata=page.metadata,
                    relevance_score=0.1,
                    error=str(e),
                    evaluation_time=time.time() - start,
                ))

        return evaluated_pages


def create_fast_crawler(
    search_client,
    llm_client,
    mode: str = "batch",
    content_filter: ContentFilter = None,
    language: str = "ja",
    **kwargs
) -> FastCrawler:
    """
    Factory function to create a FastCrawler.

    Args:
        search_client: Search client
        llm_client: LLM client
        mode: "batch", "parallel", or "sequential"
        content_filter: Content filter
        language: Language
        **kwargs: Additional options

    Returns:
        Configured FastCrawler
    """
    mode_map = {
        "batch": EvaluationMode.BATCH,
        "parallel": EvaluationMode.PARALLEL,
        "sequential": EvaluationMode.SEQUENTIAL,
    }

    return FastCrawler(
        search_client=search_client,
        llm_client=llm_client,
        evaluation_mode=mode_map.get(mode, EvaluationMode.BATCH),
        content_filter=content_filter,
        language=language,
        **kwargs
    )
