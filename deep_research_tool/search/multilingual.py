"""
Multilingual search module for Deep Research Tool.

Enables searching across multiple languages with query translation,
result aggregation, and deduplication.
"""

import asyncio
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from difflib import SequenceMatcher

from ..config import MultilingualSearchConfig, LANGUAGE_REGION_MAP


@dataclass
class TranslatedQuery:
    """A query translated to a specific language."""
    original_query: str
    translated_query: str
    target_language: str
    confidence: float = 1.0


@dataclass
class MultilingualSearchResult:
    """A search result with language metadata."""
    url: str
    title: str
    snippet: str
    source_language: str
    search_query: str
    relevance_score: float = 0.0

    # Original content before translation
    original_title: str = ""
    original_snippet: str = ""

    # Translation metadata
    is_translated: bool = False
    translation_confidence: float = 1.0

    def get_content_hash(self) -> str:
        """Generate a hash for deduplication."""
        # Use URL as primary dedup key
        return hashlib.md5(self.url.encode()).hexdigest()


@dataclass
class MultilingualSearchStats:
    """Statistics for multilingual search."""
    total_results: int = 0
    results_by_language: Dict[str, int] = field(default_factory=dict)
    duplicates_removed: int = 0
    queries_translated: int = 0
    translation_errors: int = 0

    def to_dict(self) -> Dict:
        return {
            "total_results": self.total_results,
            "results_by_language": self.results_by_language,
            "duplicates_removed": self.duplicates_removed,
            "queries_translated": self.queries_translated,
            "translation_errors": self.translation_errors,
        }

    def get_language_distribution(self) -> List[Tuple[str, int, float]]:
        """Get language distribution as (lang, count, percentage) tuples."""
        if self.total_results == 0:
            return []

        distribution = []
        for lang, count in sorted(self.results_by_language.items(),
                                   key=lambda x: x[1], reverse=True):
            percentage = (count / self.total_results) * 100
            lang_name = LANGUAGE_REGION_MAP.get(lang, {}).get("name", lang)
            distribution.append((lang_name, count, percentage))

        return distribution


class MultilingualSearcher:
    """
    Handles multilingual search operations.

    Features:
    - Query translation via LLM
    - Parallel search across multiple languages
    - Result deduplication
    - Relevance scoring with language weights
    """

    def __init__(
        self,
        config: MultilingualSearchConfig,
        search_client: Any,
        llm_client: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        self.config = config
        self.search_client = search_client
        self.llm_client = llm_client
        self.progress_callback = progress_callback
        self.stats = MultilingualSearchStats()

    def _report_progress(self, message: str, progress: float):
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(message, progress)

    def translate_query(self, query: str, target_language: str) -> TranslatedQuery:
        """
        Translate a query to the target language using LLM.

        Args:
            query: Original query text
            target_language: Target language code (e.g., 'en', 'zh')

        Returns:
            TranslatedQuery object
        """
        if not self.llm_client:
            return TranslatedQuery(
                original_query=query,
                translated_query=query,
                target_language=target_language,
                confidence=0.5
            )

        lang_info = LANGUAGE_REGION_MAP.get(target_language, {})
        lang_name = lang_info.get("name", target_language)

        try:
            prompt = f"""Translate the following search query to {lang_name}.
Only output the translated query, nothing else.
Keep the query concise and suitable for web search.

Query: {query}

Translated query:"""

            response = self.llm_client.generate(prompt, max_tokens=200)
            translated = response.strip()

            self.stats.queries_translated += 1

            return TranslatedQuery(
                original_query=query,
                translated_query=translated,
                target_language=target_language,
                confidence=0.9
            )

        except Exception as e:
            self.stats.translation_errors += 1
            # Fall back to original query
            return TranslatedQuery(
                original_query=query,
                translated_query=query,
                target_language=target_language,
                confidence=0.3
            )

    def translate_queries(self, query: str) -> List[TranslatedQuery]:
        """
        Translate query to all configured languages.

        Args:
            query: Original query text

        Returns:
            List of TranslatedQuery objects
        """
        translations = []

        for lang in self.config.search_languages:
            if self.config.query_translation == "llm":
                translated = self.translate_query(query, lang)
            else:
                # No translation - use original query
                translated = TranslatedQuery(
                    original_query=query,
                    translated_query=query,
                    target_language=lang,
                    confidence=1.0
                )
            translations.append(translated)

        return translations

    def search_single_language(
        self,
        query: TranslatedQuery,
    ) -> List[MultilingualSearchResult]:
        """
        Search in a single language.

        Args:
            query: TranslatedQuery object

        Returns:
            List of MultilingualSearchResult objects
        """
        region = self.config.get_region_for_language(query.target_language)
        print(f"[Multilingual] Searching ({query.target_language}): {query.translated_query}")

        try:
            # Use the search client with language-specific region
            raw_results = self.search_client.search(
                query.translated_query,
                max_results=self.config.results_per_language,
                region=region
            )

            results = []
            for r in raw_results:
                result = MultilingualSearchResult(
                    url=r.url if hasattr(r, 'url') else r.get('url', ''),
                    title=r.title if hasattr(r, 'title') else r.get('title', ''),
                    snippet=r.snippet if hasattr(r, 'snippet') else r.get('snippet', ''),
                    source_language=query.target_language,
                    search_query=query.translated_query,
                    original_title=r.title if hasattr(r, 'title') else r.get('title', ''),
                    original_snippet=r.snippet if hasattr(r, 'snippet') else r.get('snippet', ''),
                )
                results.append(result)

            return results

        except Exception as e:
            # Log error but don't fail entire search
            return []

    def search_parallel(
        self,
        query: str,
    ) -> Tuple[List[MultilingualSearchResult], MultilingualSearchStats]:
        """
        Search across all configured languages in parallel.

        Args:
            query: Original search query

        Returns:
            Tuple of (results list, statistics)
        """
        self.stats = MultilingualSearchStats()

        # Translate queries
        self._report_progress("Translating queries...", 10)
        translated_queries = self.translate_queries(query)

        # Search in parallel
        all_results = []
        max_workers = min(self.config.max_concurrent_searches,
                         len(self.config.search_languages))

        self._report_progress(f"Searching in {len(self.config.search_languages)} languages...", 20)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.search_single_language, tq): tq
                for tq in translated_queries
            }

            completed = 0
            for future in as_completed(futures):
                tq = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)

                    # Update stats
                    lang = tq.target_language
                    self.stats.results_by_language[lang] = len(results)

                except Exception as e:
                    pass

                completed += 1
                progress = 20 + (60 * completed / len(futures))
                lang_name = LANGUAGE_REGION_MAP.get(tq.target_language, {}).get("name", tq.target_language)
                self._report_progress(f"Searched {lang_name}", progress)

        # Deduplicate results
        self._report_progress("Deduplicating results...", 85)
        deduplicated = self._deduplicate_results(all_results)

        self.stats.total_results = len(deduplicated)
        self.stats.duplicates_removed = len(all_results) - len(deduplicated)

        # Score and sort results
        self._report_progress("Scoring results...", 95)
        scored = self._score_results(deduplicated)

        self._report_progress("Multilingual search complete", 100)

        return scored, self.stats

    def _deduplicate_results(
        self,
        results: List[MultilingualSearchResult]
    ) -> List[MultilingualSearchResult]:
        """
        Remove duplicate results based on URL and content similarity.

        Args:
            results: List of results to deduplicate

        Returns:
            Deduplicated list of results
        """
        seen_urls = set()
        seen_hashes = set()
        deduplicated = []

        for result in results:
            # Check URL
            if result.url in seen_urls:
                continue

            # Check content hash
            content_hash = result.get_content_hash()
            if content_hash in seen_hashes:
                continue

            # Check title similarity against existing results
            is_duplicate = False
            for existing in deduplicated:
                similarity = SequenceMatcher(
                    None,
                    result.title.lower(),
                    existing.title.lower()
                ).ratio()

                if similarity >= self.config.dedup_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_urls.add(result.url)
                seen_hashes.add(content_hash)
                deduplicated.append(result)

        return deduplicated

    def _score_results(
        self,
        results: List[MultilingualSearchResult]
    ) -> List[MultilingualSearchResult]:
        """
        Score results based on relevance and language weights.

        Args:
            results: List of results to score

        Returns:
            Sorted list of results by score
        """
        for result in results:
            # Base score from position (earlier results tend to be more relevant)
            base_score = 1.0

            # Apply language weight
            lang_weight = self.config.get_language_weight(result.source_language)

            result.relevance_score = base_score * lang_weight

        # Sort by score descending
        return sorted(results, key=lambda r: r.relevance_score, reverse=True)

    def translate_content(
        self,
        content: str,
        source_language: str,
        target_language: str
    ) -> Tuple[str, float]:
        """
        Translate content from source to target language.

        Args:
            content: Content to translate
            source_language: Source language code
            target_language: Target language code

        Returns:
            Tuple of (translated content, confidence score)
        """
        if source_language == target_language:
            return content, 1.0

        if not self.llm_client:
            return content, 0.5

        source_name = LANGUAGE_REGION_MAP.get(source_language, {}).get("name", source_language)
        target_name = LANGUAGE_REGION_MAP.get(target_language, {}).get("name", target_language)

        try:
            prompt = f"""Translate the following text from {source_name} to {target_name}.
Preserve the meaning and tone. Only output the translation.

Text:
{content}

Translation:"""

            response = self.llm_client.generate(prompt, max_tokens=len(content) * 2)
            return response.strip(), 0.85

        except Exception:
            return content, 0.3

    def get_stats_markdown(self) -> str:
        """Generate markdown summary of search statistics."""
        lines = ["### Multilingual Search Statistics", ""]
        lines.append(f"**Total Results:** {self.stats.total_results}")
        lines.append(f"**Duplicates Removed:** {self.stats.duplicates_removed}")
        lines.append("")

        distribution = self.stats.get_language_distribution()
        if distribution:
            lines.append("#### Results by Language")
            lines.append("| Language | Count | Percentage |")
            lines.append("|----------|-------|------------|")
            for lang_name, count, pct in distribution:
                lines.append(f"| {lang_name} | {count} | {pct:.1f}% |")

        return "\n".join(lines)


def create_multilingual_searcher(
    config: MultilingualSearchConfig,
    search_client: Any,
    llm_client: Optional[Any] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> MultilingualSearcher:
    """
    Factory function to create a MultilingualSearcher.

    Args:
        config: Multilingual search configuration
        search_client: Search client instance (e.g., DuckDuckGoSearch)
        llm_client: Optional LLM client for translations
        progress_callback: Optional progress callback

    Returns:
        Configured MultilingualSearcher instance
    """
    return MultilingualSearcher(
        config=config,
        search_client=search_client,
        llm_client=llm_client,
        progress_callback=progress_callback,
    )
