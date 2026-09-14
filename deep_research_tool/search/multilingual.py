"""
Multilingual search module for Deep Research Tool.

Enables searching across multiple languages with query translation,
result aggregation, and deduplication.
"""

import hashlib
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from ..config import (
    LANGUAGE_REGION_MAP,
    MultilingualSearchConfig,
    REGION_LOCALE_MAP,
)
from ..utils.concurrency import effective_workers
from ..utils.japanese_text import extract_keywords


@dataclass
class TranslatedQuery:
    """A query translated (or localized) for a language/region."""
    original_query: str
    translated_query: str
    target_language: str
    confidence: float = 1.0
    # region-first mode: the locale this query targets ("" in language mode)
    region: str = ""


@dataclass
class MultilingualSearchResult:
    """A search result with language metadata."""
    url: str
    title: str
    snippet: str
    source_language: str
    search_query: str
    relevance_score: float = 0.0
    # locale the result was found from (region-first mode)
    region: str = ""

    # Original content before translation
    original_title: str = ""
    original_snippet: str = ""

    # Translation metadata
    is_translated: bool = False
    translation_confidence: float = 1.0
    search_rank: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_content_hash(self) -> str:
        """Legacy URL-identity hash (not a content-similarity judgment)."""
        # Use URL as primary dedup key
        return hashlib.md5(self.url.encode()).hexdigest()


@dataclass
class MultilingualSearchStats:
    """Statistics for multilingual search."""
    total_results: int = 0
    results_by_language: Dict[str, int] = field(default_factory=dict)
    results_by_region: Dict[str, int] = field(default_factory=dict)
    duplicates_removed: int = 0
    queries_translated: int = 0
    translation_errors: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "total_results": self.total_results,
            "results_by_language": self.results_by_language,
            "results_by_region": self.results_by_region,
            "duplicates_removed": self.duplicates_removed,
            "queries_translated": self.queries_translated,
            "translation_errors": self.translation_errors,
            "errors": list(self.errors),
        }

    def get_language_distribution(self) -> List[Tuple[str, int, float]]:
        """Get language distribution as (lang, count, percentage) tuples."""
        total = sum(self.results_by_language.values())
        if total == 0:
            return []

        distribution = []
        for lang, count in sorted(self.results_by_language.items(),
                                   key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
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
        max_parallel_workers: Optional[int] = None,
    ):
        self.config = config
        self.search_client = search_client
        self.llm_client = llm_client
        self.progress_callback = progress_callback
        # app-wide parallelism ceiling (parallel_max_workers)
        self.max_parallel_workers = max_parallel_workers
        self.stats = MultilingualSearchStats()
        self._stats_lock = threading.Lock()
        self._search_lock = threading.RLock()
        self._translation_cache = {}

    def _report_progress(self, message: str, progress: float):
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(message, progress)

    def _llm_text(self, prompt: str, max_tokens: int = 200) -> str:
        """LLM call returning plain text.

        LLM clients return a response OBJECT with a .content attribute;
        the previous code called .strip() on the object, which raised and
        silently fell back to the untranslated query on every call.
        """
        response = self.llm_client.generate(prompt, max_tokens=max_tokens)
        text = getattr(response, "content", response)
        return str(text or "").strip()

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

            translated = self._llm_text(prompt)
            if not translated:
                raise ValueError("empty translation")

            with self._stats_lock:
                self.stats.queries_translated += 1

            return TranslatedQuery(
                original_query=query,
                translated_query=translated,
                target_language=target_language,
                confidence=0.9
            )

        except Exception:
            with self._stats_lock:
                self.stats.translation_errors += 1
            # Fall back to original query
            return TranslatedQuery(
                original_query=query,
                translated_query=query,
                target_language=target_language,
                confidence=0.3
            )

    def localize_query(self, query: str, region: str) -> TranslatedQuery:
        """LOCALIZE a query for one region (not a mere translation).

        The output uses the locale's language AND local vocabulary: how
        locals actually refer to the topic — local institution names,
        program/subsidy names, common local phrasing — so the search
        surfaces genuinely local sources.
        """
        info = REGION_LOCALE_MAP.get(region, {})
        language = info.get("language", "en")
        if not self.llm_client or not self.config.localize_queries:
            translated = (self.translate_query(query, language)
                          if self.llm_client else None)
            return TranslatedQuery(
                original_query=query,
                translated_query=(translated.translated_query
                                  if translated else query),
                target_language=language,
                confidence=translated.confidence if translated else 0.5,
                region=region,
            )

        try:
            prompt = f"""You are localizing a web search query for {info.get('name', region)}.
Rewrite the query in {info.get('lang_name', language)} the way A LOCAL
would search for this topic in {info.get('name', region)}:
- use the LOCAL names of institutions, agencies, programs, subsidies,
  regulations or products related to the topic (not literal translations)
- use the vocabulary locals actually use
- keep it concise and suitable for a web search engine

Original query: {query}

Output ONLY the localized query, nothing else."""
            localized = self._llm_text(prompt)
            if not localized:
                raise ValueError("empty localization")
            with self._stats_lock:
                self.stats.queries_translated += 1
            return TranslatedQuery(
                original_query=query,
                translated_query=localized,
                target_language=language,
                confidence=0.9,
                region=region,
            )
        except Exception:
            with self._stats_lock:
                self.stats.translation_errors += 1
            return TranslatedQuery(
                original_query=query,
                translated_query=query,
                target_language=language,
                confidence=0.3,
                region=region,
            )

    def translate_queries(self, query: str) -> List[TranslatedQuery]:
        """Localize independent locales concurrently and cache successful results."""
        region_mode = bool(self.config.search_regions)
        targets = self.config.search_regions if region_mode else self.config.search_languages

        def translate(target):
            key = (query, target, region_mode, self.config.query_translation, self.config.localize_queries)
            with self._stats_lock:
                cached = self._translation_cache.get(key)
            if cached is not None:
                return TranslatedQuery(**vars(cached))
            if self.config.query_translation == "llm":
                result = (self.localize_query(query, target) if region_mode
                          else self.translate_query(query, target))
            else:
                language = REGION_LOCALE_MAP.get(target, {}).get("language", "en") if region_mode else target
                result = TranslatedQuery(query, query, language, confidence=1.0,
                                         region=target if region_mode else "")
            if result.confidence >= 0.9:
                with self._stats_lock:
                    self._translation_cache[key] = result
            return result

        workers = effective_workers(self.max_parallel_workers, self.config.max_concurrent_searches, len(targets))
        if workers == 1:
            return [translate(target) for target in targets]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(translate, targets))

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
        if query.region:
            # region-first: the locale's DuckDuckGo region code (kl)
            info = REGION_LOCALE_MAP.get(query.region, {})
            region = info.get("kl", "wt-wt")
            label = f"{query.region}/{query.target_language}"
        else:
            region = self.config.get_region_for_language(query.target_language)
            label = query.target_language
        print(f"[Multilingual] Searching ({label}): {query.translated_query}")

        try:
            # Use the search client with the locale-specific region
            raw_results = self.search_client.search(
                query.translated_query,
                max_results=self.config.results_per_language,
                region=region
            )

            results = []
            for rank, r in enumerate(raw_results, 1):
                raw_metadata = getattr(r, "metadata", {}) if not isinstance(r, dict) else r.get("metadata", {})
                result = MultilingualSearchResult(
                    url=r.url if hasattr(r, 'url') else r.get('url', ''),
                    title=r.title if hasattr(r, 'title') else r.get('title', ''),
                    snippet=r.snippet if hasattr(r, 'snippet') else r.get('snippet', ''),
                    source_language=query.target_language,
                    search_query=query.translated_query,
                    original_title=r.title if hasattr(r, 'title') else r.get('title', ''),
                    original_snippet=r.snippet if hasattr(r, 'snippet') else r.get('snippet', ''),
                    region=query.region,
                    search_rank=rank,
                    metadata=dict(raw_metadata) if isinstance(raw_metadata, dict) else {},
                )
                results.append(result)

            return results

        except Exception as e:
            # Log error but don't fail entire search
            with self._stats_lock:
                self.stats.errors.append(f"Search failed for {label}: {e}")
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
        with self._search_lock:
            return self._search_parallel(query)

    def _search_parallel(self, query):
        self.stats = MultilingualSearchStats()

        # Translate queries
        self._report_progress("Translating queries...", 10)
        translated_queries = self.translate_queries(query)

        # Search in parallel (one task per query: regions or languages)
        all_results = []
        from ..utils.concurrency import effective_workers
        max_workers = effective_workers(self.max_parallel_workers,
                                        self.config.max_concurrent_searches,
                                        len(translated_queries))
        client = getattr(self.search_client, "_client", self.search_client)
        if (getattr(client, "supports_concurrent_search", False) is not True
                or getattr(client, "requires_serial_access", False) is True
                or "selenium" in type(client).__module__.lower()
                or "selenium" in type(client).__name__.lower()):
            max_workers = 1

        scope = (f"{len(self.config.search_regions)} regions"
                 if self.config.search_regions
                 else f"{len(self.config.search_languages)} languages")
        self._report_progress(f"Searching in {scope}...", 20)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.search_single_language, tq): tq
                       for tq in translated_queries}

            completed = 0
            # Collect in configured locale order, independent of network timing.
            for future in futures:
                tq = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)

                    # Update stats
                    lang = tq.target_language
                    self.stats.results_by_language[lang] = (
                        self.stats.results_by_language.get(lang, 0)
                        + len(results))
                    if tq.region:
                        self.stats.results_by_region[tq.region] = (
                            self.stats.results_by_region.get(tq.region, 0)
                            + len(results))

                except Exception as e:
                    self.stats.errors.append(f"Search failed for {tq.region or tq.target_language}: {e}")

                completed += 1
                progress = 20 + (60 * completed / len(futures))
                if tq.region:
                    label = REGION_LOCALE_MAP.get(tq.region, {}).get(
                        "name", tq.region)
                else:
                    label = LANGUAGE_REGION_MAP.get(
                        tq.target_language, {}).get("name", tq.target_language)
                self._report_progress(f"Searched {label}", progress)

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

    @staticmethod
    def _canonical_url(url):
        parsed = urlsplit(url)
        parameters = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
                      if not key.lower().startswith("utm_") and key.lower() not in ("fbclid", "gclid")]
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/",
                           urlencode(sorted(parameters)), ""))

    def _deduplicate_results(self, results):
        """Deduplicate URL identity while retaining independent source origins.

        A generic title or search snippet is insufficient to declare two primary
        documents identical. Preserve discovery locales when one URL is merged.
        """
        by_url = {}
        for result in results:
            key = self._canonical_url(result.url)
            discovery = {"language": result.source_language, "region": result.region,
                         "query": result.search_query, "rank": result.search_rank}
            if key in by_url:
                existing = by_url[key]
                existing.metadata.setdefault("discovered_in", []).append(discovery)
                existing.search_rank = min(existing.search_rank, result.search_rank)
            else:
                result.metadata.setdefault("discovered_in", []).append(discovery)
                by_url[key] = result
        return list(by_url.values())

    def _score_results(self, results):
        """Rank by lexical relevance, original SERP rank and locale preference."""
        token_cache = {}
        for result in results:
            if result.search_query not in token_cache:
                token_cache[result.search_query] = extract_keywords(result.search_query, max_keywords=20)
            terms = token_cache[result.search_query]
            text = (result.title + " " + result.snippet).casefold()
            overlap = sum(term.casefold() in text for term in terms) / max(1, len(terms))
            rank_score = 1.0 / max(1, result.search_rank)
            weight = self.config.get_language_weight(result.source_language)
            result.relevance_score = weight * (0.55 + 0.15 * rank_score + 0.30 * overlap)
            if result.region and self.config.prefer_local_sources and self._is_local_domain(result.url, result.region):
                result.relevance_score += self.config.local_source_boost
        return sorted(results, key=lambda result: (-result.relevance_score, result.search_rank,
                                                  self._canonical_url(result.url)))

    @staticmethod
    def _is_local_domain(url: str, region: str) -> bool:
        """True when the URL's host ends with the region's country TLD."""
        tld = REGION_LOCALE_MAP.get(region, {}).get("tld", "")
        if not tld or not url:
            return False
        m = re.match(r"https?://([^/]+)", url)
        host = (m.group(1) if m else url).split(":")[0].lower()
        return host.endswith(tld)

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

        if self.stats.results_by_region:
            lines.append("#### Results by Region (locale search)")
            lines.append("| Region | Count |")
            lines.append("|--------|-------|")
            for region, count in sorted(self.stats.results_by_region.items(),
                                        key=lambda x: x[1], reverse=True):
                name = REGION_LOCALE_MAP.get(region, {}).get("name", region)
                lines.append(f"| {name} | {count} |")
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
