"""
AI Crawler - LLM-driven crawling ("aicrawl" mode).

Self-built crawler with no external crawler dependencies: pages are fetched
through the existing requests-based ``search_client.get_page_content()``
interface, and an LLM reads each fetched page to decide which links are
worth following next.

Flow per section:
1. Seed a priority frontier from web search results for the section queries.
2. Pop the highest-priority URL, fetch it, apply the content filter.
3. One LLM call per page decides: page relevance, a summary to keep as
   evidence, which candidate links to enqueue (with priority), and optional
   new search queries when the trail runs dry.
4. When the frontier empties with budget remaining, re-seed from the
   LLM-suggested queries (bounded rounds).
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin

from ..evidence.content_filter import ContentFilter, create_moderate_filter
from ..utils.helpers import extract_json_from_response
from .fast_crawler import CrawledPage, CrawlResult, EvaluatedPage
from .site_crawler import (
    extract_keywords_from_topic,
    get_domain,
    is_valid_page_url,
    normalize_url,
    score_relevance_simple,
)


@dataclass
class LinkCandidate:
    """A link found on a fetched page, offered to the LLM as a candidate."""
    url: str  # absolute, normalized
    anchor_text: str = ""


@dataclass
class AICrawlDecision:
    """The LLM's judgement about one fetched page."""
    relevance_score: float = 0.0
    key_points: List[str] = field(default_factory=list)
    processed_content: str = ""
    follow_links: List[Tuple[LinkCandidate, float]] = field(default_factory=list)  # (link, priority 0-1)
    suggested_queries: List[str] = field(default_factory=list)
    used_fallback: bool = False


class AICrawler:
    """
    LLM-driven crawler: the LLM reads each page and steers the crawl.

    Unlike SiteCrawler (same-domain BFS) this follows links across domains,
    ordered by LLM-assigned priority, under page/depth/domain/LLM-call
    budgets.
    """

    # How many links from a page are offered to the LLM
    DEFAULT_MAX_LINK_CANDIDATES = 20
    # Priority decay per depth level so seeds outrank deep speculation
    DEPTH_DECAY = 0.85

    def __init__(
        self,
        search_client,
        llm_client,
        content_filter: ContentFilter = None,
        max_total_pages: int = 15,
        max_depth: int = 3,
        max_site_depth: int = 2,
        max_llm_calls: int = 25,
        max_pages_per_domain: int = 5,
        politeness_delay: float = 1.0,
        max_link_candidates: int = DEFAULT_MAX_LINK_CANDIDATES,
        max_reseed_rounds: int = 2,
        language: str = "ja",
        fetch_client=None,
    ):
        """
        Initialize AICrawler.

        Args:
            search_client: Web search client (used for search; also for page
                fetching unless fetch_client is given)
            llm_client: LLM client for crawl decisions
            content_filter: Content filter for ads/spam removal
            max_total_pages: Fetch budget per crawl_and_evaluate call
            max_depth: Maximum link depth from search-result seeds
            max_site_depth: Maximum layers followed WITHIN one site (domain);
                resets when the crawl crosses to a different domain
            max_llm_calls: Budget of LLM decision calls per crawl
            max_pages_per_domain: Cap of fetched pages per domain
            politeness_delay: Minimum seconds between fetches to the same domain
            max_link_candidates: Max links per page offered to the LLM
            max_reseed_rounds: Max re-seeding rounds from suggested queries
            language: Language for decision prompts
            fetch_client: Optional dedicated client for page fetching
                (e.g. a Selenium browser); defaults to search_client
        """
        self.search = search_client
        self.fetch = fetch_client or search_client
        self.llm = llm_client
        self.content_filter = content_filter or create_moderate_filter()
        self.max_total_pages = max_total_pages
        self.max_depth = max_depth
        self.max_site_depth = max_site_depth
        self.max_llm_calls = max_llm_calls
        self.max_pages_per_domain = max_pages_per_domain
        self.politeness_delay = politeness_delay
        self.max_link_candidates = max_link_candidates
        self.max_reseed_rounds = max_reseed_rounds
        self.language = language

    def _fetch_page(self, url: str):
        """Fetch a page via the fetch client. Subclasses may override."""
        return self.fetch.get_page_content(url)

    def crawl_and_evaluate(
        self,
        queries: List[str],
        section_context: str,
        research_topic: str = "",
        max_pages_per_query: int = 3,
        min_relevance_score: float = 0.2,
        progress_callback: Callable[[str, int, int], None] = None,
    ) -> CrawlResult:
        """
        Run an LLM-driven crawl for the given queries.

        Same signature as FastCrawler.crawl_and_evaluate so the Researcher
        call site is symmetric.

        Returns:
            CrawlResult whose pages are EvaluatedPage instances
        """
        start_time = time.time()
        keywords = extract_keywords_from_topic(research_topic or section_context)

        # Crawl state
        visited: set = set()
        # Frontier entries: (-priority, seq, url, depth, site_depth, anchor, query)
        frontier: List[Tuple[float, int, str, int, int, str, str]] = []
        seq = 0
        domain_counts: Dict[str, int] = {}
        domain_last_fetch: Dict[str, float] = {}
        llm_calls = 0
        reseeds = 0
        used_queries = set(queries)
        pending_queries: List[str] = []

        kept_pages: List[EvaluatedPage] = []
        errors: List[str] = []
        pages_fetched = 0
        pages_filtered = 0
        pages_evaluated = 0
        total_eval_time = 0.0

        def push(url: str, depth: int, site_depth: int, priority: float,
                 anchor: str, query: str) -> None:
            nonlocal seq
            heapq.heappush(frontier, (-priority, seq, url, depth, site_depth, anchor, query))
            seq += 1

        def seed_from_queries(seed_queries: List[str]) -> None:
            for query in seed_queries:
                try:
                    results = self.search.search(query, max_results=max_pages_per_query)
                except Exception as e:
                    errors.append(f"Search error for '{query}': {e}")
                    continue
                for result in results[:max_pages_per_query]:
                    url = normalize_url(result.url)
                    if url and url not in visited:
                        push(url, 0, 0, 1.0, getattr(result, "title", "") or "", query)

        if progress_callback:
            progress_callback("aicrawl: searching seed pages...", 0, self.max_total_pages)

        seed_from_queries(queries)

        while frontier and pages_fetched < self.max_total_pages:
            neg_priority, _, url, depth, site_depth, anchor, query = heapq.heappop(frontier)

            if url in visited:
                continue
            if depth > self.max_depth:
                continue
            if site_depth > self.max_site_depth:
                continue
            if not is_valid_page_url(url):
                continue
            domain = get_domain(url)
            if domain_counts.get(domain, 0) >= self.max_pages_per_domain:
                continue

            visited.add(url)
            self._politeness_wait(domain, domain_last_fetch)

            try:
                page = self._fetch_page(url)
            except Exception as e:
                errors.append(f"Fetch error for {url}: {e}")
                continue

            domain_last_fetch[domain] = time.time()
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            pages_fetched += 1

            title = page.title or anchor
            content = page.text_content or ""

            if progress_callback:
                progress_callback(
                    f"aicrawl: [{pages_fetched}/{self.max_total_pages}] {url[:60]}",
                    pages_fetched,
                    self.max_total_pages,
                )

            # Content filter (before spending an LLM call)
            if self.content_filter:
                filter_result = self.content_filter.filter_content(
                    url=url, title=title, content=content,
                )
                if not filter_result.should_include:
                    pages_filtered += 1
                    continue

            candidates = self._candidate_links(page, url, visited, domain_counts)

            # LLM decision (or keyword fallback when budget/parse fails)
            eval_start = time.time()
            if llm_calls < self.max_llm_calls:
                llm_calls += 1
                decision = self._decide(
                    url=url,
                    title=title,
                    content=content,
                    candidates=candidates,
                    research_topic=research_topic,
                    keywords=keywords,
                    section_context=section_context,
                    pages_remaining=self.max_total_pages - pages_fetched,
                )
            else:
                decision = self._fallback_decision(
                    title, content, candidates, research_topic, keywords,
                )
            total_eval_time += time.time() - eval_start
            pages_evaluated += 1

            # Keep as evidence if relevant enough
            if decision.relevance_score >= min_relevance_score:
                kept_pages.append(EvaluatedPage(
                    url=url,
                    title=title,
                    snippet=(content[:200] if content else ""),
                    content=content,
                    relevance_score=decision.relevance_score,
                    processed_content=decision.processed_content or content[:1000],
                    key_points=decision.key_points,
                    metadata={
                        "query": query,
                        "depth": depth,
                        "site_depth": site_depth,
                        "decided_by": "fallback" if decision.used_fallback else "llm",
                    },
                ))

            # Enqueue LLM-selected links with depth-dampened priority.
            # site_depth counts layers within the current domain and resets
            # to 1 when the crawl crosses to a different domain.
            for link, priority in decision.follow_links:
                if link.url not in visited:
                    dampened = priority * (self.DEPTH_DECAY ** depth)
                    link_site_depth = (
                        site_depth + 1 if get_domain(link.url) == domain else 1
                    )
                    push(link.url, depth + 1, link_site_depth, dampened,
                         link.anchor_text, query)

            for suggested in decision.suggested_queries:
                if suggested and suggested not in used_queries:
                    pending_queries.append(suggested)

            # Dead-end re-seeding: frontier drained but budget remains
            if (
                not frontier
                and pages_fetched < self.max_total_pages
                and pending_queries
                and reseeds < self.max_reseed_rounds
            ):
                reseeds += 1
                reseed_batch = []
                while pending_queries and len(reseed_batch) < 2:
                    q = pending_queries.pop(0)
                    if q not in used_queries:
                        used_queries.add(q)
                        reseed_batch.append(q)
                if reseed_batch:
                    if progress_callback:
                        progress_callback(
                            f"aicrawl: re-seeding from suggested queries: {reseed_batch}",
                            pages_fetched,
                            self.max_total_pages,
                        )
                    seed_from_queries(reseed_batch)

        total_time = time.time() - start_time

        if progress_callback:
            progress_callback(
                f"aicrawl complete: {len(kept_pages)} relevant pages "
                f"({pages_fetched} fetched, {llm_calls} LLM calls)",
                self.max_total_pages,
                self.max_total_pages,
            )

        return CrawlResult(
            pages=kept_pages,
            total_fetch_time=total_time - total_eval_time,
            total_eval_time=total_eval_time,
            pages_fetched=pages_fetched,
            pages_filtered=pages_filtered,
            pages_evaluated=pages_evaluated,
            errors=errors,
        )

    def _politeness_wait(self, domain: str, domain_last_fetch: Dict[str, float]) -> None:
        """Wait so that fetches to the same domain are spaced by politeness_delay."""
        if self.politeness_delay <= 0:
            return
        last = domain_last_fetch.get(domain)
        if last is None:
            return
        elapsed = time.time() - last
        if elapsed < self.politeness_delay:
            time.sleep(self.politeness_delay - elapsed)

    def _candidate_links(
        self,
        page,
        base_url: str,
        visited: set,
        domain_counts: Dict[str, int],
    ) -> List[LinkCandidate]:
        """Build the list of candidate links offered to the LLM."""
        raw_links: List[Dict[str, str]] = []
        if getattr(page, "links", None):
            raw_links = page.links
        elif getattr(page, "html_content", None):
            # Fallback: regex extraction from raw HTML (cross-domain allowed)
            import re
            href_pattern = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
            for href in href_pattern.findall(page.html_content)[:100]:
                if href.startswith(("#", "javascript:", "mailto:", "tel:")):
                    continue
                raw_links.append({"url": href, "text": ""})

        candidates: List[LinkCandidate] = []
        seen: set = set()
        for link in raw_links:
            href = (link.get("url") or "").strip()
            if not href:
                continue
            absolute = urljoin(base_url, href)
            if not absolute.startswith("http"):
                continue
            normalized = normalize_url(absolute)
            if normalized in visited or normalized in seen:
                continue
            if not is_valid_page_url(normalized):
                continue
            if domain_counts.get(get_domain(normalized), 0) >= self.max_pages_per_domain:
                continue
            seen.add(normalized)
            candidates.append(LinkCandidate(
                url=normalized,
                anchor_text=(link.get("text") or "").strip()[:100],
            ))
            if len(candidates) >= self.max_link_candidates:
                break

        return candidates

    def _decide(
        self,
        url: str,
        title: str,
        content: str,
        candidates: List[LinkCandidate],
        research_topic: str,
        keywords: List[str],
        section_context: str,
        pages_remaining: int,
    ) -> AICrawlDecision:
        """One LLM call: evaluate the page and pick links to follow."""
        prompt = self._build_decision_prompt(
            url, title, content, candidates,
            research_topic, keywords, section_context, pages_remaining,
        )

        try:
            response = self.llm.generate(prompt)
            data = extract_json_from_response(response.content)
        except Exception:
            return self._fallback_decision(
                title, content, candidates, research_topic, keywords,
            )

        try:
            relevance = float(data.get("relevance_score", 0.0))
        except (TypeError, ValueError):
            relevance = 0.0
        relevance = min(max(relevance, 0.0), 1.0)

        follow_links: List[Tuple[LinkCandidate, float]] = []
        for entry in (data.get("follow_links") or [])[:5]:
            if not isinstance(entry, dict):
                continue
            try:
                index = int(entry.get("index", 0))
                priority = float(entry.get("priority", 0.5))
            except (TypeError, ValueError):
                continue
            if 1 <= index <= len(candidates):
                follow_links.append(
                    (candidates[index - 1], min(max(priority, 0.0), 1.0))
                )

        key_points = [
            str(p) for p in (data.get("key_points") or []) if p
        ][:5]
        suggested = [
            str(q) for q in (data.get("suggested_queries") or []) if q
        ][:3]

        return AICrawlDecision(
            relevance_score=relevance,
            key_points=key_points,
            processed_content=str(data.get("processed_content") or ""),
            follow_links=follow_links,
            suggested_queries=suggested,
        )

    def _build_decision_prompt(
        self,
        url: str,
        title: str,
        content: str,
        candidates: List[LinkCandidate],
        research_topic: str,
        keywords: List[str],
        section_context: str,
        pages_remaining: int,
    ) -> str:
        """Build the per-page decision prompt."""
        links_text = "\n".join(
            f"{i}. [{c.anchor_text or '(アンカーテキストなし)' if self.language == 'ja' else c.anchor_text or '(no anchor text)'}] {c.url}"
            for i, c in enumerate(candidates, 1)
        ) or ("（リンク候補なし）" if self.language == "ja" else "(no candidate links)")
        keywords_str = ", ".join(keywords[:10])
        excerpt = content[:2500]

        if self.language == "ja":
            return f"""あなたはWeb調査を行うリサーチアシスタントです。今読んでいるページを評価し、次にたどるべきリンクを選んでください。

【調査テーマ】{research_topic}
【重要キーワード】{keywords_str}
【執筆中のセクション】{section_context}
【残り取得可能ページ数】{pages_remaining}ページ（残りが少ない場合は、確実に有益なリンクだけを選ぶこと）

【現在のページ】
URL: {url}
タイトル: {title}
本文（抜粋）:
{excerpt}

【このページ内のリンク候補】
{links_text}

以下を1つのJSONで回答してください:
{{
  "relevance_score": 0.0から1.0の数値,
  "processed_content": "調査テーマに関連する内容の要約（200〜500文字。関連が薄ければ空文字）",
  "key_points": ["要点1", "要点2"],
  "follow_links": [
    {{"index": リンク番号, "priority": 0.0から1.0の数値, "reason": "たどる理由を簡潔に"}}
  ],
  "suggested_queries": ["このページから判明した、新たに検索すべきクエリ"]
}}

判断基準:
- relevance_score 0.8以上: 調査テーマに直接関連する具体的情報がある／0.3未満: 無関係・広告・一覧ページのみ
- follow_links は「調査テーマの情報が得られる見込み」で選ぶこと。ナビゲーション・利用規約・SNSリンクは選ばない。最大5件、有望なリンクがなければ空配列
- suggested_queries は行き詰まりそうな場合のみ。最大3件
- 残りページ数が5以下なら follow_links は最大2件に絞る

JSONのみを出力:"""

        return f"""You are a research assistant crawling the web. Evaluate the current page and choose which links to follow next.

Research topic: {research_topic}
Key keywords: {keywords_str}
Section being written: {section_context}
Page fetch budget remaining: {pages_remaining} (if low, pick only clearly valuable links)

Current page:
URL: {url}
Title: {title}
Content (excerpt):
{excerpt}

Candidate links on this page:
{links_text}

Respond with a single JSON object:
{{
  "relevance_score": number between 0.0 and 1.0,
  "processed_content": "summary of topic-relevant content (200-500 chars; empty string if barely relevant)",
  "key_points": ["point 1", "point 2"],
  "follow_links": [
    {{"index": link number, "priority": number between 0.0 and 1.0, "reason": "brief reason"}}
  ],
  "suggested_queries": ["new search queries discovered from this page"]
}}

Guidelines:
- relevance_score >= 0.8: page has concrete information directly about the topic; < 0.3: unrelated, ads, or index-only pages
- Pick follow_links by expected topical value. Skip navigation, terms-of-service, and social links. Max 5; empty array if none look promising
- suggested_queries only when the trail looks like a dead end. Max 3
- If fewer than 5 pages remain in the budget, pick at most 2 follow_links

Output JSON only:"""

    def _fallback_decision(
        self,
        title: str,
        content: str,
        candidates: List[LinkCandidate],
        research_topic: str,
        keywords: List[str],
    ) -> AICrawlDecision:
        """Keyword-based decision when the LLM response cannot be used."""
        relevance = score_relevance_simple(
            content, title, research_topic or "", keywords,
        )

        follow_links: List[Tuple[LinkCandidate, float]] = []
        terms = [kw.lower() for kw in keywords] + [
            w.lower() for w in (research_topic or "").split() if len(w) > 1
        ]
        for candidate in candidates:
            haystack = f"{candidate.anchor_text} {candidate.url}".lower()
            if any(term in haystack for term in terms):
                follow_links.append((candidate, 0.4))
            if len(follow_links) >= 3:
                break

        return AICrawlDecision(
            relevance_score=relevance,
            processed_content=content[:500] if relevance >= 0.2 else "",
            follow_links=follow_links,
            used_fallback=True,
        )
