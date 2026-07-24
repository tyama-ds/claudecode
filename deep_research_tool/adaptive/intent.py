"""
Query intent classification + intent-templated queries + internal RRF.

STRICTLY LOCAL: classification is deterministic keyword scoring, query
templates are static strings, and Reciprocal Rank Fusion runs over
result lists returned by the EXISTING search clients (DuckDuckGo /
Selenium / requests). No external search, embedding or reranking API is
called, ever.
"""

import re
from typing import Dict, Iterable, List, Optional, Sequence

QUERY_INTENTS = (
    "definition",        # 用語・概念の定義
    "background",        # 経緯・概要
    "primary_source",    # 官公庁・一次資料
    "quantitative",      # 数値・統計・推移
    "recent_update",     # 最新動向
    "comparison",        # 比較
    "counterevidence",   # 反証・リスク
    "multi_hop",         # 複数エンティティの関係
)

# deterministic keyword rules, checked in priority order
_RULES = (
    ("quantitative", re.compile(
        r"数値|統計|推移|規模|シェア|割合|金額|億|兆|台数|件数|人口|前年比|"
        r"成長率|CAGR|market size|statistics|figures|growth rate|share",
        re.IGNORECASE)),
    ("recent_update", re.compile(
        r"最新|直近|現時点|今年|本年|足元|動向|トレンド|20\d{2}年時点|"
        r"latest|recent|current|up[- ]?to[- ]?date|trend", re.IGNORECASE)),
    ("primary_source", re.compile(
        r"一次情報|一次資料|官公庁|政府|省庁|白書|統計局|有価証券報告書|"
        r"決算|プレスリリース|official|government|primary source|"
        r"press release|annual report", re.IGNORECASE)),
    ("comparison", re.compile(
        r"比較|対比|優劣|違い|差異|どちら|vs\.?|versus|compare|difference",
        re.IGNORECASE)),
    ("counterevidence", re.compile(
        r"反証|反論|リスク|課題|問題点|懸念|批判|限界|デメリット|失敗|"
        r"counter|risk|criticism|challenge|limitation|drawback",
        re.IGNORECASE)),
    ("definition", re.compile(
        r"とは|定義|意味|何か|概念|what is|definition|meaning",
        re.IGNORECASE)),
    ("multi_hop", re.compile(
        r"影響|波及|因果|関係|経由|つながり|もたらす|impact|effect on|"
        r"relationship|causal", re.IGNORECASE)),
)


def classify_intent(text: str) -> str:
    """Deterministic intent for one requirement/question text."""
    text = text or ""
    for intent, pattern in _RULES:
        if pattern.search(text):
            return intent
    return "background"


# intent -> query suffix templates (ja / other)
_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "definition": {"ja": ["とは", "定義"], "en": ["definition", "overview"]},
    "background": {"ja": ["概要", "経緯"], "en": ["background", "overview"]},
    "primary_source": {"ja": ["統計 官公庁", "白書 政府"],
                       "en": ["official statistics", "government report"]},
    "quantitative": {"ja": ["統計 数値", "市場規模 推移"],
                     "en": ["statistics data", "market size trend"]},
    "recent_update": {"ja": ["最新 動向", "2025 2026 動向"],
                      "en": ["latest news", "recent developments"]},
    "comparison": {"ja": ["比較", "違い"], "en": ["comparison", "versus"]},
    "counterevidence": {"ja": ["課題 リスク", "批判 問題点"],
                        "en": ["criticism risks", "challenges limitations"]},
    "multi_hop": {"ja": ["影響 関係", "因果"],
                  "en": ["impact relationship", "causal effect"]},
}


def build_intent_queries(requirement_text: str, intent: str = "",
                         language: str = "ja",
                         max_queries: int = 3) -> List[str]:
    """Deterministic search queries for one requirement.

    The base query is the requirement's head (trimmed); intent-specific
    suffixes diversify the search WITHOUT changing the search client.
    """
    base = re.sub(r"\s+", " ", (requirement_text or "").strip())[:60]
    if not base:
        return []
    intent = intent or classify_intent(requirement_text)
    lang_key = "ja" if language == "ja" else "en"
    suffixes = _TEMPLATES.get(intent, _TEMPLATES["background"])[lang_key]
    queries = [base]
    for suffix in suffixes:
        queries.append(f"{base} {suffix}")
    return queries[:max_queries]


def rrf_merge(result_lists: Sequence[Sequence], k: int = 60,
              key: Optional[callable] = None,
              limit: Optional[int] = None) -> List:
    """Reciprocal Rank Fusion over N ranked result lists (INTERNAL).

    ``key(result)`` identifies duplicates across lists (defaults to the
    result's ``url`` attribute / ["url"] entry / str()). Ties break by
    first-seen order for determinism. This fuses lists produced by the
    EXISTING local search clients only.
    """
    if key is None:
        def key(r):
            url = getattr(r, "url", None)
            if url is None and isinstance(r, dict):
                url = r.get("url")
            return url or str(r)

    scores: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    items: Dict[str, object] = {}
    counter = 0
    for results in result_lists:
        for rank, result in enumerate(results or []):
            ident = key(result)
            if ident not in items:
                items[ident] = result
                first_seen[ident] = counter
                counter += 1
            scores[ident] = scores.get(ident, 0.0) + 1.0 / (k + rank + 1)

    ordered = sorted(items,
                     key=lambda i: (-scores[i], first_seen[i]))
    merged = [items[i] for i in ordered]
    return merged[:limit] if limit else merged
