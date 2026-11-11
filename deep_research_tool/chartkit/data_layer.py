"""
Data layer: the numerical data store of a run → per-section catalog of
labelled series with a certainty tag and sources on every value.

Deterministic Python only (no LLM). This is where the old pipeline went
wrong: it plotted whatever string keys each page-level extraction
happened to emit, mixed scales (billion / million / raw) in one series,
and let code-fabricated values (N×N "market shares", interpolated years)
in as if they were data. Here:

- entity / metric names are canonicalised before grouping (legal forms,
  katakana / Latin spellings and value-checked prefix aliases of one
  company fold into one entity; a metric's scope — 国内 / 世界 / 連結 — stays
  part of its key), but every name a reader sees keeps the sources' most
  frequent spelling ('Toray', 'GPa', never 'toray', 'gpa');
- an annual value and a quarterly / half-year / monthly value of one
  metric are never one series;
- every value is re-expressed in ONE display scale per series, and a
  value that had to be re-scaled is tagged 換算;
- for one (entity, metric, year, unit) the best-sourced point wins and
  the others are kept as alternatives (never averaged, never summed);
- derived values (CAGR / YoY) are 導出, interpolated ones 推定, values from
  a primary source (public statistics, regulators, company IR / press
  releases) 公表, everything else 二次.

The output has the shape of the handoff's ``data.json``: label arrays plus
series objects ``{unit, values, basis, source, note}`` referenced by key.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

BASIS_PUBLISHED = "公表"
BASIS_CONVERTED = "換算"
BASIS_DERIVED = "導出"
BASIS_ESTIMATED = "推定"
BASIS_SECONDARY = "二次"
BASIS_TAGS = (BASIS_PUBLISHED, BASIS_CONVERTED, BASIS_DERIVED,
              BASIS_ESTIMATED, BASIS_SECONDARY)

# Sources whose numbers count as 公表 (primary): public statistics,
# regulators, securities filings, academia, and company IR / press pages.
DEFAULT_PRIMARY_DOMAINS: Tuple[str, ...] = (
    ".go.jp", ".gov", ".gov.uk", ".gc.ca", ".gov.au", "europa.eu",
    ".ac.jp", ".edu", "oecd.org", "worldbank.org", "imf.org", "un.org",
    "who.int", "e-stat.go.jp", "sec.gov", "edinet-fsa.go.jp",
    "prtimes.jp", "jpx.co.jp",
)
# enhanced style: hosts whose numbers are 二次 even on an IR-looking path
# (market research firms publish their own estimates; PR wires carry
# whatever the sender wrote)
SECONDARY_DOMAINS: Tuple[str, ...] = (
    "prtimes.jp", "prnewswire.com", "businesswire.com", "globenewswire.com",
    "einpresswire.com", "openpr.com", "newswire.com", "atpress.ne.jp", "dreamnews.jp",
    "marketsandmarkets.com", "grandviewresearch.com", "imarcgroup.com",
    "researchandmarkets.com", "fortunebusinessinsights.com", "mordorintelligence.com",
    "precedenceresearch.com", "alliedmarketresearch.com", "persistencemarketresearch.com",
    "skyquestt.com", "technavio.com", "statista.com", "marketresearchfuture.com",
    "transparencymarketresearch.com", "factmr.com", "futuremarketinsights.com",
    "verifiedmarketresearch.com", "databridgemarketresearch.com", "coherentmarketinsights.com",
    "reportlinker.com", "giiresearch.com", "gii.co.jp", "yano.co.jp", "fuji-keizai.co.jp",
    "seedplanning.co.jp",
)
PRIMARY_PATH_HINTS: Tuple[str, ...] = (
    "/ir/", "/investor", "/investors", "/press-release", "/press_release",
    "/pressrelease", "/news-release", "/newsroom", "/earnings", "/results",
    "/annual-report", "/financial",
)

# scale words inside a unit string → (factor, canonical scale label)
_SCALES: Tuple[Tuple[str, float, str], ...] = (
    # compound Japanese scales first ('千億円' is 1e11 yen, not 億 + '千円')
    ("1000億", 1e11, "千億"),
    ("千億", 1e11, "千億"),
    ("百億", 1e10, "百億"),
    ("100億", 1e10, "百億"),
    ("1000万", 1e7, "千万"),
    ("千万", 1e7, "千万"),
    ("100万", 1e6, "百万"),
    ("十万", 1e5, "十万"),
    ("10万", 1e5, "十万"),
    ("兆", 1e12, "兆"),
    ("trillion", 1e12, "兆"),
    ("十億", 1e9, "10億"),
    ("10億", 1e9, "10億"),
    ("billion", 1e9, "10億"),
    ("bn", 1e9, "10億"),
    ("億", 1e8, "億"),
    ("百万", 1e6, "百万"),
    ("million", 1e6, "百万"),
    ("mn", 1e6, "百万"),
    ("thousand", 1e3, "千"),
    ("千", 1e3, "千"),
    ("万", 1e4, "万"),
)
# every scale written in a unit, longest first (to strip it in one piece)
_SCALE_ALT = "|".join(re.escape(w) for w, _f, _l in sorted(_SCALES, key=lambda t: -len(t[0]))
                      if not w.isascii() or w[0].isdigit())
_PER_POP = re.compile(r"(?:/|人口|per\s*)\s*(?:[0-9][0-9,.]*\s*(?:(?:十|百|千)?万|千|百)?|(?:(?:十|百|千)?万|千|百))"
                      r"\s*(?:人|世帯)?(?:\s*(?:当たり|あたり))?(?![0-9])"
                      r"|[0-9,.]*\s*(?:(?:十|百|千)?万|千)\s*(?:人|世帯)?\s*(?:当たり|あたり|対)")
_SCALE_LABEL_FACTOR = {"兆": 1e12, "千億": 1e11, "百億": 1e10, "10億": 1e9, "億": 1e8,
                       "千万": 1e7, "百万": 1e6, "十万": 1e5, "万": 1e4, "千": 1e3, "": 1.0}
_CURRENCY_WORDS = {
    "usd": "ドル", "us$": "ドル", "$": "ドル", "dollar": "ドル", "dollars": "ドル",
    "ドル": "ドル", "米ドル": "ドル", "jpy": "円", "yen": "円", "円": "円", "¥": "円",
    "eur": "ユーロ", "euro": "ユーロ", "€": "ユーロ", "ユーロ": "ユーロ",
    "gbp": "ポンド", "£": "ポンド", "cny": "元", "rmb": "元", "元": "元",
}


def canon(text: Optional[str]) -> str:
    """Canonical form for grouping: NFKC, lower, no quotes/brackets,
    single spaces, trailing 'market'/'市場' particles kept (they carry
    meaning) but English articles dropped."""
    s = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    s = re.sub(r"[\"'“”‘’()（）\[\]【】]", "", s)
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


# ----------------------------------------------------------------------
# grouping keys vs what readers see
# ----------------------------------------------------------------------
# canon() and the keys below only GROUP points. Every name a reader (or
# the chart-design LLM) sees is the spelling the sources used most often
# for that key (ties: the first seen) — 'Toray', never 'toray'.

def spelling(text: Any) -> str:
    """A name as the source wrote it (runs of white space collapsed)."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


class _Spellings:
    """Most frequent spelling per key; ties go to the first one seen."""

    def __init__(self) -> None:
        self._seen: Dict[str, Dict[str, List[int]]] = {}    # key -> spelling -> [count, order]

    def add(self, key: str, text: str) -> None:
        if not text:
            return
        per = self._seen.setdefault(key, {})
        if text in per:
            per[text][0] += 1
        else:
            per[text] = [1, len(per)]

    def get(self, key: str, default: str = "") -> str:
        per = self._seen.get(key)
        if not per:
            return default
        return min(per.items(), key=lambda kv: (-kv[1][0], kv[1][1]))[0]


# ---- metric key ------------------------------------------------------
# A parenthesis is dropped from the key ('売上高（億円）', 'Market size
# (2024)') unless it narrows WHAT is measured: a scope (国内 vs 世界 are
# different numbers and must never share one line), consolidation, a
# real / nominal basis, a growth basis or a period range.
_SCOPE_JA = ("国内", "世界", "日本", "米国", "欧州", "中国", "連結", "単体", "全世界", "グローバル",
             "海外", "北米", "南米", "中南米", "アジア", "欧米", "全国", "国外", "英国", "ドイツ",
             "インド", "韓国", "台湾", "東南アジア", "中東", "アフリカ",
             "実質", "名目", "前年比", "前年同期比", "前期比", "年平均", "累計", "累積", "税込", "税抜")
_SCOPE_EN = ("global", "domestic", "worldwide", "japan", "us", "u.s.", "usa", "u.s.a.",
             "united states", "north america", "europe", "eu", "china", "asia", "uk",
             "overseas", "consolidated", "non-consolidated", "standalone",
             "real", "nominal", "yoy", "cagr", "cumulative")
_SCOPE_EN_RE = re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(w) for w in
                                                    sorted(_SCOPE_EN, key=len, reverse=True))
                          + r")(?![a-z$])")
_PERIOD_RANGE = re.compile(r"(?:19|20)\d{2}\s*[-~〜～–—]\s*(?:(?:19|20)?\d{2})(?!\d)")
_PARENS = re.compile(r"[(（\[【]([^()（）\[\]【】]*)[)）\]】]")
# common metrics written in English → the Japanese key (key only: the
# display keeps the source's spelling)
_METRIC_SYNONYMS = {
    "market size": "市場規模",
    "revenue": "売上高", "revenues": "売上高", "sales": "売上高", "net sales": "売上高",
    "売上": "売上高", "売上げ": "売上高", "売り上げ": "売上高", "売上収益": "売上高",
    "operating profit": "営業利益", "operating income": "営業利益",
    "net income": "純利益", "net profit": "純利益", "当期純利益": "純利益",
    "shipments": "出荷量", "shipment": "出荷量", "shipment volume": "出荷量",
    "production": "生産量", "production volume": "生産量",
    "share": "シェア", "market share": "シェア", "市場シェア": "シェア",
    "growth rate": "成長率", "price": "価格", "prices": "価格",
    "capacity": "生産能力", "production capacity": "生産能力",
}
_METRIC_TARGETS = set(_METRIC_SYNONYMS.values())
# a scope word written in front of the metric ('国内市場規模', 'global
# market size') is the same qualifier as '市場規模（国内）'
_SCOPE_CANON = {"global": "世界", "worldwide": "世界", "全世界": "世界",
                "グローバル": "世界", "domestic": "国内", "japan": "日本", "us": "米国",
                "u.s.": "米国", "usa": "米国", "u.s.a.": "米国", "united states": "米国",
                "europe": "欧州", "eu": "欧州", "china": "中国", "asia": "アジア",
                "north america": "北米", "uk": "英国", "overseas": "海外",
                "consolidated": "連結", "non-consolidated": "単体", "standalone": "単体",
                "real": "実質", "nominal": "名目", "yoy": "前年比",
                "cagr": "年平均", "cumulative": "累計"}


def _scopes_in(text: str) -> List[str]:
    """Scope / basis qualifiers named in a (lower-cased NFKC) text."""
    found = []
    for w in sorted(_SCOPE_JA, key=len, reverse=True):
        if w in text:
            found.append(_SCOPE_CANON.get(w, w))
            text = text.replace(w, " ")
    for m in _SCOPE_EN_RE.finditer(text):
        found.append(_SCOPE_CANON.get(m.group(0), m.group(0)))
    pr = _PERIOD_RANGE.search(text)
    if pr:
        found.append(re.sub(r"\s+", "", pr.group(0)))
    return found


# a parenthetical that only gives a unit / scale / currency / year / source
_UNIT_WORD = (r"(?:(?:十|百|千)?(?:兆|億|万)|千|百)?\s*"
              r"(?:円|ドル|米ドル|ユーロ|元|ウォン|ポンド|台|トン|t|kg|件|人|社|個|本|基|戸|世帯|%|％|pt|ポイント"
              r"|usd|jpy|eur|cny|us\$|\$|million|billion|trillion|mn|bn|k|gwh|mwh|twh|kwh|gw|mw|kw|wh|w)")
_NOTE_PAREN = re.compile(
    r"(?:単位[:：]?\s*)?(?:[0-9,.]*\s*" + _UNIT_WORD + r"\s*)+"
    r"|(?:19|20)\d{2}(?:\s*[-~〜～–—]\s*(?:19|20)?\d{2})?\s*年?(?:度)?"
    r"|(?:出所|出典|source)[:：].*|推計|推定|概算|速報|確報|見込み?|予測|予想|見通し|実績"
    r"|forecast|projection|projected|estimate|estimated|actual")


def metric_key(name: Any) -> str:
    """The grouping key of a metric name: parentheses dropped unless they
    carry a scope ('市場規模（国内）' ≠ '市場規模（世界）'), common English
    metric names mapped to Japanese ('Market size' = '市場規模'), a leading
    scope word read as the same qualifier ('国内市場規模' = '市場規模（国内）')."""
    s = unicodedata.normalize("NFKC", str(name or "")).lower().strip()
    scopes: List[str] = []

    kept: List[str] = []

    def _paren(m: "re.Match") -> str:
        inner = m.group(1)
        sc = _scopes_in(inner)
        scopes.extend(sc)
        rest = inner
        if sc:                              # the scope words themselves are tags
            for w in sorted(_SCOPE_JA, key=len, reverse=True):
                rest = rest.replace(w, " ")
            rest = _PERIOD_RANGE.sub(" ", _SCOPE_EN_RE.sub(" ", rest))
        rest = canon(rest).strip(" ・,、/")
        # a unit, a scale, a currency or a year says nothing about WHAT is
        # measured; anything else (a segment, a product, 〜ベース, 予測,
        # 第1四半期) does, and stays part of the key
        if rest and not _NOTE_PAREN.fullmatch(rest):
            kept.append(rest)
        return " "
    prev = None
    while prev != s:                        # nested / several parentheses
        prev = s
        s = _PARENS.sub(_paren, s)
    base = canon(s)
    base = _METRIC_SYNONYMS.get(base, base)
    if base not in _METRIC_TARGETS:
        # '国内市場規模' / 'global market size': scope word + a known metric
        for w in sorted(set(_SCOPE_JA) | set(_SCOPE_EN), key=len, reverse=True):
            if base.startswith(w):
                if w.isascii() and not base[len(w):len(w) + 1].isspace():
                    continue                # 'user count' does not start with 'us'
                rest = base[len(w):].strip(" の・")
                rest = _METRIC_SYNONYMS.get(rest, rest)
                if rest in _METRIC_TARGETS:
                    scopes.append(_SCOPE_CANON.get(w, w))
                    base = rest
                    break
    if not base:                            # the whole name was a parenthesis ('【売上高】')
        base = canon(unicodedata.normalize("NFKC", str(name or "")).lower())
        base = _METRIC_SYNONYMS.get(base, base)
        scopes = []
        kept = []
    if kept and base:
        base = f"{base}［{'・'.join(kept)}］"
    tags = sorted(set(scopes))
    return f"{base}｜{'・'.join(tags)}" if tags and base else base


# ---- entity key --------------------------------------------------------
# Legal forms and quote brackets are folded ('トヨタ自動車株式会社' =
# 'トヨタ自動車', 'Toyota Motor Corp.' = 'Toyota Motor'); a bare 社 never
# is (A社 / B社 are different companies).
_ENTITY_QUOTES = re.compile(r"[「」『』]")
_LEGAL_PREFIX = re.compile(r"^(?:株式会社|有限会社|\(株\)|\(有\))[\s・]*")
_LEGAL_SUFFIX_JA = re.compile(r"[\s・]*(?:株式会社|有限会社|\(株\)|\(有\))$")
# グループ / ホールディングス / Group / Holdings are NOT legal forms: a group
# and its namesake company (ソフトバンクグループ / ソフトバンク, Volkswagen
# Group / Volkswagen) are different entities with different numbers
_LEGAL_SUFFIX_LATIN = re.compile(
    r"[\s,]+(?:co\.?,?\s*ltd\.?|corporation|corp\.?|inc\.?|ltd\.?|llc|plc|co\.?"
    r"|ag|s\.?a\.?)$")


def _strip_legal(text: str) -> str:
    s = text
    prev = None
    while prev != s:
        prev = s
        for pat in (_LEGAL_PREFIX, _LEGAL_SUFFIX_JA, _LEGAL_SUFFIX_LATIN):
            t = pat.sub("", s).strip(" ・,&")
            if t:                           # never strip a name down to nothing
                s = t
    return s


def entity_base_key(subject: Any) -> str:
    """The grouping key of an entity name before alias merging: NFKC,
    lower case, legal forms and 「」『』 dropped, then canon()."""
    s = unicodedata.normalize("NFKC", str(subject or "")).lower().strip()
    s = _ENTITY_QUOTES.sub("", s)
    return canon(_strip_legal(s))


# katakana → Hepburn (a small table; enough to tell トヨタ = Toyota)
_KANA = dict(zip(
    "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
    "ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポヴ",
    ["a", "i", "u", "e", "o", "ka", "ki", "ku", "ke", "ko", "sa", "shi", "su", "se", "so",
     "ta", "chi", "tsu", "te", "to", "na", "ni", "nu", "ne", "no", "ha", "hi", "fu", "he", "ho",
     "ma", "mi", "mu", "me", "mo", "ya", "yu", "yo", "ra", "ri", "ru", "re", "ro", "wa", "o", "n",
     "ga", "gi", "gu", "ge", "go", "za", "ji", "zu", "ze", "zo", "da", "ji", "zu", "de", "do",
     "ba", "bi", "bu", "be", "bo", "pa", "pi", "pu", "pe", "po", "vu"]))
_SMALL_VOWEL = {"ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o"}
_SMALL_Y = {"ャ": "a", "ュ": "u", "ョ": "o"}
_KATAKANA_NAME = re.compile(r"[ァ-ー\s]+")
_LATIN_NAME = re.compile(r"[a-z][a-z .&'\-]*")
_VOWELS = "aeiou"


def romanise(kata: str) -> Optional[str]:
    """Hepburn romanisation of a katakana-only name ('トヨタ' → 'toyota',
    'ソニー' → 'sonii': ー repeats the vowel, ッ doubles the consonant);
    None for anything that is not katakana."""
    s = unicodedata.normalize("NFKC", str(kata or "")).strip()
    if not s or not _KATAKANA_NAME.fullmatch(s):
        return None
    out: List[str] = []
    double = False
    for ch in s:
        if ch in " ・":
            continue
        if ch == "ッ":
            double = True
            continue
        if ch == "ー":
            last = next((c for c in reversed("".join(out)) if c in _VOWELS), "")
            out.append(last)
            continue
        if ch in _SMALL_Y and out:
            prev = out.pop()
            out.append((prev[:-1] if prev.endswith(("shi", "chi", "ji")) else prev[:-1] + "y")
                       + _SMALL_Y[ch])
            continue
        if ch in _SMALL_VOWEL and out:
            prev = out.pop()
            stem = {"u": "w", "i": "y"}.get(prev, prev[:-1] if prev[-1:] in _VOWELS else prev)
            out.append(stem + _SMALL_VOWEL[ch])
            continue
        roman = _KANA.get(ch) or _SMALL_VOWEL.get(ch) or _SMALL_Y.get(ch)
        if roman is None:
            return None
        if double and roman[0] not in _VOWELS:
            roman = ("t" if roman.startswith("ch") else roman[0]) + roman     # ッチ = tchi
        double = False
        out.append(roman)
    return "".join(out) or None


@lru_cache(maxsize=4096)
def _phonetic(s: str) -> str:
    """One spelling for sounds katakana cannot tell apart (l/r, v/b, c/k,
    y/i, sh/s …) and no doubled letters or long vowels."""
    s = re.sub(r"[^a-z]", "", s.lower())
    for a, b in (("ph", "f"), ("sh", "s"), ("ch", "t"), ("ck", "k"), ("qu", "kw"), ("x", "ks"),
                 ("c", "k"), ("q", "k"), ("l", "r"), ("v", "b"), ("y", "i"), ("z", "s")):
        s = s.replace(a, b)
    s = re.sub(r"(.)\1+", r"\1", s)
    return s.replace("ou", "o")


def _kana_matches_latin(kana_roman: str, latin: str) -> bool:
    """The Latin name is the romanised katakana with only the vowels that
    katakana adds after a consonant dropped ('tesura' → 'tesla',
    'panasonikku' → 'panasonic')."""
    k, lat = _phonetic(kana_roman), _phonetic(latin)
    # only vowels after the first letter can be dropped
    if len(lat) < 3 or not k or len(lat) > len(k) or k[0] != lat[0]:
        return False
    memo: Dict[Tuple[int, int], bool] = {}

    def droppable(i: int) -> bool:
        return k[i] in "uoi" and i > 0 and k[i - 1] not in _VOWELS

    def go(i: int, j: int) -> bool:
        if (i, j) in memo:
            return memo[(i, j)]
        if j == len(lat):
            ok = all(droppable(t) for t in range(i, len(k)))
        elif i == len(k):
            ok = False
        else:
            ok = (k[i] == lat[j] and go(i + 1, j + 1)) or (droppable(i) and go(i + 1, j))
        memo[(i, j)] = ok
        return ok
    return go(0, 0)


def _word_prefixes(key: str) -> List[str]:
    """Proper prefixes of a Latin name that end at a word boundary
    ('toyota motor' → ['toyota'])."""
    words = key.split(" ")
    return [" ".join(words[:n]) for n in range(1, len(words))]


def _prefix_alias(short: str, long: str, roman: Optional[str] = None,
                  prefixes: Optional[List[str]] = None) -> bool:
    """Is ``short`` a shorter way of writing ``long`` ('トヨタ' / 'トヨタ自動車',
    'toyota' / 'toyota motor', 'トヨタ' / 'toyota motor')? ``roman`` (the
    romanised ``short``) and ``prefixes`` (``long``'s word prefixes) may be
    passed in precomputed."""
    if not short or len(long) <= len(short):
        return False
    if _LATIN_NAME.fullmatch(long):
        prefixes = _word_prefixes(long) if prefixes is None else prefixes
        if short in prefixes:
            return len(short) >= 3
        roman = romanise(short) if roman is None else roman
        return bool(roman) and any(_kana_matches_latin(roman, p) for p in prefixes)
    if not long.startswith(short) or len(short) < 2:
        return False
    nxt = long[len(short)]
    if short[-1].isascii() and short[-1].isalnum() and nxt.isascii() and nxt.isalnum():
        return False                        # 'sony' is not a prefix of 'sonya'
    return True


def _close(a: float, b: float, tol: float = 0.15) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b))


def resolve_entity_aliases(cells: Dict[str, Dict[Any, float]]) -> Dict[str, str]:
    """Alias groups of entity base keys → one key per group (the smallest,
    so the result never depends on the input order).

    ``cells[key]`` holds the key's value per (metric, unit, granularity,
    period) in base units. A katakana name and a Latin name whose
    romanisation matches are one entity (トヨタ / Toyota, テスラ / Tesla).
    A prefix alias (トヨタ / トヨタ自動車, Toyota / Toyota Motor) is merged
    only when the short name has that one candidate, the extra words are
    not a complement (米国 / 米国以外), and the amounts both report agree
    (two within 15 %, or the only one within 3 %; rates and shares do not
    count) — 'Toyota' and 'Toyota Tsusho' are different companies.""" 
    keys = sorted(k for k in cells if k)
    parent = {k: k for k in keys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    kana = {k: r for k in keys if (r := romanise(k))}
    latin = [k for k in keys if _LATIN_NAME.fullmatch(k)]
    for k, roman in kana.items():
        for lat in latin:
            if _kana_matches_latin(roman, lat):
                union(k, lat)
    prefixes = {k: _word_prefixes(k) for k in latin}
    for a in keys:
        agreeing = []
        for b in keys:
            if len(b) <= len(a) or not _prefix_alias(a, b, kana.get(a, ""), prefixes.get(b)):
                continue
            extra = b[len(a):] if b.startswith(a) else b
            if _COMPLEMENT.search(extra):   # 米国 / 米国以外, excluding …
                continue
            # rates and shares are no evidence (two different companies
            # easily grow 12 % and 11 %); at least two amounts must agree
            # within 15 %, or the only shared one within 3 %
            common = [c for c in set(cells[a]) & set(cells[b]) if not _is_rate_cell(c)]
            if not common:
                continue
            tol = 0.15 if len(common) >= 2 else 0.03
            if all(_close(cells[a][c], cells[b][c], tol) for c in common):
                agreeing.append(b)
        # a short name that agrees with several longer ones (トヨタ:
        # トヨタ自動車 / トヨタ紡織) joins none — and never chains them
        if len(agreeing) == 1:
            union(a, agreeing[0])
    return {k: find(k) for k in keys}


_COMPLEMENT = re.compile(r"以外|を除く|除く|除き|その他|ほか|excluding|excl\.?|(?<![a-z])ex-|non-"
                         r"|other than|others?|rest of")


def _is_rate_cell(cell: Any) -> bool:
    metric = str(cell[0]) if isinstance(cell, tuple) and cell else ""
    unit = str(cell[1]) if isinstance(cell, tuple) and len(cell) > 1 else ""
    return (unit in ("%", "％", "pt", "ポイント", "倍", "percent")
            or any(w in metric for w in ("率", "シェア", "share", "rate", "growth", "margin")))


def parse_unit(unit: Optional[str]) -> Tuple[str, float, str]:
    """Split a unit string into (base unit, scale factor, scale label).

    'billion USD' → ('ドル', 1e9, '10億'); '百万缶' → ('缶', 1e6, '百万');
    '%' → ('%', 1, ''); 'GPa' → ('gpa', 1, '') (SI prefixes are part of the
    unit, not a presentation scale). The base is lower-cased: it is a
    GROUPING key. ``unit_spelling`` gives the same base as the source
    wrote it ('GPa'), which is what readers see.
    """
    base, _shown, factor, label = _parse_unit(unit)
    return base, factor, label


def unit_spelling(unit: Optional[str]) -> str:
    """parse_unit's base unit in the source's own spelling: 'GPa', 'MW',
    '百万kWh' → 'kWh' (the scale is still split off; currency words and
    '%' are normalised exactly as in parse_unit)."""
    return _parse_unit(unit)[1]


def _parse_unit(unit: Optional[str]) -> Tuple[str, str, float, str]:
    """(base key, base as spelled, scale factor, scale label)."""
    raw = unicodedata.normalize("NFKC", str(unit or "")).strip()
    spelled = raw.lower()
    # the scale is looked up without thousands commas ('1,000万円' → 千万);
    # a unit with no scale keeps its own spelling ('1,000 tons')
    low = re.sub(r"(?<=\d),(?=\d{3})", "", spelled)
    cased = re.sub(r"(?<=\d),(?=\d{3})", "", raw)

    def _shown(key: str, text: str) -> str:
        # str.lower() can change a string's length; the key is the fallback
        return text if text.lower() == key else key
    # a per-population denominator is part of the unit, not a scale
    # ('件（人口10万人当たり）', '人/10万人')
    if _PER_POP.search(low):
        base = spelled.strip().rstrip(" -_/・")      # '/10万人' keeps its '/'
        return base, _shown(base, raw.strip().rstrip(" -_/・")), 1.0, ""
    factor, label, hit = 1.0, "", ""
    for word, f, lab in _SCALES:
        # longest match first is guaranteed by the table order
        if word in low:
            factor, label, hit = f, lab, word
            break
    if hit:
        base = low.replace(hit, "", 1).strip()
        i = low.find(hit)
        shown = (cased[:i] + cased[i + len(hit):]).strip() if len(cased) == len(low) else base
    else:
        base, shown = spelled, raw
    base, shown = base.strip(" -_/・"), shown.strip(" -_/・")
    if base in _CURRENCY_WORDS:
        base = shown = _CURRENCY_WORDS[base]
    if base in ("％", "percent", "pct", "パーセント"):
        base = shown = "%"
    return base, _shown(base, shown), factor, label


# scale words as they appear next to a number in the ORIGINAL text
_RAW_SCALES: Tuple[Tuple[str, float], ...] = (
    ("trillion", 1e12), ("billion", 1e9), ("million", 1e6), ("thousand", 1e3),
    ("bn", 1e9), ("mn", 1e6), ("tn", 1e12), ("b", 1e9), ("m", 1e6), ("k", 1e3),
    ("兆", 1e12), ("億", 1e8), ("百万", 1e6), ("万", 1e4), ("千", 1e3),
)
# English scale words only as whole words ('BNPL', 'MNOs', 'billionaires'
# are not scales)
_RAW_NUMBER = re.compile(
    r"([0-9]+(?:[,，][0-9]{3})*(?:[.．][0-9]+)?)\s*"
    r"(trillion(?![a-z])|billion(?![a-z])|million(?![a-z])|thousand(?![a-z])|bn(?![a-z])|mn(?![a-z])"
    r"|tn(?![a-z])|兆|億|百万|万|千)?", re.I)
# Japanese compound numbers: 1兆2,000億 / 3億5,000万 / 1万2500 / 3千万 / 1千億
_JP_UNITS = (("兆", 1e12), ("千億", 1e11), ("百億", 1e10), ("十億", 1e9), ("億", 1e8),
             ("千万", 1e7), ("百万", 1e6), ("十万", 1e5), ("万", 1e4), ("千", 1e3))
_JP_PART = r"[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?\s*(?:" + "|".join(u for u, _f in _JP_UNITS) + ")"
_JP_COMPOUND = re.compile(r"(?:%s\s*)+(?:[0-9]+(?:,[0-9]{3})*(?![0-9.,]*\s*[A-Za-z]))?" % _JP_PART)
# 'bn' / 'mn' / 'tn' rebuilt into a unit must be a word parse_unit knows
_SCALE_WORD = {"bn": "billion", "mn": "million", "tn": "trillion", "b": "billion", "m": "million",
               "k": "thousand"}


def _jp_compounds(text: str) -> List[float]:
    """Totals of Japanese compound numbers ('1兆2,000億' = 1.2e12)."""
    out = []
    for m in _JP_COMPOUND.finditer(text):
        chunk = m.group(0)
        parts = re.findall(r"([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(" + "|".join(u for u, _f in _JP_UNITS) + ")?",
                           chunk)
        units = dict(_JP_UNITS)
        if sum(1 for _n, u in parts if u) < 1 or len(parts) < 2 and not any(
                u in ("千万", "千億", "百億", "十億", "十万") for _n, u in parts):
            continue                    # a plain '50億' is handled as a single number
        total, last = 0.0, float("inf")
        ok = True
        for n, u in parts:
            f = units.get(u, 1.0)
            if f >= last:               # scales must go down: '1兆2,000億'
                ok = False
                break
            total += float(n.replace(",", "")) * f
            last = f
        if ok:
            out.append(total)
    return out


# 'k' as an SI prefix written apart from its unit ('10k ohm', '4.7k Ω')
_SI_UNIT_AFTER = re.compile(r"\s*(?:Ω|ω|ohms?(?![a-z])|v(?![a-z])|w(?![a-z])|hz(?![a-z])|wh(?![a-z])"
                            r"|va(?![a-z])|var(?![a-z]))", re.I)


def _raw_numbers(raw_text: str) -> List[Tuple[float, float, str]]:
    """(number, scale factor, scale word) for every number in a text."""
    return [(n, f, w) for n, f, w, _s, _e in raw_number_spans(raw_text)]


def raw_number_spans(raw_text: str) -> List[Tuple[float, float, str, int, int]]:
    """``_raw_numbers`` with each number's span in the NFKC text (the end
    is after its scale word, if any)."""
    text = unicodedata.normalize("NFKC", str(raw_text or ""))
    out = []
    for m in _RAW_NUMBER.finditer(text):
        try:
            num = float(m.group(1).replace(",", "").replace("，", "").replace("．", "."))
        except ValueError:
            continue
        word = (m.group(2) or "").lower()
        # a compound like 千万 / 百億 is not a plain scale: leave it alone
        nxt = text[m.end():m.end() + 1]
        if word in ("千", "百万") and nxt in ("万", "億"):
            word = "?"
        if not word and text[m.end():m.end() + 2] in ("百億", "十億", "十万"):
            word = "?"                  # '3百億' is 300億 (read as a compound)
        if not word:
            # '$1.2B' / 'US$450m' / 'USD 1.2B' / '1.5k': a scale letter stuck
            # to the number (B / M / T only after a currency: '3M' may be a
            # company); never a letter that starts a unit ('10kΩ', '5km')
            letter, after = text[m.end(1):m.end(1) + 1], text[m.end(1) + 1:m.end(1) + 2]
            if letter and letter.lower() in "bmkt" and not (after.isalpha() and ord(after) < 0x3000):
                money = re.search(r"(?:[$¥€£]|\b(?:usd|jpy|eur|gbp|cny|us\$))\s*$",
                                  text[max(0, m.start() - 6):m.start()], re.I)
                if letter == "k" and _SI_UNIT_AFTER.match(text, m.end(1) + 1):
                    pass                        # '10k ohm' is 10 kΩ, not 10,000 ohm
                elif letter == "k" or (money and letter.lower() in "bmkt"):
                    word = {"b": "b", "m": "m", "k": "k", "t": "tn"}[letter.lower()]
        factor = dict(_RAW_SCALES).get(word, 1.0) if word != "?" else float("nan")
        end = m.end() if m.group(2) else (m.end(1) + 1 if word in ("b", "m", "k", "tn") else m.end(1))
        out.append((num, factor, word, m.start(1), end))
    return out


def repair_scale(value: Any, raw_text: str, unit: str) -> Tuple[str, bool]:
    """Fix a unit whose SCALE contradicts the original text.

    The LLM sometimes copies the number of '50億ドル' (value 50) but writes
    the unit of a different scale ('billion USD'), which plots the value
    10x off. When value × unit scale already equals a number × scale of
    the original text, the point is consistent (a correct conversion such
    as 5 billion USD for 50億ドル stays as it is). When the value equals a
    number of the text but the unit's scale is not that number's scale
    word, the unit is rebuilt with the text's scale word. Anything else is
    left alone. Returns (unit, changed)."""
    try:
        v = abs(float(value))           # the text's numbers carry no sign
    except (TypeError, ValueError):
        return unit, False
    if not raw_text or v == 0:
        return unit, False
    base, unit_factor, _label = parse_unit(unit)
    nums = _raw_numbers(raw_text)
    if not nums:
        return unit, False

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= 0.005 * max(abs(a), abs(b))
    if any(f == f and close(v * unit_factor, n * f) for n, f, _w in nums):
        return unit, False                          # consistent (or a correct conversion)
    for n, f, word in nums:
        if f != f or not word or not close(v, n):   # NaN: compound scale; no word: nothing to fix
            continue
        if f == unit_factor:
            return unit, False
        if base == "%":
            return unit, False
        if not base:
            # the unit is a scale word alone ('billion'): the text's scale word
            return _SCALE_WORD.get(word, word), True
        if re.fullmatch(r"[a-z]+", word):
            # English scale word: keep the unit's original currency spelling
            word = _SCALE_WORD.get(word, word)
            cur = re.sub(r"(?i)\b(trillion|billion|million|thousand|bn|mn|tn)\b|" + _SCALE_ALT, "",
                         unicodedata.normalize("NFKC", str(unit or ""))).strip() or base
            return f"{word} {cur}", True
        return f"{word}{base}", True
    return unit, False


def coerce_year(value: Any) -> Optional[int]:
    """A year as int (2024, '2024', '2024年', 'FY2024') or None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1900 <= value <= 2100 else None
    m = re.search(r"(19|20)\d{2}", str(value))
    return int(m.group(0)) if m else None


# ----------------------------------------------------------------------
# period granularity: an annual value and a quarterly / half-year /
# monthly value of the same metric are different series, never one line
# ----------------------------------------------------------------------

GRAN_ANNUAL, GRAN_HALF, GRAN_QUARTER, GRAN_MONTH = "annual", "half", "quarter", "month"
_GRAN_JA = {GRAN_HALF: "半期", GRAN_QUARTER: "四半期", GRAN_MONTH: "月次"}
_KANJI_DIGIT = {"一": 1, "二": 2, "三": 3, "四": 4,
                "first": 1, "second": 2, "third": 3, "fourth": 4}
_Q_MARKERS = (
    re.compile(r"(?<![a-z])q([1-4])(?![0-9])"),                 # Q1 / 2024Q1 / FY2024 Q2
    re.compile(r"(?<![0-9])([1-4])q(?![a-z])"),                 # 1Q / 1Q24
    re.compile(r"第?\s*([1-4一二三四])\s*四半期"),                  # 第1四半期
    re.compile(r"(?<![0-9a-z])([1-4]|first|second|third|fourth)(?:st|nd|rd|th)?\s+quarter"),
)
# '1-3月', '1月～3月', '2024年4月～2025年3月' (a 12-month span = a fiscal year)
_MONTH_RANGE = re.compile(r"(?<![0-9])(1[0-2]|0?[1-9])\s*月?\s*[-~〜～–—]\s*"
                          r"(?:(?:19|20)\d{2}\s*[年/.\-]\s*)?(1[0-2]|0?[1-9])\s*月")
_MONTH_WORD_RANGE = re.compile(
    r"(?<![a-z])(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*[-–~〜～—]\s*"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?(?![a-z])")
# cumulative spans that are neither one quarter nor a year: 'Q1-Q3',
# '1-3Q', 'YTD', '1-9月'
_QUARTER_SPAN = re.compile(r"q[1-4]\s*[-~〜～–—]\s*q?[1-4]|(?<![0-9])[1-4]\s*[-~〜～–—]\s*[1-4]\s*q(?![a-z])")
_PARTIAL_YEAR = re.compile(r"(?<![a-z])ytd(?![a-z])|year[- ]to[- ]date")
# a fiscal-year quarter ('2025年3月期第1四半期' = Apr-Jun 2024) is not the
# calendar quarter with that number
_FISCAL_QUARTER = re.compile(r"月期\s*(?:の)?\s*第?\s*[1-4一二三四]\s*四半期|fy\s*\d{2,4}\s*q[1-4]")
_H_MARKERS = (
    (re.compile(r"上半期|上期|中間期|前半期|first\s+half|1st\s+half"), 1),
    (re.compile(r"下半期|下期|後半期|second\s+half|2nd\s+half"), 2),
    (re.compile(r"(?<![a-z])h([12])(?![0-9])"), None),          # H1 / 2024H1
    (re.compile(r"(?<![0-9])([12])h(?![a-z])"), None),          # 1H / 1H24
)
_MONTH_MARKERS = (
    re.compile(r"(?:19|20)\d{2}\s*[/.\-]\s*(1[0-2]|0?[1-9])(?![0-9])"),   # 2024/03, 2024-3
    # 3月 — not 3月期 / 3月決算 (a fiscal year that ends in March)
    re.compile(r"(?<![0-9])(1[0-2]|0?[1-9])\s*月(?!\s*(?:期|決算|末決算|締))"),
)
# a fiscal-year text names its closing month without being monthly
# ('FY ending March 2024', '2024年度（3月決算）')
_FISCAL = re.compile(r"(?<![a-z])fy|fiscal|年度|決算|[0-9]\s*月?\s*期(?!間)")
# a cumulative quarter ('第3四半期累計' = nine months) is not a quarter
_CUMULATIVE = re.compile(r"累計|累積|ytd|year[- ]to[- ]date|cumulative")
_MONTH_NAMES = {m: i + 1 for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"))}
_MONTH_WORD = re.compile(r"(?<![a-z])(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
                         r"(?:uary|ruary|ch|il|e|y|ust|tember|t|ober|ember)?\.?(?![a-z])")
_GRAN_WORDS = ((GRAN_QUARTER, re.compile(r"四半期|quarter|qtr")),
               (GRAN_HALF, re.compile(r"半期|half[- ]?year|semi[- ]?annual")),
               (GRAN_MONTH, re.compile(r"月次|monthly")))


def parse_period(text: Any) -> Tuple[str, Optional[int]]:
    """(granularity, index) of a period text: ('quarter', 1) for '2024Q1',
    '1Q', '第1四半期', '1-3月期'; ('half', 2) for '下期', 'H2'; ('month', 3)
    for '2024/03', '2024年3月'; ('annual', None) for '2024', 'FY2024',
    '2024年度', '2024年3月期' (a fiscal year). The index is None when the
    text names a sub-annual period without saying which one ('四半期',
    '4-9月期' — not a calendar half)."""
    s = unicodedata.normalize("NFKC", str(text or "")).lower()
    if not s.strip():
        return GRAN_ANNUAL, None
    if _QUARTER_SPAN.search(s) or _PARTIAL_YEAR.search(s):
        return GRAN_QUARTER, None           # a partial year: no axis can place it
    if _FISCAL_QUARTER.search(s) and "fy" not in s[:2]:
        return GRAN_QUARTER, None           # fiscal quarter: not a calendar quarter
    for pat in _Q_MARKERS:
        m = pat.search(s)
        if m:
            if _CUMULATIVE.search(s):
                return GRAN_QUARTER, None   # Q1-Q3 cumulative: neither a quarter nor a year
            d = m.group(1)
            return GRAN_QUARTER, _KANJI_DIGIT.get(d) or int(d)
    m = _MONTH_RANGE.search(s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        span = (b - a) % 12 + 1
        if span == 3:
            return GRAN_QUARTER, (a + 2) // 3 if a in (1, 4, 7, 10) else None
        if span == 6:
            return GRAN_HALF, {1: 1, 7: 2}.get(a)
        if span == 1:
            return GRAN_MONTH, a
        if span == 12:
            return GRAN_ANNUAL, None
        return GRAN_MONTH, None             # '1-2月': a sub-annual span that is no period here
    m = _MONTH_WORD_RANGE.search(s)
    if m:
        a, b = _MONTH_NAMES[m.group(1)], _MONTH_NAMES[m.group(2)]
        span = (b - a) % 12 + 1
        if span == 3:
            return GRAN_QUARTER, (a + 2) // 3 if a in (1, 4, 7, 10) else None
        if span == 6:
            return GRAN_HALF, {1: 1, 7: 2}.get(a)
        if span == 12:
            return GRAN_ANNUAL, None
        return GRAN_MONTH, None
    for pat, idx in _H_MARKERS:
        m = pat.search(s)
        if m:
            return GRAN_HALF, idx if idx is not None else int(m.group(1))
    if not _FISCAL.search(s):
        for pat in _MONTH_MARKERS:
            m = pat.search(s)
            if m:
                return GRAN_MONTH, int(m.group(1))
        m = _MONTH_WORD.search(s)
        if m:
            return GRAN_MONTH, _MONTH_NAMES[m.group(1)]
    for gran, pat in _GRAN_WORDS:
        if pat.search(s):
            return gran, None
    return GRAN_ANNUAL, None


def quarter_of(text: Any) -> Optional[str]:
    """'Q1'..'Q4' when a year / period text names a quarter ('2024Q1',
    '2024年第1四半期', 'FY2024 Q2', '1-3月期'), else None."""
    gran, idx = parse_period(text)
    return f"Q{idx}" if gran == GRAN_QUARTER and idx else None


def _host_in(host: str, domains: Sequence[str]) -> bool:
    for dom in domains:
        d = dom.lower().lstrip(".")
        # label boundary: "oecd.org" matches "www.oecd.org", not "notoecd.org"
        if host == d or host.endswith("." + d):
            return True
    return False


def classify_source_basis(url: Optional[str],
                          primary_domains: Sequence[str] = DEFAULT_PRIMARY_DOMAINS,
                          secondary_domains: Sequence[str] = ()) -> str:
    """公表 for a primary source, 二次 for everything else (no URL = 二次).
    A host in ``secondary_domains`` is 二次 whatever the other rules say."""
    if not url:
        return BASIS_SECONDARY
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
    except Exception:
        return BASIS_SECONDARY
    host = (parsed.hostname or "").lower()       # no login, no port
    path = (parsed.path or "").lower()
    if secondary_domains and _host_in(host, secondary_domains):
        return BASIS_SECONDARY
    if _host_in(host, primary_domains):
        return BASIS_PUBLISHED
    if any(h in path for h in PRIMARY_PATH_HINTS):
        return BASIS_PUBLISHED
    return BASIS_SECONDARY


@dataclass
class Series:
    key: str
    kind: str                       # "time" | "comparison"
    labels_ref: str
    name: str
    metric: str
    base_unit: str
    unit: str                       # display unit incl. scale, e.g. 10億ドル
    scale_factor: float
    values: List[Optional[float]]
    basis: List[Optional[str]]
    sources: List[Dict[str, Any]]   # per value index: {url, title, evidence_id, raw_text, alternatives}
    entity: str = ""
    year: Optional[int] = None
    note: str = ""

    @property
    def n_points(self) -> int:
        return sum(1 for v in self.values if v is not None)

    def source_summary(self) -> str:
        titles = []
        for s in self.sources:
            if not s:
                continue
            t = s.get("title") or s.get("url") or ""
            if t and t not in titles:
                titles.append(t)
        return "、".join(titles[:3]) + ("…" if len(titles) > 3 else "")

    def to_data(self) -> Dict[str, Any]:
        return {
            "unit": self.unit,
            # what this series is (the legend name check uses it)
            "name": self.name,
            "kind": self.kind,
            "entity": self.entity,
            "year": self.year,
            "labels_ref": self.labels_ref,     # the label array these values belong to
            "values": list(self.values),
            "basis": list(self.basis),
            "source": self.source_summary(),
            "note": self.note,
            "sources": [dict(s) if s else None for s in self.sources],
        }


@dataclass
class SectionCatalog:
    section_id: str
    labels: Dict[str, List[str]] = field(default_factory=dict)
    series: Dict[str, Series] = field(default_factory=dict)
    dropped: List[str] = field(default_factory=list)      # why points were left out
    # yield counters: why the chapter's points did (not) become series
    stats: Dict[str, int] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.series

    def to_data(self) -> Dict[str, Any]:
        """The handoff's data.json shape: label arrays + series objects."""
        data: Dict[str, Any] = {"_meta": {"section": self.section_id,
                                          "basis_tags": list(BASIS_TAGS)}}
        data.update({k: list(v) for k, v in self.labels.items()})
        data.update({k: s.to_data() for k, s in self.series.items()})
        return data

    def catalog_for_prompt(self, max_items: int = 40) -> List[Dict[str, Any]]:
        """What the LLM sees: keys, meaning, unit, labels, certainty kinds,
        source count — never the values."""
        items = []
        ordered = sorted(self.series.values(),
                         key=lambda s: (-s.n_points, s.kind, s.key))
        for s in ordered[:max_items]:
            kinds = sorted({b for b in s.basis if b})
            items.append({
                "values_ref": s.key,
                "labels_ref": s.labels_ref,
                "kind": "時系列" if s.kind == "time" else "比較",
                "name": s.name,
                "metric": s.metric,
                "unit": s.unit,
                "labels": list(self.labels.get(s.labels_ref, [])),
                "points": s.n_points,
                "basis": kinds,
                "sources": len({(x or {}).get("url") for x in s.sources if x}),
            })
        return items


# ----------------------------------------------------------------------
# building
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SourcePolicy:
    """How a point's source is judged. classic: the v1.5 rules; enhanced:
    research firms / PR wires are 二次 and an actual value wins over a
    forecast for the same year."""
    primary_domains: Tuple[str, ...] = DEFAULT_PRIMARY_DOMAINS
    secondary_domains: Tuple[str, ...] = ()
    prefer_actual: bool = False

    @classmethod
    def for_style(cls, style: str, primary_domains: Sequence[str] = DEFAULT_PRIMARY_DOMAINS
                  ) -> "SourcePolicy":
        if style == "enhanced":
            return cls(tuple(primary_domains), SECONDARY_DOMAINS, True)
        return cls(tuple(primary_domains))


def _policy(primary_domains) -> SourcePolicy:
    if isinstance(primary_domains, SourcePolicy):
        return primary_domains
    return SourcePolicy(tuple(primary_domains))


def _point_basis(dp, primary_domains) -> str:
    raw_text = str(getattr(dp, "raw_text", "") or "")
    if getattr(dp, "is_derived", False):
        if raw_text.lower().startswith("interpolated"):
            return BASIS_ESTIMATED
        return BASIS_DERIVED
    pol = _policy(primary_domains)
    return classify_source_basis(getattr(dp, "source_url", ""), pol.primary_domains,
                                 pol.secondary_domains)


_BASIS_RANK = {BASIS_PUBLISHED: 0, BASIS_SECONDARY: 1, BASIS_CONVERTED: 1,
               BASIS_DERIVED: 2, BASIS_ESTIMATED: 3}


def _year_of(dp) -> Optional[int]:
    y = getattr(dp, "year", None)
    if y is None:
        return None
    try:
        m = re.search(r"(19|20)\d{2}", str(y))
        return int(m.group(0)) if m else None
    except Exception:
        return None


def _period_of(dp) -> Tuple[str, Optional[str], bool]:
    """(granularity, time label, period known) of a point. The quarter
    field and the period markers of its date / period text decide the
    granularity: '2024' (annual), '2024Q1', '2024H2', '2024-03'. A text
    that names a sub-annual period without saying which one ('四半期')
    is not known (the point cannot be placed on any axis)."""
    y = _year_of(dp)
    gran, idx = GRAN_ANNUAL, None
    q = str(getattr(dp, "quarter", "") or "").strip()
    if re.fullmatch(r"[1-4]", q):
        gran, idx = GRAN_QUARTER, int(q)
    else:
        texts = [q, getattr(dp, "period", None), getattr(dp, "date_context", None)]
        if isinstance(getattr(dp, "year", None), str):
            texts.append(getattr(dp, "year"))
        for text in texts:
            g, i = parse_period(text)
            if g != GRAN_ANNUAL:
                gran, idx = g, i
                break
    if gran == GRAN_ANNUAL:
        return gran, (str(y) if y is not None else None), True
    if idx is None:
        return gran, None, False
    if y is None:
        return gran, None, True
    label = {GRAN_QUARTER: f"{y}Q{idx}", GRAN_HALF: f"{y}H{idx}",
             GRAN_MONTH: f"{y}-{idx:02d}"}[gran]
    return gran, label, True


def _time_label(dp) -> Optional[str]:
    return _period_of(dp)[1]


def _pick_display_scale(points: List[Tuple[float, str]]) -> str:
    """The scale most values were reported in; ties → the larger scale."""
    counts: Dict[str, int] = {}
    for _v, label in points:
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(),
                  key=lambda kv: (-kv[1], -_SCALE_LABEL_FACTOR.get(kv[0], 1.0)))[0][0]


def _display_unit(base_unit: str, scale_label: str) -> str:
    if base_unit == "%":
        return "%"
    return f"{scale_label}{base_unit}" if scale_label else base_unit


def _choose(points: List[Any], primary_domains) -> Any:
    """Best point for one (entity, metric, year, unit): primary source,
    then not derived, then higher confidence, then earlier insertion
    (enhanced: an actual value before any forecast for that year)."""
    pol = _policy(primary_domains)

    def rank(dp):
        basis = _point_basis(dp, pol)
        conf = float(getattr(dp, "combined_confidence", 0.0) or 0.0)
        forecast = 1 if (pol.prefer_actual and getattr(dp, "is_forecast", False)) else 0
        # the URL breaks ties so the choice never depends on the
        # (parallel) extraction order
        return (forecast, _BASIS_RANK.get(basis, 9), -conf, str(getattr(dp, "source_url", "") or ""))
    return sorted(points, key=rank)[0]


def _labels_ref_for(cat: "SectionCatalog", labels: List[str], default: str) -> str:
    """Series with the same label array share one labels_ref (so the LLM
    can put them in one panel); a new array is stored under ``default``."""
    for ref, existing in cat.labels.items():
        if list(existing) == list(labels):
            return ref
    cat.labels[default] = list(labels)
    return default


def build_section_catalog(points: Iterable[Any], section_id: str,
                          section_evidence_ids: Optional[Iterable[str]] = None,
                          primary_domains: Any = DEFAULT_PRIMARY_DOMAINS,
                          min_points_time: int = 2,
                          min_entities: int = 2,
                          max_series: int = 60) -> SectionCatalog:
    """Turn the store's points that belong to ``section_id`` into labelled
    series. ``section_evidence_ids`` (the locker's section map) also claims
    points of a document that several chapters use: extraction records
    only the FIRST chapter that cited the page, the locker knows them all.
    ``primary_domains`` is a domain list or a :class:`SourcePolicy`.

    Points are grouped by (metric key, base unit, period granularity) and
    entity key (``metric_key`` / ``entity_base_key`` + aliases); series
    names, labels and units are the sources' most frequent spellings."""
    primary_domains = _policy(primary_domains)
    evidence_ids = set(section_evidence_ids or [])
    cat = SectionCatalog(section_id=section_id)
    mine = []
    for dp in points:
        sid = str(getattr(dp, "section_id", "") or "")
        eid = str(getattr(dp, "evidence_id", "") or "")
        if sid == section_id or (eid and eid in evidence_ids):
            mine.append(dp)
    st = cat.stats
    st.update({"points": len(mine), "no_metric": 0, "period_unknown": 0, "time_too_short": 0,
               "comparison_too_few": 0, "granularity_split": 0, "series": 0, "capped": 0})
    if not mine:
        return cat

    # keys (grouping only) and the spellings readers see
    metric_names, unit_names, entity_names = _Spellings(), _Spellings(), _Spellings()
    rows = []
    for dp in mine:
        metric = metric_key(getattr(dp, "metric_name", ""))
        if not metric:
            cat.dropped.append(f"{getattr(dp, 'raw_text', '')!r}: 指標名なし")
            st["no_metric"] += 1
            continue
        gran, tl, known = _period_of(dp)
        if not known:
            # '四半期' without saying which one: no axis can place it
            cat.dropped.append(f"{getattr(dp, 'raw_text', '')!r}: {_GRAN_JA.get(gran, gran)}の時期が特定できない")
            st["period_unknown"] += 1
            continue
        base_unit, unit_shown, _f, _lab = _parse_unit(getattr(dp, "unit", ""))
        rows.append((dp, metric, base_unit, gran, tl, entity_base_key(getattr(dp, "subject", ""))))
        metric_names.add(metric, spelling(getattr(dp, "metric_name", "")))
        unit_names.add(base_unit, unit_shown)

    # entity aliases (legal forms, katakana / Latin, value-checked prefixes)
    cell_points: Dict[str, Dict[Tuple[str, str, str, str], List[Any]]] = {}
    for dp, metric, base_unit, gran, tl, ent in rows:
        if ent:
            per = cell_points.setdefault(ent, {})
            if tl:
                per.setdefault((metric, base_unit, gran, tl), []).append(dp)
    cells = {ent: {c: _base_value(_choose(pts, primary_domains)) for c, pts in per.items()}
             for ent, per in cell_points.items()}
    alias = resolve_entity_aliases(cells)

    # bucket: (metric, base_unit, granularity) -> entity -> time label -> [points]
    # (an annual and a quarterly value of one metric are never one series)
    buckets: Dict[Tuple[str, str, str], Dict[str, Dict[Optional[str], List[Any]]]] = {}
    grans: Dict[Tuple[str, str, str], set] = {}
    for dp, metric, base_unit, gran, tl, ent in rows:
        entity = alias.get(ent, ent)
        entity_names.add(entity, spelling(getattr(dp, "subject", "")))
        buckets.setdefault((metric, base_unit, gran), {}).setdefault(entity, {}).setdefault(tl, []).append(dp)
        grans.setdefault((metric, base_unit, entity), set()).add(gran)
    st["granularity_split"] = sum(1 for g in grans.values() if len(g) > 1)

    n = 0
    for (metric, base_unit, gran), by_entity in sorted(buckets.items()):
        shown_metric = metric_names.get(metric, metric)
        shown_unit = unit_names.get(base_unit, base_unit)
        gran_note = f"（{_GRAN_JA[gran]}）" if gran in _GRAN_JA else ""
        # ---- time series: one per entity with >= min_points_time labels
        for entity, by_time in sorted(by_entity.items()):
            labels = sorted(t for t in by_time if t)
            if len(labels) < min_points_time:
                st["time_too_short"] += 1
                continue
            chosen = {t: _choose(by_time[t], primary_domains) for t in labels}
            scale = _pick_display_scale([(0.0, parse_unit(getattr(c, "unit", ""))[2]) for c in chosen.values()])
            shown_entity = entity_names.get(entity, entity)
            series = _series_from(chosen, labels, kind="time", metric=shown_metric, base_unit=base_unit,
                                  scale_label=scale, entity=shown_entity, primary_domains=primary_domains,
                                  unit_shown=shown_unit)
            n += 1
            key = f"ts{n}"
            labels_ref = _labels_ref_for(cat, labels, f"years_{key}")
            series.key, series.labels_ref = key, labels_ref
            series.name = _display_name(shown_entity, shown_metric, "") + gran_note
            cat.series[key] = series
            st["series"] = len(cat.series)
            if len(cat.series) >= max_series:
                st["capped"] = 1
                return cat
        # ---- comparisons: one per time label with >= min_entities entities
        by_time_all: Dict[Optional[str], Dict[str, List[Any]]] = {}
        for entity, by_time in by_entity.items():
            if not entity:
                continue
            for tl, pts in by_time.items():
                by_time_all.setdefault(tl, {}).setdefault(entity, []).extend(pts)
        for tl, by_entity2 in sorted(by_time_all.items(), key=lambda kv: (kv[0] is None, str(kv[0]))):
            if len(by_entity2) < min_entities:
                st["comparison_too_few"] += 1
                continue
            chosen = {e: _choose(pts, primary_domains) for e, pts in by_entity2.items()}
            scale = _pick_display_scale([(0.0, parse_unit(getattr(c, "unit", ""))[2]) for c in chosen.values()])
            # entities ordered by value (desc) in the display scale; the key
            # breaks ties so the order never depends on the input order
            def _val(c):
                b, f, _l = parse_unit(getattr(c, "unit", ""))
                return float(getattr(c, "value", 0.0) or 0.0) * f / _SCALE_LABEL_FACTOR.get(scale, 1.0)
            ordered = sorted(chosen.items(), key=lambda kv: (-_val(kv[1]), kv[0]))
            labels = _unique_labels([_display_entity(entity_names.get(e, e)) for e, _c in ordered])
            series = _series_from({lab: c for lab, (_e, c) in zip(labels, ordered)}, labels,
                                  kind="comparison", metric=shown_metric, base_unit=base_unit,
                                  scale_label=scale, entity="", primary_domains=primary_domains,
                                  unit_shown=shown_unit)
            n += 1
            key = f"cmp{n}"
            labels_ref = _labels_ref_for(cat, labels, f"entities_{key}")
            series.key, series.labels_ref = key, labels_ref
            series.year = int(tl[:4]) if tl else None
            series.name = _display_name("", shown_metric, tl or "") + ("" if tl else gran_note)
            cat.series[key] = series
            st["series"] = len(cat.series)
            if len(cat.series) >= max_series:
                st["capped"] = 1
                return cat
    return cat


def _base_value(dp) -> float:
    """A point's value in base units (the scale applied)."""
    _b, f, _l = parse_unit(getattr(dp, "unit", ""))
    try:
        return float(getattr(dp, "value", 0.0) or 0.0) * f
    except (TypeError, ValueError):
        return 0.0


def _unique_labels(labels: List[str]) -> List[str]:
    """Category labels must be unique (the values are looked up by label)."""
    out: List[str] = []
    for lab in labels:
        cand, n = lab, 2
        while cand in out:
            cand = f"{lab} ({n})"
            n += 1
        out.append(cand)
    return out


def _display_entity(entity: str) -> str:
    return entity if entity else "（全体）"


def _display_name(entity: str, metric: str, when: str) -> str:
    parts = [p for p in (entity, metric, when) if p]
    return " ".join(parts)


def _series_from(chosen: Dict[str, Any], labels: List[str], *, kind: str, metric: str,
                 base_unit: str, scale_label: str, entity: str, primary_domains,
                 unit_shown: Optional[str] = None) -> Series:
    disp_factor = _SCALE_LABEL_FACTOR.get(scale_label, 1.0)
    values: List[Optional[float]] = []
    basis: List[Optional[str]] = []
    sources: List[Dict[str, Any]] = []
    notes: List[str] = []
    for label in labels:
        dp = chosen.get(label)
        if dp is None:
            values.append(None); basis.append(None); sources.append(None)
            continue
        b_unit, f, lab = parse_unit(getattr(dp, "unit", ""))
        raw_value = float(getattr(dp, "value", 0.0) or 0.0)
        if base_unit == "%":
            value = raw_value
        else:
            value = raw_value * f / disp_factor
        tag = _point_basis(dp, primary_domains)
        if base_unit != "%" and lab != scale_label and tag in (BASIS_PUBLISHED, BASIS_SECONDARY):
            tag = BASIS_CONVERTED
            notes.append(f"{label}: {getattr(dp, 'raw_text', '')} を {scale_label or '基本単位'}に換算")
        if getattr(dp, "is_forecast", False):
            notes.append(f"{label}: 予測値")
        values.append(round(value, 4))
        basis.append(tag)
        sources.append({
            "url": getattr(dp, "source_url", "") or "",
            "title": getattr(dp, "source_title", "") or "",
            "evidence_id": getattr(dp, "evidence_id", "") or "",
            "raw_text": getattr(dp, "raw_text", "") or "",
            # as extracted, before the display scale (the grounding check)
            "value": raw_value,
            "unit": getattr(dp, "unit", "") or "",
            "forecast": bool(getattr(dp, "is_forecast", False)),
        })
    return Series(
        key="", kind=kind, labels_ref="", name="", metric=metric, base_unit=base_unit,
        unit=_display_unit(unit_shown or base_unit, scale_label), scale_factor=disp_factor,
        values=values, basis=basis, sources=sources, entity=entity,
        note="；".join(notes[:6]),
    )
