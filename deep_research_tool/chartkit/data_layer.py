"""
Data layer: the numerical data store of a run → per-section catalog of
labelled series with a certainty tag and sources on every value.

Deterministic Python only (no LLM). This is where the old pipeline went
wrong: it plotted whatever string keys each page-level extraction
happened to emit, mixed scales (billion / million / raw) in one series,
and let code-fabricated values (N×N "market shares", interpolated years)
in as if they were data. Here:

- entity / metric names are canonicalised before grouping;
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
_PER_POP = re.compile(r"(?:/|人口|per\s*)\s*[0-9,.]*\s*(?:十|百|千)?万\s*(?:人|世帯)?"
                      r"|[0-9,.]*\s*(?:十|百|千)?万\s*(?:人|世帯)\s*(?:当たり|あたり)")
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


def parse_unit(unit: Optional[str]) -> Tuple[str, float, str]:
    """Split a unit string into (base unit, scale factor, scale label).

    'billion USD' → ('ドル', 1e9, '10億'); '百万缶' → ('缶', 1e6, '百万');
    '%' → ('%', 1, ''); 'GPa' → ('GPa', 1, '') (SI prefixes are part of the
    unit, not a presentation scale).
    """
    raw = unicodedata.normalize("NFKC", str(unit or "")).strip()
    low = re.sub(r"(?<=\d),(?=\d{3})", "", raw.lower())      # '1,000万円' → '1000万円'
    # a per-population denominator is part of the unit, not a scale
    # ('件（人口10万人当たり）', '人/10万人')
    if _PER_POP.search(low):
        return low.strip(" -_/・"), 1.0, ""
    factor, label, hit = 1.0, "", ""
    for word, f, lab in _SCALES:
        # longest match first is guaranteed by the table order
        if word in low:
            factor, label, hit = f, lab, word
            break
    base = low.replace(hit, "", 1).strip() if hit else low
    base = base.strip(" -_/・")
    base = _CURRENCY_WORDS.get(base, base)
    if base in ("％", "percent", "pct", "パーセント"):
        base = "%"
    return base, factor, label


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


def _raw_numbers(raw_text: str) -> List[Tuple[float, float, str]]:
    """(number, scale factor, scale word) for every number in a text."""
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
                if letter == "k" or (money and letter.lower() in "bmkt"):
                    word = {"b": "b", "m": "m", "k": "k", "t": "tn"}[letter.lower()]
        factor = dict(_RAW_SCALES).get(word, 1.0) if word != "?" else float("nan")
        out.append((num, factor, word))
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


def _time_label(dp) -> Optional[str]:
    y = _year_of(dp)
    if y is None:
        return None
    q = str(getattr(dp, "quarter", "") or "").upper().strip()
    return f"{y}{q}" if q and q.startswith("Q") else str(y)


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
    ``primary_domains`` is a domain list or a :class:`SourcePolicy`."""
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
    st.update({"points": len(mine), "no_metric": 0, "time_too_short": 0,
               "comparison_too_few": 0, "series": 0, "capped": 0})
    if not mine:
        return cat

    # bucket: (metric, base_unit) -> entity -> time label -> [points]
    buckets: Dict[Tuple[str, str], Dict[str, Dict[Optional[str], List[Any]]]] = {}
    for dp in mine:
        metric = canon(getattr(dp, "metric_name", ""))
        if not metric:
            cat.dropped.append(f"{getattr(dp, 'raw_text', '')!r}: 指標名なし")
            st["no_metric"] += 1
            continue
        base_unit, _f, _lab = parse_unit(getattr(dp, "unit", ""))
        entity = canon(getattr(dp, "subject", "")) or ""
        tl = _time_label(dp)
        buckets.setdefault((metric, base_unit), {}).setdefault(entity, {}).setdefault(tl, []).append(dp)

    n = 0
    for (metric, base_unit), by_entity in sorted(buckets.items()):
        # ---- time series: one per entity with >= min_points_time labels
        for entity, by_time in sorted(by_entity.items()):
            labels = sorted(t for t in by_time if t)
            if len(labels) < min_points_time:
                st["time_too_short"] += 1
                continue
            chosen = {t: _choose(by_time[t], primary_domains) for t in labels}
            scale = _pick_display_scale([(0.0, parse_unit(getattr(c, "unit", ""))[2]) for c in chosen.values()])
            series = _series_from(chosen, labels, kind="time", metric=metric, base_unit=base_unit,
                                  scale_label=scale, entity=entity, primary_domains=primary_domains)
            n += 1
            key = f"ts{n}"
            labels_ref = _labels_ref_for(cat, labels, f"years_{key}")
            series.key, series.labels_ref = key, labels_ref
            series.name = _display_name(entity, metric, "")
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
            # entities ordered by value (desc) in the display scale
            def _val(c):
                b, f, _l = parse_unit(getattr(c, "unit", ""))
                return float(getattr(c, "value", 0.0) or 0.0) * f / _SCALE_LABEL_FACTOR.get(scale, 1.0)
            ordered = sorted(chosen.items(), key=lambda kv: -_val(kv[1]))
            labels = [_display_entity(e) for e, _c in ordered]
            series = _series_from({_display_entity(e): c for e, c in ordered}, labels, kind="comparison",
                                  metric=metric, base_unit=base_unit, scale_label=scale, entity="",
                                  primary_domains=primary_domains)
            n += 1
            key = f"cmp{n}"
            labels_ref = _labels_ref_for(cat, labels, f"entities_{key}")
            series.key, series.labels_ref = key, labels_ref
            series.year = int(tl[:4]) if tl else None
            series.name = _display_name("", metric, tl or "")
            cat.series[key] = series
            st["series"] = len(cat.series)
            if len(cat.series) >= max_series:
                st["capped"] = 1
                return cat
    return cat


def _display_entity(entity: str) -> str:
    return entity if entity else "（全体）"


def _display_name(entity: str, metric: str, when: str) -> str:
    parts = [p for p in (entity, metric, when) if p]
    return " ".join(parts)


def _series_from(chosen: Dict[str, Any], labels: List[str], *, kind: str, metric: str,
                 base_unit: str, scale_label: str, entity: str, primary_domains) -> Series:
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
        unit=_display_unit(base_unit, scale_label), scale_factor=disp_factor,
        values=values, basis=basis, sources=sources, entity=entity,
        note="；".join(notes[:6]),
    )
