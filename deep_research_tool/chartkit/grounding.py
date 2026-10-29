"""
Grounding check: every plotted value against the text it came from.

The numbers on a chart come from the data layer, which took them from an
LLM extraction of the evidence pages. Two things can still be wrong
there, and neither shows on the chart: the quoted text (``raw_text``) is
not in the page at all, or the value is not a number of that quote. Both
are checked here, deterministically, for every 公表 / 二次 / 換算 value
(導出 / 推定 values are computed, not quoted).

Both chart styles run the check and report what failed (要確認事項); the
enhanced style also says so under the chart.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional

from .data_layer import _JP_COMPOUND, _jp_compounds, _raw_numbers, parse_unit

QUOTED_BASIS = ("公表", "二次", "換算")
VALUE_NOT_IN_QUOTE = "value_not_in_quote"
QUOTE_NOT_IN_SOURCE = "quote_not_in_source"
PROBLEM_TEXT = {
    VALUE_NOT_IN_QUOTE: "値が引用元の記述と一致しません",
    QUOTE_NOT_IN_SOURCE: "引用元の記述が出典本文に見つかりません",
}


def _norm(text: str) -> str:
    """NFKC, lower case, no thousands separators, no spaces — except that
    the space between two numbers (a table row '1,100  1,200  1,350')
    becomes '|' so the numbers do not fuse into one."""
    s = unicodedata.normalize("NFKC", str(text or "")).lower()
    s = re.sub(r"(?<=\d)[,，](?=\d{3})", "", s)
    s = re.sub(r"(\d\.\d*?)0+(?![\d])", r"\1", s)           # 12.50 → 12.5
    s = re.sub(r"(\d)\.(?![\d])", r"\1", s)                  # 12. → 12
    s = re.sub(r"(?<=[\d.%])\s+(?=\d)", "|", s)
    return re.sub(r"\s+", "", s)


def _close(a: float, b: float) -> bool:
    # 0.5 %: rounding of a converted value, never 1,234 for '1,243'
    return abs(a - b) <= max(0.005 * max(abs(a), abs(b)), 1e-9)


_KANJI_DIGIT = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _percent_forms(text: str) -> List[float]:
    """Percentages written as 割 / 分の / 人に: '3割5分' 35, '約三割' 30,
    '3分の1' 33.3, '4人に1人' 25."""
    out = []
    for m in re.finditer(r"([0-9]+(?:\.[0-9]+)?|[一二三四五六七八九十])\s*割(?:\s*([0-9一二三四五六七八九])\s*分)?", text):
        a = m.group(1)
        w = float(a) if a[0].isdigit() else float(_KANJI_DIGIT[a])
        b = m.group(2)
        f = (float(b) if b and b.isdigit() else float(_KANJI_DIGIT.get(b, 0))) if b else 0.0
        out.append(w * 10 + f)
    for m in re.finditer(r"([0-9]+)\s*分の\s*([0-9]+)", text):
        if float(m.group(1)):
            out.append(float(m.group(2)) / float(m.group(1)) * 100)
    for m in re.finditer(r"([0-9]+)\s*(人|社|件|台|世帯|店|校)に\s*([0-9]+)\s*\2", text):
        if float(m.group(1)):
            out.append(float(m.group(3)) / float(m.group(1)) * 100)
    return out


def value_in_quote(value: float, unit: str, raw_text: str) -> bool:
    """The value (in the series' display unit) is a number of the quote:
    the same number, or the same quantity once both scales are applied
    (3.1 in 10億ドル for '約31億ドル')."""
    v = abs(float(value))
    base, factor, _label = parse_unit(unit)
    text = unicodedata.normalize("NFKC", str(raw_text or ""))
    if base == "%":
        # 3割 / 3割5分 / 3分の1 / 4人に1人 are percentages
        # (a fraction may be rounded: '3人に1人' as 33 or 33.3 %)
        if any(_close(v, p) or abs(v - round(p)) < 1e-9 or abs(v - round(p, 1)) < 1e-9
               for p in _percent_forms(text)):
            return True
        # a number that is part of '3割' / '3分の1' is not '3 %'
        text = re.sub(r"[0-9一二三四五六七八九十]+(?:\.[0-9]+)?\s*割(?:\s*[0-9一二三四五六七八九]\s*分)?"
                      r"|[0-9]+\s*分の\s*[0-9]+|[0-9]+\s*(人|社|件|台|世帯|店|校)に\s*[0-9]+\s*\1", " ", text)
    elif any(_close(v * factor, total) for total in _jp_compounds(text)):
        return True                     # 1兆2,000億円 / 3億5,000万円
    for n, f, w in _raw_numbers(text):
        if f != f:                      # compound scale (千万): not comparable
            continue
        if base == "%":
            # a percentage is a plain number: '50万人' is not 50 %
            if not w and _close(v, n):
                return True
            continue
        # the same quantity once both scales are applied; a number with a
        # scale word must match WITH it (50 in 10億ドル is not '50億ドル'),
        # a bare number (the scale is in a table header) matches as is
        if _close(v * factor, n * f):
            return True
        if not w and _close(v, n):
            return True
    return False


_YEAR = re.compile(r"^(19|20)\d{2}$")


def quote_in_source(raw_text: str, source_text: str) -> bool:
    """The quote is in the page — verbatim, or (the extraction trims and
    re-spaces quotes) every non-year number of it appears there."""
    q, src = _norm(raw_text), _norm(source_text)
    if not q or not src:
        return False
    # verbatim, but never inside a longer number ('3.5%' is not in '13.5%')
    pat = (r"(?<![\d.])" if q[0].isdigit() else "") + re.escape(q) + \
        (r"(?![\d]|\.\d)" if q[-1].isdigit() else "")
    if re.search(pat, src):
        return True
    nums = re.findall(r"\d+(?:\.\d+)?", q)
    key = [n for n in nums if not _YEAR.match(n)] or nums
    # whole numbers only: '50' is not in '2050' or '150' — and the scale
    # must agree ('$5.2B' is not '$5.2 million'; '35万人' is not '35人')
    if key and all(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d]|\.\d)", src) for n in key) \
            and not _scale_conflict(raw_text, source_text):
        return True
    # the same quantities written another way ('1.2兆円' for '1兆2,000億円'):
    # exactly the same — the 0.5 % rounding of a converted value is not
    # for the quote itself ('1,234億円' is not in a page that says '1,240億円')
    want = _quantities(raw_text)
    have = _quantities(source_text)
    return bool(want) and all(any(abs(w - h) <= 1e-9 * max(abs(w), abs(h)) for h in have) for w in want)


def _quantities(text: str) -> List[float]:
    """Every scaled quantity of a text (years left out); a Japanese
    compound ('1兆2,000億') counts as its total only."""
    t = unicodedata.normalize("NFKC", str(text or ""))
    out: List[float] = []
    rest = t
    for m in _JP_COMPOUND.finditer(t):
        totals = _jp_compounds(m.group(0))
        if totals:
            out.extend(totals)
            rest = rest.replace(m.group(0), " ", 1)
    out.extend(n * f for n, f, w in _raw_numbers(rest) if f == f and (w or not _YEAR.match(f"{n:g}")))
    return out


def _scale_conflict(raw_text: str, source_text: str) -> bool:
    """A number the quote writes WITH a scale appears in the page only
    with a different scale."""
    page = [(n, f) for n, f, _w in _raw_numbers(source_text) if f == f]
    for n, f, w in _raw_numbers(raw_text):
        if not w or f != f:
            continue
        same = [pf for pn, pf in page if _close(pn, n)]
        if same and all(not _close(pf, f) for pf in same):
            return True
    return False


def ground_panel(panel: Dict[str, Any],
                 source_text: Callable[[str, str], Optional[str]]) -> Dict[str, Any]:
    """Check one RESOLVED panel. ``source_text(evidence_id, url)`` returns
    the page text the value was extracted from (None when unknown — the
    quote is then only checked against the value). Returns
    ``{"checked": n, "issues": [{series, label, value, problem, raw_text, url}]}``."""
    checked = 0
    issues: List[Dict[str, Any]] = []
    labels = list(panel.get("labels") or [])
    for s in panel.get("series") or []:
        unit = str(s.get("unit") or panel.get("unit") or "")
        values = s.get("values") or []
        basis = s.get("basis") or []
        sources = s.get("sources") or []
        for j, v in enumerate(values):
            label = labels[j] if j < len(labels) else None
            if isinstance(v, dict):                     # line_xy point
                label, v = v.get("x"), v.get("y")
                if isinstance(label, float) and label.is_integer():
                    label = int(label)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            b = basis[j] if j < len(basis) else None
            src = sources[j] if j < len(sources) else None
            if b not in QUOTED_BASIS or not isinstance(src, dict):
                continue
            raw = str(src.get("raw_text") or "")
            if not raw:
                continue
            checked += 1
            problem = None
            # the plotted value, or the extracted one before the display
            # scale was applied (a bare '5,000' quote of 5 in 10億ドル)
            orig = src.get("value")
            if not value_in_quote(float(v), unit, raw) and not (
                    isinstance(orig, (int, float)) and not isinstance(orig, bool)
                    and value_in_quote(float(orig), str(src.get("unit") or ""), raw)):
                problem = VALUE_NOT_IN_QUOTE
            else:
                text = source_text(str(src.get("evidence_id") or ""), str(src.get("url") or ""))
                if text and not quote_in_source(raw, text):
                    problem = QUOTE_NOT_IN_SOURCE
            if problem:
                issues.append({"series": s.get("name"), "label": label, "value": v,
                               "problem": problem, "raw_text": raw[:160],
                               "url": src.get("url") or ""})
    return {"checked": checked, "issues": issues}
