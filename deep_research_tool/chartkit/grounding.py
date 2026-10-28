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

from .data_layer import _raw_numbers, parse_unit

QUOTED_BASIS = ("公表", "二次", "換算")
VALUE_NOT_IN_QUOTE = "value_not_in_quote"
QUOTE_NOT_IN_SOURCE = "quote_not_in_source"
PROBLEM_TEXT = {
    VALUE_NOT_IN_QUOTE: "値が引用元の記述と一致しません",
    QUOTE_NOT_IN_SOURCE: "引用元の記述が出典本文に見つかりません",
}


def _norm(text: str) -> str:
    """NFKC, lower case, no spaces, no thousands separators."""
    s = unicodedata.normalize("NFKC", str(text or "")).lower()
    s = re.sub(r"(?<=\d)[,，](?=\d{3})", "", s)
    return re.sub(r"\s+", "", s)


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.01 * max(abs(a), abs(b)), 1e-9)


def value_in_quote(value: float, unit: str, raw_text: str) -> bool:
    """The value (in the series' display unit) is a number of the quote:
    the same number, or the same quantity once both scales are applied
    (3.1 in 10億ドル for '約31億ドル')."""
    v = abs(float(value))
    base, factor, _label = parse_unit(unit)
    for n, f, w in _raw_numbers(raw_text):
        if f != f:                      # compound scale (千万): not comparable
            continue
        if base == "%":
            if _close(v, n):
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
    # whole numbers only: '50' is not in '2050' or '150'
    return bool(key) and all(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d]|\.\d)", src) for n in key)


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
