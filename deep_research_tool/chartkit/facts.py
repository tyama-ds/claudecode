"""
Facts of a panel, computed from its values (never from the LLM), and the
check of the LLM's judgment against them.

The LLM writes the judgment ("what to look at") without seeing a single
value, so it can call a falling line 一貫して増加 or name the wrong label
as the largest. ``judgment_conflict`` finds only PLAIN contradictions —
a sentence a reader would read as one unambiguous claim:

- 'A社が最大。' / '2024年が過去最高' / 'B社が最も低い': the named label
  directly followed by が / は and a maximum / minimum word (not 最大手,
  最大の…, a drop being 最大, a forecast year being higher);
- '一貫して増加' / '年々減少' / '右肩上がり': a whole-span trend word with
  one direction, no negation, no loss, no cause ('…により', '…ため').

Anything else is not checked: a missed contradiction costs less than a
true sentence called wrong. classic never rewrites the judgment (the
contradiction is reported in 要確認事項); enhanced replaces a contradicted
judgment with ``template_judgment`` and appends the computed fact to the
others. ``source_line`` is the data's own account of the sources,
certainty and forecasts (enhanced prints it under the chart).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_TIME_LABEL = re.compile(r"^(19|20)\d{2}(\s*Q[1-4])?$", re.I)
_UP = ("増加", "拡大", "伸び", "上昇", "成長", "増え", "伸長", "右肩上がり", "増収", "増益", "高ま",
       "回復", "持ち直", "反発", "上向")
_DOWN = ("減少", "縮小", "低下", "下落", "落ち込", "減り", "減収", "減益", "右肩下がり", "目減り",
         "反落", "下向", "頭打ち")
# a claim about the WHOLE plotted span
_TREND = ("一貫して", "一貫した", "年々", "毎年", "継続的に", "継続して", "右肩上がり", "右肩下がり",
          "期間を通じて", "期間中一貫", "一本調子")
# negated / softened claims are not checked
_NEG = ("ない", "せず", "ず、", "止ま", "とどま", "鈍化", "伸び悩", "横ばい", "一服", "限定的", "遅れ")
# losses and deficits: 縮小 of a negative amount is a rise of the value
_LOSS = ("赤字", "損失", "欠損", "純損", "マイナス")
# a cause or another subject in the sentence: the direction may not be the series'
_CAUSE = ("により", "によって", "ため", "ので", "を受け", "背景", "影響", "に伴", "につれ", "から")
_MAX_WORDS = ("過去最高", "過去最大", "最高", "最大", "最多", "首位", "トップ", "1位", "一位",
              "最も多", "最も大き", "最も高")
_MIN_WORDS = ("過去最低", "過去最小", "最低", "最小", "最少", "最下位", "最も少な", "最も小さ", "最も低")
# what may follow the max / min word for it to describe the subject itself
# ('A社が最大。', '最高水準', '首位を維持'); '最大手', '最大の成長余地' are not
_AFTER = (r"(?=$|[。、，,．.）)」』]|である|だ|で(?!は)|い|く|水準|値|額"
          r"|を(?:維持|保|守|堅持|占め|記録|更新|奪)|に(?:なっ|なる|立|躍|浮上|返り咲)|と(?:なっ|なる))")


def _num(v: Any) -> Optional[float]:
    if isinstance(v, dict):
        v = v.get("y")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def _points(panel: Dict[str, Any], s: Dict[str, Any]) -> List[Tuple[str, float, bool]]:
    """(label, value, forecast) for every plotted value of a series."""
    labels = list(panel.get("labels") or [])
    sources = s.get("sources") or []
    out = []
    for j, v in enumerate(s.get("values") or []):
        y = _num(v)
        if y is None:
            continue
        if isinstance(v, dict):
            x = v.get("x")
            lab = str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
        else:
            lab = str(labels[j]) if j < len(labels) else str(j)
        src = sources[j] if j < len(sources) else None
        out.append((lab, y, bool(isinstance(src, dict) and src.get("forecast"))))
    return out


def _natural(v: float, unit: str) -> Tuple[float, str]:
    """A display unit that starts with a number reads naturally: 5 in
    '10億ドル' is '50億ドル' (glued, it read '510億ドル')."""
    m = re.match(r"^(\d+)(\D.*)$", unit or "")
    if m:
        return v * int(m.group(1)), m.group(2)
    return v, unit


def _fmt(v: float, unit: str) -> str:
    v, unit = _natural(v, unit)
    if float(v).is_integer():
        s = f"{int(v):,}"
    elif abs(v) < 1:
        s = f"{v:.3g}"
    else:
        s = f"{v:,.1f}"
    return s + ("%" if unit == "%" else unit)


def panel_facts(panel: Dict[str, Any]) -> Dict[str, Any]:
    """{kind, unit, points, first, last, direction, change, change_pct,
    max, min, max_actual, min_actual} for a single-series panel;
    {kind: "multi"} otherwise. Points are (label, value)."""
    series = panel.get("series") or []
    unit = str(panel.get("unit") or (series[0].get("unit") if series else "") or "")
    if len(series) != 1 or str(panel.get("form") or "").startswith("stacked"):
        return {"kind": "multi", "unit": unit}
    raw = _points(panel, series[0])
    if len(raw) < 2:
        return {"kind": "multi", "unit": unit}
    pts = [(l, v) for l, v, _f in raw]
    actual = [(l, v) for l, v, f in raw if not f] or pts
    time = panel.get("form") == "line_xy" or all(_TIME_LABEL.match(l) for l, _v in pts)
    first, last = pts[0], pts[-1]
    change = last[1] - first[1]
    return {
        "kind": "time" if time else "comparison",
        "unit": unit,
        "points": pts,
        "forecast": {l for l, _v, f in raw if f},
        "first": first, "last": last,
        "direction": _move(first[1], last[1], unit),
        "change": change,
        "change_pct": (change / abs(first[1]) * 100.0) if first[1] and unit != "%" else None,
        "max": max(pts, key=lambda p: p[1]),
        "min": min(pts, key=lambda p: p[1]),
        "max_actual": max(actual, key=lambda p: p[1]),
        "min_actual": min(actual, key=lambda p: p[1]),
    }


def _material(a: float, b: float, unit: str) -> bool:
    return abs(b - a) >= (1.0 if unit == "%" else 0.05 * max(abs(a), abs(b), 1e-9))


def _move(a: float, b: float, unit: str) -> str:
    if not _material(a, b, unit):
        return "flat"
    return "up" if b > a else "down"


def _names(label: str, text: str) -> List[re.Match]:
    """Where the label is named as a subject: 'A社が' / '2024年は' (not
    inside '2020年代' or '12024')."""
    lead = r"(?<![0-9０-９A-Za-z])" if label[:1].isalnum() else ""
    tail = r"(?:年度|年)?(?:が|は|には|に)" if _TIME_LABEL.match(label) else r"(?:が|は)"
    return list(re.finditer(lead + re.escape(label) + tail, text))


# a period is never 'ranked' (2021年に首位 = a company's rank that year)
_RANK = ("首位", "トップ", "1位", "一位", "最下位")
# 'に' names a period only together with an explicit record word
_RECORD = ("過去最高", "過去最大", "過去最低", "過去最小")


def judgment_conflict(judgment: str, facts: Dict[str, Any]) -> Optional[str]:
    """Why the judgment plainly contradicts the data, or None (see the
    module docstring for what counts as a plain claim). A difference
    below 5 % (1 pt for '%') is never a contradiction."""
    text = str(judgment or "")
    if not text or facts.get("kind") not in ("time", "comparison"):
        return None
    unit = facts.get("unit") or ""
    pts: List[Tuple[str, float]] = facts["points"]
    if any(w in text for w in _LOSS) or (unit != "%" and any(v < 0 for _l, v in pts)):
        return None
    forecast = facts.get("forecast") or set()
    # 1. 'X が 最大 / 最小'
    for lab, v in pts:
        period = bool(_TIME_LABEL.match(lab))
        for m in _names(lab, text):
            rest = text[m.end():]
            words_max = [w for w in _MAX_WORDS if not (period and w in _RANK)]
            words_min = [w for w in _MIN_WORDS if not (period and w in _RANK)]
            if m.group(0).endswith("に"):
                words_max = [w for w in words_max if w in _RECORD]
                words_min = [w for w in words_min if w in _RECORD]
            is_max = any(re.match(re.escape(w) + _AFTER, rest) for w in words_max)
            is_min = any(re.match(re.escape(w) + _AFTER, rest) for w in words_min)
            # a forecast year is not part of 'the highest so far' for an actual one
            top = facts["max"] if lab in forecast else facts["max_actual"]
            bottom = facts["min"] if lab in forecast else facts["min_actual"]
            if is_max and v < top[1] and _material(v, top[1], unit):
                return f"「{lab}」が最大とされていますが、最大は「{top[0]}」です"
            if is_min and v > bottom[1] and _material(bottom[1], v, unit):
                return f"「{lab}」が最小とされていますが、最小は「{bottom[0]}」です"
    # 2. a whole-span trend
    if facts["kind"] != "time" or not any(w in text for w in _TREND):
        return None
    if any(w in text for w in _NEG) or any(w in text for w in _CAUSE):
        return None
    if any(_names(lab, text) or re.search(r"(?<![0-9])" + re.escape(lab) + r"(?![0-9])", text)
           for lab, _v in pts):
        return None                       # a named period: not the whole span
    up, down = any(w in text for w in _UP), any(w in text for w in _DOWN)
    if up == down:
        return None
    said = "up" if up else "down"
    real = facts["direction"]
    if real != "flat" and real != said:
        (fl, _fv), (ll, _lv) = facts["first"], facts["last"]
        return f"{fl}→{ll}は{'増加' if real == 'up' else '減少'}ですが、逆の方向で書かれています"
    return None


def fact_sentence(facts: Dict[str, Any]) -> str:
    """One sentence of computed facts ('' for a multi-series panel)."""
    unit = facts.get("unit") or ""
    if facts.get("kind") == "time":
        (fl, fv), (ll, lv) = facts["first"], facts["last"]
        word = {"up": "増加", "down": "減少", "flat": "ほぼ横ばい"}[facts["direction"]]
        pct = facts.get("change_pct")
        extra = ""
        if facts["direction"] != "flat":
            if pct is not None:
                extra = f"、{pct:+.0f}%"
            elif unit == "%":
                extra = f"、{facts['change']:+.1f}pt".replace(".0pt", "pt")
        return f"{fl}→{ll}で{_fmt(fv, unit)}→{_fmt(lv, unit)}（{word}{extra}）"
    if facts.get("kind") == "comparison":
        (ml, mv), (nl, nv) = facts["max"], facts["min"]
        return f"最大は{ml}（{_fmt(mv, unit)}）、最小は{nl}（{_fmt(nv, unit)}）"
    return ""


def template_judgment(facts: Dict[str, Any]) -> str:
    s = fact_sentence(facts)
    return f"{s}。" if s else ""


def source_line(panel: Dict[str, Any], max_names: int = 3) -> str:
    """'出所: A、B ほか2件（公表3・二次2）。予測値: 2028・2033。' from the data."""
    names: List[str] = []
    counts: Dict[str, int] = {}
    forecast: List[str] = []
    labels = list(panel.get("labels") or [])
    for s in panel.get("series") or []:
        basis = s.get("basis") or []
        for j, src in enumerate(s.get("sources") or []):
            if not isinstance(src, dict):
                continue
            name = str(src.get("title") or "").strip() or _host(src.get("url"))
            if name and name not in names:
                names.append(name)
            b = basis[j] if j < len(basis) else None
            if b:
                counts[b] = counts.get(b, 0) + 1
            if src.get("forecast"):
                vals = s.get("values") or []
                v = vals[j] if j < len(vals) else None
                if isinstance(v, dict):
                    x = v.get("x")
                    lab = str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
                else:
                    lab = str(labels[j]) if j < len(labels) else ""
                if lab and lab not in forecast:
                    forecast.append(lab)
    if not names and not counts:
        return ""
    short = [n if len(n) <= 30 else n[:29] + "…" for n in names[:max_names]]
    more = f" ほか{len(names) - max_names}件" if len(names) > max_names else ""
    order = ("公表", "換算", "導出", "推定", "二次")
    tags = "・".join(f"{b}{counts[b]}" for b in order if counts.get(b))
    line = f"出所: {'、'.join(short)}{more}" + (f"（{tags}）" if tags else "") + "。"
    if forecast:
        line += f"予測値: {'・'.join(forecast[:6])}{'…' if len(forecast) > 6 else ''}。"
    return line


def _host(url: Any) -> str:
    m = re.match(r"^[a-z]+://([^/]+)", str(url or ""), re.I)
    return m.group(1).lower().removeprefix("www.") if m else ""


def apply_facts(panel: Dict[str, Any], style: str = "classic") -> Dict[str, Any]:
    """Check a RESOLVED panel in place; returns what was found / changed.

    classic: nothing is rewritten — a plain contradiction is only returned
    (``conflict``) for 要確認事項. enhanced: a contradicted judgment is
    replaced by the computed sentence (``judgment_replaced``), the
    computed fact is appended to the others, the data's source / certainty
    / forecast line to the caption and the unit to a title without one."""
    from .spec import fix_title_unit

    record: Dict[str, Any] = {}
    facts = panel_facts(panel)
    judgment = str(panel.get("judgment") or "")
    reason = judgment_conflict(judgment, facts)
    if reason:
        record["conflict"] = {"judgment": judgment, "reason": reason}
    if style != "enhanced":
        return record
    if reason:
        new = template_judgment(facts)
        if new:
            record["judgment_replaced"] = {"from": judgment, "to": new, "reason": reason}
            panel["judgment"] = judgment = new
    fact = fact_sentence(facts)
    if fact and not reason:
        shown = [_fmt(p[1], facts.get("unit") or "") for p in (facts.get("last"), facts.get("max")) if p]
        nums = [re.sub(r"[^0-9.]", "", n) for n in shown]
        plain = judgment.replace(",", "")
        # the judgment already quotes the number (as a whole number: '2' is not in '2020')
        if not any(n and re.search(r"(?<![0-9.])" + re.escape(n) + r"(?![0-9])", plain) for n in nums):
            judgment = judgment.rstrip().rstrip("。.") + f"。データ: {fact}。"
            panel["judgment"] = judgment
            record["fact_added"] = fact
    line = source_line(panel)
    if line:
        caption = str(panel.get("caption") or "").rstrip()
        panel["caption"] = f"{caption}　{line}" if caption else line
        record["source_line"] = line
    unit = str(panel.get("unit") or "")
    if unit and panel.get("title"):
        panel["title"] = fix_title_unit(str(panel["title"]), unit, append=True)
    return record
