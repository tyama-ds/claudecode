"""
Facts of a panel, computed from its values (never from the LLM), and the
check of the LLM's judgment against them.

The LLM writes the judgment ("what to look at") without seeing a single
value, so it can say 増加 over a falling line or name the wrong label as
the largest. ``judgment_conflict`` finds the plain contradictions (one
direction word against the data's direction; 最大 / 最小 on a label
that is not), and ``template_judgment`` is the sentence that replaces a
contradicted judgment. ``source_line`` is the data's own account of the
sources, certainty and forecasts (the enhanced style prints it under the
chart).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_TIME_LABEL = re.compile(r"^(19|20)\d{2}(\s*Q[1-4])?$", re.I)
_UP = ("増加", "拡大", "伸び", "上昇", "成長", "増え", "伸長", "右肩上がり", "増収", "増益", "高ま",
       "回復", "持ち直", "反発", "上向")
_DOWN = ("減少", "縮小", "低下", "下落", "落ち込", "減り", "減収", "減益", "右肩下がり", "目減り",
         "反落", "下向", "頭打ち")
_MAX = ("最大", "最高", "ピーク", "トップ", "首位", "最も多", "最も大き")
_MIN = ("最小", "最低", "最も少な", "最も小さ")
# negated / softened claims are not checked
_NEG = ("ない", "せず", "ず、", "止ま", "とどま", "鈍化", "伸び悩", "横ばい", "一服")
# losses and deficits: 縮小 of a negative amount is a rise of the value
_LOSS = ("赤字", "損失", "欠損", "純損", "マイナス")
# a claim about the CHANGE, not about the plotted level
_CHANGE = ("前年比", "前期比", "前月比", "伸び率", "増加率", "減少率", "成長率", "増減", "変化",
           "伸び幅", "落ち込み幅", "増加幅", "減少幅", "差")
# '2018年以降 増加' / '2018年比で増加' / 'その後減少': from that label to the end
_SPAN = ("以降", "以後", "以来", "から", "その後", "比", "より")
# '前年比' etc.: against the period before the named label (checked first)
_PREV = ("前年比", "前期比", "前月比", "前年同期比", "前年から", "前年より", "前期から", "前期より")
_RANGE = ("から", "→", "〜", "～", "にかけて")


def _num(v: Any) -> Optional[float]:
    if isinstance(v, dict):
        v = v.get("y")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def _points(panel: Dict[str, Any], s: Dict[str, Any]) -> List[Tuple[str, float]]:
    labels = list(panel.get("labels") or [])
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
        out.append((lab, y))
    return out


def _fmt(v: float, unit: str) -> str:
    if float(v).is_integer():
        s = f"{int(v):,}"
    elif abs(v) < 1:
        s = f"{v:.3g}"
    else:
        s = f"{v:,.1f}"
    return s + ("%" if unit == "%" else unit)


def panel_facts(panel: Dict[str, Any]) -> Dict[str, Any]:
    """{kind, unit, points, first, last, direction, change, change_pct,
    max, min} for a single-series panel; {kind: "multi"} otherwise."""
    series = panel.get("series") or []
    unit = str(panel.get("unit") or (series[0].get("unit") if series else "") or "")
    if len(series) != 1 or str(panel.get("form") or "").startswith("stacked"):
        return {"kind": "multi", "unit": unit}
    pts = _points(panel, series[0])
    if len(pts) < 2:
        return {"kind": "multi", "unit": unit}
    time = panel.get("form") == "line_xy" or all(_TIME_LABEL.match(l) for l, _v in pts)
    first, last = pts[0], pts[-1]
    change = last[1] - first[1]
    material = abs(change) >= (1.0 if unit == "%" else 0.05 * max(abs(first[1]), abs(last[1]), 1e-9))
    direction = "flat" if not material else ("up" if change > 0 else "down")
    return {
        "kind": "time" if time else "comparison",
        "unit": unit,
        "points": pts,
        "first": first, "last": last,
        "direction": direction,
        "change": change,
        "change_pct": (change / abs(first[1]) * 100.0) if first[1] and unit != "%" else None,
        "max": max(pts, key=lambda p: p[1]),
        "min": min(pts, key=lambda p: p[1]),
    }


def _direction_words(text: str) -> Optional[str]:
    up = any(w in text for w in _UP)
    down = any(w in text for w in _DOWN)
    if up == down or any(w in text for w in _NEG):
        return None                 # both, neither, or negated: not a plain claim
    return "up" if up else "down"


def _material(a: float, b: float, unit: str) -> bool:
    return abs(b - a) >= (1.0 if unit == "%" else 0.05 * max(abs(a), abs(b), 1e-9))


def _move(a: float, b: float, unit: str) -> str:
    if not _material(a, b, unit):
        return "flat"
    return "up" if b > a else "down"


def judgment_conflict(judgment: str, facts: Dict[str, Any]) -> Optional[str]:
    """Why the judgment contradicts the data, or None.

    Conservative on purpose (a TRUE sentence must never be replaced):
    only plain claims are checked — one direction word, not negated, not
    about a loss or a change rate — over the whole span, from one named
    label onwards ('以降', '比', 'その後'), between two named labels joined
    by 'から' / '→', or for one named label against the period before;
    and 最大 / 最小 on the one label it names. A difference below 5 %
    (1 pt for '%') is never a contradiction."""
    text = str(judgment or "")
    if not text or facts.get("kind") not in ("time", "comparison"):
        return None
    unit = facts.get("unit") or ""
    pts: List[Tuple[str, float]] = facts["points"]
    if any(w in text for w in _LOSS) or (unit != "%" and any(v < 0 for _l, v in pts)):
        return None
    named = [i for i, (lab, _v) in enumerate(pts) if lab and lab in text]
    if len(named) == 1 and not any(w in text for w in _CHANGE):
        i = named[0]
        lab, v = pts[i]
        has_max, has_min = any(w in text for w in _MAX), any(w in text for w in _MIN)
        top, bottom = facts["max"], facts["min"]
        if has_max and not has_min and v < top[1] and _material(v, top[1], unit):
            return f"「{lab}」が最大とされていますが、最大は「{top[0]}」です"
        if has_min and not has_max and v > bottom[1] and _material(bottom[1], v, unit):
            return f"「{lab}」が最小とされていますが、最小は「{bottom[0]}」です"
    if facts["kind"] != "time":
        return None
    said = _direction_words(text)
    if said is None:
        return None
    if not named:
        (fl, fv), (ll, lv) = facts["first"], facts["last"]
    elif len(named) == 1:
        i = named[0]
        if any(w in text for w in _PREV):
            if i == 0:
                return None
            (fl, fv), (ll, lv) = pts[i - 1], pts[i]
        elif any(w in text for w in _SPAN):
            if i == len(pts) - 1:
                return None
            (fl, fv), (ll, lv) = pts[i], pts[-1]
        elif i > 0:
            (fl, fv), (ll, lv) = pts[i - 1], pts[i]
        else:
            return None
    elif len(named) == 2 and any(w in text for w in _RANGE):
        (fl, fv), (ll, lv) = pts[named[0]], pts[named[1]]
    else:
        return None
    real = _move(fv, lv, unit)
    if real != "flat" and real != said:
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
    """Check (and fix) a RESOLVED panel in place; returns what changed.

    Both styles: a judgment that contradicts the data is replaced by the
    computed sentence. enhanced also appends the computed fact to the
    judgment, the data's source / certainty / forecast line to the caption
    and the unit to a title without one."""
    from .spec import fix_title_unit

    record: Dict[str, Any] = {}
    facts = panel_facts(panel)
    judgment = str(panel.get("judgment") or "")
    reason = judgment_conflict(judgment, facts)
    if reason:
        new = template_judgment(facts)
        if new:
            record["judgment_replaced"] = {"from": judgment, "to": new, "reason": reason}
            panel["judgment"] = judgment = new
    if style != "enhanced":
        return record
    fact = fact_sentence(facts)
    if fact and not reason:
        shown = [_fmt(p[1], "").replace(",", "") for p in (facts.get("last"), facts.get("max")) if p]
        if not any(n and n in judgment.replace(",", "") for n in shown):
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
