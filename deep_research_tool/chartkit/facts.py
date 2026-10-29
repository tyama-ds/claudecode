"""
Facts of a panel, computed from its values (never from the LLM), and the
check of the LLM's judgment against them.

The LLM writes the judgment ("what to look at") without seeing a single
value. Its sentence is NEVER rewritten: free-form Japanese ('2020年は最も
大きく落ち込んだ', '首位を奪われた', '原料高が毎年上昇し…') cannot be checked
reliably by rules, and replacing a true sentence is worse than leaving a
wrong one. Instead

- enhanced appends the computed facts ('データ: 2019→2023で…') so the
  reader sees the numbers next to the sentence;
- ``judgment_conflict`` reports (要確認事項, both styles) only the one
  claim shape that rules can read: a category named as the subject and
  directly called the largest / smallest, ending there — 'A社が最大。',
  '2024年が過去最高。', 'B社が最も低い。'.

``source_line`` is the data's own account of the sources, certainty and
forecasts (enhanced prints it under the chart).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .data_layer import _PER_POP

_TIME_LABEL = re.compile(r"^(19|20)\d{2}(\s*Q[1-4])?$", re.I)
_MAX_WORDS = ("過去最高", "過去最大", "最高", "最大", "最多", "首位", "トップ", "1位", "一位",
              "最も多い", "最も大きい", "最も高い")
_MIN_WORDS = ("過去最低", "過去最小", "最低", "最小", "最少", "最下位", "最も少ない", "最も小さい", "最も低い")
# a period is never 'ranked' (2021年に首位 = a company's rank that year)
_RANK = ("首位", "トップ", "1位", "一位", "最下位")
# the claim must END at the max / min word: 'A社が最大。' — not '最大手',
# '最大の…', '最も大きく落ち込んだ', '首位を奪われ', '過去最高だった2019年'
_END = r"(?:です|である|だ|となった|になった)?(?=$|[。．.！!？?」』）)]|、|，|,)"
# a negation anywhere in the sentence: not a plain claim
_NEG = ("ない", "なかった", "ず", "れず", "外れ", "陥落", "転落", "譲", "奪われ", "落とし", "後退")
# losses and deficits: '最大' of a negative amount is ambiguous
_LOSS = ("赤字", "損失", "欠損", "純損", "マイナス")
# a scope or a change: '国内勢ではB社が最大', 'A社を除くと…', 'コロナ後では…',
# '増加幅は2019年が最大' — the claim is not about the plotted values as a whole
_SCOPE = ("除", "以外", "以降", "以来", "以後", "後は", "後では", "では", "のうち", "の中で", "勢",
          "に限", "限定", "に絞", "だけで", "のみで", "で見ると", "で見れば", "において",
          "前年比", "前年同期比", "前期比", "増加幅", "減少幅", "上昇幅", "下落幅", "増加率", "減少率",
          "上昇率", "下落率", "伸び", "落ち込み", "増減", "成長率", "変化", "振れ幅",
          "前年差", "前年対比", "上げ幅", "下げ幅", "伸長率", "増加額", "減少額")


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
    """A display unit that starts with a number times a scale reads
    naturally: 5 in '10億ドル' is '50億ドル' (glued, it read '510億ドル').
    Other units that start with a number ('2015年=100', '100世帯当たり台')
    are not scales and stay as they are."""
    m = re.match(r"^(\d+)(兆|億|万|千|百万|十億)(.*)$", unit or "")
    # '10万人当たり' / '/10万人' are rates per 100,000 people, not 10 × 万
    if m and not _PER_POP.search(unit):
        return v * int(m.group(1)), m.group(2) + m.group(3)
    return v, unit


def _fmt(v: float, unit: str) -> str:
    v, unit = _natural(v, unit)
    if float(v).is_integer():
        s = f"{int(v):,}"
    elif abs(v) < 1:
        s = f"{v:.3g}"
    else:
        s = f"{v:,.1f}"
    if unit[:1].isdigit() or (_PER_POP.search(unit) and "/" not in unit):
        # '98.5（2015年=100）', '15.7（人口10万人当たり）' — never '98.52015年=100';
        # '12.1億t-co2', '15.7件/10万人' read as they are
        return f"{s}（{unit}）"
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
        # no percentage across zero or from a negative base (−120→−20 is not +83 %)
        "change_pct": (change / first[1] * 100.0)
        if unit != "%" and first[1] > 0 and last[1] >= 0 else None,
        "max": max(pts, key=lambda p: p[1]),
        "min": min(pts, key=lambda p: p[1]),
        "max_actual": max(actual, key=lambda p: p[1]),
        "min_actual": min(actual, key=lambda p: p[1]),
    }


def _material(a: float, b: float, unit: str) -> bool:
    return abs(b - a) >= (1.0 if unit == "%" else 0.05 * max(abs(a), abs(b), 1e-9))


def _move(a: float, b: float, unit: str) -> str:
    """The direction the reader is told: 5 % of the larger value, or 1 pt
    on a '%' series (0.4 %→0.9 % is a rise, not 'ほぼ横ばい')."""
    d = abs(b - a)
    if d < 0.05 * max(abs(a), abs(b), 1e-9) and not (unit == "%" and d >= 1.0):
        return "flat"
    return "up" if b > a else "down"


def _names(label: str, text: str) -> List[re.Match]:
    """Where the label is the subject: 'A社が' / '2024年は' — not inside a
    longer word ('京都' in '東京都', '2020' in '2020年代' or '12020')."""
    lead = r"(?<![0-9０-９A-Za-z\u4e00-\u9fff\u30a0-\u30ff])"
    tail = r"(?:年度|年)?(?:が|は)" if _TIME_LABEL.match(label) else r"(?:が|は)"
    return list(re.finditer(lead + re.escape(label) + tail, text))


def judgment_conflict(judgment: str, facts: Dict[str, Any]) -> Optional[str]:
    """Why the judgment plainly contradicts the data, or None: only
    '<label>が/は <max|min word>' ending there (see the module docstring),
    with no negation or loss in the sentence. A difference below 5 % (1 pt
    for '%') is never a contradiction; a rank chart ('位') is not checked
    (a smaller number is the better rank)."""
    import unicodedata
    # the catalog lower-cases names ('NXP' → 'nxp'); full-width digits too
    text = unicodedata.normalize("NFKC", str(judgment or "")).lower()
    if not text or facts.get("kind") not in ("time", "comparison"):
        return None
    unit = facts.get("unit") or ""
    pts: List[Tuple[str, float]] = facts["points"]
    if "位" in unit or any(w in text for w in _LOSS) or any(w in text for w in _NEG) \
            or any(w in text for w in _SCOPE) or (unit != "%" and any(v < 0 for _l, v in pts)):
        return None
    forecast = facts.get("forecast") or set()
    for lab, v in pts:
        period = bool(_TIME_LABEL.match(lab))
        for m in _names(unicodedata.normalize("NFKC", lab).lower(), text):
            # a topic before the name in the same sentence is a scope
            # ('欧州メーカーはVolkswagenが首位、Nissanが最下位', '増加率は2020年が最大')
            prefix = re.split(r"[。．!?！？]", text[:m.start()])[-1]
            if "は" in prefix or re.search(r"で[、,]?\s*$", prefix):
                continue                        # also '国内でB社が首位', '前年差で2020年が最大'
            rest = text[m.end():]
            words_max = [w for w in _MAX_WORDS if not (period and w in _RANK)]
            words_min = [w for w in _MIN_WORDS if not (period and w in _RANK)]
            is_max = any(re.match(re.escape(w) + _END, rest) for w in words_max)
            is_min = any(re.match(re.escape(w) + _END, rest) for w in words_min)
            # a forecast year is not part of 'the highest so far' for an actual one
            top = facts["max"] if lab in forecast else facts["max_actual"]
            bottom = facts["min"] if lab in forecast else facts["min_actual"]
            if is_max and v < top[1] and _material(v, top[1], unit):
                return f"「{lab}」が最大とされていますが、最大は「{top[0]}」です"
            if is_min and v > bottom[1] and _material(bottom[1], v, unit):
                return f"「{lab}」が最小とされていますが、最小は「{bottom[0]}」です"
    return None


def fact_sentence(facts: Dict[str, Any]) -> str:
    """One sentence of computed facts ('' for a multi-series panel)."""
    unit = facts.get("unit") or ""
    forecast = facts.get("forecast") or set()

    def mark(label: str) -> str:
        return f"{label}（予測）" if label in forecast else label
    if facts.get("kind") == "time":
        (fl, fv), (ll, lv) = facts["first"], facts["last"]
        fl, ll = mark(fl), mark(ll)
        if "位" in unit:                         # a rank: a smaller number is a rise
            word = {"up": "順位を下げた", "down": "順位を上げた", "flat": "ほぼ同じ順位"}[facts["direction"]]
            return f"{fl}→{ll}で{_fmt(fv, unit)}→{_fmt(lv, unit)}（{word}）"
        word = {"up": "増加", "down": "減少", "flat": "ほぼ横ばい"}[facts["direction"]]
        pct = facts.get("change_pct")
        extra = ""
        if facts["direction"] != "flat":
            if pct is not None:
                extra = f"、{pct:+.0f}%"
            elif unit == "%":
                c = facts["change"]             # '+0.03pt', never '+0pt'
                extra = "、" + (f"{c:+.2g}" if abs(c) < 0.1 else f"{c:+.1f}".replace(".0", "")) + "pt"
        return f"{fl}→{ll}で{_fmt(fv, unit)}→{_fmt(lv, unit)}（{word}{extra}）"
    if facts.get("kind") == "comparison":
        (ml, mv), (nl, nv) = facts["max"], facts["min"]
        if "位" in unit:                         # a rank: the smallest number is the top
            return f"最上位は{nl}（{_fmt(nv, unit)}）、最下位は{ml}（{_fmt(mv, unit)}）"
        return f"最大は{ml}（{_fmt(mv, unit)}）、最小は{nl}（{_fmt(nv, unit)}）"
    return ""


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
    line = f"出所（データ）: {'、'.join(short)}{more}" + (f"（{tags}）" if tags else "") + "。"
    if forecast:
        line += f"予測値: {'・'.join(forecast[:6])}{'…' if len(forecast) > 6 else ''}。"
    return line


def _host(url: Any) -> str:
    m = re.match(r"^[a-z]+://([^/]+)", str(url or ""), re.I)
    return m.group(1).lower().removeprefix("www.") if m else ""


def apply_facts(panel: Dict[str, Any], style: str = "classic") -> Dict[str, Any]:
    """Check a RESOLVED panel in place; returns what was found / added.

    The judgment is never rewritten. Both styles: a plain contradiction is
    returned (``conflict``) for 要確認事項. enhanced also appends the
    computed fact to the judgment, the data's source / certainty /
    forecast line to the caption and the unit to a title without one."""
    from .spec import fix_title_unit

    record: Dict[str, Any] = {}
    facts = panel_facts(panel)
    judgment = str(panel.get("judgment") or "")
    reason = judgment_conflict(judgment, facts)
    if reason:
        record["conflict"] = {"judgment": judgment, "reason": reason}
    if style != "enhanced":
        return record
    fact = fact_sentence(facts)
    if fact:
        shown = [_fmt(p[1], facts.get("unit") or "") for p in (facts.get("last"), facts.get("max")) if p]
        nums = [re.sub(r"[^0-9.]", "", n) for n in shown]
        plain = judgment.replace(",", "")
        # the judgment already quotes the number (as a whole number: '2' is not in '2020')
        if not any(n and re.search(r"(?<![0-9.])" + re.escape(n) + r"(?![0-9])", plain) for n in nums):
            judgment = (judgment.rstrip().rstrip("。.") + "。" if judgment.strip() else "") + f"データ: {fact}。"
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
