"""
Spec layer: the LLM's chart specification (panels.json) — prompt,
parsing, resolution against the data and validation.

A panel: {id, form, title, labels_ref, series[{name, values_ref, role}],
highlight?, unit?, y_max?, height?, caption, judgment}. The LLM never
writes values; ``resolve`` fills ``labels`` / ``values`` / ``basis`` from
the data by key, and ``validate`` refuses anything that does not line up.
"""

from __future__ import annotations

import json
import re
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

from .data_layer import _CURRENCY_WORDS, canon, parse_unit

FORMS = ("bar", "hbar", "stacked_bar", "stacked_hbar", "line_xy")
ROLES = ("emphasis", "context", "context2", "part")
MAX_SERIES = 8
MIN_CV = 0.01


class SpecError(ValueError):
    """A panel that does not line up with the data."""


# ----------------------------------------------------------------------
# prompt
# ----------------------------------------------------------------------

STYLE_GUIDE_JA = """あなたは比較判断用チャートの設計者です。与えられたデータカタログから、人が比較・判断しやすい
チャート構成を JSON 配列で出力します。
1. 数値は書かない。系列は values_ref でカタログのキー（例: ts3, cmp7）をそのまま参照する。計算・換算・推定は禁止。
   labels_ref もカタログに書かれたものをそのまま使う。1 パネル内の系列は同じ labels_ref を持つものだけを組み合わせる。
2. form は bar / hbar / stacked_bar / stacked_hbar / line_xy から選ぶ。選択基準:
   量の比較=bar、項目が多い比較（6 以上）=hbar、構成比=stacked_hbar、時系列の推計比較=line_xy、能力と実績=stacked_bar。
3. 1 パネル=1 つの問い。パネルは最大 {max_panels}、系列は最大 8、二軸は禁止。単位の違う系列を 1 パネルに混ぜない。
   情報量のないパネル（値が 1 つだけ、全部同じ値、章の論点と無関係）は作らない。何も作れなければ [] を返す。
4. 強調する系列は 1 つだけ role="emphasis"、残りは "context" または "context2"。構成比は "part"。
   単一系列で特定の項目だけ目立たせたい場合は highlight にそのラベルを列挙する。
5. caption には出所と確度（公表/換算/導出/推定/二次）を必ず書く（カタログの basis と sources を使う）。
   judgment には「このグラフで何を見るべきか」を 1 文で書く。title には単位を括弧で含める。
6. 出力はスキーマに従う JSON 配列のみ。説明文やコードフェンスを付けない。

スキーマ:
[{"id": "[a-z0-9_]+", "form": "bar|hbar|stacked_bar|stacked_hbar|line_xy", "title": "…（単位）",
  "labels_ref": "カタログの labels_ref", "series": [{"name": "表示名", "values_ref": "カタログの values_ref", "role": "emphasis|context|context2|part"}],
  "highlight": ["ラベル"], "unit": "単位", "y_max": 数値, "caption": "出所と確度", "judgment": "見るべき点を 1 文"}]
"""

STYLE_GUIDE_EN = """You design charts for comparison and judgment. From the data catalog, output a JSON array of panels.
1. Never write numbers: series reference catalog keys through values_ref (e.g. ts3, cmp7); labels_ref must be the catalog's. Only combine series that share one labels_ref in a panel.
2. form is one of bar / hbar / stacked_bar / stacked_hbar / line_xy (quantity comparison=bar, many items=hbar, composition=stacked_hbar, time series of estimates=line_xy, capacity vs actual=stacked_bar).
3. One panel = one question. At most {max_panels} panels, at most 8 series, no dual axes, never mix units in one panel. Skip uninformative panels (single value, flat values, unrelated to the chapter). Return [] if nothing is worth plotting.
4. Exactly one role="emphasis" series at most; others "context" / "context2"; composition parts are "part". Use highlight for single-series bars.
5. caption states sources and certainty (公表/換算/導出/推定/二次 from the catalog's basis and sources); judgment says in one sentence what to look at; the title includes the unit in parentheses.
6. Output the JSON array only, no prose, no code fences.
Schema: [{"id","form","title","labels_ref","series":[{"name","values_ref","role"}],"highlight":[],"unit","y_max","caption","judgment"}]
"""


def build_messages(section_title: str, section_goal: str, catalog: List[Dict[str, Any]],
                   max_panels: int = 6, language: str = "ja",
                   research_topic: str = "") -> Dict[str, str]:
    """{"system", "user"} for one chapter; the catalog carries no values."""
    guide = STYLE_GUIDE_JA if language == "ja" else STYLE_GUIDE_EN
    system = guide.replace("{max_panels}", str(int(max_panels)))
    cat_json = json.dumps(catalog, ensure_ascii=False, indent=1)
    if language == "ja":
        user = (f"調査テーマ: {research_topic}\n章: {section_title}\n章の目的: {section_goal}\n"
                f"パネル数の上限: {max_panels}\n"
                f"データカタログ（キー・意味・単位・ラベル・確度の種類・出典数。値は含めない）:\n{cat_json}")
    else:
        user = (f"Topic: {research_topic}\nChapter: {section_title}\nGoal: {section_goal}\n"
                f"Max panels: {max_panels}\nData catalog (keys, meaning, unit, labels, certainty kinds, source count; no values):\n{cat_json}")
    return {"system": system, "user": user}


# ----------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------

def parse_panels(text: str, status: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The JSON array in an LLM reply (code fences and prose tolerated).
    ``status["ok"]`` tells an explicit [] (nothing worth plotting) from a
    reply that held no JSON array at all."""
    if status is not None:
        status["ok"] = False
    if not text:
        return []
    s = str(text).strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1).strip()
    start, end = s.find("["), s.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        data = json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    if status is not None:
        status["ok"] = True
    return [p for p in data if isinstance(p, dict)]


# ----------------------------------------------------------------------
# resolve + validate
# ----------------------------------------------------------------------

def resolve(panels: List[Dict[str, Any]], data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fill labels / values / basis / unit from the data by key (values only
    ever come from the data layer). Missing keys raise SpecError; so does
    a series whose own label array is not the panel's (its values would
    be drawn under the wrong labels)."""
    out = []
    for p in panels:
        q = dict(p)
        q["series"] = [dict(s) for s in (p.get("series") or []) if isinstance(s, dict)]
        ref = q.get("labels_ref")
        xy_form = q.get("form") == "line_xy"
        if ref is not None or not xy_form:
            if ref not in data or not isinstance(data.get(ref), list):
                raise SpecError(f"{q.get('id')}: labels_ref '{ref}' がデータにありません")
            q["labels"] = list(data[ref])
        else:
            q["labels"] = []            # line_xy over {x, y} points needs no label axis
        for s in q["series"]:
            vref = s.get("values_ref")
            d = data.get(vref)
            if not isinstance(d, dict) or "values" not in d:
                raise SpecError(f"{q.get('id')}/{s.get('name')}: values_ref '{vref}' がデータにありません")
            s_ref = d.get("labels_ref")
            if s_ref and ref and s_ref != ref and list(data.get(s_ref) or []) != q["labels"]:
                raise SpecError(f"{q.get('id')}/{s.get('name')}: 系列のラベル配列 '{s_ref}' が"
                                f"パネルの labels_ref '{ref}' と一致しません")
            s["values"] = list(d["values"])
            s["basis"] = list(d.get("basis") or [None] * len(s["values"]))
            s["unit"] = d.get("unit")            # the DATA's unit, never the LLM's
            # what the series IS (for the legend name check)
            s["_data"] = {"name": d.get("name") or "", "kind": d.get("kind") or "",
                          "entity": d.get("entity") or "", "year": d.get("year")}
            s["source"] = d.get("source", "")
            s["sources"] = list(d.get("sources") or [])
            if xy_form and not all(isinstance(v, dict) for v in s["values"] if v is not None):
                # a labelled series: its numeric labels (years) are the x axis
                if not q["labels"]:
                    raise SpecError(f"{q.get('id')}/{s.get('name')}: line_xy にはラベル配列（年）が必要です")
                _to_xy(q, s)
        _check_series_names(q)
        _normalize_panel(q)
        out.append(q)
    return out


def _series_keys(s: Dict[str, Any]) -> List[str]:
    """What tells this series apart from the others in a panel: the
    entity of a time series, the year of a comparison."""
    d = s.get("_data") or {}
    if d.get("kind") == "time" and d.get("entity"):
        return [canon(d["entity"])]
    if d.get("kind") == "comparison" and d.get("year"):
        return [str(d["year"])]
    return []


def _check_series_names(q: Dict[str, Any]) -> None:
    """The legend name is the LLM's (it never sees the values): a name that
    names ANOTHER series of the panel (company A's line labelled 'B社')
    would swap the subjects while every number stays right. Such a name
    is replaced by the data's own name; a name that names neither (a
    generic '当社') is kept."""
    series = q.get("series") or []
    keys = [_series_keys(s) for s in series]
    fixes = []
    for i, s in enumerate(series):
        name = canon(s.get("name") or "")
        own = keys[i]
        if not name or not own or any(k and k in name for k in own):
            continue
        others = [k for j, ks in enumerate(keys) if j != i for k in ks if k and k not in own]
        if any(k in name for k in others):
            data_name = (s.get("_data") or {}).get("name") or s.get("values_ref") or ""
            fixes.append(f"{s.get('name')} → {data_name}")
            s["name"] = data_name
    if fixes:
        q["name_fixes"] = fixes


_CURRENCY_BASES = set(_CURRENCY_WORDS.values())
_TITLE_UNIT = re.compile(r"[（(]([^（）()]{1,16})[）)]\s*$")


def fix_title_unit(title: str, unit: str, append: bool = False) -> str:
    """The unit in a title's closing parentheses must be the DATA's unit
    (the LLM wrote it without seeing the data: '（億円）' over 百万円
    values). A parenthesis that is not a unit ('（調査会社別）') is kept;
    ``append`` adds '（unit）' to a title without one."""
    title = str(title or "")
    unit = str(unit or "")
    if not unit:
        return title
    m = _TITLE_UNIT.search(title)
    if m:
        inner = m.group(1).strip()
        if inner == unit:
            return title
        ib, i_f, _l = parse_unit(inner)
        ub, u_f, _l2 = parse_unit(unit)
        # only a parenthesis that IS a unit is replaced: '（2024年）',
        # '（前年比）', '（主要5社）' are kept even over a '%' series
        inner_is_unit = ib == ub or ib == "%" or ib in _CURRENCY_BASES or i_f != 1.0
        related = (ib == ub) or (ib in _CURRENCY_BASES and ub in _CURRENCY_BASES) \
            or "%" in (ib, ub)
        if inner_is_unit and related and (ib, i_f) != (ub, u_f):
            return title[:m.start(1)] + unit + title[m.end(1):]
        return title
    return f"{title}（{unit}）" if append else title


def _to_xy(q: Dict[str, Any], s: Dict[str, Any]) -> None:
    """line_xy plots a labelled series against numeric labels (years):
    values become {x, y} points; non-numeric labels cannot be an x axis."""
    xs = []
    for lab in q["labels"]:
        try:
            xs.append(float(str(lab).strip()))
        except ValueError:
            raise SpecError(f"{q.get('id')}: line_xy はラベルが数値（年）の系列だけです（{lab!r}）")
    pts, basis, sources = [], [], []
    srcs = list(s.get("sources") or []) + [None] * len(xs)
    for x, v, b, src in zip(xs, s["values"], s["basis"] + [None] * len(xs), srcs):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            pts.append({"x": x, "y": float(v)})
            basis.append(b)
            sources.append(src)            # stays aligned with the points
    s["values"], s["basis"], s["sources"] = pts, basis, sources


def _data_max(q: Dict[str, Any]) -> float:
    """The largest plotted value (stack total for stacked forms)."""
    series = q["series"]
    if q.get("form") == "line_xy":
        ys = [float(v["y"]) for s in series for v in (s.get("values") or [])
              if isinstance(v, dict) and isinstance(v.get("y"), (int, float))]
        return max(ys) if ys else 0.0
    if str(q.get("form") or "").startswith("stacked"):
        n = max((len(s.get("values") or []) for s in series), default=0)
        totals = []
        for j in range(n):
            tot = 0.0
            for s in series:
                vals = s.get("values") or []
                v = vals[j] if j < len(vals) else None
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    tot += float(v)
            totals.append(tot)
        return max(totals) if totals else 0.0
    vals = [float(v) for s in series for v in (s.get("values") or [])
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return max(vals) if vals else 0.0


def _normalize_panel(q: Dict[str, Any]) -> None:
    """Hints the LLM wrote WITHOUT seeing the values: the panel unit is
    the data's unit, y_max never clips a bar (raised to the data max),
    height is an int in pixels, highlight only applies to one series."""
    units = [str(s.get("unit") or "") for s in q["series"]]
    if units and len(set(units)) == 1:
        q["unit"] = units[0]
    y_max = q.get("y_max")
    if isinstance(y_max, bool) or not isinstance(y_max, (int, float)):
        q.pop("y_max", None)
    else:
        top = _data_max(q)
        # an axis far above the data (the LLM guessed 5000 for values up to
        # 300) squashes every bar; % axes may stop at 100
        roomy = max(2.0 * top, 100.0 if q.get("unit") == "%" else 0.0)
        if top <= 0 or float(y_max) > roomy:
            q.pop("y_max", None)
        else:
            q["y_max"] = max(float(y_max), top)
    if q.get("title") and q.get("unit"):
        q["title"] = fix_title_unit(q["title"], q["unit"])
    height = q.get("height")
    if height is not None:
        try:
            q["height"] = max(60, min(2000, int(float(str(height).strip().lower().removesuffix("px")))))
        except (TypeError, ValueError):
            q.pop("height", None)


def _cv(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (pstdev(values) / abs(m)) if abs(m) > 1e-12 else 1.0


def validate(panels: List[Dict[str, Any]]) -> None:
    """Raise SpecError on the first invalid (resolved) panel."""
    seen_ids = set()
    for p in panels:
        pid = str(p.get("id") or "")
        if not re.fullmatch(r"[a-z0-9_]+", pid):
            raise SpecError(f"{pid!r}: id は [a-z0-9_]+ で指定してください")
        if pid in seen_ids:
            raise SpecError(f"{pid}: id が重複しています")
        seen_ids.add(pid)
        form = p.get("form")
        if form not in FORMS:
            raise SpecError(f"{pid}: form が不正です（{form!r}）")
        series = p.get("series") or []
        if not 1 <= len(series) <= MAX_SERIES:
            raise SpecError(f"{pid}: 系列数は 1〜{MAX_SERIES} です（{len(series)}）")
        if form.startswith("stacked") and not 2 <= len(series) <= 4:
            raise SpecError(f"{pid}: stacked_* の系列は 2〜4 本です")
        if form.startswith("stacked") and any(
                isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0
                for s in series for v in (s.get("values") or [])):
            # a negative part cannot be stacked (the segment would vanish)
            raise SpecError(f"{pid}: 積み上げ形式に負の値は描けません")
        if sum(1 for s in series if s.get("role") == "emphasis") > 1:
            raise SpecError(f"{pid}: emphasis は 1 系列までです")
        for s in series:
            if s.get("role") not in ROLES:
                raise SpecError(f"{pid}/{s.get('name')}: role が不正です")
        if not str(p.get("caption") or "").strip() or not str(p.get("judgment") or "").strip():
            raise SpecError(f"{pid}: caption / judgment は必須です")
        if not str(p.get("title") or "").strip():
            raise SpecError(f"{pid}: title は必須です")
        units = {str(s.get("unit") or "") for s in series}
        if len(units) > 1:
            raise SpecError(f"{pid}: 単位の違う系列が混在しています（{sorted(units)}）")
        if form == "line_xy":
            for s in series:
                vals = s.get("values") or []
                if not vals or not all(isinstance(v, dict) and "x" in v and "y" in v for v in vals):
                    raise SpecError(f"{pid}/{s.get('name')}: line_xy の値は {{x, y}} の配列です")
                if len(s.get("basis") or []) != len(vals):
                    raise SpecError(f"{pid}/{s.get('name')}: basis の長さが不一致です")
            continue
        labels = p.get("labels") or []
        if len(labels) < 1:
            raise SpecError(f"{pid}: ラベルがありません")
        for s in series:
            vals = s.get("values") or []
            if len(vals) != len(labels):
                raise SpecError(f"{pid}/{s.get('name')}: ラベル数と値数が不一致です（{len(labels)} vs {len(vals)}）")
            if len(s.get("basis") or []) != len(vals):
                raise SpecError(f"{pid}/{s.get('name')}: basis の長さが不一致です")
            numeric = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if not numeric:
                raise SpecError(f"{pid}/{s.get('name')}: 値がすべて空です")
        hl = p.get("highlight")
        if hl is not None:
            if not isinstance(hl, list) or any(h not in labels for h in hl):
                raise SpecError(f"{pid}: highlight にラベルにない項目があります")
            if len(series) != 1:
                # highlight is a single-series device (the prompt says so);
                # on a multi-series panel it would recolour every series
                # while the legend keeps the role colours
                p.pop("highlight", None)
        # informativeness: a single series that is flat or a single point
        # says nothing (the old pipeline drew those by the dozen)
        all_numeric = [float(v) for s in series for v in (s.get("values") or [])
                       if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(all_numeric) < 2:
            raise SpecError(f"{pid}: 値が 1 つしかありません")
        if len(series) == 1 and not form.startswith("stacked") and _cv(all_numeric) < MIN_CV:
            raise SpecError(f"{pid}: 値がすべて同じです")


def validate_panels(panels: List[Dict[str, Any]], data: Dict[str, Any],
                    max_panels: Optional[int] = None,
                    stats: Optional[Dict[str, Any]] = None,
                    ) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], str]]]:
    """Resolve + validate panel by panel: (valid resolved panels, rejected
    [(panel, reason)]). One bad panel never discards the others.
    ``stats["over_limit"]`` counts the panels left unchecked once
    ``max_panels`` valid ones were found."""
    valid: List[Dict[str, Any]] = []
    rejected: List[Tuple[Dict[str, Any], str]] = []
    seen: set = set()
    ids: set = set()
    for p in panels:
        try:
            r = resolve([p], data)
            validate(r)
            pid = str(r[0].get("id"))
            if pid in ids:          # two panels, one PNG path / canvas id
                raise SpecError(f"{pid}: id が重複しています")
            # the same series in the same form twice says nothing new
            sig = (r[0].get("form"), r[0].get("labels_ref"),
                   tuple(sorted(str(s.get("values_ref")) for s in r[0].get("series") or [])))
            if sig in seen:
                raise SpecError(f"{pid}: 同じ系列・同じ形式のパネルが重複しています")
            seen.add(sig)
            ids.add(pid)
            valid.append(r[0])
        except SpecError as e:
            rejected.append((p, str(e)))
        except Exception as e:                      # malformed JSON shapes
            rejected.append((p, f"{type(e).__name__}: {e}"))
        if max_panels is not None and len(valid) >= max_panels:
            break
    if stats is not None:
        stats["over_limit"] = max(0, len(panels) - len(valid) - len(rejected))
    return valid, rejected


def panel_semantics(panel: Dict[str, Any]) -> Dict[str, Any]:
    """The verifiable content of a resolved panel (joins the semantic
    manifest; a changed number after the freeze changes the hash)."""
    series: Dict[str, List[Any]] = {}
    detail: List[Dict[str, Any]] = []
    for s in panel.get("series") or []:
        name = str(s.get("name") or s.get("values_ref") or "")
        base, n = name, 2
        while name in series:                    # two series with one display name
            name = f"{base} ({n})"; n += 1
        vals = s.get("values") or []
        # line_xy points are {x, y}: keep them readable in the verified text
        series[name] = [f"{v.get('x')}:{v.get('y')}" if isinstance(v, dict) else v for v in vals]
        detail.append({"name": name, "role": s.get("role"), "values_ref": s.get("values_ref"),
                       "basis": list(s.get("basis") or [])})
    return {
        "form": panel.get("form"),
        "title": panel.get("title"),
        "unit": panel.get("unit"),
        "labels": list(panel.get("labels") or []),
        # {name: values} — the shape the semantic manifest verifies/freezes
        "series": series,
        "series_detail": detail,
        "highlight": list(panel.get("highlight") or []),
        "caption": panel.get("caption"),
        "judgment": panel.get("judgment"),
    }
