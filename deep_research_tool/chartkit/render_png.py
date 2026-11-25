"""
PNG renderer (matplotlib) with the SAME rules as the HTML page, for the
Markdown / Word / PDF outputs: one emphasised series in blue, context in
grey, negative values red, category colours for parts and line_xy,
striped fill for every bar value that is not 公表 (a hollow marker on a
line), forecasts drawn dashed and hollow, value labels, the unit in the
title, a single axis, a legend built from the roles (none for a single
series), and the caption and judgment under the chart (left out with
``embed_text=False``: the report then prints them once as text).

Two styles (``report.chart_style``):

- ``classic``  — the v1.5 look (10 in wide, the same sizes and layout)
- ``enhanced`` — drawn at the size it is printed (5.5 in, the width Word
  and PDF use, so the text is not shrunk), a bar chart with many or long
  labels drawn as horizontal bars, and a layout QA pass (overlapping
  tick labels rotated, the plot enlarged when text spills out of the
  figure)

Both styles get the correctness fixes: negative values are drawn (the
value axis is no longer pinned at 0), a line chart's value axis follows
the data (0 only when the values are near 0), the value axis prints
plain numbers with thousands separators (never a '1e6' offset), point /
value labels never print over each other or over another series'
markers, lines or bars (they move, or are dropped; the emphasised series
keeps its labels), long category names set the left margin, small
numbers keep their digits (0.04, not 0.0), text never breaks inside a
number, proportional CJK fonts come first, and rendering is serialized
(pyplot keeps global state; the Web UI runs several jobs in one process).
"""

from __future__ import annotations

import math
import re
import textwrap
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

ROLE = {"emphasis": "#2a78d6", "context": "#888780", "context2": "#B4B2A9", "negative": "#e34948"}
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEG = [("#85B7EB", "#042C53"), ("#F0997B", "#4A1B0C"), ("#B4B2A9", "#2C2C2A"), ("#A9D8C2", "#0E3D2A")]
# every marker can be drawn hollow (an 'x' could not)
MARKERS = ["o", "s", "^", "D", "*", "v"]
INK = "#52514e"
GRID = "#e1e0d9"
# notes under the legend (prelude.js prints the same words): a value that
# is not 公表 is striped on a bar and hollow on a line; a forecast is dashed
# and hollow (a bar: hollow, dotted, dashed outline)
STRIPE_NOTE = "斜線＝換算・導出・推定・二次"
HOLLOW_NOTE = "白抜き＝換算・導出・推定・二次"
FORECAST_NOTE = "破線・白抜き＝予測値"
FORECAST_DASH = (0, (4, 3))
FORECAST_HATCH = "...."

STYLES = ("classic", "enhanced")
# classic: the v1.5 sizes; enhanced: drawn at the printed width (Word / PDF
# place charts 5.5 in wide) so 8-11 pt text stays 8-11 pt on paper
_LAYOUT = {
    # cat: category names on a horizontal chart; min_plot: the plot area
    # never shrinks below this height (inches) under a long caption / legend
    "classic": {"width": 10.0, "dpi": 150, "title": 12, "tick": 9, "cat": 10, "value": 8, "caption": 8,
                "judgment": 9, "legend": 9, "wrap": 70, "bar_h": 5.2, "min_plot": 1.8, "cat_wrap": 30},
    "enhanced": {"width": 5.5, "dpi": 200, "title": 11, "tick": 8.5, "cat": 8.5, "value": 7.5, "caption": 7.5,
                 "judgment": 8, "legend": 8, "wrap": 46, "bar_h": 3.4, "min_plot": 1.9, "cat_wrap": 16},
}

# both styles: proportional CJK fonts first (Noto / BIZ UDP / IPAex /
# IPAP; a fixed-pitch font set '1,234' and Latin names with wide gaps),
# the fixed-pitch ones (IPAGothic, MS Gothic…) only when nothing else is
# installed. Hints match by substring: 'IPAGothic' does not match
# 'IPAPGothic' or 'IPAexGothic'.
_PROPORTIONAL_FIRST = ("Noto Sans CJK JP", "Noto Sans JP", "NotoSansCJK", "NotoSansJP", "BIZ UDPGothic",
                       "IPAexGothic", "IPAPGothic", "Source Han Sans", "Yu Gothic", "YuGothic",
                       "Hiragino Sans", "Meiryo", "MS PGothic", "TakaoPGothic", "VL PGothic",
                       "IPAGothic", "MS Gothic", "TakaoGothic", "VL Gothic", "WenQuanYi", "Unifont")
_JP_FONT_HINTS = {"classic": _PROPORTIONAL_FIRST, "enhanced": _PROPORTIONAL_FIRST}
_font_cache: Dict[str, Optional[str]] = {}
# pyplot (figure manager, rcParams) is process-global: one render at a time
_RENDER_LOCK = threading.Lock()
# set when no CJK-capable font was found (the caller warns once)
FONT_STATE = {"japanese_missing": False}


def setup_japanese_font(plt, fm, style: str = "classic") -> Optional[str]:
    """Pick a CJK-capable font by substring (the old exact-name search
    silently fell back to DejaVu and drew □□□); returns the family name."""
    key = f"name:{style}"
    if key in _font_cache:
        name = _font_cache[key]
    else:
        name = None
        try:
            names = [f.name for f in fm.fontManager.ttflist]
            for hint in _JP_FONT_HINTS.get(style, _JP_FONT_HINTS["classic"]):
                hits = [n for n in names if hint.lower().replace(" ", "") in n.lower().replace(" ", "")]
                if hits:
                    # a family's fixed-pitch variant ('… Mono') only when it is the only one
                    name = next((n for n in hits if "mono" not in n.lower()), hits[0])
                    break
        except Exception:
            name = None
        _font_cache[key] = name
    FONT_STATE["japanese_missing"] = name is None
    if name:
        plt.rcParams["font.family"] = name
    plt.rcParams["axes.unicode_minus"] = False
    return name


def _published(b) -> bool:
    return b is None or b == "公表"


_GROWTH_RE = re.compile(r"前年比|前期比|成長|伸び|増減|変化|増加率|growth|yoy|change|cagr", re.I)


def _signed_pct(panel: Dict[str, Any]) -> bool:
    """'%' values carry a sign only on change panels (growth / YoY /
    negatives present); shares and ratios stay unsigned — same rule as
    prelude.js."""
    if any(isinstance(v, (int, float)) and v < 0
           for s in (panel.get("series") or []) for v in (s.get("values") or [])):
        return True
    return bool(_GROWTH_RE.search(str(panel.get("title") or "")))


def _fmt(value: float, unit: str, signed: bool = False) -> str:
    if unit == "%":
        return f"{'+' if signed and value > 0 else ''}{value:g}%"
    if float(value).is_integer():
        return f"{int(value):,}"
    if abs(value) < 1:
        # 0.04 must not print as 0.0: three significant digits
        return f"{value:.3g}"
    return f"{value:,.1f}"


# characters that must not start a line (Japanese line-breaking rules:
# closing punctuation, small kana, the long-vowel mark, iteration marks)
_NO_LINE_START = set("、。，．・：；？！）］｝〕〉》」』】〙〗ー々ゝゞ～%％,.:;!?)]}”’"
                     "ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶ")
# … and that must not end one (opening brackets)
_NO_LINE_END = set("（［｛〔〈《「『【〘〖([{“‘")
# a number keeps its multiplier ('6億缶', '1.6億円', '3千5百'); a Latin word stays whole
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,%％'’\-/:+]*(?:[千百万億兆京][0-9.,]*)*|\s+|.", re.S)
# ASCII spaces and the ideographic space never start or end a line
_EDGE_SPACE = " \u3000"


def _break_at(chunk: str, room: int, width: int, level: int = 0) -> int:
    """How many characters of ``chunk`` (longer than ``room``) go on this
    line, at most ``room``: after the last hyphen that fits (textwrap's
    rule), otherwise between CJK characters anywhere, but never inside a
    Latin word or a number with its multiplier that fits a line (a URL
    longer than a line is cut), never before closing punctuation and never
    after an opening bracket. 0 when no such break fits.
    ``level`` 1 drops the punctuation rules, 2 also the word rule (a line
    that must break somewhere: cut at ``room``)."""
    if level >= 2:
        return room
    hyphen = chunk.rfind("-", 0, room)
    if hyphen > 0 and any(c != "-" for c in chunk[:hyphen]):
        return hyphen + 1
    whole = [(m.start(), m.end()) for m in _TOKEN.finditer(chunk) if 1 < m.end() - m.start() <= width]
    for end in range(min(room, len(chunk) - 1), 0, -1):
        if any(a < end < b for a, b in whole):
            continue
        if level == 0 and (chunk[end] in _NO_LINE_START or chunk[end - 1] in _NO_LINE_END):
            continue
        return end
    return 0


class _Wrapper(textwrap.TextWrapper):
    """textwrap's chunks and spaces (v1.5 wrapped with textwrap: words
    move to the next line whole, Latin words break at spaces and hyphens),
    except where a word is longer than the room left on a line: it is
    broken by ``_break_at`` (CJK anywhere, Latin words and numbers kept
    whole, Japanese line-breaking rules), or moved to the next line whole.
    A line is never longer than the width."""

    def _handle_long_word(self, reversed_chunks, cur_line, cur_len, width):
        chunk = reversed_chunks[-1]
        if not chunk.strip(_EDGE_SPACE):
            reversed_chunks.pop()               # a run of spaces wider than a line
            return
        width = max(1, width)
        room = width - cur_len
        end = _break_at(chunk, room, width) if room > 0 else 0
        if end <= 0 and any(c.strip(_EDGE_SPACE) for c in cur_line):
            # the whole word goes to the next line: end this one without its
            # trailing space (textwrap 3.11/3.12 kept it when the line was full)
            while cur_line and not cur_line[-1].strip(_EDGE_SPACE):
                cur_line.pop()
            return
        if end <= 0:
            # at the start of a line (or after nothing but spaces): break
            # somewhere, at the latest at the full width
            cur_line.clear()
            room, level = width, 0
            while end <= 0:
                end = _break_at(chunk, room, width, level)
                level += 1
        cur_line.append(chunk[:end])
        if end >= len(chunk):
            reversed_chunks.pop()
        else:
            reversed_chunks[-1] = chunk[end:]


def _wrap(text: str, width: int) -> str:
    """Wrap to ``width`` characters (each character counts 1): Latin text
    breaks at spaces and hyphens as textwrap does (v1.5 used textwrap),
    CJK text between any two characters under the Japanese line-breaking
    rules, a number with its multiplier ('6億缶', '1.5') and a Latin word
    that fits a line are never split; no line starts or ends with a space
    and none is longer than ``width``."""
    text = str(text or "")
    if not text:
        return ""
    wrapper = _Wrapper(width=max(1, width))
    text = wrapper._munge_whitespace(text).strip(_EDGE_SPACE)
    if not text:
        return ""
    lines = (ln.strip(_EDGE_SPACE) for ln in wrapper.wrap(text))
    return "\n".join(ln for ln in lines if ln)


def _overlaps(a, b, pad: float = 1.0) -> bool:
    return not (a.x1 + pad < b.x0 or b.x1 + pad < a.x0 or a.y1 + pad < b.y0 or b.y1 + pad < a.y0)


def _drop_overlapping(fig, texts: List[Any]) -> int:
    """Remove texts that overlap an earlier kept one (value labels of
    close points would otherwise print over each other: '9.0' and '9.5'
    read '905'). Returns how many were dropped."""
    texts = [t for t in texts if t.figure is not None]     # skip labels already removed
    if not texts:
        return 0
    renderer = fig.canvas.get_renderer()
    kept: List[Any] = []
    dropped = 0
    for t in texts:
        bb = t.get_window_extent(renderer=renderer)
        # boxes that only touch are still readable (v1.5 kept them)
        if any(_overlaps(bb, k, pad=-0.5) for k in kept):
            t.remove()
            dropped += 1
        else:
            kept.append(bb)
    return dropped


def _forecast_bar(color: str) -> Dict[str, Any]:
    """A forecast bar / segment: hollow, dotted and outlined dashed in its
    colour (a value that is not 公表 is striped '////' instead)."""
    return dict(facecolor="white", edgecolor=color, linewidth=1.0, linestyle=FORECAST_DASH,
                hatch=FORECAST_HATCH)


def _segment_hits(p, q, box, pad: float) -> bool:
    """Does the segment p–q (display coordinates, ``pad`` px thick each
    side) cross the box? (Liang–Barsky clipping)"""
    x0, y0, x1, y1 = box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for d, room in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if d == 0:
            if room < 0:
                return False
            continue
        t = room / d
        if d < 0:
            if t > t1:
                return False
            t0 = max(t0, t)
        else:
            if t < t0:
                return False
            t1 = min(t1, t)
    return t0 <= t1


def _mark_geometry(fig, ax, marks: List[Any]) -> Dict[int, Dict[str, List[Any]]]:
    """Per series, in display coordinates: the boxes of its markers and
    bars and the segments of its line."""
    from matplotlib.transforms import Bbox
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    geo: Dict[int, Dict[str, List[Any]]] = {}
    for s, kind, what in marks:
        g = geo.setdefault(s, {"boxes": [], "segs": []})
        if kind == "bar":
            if what.figure is not None:
                g["boxes"].append(what.get_window_extent(renderer=renderer))
            continue
        pts, ms, lw = what
        disp = ax.transData.transform(pts)
        r = ms * px / 2.0 + 1.0
        g["boxes"].extend(Bbox.from_extents(x - r, y - r, x + r, y + r) for x, y in disp)
        g["segs"].extend((p, q, lw * px / 2.0) for p, q in zip(disp, disp[1:]))
    return geo


def _hits_marks(bb, geo: Dict[int, Dict[str, List[Any]]], own: Optional[int], with_own: bool) -> bool:
    for s, g in geo.items():
        if s == own and not with_own:
            continue
        if any(_overlaps(bb, b, pad=-1.0) for b in g["boxes"]):
            return True
        if any(_segment_hits(p, q, bb, pad - 1.0) for p, q, pad in g["segs"]):
            return True
    return False


def _avoid_marks(fig, ax, labels: List[Any], label_series: Dict[int, int], label_moves: Dict[int, List[Any]],
                 marks: List[Any], protected: set, drop: bool) -> int:
    """Point / value labels (in rank order: the emphasised series first)
    never sit on another series' marker, line or bar, nor on a label
    already placed. A point label tries its own place, then below / above,
    then right of its point (a moved label also stays off its own line and
    inside the plot); a bar's label stays at the bar end. A label with no
    free place goes back to its own place, and with ``drop`` is removed —
    except the emphasised series' (``protected``). A label of no series (a
    stack total, a segment label) stays where it is, as an obstacle.
    Returns how many were removed."""
    labels = [t for t in labels if t.figure is not None and id(t) in label_series]
    if not labels or not marks:
        return 0
    renderer = fig.canvas.get_renderer()
    geo = _mark_geometry(fig, ax, marks)
    plot = ax.get_window_extent(renderer=renderer)
    handled = {id(t) for t in labels}
    kept = [t.get_window_extent(renderer=renderer) for t in ax.texts
            if id(t) not in handled and t.get_visible() and t.get_text()]

    def place(t, where):
        if where is not None:
            t.xyann = where[0]
            t.set_va(where[1])
            t.set_ha(where[2])

    removed = 0
    for t in labels:
        own = label_series.get(id(t))
        places = label_moves.get(id(t)) or [None]
        chosen = None
        for k, where in enumerate(places):
            place(t, where)
            bb = t.get_window_extent(renderer=renderer)
            if k > 0 and not (plot.x0 - 1 <= bb.x0 and bb.x1 <= plot.x1 + 1
                              and plot.y0 - 1 <= bb.y0 and bb.y1 <= plot.y1 + 1):
                continue
            if _hits_marks(bb, geo, own, with_own=True):
                continue
            if any(_overlaps(bb, o, pad=-0.5) for o in kept):
                continue
            chosen = bb
            break
        if chosen is None:
            place(t, places[0])
            if drop and id(t) not in protected:
                t.remove()
                removed += 1
                continue
            chosen = t.get_window_extent(renderer=renderer)
        kept.append(chosen)
    return removed


def _labels_on_marks(fig, ax, labels: List[Any], label_series: Dict[int, int], marks: List[Any]) -> int:
    """(QA) drawn labels that sit on another series' marker, line or bar."""
    live = [t for t in labels if t.figure is not None and t.get_visible() and t.get_text()
            and id(t) in label_series]
    if not live or not marks:
        return 0
    renderer = fig.canvas.get_renderer()
    geo = _mark_geometry(fig, ax, marks)
    return sum(1 for t in live
               if _hits_marks(t.get_window_extent(renderer=renderer), geo, label_series.get(id(t)), False))


def _value_range(series: List[Dict[str, Any]], line: bool) -> (float, float):
    vals: List[float] = []
    for s in series:
        for v in (s.get("values") or []):
            if line and isinstance(v, dict):
                v = v.get("y")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.append(float(v))
    if not vals:
        return 0.0, 0.0
    return min(vals), max(vals)


def _point_label(v: Any, labels: List[Any], j: int) -> str:
    """The label a value is plotted at ('2024' for a line_xy point at x=2024.0)."""
    if isinstance(v, dict):
        x = v.get("x")
        return str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
    return str(labels[j]) if j < len(labels) else str(j)


def forecast_flags(series: Dict[str, Any], labels: Optional[List[Any]] = None) -> List[bool]:
    """One flag per value: True for a forecast. The source of each point
    says so (``sources[j]["forecast"]``, always set by the data layer).
    Only when a point has no source dict does the series note count, and
    then only an exact entry '<label>: 予測値' — free text (a quoted source
    sentence that mentions a forecast) never marks actual values."""
    values = series.get("values") or []
    sources = series.get("sources") or []
    labels = list(labels or [])
    noted = set()
    for part in re.split(r"[；;]", str(series.get("note") or "")):
        m = re.fullmatch(r"\s*([^:：]{1,20})\s*[:：]\s*予測値\s*", part)
        if m:
            noted.add(m.group(1).strip())
    out = []
    for j, v in enumerate(values):
        src = sources[j] if j < len(sources) else None
        if isinstance(src, dict) and "forecast" in src:
            out.append(bool(src.get("forecast")))
        else:
            out.append(v is not None and _point_label(v, labels, j) in noted)
    return out


def _line_floor(lo: float, hi: float) -> float:
    """Bottom of a line chart's value axis (no negative values): the data
    range padded by 10 %, or 0 when the values are near 0 (the smallest is
    at most a fifth of the largest) or the line is flat. prelude.js
    (lineFromZero) applies the same rule."""
    span = hi - lo
    if hi <= 0 or span <= 0 or lo <= 0.2 * hi:
        return 0.0
    return max(0.0, lo - 0.1 * span)


def _plain_number(v, _pos=None) -> str:
    """A year / x value as written ('2024', never '2.024e3' or an offset)."""
    r = round(v)
    return str(int(r)) if abs(v - r) < 1e-9 else f"{v:g}"


# the forms this renderer can draw (spec.FORMS_BY_STYLE decides which a
# style may choose)
_DRAWN_FORMS = ("bar", "hbar", "stacked_bar", "stacked_hbar", "line_xy")


def _pick_form(panel: Dict[str, Any], style: str) -> str:
    """enhanced: a vertical bar chart whose labels would collide (many or
    long labels) is drawn as horizontal bars; the panel's semantic form is
    unchanged (recorded as form_rendered)."""
    form = panel.get("form", "bar")
    # several series side by side as horizontal bars would grow the chart
    # by labels x series (far taller than a page): only one series turns
    if style != "enhanced" or form != "bar" or len(panel.get("series") or []) != 1:
        return form
    labels = [str(l) for l in (panel.get("labels") or [])]
    if len(labels) > 6 or any(len(l) > 10 for l in labels):
        panel["form_rendered"] = "hbar"
        return "hbar"
    return form


def render_png(panel: Dict[str, Any], path: Path, language: str = "ja", dpi: Optional[int] = None,
               style: str = "classic", qa: Optional[Dict[str, Any]] = None, embed_text: bool = True) -> Path:
    """Draw one RESOLVED panel to ``path`` (PNG). Returns the path.
    ``qa`` (a dict) receives the layout facts of the drawn figure: value
    axis range, overlapping labels, labels on another series' marks, text
    outside the image, pixel size.
    ``embed_text`` False leaves the title, the 見るべき点 line and the
    caption out of the image (a report prints them once as text around
    it); the legend and its notes stay. True (the default) draws them, for
    an image used on its own."""
    with _RENDER_LOCK:
        return _render_locked(panel, Path(path), language, dpi, style if style in STYLES else "classic", qa,
                              bool(embed_text))


def _render_locked(panel: Dict[str, Any], path: Path, language: str, dpi: Optional[int],
                   style: str, qa: Optional[Dict[str, Any]] = None, embed_text: bool = True) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    L = _LAYOUT[style]
    with plt.rc_context():
        setup_japanese_font(plt, fm, style)
        plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                             "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK,
                             "ytick.color": INK, "font.size": 10 if style == "classic" else 8.5,
                             "figure.facecolor": "white"})
        form = _pick_form(panel, style)
        if form not in _DRAWN_FORMS:
            # never drawn as bars by default: the caller counts it as failed
            raise ValueError(f"{panel.get('id')}: この形式は描画できません（{form!r}）")
        series = panel.get("series") or []
        labels = list(panel.get("labels") or [])
        unit = str(panel.get("unit") or (series[0].get("unit") if series else "") or "")
        n_series = len(series)
        signed = _signed_pct(panel)
        fmt = lambda v: _fmt(v, unit, signed)  # noqa: E731
        horiz = form.endswith("hbar")
        stacked = form.startswith("stacked")
        seg_row = form == "stacked_hbar"
        line = form == "line_xy"
        lo, hi = _value_range(series, line)

        # a category name too long for the left margin is wrapped (it was cut
        # off); each extra line adds to the row height
        cat_labels = [_wrap(l, L["cat_wrap"]) for l in labels] if horiz else labels
        rows_h = sum(1.0 + 0.6 * l.count("\n") for l in cat_labels) if horiz else len(labels)
        if style == "classic":                      # the v1.5 heights
            if seg_row:
                fig_h = max(2.6, 0.55 * max(1, rows_h) + 2.0)
            elif horiz:
                fig_h = max(3.2, 0.45 * max(1, rows_h) * max(1, n_series) + 2.0)
            else:
                fig_h = L["bar_h"]
        else:
            scale = L["width"] / 10.0
            if seg_row:
                fig_h = max(2.6 * scale + 0.6, 0.55 * scale * max(1, rows_h) + 2.0 * scale + 0.4)
            elif horiz:
                # the whole chart stays within a page (10.5 in): a category
                # gets at most 0.6 in and shrinks down to 0.16 in (bars get
                # thinner) before the chart grows past the page
                budget = 10.5 - 0.9 - (2.0 * scale + 0.6)
                per_cat = min(0.45 * scale * max(1, n_series), 0.6, max(0.16, budget / max(1.0, rows_h)))
                fig_h = max(3.2 * scale + 0.6, per_cat * max(1, rows_h) + 2.0 * scale + 0.6)
            else:
                fig_h = L["bar_h"]
            # room for the legend / caption / judgment lines under the plot
            # (only the legend when the text is printed outside the image)
            fig_h += 0.9 if embed_text else 0.45
        fig, ax = plt.subplots(figsize=(L["width"], fig_h))
        forecast = [forecast_flags(s, labels) for s in series]
        any_forecast = any(any(f) for f in forecast)
        # a value that is neither 公表 nor a forecast: striped bar / hollow point
        any_estimated = any(not _published(b) and not (j < len(f) and f[j])
                            for s, f in zip(series, forecast) for j, b in enumerate(s.get("basis") or []))
        handles: List[Any] = []
        point_labels: List[Any] = []
        value_labels: List[Any] = []
        seg_labels: List[Any] = []
        label_rank: Dict[int, int] = {}     # id(text) -> 0 for the emphasised series
        role_rank = {"emphasis": 0, "part": 1, "context": 2, "context2": 3}
        # what each series drew (markers / lines / bars) and which series a
        # label belongs to: a label must not sit on another series' marks
        marks: List[Any] = []
        label_series: Dict[int, int] = {}
        label_moves: Dict[int, List[Any]] = {}  # id(point label) -> its places, the first is its own

        if line:
            lw0 = 2 if style == "classic" else 1.6
            ms = 6 if style == "classic" else 4.5
            for i, s in enumerate(series):
                color = CAT[i % len(CAT)]
                marker = MARKERS[i % len(MARKERS)]
                basis = s.get("basis") or []
                fcs = forecast[i]
                pts = [(float(v["x"]), float(v["y"]), basis[j] if j < len(basis) else None,
                        bool(j < len(fcs) and fcs[j]))
                       for j, v in enumerate(s.get("values") or []) if isinstance(v, dict)]
                if not pts:
                    continue
                pts.sort(key=lambda p: p[0])            # (basis / forecast stay with their point)
                # actual values: a solid line in the series' own colour (a second
                # series is not dashed: dashes mean a forecast); a segment into or
                # out of a forecast is dashed; the emphasised series is drawn
                # thicker, on top
                emph = s.get("role") == "emphasis" and n_series > 1
                lw = lw0 * (1.3 if emph else 1.0)
                z = 3 if emph else 2
                runs: List[Any] = []
                for a, b in zip(pts, pts[1:]):
                    dashed = a[3] or b[3]
                    if runs and runs[-1][0] == dashed:
                        runs[-1][1].append(b)
                    else:
                        runs.append((dashed, [a, b]))
                for dashed, run in runs:
                    ax.plot([p[0] for p in run], [p[1] for p in run], color=color, linewidth=lw,
                            linestyle=FORECAST_DASH if dashed else "-", zorder=z)
                # a forecast, or a value that is not 公表, is a hollow marker
                solid = [(x, y) for x, y, b, f in pts if _published(b) and not f]
                hollow = [(x, y) for x, y, b, f in pts if not _published(b) or f]
                if solid:
                    ax.plot(*zip(*solid), linestyle="none", marker=marker, markersize=ms, color=color,
                            zorder=z + 0.5)
                if hollow:
                    ax.plot(*zip(*hollow), linestyle="none", marker=marker, markersize=ms + 1,
                            markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.4,
                            zorder=z + 0.5)
                marks.append((i, "line", ([(p[0], p[1]) for p in pts], ms + 1, lw)))
                for x, y, _b, _f in pts:
                    # (v1.5 placement above the point; a negative point's label
                    # goes under it — above, it sat on the line)
                    up = (((0, 7), "baseline", "center"), ((0, -7), "top", "center"))
                    places = list(up if y >= 0 else up[::-1]) + [((7, 0), "center", "left")]
                    t = ax.annotate(fmt(y), (x, y), textcoords="offset points", xytext=places[0][0],
                                    va=places[0][1], ha=places[0][2], fontsize=L["value"], color=INK)
                    point_labels.append(t)
                    label_rank[id(t)] = role_rank.get(s.get("role", "context"), 2)
                    label_series[id(t)] = i
                    label_moves[id(t)] = places
                handles.append(Line2D([0], [0], color=color, marker=marker, markersize=ms, linewidth=lw,
                                      label=s.get("name", "")))
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_plain_number))
            ax.set_ylabel(unit)
            # negative values (a fall in growth, a loss) must stay visible
            if lo < 0:
                span = (max(hi, 0.0) - lo) or abs(lo)
                ax.set_ylim(bottom=lo - 0.08 * span, top=panel.get("y_max") or None)
                ax.axhline(0, color="#b9b8b1", linewidth=0.8)
            else:
                # the data range, padded (a line is read by its change; 0 only
                # when the values are near it)
                ax.set_ylim(bottom=_line_floor(lo, hi), top=panel.get("y_max") or None)
            ax.grid(axis="y", color=GRID)
        elif seg_row:
            # one segment row PER LABEL (the HTML page draws the same rows);
            # a missing value is an empty segment, never a later label's value
            rows = max(1, len(labels))
            lefts = [0.0] * rows
            widest = 1e-9
            for i, s in enumerate(series):
                fill, text_color = SEG[i % len(SEG)]
                handles.append(Patch(facecolor=fill, label=s.get("name", "")))
                vals = s.get("values") or []
                basis = s.get("basis") or []
                fcs = forecast[i]
                for j in range(rows):
                    v = vals[j] if j < len(vals) else None
                    if not isinstance(v, (int, float)) or isinstance(v, bool):
                        continue
                    v = float(v)
                    b = basis[j] if j < len(basis) else None
                    if j < len(fcs) and fcs[j]:
                        kw = _forecast_bar(text_color)
                    else:
                        kw = dict(color=fill, hatch="////" if not _published(b) else None, edgecolor="white",
                                  linewidth=0.5)
                    bc = ax.barh([j], [v], left=lefts[j], height=0.6, **kw)
                    marks.append((i, "bar", bc.patches[0]))
                    if v > 0:
                        t = ax.text(lefts[j] + v / 2, j, f"{s.get('name', '')} {fmt(v)}",
                                    ha="center", va="center", fontsize=L["value"] + 1, color=text_color)
                        value_labels.append(t)
                        seg_labels.append((t, lefts[j], lefts[j] + v, fmt(v)))
                    lefts[j] += v
                    widest = max(widest, lefts[j])
            y_max = panel.get("y_max")
            ax.set_xlim(0, max(float(y_max) if isinstance(y_max, (int, float)) else 0.0, widest))
            if rows > 1:
                ax.set_yticks(range(rows)); ax.set_yticklabels(cat_labels, fontsize=L["cat"]); ax.invert_yaxis()
            else:
                ax.set_yticks([])
            ax.set_xticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        else:
            idx = list(range(len(labels)))
            width = 0.8 / max(1, n_series) if not stacked else 0.6
            bottoms = [0.0] * len(labels)
            span = (max(hi, 0.0) - min(lo, 0.0)) or 1.0
            for i, s in enumerate(series):
                vals = s.get("values") or []
                basis = s.get("basis") or [None] * len(vals)
                role = s.get("role", "context")
                base = CAT[i % len(CAT)] if role == "part" else ROLE.get(role, ROLE["context"])
                hl = panel.get("highlight") or []
                colors = []
                for j, v in enumerate(vals):
                    if hl:
                        colors.append(ROLE["emphasis"] if labels[j] in hl else ROLE["context"])
                    elif n_series == 1 and isinstance(v, (int, float)) and v < 0:
                        colors.append(ROLE["negative"])
                    else:
                        colors.append(base)
                numeric = [float(v) if isinstance(v, (int, float)) else 0.0 for v in vals]
                present = [isinstance(v, (int, float)) for v in vals]
                fcs = forecast[i]
                if stacked:
                    pos = idx
                else:
                    offset = (i - (n_series - 1) / 2) * width
                    pos = [x + offset for x in idx]
                for j in range(len(labels)):
                    if not present[j]:
                        continue
                    if j < len(fcs) and fcs[j]:
                        kw = _forecast_bar(colors[j])
                    else:
                        kw = dict(color=colors[j], edgecolor="white", linewidth=0.5)
                        if not _published(basis[j]):
                            kw["hatch"] = "////"
                    if horiz:
                        bc = ax.barh(pos[j], numeric[j], height=width, left=bottoms[j] if stacked else 0, **kw)
                    else:
                        bc = ax.bar(pos[j], numeric[j], width=width, bottom=bottoms[j] if stacked else 0, **kw)
                    marks.append((i, "bar", bc.patches[0]))
                    if not stacked:
                        if horiz:
                            gap = 0.01 * span
                            if numeric[j] < 0:      # the label sits left of a negative bar
                                t = ax.text(numeric[j] - gap, pos[j], fmt(numeric[j]) + " ",
                                            va="center", ha="right", fontsize=L["value"], color=INK)
                            else:
                                # classic: the v1.5 gap (1% of the bar + 0.02), unless
                                # that fixed 0.02 is large against the data (0.000004)
                                v15 = abs(numeric[j]) * 0.01 + 0.02
                                at = numeric[j] + (v15 if style == "classic" and v15 <= 0.05 * span else gap)
                                t = ax.text(at, pos[j], " " + fmt(numeric[j]),
                                            va="center", ha="left", fontsize=L["value"], color=INK)
                        else:
                            t = ax.text(pos[j], numeric[j], fmt(numeric[j]), ha="center",
                                        va="bottom" if numeric[j] >= 0 else "top",
                                        fontsize=L["value"], color=INK)
                        value_labels.append(t)
                        # highlighted bars and the emphasised series keep their labels first
                        label_rank[id(t)] = 0 if (hl and labels[j] in hl) else role_rank.get(role, 2)
                        label_series[id(t)] = i
                if stacked:
                    bottoms = [bottoms[j] + (numeric[j] if present[j] else 0.0) for j in range(len(labels))]
                handles.append(Patch(facecolor=base, label=s.get("name", "")))
            if stacked:
                for j, total in enumerate(bottoms):
                    if total:
                        if horiz:
                            value_labels.append(ax.text(total, j, " " + fmt(total), va="center", ha="left",
                                                        fontsize=L["value"], color=INK))
                        else:
                            value_labels.append(ax.text(j, total, fmt(total), ha="center", va="bottom",
                                                        fontsize=L["value"], color=INK))
            if horiz:
                # the unit is in the title; an x-axis label would sit where
                # the legend goes
                ax.set_yticks(idx); ax.set_yticklabels(cat_labels, fontsize=L["cat"]); ax.invert_yaxis()
                ax.grid(axis="x", color=GRID)
                if lo < 0:
                    ax.axvline(0, color="#b9b8b1", linewidth=0.8)
                if panel.get("y_max"):
                    # never cut a negative bar: the left edge follows the data
                    left = lo - 0.12 * span if lo < 0 else 0
                    ax.set_xlim(left, float(panel["y_max"]))
                elif lo < 0:
                    ax.set_xlim(lo - 0.15 * span, max(hi, 0.0) + 0.15 * span)
            else:
                wrap_w = 10 if style == "classic" else 8
                ax.set_xticks(idx); ax.set_xticklabels([_wrap(l, wrap_w) for l in labels], fontsize=L["tick"])
                ax.set_ylabel(unit); ax.grid(axis="y", color=GRID)
                if panel.get("y_max"):
                    ax.set_ylim(top=float(panel["y_max"]))
                ax.axhline(0, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)

        if not seg_row:
            # both styles: plain numbers with thousands separators on the
            # value axis (matplotlib's own formatter printed '0.5 … 3.5' and
            # a '1e6' offset), as many decimals as the tick step needs (1.25
            # never prints as 1.2); classic keeps its tick size
            axis = ax.xaxis if horiz else ax.yaxis
            axis.set_major_formatter(mticker.FuncFormatter(_step_formatter(axis)))
            if style == "enhanced":
                ax.tick_params(labelsize=L["tick"])

        if embed_text:
            ax.set_title(str(panel.get("title", "")), fontsize=L["title"], loc="left", color="#0b0b0b", pad=10)
        # Below the axes, bottom-up, in INCHES: judgment, caption, legend —
        # then the axes bottom margin is set above all of them so tick
        # labels, legend and text never overlap; a long caption or legend
        # makes the figure taller instead of squeezing the plot away.
        line_in = 0.16 if style == "classic" else 0.14
        text_x = 0.05 if style == "classic" else 0.02
        judgment_text = str(panel.get("judgment", "") or "") if embed_text else ""
        caption_text = str(panel.get("caption", "") or "") if embed_text else ""
        if style == "classic":
            # the v1.5 wrap ('見るべき点: ' + 70 characters), narrowed only
            # when the measured line would run past the image edge
            wrap_judgment = lambda w: ("見るべき点: " + _wrap(judgment_text, w)) if judgment_text else ""  # noqa: E731
            j_width = L["wrap"]
        else:
            # larger type than the caption and carries its prefix: the whole
            # line is wrapped to the width that is left
            wrap_judgment = lambda w: _wrap("見るべき点: " + judgment_text, w) if judgment_text else ""  # noqa: E731
            j_width = int((L["width"] * (1 - text_x) - 0.15) * 72 / L["judgment"]) - 2
        # (without the text the legend sits at the bottom edge)
        y_in = 0.4 if style == "classic" and embed_text else 0.12
        # classic also keeps the v1.5 arithmetic in figure fractions, so an
        # unchanged layout lands on exactly the same pixels
        line_f = 0.16 / fig_h
        y_f = y_in / fig_h
        placed: List[Any] = []                       # (artist, y in inches, v1.5 y fraction)
        if judgment_text:
            t = fig.text(text_x, 0, "", fontsize=L["judgment"], color="#0b0b0b", va="bottom")
            judgment = _fit_text_width(fig, t, wrap_judgment, j_width)
            placed.append((t, y_in, y_f))
            y_in += line_in * (judgment.count("\n") + 1) + line_in * 0.4
            y_f += line_f * (judgment.count("\n") + 1) + line_f * 0.4
        if caption_text:
            t = fig.text(text_x, 0, "", fontsize=L["caption"], color="#898781", va="bottom")
            caption = _fit_text_width(fig, t, lambda w: _wrap(caption_text, w), L["wrap"])
            placed.append((t, y_in, y_f))
            y_in += line_in * (caption.count("\n") + 1) + line_in * 0.6
            y_f += line_f * (caption.count("\n") + 1) + line_f * 0.6
        legend = None
        # one series needs no legend (the title names it); the notes stay
        handles = handles if n_series > 1 else []
        if any_estimated:
            handles.append(Line2D([0], [0], linestyle="none", marker="o", markersize=7, markerfacecolor="white",
                                  markeredgecolor=INK, markeredgewidth=1.4, label=HOLLOW_NOTE) if line
                           else Patch(facecolor="white", edgecolor=INK, hatch="////", label=STRIPE_NOTE))
        if any_forecast:
            handles.append(Line2D([0], [0], color=INK, linestyle=FORECAST_DASH, marker="o", markersize=7,
                                  markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.4,
                                  label=FORECAST_NOTE) if line
                           else Patch(label=FORECAST_NOTE, **_forecast_bar(INK)))
        if handles:
            ncol = min(4 if style == "classic" else 3, len(handles))
            renderer = fig.canvas.get_renderer()
            wrapped_names = False
            while True:
                legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0),
                                    ncol=ncol, frameon=False, fontsize=L["legend"])
                width_px = legend.get_window_extent(renderer=renderer).width
                # long series names: fewer columns rather than a legend cut at the edge
                if width_px <= fig.bbox.width * 0.97:
                    break
                legend.remove()
                if ncol > 1:
                    ncol -= 1
                    continue
                if wrapped_names:
                    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0),
                                        ncol=1, frameon=False, fontsize=L["legend"])
                    break
                # one column and still too wide: the names themselves wrap
                longest = max(len(h.get_label()) for h in handles) or 1
                per_line = max(8, int(longest * fig.bbox.width * 0.85 / max(1.0, width_px)))
                for h in handles:
                    h.set_label(_wrap(h.get_label(), per_line))
                wrapped_names = True
            legend_y, legend_f = y_in, y_f
            rows = -(-len(handles) // ncol)
            extra = sum(h.get_label().count("\n") for h in handles) // max(1, ncol)
            y_in += line_in * 1.3 * (rows + extra)   # legend rows
            y_f += line_f * 1.3 * (rows + extra)
        ticks_in = 0.0 if seg_row else 0.45                # tick labels under the axes
        bottom_in = y_in + ticks_in + line_in * 0.5
        bottom_f = y_f + (0.0 if seg_row else 0.45 / fig_h) + line_f * 0.5
        top = (0.86 if style == "classic" else 0.88) if embed_text else 0.95     # (no title: room above)
        # classic grows only where v1.5 hit its cap (bottom margin > 75 %:
        # the legend ran into the tick labels); enhanced keeps a minimum plot
        squeezed = (bottom_f > 0.75) if style == "classic" \
            else (fig_h * top - bottom_in < L["min_plot"])
        if squeezed:
            fig_h = max(fig_h, (bottom_in + L["min_plot"]) / top)
            fig.set_size_inches(L["width"], fig_h)
        as_v15 = style == "classic" and not squeezed
        for artist, at, at_f in placed:
            artist.set_y(at_f if as_v15 else at / fig_h)
        if legend is not None:
            legend.set_bbox_to_anchor((0.5, legend_f if as_v15 else legend_y / fig_h),
                                      transform=fig.transFigure)
        fig.subplots_adjust(bottom=min(0.75, bottom_f) if as_v15 else bottom_in / fig_h, top=top,
                            left=0.09 if style == "classic" else 0.12, right=0.98)

        fig.canvas.draw()
        by_rank = lambda ts: sorted(ts, key=lambda t: label_rank.get(id(t), 1))  # noqa: E731
        protected = {k for k, r in label_rank.items() if r == 0}
        # a label on another series' marker / line / bar moves (above →
        # below → right) where it is free; then close points: drop the labels
        # that would print over each other (the emphasised series keeps its labels)
        _avoid_marks(fig, ax, by_rank(point_labels + value_labels), label_series, label_moves, marks,
                     protected, drop=False)
        _drop_overlapping(fig, by_rank(point_labels))
        _fit_left_margin(fig, ax)
        _fit_title(fig, ax)
        if seg_labels:
            _fit_segment_labels(fig, ax, seg_labels)
        elif value_labels or point_labels:
            _fit_value_labels(fig, ax, value_labels + point_labels, horiz, bool(panel.get("y_max")))
        if not seg_row:
            _thin_value_ticks(fig, ax, horiz)
        if not horiz:
            _rotate_crowded_ticks(fig, ax)
        _fit_axis_edges(fig, ax)
        _fit_left_margin(fig, ax)                    # ticks may have widened meanwhile
        if not seg_row:
            _thin_value_ticks(fig, ax, horiz)        # the plot may have narrowed
        # the final layout: a label still on another series' marks moves or,
        # unless it is the emphasised series', is dropped; labels that touch
        # now (the plot may have narrowed) are dropped too
        _avoid_marks(fig, ax, by_rank(point_labels + value_labels), label_series, label_moves, marks,
                     protected, drop=True)
        _drop_overlapping(fig, by_rank(point_labels))
        _drop_overlapping(fig, by_rank(value_labels))
        _qa_layout(fig, ax, style)
        if qa is not None:
            qa.update(_layout_facts(fig, ax, horiz, seg_row, dpi or L["dpi"]))
            qa["label_on_marks"] = _labels_on_marks(fig, ax, point_labels + value_labels, label_series, marks)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi or L["dpi"], facecolor="white")
        plt.close(fig)
        return path


def _step_formatter(axis):
    """Tick text with thousands separators and the decimals the tick step
    needs (0.25 steps print 1.25, 1,000 steps print 12,000)."""
    def fmt(v, _pos=None):
        locs = list(axis.get_majorticklocs())
        steps = [abs(b - a) for a, b in zip(locs, locs[1:]) if b != a]
        step = min(steps) if steps else (abs(v) or 1.0)
        d = 0
        while d < 15 and abs(step * 10 ** d - round(step * 10 ** d)) > 1e-6 * abs(step * 10 ** d):
            d += 1
        if d >= 15:                                  # no exact decimal step: significant digits
            return f"{v:.3g}"
        text = f"{v:,.{d}f}"
        return "0" if not text.strip("-0.,") else text      # never '-0'
    return fmt


def _title_artist(ax):
    """The title text actually drawn (set with loc='left')."""
    for t in (getattr(ax, "_left_title", None), ax.title, getattr(ax, "_right_title", None)):
        if t is not None and t.get_text():
            return t
    return ax.title


def _shown_ticks(axis, lim) -> List[Any]:
    """Tick labels inside the axis limits (the others are never drawn)."""
    lo, hi = sorted(lim)
    span = (hi - lo) or 1.0
    return [t for loc, t in zip(axis.get_ticklocs(), axis.get_ticklabels())
            if t.get_visible() and t.get_text() and lo - 1e-6 * span <= loc <= hi + 1e-6 * span]


def _fit_text_width(fig, artist, wrap, width: int) -> str:
    """Set ``wrap(width)`` on a figure text, narrowing ``width`` while the
    measured line runs past the image edge. Returns the text."""
    renderer = fig.canvas.get_renderer()
    limit = fig.bbox.x1 - 6
    text = wrap(width)
    for _ in range(12):
        artist.set_text(text)
        box = artist.get_window_extent(renderer=renderer)
        if box.x1 <= limit or width <= 8:
            break
        width = max(8, int(width * (limit - box.x0) / max(1.0, box.width)) - 1)
        text = wrap(width)
    return text


def _fit_left_margin(fig, ax) -> None:
    """Category names, value ticks ('12,345,678') and the unit label set
    the left margin (a fixed margin cut them off at the image edge)."""
    renderer = fig.canvas.get_renderer()
    items = _shown_ticks(ax.yaxis, ax.get_ylim())
    if ax.yaxis.label.get_text():
        items.append(ax.yaxis.label)
    if not items:
        return
    x0 = min(t.get_window_extent(renderer=renderer).x0 for t in items)
    if x0 >= 4:
        return
    sp = fig.subplotpars
    fig.subplots_adjust(left=min(0.5, sp.left + (4 - x0) / fig.bbox.width))
    fig.canvas.draw()


def _fit_title(fig, ax) -> None:
    """A title wider than the image is wrapped (and the plot moves down)."""
    renderer = fig.canvas.get_renderer()
    title = _title_artist(ax)
    text = title.get_text()
    if not text:
        return
    box = title.get_window_extent(renderer=renderer)
    avail = fig.bbox.x1 - 6 - box.x0
    if box.x1 <= fig.bbox.x1 - 2 or avail <= 0:
        return
    per_line = max(8, int(len(text) * avail / max(1.0, box.width)))
    for _ in range(6):
        title.set_text(_wrap(text, per_line))
        fig.canvas.draw()
        box = title.get_window_extent(renderer=renderer)
        if box.x1 <= fig.bbox.x1 - 2 or per_line <= 8:
            break
        per_line -= 2
    if box.y1 > fig.bbox.y1 - 2:
        sp = fig.subplotpars
        fig.subplots_adjust(top=max(sp.bottom + 0.2, sp.top - (box.y1 - fig.bbox.y1 + 6) / fig.bbox.height))
        fig.canvas.draw()


def _fit_value_labels(fig, ax, labels: List[Any], horiz: bool, fixed_max: bool) -> None:
    """Value / point labels that would be CUT (past the image edge), print
    over the title, or print over the category names / tick labels: the
    axis is extended to hold them (or, for a horizontal chart whose maximum
    the panel fixes, the plot narrows). A label inside the image is left
    where v1.5 drew it."""
    renderer = fig.canvas.get_renderer()
    for _ in range(4):
        live = [t for t in labels if t.figure is not None and t.get_visible() and t.get_text()]
        if not live:
            return
        box = ax.get_window_extent(renderer=renderer)
        ends = [t.get_window_extent(renderer=renderer) for t in live]
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        changed = False
        # right: past the image edge
        over_r = max(b.x1 for b in ends) - (fig.bbox.x1 - 2)
        if over_r > 0:
            if horiz and fixed_max:
                sp = fig.subplotpars
                fig.subplots_adjust(right=max(sp.left + 0.3, sp.right - (over_r + 4) / fig.bbox.width))
            else:
                x1 += (x1 - x0) * (over_r + 4) / max(1.0, box.width - over_r - 4)
            changed = True
        # left: over the category names / value ticks (a negative bar's label,
        # kept 4 px clear of the axis so its tick does not read as a minus)
        under_l = (box.x0 + (4 if horiz else 0)) - min(b.x0 for b in ends)
        if under_l > 1:
            x0 -= (x1 - x0) * (under_l + 4) / max(1.0, box.width - under_l - 4)
            changed = True
        if not horiz:
            # top: past the image edge, or ON the title (both ways overlapping)
            title = _title_artist(ax)
            tb = title.get_window_extent(renderer=renderer) if title.get_text() else None
            over_t = max(b.y1 for b in ends) - (fig.bbox.y1 - 2)
            if tb is not None:
                hits = [b for b in ends if _overlaps(b, tb, pad=0.0)]
                if hits:
                    over_t = max(over_t, max(b.y1 for b in hits) - (tb.y0 - 1))
            if over_t > 0:
                y1 += (y1 - y0) * (over_t + 4) / max(1.0, box.height - over_t - 4)
                changed = True
            # bottom: under the axis, over the category / tick labels
            under_b = box.y0 - min(b.y0 for b in ends)
            if under_b > 1:
                y0 -= (y1 - y0) * (under_b + 4) / max(1.0, box.height - under_b - 4)
                changed = True
        if not changed:
            return
        # never more than double the range in one pass (a stray label must
        # not shrink the bars to nothing)
        ox0, ox1 = ax.get_xlim()
        oy0, oy1 = ax.get_ylim()
        cap_x, cap_y = abs(ox1 - ox0), abs(oy1 - oy0)
        ax.set_xlim(max(x0, ox0 - cap_x), min(x1, ox1 + cap_x))
        ax.set_ylim(max(y0, oy0 - cap_y), min(y1, oy1 + cap_y))
        fig.canvas.draw()


def _fit_axis_edges(fig, ax) -> None:
    """A drawn tick label at either end of the horizontal axis ('3,000')
    must not be cut at the image edge."""
    renderer = fig.canvas.get_renderer()
    ticks = [t.get_window_extent(renderer=renderer) for t in _shown_ticks(ax.xaxis, ax.get_xlim())]
    if not ticks:
        return
    sp = fig.subplotpars
    spill_r = max(b.x1 for b in ticks) - (fig.bbox.x1 - 2)
    spill_l = (fig.bbox.x0 + 2) - min(b.x0 for b in ticks)
    if spill_r <= 0 and spill_l <= 0:
        return
    fig.subplots_adjust(right=sp.right - max(0.0, spill_r + 2) / fig.bbox.width,
                        left=sp.left + max(0.0, spill_l + 2) / fig.bbox.width)
    fig.canvas.draw()


def _thin_value_ticks(fig, ax, horiz: bool) -> None:
    """Value-axis tick labels that print over each other ('1,000,000' on a
    narrow horizontal axis) get fewer ticks."""
    import matplotlib.ticker as mticker
    renderer = fig.canvas.get_renderer()
    axis = ax.xaxis if horiz else ax.yaxis
    lim = ax.get_xlim() if horiz else ax.get_ylim()
    for _ in range(8):
        ticks = _shown_ticks(axis, lim)
        boxes = [t.get_window_extent(renderer=renderer) for t in ticks]
        if len(boxes) <= 2 or not any(_overlaps(a, b, pad=2.0) for a, b in zip(boxes, boxes[1:])):
            return
        axis.set_major_locator(mticker.MaxNLocator(nbins=max(2, len(boxes) - 2)))
        fig.canvas.draw()


def _fit_segment_labels(fig, ax, labels: List[Any]) -> None:
    """A segment label wider than its segment ('B社 3%' in a 3% slice)
    printed over the next one: it shrinks to the value alone, and is
    dropped when even that does not fit (the legend still names it)."""
    renderer = fig.canvas.get_renderer()
    for t, x0, x1, short in labels:
        # (a label that just fits its segment, as v1.5 drew it, stays)
        width = abs(ax.transData.transform((x1, 0))[0] - ax.transData.transform((x0, 0))[0]) + 2
        if t.get_window_extent(renderer=renderer).width <= width:
            continue
        t.set_text(short)
        if t.get_window_extent(renderer=renderer).width > width:
            t.remove()


def _angled_overlap(fig, ticks: List[Any]) -> bool:
    for a, b in zip(ticks, ticks[1:]):
        rot = math.radians(a.get_rotation() % 180)
        xa = a.get_transform().transform(a.get_position())[0]
        xb = b.get_transform().transform(b.get_position())[0]
        if abs(xb - xa) * math.sin(rot) < a.get_fontsize() * fig.dpi / 72.0:
            return True
    return False


def _rotate_crowded_ticks(fig, ax) -> None:
    """Category labels that print over each other are set at 30°, then
    upright, and when even that is too dense only every n-th one is
    shown; the plot moves up when the angled labels reach the legend."""
    renderer = fig.canvas.get_renderer()
    ticks = [t for t in ax.get_xticklabels() if t.get_text()]
    boxes = [t.get_window_extent(renderer=renderer) for t in ticks]
    if not any(_overlaps(a, b) for a, b in zip(boxes, boxes[1:])):
        return
    before = min(b.y0 for b in boxes)
    for t in ticks:
        t.set_rotation(30)
        t.set_ha("right")
        t.set_rotation_mode("anchor")
    fig.canvas.draw()
    if _angled_overlap(fig, ticks):
        for t in ticks:
            t.set_rotation(90)
        fig.canvas.draw()
        if _angled_overlap(fig, ticks) and len(ticks) > 1:
            xa = ticks[0].get_transform().transform(ticks[0].get_position())[0]
            xb = ticks[1].get_transform().transform(ticks[1].get_position())[0]
            gap = max(1.0, abs(xb - xa))
            every = math.ceil(ticks[0].get_fontsize() * fig.dpi / 72.0 * 1.2 / gap)
            for i, t in enumerate(ticks):
                t.set_visible(i % every == 0)
            fig.canvas.draw()
    after = min(t.get_window_extent(renderer=renderer).y0 for t in ticks if t.get_visible())
    if after < before:
        sp = fig.subplotpars
        fig.subplots_adjust(bottom=min(0.8, sp.bottom + (before - after) / fig.bbox.height))
        fig.canvas.draw()


def _qa_layout(fig, ax, style: str) -> None:
    """Last pass: a figure text still past the edge is re-wrapped; a label
    still past the edge is pulled inside. Only an enhanced chart where
    that is not enough grows wider (it then prints a little smaller)."""
    renderer = fig.canvas.get_renderer()
    fb = fig.bbox

    def out(t):
        b = t.get_window_extent(renderer=renderer)
        return b.x0 < fb.x0 - 1 or b.x1 > fb.x1 + 1
    for t in fig.texts:
        if t.get_visible() and t.get_text() and out(t):
            raw = t.get_text().replace("\n", "")
            _fit_text_width(fig, t, lambda w: _wrap(raw, w), max(8, len(raw)))
    for t in ax.texts:
        if t.get_visible() and t.get_text() and out(t):
            b = t.get_window_extent(renderer=renderer)
            t.set_ha("right" if b.x1 > fb.x1 else "left")
    fig.canvas.draw()
    if style == "enhanced" and any(t.get_visible() and t.get_text() and out(t) for t in fig.texts + ax.texts):
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * 1.15, h)
        fig.canvas.draw()


def _layout_facts(fig, ax, horiz: bool, seg_row: bool, dpi: int) -> Dict[str, Any]:
    """What a reader would see go wrong, measured on the drawn figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [t for t in ax.texts if t.get_visible() and t.get_text()]
    boxes = [t.get_window_extent(renderer=renderer) for t in labels]
    overlaps = sum(1 for i in range(len(boxes)) for j in range(i + 1, len(boxes))
                   if _overlaps(boxes[i], boxes[j], pad=-1.0))

    xt = _shown_ticks(ax.xaxis, ax.get_xlim())
    yt = _shown_ticks(ax.yaxis, ax.get_ylim())
    tick_overlaps = 0
    for a, b in zip(xt, xt[1:]):
        rot = math.radians(a.get_rotation() % 180)
        if rot:
            # angled labels are parallel: they collide when the gap across
            # them is below one line of text (their boxes always overlap)
            ax_a = a.get_transform().transform(a.get_position())[0]
            ax_b = b.get_transform().transform(b.get_position())[0]
            if abs(ax_b - ax_a) * math.sin(rot) < a.get_fontsize() * fig.dpi / 72.0:
                tick_overlaps += 1
        elif _overlaps(a.get_window_extent(renderer=renderer),
                       b.get_window_extent(renderer=renderer), pad=-1.0):
            tick_overlaps += 1
    texts = list(fig.texts) + labels + xt + yt + [t for t in (_title_artist(ax), ax.xaxis.label,
                                                               ax.yaxis.label)
                                                  if t.get_visible() and t.get_text()]
    for leg in fig.legends:
        texts.extend(leg.get_texts())
    fb = fig.bbox
    spill = []
    for t in texts:
        b = t.get_window_extent(renderer=renderer)
        if b.x0 < fb.x0 - 1 or b.x1 > fb.x1 + 1 or b.y0 < fb.y0 - 1 or b.y1 > fb.y1 + 1:
            spill.append(t.get_text())
    title = _title_artist(ax)
    tb = title.get_window_extent(renderer=renderer) if title.get_text() else None
    on_title = sum(1 for b in boxes if tb is not None and _overlaps(b, tb, pad=-1.0))
    tick_boxes = [t.get_window_extent(renderer=renderer) for t in yt + xt]
    on_ticks = sum(1 for b in boxes if any(_overlaps(b, k, pad=-1.0) for k in tick_boxes))
    w, h = fig.get_size_inches()
    return {"value_lim": tuple(ax.get_xlim() if (horiz or seg_row) else ax.get_ylim()),
            "label_on_title": on_title, "label_on_ticks": on_ticks,
            "label_overlaps": overlaps, "tick_overlaps": tick_overlaps, "spill": spill,
            "size_px": (round(w * dpi), round(h * dpi)),
            "plot_in": round(ax.get_position().height * h, 3)}


def layout_report(png_path: Path) -> Dict[str, Any]:
    """(for tests) image size of a rendered PNG."""
    from PIL import Image
    with Image.open(png_path) as im:
        return {"width": im.width, "height": im.height}
