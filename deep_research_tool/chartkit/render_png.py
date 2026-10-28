"""
PNG renderer (matplotlib) with the SAME rules as the HTML page, for the
Markdown / Word / PDF outputs: one emphasised series in blue, context in
grey, negative values red, category colours for parts and line_xy,
striped fill for every value that is not 公表, value labels, the unit in
the title, a single axis, a legend built from the roles, and the caption
and judgment under the chart.

Two styles (``report.chart_style``):

- ``classic``  — the v1.5 look (10 in wide, the same fonts and layout)
- ``enhanced`` — drawn at the size it is printed (5.5 in, the width Word
  and PDF use, so the text is not shrunk), thousands separators on the
  value axis, a bar chart with many or long labels drawn as horizontal
  bars, and a layout QA pass (overlapping tick labels rotated, the plot
  enlarged when text spills out of the figure)

Both styles get the correctness fixes: negative values are drawn (the
value axis is no longer pinned at 0), overlapping point labels are
dropped instead of printed over each other, long category names set the
left margin, small numbers keep their digits (0.04, not 0.0), text never
breaks inside a number, and rendering is serialized (pyplot keeps global
state; the Web UI runs several jobs in one process).
"""

from __future__ import annotations

import math
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

ROLE = {"emphasis": "#2a78d6", "context": "#888780", "context2": "#B4B2A9", "negative": "#e34948"}
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEG = [("#85B7EB", "#042C53"), ("#F0997B", "#4A1B0C"), ("#B4B2A9", "#2C2C2A"), ("#A9D8C2", "#0E3D2A")]
MARKERS = ["o", "s", "^", "D", "*", "x"]
DASHES = ["-", "--", ":", "-.", (0, (4, 4)), (0, (1, 2))]
INK = "#52514e"
GRID = "#e1e0d9"
STRIPE_NOTE = "斜線＝換算・導出・推定・二次"

STYLES = ("classic", "enhanced")
# classic: the v1.5 sizes; enhanced: drawn at the printed width (Word / PDF
# place charts 5.5 in wide) so 8-11 pt text stays 8-11 pt on paper
_LAYOUT = {
    "classic": {"width": 10.0, "dpi": 150, "title": 12, "tick": 9, "value": 8, "caption": 8,
                "judgment": 9, "legend": 9, "wrap": 70, "bar_h": 5.2},
    "enhanced": {"width": 5.5, "dpi": 200, "title": 11, "tick": 8.5, "value": 7.5, "caption": 7.5,
                 "judgment": 8, "legend": 8, "wrap": 46, "bar_h": 3.4},
}

_JP_FONT_HINTS = ("Noto Sans CJK JP", "Noto Sans JP", "NotoSansCJK", "NotoSansJP", "BIZ UDPGothic",
                  "IPAexGothic", "IPAGothic", "IPAPGothic", "Source Han Sans", "Yu Gothic",
                  "YuGothic", "Hiragino Sans", "Meiryo", "MS Gothic", "TakaoGothic", "VL Gothic",
                  "WenQuanYi", "Unifont")
_font_cache: Dict[str, Optional[str]] = {}
# pyplot (figure manager, rcParams) is process-global: one render at a time
_RENDER_LOCK = threading.Lock()
# set when no CJK-capable font was found (the caller warns once)
FONT_STATE = {"japanese_missing": False}


def setup_japanese_font(plt, fm) -> Optional[str]:
    """Pick a CJK-capable font by substring (the old exact-name search
    silently fell back to DejaVu and drew □□□); returns the family name."""
    if "name" in _font_cache:
        name = _font_cache["name"]
    else:
        name = None
        try:
            names = [f.name for f in fm.fontManager.ttflist]
            for hint in _JP_FONT_HINTS:
                hit = next((n for n in names if hint.lower().replace(" ", "") in n.lower().replace(" ", "")), None)
                if hit:
                    name = hit
                    break
        except Exception:
            name = None
        _font_cache["name"] = name
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


# characters that must not start a line (Japanese line-breaking rules)
_NO_LINE_START = set("、。，．・：；？！）］｝〕〉》」』】ー々ゝゞ～%％,.:;)]}")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,%％'’\-/:+]*|\s+|.", re.S)


def _wrap(text: str, width: int) -> str:
    """Wrap to ``width`` characters without breaking inside a number or a
    Latin word ('6億缶' and '1.5' stay whole) and without starting a line
    with closing punctuation."""
    text = str(text or "")
    if not text:
        return ""
    lines: List[str] = []
    cur = ""
    for tok in _TOKEN.findall(text):
        if tok.isspace():
            if cur:
                cur += " "
            continue
        if len(cur) + len(tok) > width and cur.strip():
            if tok[0] in _NO_LINE_START:
                cur += tok                      # keep closing punctuation on this line
                continue
            lines.append(cur.rstrip())
            cur = tok
        else:
            cur += tok
    if cur.strip():
        lines.append(cur.rstrip())
    return "\n".join(lines)


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
        if any(_overlaps(bb, k) for k in kept):
            t.remove()
            dropped += 1
        else:
            kept.append(bb)
    return dropped


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


def _pick_form(panel: Dict[str, Any], style: str) -> str:
    """enhanced: a vertical bar chart whose labels would collide (many or
    long labels) is drawn as horizontal bars; the panel's semantic form is
    unchanged (recorded as form_rendered)."""
    form = panel.get("form", "bar")
    if style != "enhanced" or form != "bar":
        return form
    labels = [str(l) for l in (panel.get("labels") or [])]
    if len(labels) > 6 or any(len(l) > 10 for l in labels):
        panel["form_rendered"] = "hbar"
        return "hbar"
    return form


def render_png(panel: Dict[str, Any], path: Path, language: str = "ja", dpi: Optional[int] = None,
               style: str = "classic", qa: Optional[Dict[str, Any]] = None) -> Path:
    """Draw one RESOLVED panel to ``path`` (PNG). Returns the path.
    ``qa`` (a dict) receives the layout facts of the drawn figure: value
    axis range, overlapping labels, text outside the image, pixel size."""
    with _RENDER_LOCK:
        return _render_locked(panel, Path(path), language, dpi, style if style in STYLES else "classic", qa)


def _render_locked(panel: Dict[str, Any], path: Path, language: str, dpi: Optional[int],
                   style: str, qa: Optional[Dict[str, Any]] = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    L = _LAYOUT[style]
    with plt.rc_context():
        setup_japanese_font(plt, fm)
        plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                             "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK,
                             "ytick.color": INK, "font.size": 10 if style == "classic" else 8.5,
                             "figure.facecolor": "white"})
        form = _pick_form(panel, style)
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

        scale = L["width"] / 10.0
        if seg_row:
            fig_h = max(2.6 * scale + 0.6, 0.55 * scale * max(1, len(labels)) + 2.0 * scale + 0.4)
        elif horiz:
            fig_h = max(3.2 * scale + 0.6, 0.45 * scale * max(1, len(labels)) * max(1, n_series) + 2.0 * scale + 0.6)
        else:
            fig_h = L["bar_h"]
        if style == "enhanced":
            # room for the legend / caption / judgment lines under the plot
            fig_h += 0.9
        fig, ax = plt.subplots(figsize=(L["width"], fig_h))
        any_striped = any(not _published(b) for s in series for b in (s.get("basis") or []))
        handles: List[Any] = []
        point_labels: List[Any] = []
        value_labels: List[Any] = []
        seg_labels: List[Any] = []

        if line:
            for i, s in enumerate(series):
                color = CAT[i % len(CAT)]
                pts = [(float(v["x"]), float(v["y"])) for v in (s.get("values") or []) if isinstance(v, dict)]
                if not pts:
                    continue
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, color=color, marker=MARKERS[i % len(MARKERS)], linestyle=DASHES[i % len(DASHES)],
                        linewidth=2 if style == "classic" else 1.6, markersize=6 if style == "classic" else 4.5)
                for x, y, b in zip(xs, ys, s.get("basis") or [None] * len(xs)):
                    if not _published(b):
                        ax.plot([x], [y], marker=MARKERS[i % len(MARKERS)],
                                markersize=9 if style == "classic" else 7, markerfacecolor="none",
                                markeredgecolor=color, linestyle="none")
                    # a negative point's label goes under it (above, it sat on the line)
                    point_labels.append(ax.annotate(fmt(y), (x, y), textcoords="offset points",
                                                    xytext=(0, 7) if y >= 0 else (0, -7),
                                                    va="bottom" if y >= 0 else "top",
                                                    ha="center", fontsize=L["value"], color=INK))
                handles.append(Line2D([0], [0], color=color, marker=MARKERS[i % len(MARKERS)],
                                      linestyle=DASHES[i % len(DASHES)], label=s.get("name", "")))
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.set_ylabel(unit)
            # negative values (a fall in growth, a loss) must stay visible
            if lo < 0:
                span = (max(hi, 0.0) - lo) or abs(lo)
                ax.set_ylim(bottom=lo - 0.08 * span, top=panel.get("y_max") or None)
                ax.axhline(0, color="#b9b8b1", linewidth=0.8)
            else:
                ax.set_ylim(bottom=0, top=panel.get("y_max") or None)
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
                for j in range(rows):
                    v = vals[j] if j < len(vals) else None
                    if not isinstance(v, (int, float)) or isinstance(v, bool):
                        continue
                    v = float(v)
                    b = basis[j] if j < len(basis) else None
                    ax.barh([j], [v], left=lefts[j], color=fill, height=0.6,
                            hatch="////" if not _published(b) else None, edgecolor="white", linewidth=0.5)
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
                ax.set_yticks(range(rows)); ax.set_yticklabels(labels, fontsize=L["tick"]); ax.invert_yaxis()
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
                if stacked:
                    pos = idx
                else:
                    offset = (i - (n_series - 1) / 2) * width
                    pos = [x + offset for x in idx]
                for j in range(len(labels)):
                    if not present[j]:
                        continue
                    kw = dict(color=colors[j], edgecolor="white", linewidth=0.5)
                    if not _published(basis[j]):
                        kw["hatch"] = "////"
                    if horiz:
                        ax.barh(pos[j], numeric[j], height=width, left=bottoms[j] if stacked else 0, **kw)
                    else:
                        ax.bar(pos[j], numeric[j], width=width, bottom=bottoms[j] if stacked else 0, **kw)
                    if not stacked:
                        if horiz:
                            gap = 0.01 * span
                            if numeric[j] < 0:      # the label sits left of a negative bar
                                value_labels.append(ax.text(numeric[j] - gap, pos[j], fmt(numeric[j]) + " ",
                                                            va="center", ha="right", fontsize=L["value"], color=INK))
                            else:
                                value_labels.append(ax.text(numeric[j] + gap, pos[j], " " + fmt(numeric[j]),
                                                            va="center", ha="left", fontsize=L["value"], color=INK))
                        else:
                            value_labels.append(ax.text(pos[j], numeric[j], fmt(numeric[j]), ha="center",
                                                        va="bottom" if numeric[j] >= 0 else "top",
                                                        fontsize=L["value"], color=INK))
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
                ax.set_yticks(idx); ax.set_yticklabels(labels, fontsize=L["tick"]); ax.invert_yaxis()
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

        if style == "enhanced" and not seg_row:
            # thousands separators / significant digits on the value axis
            axis = ax.xaxis if horiz else ax.yaxis
            axis.set_major_formatter(mticker.FuncFormatter(
                lambda v, _pos: _fmt(v, "" if unit == "%" else unit) if unit != "%" else f"{v:g}"))
            ax.tick_params(labelsize=L["tick"])

        ax.set_title(str(panel.get("title", "")), fontsize=L["title"], loc="left", color="#0b0b0b", pad=10)
        # Below the axes, bottom-up in FIGURE coordinates: judgment, caption,
        # legend — then the axes bottom margin is set above all of them so
        # tick labels, legend and text never overlap whatever the height.
        line_h = (0.16 if style == "classic" else 0.14) / fig_h
        caption = _wrap(panel.get("caption", ""), L["wrap"])
        judgment = _wrap(panel.get("judgment", ""), L["wrap"])
        y = (0.4 if style == "classic" else 0.12) / fig_h
        if judgment:
            fig.text(0.05 if style == "classic" else 0.02, y, "見るべき点: " + judgment,
                     fontsize=L["judgment"], color="#0b0b0b", va="bottom")
            y += line_h * (judgment.count("\n") + 1) + line_h * 0.4
        if caption:
            fig.text(0.05 if style == "classic" else 0.02, y, caption, fontsize=L["caption"],
                     color="#898781", va="bottom")
            y += line_h * (caption.count("\n") + 1) + line_h * 0.6
        if handles or any_striped:
            if any_striped:
                handles.append(Patch(facecolor="white", edgecolor=INK, hatch="////", label=STRIPE_NOTE))
            ncol = min(4 if style == "classic" else 3, len(handles))
            renderer = fig.canvas.get_renderer()
            while True:
                legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y),
                                    ncol=ncol, frameon=False, fontsize=L["legend"])
                # long series names: fewer columns rather than a legend cut at the edge
                if ncol == 1 or legend.get_window_extent(renderer=renderer).width <= fig.bbox.width * 0.97:
                    break
                legend.remove()
                ncol -= 1
            y += line_h * 1.3 * -(-len(handles) // ncol)   # legend rows
        ticks = 0.0 if seg_row else 0.45 / fig_h          # tick labels under the axes
        bottom = min(0.75, y + ticks + line_h * 0.5)
        fig.subplots_adjust(bottom=bottom, top=0.86 if style == "classic" else 0.88,
                            left=0.09 if style == "classic" else 0.12, right=0.98)

        fig.canvas.draw()
        # close points: drop the labels that would print over each other
        _drop_overlapping(fig, point_labels)
        if horiz:
            _fit_left_margin(fig, ax)
        if seg_labels:
            _fit_segment_labels(fig, ax, seg_labels)
        elif value_labels:
            _fit_value_labels(fig, ax, value_labels, horiz, bool(panel.get("y_max")))
        if not horiz:
            _rotate_crowded_ticks(fig, ax)
        _drop_overlapping(fig, value_labels)
        if style == "enhanced":
            _qa_layout(fig, ax)
        if qa is not None:
            qa.update(_layout_facts(fig, ax, horiz, seg_row, dpi or L["dpi"]))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi or L["dpi"], facecolor="white")
        plt.close(fig)
        return path


def _fit_left_margin(fig, ax) -> None:
    """Long category names on a horizontal chart set the left margin
    (a fixed margin cut them off)."""
    renderer = fig.canvas.get_renderer()
    widths = [t.get_window_extent(renderer=renderer).width for t in ax.get_yticklabels() if t.get_text()]
    if not widths:
        return
    fig_w = fig.get_size_inches()[0] * fig.dpi
    need = (max(widths) + 12) / fig_w
    sp = fig.subplotpars
    if need > sp.left:
        fig.subplots_adjust(left=min(0.45, need))
        fig.canvas.draw()


def _fit_value_labels(fig, ax, labels: List[Any], horiz: bool, fixed_max: bool) -> None:
    """The label past the longest bar must not be cut off at the figure
    edge (a horizontal chart's '2,707') or sit under the title: the value
    axis is extended to hold it — or, when the panel fixes the axis
    maximum, the plot narrows instead."""
    renderer = fig.canvas.get_renderer()
    for _ in range(3):
        box = ax.get_window_extent(renderer=renderer)
        ends = [t.get_window_extent(renderer=renderer) for t in labels
                if t.figure is not None and t.get_visible() and t.get_text()]
        if not ends:
            return
        over = (max(b.x1 for b in ends) - box.x1) if horiz else (max(b.y1 for b in ends) - box.y1)
        if over <= 1:
            return
        if fixed_max:
            if horiz:
                sp = fig.subplotpars
                fig.subplots_adjust(right=max(sp.left + 0.3, sp.right - (over + 4) / fig.bbox.width))
                fig.canvas.draw()
            return
        lo, hi = ax.get_xlim() if horiz else ax.get_ylim()
        size = box.width if horiz else box.height
        grow = (hi - lo) * (over + 4) / max(1.0, size - over - 4)
        if horiz:
            ax.set_xlim(lo, hi + grow)
        else:
            ax.set_ylim(lo, hi + grow)
        fig.canvas.draw()
        if horiz:
            # the last tick label of the longer axis ('3,000') must not be cut
            ticks = [t.get_window_extent(renderer=renderer) for t in ax.get_xticklabels() if t.get_text()]
            spill = max((b.x1 for b in ticks), default=0) - fig.bbox.x1
            if spill > 0:
                sp = fig.subplotpars
                fig.subplots_adjust(right=sp.right - (spill + 4) / fig.bbox.width)
                fig.canvas.draw()


def _fit_segment_labels(fig, ax, labels: List[Any]) -> None:
    """A segment label wider than its segment ('B社 3%' in a 3% slice)
    printed over the next one: it shrinks to the value alone, and is
    dropped when even that does not fit (the legend still names it)."""
    renderer = fig.canvas.get_renderer()
    for t, x0, x1, short in labels:
        width = abs(ax.transData.transform((x1, 0))[0] - ax.transData.transform((x0, 0))[0]) - 4
        if t.get_window_extent(renderer=renderer).width <= width:
            continue
        t.set_text(short)
        if t.get_window_extent(renderer=renderer).width > width:
            t.remove()


def _rotate_crowded_ticks(fig, ax) -> None:
    """Category labels that print over each other are set at an angle,
    and the plot moves up when the angled labels reach the legend."""
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
    after = min(t.get_window_extent(renderer=renderer).y0 for t in ticks)
    if after < before:
        sp = fig.subplotpars
        fig.subplots_adjust(bottom=min(0.8, sp.bottom + (before - after) / fig.bbox.height))
        fig.canvas.draw()


def _qa_layout(fig, ax) -> None:
    """enhanced: the figure grows when text spills out of it."""
    renderer = fig.canvas.get_renderer()
    fig_bb = fig.bbox
    spill = [t for t in fig.texts + ax.texts
             if t.get_visible() and t.get_text()
             and not (fig_bb.x0 - 1 <= t.get_window_extent(renderer=renderer).x0
                      and t.get_window_extent(renderer=renderer).x1 <= fig_bb.x1 + 1)]
    if spill:
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

    def shown(axis, lim):
        # tick labels past the axis limits exist but are never drawn
        lo, hi = sorted(lim)
        span = (hi - lo) or 1.0
        return [t for loc, t in zip(axis.get_ticklocs(), axis.get_ticklabels())
                if t.get_visible() and t.get_text() and lo - 1e-6 * span <= loc <= hi + 1e-6 * span]
    xt = shown(ax.xaxis, ax.get_xlim())
    yt = shown(ax.yaxis, ax.get_ylim())
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
    texts = list(fig.texts) + labels + xt + yt
    for leg in fig.legends:
        texts.extend(leg.get_texts())
    fb = fig.bbox
    spill = []
    for t in texts:
        b = t.get_window_extent(renderer=renderer)
        if b.x0 < fb.x0 - 1 or b.x1 > fb.x1 + 1 or b.y0 < fb.y0 - 1 or b.y1 > fb.y1 + 1:
            spill.append(t.get_text())
    w, h = fig.get_size_inches()
    return {"value_lim": tuple(ax.get_xlim() if (horiz or seg_row) else ax.get_ylim()),
            "label_overlaps": overlaps, "tick_overlaps": tick_overlaps, "spill": spill,
            "size_px": (round(w * dpi), round(h * dpi))}


def layout_report(png_path: Path) -> Dict[str, Any]:
    """(for tests) image size of a rendered PNG."""
    from PIL import Image
    with Image.open(png_path) as im:
        return {"width": im.width, "height": im.height}
