"""
PNG renderer (matplotlib) with the SAME rules as the HTML page, for the
Markdown / Word / PDF outputs: one emphasised series in blue, context in
grey, negative values red, category colours for parts and line_xy,
striped fill for every value that is not 公表, value labels, the unit in
the title, a single axis, a legend built from the roles, and the caption
and judgment under the chart.
"""

from __future__ import annotations

import math
import re
import textwrap
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

_JP_FONT_HINTS = ("IPAGothic", "IPAexGothic", "IPAPGothic", "Noto Sans CJK JP", "Noto Sans JP",
                  "NotoSansCJK", "NotoSansJP", "Source Han Sans", "Yu Gothic", "YuGothic",
                  "Hiragino Sans", "Meiryo", "MS Gothic", "TakaoGothic", "VL Gothic",
                  "WenQuanYi", "Unifont")
_font_cache: Dict[str, Optional[str]] = {}


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
    return f"{value:,.1f}"


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(str(text or ""), width=width)) if text else ""


def render_png(panel: Dict[str, Any], path: Path, language: str = "ja", dpi: int = 150) -> Path:
    """Draw one RESOLVED panel to ``path`` (PNG). Returns the path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    setup_japanese_font(plt, fm)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "font.size": 10, "figure.facecolor": "white"})
    form = panel.get("form", "bar")
    series = panel.get("series") or []
    labels = list(panel.get("labels") or [])
    unit = str(panel.get("unit") or (series[0].get("unit") if series else "") or "")
    n_series = len(series)
    signed = _signed_pct(panel)
    fmt = lambda v: _fmt(v, unit, signed)  # noqa: E731
    horiz = form.endswith("hbar")
    stacked = form.startswith("stacked")
    seg_row = form == "stacked_hbar"

    if seg_row:
        fig_h = 2.6
    elif horiz:
        fig_h = max(3.2, 0.45 * max(1, len(labels)) * max(1, n_series) + 2.0)
    else:
        fig_h = 5.2
    fig, ax = plt.subplots(figsize=(10, fig_h))
    any_striped = any(not _published(b) for s in series for b in (s.get("basis") or []))
    handles: List[Any] = []

    if form == "line_xy":
        for i, s in enumerate(series):
            color = CAT[i % len(CAT)]
            pts = [(float(v["x"]), float(v["y"])) for v in (s.get("values") or []) if isinstance(v, dict)]
            if not pts:
                continue
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys, color=color, marker=MARKERS[i % len(MARKERS)], linestyle=DASHES[i % len(DASHES)],
                    linewidth=2, markersize=6)
            for x, y, b in zip(xs, ys, s.get("basis") or [None] * len(xs)):
                if not _published(b):
                    ax.plot([x], [y], marker=MARKERS[i % len(MARKERS)], markersize=9, markerfacecolor="none",
                            markeredgecolor=color, linestyle="none")
                ax.annotate(fmt(y), (x, y), textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=8, color=INK)
            handles.append(Line2D([0], [0], color=color, marker=MARKERS[i % len(MARKERS)],
                                  linestyle=DASHES[i % len(DASHES)], label=s.get("name", "")))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_ylabel(unit)
        ax.set_ylim(bottom=0, top=panel.get("y_max") or None)
        ax.grid(axis="y", color=GRID)
    elif seg_row:
        left = 0.0
        for i, s in enumerate(series):
            v = next((float(x) for x in (s.get("values") or []) if isinstance(x, (int, float))), None)
            if v is None:
                continue
            fill, text_color = SEG[i % len(SEG)]
            b = (s.get("basis") or [None])[0]
            ax.barh([0], [v], left=left, color=fill, height=0.6,
                    hatch="////" if not _published(b) else None, edgecolor="white", linewidth=0.5)
            ax.text(left + v / 2, 0, f"{s.get('name', '')} {fmt(v)}", ha="center", va="center",
                    fontsize=10, color=text_color)
            handles.append(Patch(facecolor=fill, label=s.get("name", "")))
            left += v
        ax.set_xlim(0, panel.get("y_max") or max(left, 1e-9))
        ax.set_yticks([]); ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    else:
        idx = list(range(len(labels)))
        width = 0.8 / max(1, n_series) if not stacked else 0.6
        bottoms = [0.0] * len(labels)
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
                        ax.text(numeric[j] + (abs(numeric[j]) * 0.01 + 0.02), pos[j], " " + fmt(numeric[j]),
                                va="center", ha="left", fontsize=8, color=INK)
                    else:
                        ax.text(pos[j], numeric[j], fmt(numeric[j]), ha="center",
                                va="bottom" if numeric[j] >= 0 else "top", fontsize=8, color=INK)
            if stacked:
                bottoms = [bottoms[j] + (numeric[j] if present[j] else 0.0) for j in range(len(labels))]
            handles.append(Patch(facecolor=base, label=s.get("name", "")))
        if stacked:
            for j, total in enumerate(bottoms):
                if total:
                    if horiz:
                        ax.text(total, j, " " + fmt(total), va="center", ha="left", fontsize=8, color=INK)
                    else:
                        ax.text(j, total, fmt(total), ha="center", va="bottom", fontsize=8, color=INK)
        if horiz:
            # the unit is in the title; an x-axis label would sit where
            # the legend goes
            ax.set_yticks(idx); ax.set_yticklabels(labels); ax.invert_yaxis()
            ax.grid(axis="x", color=GRID)
            if panel.get("y_max"):
                ax.set_xlim(0, float(panel["y_max"]))
        else:
            ax.set_xticks(idx); ax.set_xticklabels([_wrap(l, 10) for l in labels], fontsize=9)
            ax.set_ylabel(unit); ax.grid(axis="y", color=GRID)
            if panel.get("y_max"):
                ax.set_ylim(top=float(panel["y_max"]))
            ax.axhline(0, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    ax.set_title(str(panel.get("title", "")), fontsize=12, loc="left", color="#0b0b0b", pad=10)
    # Below the axes, bottom-up in FIGURE coordinates: judgment, caption,
    # legend — then the axes bottom margin is set above all of them so
    # tick labels, legend and text never overlap whatever the height.
    line = 0.16 / fig_h                       # one text line (~0.16 in)
    caption = _wrap(panel.get("caption", ""), 70)
    judgment = _wrap(panel.get("judgment", ""), 70)
    y = 0.4 / fig_h
    if judgment:
        fig.text(0.05, y, "見るべき点: " + judgment, fontsize=9, color="#0b0b0b", va="bottom")
        y += line * (judgment.count("\n") + 1) + line * 0.4
    if caption:
        fig.text(0.05, y, caption, fontsize=8, color="#898781", va="bottom")
        y += line * (caption.count("\n") + 1) + line * 0.6
    if handles or any_striped:
        if any_striped:
            handles.append(Patch(facecolor="white", edgecolor=INK, hatch="////", label=STRIPE_NOTE))
        ncol = min(4, len(handles))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y),
                   ncol=ncol, frameon=False, fontsize=9)
        y += line * 1.3 * -(-len(handles) // ncol)   # legend rows
    ticks = 0.0 if seg_row else 0.45 / fig_h          # tick labels under the axes
    fig.subplots_adjust(bottom=min(0.75, y + ticks + line * 0.5), top=0.86, left=0.09, right=0.98)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path
