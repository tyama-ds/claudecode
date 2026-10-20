"""
HTML renderer: one self-contained page (Chart.js 4.4.1 embedded from
``vendor/chart.umd.js``, no network, no LLM-written script) drawing the
resolved panels with the fixed rules in ``prelude.js``.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).parent
VENDOR_CHARTJS = HERE / "vendor" / "chart.umd.js"


def _read(name: str) -> str:
    return (HERE / name).read_text(encoding="utf-8")


def kpi_card(k: Dict[str, Any]) -> str:
    e = html.escape
    return (f'<div class="kpi"><p class="kl">{e(str(k.get("label", "")))}</p>'
            f'<p class="kv">{e(str(k.get("value", "")))}</p>'
            f'<p class="ks">{e(str(k.get("sub", "")))}</p></div>')


def panel_html(p: Dict[str, Any]) -> str:
    e = html.escape
    form = p.get("form", "bar")
    default_h = 80 if form == "stacked_hbar" else (
        max(260, len(p.get("series") or []) * len(p.get("labels") or []) * 40 + 80) if form == "hbar" else 260)
    h = _px(p.get("height"), default_h)
    return (f'<section class="panel"><p class="lb">{e(str(p.get("title", "")))}</p>'
            f'<div class="lg" id="lg_{e(str(p["id"]))}"></div>'
            f'<div style="position:relative;width:100%;height:{h}px">'
            f'<canvas id="cv_{e(str(p["id"]))}" role="img" aria-label="{e(str(p.get("title", "")))}"></canvas></div>'
            f'<p class="cap">{e(str(p.get("caption", "")))}</p>'
            f'<p class="jd">{e(str(p.get("judgment", "")))}</p></section>')


def _px(value: Any, default: int) -> int:
    """An LLM-written height ("300px", "300.0", 300) as pixels, clamped;
    anything unusable falls back to the default instead of raising."""
    try:
        n = int(float(str(value).strip().lower().removesuffix("px")))
    except (TypeError, ValueError):
        return default
    return max(60, min(2000, n))


def _spec_json(panels: Sequence[Dict[str, Any]]) -> str:
    keep = ("id", "form", "title", "labels", "series", "highlight", "unit", "y_max", "height")
    slim = []
    for p in panels:
        q = {k: p[k] for k in keep if k in p}
        q["series"] = [{k: s.get(k) for k in ("name", "role", "values", "basis")}
                       for s in (p.get("series") or [])]
        slim.append(q)
    # no "<" at all inside the <script>: "</script" / "<!--" in a label
    # can never end or confuse the script element (< is plain JSON)
    return json.dumps(slim, ensure_ascii=False).replace("<", "\\u003c")


def render_html(panels: Sequence[Dict[str, Any]], kpis: Optional[Sequence[Dict[str, Any]]] = None,
                title: str = "チャート", subtitle: str = "",
                sections: Optional[List[Dict[str, Any]]] = None) -> str:
    """A single HTML page. ``panels`` are RESOLVED panels (see spec.resolve).
    ``sections`` = [{"title": str, "panel_ids": [...]}] groups the panels
    under chapter headings; without it they are listed in order."""
    tpl = _read("template.html")
    chartjs = VENDOR_CHARTJS.read_text(encoding="utf-8") if VENDOR_CHARTJS.is_file() else ""
    prelude = _read("prelude.js")
    by_id = {str(p["id"]): p for p in panels}
    if sections:
        body = []
        for sec in sections:
            ids = [i for i in sec.get("panel_ids", []) if i in by_id]
            if not ids:
                continue
            body.append(f'<h2 class="sec">{html.escape(str(sec.get("title", "")))}</h2>')
            body.extend(panel_html(by_id[i]) for i in ids)
        panels_html = "".join(body)
    else:
        panels_html = "".join(panel_html(p) for p in panels)
    if not panels_html:
        panels_html = '<p class="empty">描けるチャートはありませんでした。</p>'
    if not chartjs:
        panels_html = ('<p class="empty">Chart.js が同梱されていないため描画できません'
                       '（deep_research_tool/chartkit/vendor/chart.umd.js）。</p>' + panels_html)
    # one pass: a placeholder-like string inside a title / label is never
    # substituted by a later replacement
    fills = {"TITLE": html.escape(title), "SUBTITLE": html.escape(subtitle),
             "KPIS": "".join(kpi_card(k) for k in (kpis or [])), "PANELS": panels_html,
             "CHARTJS": chartjs, "SPEC": _spec_json(panels), "PRELUDE": prelude}
    import re as _re
    return _re.sub(r"\{\{(TITLE|SUBTITLE|KPIS|PANELS|CHARTJS|SPEC|PRELUDE)\}\}",
                   lambda m: fills[m.group(1)], tpl)


def to_jupyter(page: str, height: int = 900):
    """IPython.display.HTML with a sandboxed iframe (the page string when
    IPython is not installed)."""
    try:
        from IPython.display import HTML
    except Exception:
        return page
    return HTML(f'<iframe sandbox="allow-scripts" style="width:100%;height:{int(height)}px;border:0" '
                f'srcdoc="{html.escape(page)}"></iframe>')
