"""
chartkit — charts as "spec JSON + fixed renderer" (B案).

The LLM never writes a number: a deterministic data layer turns the
numerical data store of a research run into a per-section catalog of
labelled series with a certainty tag (公表 / 換算 / 導出 / 推定 / 二次) and
sources on every value; the LLM only chooses WHICH series to plot, in
which form, with which emphasis, and writes the caption (source +
certainty) and the judgment (what to look at). A validator checks the
spec against the data; fixed renderers draw it — Chart.js in a single
offline HTML page for the browser, and matplotlib PNGs with the same
rules for Markdown / Word / PDF.
"""

from .data_layer import (SectionCatalog, Series, build_section_catalog,
                         classify_source_basis, parse_unit)
from .spec import (FORMS, ROLES, SpecError, build_messages, panel_semantics,
                   parse_panels, resolve, validate, validate_panels)
from .render_html import render_html
from .render_png import render_png

__all__ = [
    "SectionCatalog", "Series", "build_section_catalog",
    "classify_source_basis", "parse_unit",
    "FORMS", "ROLES", "SpecError", "build_messages", "panel_semantics",
    "parse_panels", "resolve", "validate", "validate_panels",
    "render_html", "render_png",
]
