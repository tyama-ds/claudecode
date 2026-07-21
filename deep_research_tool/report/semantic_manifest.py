"""
Semantic manifest - the canonical snapshot of everything the report SAYS.

At freeze time (right after the finalization loop verified the final
body) a manifest of all SEMANTIC content is built and hashed:

- chapter bodies (including limitations, Fermi, glossary, warnings —
  they are chapters by the time the loop runs)
- figure/chart titles, captions and alt texts
- table titles, captions, headers and every cell
- chart insights text

Everything in the manifest was part of what the verifier saw. After the
freeze, renderers may only perform NON-semantic work (citation number
substitution — applied before hashing —, layout, fonts, DOCX/PDF
conversion, drawing pixels from already-frozen chart data), so the
manifest recomputed just before the final output must hash IDENTICALLY.
A mismatch means something generated or altered meaning after
verification — a pipeline defect, reported loudly.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional


def figure_semantics(collection) -> List[Dict[str, Any]]:
    """All semantic content of a FigureTableCollection, canonically ordered."""
    if collection is None:
        return []
    items: List[Dict[str, Any]] = []
    for kind, figs in (("figure", getattr(collection, "figures", []) or []),
                       ("chart", getattr(collection, "charts", []) or [])):
        for f in figs:
            items.append({
                "kind": kind,
                "id": str(getattr(f, "figure_id", "") or ""),
                "section": str(getattr(f, "section_id", "") or ""),
                "title": str(getattr(f, "title", "") or ""),
                "caption": str(getattr(f, "caption", "") or ""),
                "alt": str(getattr(f, "alt_text", "") or ""),
            })
    for t in getattr(collection, "tables", []) or []:
        items.append({
            "kind": "table",
            "id": str(getattr(t, "table_id", "") or ""),
            "section": str(getattr(t, "section_id", "") or ""),
            "title": str(getattr(t, "title", "") or ""),
            "caption": str(getattr(t, "caption", "") or ""),
            "headers": [str(h) for h in (getattr(t, "headers", []) or [])],
            "rows": [[str(c) for c in row]
                     for row in (getattr(t, "rows", []) or [])],
        })
    items.sort(key=lambda i: (i["kind"], i["section"], i["id"], i["title"]))
    return items


def build_semantic_manifest(chapters: Dict[str, str],
                            figure_collection=None) -> Dict[str, Any]:
    """Canonical semantic snapshot: chapter texts + figure semantics."""
    return {
        "chapters": {str(k): str(v or "") for k, v in
                     sorted((chapters or {}).items())},
        "figures": figure_semantics(figure_collection),
    }


def manifest_hash(manifest: Dict[str, Any]) -> str:
    """Deterministic sha256 over the canonical JSON form."""
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def figure_semantics_markdown(collection, language: str = "ja",
                              section_id: str = "figures") -> str:
    """The figure semantics as a verifiable markdown section.

    This text joins the finalization chapters, so every LLM-generated
    figure title, caption and table cell passes through the SAME
    verification as the body — BEFORE the freeze. It is rendered
    verbatim afterwards (render-only section, never LLM-edited).
    """
    items = figure_semantics(collection)
    if not items:
        return ""
    title = "図表一覧" if language == "ja" else "Figures and Tables"
    lines = [f"## {section_id}. {title}", ""]
    for item in items:
        label = {"figure": "図", "chart": "図", "table": "表"}.get(
            item["kind"], "図") if language == "ja" else item["kind"].title()
        head = item["title"] or item["id"]
        lines.append(f"### {label}: {head}")
        if item["caption"]:
            lines.append(item["caption"])
        if item.get("alt"):
            lines.append(item["alt"])
        if item["kind"] == "table" and item.get("headers"):
            lines.append("")
            lines.append("| " + " | ".join(item["headers"]) + " |")
            lines.append("|" + "---|" * len(item["headers"]))
            for row in item.get("rows", [])[:50]:
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines).strip()
