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
                            figure_collection=None,
                            extras: Optional[Dict[str, Any]] = None,
                            ) -> Dict[str, Any]:
    """Canonical semantic snapshot: chapter texts + figure semantics.

    ``extras`` carries semantic content that lives OUTSIDE the chapter
    map — the executive summary, key findings, recommendations, glossary
    entries, footnotes. Everything the reader will read is in the
    manifest; nothing semantic escapes the freeze because it happens to
    be stored under another key.
    """
    manifest = {
        "chapters": {str(k): str(v or "") for k, v in
                     sorted((chapters or {}).items())},
        "figures": figure_semantics(figure_collection),
    }
    if extras:
        def _canon(v):
            if isinstance(v, dict):
                return {str(k): _canon(x) for k, x in sorted(v.items())}
            if isinstance(v, (list, tuple)):
                return [_canon(x) for x in v]
            return str(v)
        manifest["extras"] = {str(k): _canon(v)
                              for k, v in sorted(extras.items())}
    return manifest


def manifest_hash(manifest: Dict[str, Any]) -> str:
    """Deterministic sha256 over the canonical JSON form."""
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Artifact-level freeze verification (audit item D)
#
# Hashing the in-memory chapters twice only proves the OBJECT did not
# change; a renderer bug could still alter or drop content on the way to
# disk. The functions below REBUILD the semantic text from the ACTUAL
# saved artifact and check that every frozen chapter paragraph survived
# into it (normalized: markup, citation numbers and whitespace removed —
# rendering may restyle, it must never change meaning-bearing text).
# ---------------------------------------------------------------------------

import re as _re

_CITATION_RE = _re.compile(r"\[(?:SOURCE:?\s*)?\d+\]")
_MD_IMAGE_RE = _re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = _re.compile(r"\[([^\]]*)\]\([^)]*\)")
_TAG_RE = _re.compile(r"<[^>]+>")


def normalize_semantic_text(text: str) -> str:
    """Reduce text to its meaning-bearing characters.

    Removes markdown/HTML markup, citation tags/numbers and ALL
    whitespace, so a markdown source and its rendered artifact compare
    equal exactly when the words are the same.
    """
    text = text or ""
    text = _MD_IMAGE_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _TAG_RE.sub("", text)
    text = _CITATION_RE.sub("", text)
    text = _re.sub(r"[*_`>#|\\-]+", "", text)
    text = _re.sub(r"[：:]\s*$", "", text)
    return _re.sub(r"\s+", "", text)


def extract_artifact_text(path) -> Optional[str]:
    """Best-effort text of a SAVED report artifact (md/txt/html/docx).

    Returns None when the format cannot be read back (the caller reports
    the check as SKIPPED — never silently passed).
    """
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return None
    suffix = p.suffix.lower()
    if suffix in (".md", ".txt", ".markdown"):
        return p.read_text(encoding="utf-8", errors="replace")
    if suffix in (".html", ".htm"):
        import html as _htmlmod
        return _htmlmod.unescape(
            p.read_text(encoding="utf-8", errors="replace"))
    if suffix == ".docx":
        try:
            import docx
            doc = docx.Document(str(p))
            parts = [para.text for para in doc.paragraphs]
            for table in doc.tables:
                for row in table.rows:
                    parts.extend(cell.text for cell in row.cells)
            return "\n".join(parts)
        except Exception:
            return None
    return None


def verify_frozen_in_artifact(manifest: Dict[str, Any],
                              artifact_text: str,
                              min_fragment_chars: int = 24,
                              ) -> Dict[str, Any]:
    """Check that the frozen semantic content is IN the saved artifact.

    Every sufficiently long paragraph of every frozen chapter (and every
    figure/table title, caption and cell) must appear — normalized — in
    the artifact text. Returns {"ok", "checked", "missing"} where
    ``missing`` lists the fragments that did not survive rendering.
    """
    haystack = normalize_semantic_text(artifact_text)
    checked = 0
    missing: List[str] = []

    def _check(fragment: str) -> None:
        nonlocal checked
        needle = normalize_semantic_text(fragment)
        if len(needle) < min_fragment_chars:
            return
        checked += 1
        if needle not in haystack:
            missing.append(fragment.strip()[:120])

    for text in (manifest.get("chapters") or {}).values():
        for para in (text or "").split("\n\n"):
            for line in para.split("\n"):
                _check(line)
    for item in manifest.get("figures") or []:
        _check(item.get("title", ""))
        _check(item.get("caption", ""))
        for row in item.get("rows", []) or []:
            _check(" ".join(row))
    extras = manifest.get("extras") or {}

    def _walk(value):
        if isinstance(value, dict):
            for v in value.values():
                _walk(v)
        elif isinstance(value, list):
            for v in value:
                _walk(v)
        else:
            _check(str(value))
    _walk(extras)

    return {"ok": not missing, "checked": checked, "missing": missing}


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
