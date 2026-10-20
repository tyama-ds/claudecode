"""Turn an EXISTING report (markdown / text / Word) into the session form
the finalization pipeline verifies: one chapter per heading, numbered
"1", "2", … with a research plan built from the headings.

This is the input side of the "verify only" mode: the body comes from
the file, the evidence comes from the optional evidence JSON or from
the finalization research rounds (web search for the claims that lack
support). Nothing here calls an LLM or the network.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..research.query_generator import (ResearchPlan, TableOfContents,
                                        TableOfContentsItem)
from ..research.researcher import ResearchSession, ResearchState

# headings that are not report body (never verified as claims)
_NON_BODY = re.compile(
    r"^(参考文献|引用文献|出典|文献|references?|sources?|bibliography|"
    r"目次|table of contents|contents|図表一覧|figures and tables|"
    r"用語集|glossary|付録[A-Za-z]?|appendix)\b", re.I)
_LEADING_NUMBER = re.compile(
    r"^\s*(?:第\s*\d+\s*[章節部]|\d+(?:[.．]\d+)*[.．]?|[IVX]+\.|[（(]\d+[)）])\s*[:：.．\-–—]?\s*")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def _clean_title(text: str) -> str:
    return _LEADING_NUMBER.sub("", (text or "").strip()).strip() or (text or "").strip()


def report_text_from_file(path) -> Tuple[str, str]:
    """(title, markdown-ish text) of a report file.

    .md / .markdown / .txt are read as they are; .docx paragraphs become
    lines with Word's heading styles turned into '#' headings and tables
    into pipe rows (python-docx; the DocumentReader is not used because
    it drops the heading structure)."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".md", ".markdown", ".txt", ".text"):
        return p.stem, p.read_text(encoding="utf-8", errors="replace")
    if suffix == ".docx":
        try:
            import docx
        except ImportError as e:                     # pragma: no cover
            raise ValueError("python-docx is required to read .docx reports") from e
        doc = docx.Document(str(p))
        lines: List[str] = []
        title = ""
        for block in _iter_docx_blocks(doc):
            kind, payload = block
            if kind == "para":
                para = payload
                text = para.text.strip()
                if not text:
                    lines.append("")
                    continue
                style = (para.style.name if para.style is not None else "") or ""
                m = re.match(r"(?i)^(heading|見出し)\s*(\d+)", style)
                if style.lower() == "title" or style == "表題":
                    title = title or text
                    lines.append(f"# {text}")
                elif m:
                    level = min(6, max(1, int(m.group(2))) + 1)   # Heading 1 -> ##
                    lines.append("#" * level + " " + text)
                else:
                    lines.append(text)
            else:
                table = payload
                rows = []
                for row in table.rows:
                    rows.append("| " + " | ".join(c.text.strip().replace("\n", " ")
                                                 for c in row.cells) + " |")
                if rows:
                    lines.append(rows[0])
                    lines.append("|" + "---|" * len(table.rows[0].cells))
                    lines.extend(rows[1:])
                lines.append("")
        core_title = ""
        try:
            core_title = (doc.core_properties.title or "").strip()
        except Exception:
            pass
        return (title or core_title or p.stem), "\n".join(lines)
    raise ValueError(f"unsupported report format: {suffix or '(none)'} "
                     "(use .md, .txt or .docx)")


def _iter_docx_blocks(doc):
    """Paragraphs and tables in document order."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    body = doc.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            yield "para", Paragraph(child, doc)
        elif tag == "tbl":
            yield "table", Table(child, doc)


def split_report_sections(text: str, fallback_title: str = "") -> Tuple[str, List[Dict[str, str]]]:
    """(title, [{"title", "content"}]) from markdown-style headings.

    The first level-1 heading is the report title. Chapters are the
    level-2 headings (level-3+ headings stay inside their chapter as
    sub-headings); when the file has only level-1 headings those become
    the chapters; a file without headings is ONE chapter. Reference /
    contents / glossary / appendix sections are dropped (they are not
    claims to verify)."""
    lines = (text or "").replace("\r\n", "\n").split("\n")
    headings = [(i, len(m.group(1)), m.group(2).strip())
                for i, line in enumerate(lines)
                if (m := _HEADING.match(line))]
    title = fallback_title
    h1 = [h for h in headings if h[1] == 1]
    h2 = [h for h in headings if h[1] == 2]
    if h1:
        title = _clean_title(h1[0][2]) or title
    if h2:
        chapter_level = 2
    elif len(h1) >= 2:
        chapter_level = 1
        title = fallback_title
    elif len(h1) == 1 and len(headings) > 1:
        # "# Title" + "### ..." style: the deeper headings are chapters
        chapter_level = min(h[1] for h in headings if h[1] > 1)
    else:
        chapter_level = None

    sections: List[Dict[str, str]] = []
    if chapter_level is None:
        body = "\n".join(l for i, l in enumerate(lines)
                         if not (h1 and i == h1[0][0])).strip()
        if body:
            sections.append({"title": title or "本文", "content": body})
        return title, sections

    starts = [h for h in headings if h[1] == chapter_level]
    for n, (idx, _lvl, raw) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        # a shallower heading between two chapters ends the chapter too
        for j in range(idx + 1, end):
            m = _HEADING.match(lines[j])
            if m and len(m.group(1)) < chapter_level:
                end = j
                break
        heading = _clean_title(raw)
        if _NON_BODY.match(heading):
            continue
        content = "\n".join(lines[idx + 1:end]).strip()
        if not content:
            continue
        sections.append({"title": heading, "content": content})
    return title, sections


def session_from_report(report_path, query: str = "", requirements: str = "",
                        language: str = "ja", session_id: Optional[str] = None) -> ResearchSession:
    """A ResearchSession whose chapters are the report's sections and
    whose plan is built from its headings — ready for the finalization
    runner. ``query`` defaults to the report title."""
    title, text = report_text_from_file(report_path)
    title, sections = split_report_sections(text, fallback_title=title)
    if not sections:
        raise ValueError("the report has no body text to verify")
    items = [TableOfContentsItem(section=str(i + 1), title=s["title"],
                                 description=s["title"])
             for i, s in enumerate(sections)]
    toc = TableOfContents(title=title or (query or "report"), items=items)
    plan = ResearchPlan(title=title or (query or "report"), summary="",
                        table_of_contents=toc, search_queries=[],
                        methodology_notes="external report (verify only)")
    session = ResearchSession(query=query or title or Path(report_path).stem,
                              requirements=requirements or "",
                              state=ResearchState.COMPLETED,
                              research_plan=plan)
    if session_id:
        session.session_id = session_id
    for item, s in zip(items, sections):
        session.section_contents[item.section] = {
            "title": s["title"], "content": s["content"],
            "summary": "", "confidence": "unknown",
            "sources": [], "extracted_content": [],
        }
        item.content = s["content"]
    session.run_config = {"source": "external_report",
                          "report_path": str(report_path), "language": language}
    return session
