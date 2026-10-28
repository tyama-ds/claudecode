"""External research material for the feedback re-research (起案 C).

The reader can hand the feedback re-research reports produced elsewhere —
OpenAI Deep Research, Gemini, Copilot, an in-house memo — as files
(md / txt / docx / pdf / html) or pasted text. This module:

1. reads them (``load_external_reports``): title, chapters (same heading
   rules as the verify-only loader), the URLs they cite (Sources section,
   markdown links, bare URLs) and per-chapter digests (first sentences +
   the numbers / years they state) — no LLM involved
2. compares them with the report being revised (``compare_reports``): one
   LLM call returns topics classified as agree / conflict / external_only /
   report_only with the chapter they belong to; a deterministic overlap
   fallback keeps the feature usable when the answer is unusable
3. turns the comparison into instructions for the revision planner
   (``comparison_instructions``): a conflict revises its chapter ("check
   with primary sources; when they differ, give both with evidence"), an
   external-only topic becomes a new chapter (add mode), agreement is
   recorded only
4. registers the material for the research loop
   (``external_documents``): each chapter as a local document with
   ``metadata.source = "external"`` (SECONDARY information: the researcher
   caps its importance and the writing rule below makes the text say where
   it comes from); the cited URLs are fetched like 指定 URL (primary)
5. summarises the outcome (``external_summary``): materials, the
   comparison table with what the re-research found per topic, and the
   chapters that rest on the external material alone (要確認)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .utils.helpers import ResearchWarnings, extract_json_from_response

MAX_EXTERNAL_REPORTS = 5
MAX_EXTERNAL_CHARS = 300_000         # per material
MAX_CITED_URLS = 30                  # fetched as primary sources (all materials)
MAX_TOPICS = 40
DIGEST_CHARS = 200
EXTERNAL_MAX_IMPORTANCE = 0.5        # secondary information never outranks a primary page
FILE_SUFFIXES = (".md", ".markdown", ".txt", ".text", ".docx", ".pdf", ".html", ".htm")

STATUS_LABELS = {"agree": "一致", "conflict": "食い違い", "external_only": "外部のみ",
                 "report_only": "本レポートのみ"}

WRITING_RULE = ("外部調査資料（二次情報）だけが根拠の文は「外部調査（出所名）によれば」と出所を明示し、"
                "数値・固有の主張は一次資料（公的統計・企業の公表資料・原典のページ）を優先する。"
                "外部資料と一次資料が食い違うときは両方を根拠つきで示す")

# a URL ends at whitespace, ASCII / Japanese brackets and Japanese
# punctuation; one level of balanced parentheses stays inside
# (https://en.wikipedia.org/wiki/Foo_(bar))
_URL_STOP = "\\s<>()\\[\\]\"'（）【】「」『』〔〕《》｛｝，；：、。！？　"
_URL_RE = re.compile(r"https?://(?:[^" + _URL_STOP + r"]|\([^\s()]*\))+")
_MD_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://(?:[^()\s]|\([^()\s]*\))+)\)")
_URL_TRAIL = ".,;:!?"
# numbers worth keeping in a digest: amounts with units, percentages, years
_FIGURE_RE = re.compile(
    r"(?:約|およそ)?\d[\d,，.．]*\s*(?:兆|億|万|千)?\s*(?:円|ドル|USD|米ドル|ユーロ|元|t|トン|件|社|人|台|%|％|倍|GW|MW|kWh)"
    r"|(?:19|20)\d{2}\s*年(?:度)?")


@dataclass
class ExternalReport:
    label: str
    title: str = ""
    text: str = ""
    sections: List[Dict[str, str]] = field(default_factory=list)
    urls: List[Dict[str, Any]] = field(default_factory=list)
    origin: str = "paste"            # file | paste
    path: Optional[str] = None
    truncated: bool = False
    error: Optional[str] = None

    @property
    def chars(self) -> int:
        return len(self.text)

    def info(self) -> Dict[str, Any]:
        return {"label": self.label, "title": self.title, "origin": self.origin,
                "file": Path(self.path).name if (self.path and self.origin == "file") else None,
                "chars": self.chars, "sections": len(self.sections),
                "section_titles": [s["title"] for s in self.sections][:20],
                "cited_urls": len(self.urls), "truncated": self.truncated, "error": self.error}


# ----------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------

def html_to_markdown(raw: str) -> Tuple[str, str]:
    """(title, markdown-ish text) of an HTML page: h1-h6 become '#'
    headings, paragraphs / list items / table rows become lines and links
    are kept as [text](url), so chapters and cited URLs survive."""
    from html.parser import HTMLParser

    class _P(HTMLParser):
        BLOCK = {"p", "div", "section", "article", "li", "tr", "br", "ul", "ol", "table",
                 "blockquote", "pre", "header", "footer", "main"}
        HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
        SKIP = {"script", "style", "noscript", "svg", "nav"}

        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.out: List[str] = []
            self.title = ""
            self._in_title = False
            self._skip = 0
            self._href: Optional[str] = None
            self._link_text: List[str] = []
            self._heading: Optional[int] = None       # level while inside h1-h6
            self._heading_text: List[str] = []

        def _flush_link(self) -> None:
            """Write an open <a> (also when the page never closes it)."""
            if self._href is None:
                return
            text = " ".join("".join(self._link_text).split()) or self._href
            self._write(f"[{text}]({self._href})")
            self._href, self._link_text = None, []

        def _write(self, text: str) -> None:
            (self._heading_text if self._heading is not None else self.out).append(text)

        def _flush_heading(self) -> None:
            if self._heading is None:
                return
            level, text = self._heading, " ".join("".join(self._heading_text).split())
            self._heading, self._heading_text = None, []
            if text:
                self.out.append("\n\n" + "#" * level + " " + text + "\n")

        def handle_starttag(self, tag, attrs):
            if tag in self.SKIP:
                self._skip += 1
            elif tag == "title":
                self._in_title = True
            elif tag in self.HEADINGS:
                self._flush_link()
                self._flush_heading()
                self._heading, self._heading_text = int(tag[1]), []
            elif tag in self.BLOCK:
                self._flush_link()
                self._write(" " if self._heading is not None else "\n")
            elif tag == "a":
                self._flush_link()
                href = dict(attrs).get("href") or ""
                if href.startswith(("http://", "https://")):
                    self._href, self._link_text = href, []

        def handle_endtag(self, tag):
            if tag in self.SKIP:
                self._skip = max(0, self._skip - 1)
            elif tag == "title":
                self._in_title = False
            elif tag in self.HEADINGS:
                self._flush_link()
                self._flush_heading()
            elif tag in self.BLOCK:
                self._flush_link()
                self._write(" " if self._heading is not None else "\n")
            elif tag == "a":
                self._flush_link()

        def handle_data(self, data):
            if self._skip:
                return
            if self._in_title:
                self.title += data
            elif self._href is not None:
                self._link_text.append(data)
            else:
                self._write(data)

        def close(self):
            super().close()
            self._flush_link()
            self._flush_heading()

    parser = _P()
    parser.feed(raw)
    parser.close()
    text = "".join(parser.out)
    lines = [" ".join(l.split()) if not l.lstrip().startswith("#") else l.strip() for l in text.split("\n")]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return " ".join(parser.title.split()), text


def _read_file(path: Path) -> Tuple[str, str]:
    """(title, markdown-ish text) of one material file."""
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown", ".txt", ".text", ".docx"):
        from .report.report_loader import report_text_from_file
        return report_text_from_file(path)
    if suffix in (".html", ".htm"):
        from .report.report_loader import _decode_text
        title, text = html_to_markdown(_decode_text(path.read_bytes()))
        return (title or path.stem), text
    if suffix == ".pdf":
        from .utils.document_reader import DocumentReader
        doc = DocumentReader(extract_images=False).read_document(path)
        if getattr(doc, "error", None):
            raise ValueError(str(doc.error))
        return (getattr(doc, "title", "") or path.stem), str(getattr(doc, "content", "") or "")
    raise ValueError(f"対応していない形式です: {suffix or '(拡張子なし)'}"
                     f"（{' / '.join(s.lstrip('.') for s in FILE_SUFFIXES)}）")


def extract_cited_urls(text: str, limit: int = MAX_CITED_URLS) -> List[Dict[str, Any]]:
    """URLs a material cites: its Sources / 参考文献 list first (with
    titles), then markdown links, then bare URLs; canonical de-dup."""
    from .engines.ingest import canonicalize_url, parse_sources_section
    out: List[Dict[str, Any]] = []
    seen = set()

    def _add(url: str, title: Optional[str]) -> None:
        url = url.rstrip(_URL_TRAIL)
        key = canonicalize_url(url)
        if not key or key in seen or len(out) >= limit:
            return
        seen.add(key)
        out.append({"url": url, "title": (title or "").strip() or None})

    for s in parse_sources_section(text):
        _add(s["url"], s.get("title"))
    for m in _MD_LINK_RE.finditer(text or ""):
        _add(m.group(2), m.group(1))
    for m in _URL_RE.finditer(text or ""):
        _add(m.group(0), None)
    return out


def digest(text: str, limit: int = DIGEST_CHARS) -> str:
    """First sentences (up to ``limit`` characters) + the figures the text
    states — what the comparison needs from a long chapter."""
    flat = " ".join(re.sub(r"(?m)^\s*(#+|[-*+]|\|)\s*", "", text or "").split())
    flat = _MD_LINK_RE.sub(lambda m: m.group(1), flat)
    flat = _URL_RE.sub("", flat)
    head = flat[:limit]
    figures = []
    for m in _FIGURE_RE.finditer(flat):
        f = m.group(0).strip()
        if f not in figures and f not in head:
            figures.append(f)
        if len(figures) >= 8:
            break
    return head + (f"（数値: {'、'.join(figures)}）" if figures else "")


def load_external_reports(items: Sequence[Dict[str, Any]], *,
                          max_reports: int = MAX_EXTERNAL_REPORTS,
                          max_chars: int = MAX_EXTERNAL_CHARS) -> List[ExternalReport]:
    """``items`` = [{label?, path? | text?}]. A material that cannot be
    read is returned with ``error`` set (the caller reports it). Raises
    ValueError for more than ``max_reports`` materials."""
    from .report.report_loader import split_report_sections
    items = [i for i in (items or []) if isinstance(i, dict) and (i.get("path") or (i.get("text") or "").strip())]
    if len(items) > max_reports:
        raise ValueError(f"外部の調査資料は {max_reports} 件までです（{len(items)} 件）")
    out: List[ExternalReport] = []
    used_labels = set()
    for n, item in enumerate(items, 1):
        path = item.get("path")
        label = " ".join(str(item.get("label") or "").split())[:80]
        if not label:
            label = Path(str(path)).stem if path else f"外部資料{n}"
        base, k = label, 2
        while label in used_labels:
            label = f"{base} ({k})"
            k += 1
        used_labels.add(label)
        # the Web UI saves pasted text to a file and marks it "pasted"
        pasted = bool(item.get("pasted")) or not path
        rep = ExternalReport(label=label, origin="paste" if pasted else "file",
                             path=str(path) if path else None)
        try:
            if path:
                p = Path(str(path))
                if not p.is_file():
                    raise ValueError(f"ファイルが見つかりません: {p.name}")
                title, text = _read_file(p)
            else:
                text = str(item.get("text") or "")
                title = ""
            text = text.replace("\r\n", "\n").strip()
            if not text:
                raise ValueError("本文が空です")
            if len(text) > max_chars:
                text = text[:max_chars]
                rep.truncated = True
            rep.urls = extract_cited_urls(text)
            title, sections = split_report_sections(text, fallback_title=title or label)
            rep.title = title or label
            rep.sections = [{"title": s["title"], "content": s["content"],
                             "digest": digest(s["content"])} for s in sections]
            if not rep.sections:
                raise ValueError("本文に章として読める部分がありません")
            rep.text = text
        except Exception as e:  # noqa: BLE001 - one unreadable material is reported
            rep.error = f"{e}"[:300]
        out.append(rep)
    return out


# ----------------------------------------------------------------------
# comparison
# ----------------------------------------------------------------------

@dataclass
class Topic:
    topic: str
    status: str                       # agree | conflict | external_only | report_only
    section: Optional[str] = None     # the report chapter it belongs to
    report: str = ""                  # what the report says
    external: str = ""                # what the material says
    label: str = ""                   # which material
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"topic": self.topic, "status": self.status,
                "status_label": STATUS_LABELS.get(self.status, self.status),
                "section": self.section, "report": self.report, "external": self.external,
                "label": self.label, "note": self.note}


@dataclass
class Comparison:
    topics: List[Topic] = field(default_factory=list)
    source: str = "llm"               # llm | fallback | none

    def by_status(self, status: str) -> List[Topic]:
        return [t for t in self.topics if t.status == status]

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "topics": [t.to_dict() for t in self.topics]}


def build_comparison_prompt(session, reports: Sequence[ExternalReport], language: str = "ja") -> str:
    from .revision import flat_sections, section_digest
    toc = "\n".join(f"- {it.section}: {it.title} — {section_digest(session, it) or '(本文なし)'}"
                    for it in flat_sections(session))
    mats = []
    for r in reports:
        if r.error:
            continue
        lines = [f"■ {r.label}（{r.title}）"]
        for s in r.sections[:30]:
            lines.append(f"  - {s['title']} — {s['digest']}")
        mats.append("\n".join(lines))
    return f"""あなたは調査レポートの編集者です。本レポートと、別の調査で作られた外部資料を話題ごとに照合し、JSON で答えてください。

【調査テーマ】{session.query}

【本レポートの章と要旨】
{toc or "(章なし)"}

【外部の調査資料（章と要旨・数値）】
{chr(10).join(mats) or "(なし)"}

規則:
- 話題は「市場規模」「主要メーカーの戦略」のような短い名前にし、{MAX_TOPICS} 件まで。
- status は次のいずれか:
  "agree"（両方にあり内容が一致）、"conflict"（両方にあるが数値・年・結論が食い違う）、
  "external_only"（外部資料にだけある）、"report_only"（本レポートにだけある）。
- section は、その話題が属する本レポートの章番号（例 "2"）。external_only で該当する章が無ければ null。
- report / external には、それぞれが述べている内容を 1 文で（数値・年を含める）。conflict は何がどう違うかが分かるように書く。
- label は外部資料の名前（■ の後ろの名前）。
- 推測で食い違いを作らない。要旨から判断できないものは agree か report_only にしない。分からなければその話題は出さない。

JSON のみを出力:
{{"topics": [{{"topic": "市場規模", "status": "conflict", "section": "2", "report": "1,900 億円（2023 年）", "external": "2,400 億円（2025 年）", "label": "...", "note": "年が異なる"}}]}}"""


def parse_comparison(text: str, session, reports: Sequence[ExternalReport]) -> Comparison:
    from .revision import flat_sections
    data = extract_json_from_response(text or "")
    if not isinstance(data, dict) or not isinstance(data.get("topics"), list):
        raise ValueError("no topics in the comparison answer")
    sections = {str(s.section) for s in flat_sections(session)}
    labels = [r.label for r in reports if not r.error]
    topics: List[Topic] = []
    for t in data["topics"][:MAX_TOPICS]:
        if not isinstance(t, dict):
            continue
        name = " ".join(str(t.get("topic") or "").split())[:80]
        status = str(t.get("status") or "").strip().lower()
        if not name or status not in STATUS_LABELS:
            continue
        sec = str(t.get("section") or "").strip() or None
        if sec is not None and sec not in sections:
            sec = sec.split(".")[0] if sec.split(".")[0] in sections else None
        label = " ".join(str(t.get("label") or "").split())
        if label not in labels:
            label = labels[0] if len(labels) == 1 else (label or "")
        topics.append(Topic(topic=name, status=status, section=sec,
                            report=" ".join(str(t.get("report") or "").split())[:300],
                            external=" ".join(str(t.get("external") or "").split())[:300],
                            label=label, note=" ".join(str(t.get("note") or "").split())[:200]))
    if not topics:
        raise ValueError("the comparison lists no topic")
    return Comparison(topics=topics, source="llm")


def _bigrams(text: str) -> set:
    s = re.sub(r"\s+", "", text or "")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def fallback_comparison(session, reports: Sequence[ExternalReport],
                        threshold: float = 0.25) -> Comparison:
    """No usable LLM answer: each external chapter is matched to the report
    chapter whose text it overlaps most (character bigrams). Matched →
    agree (no conflict is claimed without reading), unmatched →
    external_only."""
    from .revision import flat_sections
    secs = [(str(it.section), it.title,
             _bigrams(it.title + " " + ((session.section_contents.get(str(it.section)) or {})
                                        .get("content") or "")[:4000]))
            for it in flat_sections(session)]
    topics: List[Topic] = []
    for r in reports:
        if r.error:
            continue
        for s in r.sections:
            grams = _bigrams(s["title"] + " " + s["content"][:4000])
            best, score = None, 0.0
            for sid, _title, g in secs:
                if not grams or not g:
                    continue
                sc = len(grams & g) / max(1, min(len(grams), 2000))
                if sc > score:
                    best, score = sid, sc
            status = "agree" if best is not None and score >= threshold else "external_only"
            topics.append(Topic(topic=s["title"][:80], status=status,
                                section=best if status == "agree" else None,
                                external=s["digest"][:300], label=r.label,
                                note="規則による照合（内容の一致・食い違いは未判定）"))
            if len(topics) >= MAX_TOPICS:
                break
    return Comparison(topics=topics, source="fallback")


def compare_reports(llm, session, reports: Sequence[ExternalReport], *,
                    language: str = "ja") -> Comparison:
    usable = [r for r in reports if not r.error]
    if not usable:
        return Comparison(topics=[], source="none")
    try:
        response = llm.generate(build_comparison_prompt(session, usable, language))
        return parse_comparison(str(getattr(response, "content", response)), session, usable)
    except Exception as e:  # noqa: BLE001 - never abort on the comparison
        ResearchWarnings.get_instance().add(
            "MEDIUM", "External",
            f"外部資料との照合を LLM から得られなかったため、本文の重なりで照合しました（{e}）")
        return fallback_comparison(session, usable)


def comparison_instructions(comparison: Comparison, mode: str) -> str:
    """The comparison as reader feedback for the revision planner."""
    lines: List[str] = []
    for t in comparison.by_status("conflict"):
        where = f"第{t.section}章の" if t.section else ""
        lines.append(f"- {where}「{t.topic}」: 外部資料（{t.label}）は「{t.external}」、本レポートは"
                     f"「{t.report}」と食い違う。一次資料で確認し、違いがあれば両論を根拠つきで書く。")
    ext_only = comparison.by_status("external_only")
    for t in ext_only:
        if t.section:
            lines.append(f"- 第{t.section}章に、外部資料（{t.label}）にある「{t.topic}」"
                         f"（{t.external}）を一次資料で確認して加える。")
        elif mode == "add":
            lines.append(f"- 外部資料（{t.label}）にあり本レポートに無い話題「{t.topic}」"
                         f"（{t.external}）を、一次資料で確認して新しい章として追加する。")
        else:
            lines.append(f"- 外部資料（{t.label}）にある話題「{t.topic}」は、関係する章があれば"
                         "一次資料で確認して加える（章は追加しない）。")
    agree = comparison.by_status("agree")
    if agree:
        lines.append("- 次の話題は外部資料と一致（記録のみ・章は変えない）: "
                     + "、".join(t.topic for t in agree[:15]))
    if not lines:
        return ""
    return "【外部の調査資料との照合結果】\n" + "\n".join(lines)


def comparison_preview(comparison: Comparison) -> str:
    """One text block for the plan review panel."""
    rows = []
    for t in comparison.topics:
        sec = f"第{t.section}章" if t.section else "（章なし）"
        rows.append(f"【{STATUS_LABELS.get(t.status, t.status)}】{t.topic} {sec}"
                    + (f" 本: {t.report}" if t.report else "")
                    + (f" / 外部（{t.label}）: {t.external}" if t.external else ""))
    return "外部資料との照合:\n" + "\n".join(rows) if rows else ""


# ----------------------------------------------------------------------
# material for the research loop
# ----------------------------------------------------------------------

def external_documents(reports: Sequence[ExternalReport]) -> List[Dict[str, Any]]:
    """Each chapter of each material as a local document (secondary
    information, cited as 外部調査（label）)."""
    docs: List[Dict[str, Any]] = []
    for r in reports:
        if r.error:
            continue
        for i, s in enumerate(r.sections, 1):
            docs.append({
                "path": f"external/{r.label}/{i:02d}",
                "title": f"外部調査（{r.label}）: {s['title']}",
                "content": s["content"],
                "metadata": {"source": "external", "external_label": r.label,
                             "external_title": r.title, "external_section": s["title"],
                             "external_file": Path(r.path).name if r.path else None},
            })
    return docs


def cited_urls(reports: Sequence[ExternalReport], limit: int = MAX_CITED_URLS) -> List[str]:
    out: List[str] = []
    for r in reports:
        for u in r.urls:
            if u["url"] not in out:
                out.append(u["url"])
            if len(out) >= limit:
                return out
    return out


# ----------------------------------------------------------------------
# outcome
# ----------------------------------------------------------------------

def _is_external(ev) -> bool:
    return (getattr(ev, "access_method", "") == "external"
            or ((getattr(ev, "metadata", None) or {}).get("source") == "external"))


def external_summary(reports: Sequence[ExternalReport], comparison: Comparison, *,
                     locker=None, revision: Optional[Dict[str, Any]] = None,
                     cited: Sequence[str] = (), url_report: Optional[Dict[str, Any]] = None,
                     body_as_evidence: bool = True, fetch_cited: bool = True,
                     parent_evidence_ids: Optional[set] = None) -> Dict[str, Any]:
    """Materials, the comparison with the outcome per topic, and the
    chapters whose new text rests on the external material alone.

    Only the evidence THIS run gathered counts (``parent_evidence_ids`` =
    the original report's evidence copied into the new locker). In full
    mode the new report has its own chapter numbering: every topic is
    matched to a new chapter by title, never by the original number."""
    full = bool(revision) and (revision.get("mode") == "full")
    sec_rows = list((revision or {}).get("sections") or [])
    changed = set()
    if revision:
        changed = set(revision.get("revised") or []) | set(revision.get("added") or [])
        if full:
            changed = {str(r.get("section")) for r in sec_rows}
    skip_ids = set(parent_evidence_ids or ())
    per_section: Dict[str, Dict[str, int]] = {}
    if locker is not None:
        try:
            for ev in locker.get_all_evidence():
                if getattr(ev, "id", None) in skip_ids:
                    continue
                sid = str(getattr(ev, "section_reference", "") or "")
                if not sid:
                    continue
                c = per_section.setdefault(sid, {"external": 0, "primary": 0})
                c["external" if _is_external(ev) else "primary"] += 1
        except Exception:
            per_section = {}
    # an external-only topic has no chapter of the report (and in full mode
    # no number survives): the chapter for it is found by its title. The
    # overlap is measured against the LONGER title and must exceed half,
    # so a shared word ("メーカーの") does not make a match
    def _chapter_for(topic: str, any_action: bool) -> Optional[str]:
        best, score = None, 0.0
        tg = _bigrams(topic)
        for r in sec_rows:
            if not any_action and r.get("action") not in ("add", "revise"):
                continue
            title = str(r.get("title") or "")
            if topic and title and (topic in title or title in topic):
                return str(r.get("section"))
            g = _bigrams(title)
            if tg and g:
                sc = len(tg & g) / max(len(tg), len(g))
                if sc > score:
                    best, score = str(r.get("section")), sc
        return best if score > 0.5 else None

    rows = []
    for t in comparison.topics:
        row = t.to_dict()
        section = t.section
        if full and t.status in ("conflict", "external_only"):
            section = _chapter_for(t.topic, any_action=True)
            row["matched_section"] = section
        elif t.status == "external_only" and not section:
            section = _chapter_for(t.topic, any_action=False)
            if section:
                row["matched_section"] = section
        counts = per_section.get(str(section or ""), {})
        if t.status in ("conflict", "external_only") and section in changed:
            if full:
                prefix = f"全面再調査・第{section}章・"
            elif section != t.section and section in set((revision or {}).get("added") or []):
                prefix = f"第{section}章を追加・"
            else:
                prefix = ""
            if counts.get("primary"):
                row["outcome"] = f"{prefix}一次資料 {counts['primary']} 件で再調査"
            elif counts.get("external"):
                row["outcome"] = f"{prefix}一次資料なし（外部資料のみ・要確認）"
            else:
                row["outcome"] = f"{prefix}再調査（根拠の記録なし）"
        elif full and t.status in ("conflict", "external_only"):
            row["outcome"] = "全面再調査（対応する章を特定できず）"
        elif t.status == "agree":
            row["outcome"] = "一致（そのまま）"
        elif t.status == "report_only":
            row["outcome"] = "本レポートのみ（そのまま）"
        elif t.status == "external_only" and changed:
            row["outcome"] = "対応する章なし（修正計画で扱われず）"
        else:
            row["outcome"] = "対応なし"
        rows.append(row)
    external_only_sections = sorted(
        (sid for sid in changed
         if per_section.get(sid, {}).get("external") and not per_section.get(sid, {}).get("primary")),
        key=lambda s: [int(p) if p.isdigit() else p for p in s.split(".")])
    fetched = None
    failed: List[str] = []
    if url_report:
        cited_set = set(cited)
        fetched = sum(1 for u in (url_report.get("urls") or [])
                      if isinstance(u, dict) and u.get("url") in cited_set and u.get("ok"))
        failed = [u.get("url") for u in (url_report.get("urls") or [])
                  if isinstance(u, dict) and u.get("url") in cited_set and not u.get("ok")]
    return {
        "materials": [r.info() for r in reports],
        "comparison_source": comparison.source,
        "topics": rows,
        "counts": {s: len(comparison.by_status(s)) for s in STATUS_LABELS},
        "cited_urls": list(cited),
        "cited_fetched": fetched,
        "cited_failed": failed[:30],
        "body_as_evidence": body_as_evidence,
        "fetch_cited": fetch_cited,
        "external_only_sections": external_only_sections,
    }


__all__ = [
    "ExternalReport", "Topic", "Comparison", "load_external_reports", "extract_cited_urls",
    "digest", "compare_reports", "fallback_comparison", "parse_comparison",
    "build_comparison_prompt", "comparison_instructions", "comparison_preview",
    "external_documents", "cited_urls", "external_summary", "WRITING_RULE",
    "EXTERNAL_MAX_IMPORTANCE", "MAX_EXTERNAL_REPORTS", "MAX_EXTERNAL_CHARS", "MAX_CITED_URLS",
    "FILE_SUFFIXES",
]
