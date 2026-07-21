"""
Live report sinks - progressive writing of the report while research runs.

The pipeline emits events at natural stage boundaries and one or more
sinks consume them:

    on_plan(title, toc)                research plan confirmed
    on_section(sid, title, text)       a chapter draft was written/updated
    on_figure(sid, image_path, ...)    a chart/figure was generated
    on_finalized(chapters, references) verified final body (post-freeze)
    on_status(message)                 free-form progress note
    close()                            run finished (always called)

IMPORTANT: everything before ``on_finalized`` is a DRAFT. The
finalization loop may rewrite chapters and the citation numbers are
substituted only at freeze, so live output is watermarked as 草稿 and
replaced wholesale when the final body arrives.

Sinks must never break the research run: every event is wrapped so a
failing sink logs, disables itself, and the pipeline continues.

Provided sinks:
- CompositeSink: fan-out with per-sink error isolation
- WebUILiveSink: pushes snapshots into a thread-safe store (the Web UI
  job object) that /api/live-report serves
- WordComSink: writes into a VISIBLE Microsoft Word window via COM
  (Windows + installed Word). All COM calls run on one dedicated STA
  worker thread; the pipeline only enqueues events.
"""

import queue
import re
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional


class LiveReportSink:
    """Base sink: all events are optional no-ops."""

    def on_plan(self, title: str, toc: List[Dict[str, str]]) -> None:
        pass

    def on_section(self, section_id: str, title: str, text: str,
                   draft: bool = True) -> None:
        pass

    def on_figure(self, section_id: str, image_path: str,
                  caption: str = "") -> None:
        pass

    def on_finalized(self, chapters: Dict[str, str],
                     references: List[str]) -> None:
        pass

    def on_status(self, message: str) -> None:
        pass

    def close(self) -> None:
        pass


class CompositeSink(LiveReportSink):
    """Fan out to several sinks; one failing sink never affects others."""

    def __init__(self, sinks: List[LiveReportSink]):
        self.sinks = [s for s in (sinks or []) if s is not None]

    def _each(self, method: str, *args, **kwargs) -> None:
        for sink in list(self.sinks):
            try:
                getattr(sink, method)(*args, **kwargs)
            except Exception as e:
                print(f"[LiveReport] sink {type(sink).__name__}.{method} "
                      f"failed: {e}; disabling this sink")
                try:
                    self.sinks.remove(sink)
                except ValueError:
                    pass

    def on_plan(self, title, toc):
        self._each("on_plan", title, toc)

    def on_section(self, section_id, title, text, draft=True):
        self._each("on_section", section_id, title, text, draft=draft)

    def on_figure(self, section_id, image_path, caption=""):
        self._each("on_figure", section_id, image_path, caption=caption)

    def on_finalized(self, chapters, references):
        self._each("on_finalized", chapters, references)

    def on_status(self, message):
        self._each("on_status", message)

    def close(self):
        self._each("close")


class WebUILiveSink(LiveReportSink):
    """Thread-safe snapshot store consumed by the Web UI live panel.

    The snapshot shape (all strings JSON-safe):
        {"title": str, "toc": [{"section","title"}],
         "sections": {sid: {"title","text","draft","figures":[...]}},
         "finalized": bool, "references": [str], "rev": int}
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {
            "title": "", "toc": [], "sections": {},
            "finalized": False, "references": [], "rev": 0,
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            # sections copied shallowly; values are replaced, not mutated
            return {
                "title": self._data["title"],
                "toc": list(self._data["toc"]),
                "sections": {k: dict(v)
                             for k, v in self._data["sections"].items()},
                "finalized": self._data["finalized"],
                "references": list(self._data["references"]),
                "rev": self._data["rev"],
            }

    def _bump(self):
        self._data["rev"] += 1

    def on_plan(self, title, toc):
        with self._lock:
            self._data["title"] = title
            self._data["toc"] = [dict(t) for t in toc]
            self._bump()

    def on_section(self, section_id, title, text, draft=True):
        with self._lock:
            entry = self._data["sections"].setdefault(
                section_id, {"figures": []})
            entry.update({"title": title, "text": text, "draft": draft})
            self._bump()

    def on_figure(self, section_id, image_path, caption=""):
        with self._lock:
            entry = self._data["sections"].setdefault(
                section_id, {"title": section_id, "text": "", "draft": True,
                             "figures": []})
            entry.setdefault("figures", []).append(
                {"path": str(image_path), "caption": caption})
            self._bump()

    def on_finalized(self, chapters, references):
        with self._lock:
            for sid, text in chapters.items():
                entry = self._data["sections"].setdefault(
                    sid, {"title": sid, "figures": []})
                entry.update({"text": text, "draft": False})
            self._data["finalized"] = True
            self._data["references"] = list(references)
            self._bump()


# ---------------------------------------------------------------------------
# Word COM live writing (Windows + Microsoft Word)
# ---------------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s*(.*)$")
_MD_TOKEN_RE = re.compile(r"[*_`]+")

DRAFT_MARK_JA = "【草稿】検証前の内容です。最終確定時に自動で置き換わります。"
DRAFT_MARK_EN = "[DRAFT] Pre-verification content; replaced automatically on finalization."


def _plain_paragraphs(text: str) -> List[Dict[str, Any]]:
    """Minimal markdown -> [{style, text}] for Word paragraphs."""
    out = []
    for block in (text or "").split("\n"):
        line = block.rstrip()
        if not line.strip():
            continue
        m = _MD_HEADING_RE.match(line.strip())
        if m:
            level = min(len(m.group(1)), 4)
            out.append({"style": f"h{level}",
                        "text": _MD_TOKEN_RE.sub("", m.group(2))})
            continue
        stripped = line.strip()
        if stripped.startswith(("- ", "* ")):
            out.append({"style": "list",
                        "text": _MD_TOKEN_RE.sub("", stripped[2:])})
            continue
        out.append({"style": "body", "text": _MD_TOKEN_RE.sub("", stripped)})
    return out


class RealWordAdapter:
    """Thin wrapper over the Word COM object model.

    Kept minimal so tests can inject a fake with the same surface:
        start(), add_title(), begin_section(sid), write_paragraph(),
        add_picture(), replace_section(), remove_draft_marks(),
        save_as(), quit_keep_open()
    Must only be used from the ONE thread that called start()
    (COM STA requirement).
    """

    def __init__(self, visible: bool = True):
        self.visible = visible
        self.word = None
        self.doc = None
        self._pythoncom = None

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        import pythoncom                     # noqa: F401 (pywin32)
        import win32com.client
        self._pythoncom = pythoncom
        pythoncom.CoInitialize()
        self.word = win32com.client.Dispatch("Word.Application")
        self.word.Visible = self.visible
        self.doc = self.word.Documents.Add()

    def quit_keep_open(self) -> None:
        """Leave the document open for the user; release COM."""
        if self._pythoncom is not None:
            self.word = None
            self.doc = None
            self._pythoncom.CoUninitialize()

    # -- writing --------------------------------------------------------

    def _append(self, text: str, style: str) -> None:
        rng = self.doc.Range()
        rng.Collapse(0)                       # wdCollapseEnd
        para = rng.Paragraphs.Add()
        para.Range.Text = text
        try:
            styles = {"h1": -2, "h2": -3, "h3": -4, "h4": -5,
                      "title": -63, "list": -66, "body": -1}
            para.Range.Style = styles.get(style, -1)   # builtin style ids
        except Exception:
            pass

    def add_title(self, title: str) -> None:
        self._append(title, "title")

    def begin_section(self, section_id: str) -> None:
        """Bookmark the start of a section so it can be replaced later."""
        rng = self.doc.Range()
        rng.Collapse(0)
        self.doc.Bookmarks.Add(f"drt_start_{_bm(section_id)}", rng)

    def end_section(self, section_id: str) -> None:
        rng = self.doc.Range()
        rng.Collapse(0)
        self.doc.Bookmarks.Add(f"drt_end_{_bm(section_id)}", rng)

    def write_paragraph(self, text: str, style: str = "body") -> None:
        self._append(text, style)

    def add_picture(self, image_path: str, caption: str = "") -> None:
        rng = self.doc.Range()
        rng.Collapse(0)
        self.doc.InlineShapes.AddPicture(str(image_path), False, True, rng)
        if caption:
            self._append(caption, "body")

    def replace_section(self, section_id: str,
                        paragraphs: List[Dict[str, Any]]) -> None:
        """Replace everything between the section's bookmarks."""
        start_name = f"drt_start_{_bm(section_id)}"
        end_name = f"drt_end_{_bm(section_id)}"
        if not (self.doc.Bookmarks.Exists(start_name)
                and self.doc.Bookmarks.Exists(end_name)):
            for p in paragraphs:            # unknown section: append
                self._append(p["text"], p["style"])
            return
        start = self.doc.Bookmarks(start_name).Range.Start
        end = self.doc.Bookmarks(end_name).Range.End
        rng = self.doc.Range(start, end)
        rng.Delete()
        rng.Collapse(1)                      # wdCollapseStart
        for p in paragraphs:
            para = rng.Paragraphs.Add()
            para.Range.Text = p["text"]
        # re-add bookmarks around the new content
        self.doc.Bookmarks.Add(start_name, self.doc.Range(start, start))

    def remove_draft_marks(self, mark: str) -> None:
        find = self.doc.Content.Find
        find.ClearFormatting()
        find.Execute(FindText=mark, ReplaceWith="", Replace=2)  # wdReplaceAll

    def save_as(self, path: str) -> None:
        self.doc.SaveAs2(str(path))


def _bm(section_id: str) -> str:
    """Word bookmark names: letters/digits/underscore only."""
    return re.sub(r"[^0-9A-Za-z_]", "_", str(section_id)) or "sec"


class WordComSink(LiveReportSink):
    """Live writing into a visible Word window.

    Pipeline threads only enqueue events; ONE worker thread owns the COM
    objects (STA). If Word/pywin32 is unavailable or a COM call fails
    repeatedly, the sink disables itself and the run continues.
    """

    MAX_FAILURES = 5

    def __init__(self, output_path=None, language: str = "ja",
                 adapter_factory: Optional[Callable] = None,
                 visible: bool = True):
        self.output_path = str(output_path) if output_path else ""
        self.language = language
        self.draft_mark = DRAFT_MARK_JA if language == "ja" else DRAFT_MARK_EN
        self._adapter_factory = adapter_factory or \
            (lambda: RealWordAdapter(visible=visible))
        self._queue: "queue.Queue" = queue.Queue()
        self._known_sections = set()
        self._failures = 0
        self.disabled = False
        self.started = threading.Event()
        self.failed_to_start = False
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name="word-live-sink")
        self._worker.start()

    # -- event API (pipeline side: enqueue only) -------------------------

    def _put(self, fn: Callable) -> None:
        if not self.disabled:
            self._queue.put(fn)

    def on_plan(self, title, toc):
        def _do(w):
            w.add_title(title)
            w.write_paragraph(self.draft_mark, "body")
            heading = "目次" if self.language == "ja" else "Table of Contents"
            w.write_paragraph(heading, "h2")
            for item in toc:
                w.write_paragraph(
                    f"{item.get('section', '')}. {item.get('title', '')}",
                    "list")
        self._put(_do)

    def on_section(self, section_id, title, text, draft=True):
        paragraphs = _plain_paragraphs(text)
        mark = self.draft_mark if draft else ""
        is_new = section_id not in self._known_sections
        self._known_sections.add(section_id)

        def _do(w):
            if not is_new:
                w.replace_section(section_id, paragraphs)
                return
            w.begin_section(section_id)
            if not paragraphs or paragraphs[0]["style"] not in (
                    "h1", "h2", "h3", "h4"):
                w.write_paragraph(f"{section_id}. {title}", "h2")
            if mark:
                w.write_paragraph(mark, "body")
            for p in paragraphs:
                w.write_paragraph(p["text"], p["style"])
            w.end_section(section_id)
        self._put(_do)

    def on_figure(self, section_id, image_path, caption=""):
        def _do(w):
            w.add_picture(str(image_path), caption)
        self._put(_do)

    def on_finalized(self, chapters, references):
        def _do(w):
            for sid, text in chapters.items():
                w.replace_section(sid, _plain_paragraphs(text))
            heading = "参考文献" if self.language == "ja" else "References"
            w.write_paragraph(heading, "h2")
            for i, ref in enumerate(references, 1):
                w.write_paragraph(f"{i}. {ref}", "list")
            w.remove_draft_marks(self.draft_mark)
            if self.output_path:
                w.save_as(self.output_path)
        self._put(_do)

    def on_status(self, message):
        pass    # progress stays in the console / Web UI

    def close(self):
        self._queue.put(None)
        self._worker.join(timeout=30)

    # -- worker (COM side) ------------------------------------------------

    def _run(self):
        adapter = None
        try:
            adapter = self._adapter_factory()
            adapter.start()
        except Exception as e:
            print(f"[LiveReport] Word COM unavailable: {e}. "
                  f"Live Word writing disabled (the run continues; "
                  f"requires Windows + Microsoft Word + pywin32).")
            self.disabled = True
            self.failed_to_start = True
            self.started.set()
            self._drain()
            return
        self.started.set()

        while True:
            fn = self._queue.get()
            if fn is None:
                break
            try:
                fn(adapter)
            except Exception as e:
                self._failures += 1
                print(f"[LiveReport] Word write failed: {e}")
                if self._failures >= self.MAX_FAILURES:
                    print("[LiveReport] too many Word failures; "
                          "disabling live Word writing")
                    self.disabled = True
                    self._drain()
                    break
        try:
            adapter.quit_keep_open()
        except Exception:
            traceback.print_exc()

    def _drain(self):
        while True:
            try:
                if self._queue.get_nowait() is None:
                    break
            except queue.Empty:
                break
