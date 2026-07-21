"""
Tests for the live report sinks (Web UI preview + Word COM writing).

The Word COM boundary is exercised through an injected fake adapter, so
these tests run anywhere (no Windows / Word / pywin32 required) while
covering the sink's threading, ordering, draft->final replacement and
failure isolation.
"""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from deep_research_tool.report.live_report import (
    DRAFT_MARK_JA,
    CompositeSink,
    LiveReportSink,
    WebUILiveSink,
    WordComSink,
    _plain_paragraphs,
)


class FakeWordAdapter:
    """Same surface as RealWordAdapter; records every call in order."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on or set()
        self.saved_path = None

    def _rec(self, name, *args):
        if name in self.fail_on:
            raise RuntimeError(f"COM failure in {name}")
        self.calls.append((name,) + args)

    def start(self):
        self._rec("start")

    def quit_keep_open(self):
        self._rec("quit_keep_open")

    def add_title(self, title):
        self._rec("add_title", title)

    def begin_section(self, sid):
        self._rec("begin_section", sid)

    def end_section(self, sid):
        self._rec("end_section", sid)

    def write_paragraph(self, text, style="body"):
        self._rec("write_paragraph", text, style)

    def add_picture(self, path, caption=""):
        self._rec("add_picture", path, caption)

    def replace_section(self, sid, paragraphs):
        self._rec("replace_section", sid,
                  [p["text"] for p in paragraphs])

    def remove_draft_marks(self, mark):
        self._rec("remove_draft_marks", mark)

    def save_as(self, path):
        self.saved_path = path
        self._rec("save_as", path)

    def names(self):
        return [c[0] for c in self.calls]


class TestWordComSink(unittest.TestCase):

    def _sink(self, adapter):
        sink = WordComSink(output_path="C:/out/report_live.docx",
                           language="ja",
                           adapter_factory=lambda: adapter)
        sink.started.wait(5)
        return sink

    def test_full_flow_draft_then_finalized(self):
        adapter = FakeWordAdapter()
        sink = self._sink(adapter)

        sink.on_plan("市場調査レポート",
                     [{"section": "1", "title": "シェア"}])
        sink.on_section("1", "シェア",
                        "## 1. シェア\n\nA社は52% [SOURCE 1]。", draft=True)
        sink.on_figure("1", "/figs/chart.png", "図1 シェア推移")
        # finalization rewrote the chapter with display numbers
        sink.on_finalized({"1": "## 1. シェア\n\nA社は52% [1]。"},
                          ["出典A (2026)"])
        sink.close()

        names = adapter.names()
        # lifecycle & ordering
        self.assertEqual(names[0], "start")
        self.assertEqual(names[-1], "quit_keep_open")
        self.assertIn("add_title", names)
        self.assertIn("begin_section", names)
        self.assertIn("add_picture", names)
        # draft watermark written, then removed at finalize
        drafts = [c for c in adapter.calls
                  if c[0] == "write_paragraph" and c[1] == DRAFT_MARK_JA]
        self.assertTrue(drafts)
        self.assertIn(("remove_draft_marks", DRAFT_MARK_JA), adapter.calls)
        # the final body REPLACED the section (not appended twice)
        replaced = [c for c in adapter.calls if c[0] == "replace_section"]
        self.assertEqual(len(replaced), 1)
        self.assertEqual(replaced[0][1], "1")
        self.assertTrue(any("[1]" in t for t in replaced[0][2]))
        # document saved to the requested path
        self.assertEqual(adapter.saved_path, "C:/out/report_live.docx")

    def test_second_emit_of_same_section_replaces(self):
        adapter = FakeWordAdapter()
        sink = self._sink(adapter)
        sink.on_section("2", "動向", "初稿", draft=True)
        sink.on_section("2", "動向", "推敲後の本文", draft=True)
        sink.close()
        self.assertEqual(adapter.names().count("begin_section"), 1)
        replaced = [c for c in adapter.calls if c[0] == "replace_section"]
        self.assertEqual(len(replaced), 1)
        self.assertEqual(replaced[0][2], ["推敲後の本文"])

    def test_word_unavailable_disables_gracefully(self):
        def boom():
            raise ImportError("No module named 'win32com'")
        sink = WordComSink(adapter_factory=boom)
        sink.started.wait(5)
        self.assertTrue(sink.failed_to_start)
        # events become no-ops, nothing raises
        sink.on_section("1", "t", "text")
        sink.on_finalized({"1": "x"}, [])
        sink.close()

    def test_repeated_com_failures_disable_sink(self):
        adapter = FakeWordAdapter(fail_on={"write_paragraph"})
        sink = self._sink(adapter)
        for i in range(WordComSink.MAX_FAILURES + 2):
            sink.on_section(str(i), "t", f"本文{i}")
        sink.close()
        self.assertTrue(sink.disabled)


class TestWebUILiveSink(unittest.TestCase):

    def test_snapshot_progression(self):
        sink = WebUILiveSink()
        self.assertEqual(sink.snapshot()["rev"], 0)

        sink.on_plan("タイトル", [{"section": "1", "title": "章1"}])
        sink.on_section("1", "章1", "草稿本文", draft=True)
        sink.on_figure("1", "/figs/a.png", "図")
        snap = sink.snapshot()
        self.assertEqual(snap["title"], "タイトル")
        self.assertTrue(snap["sections"]["1"]["draft"])
        self.assertEqual(snap["sections"]["1"]["figures"][0]["path"],
                         "/figs/a.png")
        self.assertFalse(snap["finalized"])

        sink.on_finalized({"1": "確定本文 [1]"}, ["出典A"])
        snap = sink.snapshot()
        self.assertTrue(snap["finalized"])
        self.assertFalse(snap["sections"]["1"]["draft"])
        self.assertEqual(snap["sections"]["1"]["text"], "確定本文 [1]")
        self.assertEqual(snap["references"], ["出典A"])
        self.assertGreater(snap["rev"], 0)


class TestCompositeSink(unittest.TestCase):

    def test_failing_sink_is_isolated_and_removed(self):
        good = WebUILiveSink()
        bad = MagicMock(spec=LiveReportSink)
        bad.on_section.side_effect = RuntimeError("boom")
        combo = CompositeSink([bad, good])
        combo.on_section("1", "t", "text")     # bad fails, gets removed
        combo.on_section("1", "t", "text2")    # good keeps receiving
        self.assertEqual(good.snapshot()["sections"]["1"]["text"], "text2")
        self.assertEqual(bad.on_section.call_count, 1)


class TestMarkdownParagraphs(unittest.TestCase):

    def test_headings_lists_and_body(self):
        paras = _plain_paragraphs(
            "## 1. 見出し\n\n本文の段落。**強調**あり。\n- 箇条書き1\n")
        self.assertEqual(paras[0], {"style": "h2", "text": "1. 見出し"})
        self.assertEqual(paras[1]["style"], "body")
        self.assertNotIn("**", paras[1]["text"])
        self.assertEqual(paras[2], {"style": "list", "text": "箇条書き1"})


class TestPipelineWiring(unittest.TestCase):
    """The real run() emits plan -> sections -> finalized to the sink."""

    def test_run_emits_live_events(self):
        from deep_research_tool.tests.test_finalization_integration import (
            FakeSearch, make_llm, make_locker, build_tool, run_tool,
        )
        from deep_research_tool.tests import test_finalization_integration as it
        from deep_research_tool.utils.helpers import ResearchWarnings

        ResearchWarnings.reset()
        tmp = Path(tempfile.mkdtemp())
        llm = make_llm()
        locker = make_locker(tmp, it.SUPPORTED_EVIDENCE)
        tool = build_tool(tmp, llm, FakeSearch())
        sink = WebUILiveSink()

        plan = it.make_plan([("1", "市場シェア", "A社のシェア")])
        session = it.make_session(plan, {
            "1": {"title": "市場シェア",
                  "content": it.long_body("1", "市場シェア",
                                          it.CLAIM_SENTENCE),
                  "evidence": it.SUPPORTED_EVIDENCE},
        })

        from unittest.mock import patch
        fake = it.FakeResearcher(session, locker)
        with patch("deep_research_tool.main.Researcher",
                   new=lambda **kw: fake):
            tool.run(query="市場調査", live_sink=sink)

        snap = sink.snapshot()
        self.assertEqual(snap["title"], "統合テストレポート")   # on_plan
        self.assertIn("1", snap["sections"])                    # on_section
        self.assertTrue(snap["finalized"])                      # on_finalized
        # the live final body is the VERIFIED, display-numbered text
        self.assertIn("[1]", snap["sections"]["1"]["text"])
        self.assertNotIn("[SOURCE", snap["sections"]["1"]["text"])
        self.assertTrue(snap["references"])


if __name__ == "__main__":
    unittest.main()
