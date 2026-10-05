"""Avoid redundant rewrites and reject factual changes in prose-only edits."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deep_research_tool.main import DeepResearchTool
from deep_research_tool.report.v2.generator import ChapterContent, ReportGeneratorV2
from deep_research_tool.report.v2.context import ReportContext
from deep_research_tool.utils.timing import StageTimings


@pytest.mark.parametrize("old,new", [
    ("10 mg [SOURCE 1]", "90 mg [SOURCE 1]"),
    ("10 mg [SOURCE 1]", "10 mg [SOURCE 9]"),
    ("10 mg and 20 kg [SOURCE 1]", "20 mg and 10 kg [SOURCE 1]"),
    ("Revenue in 2025 [SOURCE 1]", "Revenue in 2026 [SOURCE 1]"),
    ("Reported growth [SOURCE:1]", "Reported growth 1"),
    ("Alpha 10 mg, Beta 20 mg [SOURCE 1]", "Alpha 20 mg, Beta 10 mg [SOURCE 1]"),
])
def test_prose_edit_rejects_changed_quantities_and_citations(old, new):
    original = "## 1. Results\n\n" + old
    llm = Mock()
    llm.generate.return_value = SimpleNamespace(content="## 1. Results\n\n" + new)
    generator = ReportGeneratorV2(llm, language="en")
    chapters = {"1": ChapterContent("1", "Results", original)}
    result = generator._polish_chapters(chapters, ReportContext(research_topic="Results", language="en"))
    assert result["1"].content == original
    assert result["1"].is_draft


def test_polish_only_selected_chapter():
    llm = Mock()
    body = "## 2. Results\n\nA useful discussion of the evidence."
    llm.generate.return_value = SimpleNamespace(content=body)
    generator = ReportGeneratorV2(llm, language="en")
    chapters = {"1": ChapterContent("1", "First", "## 1. First\n\nFirst chapter remains intact."),
                "2": ChapterContent("2", "Results", body)}
    generator._polish_chapters(chapters, ReportContext(research_topic="Results", language="en"), section_ids={"2"})
    assert llm.generate.call_count == 1
    assert "First chapter remains intact" in llm.generate.call_args.args[0]


def test_complete_v1_draft_does_not_refetch_or_rewrite():
    tool = DeepResearchTool.__new__(DeepResearchTool)
    tool.search_client = Mock()
    tool._enhance_section_content = Mock()
    locker = Mock()
    session = SimpleNamespace(section_contents={"1": {"content": "The supported finding. " * 20 + "[SOURCE 1]", "sources": ["https://example.com"]}})
    assert tool._enhance_sections_with_full_evidence(session, locker, "question") is session
    tool.search_client.get_page_content.assert_not_called()
    tool._enhance_section_content.assert_not_called()


@pytest.mark.parametrize("quality", [{"confidence": "low"}, {"gaps": ["missing fact"]}])
def test_v1_incomplete_persisted_draft_remains_selected(quality):
    tool = DeepResearchTool.__new__(DeepResearchTool)
    tool.search_client = Mock()
    tool._enhance_section_content = Mock(return_value="Improved. " * 50)
    tool._build_evidence_overview = Mock(return_value="Evidence")
    locker = Mock()
    locker.get_all_evidence.return_value = [SimpleNamespace(url="https://example.com", title="Source", content_excerpt="Evidence", relevance_score=0.1)]
    session = SimpleNamespace(section_contents={"1": {"content": "Supported finding. " * 20 + "[SOURCE 1]", "sources": ["https://example.com"], **quality}})
    tool._enhance_sections_with_full_evidence(session, locker, "question")
    tool._enhance_section_content.assert_called_once()


def test_stage_timings_record_failures_without_hiding_exception():
    timings = StageTimings()
    with pytest.raises(ValueError, match="failed"):
        timings.call("search", lambda: (_ for _ in ()).throw(ValueError("failed")))
    assert timings.call("writing", lambda: "done") == "done"
    snapshot = timings.snapshot()
    assert snapshot["stages"]["search"]["failures"] == 1
    assert snapshot["stages"]["writing"]["calls"] == 1


@pytest.mark.parametrize("kind", ["llm", "search"])
@pytest.mark.parametrize("cancel_at", [1, 2])
def test_leaf_calls_recheck_cancel_after_waiting_without_leaking_permits(kind, cancel_at):
    from deep_research_tool.api.base import BaseLLMClient
    from deep_research_tool.search.base import BaseSearchClient
    from deep_research_tool.utils.concurrency import RunLimits
    from deep_research_tool.main import RunCancelled

    limiter = RunLimits(1)
    calls = []
    def check():
        calls.append(True)
        if len(calls) == cancel_at:
            raise RunCancelled("stopped")
    client = SimpleNamespace(concurrency_limiter=limiter, cancel_check=check)
    permit = BaseLLMClient._leaf_permit if kind == "llm" else BaseSearchClient._leaf_permit
    with pytest.raises(RunCancelled):
        with permit(client):
            pytest.fail("network call must not start")
    assert limiter.run_limiter.active == 0


def test_composed_permit_consumes_one_deadline(monkeypatch):
    from contextlib import contextmanager
    from deep_research_tool.utils.concurrency import RunLimits
    from deep_research_tool.utils import concurrency
    clock = iter([10.0, 10.07])
    monkeypatch.setattr(concurrency.time, "monotonic", lambda: next(clock))
    received = []
    @contextmanager
    def permit(timeout=None):
        received.append(timeout)
        yield
    limiter = RunLimits(1)
    limiter.run_limiter = SimpleNamespace(permit=permit)
    limiter.process_limiter = SimpleNamespace(permit=permit)
    with limiter.permit(timeout=0.1):
        pass
    assert received == pytest.approx([0.1, 0.03])
