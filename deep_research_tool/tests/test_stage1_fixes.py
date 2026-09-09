"""
Regression tests for the GUI/操作性 review — Stage 1 (bug fixes).

1. Tkinter start contract: GUI values -> build_gui_config -> run_research
   entry (no duplicate kwargs), knobs forwarded, error message bound.
2. Checkpoint / stop / resume: atomic saves at section completion and on
   cancel; nothing new starts after a cancel; resume reuses completed
   sections and persists.
3. Live-report job switching is covered by the browser test; here the
   server-side pieces: log sequence ids, finished_at, queue, plan-review
   cancel, unique persistent job ids, ledger.
4. GUI config mapping: 0 vs empty vs invalid for numeric fields.
5. Process / verification / quality are separate.
6. Per-job warning isolation.
7. fast_parallel with zero pages; standard-mode URL dedup / independent
   source counting.
8. Fermi GUI save state.
"""

import json
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from deep_research_tool.config import CrawlMode, ResearchSourceMode
from deep_research_tool.gui_config import build_gui_config, describe_outcome
from deep_research_tool.research.researcher import (
    Researcher,
    ResearchSession,
    ResearchState,
)
from deep_research_tool.tests.test_e2e_pipeline import (
    FakeLLM,
    FakeSearchClient,
)
from deep_research_tool.utils.cancellation import RunCancelled
from deep_research_tool.utils.concurrency import ContextThreadPoolExecutor
from deep_research_tool.utils.helpers import ResearchWarnings
from deep_research_tool.webui.server import (
    FieldError,
    JobManager,
    ResearchJob,
    build_config_kwargs,
    scrub_secrets,
)


# ===========================================================================
# 1. Tkinter start contract
# ===========================================================================

GUI_VALUES = {
    "topic": "炭素繊維の市場調査", "provider": "openai",
    "openai_api_key": "sk-test", "openai_model": "gpt-5-mini",
    "min_iterations": 2, "max_iterations": 4, "parallel_max_workers": 4,
    "temperature": 0.3, "max_tokens": 2048, "region": "jp-jp",
    "include_images": False, "include_citations": True, "include_toc": False,
    "verification_strictness": "high",     # verify-command-only: dropped
}


class TestTkStartContract:

    def test_gui_config_uses_run_research_contract(self):
        cfg = build_gui_config(GUI_VALUES)
        assert cfg["iterations"] == 2          # NOT research_iterations
        assert "research_iterations" not in cfg
        assert cfg["temperature"] == 0.3 and cfg["max_tokens"] == 2048
        assert cfg["search_region"] == "jp-jp"
        assert cfg["include_images"] is False and cfg["include_toc"] is False
        assert "verification_strictness" not in cfg

    def test_gui_values_reach_run_research_without_type_error(self):
        """The REAL entry: run_research(**build_gui_config(...))."""
        from deep_research_tool import main as main_mod

        captured = {}

        class FakeTool:
            def __init__(self, config):
                captured["config"] = config

            def run(self, **kwargs):
                captured["run_kwargs"] = kwargs
                return {"report_path": "x.md", "warning_count": 0,
                        "status": {"process": "completed",
                                   "verification": "performed",
                                   "quality": "passed"}}

        cfg = build_gui_config(GUI_VALUES)
        topic = cfg.pop("topic")
        cancel = threading.Event()
        with patch.object(main_mod, "DeepResearchTool", FakeTool):
            result = main_mod.run_research(
                query=topic, progress_callback=lambda m, p: None,
                cancel_event=cancel, **cfg)
        config = captured["config"]
        assert config.research.min_iterations == 2
        assert config.research.max_iterations == 4
        assert config.research.parallel_max_workers == 4
        assert config.api.temperature == 0.3
        assert config.api.max_tokens == 2048
        assert config.search.region == "jp-jp"
        assert config.report.include_images is False
        assert config.report.include_toc is False
        # progress callback + cancel token are forwarded to the tool
        assert callable(captured["run_kwargs"]["progress_callback"])
        assert captured["run_kwargs"]["cancel_event"] is cancel
        assert result["report_path"] == "x.md"

    def test_outcome_classification_is_three_axes(self):
        # a failed artifact check is never a plain success
        bad = describe_outcome({"run_status": "failed_semantic_check",
                                "status": {"process": "completed",
                                           "verification": "performed",
                                           "quality": "failed"}})
        assert bad["ok"] is False and "検査" in bad["headline"]
        cancelled = describe_outcome({"verification_cancelled": True})
        assert cancelled["ok"] is False and "中止" in cancelled["headline"]
        unverified = describe_outcome({"status": {
            "process": "completed", "verification": "skipped",
            "quality": "unverified"}})
        assert unverified["ok"] is True and "未検証" in unverified["headline"]
        limited = describe_outcome({"status": {
            "process": "completed", "verification": "performed",
            "quality": "limitations"}})
        assert "要確認" in limited["headline"]
        passed = describe_outcome({"status": {
            "process": "completed", "verification": "performed",
            "quality": "passed"}})
        assert passed["ok"] and "品質基準" in passed["headline"]


# ===========================================================================
# 2. checkpoint / stop / resume
# ===========================================================================

def make_researcher(tmp_path, llm=None, search=None, **overrides):
    kwargs = dict(
        llm_client=llm or FakeLLM(), search_client=search or FakeSearchClient(),
        output_dir=tmp_path, source_mode=ResearchSourceMode.WEB,
        crawl_mode=CrawlMode.STANDARD, use_enhanced_synthesis=False,
        filter_mode="none", max_gap_fill_rounds=0,
        max_queries_per_iteration=2, max_pages_per_query=2,
    )
    kwargs.update(overrides)
    return Researcher(**kwargs)


class TestCheckpointStopResume:

    def test_cancel_after_first_section_persists_and_blocks_new_work(
            self, tmp_path):
        llm, search = FakeLLM(), FakeSearchClient()
        holder = {}
        counts_at_cancel = {}

        def cancel_check():
            r = holder.get("researcher")
            if r and r.session and "1" in r.session.completed_sections:
                # first checkpoint after section 1 -> the run is cancelled
                counts_at_cancel.setdefault("search", search.search_calls)
                counts_at_cancel.setdefault("fetch", search.fetch_calls)
                counts_at_cancel.setdefault("llm", len(llm.calls))
                raise RunCancelled("user stop")

        researcher = make_researcher(tmp_path, llm, search,
                                     cancel_check=cancel_check,
                                     run_config={"provider": "fake"})
        holder["researcher"] = researcher
        with pytest.raises(RunCancelled):
            researcher.conduct_research(query="炭素繊維の市場調査",
                                        requirements="目次は2章構成でよい")

        session = researcher.session
        assert session.state == ResearchState.CANCELLED
        assert session.completed_sections == ["1"]
        assert session.section_contents["1"]["content"]
        assert "2" not in session.section_contents
        # nothing NEW started after the cancel was accepted
        assert search.search_calls == counts_at_cancel["search"]
        assert search.fetch_calls == counts_at_cancel["fetch"]
        assert len(llm.calls) == counts_at_cancel["llm"]
        # atomic checkpoint: session + evidence really exist on disk and
        # the "saved" list names only files that exist
        saved = session.saved_artifacts
        assert set(saved) == {"session", "evidence_json"}
        for path in saved.values():
            assert Path(path).exists()
        data = json.loads(Path(saved["session"]).read_text(encoding="utf-8"))
        assert data["state"] == "cancelled"
        assert data["completed_sections"] == ["1"]
        assert data["stage"] == "cancelled"
        assert data["run_config"] == {"provider": "fake"}
        assert data["checkpoints"][-1]["stage"] == "cancelled"
        assert any(c["stage"] == "section_completed" for c in data["checkpoints"])
        ev = json.loads(Path(saved["evidence_json"]).read_text(encoding="utf-8"))
        assert ev["total_evidence"] >= 1
        assert "1" in ev["section_evidence"]

    def test_resume_reuses_completed_sections_and_persists(self, tmp_path):
        llm, search = FakeLLM(), FakeSearchClient()
        holder = {}

        def cancel_check():
            r = holder.get("researcher")
            if r and r.session and "1" in r.session.completed_sections \
                    and not holder.get("resuming"):
                raise RunCancelled("user stop")

        researcher = make_researcher(tmp_path, llm, search,
                                     cancel_check=cancel_check)
        holder["researcher"] = researcher
        with pytest.raises(RunCancelled):
            researcher.conduct_research(query="炭素繊維の市場調査",
                                        requirements="目次は2章構成でよい")
        session_path = Path(researcher.session.saved_artifacts["session"])

        # --- resume with a FRESH researcher (as after a restart) ---
        holder["resuming"] = True
        llm2, search2 = FakeLLM(), FakeSearchClient()
        resumed = make_researcher(tmp_path, llm2, search2)
        section1_before = json.loads(session_path.read_text(
            encoding="utf-8"))["section_contents"]["1"]["content"]
        session = resumed.resume_research(session_path)

        assert resumed.resume_plan == {"reused": ["1"], "to_process": ["2"]}
        assert session.state == ResearchState.COMPLETED
        # section 1 was NOT regenerated (identical text, one synthesis only)
        assert session.section_contents["1"]["content"] == section1_before
        assert sum("===SECTION_META===" in p for p in llm2.calls) == 1
        assert "2" in session.section_contents
        assert sorted(session.completed_sections) == ["1", "2"]
        # the resumed result is PERSISTED
        data = json.loads(session_path.read_text(encoding="utf-8"))
        assert data["state"] == "completed"
        assert sorted(data["completed_sections"]) == ["1", "2"]
        assert data["checkpoints"][-1]["stage"] == "completed"

    def test_session_save_is_atomic(self, tmp_path):
        s = ResearchSession(query="q")
        target = tmp_path / "s.json"
        s.save(target)
        assert target.exists()
        assert not (tmp_path / "s.json.tmp").exists()
        loaded = ResearchSession.load(target)
        assert loaded.query == "q"
        assert loaded.completed_sections == []

    def test_is_section_done_marker_and_legacy_fallback(self):
        s = ResearchSession(query="q")
        s.section_contents["1"] = {"content": "本文"}
        assert s.is_section_done("1") is True       # legacy fallback
        s.completed_sections.append("2")
        assert s.is_section_done("2") is True
        assert s.is_section_done("1") is False      # markers now authoritative


# ===========================================================================
# 7. crawl / dedup
# ===========================================================================

class SameUrlSearch(FakeSearchClient):
    """Every query returns the SAME page (worst-case duplication)."""

    def search(self, query, max_results=10):
        self.search_calls += 1
        return [SimpleNamespace(url="https://example.com/dup?utm_source=x",
                                title="同一ページ", snippet="s"),
                SimpleNamespace(url="https://example.com/dup/",
                                title="同一ページ", snippet="s")]


class DocLinkSearch(FakeSearchClient):
    """Result pages link to the SAME PDF under different spellings."""

    def __init__(self):
        super().__init__()
        self.fetched = []

    def get_page_content(self, url):
        self.fetched.append(url)
        page = super().get_page_content(url)
        if "report.pdf" in url:
            page.title = "業界レポート"
            page.text_content = "PDF本文: 炭素繊維の需要は年率8%で成長。" * 20
            page.links = []
        else:
            host = "example.com" if url.endswith("/0") else "EXAMPLE.com"
            page.links = [{"url": f"https://{host}/report.pdf",
                           "text": "業界レポート(PDF)"}]
        return page


def DocLinkSearch_pair():
    return FakeLLM(), DocLinkSearch()


class TestCrawlDedup:

    def test_fast_parallel_zero_pages_returns_empty(self):
        from deep_research_tool.research.fast_crawler import (
            EvaluationMode, FastCrawler)
        crawler = FastCrawler(search_client=MagicMock(), llm_client=MagicMock(),
                              evaluation_mode=EvaluationMode.PARALLEL)
        assert crawler._parallel_evaluate([], "ctx", {"topic": ""}) == []
        # whole crawl with searches returning nothing -> empty, no exception
        crawler.search.search.return_value = []
        result = crawler.crawl_and_evaluate(["q1", "q2"], "1. 章")
        assert result.pages == [] and result.pages_fetched == 0

    def test_standard_mode_fetches_and_extracts_a_url_once(self, tmp_path):
        llm, search = FakeLLM(), SameUrlSearch()
        researcher = make_researcher(tmp_path, llm, search, max_iterations=1)
        researcher.conduct_research(query="炭素繊維の市場調査",
                                    requirements="目次は2章構成でよい")
        # 2 sections x 2 queries x 2 results all point at one page:
        # fetched once per section (page cache: once per RUN), extracted
        # once per section, counted as ONE independent source
        assert search.fetch_calls == 1
        extractions = sum("processed_content" in p for p in llm.calls)
        assert extractions == 2
        parts_sources = researcher.session.section_contents["1"]["sources"]
        assert len(parts_sources) == 1
        assert researcher._independent_source_count(
            [SimpleNamespace(source_url="https://example.com/dup/"),
             SimpleNamespace(source_url="https://EXAMPLE.com/dup?utm_source=a")]
        ) == 1
        # evidence locker is not inflated either
        urls = [e.url for e in researcher.evidence_locker.get_all_evidence()]
        assert len(urls) <= 2          # at most one per section

    def test_document_links_are_deduped_and_cached(self, tmp_path):
        """The same PDF linked from several result pages (utm / slash
        variants) is fetched once per run and cited once per section."""
        llm, search = DocLinkSearch_pair()
        researcher = make_researcher(tmp_path, llm, search, max_iterations=1)
        researcher.conduct_research(query="炭素繊維の市場調査",
                                    requirements="目次は2章構成でよい")
        pdf_fetches = [u for u in search.fetched if "report.pdf" in u.lower()]
        assert len(pdf_fetches) == 1
        for sid in ("1", "2"):
            cited = [u for u in researcher.session.section_contents[sid]["sources"]
                     if "report.pdf" in u.lower()]
            assert len(cited) <= 1
        pdf_evidence = [e for e in researcher.evidence_locker.get_all_evidence()
                        if "report.pdf" in e.url.lower()]
        assert len(pdf_evidence) <= 2            # at most one per section

    def test_expansion_dedupes_and_uses_the_page_cache(self, tmp_path):
        llm, search = FakeLLM(), FakeSearchClient()
        researcher = make_researcher(tmp_path, llm, search, max_iterations=1)
        researcher.conduct_research(query="炭素繊維の市場調査",
                                    requirements="目次は2章構成でよい")
        # expansion queries all return spellings of ONE new page
        search.search = lambda query, max_results=10: [
            SimpleNamespace(url="https://example.com/expand?utm_source=x",
                            title="追加", snippet="s"),
            SimpleNamespace(url="https://EXAMPLE.com/expand/", title="追加",
                            snippet="s")]
        researcher._generate_expansion_queries = lambda *a, **k: ["q1", "q2"]
        before_fetch, before_llm = search.fetch_calls, len(llm.calls)
        researcher.expand_section_content(["1"], additional_iterations=2,
                                          focus_on_gaps=False)
        assert search.fetch_calls - before_fetch == 1       # one real fetch
        extractions = sum("processed_content" in c for c in llm.calls[before_llm:])
        assert extractions <= 1

    def test_url_normalization(self):
        n = Researcher._normalize_url
        assert n("https://Example.com/a/?utm_source=x") == n("https://example.com/a")
        assert n("https://example.com/a#frag") == n("https://example.com/a")
        assert n("https://example.com/a?b=1&a=2") == n("https://example.com/a?a=2&b=1")
        assert n("https://example.com/a?x=1") != n("https://example.com/a?x=2")


# ===========================================================================
# 6. per-job warnings
# ===========================================================================

class TestWarningIsolation:

    def test_two_jobs_do_not_share_warnings(self):
        ResearchWarnings.reset()
        seen = {}

        def job(name, n):
            collector = ResearchWarnings.bind()
            for i in range(n):
                ResearchWarnings.get_instance().add("LOW", name, f"{name}-{i}")
            # parallel workers inherit the binding
            with ContextThreadPoolExecutor(max_workers=2) as ex:
                ex.submit(lambda: ResearchWarnings.get_instance().add(
                    "LOW", name, f"{name}-worker")).result()
            seen[name] = [w["message"] for w in collector.to_dict_list()]
            ResearchWarnings.unbind()

        threads = [threading.Thread(target=job, args=("A", 2)),
                   threading.Thread(target=job, args=("B", 3))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert seen["A"] == ["A-0", "A-1", "A-worker"]
        assert seen["B"] == ["B-0", "B-1", "B-2", "B-worker"]
        # the process-wide singleton saw none of it
        assert ResearchWarnings.get_instance().count() == 0


# ===========================================================================
# 3/8. server-side job state
# ===========================================================================

class TestJobManagerState:

    def test_job_ids_are_unique_and_persist_across_restart(self, tmp_path):
        m1 = JobManager(output_dir=str(tmp_path))
        m1._launch = lambda job, params: None          # no real run
        a = m1.start({"query": "A", "openai_api_key": "sk-secret"})
        b = m1.start({"query": "B"})
        assert a.job_id != b.job_id
        assert not a.job_id.startswith("job-1")        # never job-1 again
        ledger = json.loads((tmp_path / "jobs_ledger.json").read_text("utf-8"))
        assert set(ledger["jobs"]) == {a.job_id, b.job_id}
        assert "sk-secret" not in json.dumps(ledger)   # secrets never persisted
        # a NEW manager (restart) still lists both runs
        m2 = JobManager(output_dir=str(tmp_path))
        ids = [r["job_id"] for r in m2.history()]
        assert set(ids) == {a.job_id, b.job_id}
        c = m2.start({"query": "C"}) if False else None   # (no launch needed)
        assert JobManager.new_job_id() != JobManager.new_job_id()

    def test_queue_positions_and_cancel_while_queued(self, tmp_path):
        m = JobManager(output_dir=str(tmp_path))
        launched = []
        m._launch = lambda job, params: launched.append(job.job_id)
        for i in range(m.MAX_CONCURRENT):
            j = ResearchJob(f"run-{i}", f"q{i}")         # state running
            m.jobs[j.job_id] = j
        q1 = m.start({"query": "queued-1"})
        q2 = m.start({"query": "queued-2"})
        assert q1.state == "queued" and q2.state == "queued"
        m.list_jobs()
        assert (q1.queue_position, q2.queue_position) == (1, 2)
        assert launched == []                            # nothing started
        assert m.cancel(q1.job_id) is True
        assert q1.state == "cancelled" and q1.finished_at is not None
        m.list_jobs()
        assert q2.queue_position == 1
        # capacity frees up -> the queued job is launched with its params
        for j in list(m.jobs.values()):
            if j.job_id.startswith("run-"):
                j.finish("completed")
        m._pump_queue()
        assert launched == [q2.job_id]
        assert q2.state == "running"

    def test_elapsed_time_is_frozen_after_finish(self):
        job = ResearchJob("j", "q")
        job.started_at = time.time() - 60
        job.finish("completed")
        e1 = job.elapsed_seconds()
        time.sleep(0.05)
        assert job.elapsed_seconds() == e1
        assert 59 <= e1 <= 61

    def test_log_sequence_advances_beyond_window(self):
        job = ResearchJob("j", "q")
        for i in range(70):
            job.update(f"m{i}", i)
        d = job.to_dict()
        assert len(d["log"]) == 50
        assert d["log_seq"] == 70
        assert d["log"][-1]["seq"] == 70

    def test_plan_review_explicit_mode_waits_and_cancel_releases(self):
        job = ResearchJob("j", "q")
        out = {}

        def waiter():
            out["resp"] = job.begin_plan_review({"title": "p"}, timeout=0)
        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.2)
        assert t.is_alive()                       # no auto-start
        assert job.state == "plan_review"
        assert job.cancel_run() is True           # releases the wait
        t.join(timeout=2)
        assert not t.is_alive()
        assert out["resp"]["action"] == "cancel"  # not mistaken for approve

    def test_plan_review_approve_is_distinct_from_cancel(self):
        job = ResearchJob("j", "q")
        out = {}
        t = threading.Thread(target=lambda: out.update(
            resp=job.begin_plan_review({"title": "p"}, timeout=0)))
        t.start()
        time.sleep(0.1)
        assert job.respond_plan_review("approve")
        t.join(timeout=2)
        assert out["resp"]["action"] == "approve"

    def test_scrub_secrets_deep(self):
        params = {"query": "q", "openai_api_key": "sk", "proxy_password": "p",
                  "stage_llm": {"writing": {"provider": "local",
                                            "api_key": "tok"}}}
        clean = scrub_secrets(params)
        assert "openai_api_key" not in clean and "proxy_password" not in clean
        assert clean["stage_llm"]["writing"] == {"provider": "local"}
        assert params["openai_api_key"] == "sk"       # input untouched


# ===========================================================================
# 4. numeric config mapping (0 vs empty vs invalid)
# ===========================================================================

class TestNumericMapping:

    def test_zero_is_kept_and_invalid_names_the_field(self):
        kwargs = build_config_kwargs({"query": "q", "ai_crawl_site_depth": 0,
                                      "gap_fill_rounds": 0,
                                      "plan_review_timeout": 0})
        assert kwargs["ai_crawl_site_depth"] == 0
        assert kwargs["max_gap_fill_rounds"] == 0
        assert kwargs["plan_review_timeout"] == 0
        # empty string -> omitted (server default), never coerced to 2
        kwargs = build_config_kwargs({"query": "q", "ai_crawl_site_depth": ""})
        assert "ai_crawl_site_depth" not in kwargs
        with pytest.raises(FieldError) as exc:
            build_config_kwargs({"query": "q", "ai_crawl_site_depth": "abc"})
        assert exc.value.field == "ai_crawl_site_depth"
        with pytest.raises(FieldError) as exc:
            build_config_kwargs({"query": "q", "max_pages_per_query": 0})
        assert exc.value.field == "max_pages_per_query"


# ===========================================================================
# 8. Fermi GUI save state (tkinter stubbed: module logic only)
# ===========================================================================

class TestFermiGuiSaveState:

    @pytest.fixture
    def fermi_module(self):
        fake_tk = types.ModuleType("tkinter")
        fake_tk.END = "end"
        fake_tk.DISABLED = "disabled"
        fake_tk.NORMAL = "normal"
        for name in ("ttk", "filedialog", "messagebox", "scrolledtext"):
            setattr(fake_tk, name, MagicMock())
        fake_tk.Tk = MagicMock
        fake_tk.StringVar = MagicMock
        fake_tk.Frame = MagicMock
        fake_tk.Text = MagicMock
        mods = {"tkinter": fake_tk, "tkinter.ttk": fake_tk.ttk,
                "tkinter.filedialog": fake_tk.filedialog,
                "tkinter.messagebox": fake_tk.messagebox,
                "tkinter.scrolledtext": fake_tk.scrolledtext}
        with patch.dict(sys.modules, mods):
            sys.modules.pop("deep_research_tool.fermi_gui", None)
            import importlib
            module = importlib.import_module("deep_research_tool.fermi_gui")
            yield module
        sys.modules.pop("deep_research_tool.fermi_gui", None)

    def test_rerun_and_failure_clear_previous_result(self, fermi_module):
        cls = [c for c in vars(fermi_module).values()
               if isinstance(c, type) and c.__module__ == fermi_module.__name__
               and hasattr(c, "_on_run")][0]
        gui = cls.__new__(cls)
        gui.question_entry = MagicMock(); gui.question_entry.get.return_value = "B?"
        gui.run_button = MagicMock(); gui.status_var = MagicMock()
        gui.result_text = MagicMock()
        gui.save_docx_button = MagicMock(); gui.save_pdf_button = MagicMock()
        gui.last_estimate = "OLD RESULT FOR QUESTION A"
        with patch.object(fermi_module.threading, "Thread") as thread:
            thread.return_value = MagicMock()
            gui._on_run()
        # re-run clears the previous estimate and disables saving
        assert gui.last_estimate is None
        gui.save_docx_button.config.assert_any_call(state="disabled")
        gui.save_pdf_button.config.assert_any_call(state="disabled")
        # a failure never leaves the old estimate savable either
        gui.last_estimate = "STALE"
        gui.root = MagicMock()
        gui._show_error("boom")
        assert gui.last_estimate is None
