"""v1.6.0 — stages that used to run one LLM call at a time now use the
worker pool: sections in the research loop, post-verification edits
(sections and sentence patches), judge batches sized to the workers,
the finalization round's page fetches. No network, no real LLM."""
import threading
import time
from types import SimpleNamespace

import pytest

from deep_research_tool.config import CrawlMode, ResearchSourceMode, create_config
from deep_research_tool.research.researcher import Researcher, ResearchState
from deep_research_tool.tests.test_e2e_pipeline import FakeLLM, FakeSearchClient


class RecordingSearch(FakeSearchClient):
    def __init__(self):
        super().__init__()
        self.queries = []
        self.threads = set()
        self._lock = threading.Lock()

    def search(self, query, max_results=10):
        with self._lock:
            self.queries.append(query)
            self.threads.add(threading.current_thread().name)
        return super().search(query, max_results)


def _researcher(tmp_path, workers, search=None, **kw):
    llm = FakeLLM()
    search = search or RecordingSearch()
    r = Researcher(
        llm_client=llm, search_client=search, output_dir=tmp_path,
        source_mode=ResearchSourceMode.WEB, crawl_mode=CrawlMode.STANDARD,
        use_enhanced_synthesis=False, filter_mode="none",
        min_iterations=1, max_iterations=1, section_workers=workers, **kw)
    return r, llm, search


class TestParallelSections:
    def test_sections_researched_concurrently_and_checkpointed(self, tmp_path):
        r, llm, search = _researcher(tmp_path, 2)
        session = r.conduct_research(query="炭素繊維の市場調査")
        assert session.state == ResearchState.COMPLETED
        assert sorted(session.completed_sections) == ["1", "2"]
        for sid in ("1", "2"):
            assert session.section_contents[sid]["content"]
        stages = [c["stage"] for c in session.checkpoints]
        assert stages.count("section_completed") == 2
        assert stages[-1] == "completed"
        assert (tmp_path / f"session_{session.session_id}.json").exists()

    def test_query_slicing_matches_the_sequential_loop(self, tmp_path):
        r1, _, s1 = _researcher(tmp_path / "seq", 1)
        r1.conduct_research(query="炭素繊維の市場調査")
        r2, _, s2 = _researcher(tmp_path / "par", 2)
        r2.conduct_research(query="炭素繊維の市場調査")
        # each section takes the next max_queries_per_iteration plan queries
        assert sorted(s1.queries) == sorted(s2.queries)

    def test_worker_count_is_bounded_and_falls_back_for_serial_clients(self, tmp_path):
        r, _, _ = _researcher(tmp_path, 4, parallel_max_workers=3)
        assert r._section_workers(10) == 3          # parallel_max_workers
        assert r._section_workers(2) == 2           # section count
        assert r._section_workers(1) == 1
        r.crawl_mode = CrawlMode.AI_CRAWL
        assert r._section_workers(10) == 1          # AI crawl stays serial
        r.crawl_mode = CrawlMode.STANDARD
        r.search.requires_serial_access = True
        assert r._section_workers(10) == 1          # Selenium-style client
        r0, _, _ = _researcher(tmp_path / "z", 1)
        assert r0._section_workers(10) == 1

    def test_status_line_names_the_running_sections(self, tmp_path):
        from deep_research_tool.main import DeepResearchTool
        from deep_research_tool.utils.progress import ProgressPlan
        r, _, _ = _researcher(tmp_path, 2)
        r.session = SimpleNamespace(completed_sections=["1"])
        r._sections_running = ["2", "3"]
        sec = SimpleNamespace(section="2", title="主要メーカー")
        r._set_activity(sec, 1, 3)
        assert r.current_activity["running"] == ["2", "3"]
        tool = DeepResearchTool.__new__(DeepResearchTool)
        tool.config = create_config(provider="openai", openai_api_key="t")
        tool.progress_plan = ProgressPlan.for_config(tool.config)
        tool.progress_plan.enter("research")
        tool.researcher = r
        text = tool.current_activity()["text"]
        assert "並列" in text and "2・3" in text and "1/3" in text

    def test_resume_reuses_completed_sections_in_parallel_mode(self, tmp_path):
        r, _, search = _researcher(tmp_path, 2)
        session = r.conduct_research(query="炭素繊維の市場調査")
        path = tmp_path / f"session_{session.session_id}.json"
        # pretend section 2 never finished
        session.completed_sections = ["1"]
        session.state = ResearchState.RESEARCHING
        session.save(path)
        r2, _, search2 = _researcher(tmp_path, 2)
        resumed = r2.resume_research(path)
        assert resumed.state == ResearchState.COMPLETED
        assert r2.resume_plan == {"reused": ["1"], "to_process": ["2"]}
        assert sorted(resumed.completed_sections) == ["1", "2"]

    def test_config_knob_and_validation(self):
        cfg = create_config(provider="openai", openai_api_key="t")
        assert cfg.research.research_section_workers == 2
        cfg = create_config(provider="openai", openai_api_key="t", research_section_workers=1)
        assert cfg.research.research_section_workers == 1
        for bad in (0, 9, True, "2"):
            with pytest.raises(ValueError):
                create_config(provider="openai", openai_api_key="t", research_section_workers=bad)


class TestParallelEdits:
    def _controller(self, workers, rewrite):
        from deep_research_tool.report.finalization import (FinalizationController,
                                                             LoopBudget, StructuredVerdict)
        return FinalizationController(
            verify_fn=lambda ch: StructuredVerdict(), rewrite_fn=rewrite,
            compress_fn=rewrite, hedge_fn=rewrite, budget=LoopBudget(),
            edit_workers=workers)

    def test_section_edits_run_in_parallel_and_apply_in_order(self):
        from deep_research_tool.report.finalization import (ISSUE_UNSUPPORTED,
                                                             StructuredVerdict, VerificationIssue)
        threads, lock = [], threading.Lock()

        def rewrite(sid, issues):
            with lock:
                threads.append(threading.current_thread().name)
            time.sleep(0.05)
            return f"edited {sid}"

        ctl = self._controller(3, rewrite)
        chapters = {"1": "a", "2": "b", "3": "c"}
        verdict = StructuredVerdict()
        for sid in chapters:
            verdict.issues.append(VerificationIssue(type=ISSUE_UNSUPPORTED, section_id=sid, claim="x"))
        t0 = time.time()
        changed = ctl._do_revision(chapters, verdict, ctl.rewrite_fn, (ISSUE_UNSUPPORTED,))
        assert changed and chapters == {"1": "edited 1", "2": "edited 2", "3": "edited 3"}
        assert len(set(threads)) >= 2                 # really concurrent
        assert time.time() - t0 < 0.14                # not 3 × 50 ms

    def test_sequential_default_and_failure_isolation(self):
        def rewrite(sid, issues):
            if sid == "2":
                raise RuntimeError("boom")
            return f"edited {sid}"
        from deep_research_tool.report.finalization import (ISSUE_UNSUPPORTED,
                                                             StructuredVerdict, VerificationIssue)
        ctl = self._controller(1, rewrite)
        assert ctl.edit_workers == 1
        chapters = {"1": "a", "2": "b"}
        verdict = StructuredVerdict()
        for sid in chapters:
            verdict.issues.append(VerificationIssue(type=ISSUE_UNSUPPORTED, section_id=sid, claim="x"))
        ctl._do_revision(chapters, verdict, ctl.rewrite_fn, (ISSUE_UNSUPPORTED,))
        assert chapters == {"1": "edited 1", "2": "b"}

    def test_cancellation_propagates_from_a_worker(self):
        from deep_research_tool.utils.cancellation import RunCancelled
        from deep_research_tool.report.finalization import (ISSUE_UNSUPPORTED,
                                                             StructuredVerdict, VerificationIssue)

        def rewrite(sid, issues):
            raise RunCancelled("verification")
        ctl = self._controller(2, rewrite)
        chapters = {"1": "a", "2": "b"}
        verdict = StructuredVerdict()
        for sid in chapters:
            verdict.issues.append(VerificationIssue(type=ISSUE_UNSUPPORTED, section_id=sid, claim="x"))
        with pytest.raises(RunCancelled):
            ctl._do_revision(chapters, verdict, ctl.rewrite_fn, (ISSUE_UNSUPPORTED,))

    def test_runner_edit_workers_and_order_preserving_map(self):
        from deep_research_tool.report.finalization_runner import FinalizationRunner
        runner = FinalizationRunner.__new__(FinalizationRunner)
        runner.edit_workers = 3
        out = runner._map_parallel(lambda x: (threading.current_thread().name, x * 2), [1, 2, 3, 4])
        assert [v for _t, v in out] == [2, 4, 6, 8]
        runner.edit_workers = 1
        assert runner._map_parallel(lambda x: x, [5]) == [5]


class TestJudgeBatching:
    def test_batches_shrink_to_fill_idle_workers(self):
        from deep_research_tool.verification.claim_verifier import auto_batch_size
        assert auto_batch_size(10, 10, 8) == 2       # 5 batches instead of 1
        assert auto_batch_size(30, 10, 8) == 4       # 8 batches instead of 3
        assert auto_batch_size(100, 10, 8) == 10     # already enough batches
        assert auto_batch_size(3, 10, 8) == 2
        assert auto_batch_size(1, 10, 8) == 10       # single claim: untouched
        assert auto_batch_size(10, 10, 1) == 10      # one worker: untouched
        assert auto_batch_size(10, 10, 8, explicit=True) == 10

    def test_explicit_batch_size_is_remembered(self):
        from deep_research_tool.verification.profiles import resolve_verification_settings
        assert resolve_verification_settings("balanced").batch_size_explicit is False
        assert resolve_verification_settings("balanced", {"batch_size": 4}).batch_size_explicit is True
        assert resolve_verification_settings("custom", {"batch_size": None}).batch_size_explicit is False


class TestFinalizationPrefetch:
    def _runner(self, search):
        from deep_research_tool.report.finalization import LoopBudget
        from deep_research_tool.report.finalization_runner import FinalizationRunner
        from deep_research_tool.verification.profiles import VerificationSettings
        runner = FinalizationRunner.__new__(FinalizationRunner)
        runner.search_client = search
        runner.budget = LoopBudget()
        runner.rc = None
        runner.verification_settings = VerificationSettings(max_workers=4)
        return runner

    def test_pages_are_fetched_concurrently_then_registered_in_order(self):
        threads, lock = set(), threading.Lock()

        class Search:
            def get_page_content(self, url):
                with lock:
                    threads.add(threading.current_thread().name)
                time.sleep(0.03)
                if url.endswith("/bad"):
                    raise RuntimeError("fetch failed")
                return SimpleNamespace(url=url, title="t", text_content="x" * 300, metadata={})

        runner = self._runner(Search())
        runner.budget.seen_urls.add("https://e.com/seen")
        urls = ["https://e.com/1", "https://e.com/2", "https://e.com/bad",
                "https://e.com/1", "https://e.com/seen", ""]
        pages = runner._prefetch_pages(urls)
        assert set(pages) == {"https://e.com/1", "https://e.com/2", "https://e.com/bad"}
        assert pages["https://e.com/bad"] is False
        assert pages["https://e.com/1"].text_content.startswith("x")
        assert len(threads) >= 2

    def test_serial_clients_fetch_one_at_a_time(self):
        class Serial:
            requires_serial_access = True

            def __init__(self):
                self.threads = set()

            def get_page_content(self, url):
                self.threads.add(threading.current_thread().name)
                return SimpleNamespace(url=url, title="t", text_content="y" * 300, metadata={})

        s = Serial()
        runner = self._runner(s)
        pages = runner._prefetch_pages(["https://e.com/a", "https://e.com/b"])
        assert len(pages) == 2 and len(s.threads) == 1


class TestDelegatedCapDisplay:
    """The 'LLM 同時実行 設定' row used to show the main provider's limit
    (8) while a delegated local client capped verification at 1."""

    def _tool(self, local_cap=1):
        from unittest.mock import Mock
        from deep_research_tool.api.delegation import DelegatedLLMClient
        from deep_research_tool.main import DeepResearchTool
        tool = DeepResearchTool.__new__(DeepResearchTool)
        tool.config = create_config(provider="openai", openai_api_key="t")
        tool.llm_client = Mock(spec=["generate"])          # cloud: no max_concurrency
        local = SimpleNamespace(max_concurrency=local_cap, generate=lambda *a, **k: None)
        tool.role_llm_clients = {"verification": DelegatedLLMClient(
            local, fallback=tool.llm_client, role_label="検証", setting_label="ロールルーティング")}
        tool.stage_llm_clients = {}
        return tool

    def test_snapshot_reports_the_binding_caps(self):
        snap = self._tool(1).concurrency_snapshot()
        c = snap["configured"]
        assert c["llm"] == 8 and c["verification_llm"] == 1 and c["revision_llm"] == 8
        assert c["delegated"] == {"role:verification": 1}
        snap = self._tool(None).concurrency_snapshot()      # explicit "no cap"
        assert snap["configured"]["verification_llm"] == 8

    def test_tripped_delegate_reports_the_main_cap(self):
        tool = self._tool(1)
        tool.role_llm_clients["verification"]._state.tripped = True
        assert tool.concurrency_snapshot()["configured"]["verification_llm"] == 8

    def test_warnings_name_the_local_cap(self):
        from deep_research_tool.main import DeepResearchTool
        from deep_research_tool.utils.helpers import ResearchWarnings
        tool = DeepResearchTool.__new__(DeepResearchTool)
        col = ResearchWarnings.bind()
        tool._warn_if_llm_serialized({"provider": "openai", "configured": {"llm": 8, "verification_llm": 1},
                                      "activity": {"llm": {"peak": 4, "total": 40}, "verify": {"total": 12}}})
        msgs = [w["message"] for w in col.to_dict_list()]
        assert any("委譲先ローカルLLM" in m for m in msgs)
        col = ResearchWarnings.bind()
        tool._warn_if_llm_serialized({"provider": "local", "configured": {"llm": 1},
                                      "activity": {"llm": {"peak": 1, "total": 25}}})
        msgs = [w["message"] for w in col.to_dict_list()]
        assert any("上限が 1（既定）" in m for m in msgs)
        col = ResearchWarnings.bind()
        tool._warn_if_llm_serialized({"provider": "openai", "configured": {"llm": 8, "verification_llm": 8},
                                      "activity": {"llm": {"peak": 6, "total": 40}, "verify": {"total": 12}}})
        assert col.to_dict_list() == []
        ResearchWarnings.unbind()
