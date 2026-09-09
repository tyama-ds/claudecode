"""
Stage 3 — reuse / time-saving features.

- FetchCache: page + extraction reuse, freshness bypass, atomic files
- regenerate_from_session: no search / no LLM, prefers the FINAL verified
  body over the stale session text, summary mode
- DeepResearchTool.run(resume_from=...) reuses completed sections
- Web UI: precheck / upload / history / resume / regenerate endpoints,
  shared cache dir per server, measured timings in results
"""

import io
import json
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from deep_research_tool.config import CrawlMode, ResearchSourceMode
from deep_research_tool.research.fetch_cache import (
    EXTRACTION_PROMPT_VERSION,
    FetchCache,
)
from deep_research_tool.research.researcher import Researcher, ResearchState
from deep_research_tool.tests.test_e2e_pipeline import FakeLLM, FakeSearchClient
from deep_research_tool.utils.timing import StageTimer
from deep_research_tool.webui.server import (
    JobManager,
    ResearchJob,
    WebUIHandler,
    precheck_documents,
)
from http.server import ThreadingHTTPServer


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


# ===========================================================================
# FetchCache
# ===========================================================================

class TestFetchCache:

    def test_page_hit_miss_and_freshness(self, tmp_path):
        cache = FetchCache(tmp_path / "c", max_age_hours=1)
        assert cache.get_page("https://a.example.com/x") is None
        cache.put_page("https://a.example.com/x", "本文", title="t")
        hit = cache.get_page("https://a.example.com/x")
        assert hit["text"] == "本文" and hit["content_hash"]
        # a refresh demand bypasses the cached page (and is counted)
        assert cache.get_page("https://a.example.com/x", refresh=True) is None
        st = cache.stats()
        assert st["page_hits"] == 1 and st["refreshed"] == 1
        # stale entries are re-fetched
        stale = FetchCache(tmp_path / "c", max_age_hours=0.0000001)
        time.sleep(0.01)
        assert stale.get_page("https://a.example.com/x") is None
        # no temp files left behind
        assert not list((tmp_path / "c" / "pages").glob("*.tmp"))

    def test_extraction_key_is_content_addressed(self, tmp_path):
        cache = FetchCache(tmp_path / "c")
        k1 = cache.extraction_key("text A", "1. 章", "q", "gpt-x")
        k2 = cache.extraction_key("text A", "1. 章", "q", "gpt-x")
        k3 = cache.extraction_key("text B", "1. 章", "q", "gpt-x")
        k4 = cache.extraction_key("text A", "2. 章", "q", "gpt-x")
        k5 = cache.extraction_key("text A", "1. 章", "q", "other-model")
        k6 = cache.extraction_key("text A", "1. 章", "q", "gpt-x", "extract-v9")
        assert k1 == k2
        assert len({k1, k3, k4, k5, k6}) == 5
        cache.put_extraction(k1, {"processed_content": "p", "relevance_score": 0.8})
        assert cache.get_extraction(k1)["extracted"]["processed_content"] == "p"
        assert cache.get_extraction(k3) is None

    def test_disabled_cache_is_inert(self, tmp_path):
        cache = FetchCache(tmp_path / "c", enabled=False)
        cache.put_page("u", "t")
        assert cache.get_page("u") is None
        assert not (tmp_path / "c").exists()

    def test_second_run_reuses_pages_and_extractions(self, tmp_path):
        cache = FetchCache(tmp_path / "cache")
        llm1, s1 = FakeLLM(), FakeSearchClient()
        r1 = make_researcher(tmp_path / "run1", llm1, s1, max_iterations=1,
                             fetch_cache=cache)
        r1.conduct_research(query="炭素繊維の市場調査",
                            requirements="目次は2章構成でよい")
        extractions_1 = sum("processed_content" in p for p in llm1.calls)
        assert s1.fetch_calls > 0 and extractions_1 > 0

        # identical second run: pages come from disk, extractions replay
        llm2, s2 = FakeLLM(), FakeSearchClient()
        r2 = make_researcher(tmp_path / "run2", llm2, s2, max_iterations=1,
                             fetch_cache=cache)
        r2.conduct_research(query="炭素繊維の市場調査",
                            requirements="目次は2章構成でよい")
        assert s2.fetch_calls == 0
        assert sum("processed_content" in p for p in llm2.calls) == 0
        st = cache.stats()
        assert st["page_hits"] >= 1 and st["extract_hits"] >= 1

    def test_freshness_demand_refetches(self, tmp_path):
        cache = FetchCache(tmp_path / "cache")
        # first run fills the cache
        r0 = make_researcher(tmp_path / "run0", FakeLLM(), FakeSearchClient(),
                             max_iterations=1, fetch_cache=cache)
        r0.conduct_research(query="炭素繊維の市場調査",
                            requirements="目次は2章構成でよい")
        # a run that DEMANDS fresh pages ignores them and re-fetches
        llm, s = FakeLLM(), FakeSearchClient()
        r = make_researcher(tmp_path / "run", llm, s, max_iterations=1,
                            fetch_cache=cache, refresh_fetched=True)
        r.conduct_research(query="炭素繊維の市場調査",
                           requirements="目次は2章構成でよい")
        assert s.fetch_calls > 0            # cached pages were NOT reused
        assert cache.stats()["refreshed"] >= 1


# ===========================================================================
# regenerate from a saved research
# ===========================================================================

class TestRegenerate:

    def _finished_session(self, tmp_path):
        llm, s = FakeLLM(), FakeSearchClient()
        r = make_researcher(tmp_path, llm, s, max_iterations=1)
        session = r.conduct_research(query="炭素繊維の市場調査",
                                     requirements="目次は2章構成でよい")
        return r, session, Path(session.saved_artifacts["session"])

    def test_regenerate_prefers_final_verified_body(self, tmp_path):
        from deep_research_tool.main import regenerate_from_session
        r, session, session_path = self._finished_session(tmp_path)
        # a frozen final body exists and DIFFERS from the session text
        final = {"session_id": session.session_id,
                 "chapters": {"1": "## 1. 市場規模の推移\n\n検証済みの最終本文である。",
                              "2": "## 2. 主要メーカーの動向\n\n検証済みの第2章である。"},
                 "status": {"verification": "performed", "quality": "passed"}}
        (tmp_path / f"final_body_{session.session_id}.json").write_text(
            json.dumps(final, ensure_ascii=False), encoding="utf-8")
        out = regenerate_from_session(session_path, output_format="markdown")
        assert out["source"] == "final_body"
        text = Path(out["report_path"]).read_text(encoding="utf-8")
        assert "検証済みの最終本文である" in text
        # the stale pre-finalization body is NOT what shipped
        assert session.section_contents["1"]["content"][:30] not in text
        # the original session file is untouched
        assert json.loads(session_path.read_text("utf-8"))["session_id"] == \
            session.session_id

    def test_regenerate_summary_mode_without_llm_or_search(self, tmp_path):
        from deep_research_tool.main import regenerate_from_session
        r, session, session_path = self._finished_session(tmp_path)
        with patch("deep_research_tool.search.get_search_client") as gs, \
                patch("deep_research_tool.api.get_client") as gc:
            out = regenerate_from_session(session_path, mode="summary",
                                          output_format="markdown")
            assert not gs.called and not gc.called
        assert out["mode"] == "summary"
        text = Path(out["report_path"]).read_text(encoding="utf-8")
        assert "章別要約" in text
        assert "市場規模の推移" in text and "主要メーカーの動向" in text


# ===========================================================================
# DeepResearchTool.run(resume_from=...)
# ===========================================================================

class TestResumeEntry:

    def test_run_resume_reuses_completed_sections(self, tmp_path):
        from deep_research_tool.tests.test_finalization_integration import (
            build_tool, make_llm)
        from deep_research_tool.utils.cancellation import RunCancelled

        # 1) a run cancelled after section 1 (real Researcher + fakes)
        llm, search = FakeLLM(), FakeSearchClient()
        holder = {}

        def cancel_check():
            r = holder.get("r")
            if r and r.session and "1" in r.session.completed_sections:
                raise RunCancelled("stop")
        r = make_researcher(tmp_path, llm, search, cancel_check=cancel_check)
        holder["r"] = r
        with pytest.raises(RunCancelled):
            r.conduct_research(query="炭素繊維の市場調査",
                               requirements="目次は2章構成でよい")
        session_path = r.session.saved_artifacts["session"]

        # 2) resume through the TOOL entry point with the real Researcher
        tool = build_tool(tmp_path, FakeLLM(), FakeSearchClient(),
                          enable_verification=False, auto_figures=False)
        tool.config.research.use_enhanced_synthesis = False
        tool.config.research.max_gap_fill_rounds = 0
        tool.config.research.max_queries_per_iteration = 2
        tool.config.research.max_pages_per_query = 2
        tool.config.research.content_filter_mode = \
            type(tool.config.research.content_filter_mode)("none")
        result = tool.run(query="炭素繊維の市場調査",
                          requirements="目次は2章構成でよい",
                          resume_from=session_path)
        assert result["resume_plan"] == {"reused": ["1"], "to_process": ["2"]}
        assert Path(result["report_path"]).exists()
        assert result["timings"]["stages"][0]["stage"] == "research"
        assert "api_calls" in result["timings"]["counters"]
        data = json.loads(Path(session_path).read_text("utf-8"))
        assert data["state"] == "completed"
        assert sorted(data["completed_sections"]) == ["1", "2"]


# ===========================================================================
# Web UI endpoints
# ===========================================================================

@pytest.fixture
def server(tmp_path):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WebUIHandler)
    httpd.job_manager = JobManager(output_dir=str(tmp_path))
    httpd.output_dir = str(tmp_path)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd
    httpd.shutdown()


def _post(server, path, payload):
    port = server.server_address[1]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(server, path):
    port = server.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as res:
        return res.status, json.loads(res.read())


class TestWebEndpoints:

    def test_precheck_reports_readability(self, server, tmp_path):
        good = tmp_path / "doc.txt"
        good.write_text("読める本文です。" * 10, encoding="utf-8")
        bad = tmp_path / "image.png"
        bad.write_bytes(b"\x89PNG")
        status, data = _post(server, "/api/precheck",
                             {"paths": [str(good), str(bad),
                                        str(tmp_path / "missing.pdf")]})
        assert status == 200
        by_name = {f["name"]: f for f in data["files"]}
        assert by_name["doc.txt"]["readable"] is True
        assert by_name["doc.txt"]["chars"] > 0
        assert by_name["image.png"]["readable"] is False
        assert "未対応" in by_name["image.png"]["error"]
        assert by_name["missing.pdf"]["readable"] is False

    def test_upload_stores_and_prechecks(self, server, tmp_path):
        port = server.server_address[1]
        boundary = "----drtboundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="files"; filename="memo.md"\r\n'
            "Content-Type: text/markdown\r\n\r\n"
            "# メモ\n\n手元の資料の本文。\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/upload", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST")
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read())
        assert len(data["files"]) == 1
        f = data["files"][0]
        assert f["name"] == "memo.md" and f["readable"] is True
        # stored under the SERVER output dir (not the browser's path)
        assert Path(f["path"]).is_file()
        assert str(tmp_path) in f["path"]

    def test_history_resume_and_regenerate_flow(self, server, tmp_path):
        manager = server.job_manager
        # a finished run with a saved session (as the researcher writes it)
        llm, s = FakeLLM(), FakeSearchClient()
        job_dir = tmp_path / "job-old"
        r = make_researcher(job_dir, llm, s, max_iterations=1)
        session = r.conduct_research(query="炭素繊維の市場調査",
                                     requirements="目次は2章構成でよい")
        job = ResearchJob("job-old", "炭素繊維の市場調査",
                          params={"query": "炭素繊維の市場調査",
                                  "openai_api_key": "sk-secret"})
        job.result = {"saved_artifacts": dict(session.saved_artifacts),
                      "report_path": None}
        job.finish("completed")
        manager.jobs[job.job_id] = job
        manager.record(job, output_dir=str(job_dir))

        status, hist = _get(server, "/api/history")
        assert status == 200
        rec = [h for h in hist["history"] if h["job_id"] == "job-old"][0]
        assert rec["params_summary"]["query"] == "炭素繊維の市場調査"
        assert "openai_api_key" not in json.dumps(rec)     # never echoed

        # regenerate: a NEW lightweight job that re-outputs without search
        launched = []
        manager._launch = lambda j, p: launched.append((j, p))
        status, data = _post(server, "/api/regenerate",
                             {"job_id": "job-old", "output_format": "markdown",
                              "mode": "summary"})
        assert status == 202
        j, p = launched[-1]
        assert p["regenerate"]["session_path"] == session.saved_artifacts["session"]
        assert p["regenerate"]["mode"] == "summary"
        assert p["output_dir"] == str(job_dir)

        # resume: params come from the ledger, resume_from points at the
        # saved session, secrets only from THIS request
        status, data = _post(server, "/api/resume",
                             {"job_id": "job-old", "openai_api_key": "sk-now"})
        assert status == 202
        j, p = launched[-1]
        assert p["resume_from"] == session.saved_artifacts["session"]
        assert p["output_dir"] == str(job_dir)
        assert p["openai_api_key"] == "sk-now"
        assert p["cache_dir"] == str(tmp_path / ".cache")   # shared cache

        # unknown job -> 404; job without a saved session -> 409
        assert _post(server, "/api/resume", {"job_id": "nope"})[0] == 404
        j2 = ResearchJob("job-nosession", "q")
        j2.finish("error")
        manager.jobs[j2.job_id] = j2
        manager.record(j2, output_dir=str(tmp_path / "job-nosession"))
        assert _post(server, "/api/resume", {"job_id": "job-nosession"})[0] == 409

    def test_regenerate_job_runs_offline(self, server, tmp_path):
        manager = server.job_manager
        llm, s = FakeLLM(), FakeSearchClient()
        job_dir = tmp_path / "job-src"
        r = make_researcher(job_dir, llm, s, max_iterations=1)
        session = r.conduct_research(query="炭素繊維の市場調査",
                                     requirements="目次は2章構成でよい")
        job = manager.start({
            "query": "再出力",
            "regenerate": {"session_path": session.saved_artifacts["session"],
                           "output_format": "markdown", "mode": "full"},
            "regenerate_output_dir": str(job_dir),
        })
        for _ in range(100):
            if job.is_terminal:
                break
            time.sleep(0.1)
        assert job.state == "completed", job.error
        assert Path(job.result["report_path"]).exists()
        assert job.result["regenerated_from"] == "session"
        assert job.result["status"]["verification"] == "skipped"


class TestTimer:

    def test_stage_timer_measures_only(self):
        t = StageTimer()
        with t.stage("research"):
            time.sleep(0.01)
        t.count("api_calls", 3)
        t.start("render")
        t.finish()                    # open stage closed as interrupted
        d = t.to_dict()
        assert d["stages"][0]["stage"] == "research"
        assert d["stages"][0]["seconds"] >= 0.01
        assert d["stages"][1]["interrupted"] is True
        assert d["counters"] == {"api_calls": 3}
        assert "eta" not in json.dumps(d) and "speedup" not in json.dumps(d)


# ===========================================================================
# Follow-ups from the review audit
# ===========================================================================

class TestLedgerAndStatusFollowups:

    def test_ledger_marks_orphaned_records_interrupted_on_load(self, tmp_path):
        """A record left 'running' by a dead server process can never finish:
        it is shown as interrupted (never as running) after a restart."""
        ledger = tmp_path / "jobs_ledger.json"
        ledger.write_text(json.dumps({"version": 1, "jobs": {
            "job-a": {"job_id": "job-a", "query": "q", "state": "running",
                      "started_at": 1.0, "params_summary": {}},
            "job-b": {"job_id": "job-b", "query": "q", "state": "completed",
                      "started_at": 2.0, "params_summary": {}},
        }}), encoding="utf-8")
        manager = JobManager(output_dir=str(tmp_path))
        by_id = {r["job_id"]: r for r in manager.history()}
        assert by_id["job-a"]["state"] == "interrupted"
        assert by_id["job-a"]["finished_at"] == 1.0
        assert "再起動" in by_id["job-a"]["error"]
        assert by_id["job-b"]["state"] == "completed"
        # persisted, and history() hands out copies (no shared mutation)
        on_disk = json.loads(ledger.read_text("utf-8"))["jobs"]
        assert on_disk["job-a"]["state"] == "interrupted"
        by_id["job-b"]["state"] = "mutated"
        assert manager.history()[0]["state"] != "mutated"

    def test_ledger_write_survives_non_json_values(self, tmp_path):
        manager = JobManager(output_dir=str(tmp_path))
        job = ResearchJob("job-x", "q", params={"query": "q"})
        job.result = {"report_path": Path("/tmp/x.md")}     # a Path, not str
        manager.record(job, output_dir=str(tmp_path))
        data = json.loads((tmp_path / "jobs_ledger.json").read_text("utf-8"))
        assert data["jobs"]["job-x"]["result"]["report_path"] == "/tmp/x.md"

    def test_job_status_reports_live_saved_artifacts(self, tmp_path):
        job = ResearchJob("job-live", "q", params={"query": "q"})
        assert job.to_dict()["saved_artifacts"] == {}
        current = {}
        job.artifacts_source = lambda: current
        current["session"] = str(tmp_path / "session_1.json")
        assert job.to_dict()["saved_artifacts"] == {"session": current["session"]}
        # once the job has a result, the result is authoritative
        job.result = {"saved_artifacts": {"report": "r.md"}}
        assert job.to_dict()["saved_artifacts"] == {"report": "r.md"}

    def test_regenerate_status_travels_with_the_final_body(self, tmp_path):
        """A final body frozen after a CANCELLED verification is re-output
        as 'verification: cancelled' — never relabelled verified/passed."""
        from deep_research_tool.main import regenerate_from_session
        llm, s = FakeLLM(), FakeSearchClient()
        r = make_researcher(tmp_path, llm, s, max_iterations=1)
        session = r.conduct_research(query="炭素繊維の市場調査",
                                     requirements="目次は2章構成でよい")
        sid = session.session_id
        (tmp_path / f"final_body_{sid}.json").write_text(json.dumps({
            "session_id": sid, "decision": "cancelled",
            "status": {"process": "completed", "verification": "cancelled",
                       "quality": "unverified"},
            "chapters": {"1": "# 1\n\n検証途中の本文", "2": "# 2\n\n本文2",
                         "要旨": "全体の要旨"},          # mixed keys
        }, ensure_ascii=False), encoding="utf-8")
        session_path = session.saved_artifacts["session"]
        fetches, llm_n = s.fetch_calls, len(llm.calls)
        out = regenerate_from_session(session_path, "markdown", "full")
        assert out["source"] == "final_body"
        assert out["status"]["verification"] == "cancelled"
        assert out["decision"] == "cancelled"
        # summary mode must not crash on mixed numeric / named chapter keys
        out2 = regenerate_from_session(session_path, "markdown", "summary")
        assert Path(out2["report_path"]).exists()
        assert s.fetch_calls == fetches and len(llm.calls) == llm_n   # offline

    def test_regenerate_job_copies_status_from_final_body(self, tmp_path):
        llm, s = FakeLLM(), FakeSearchClient()
        job_dir = tmp_path / "job-old"
        r = make_researcher(job_dir, llm, s, max_iterations=1)
        session = r.conduct_research(query="炭素繊維の市場調査",
                                     requirements="目次は2章構成でよい")
        sid = session.session_id
        (job_dir / f"final_body_{sid}.json").write_text(json.dumps({
            "session_id": sid, "decision": "timeout",
            "status": {"process": "completed", "verification": "timeout",
                       "quality": "unverified"},
            "chapters": {"1": "本文1", "2": "本文2"}}), encoding="utf-8")
        manager = JobManager(output_dir=str(tmp_path))
        job = manager.start({
            "query": "再出力",
            "regenerate": {"session_path": session.saved_artifacts["session"],
                           "output_format": "markdown", "mode": "full"},
            "regenerate_output_dir": str(job_dir),
        })
        for _ in range(100):
            if job.is_terminal:
                break
            time.sleep(0.1)
        assert job.state == "completed", job.error
        assert job.result["status"]["verification"] == "timeout"
        assert job.result["status"]["quality"] == "unverified"
        assert job.result["decision"] == "timeout"

    def test_regenerate_endpoint_validates_format_and_mode(self, server, tmp_path):
        llm, s = FakeLLM(), FakeSearchClient()
        r = make_researcher(tmp_path / "j", llm, s, max_iterations=1)
        session = r.conduct_research(query="炭素繊維の市場調査",
                                     requirements="目次は2章構成でよい")
        sp = session.saved_artifacts["session"]
        status, data = _post(server, "/api/regenerate",
                             {"session_path": sp, "output_format": "xlsx"})
        assert status == 400 and data["field"] == "output_format"
        status, data = _post(server, "/api/regenerate",
                             {"session_path": sp, "mode": "brief"})
        assert status == 400 and data["field"] == "mode"
        assert server.job_manager.list_jobs() == []      # nothing started
