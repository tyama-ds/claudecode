"""v1.6.0 — a run that dies after the research loop resumes from the
first stage that did not finish: stage state file, persisted
verification judgements, the edited-body draft, retries on transient
network errors, and the Web UI error path. No network, no real LLM."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from deep_research_tool.tests.test_concurrency_visibility import _post, web  # noqa: F401
from deep_research_tool.tests.test_finalization_integration import (
    CLAIM_SENTENCE, SUPPORTED_EVIDENCE, FakeResearcher, FakeSearch, build_tool,
    long_body, make_llm, make_locker, make_plan, make_session)
from deep_research_tool.utils.run_state import STAGE_ORDER, RunState
from deep_research_tool.verification.runtime import VerificationCache


class TestRunState:
    def test_mark_done_and_later_stages_dropped(self, tmp_path):
        s = RunState(tmp_path / "run_state_x.json")
        s.mark("research", session="a.json")
        s.mark("figures", figures="f.json")
        s.mark("verification", final_body="fb.json")
        assert s.completed_stages() == ["research", "figures", "verification"]
        s2 = RunState.load(tmp_path / "run_state_x.json")
        assert s2.done("verification") and s2.artifact("figures") == "f.json"
        s2.mark("figures")           # rerunning figures invalidates what followed
        assert s2.done("research") and s2.done("figures") and not s2.done("verification")
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        assert RunState.load(tmp_path / "broken.json").completed_stages() == []
        assert STAGE_ORDER[0] == "research" and STAGE_ORDER[-1] == "rendering"
        assert RunState(None).to_dict()["completed"] == []


class TestVerificationCachePersistence:
    def test_roundtrip_and_dirty_flag(self, tmp_path):
        c = VerificationCache(enabled=True)
        assert not c.dirty and bool(c)              # an empty cache stays truthy
        c.put("k1", {"status": "supported", "ids": ["E1"]})
        c.put("k2", {1: {"answered": True}})
        assert c.dirty and c.save(tmp_path / "vc.json") and not c.dirty
        c2 = VerificationCache(enabled=True)
        assert c2.load(tmp_path / "vc.json") == 2 and c2.loaded == 2 and c2.size == 2
        assert c2.get("k1") == {"status": "supported", "ids": ["E1"]}
        assert c2.get("k2") == {"1": {"answered": True}}   # JSON keys; callers re-int them
        assert VerificationCache(enabled=False).save(tmp_path / "x.json") is False
        assert VerificationCache(enabled=True).load(tmp_path / "missing.json") == 0


class _ResumableResearcher(FakeResearcher):
    """The fake researcher with the two things a resume needs: loading
    the saved session and the atomic checkpoint the stages call."""

    def resume_research(self, session_path, additional_iterations=2):
        from deep_research_tool.research.researcher import ResearchSession
        self.session = ResearchSession.load(Path(session_path))
        self.resume_plan = {"reused": sorted(self.session.section_contents), "to_process": []}
        return self.session

    def _checkpoint(self, stage, reason=""):
        path = Path(self.locker.output_dir).parent / f"session_{self.session.session_id}.json"
        self.session.stage = stage
        self.session.saved_artifacts = {"session": str(path)}
        self.session.save(path)
        return dict(self.session.saved_artifacts)


def _first_run(tmp, llm=None):
    llm = llm or make_llm()
    plan = make_plan([("1", "市場シェア", "A社のシェアを明らかにする")])
    session = make_session(plan, {"1": {"title": "市場シェア",
                                        "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                                        "evidence": SUPPORTED_EVIDENCE}})
    locker = make_locker(tmp, SUPPORTED_EVIDENCE)
    tool = build_tool(tmp, llm, FakeSearch())
    fake = _ResumableResearcher(session, locker)
    with patch("deep_research_tool.main.Researcher", new=lambda **kw: fake):
        result = tool.run(query="市場調査")
    return tool, fake, result, llm


def _resume(tmp, fake, session_id, llm=None):
    llm = llm or make_llm()
    tool = build_tool(tmp, llm, FakeSearch())
    fake2 = _ResumableResearcher(fake.session, fake.locker)
    with patch("deep_research_tool.main.Researcher", new=lambda **kw: fake2):
        result = tool.run(query="市場調査", resume_from=str(tmp / f"session_{session_id}.json"))
    return tool, result, llm


class TestStageCheckpoints:
    def test_run_writes_state_cache_and_final_body(self, tmp_path):
        tool, fake, result, llm = _first_run(tmp_path)
        sid = result["session_id"]
        state = json.loads((tmp_path / f"run_state_{sid}.json").read_text(encoding="utf-8"))
        assert [s for s in STAGE_ORDER if s in state["stages"]] == \
            ["research", "evidence_review", "verification", "rendering"]
        assert Path(state["artifacts"]["final_body"]).exists()
        assert Path(state["artifacts"]["report"]).exists()
        assert (tmp_path / f"verification_cache_{sid}.json").exists()
        assert not (tmp_path / f"final_body_draft_{sid}.json").exists()   # frozen body replaced it
        assert (tmp_path / f"session_{sid}.json").exists()                 # checkpoint after the review
        assert result["resume_plan"] is None

    def test_resume_skips_completed_verification(self, tmp_path):
        tool, fake, result, llm = _first_run(tmp_path)
        sid = result["session_id"]
        tool2, res2, llm2 = _resume(tmp_path, fake, sid)
        skipped = res2["resume_plan"]["stages_skipped"]
        assert "evidence_review" in skipped and "verification" in skipped
        assert not llm2.prompts("extract") and not llm2.prompts("judge")   # no verification call
        assert res2["status"]["verification"] == "performed"
        assert res2["run_status"] == "completed"
        assert res2["resume_plan"]["reused"] == ["1"]        # the researcher's breakdown survives
        report = Path(res2["report_path"]).read_text(encoding="utf-8")
        assert CLAIM_SENTENCE in report
        assert tool2.finalization_outcome.get("restored") is True

    def test_real_researcher_reports_every_section_reused_for_a_finished_loop(self, tmp_path):
        from deep_research_tool.research.researcher import Researcher
        from deep_research_tool.tests.test_e2e_pipeline import FakeLLM, FakeSearchClient
        plan = make_plan([("1", "市場シェア", "a"), ("2", "価格動向", "b")])
        session = make_session(plan, {"1": {"title": "市場シェア", "content": long_body("1", "市場シェア", CLAIM_SENTENCE)},
                                      "2": {"title": "価格動向", "content": long_body("2", "価格動向", CLAIM_SENTENCE)}})
        path = tmp_path / "session_itest.json"
        session.save(path)
        researcher = Researcher(FakeLLM(), FakeSearchClient(), output_dir=tmp_path)
        resumed = researcher.resume_research(path)
        assert resumed.session_id == "itest"
        assert researcher.resume_plan == {"reused": ["1", "2"], "to_process": []}
        assert resumed.section_contents["1"]["content"] == session.section_contents["1"]["content"]

    def test_resume_replays_judgements_when_verification_did_not_finish(self, tmp_path):
        tool, fake, result, llm = _first_run(tmp_path)
        sid = result["session_id"]
        # simulate a crash during verification: the stage never got its
        # mark and no final body exists, but the judgements were saved
        st = RunState.load(tmp_path / f"run_state_{sid}.json")
        st.mark("evidence_review")
        (tmp_path / f"final_body_{sid}.json").unlink()
        tool2, res2, llm2 = _resume(tmp_path, fake, sid)
        assert res2["resume_plan"]["verification_cache_entries"] >= 1
        assert not llm2.prompts("extract") and not llm2.prompts("judge")   # replayed from the cache
        assert "verification" not in res2["resume_plan"]["stages_skipped"]
        assert res2["status"]["verification"] == "performed"
        assert tool2.finalization_outcome["decision"] == "accept"
        assert (tmp_path / f"final_body_{sid}.json").exists()

    def test_resume_starts_from_the_edited_draft(self, tmp_path):
        tool, fake, result, llm = _first_run(tmp_path)
        sid = result["session_id"]
        st = RunState.load(tmp_path / f"run_state_{sid}.json")
        st.mark("evidence_review")
        (tmp_path / f"final_body_{sid}.json").unlink()
        edited = long_body("1", "市場シェア", CLAIM_SENTENCE) + "\n\n再開前に修正された段落。"
        (tmp_path / f"final_body_draft_{sid}.json").write_text(json.dumps({
            "session_id": sid, "chapters": {"1": edited}}, ensure_ascii=False), encoding="utf-8")
        tool2, res2, llm2 = _resume(tmp_path, fake, sid)
        assert res2["resume_plan"]["resumed_from_draft"] is True
        assert "再開前に修正された段落" in tool2.finalization_outcome["chapters"]["1"]
        assert "再開前に修正された段落" in Path(res2["report_path"]).read_text(encoding="utf-8")
        assert not (tmp_path / f"final_body_draft_{sid}.json").exists()


class TestFigureCollectionRoundtrip:
    def test_from_dict_rebuilds_figures_tables_and_charts(self, tmp_path):
        from deep_research_tool.report.figure_table_generator import (
            Figure, FigureTableCollection, FigureType, TableData)
        coll = FigureTableCollection(
            charts=[Figure(figure_id="c1", figure_type=FigureType.CHART, title="t", caption="c",
                           section_id="1", image_path=tmp_path / "c1.png",
                           chart_data={"series": {"a": [1, 2]}, "labels": ["x", "y"]})],
            tables=[TableData(table_id="t1", title="表", caption="cap", headers=["a", "b"],
                              rows=[["1", "2"]], section_id="2", unit="億円")])
        data = json.loads(json.dumps(coll.to_dict(), ensure_ascii=False))
        back = FigureTableCollection.from_dict(data)
        assert back.charts[0].figure_type == FigureType.CHART
        assert back.charts[0].image_path == tmp_path / "c1.png"
        assert back.charts[0].chart_data["series"] == {"a": [1, 2]}
        assert back.tables[0].headers == ["a", "b"] and back.tables[0].unit == "億円"
        assert back.get_charts_for_section("1") and back.get_tables_for_section("2")


class TestTransientRetry:
    def _obj(self, attempts, cancel=None):
        return SimpleNamespace(retry_attempts=attempts, retry_backoff_seconds=0.0,
                               cancel_check=cancel or (lambda: None))

    def test_retries_connection_errors_then_succeeds(self):
        from deep_research_tool.api.base import BaseLLMClient
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("reset by peer")
            return "ok"
        assert BaseLLMClient._call_with_retry(self._obj(3), fn) == "ok"
        assert len(calls) == 3

    def test_non_transient_errors_propagate_and_attempts_are_bounded(self):
        from deep_research_tool.api.base import BaseLLMClient

        def bad():
            raise ValueError("bad request")
        with pytest.raises(ValueError):
            BaseLLMClient._call_with_retry(self._obj(3), bad)
        calls = []

        def always():
            calls.append(1)
            raise TimeoutError("t")
        with pytest.raises(TimeoutError):
            BaseLLMClient._call_with_retry(self._obj(1), always)
        assert len(calls) == 2
        with pytest.raises(TimeoutError):
            BaseLLMClient._call_with_retry(self._obj(0), always)

    def test_cancel_between_attempts_and_sdk_error_names(self):
        from deep_research_tool.api.base import BaseLLMClient
        from deep_research_tool.utils.cancellation import RunCancelled

        def cancel():
            raise RunCancelled("stop")

        def boom():
            raise ConnectionError("c")
        with pytest.raises(RunCancelled):
            BaseLLMClient._call_with_retry(self._obj(2, cancel), boom)

        class APIConnectionError(Exception):          # the SDK's class, by name
            pass
        assert BaseLLMClient.is_transient_error(APIConnectionError("x"))
        assert BaseLLMClient.is_transient_error(TimeoutError("x"))
        assert not BaseLLMClient.is_transient_error(ValueError("x"))

    def test_config_wires_retry_onto_clients_and_delegates(self):
        from deep_research_tool.api.delegation import DelegatedLLMClient
        from deep_research_tool.config import create_config
        from deep_research_tool.main import DeepResearchTool
        cfg = create_config(provider="openai", openai_api_key="t",
                            llm_retry_attempts=5, llm_retry_backoff_seconds=0.5)
        assert cfg.api.llm_retry_attempts == 5 and cfg.api.llm_retry_backoff_seconds == 0.5
        tool = DeepResearchTool.__new__(DeepResearchTool)
        tool.config = cfg
        client = SimpleNamespace()
        tool._wire_retry(client)
        assert client.retry_attempts == 5 and client.retry_backoff_seconds == 0.5
        local = SimpleNamespace(generate=lambda *a, **k: None)
        d = DelegatedLLMClient(local, fallback=None, role_label="検証", setting_label="x")
        d.retry_attempts = 4
        assert local.retry_attempts == 4            # forwarded to the local client
        # out-of-range values are rejected, never silently clamped
        for bad in ({"llm_retry_attempts": 99}, {"llm_retry_attempts": True},
                    {"llm_retry_attempts": 2.5}, {"llm_retry_backoff_seconds": -1}):
            with pytest.raises(ValueError):
                create_config(provider="openai", openai_api_key="t", **bad)


class _DyingTool:
    """A tool whose run dies after saving a session checkpoint."""
    instances = []

    def __init__(self, config):
        self.config = config
        self.verification_progress = None
        self.partial_artifacts = {}
        sess = SimpleNamespace(session_id="s-dead")
        self.researcher = SimpleNamespace(get_session=lambda: sess,
                                          get_evidence_locker=lambda: None)
        _DyingTool.instances.append(self)

    def request_cancel(self):
        pass

    def current_saved_artifacts(self):
        return {"session": self.session_path}

    def concurrency_snapshot(self):
        return {}

    def llm_routing_snapshot(self):
        return {}

    def progress_snapshot(self):
        return None

    def current_plan(self):
        return None

    def run(self, **kw):
        out = Path(self.config.report.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.session_path = str(out / "session_s-dead.json")
        Path(self.session_path).write_text("{}", encoding="utf-8")
        if kw.get("resume_from"):
            return {"report_path": self.session_path, "session_id": "s-dead",
                    "status": {"process": "completed", "verification": "skipped",
                               "quality": "unverified"}, "run_status": "completed",
                    "saved_artifacts": {"session": self.session_path}, "resume_plan": {
                        "reused": ["1"], "to_process": [], "stages_skipped": ["verification"]}}
        raise RuntimeError("connection dropped mid-verification")


class TestErrorPathResume:
    def _wait(self, web, job_id):
        import time
        import urllib.request
        port = web.server_address[1]
        for _ in range(100):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/jobs", timeout=10) as r:
                jobs = json.loads(r.read())["jobs"]
            job = next((x for x in jobs if x["job_id"] == job_id), None)
            if job and job["state"] in ("completed", "error", "cancelled"):
                return job
            time.sleep(0.1)
        raise AssertionError("job did not finish")

    def test_errored_job_keeps_its_artifacts_and_can_be_resumed(self, web):
        with patch("deep_research_tool.main.DeepResearchTool", _DyingTool):
            status, d = _post(web, "/api/research", {"query": "q", "provider": "openai",
                                                      "openai_api_key": "sk-test",
                                                      "plan_review": False})
            assert status == 202, d
            job = self._wait(web, d["job_id"])
            assert job["state"] == "error"
            res = job["result"]
            assert res["run_status"] == "error" and res["saved_artifacts"]["session"]
            assert res["session_id"] == "s-dead" and "connection dropped" in res["error"]
            assert "再開" in job["message"]
            # the history row can resume it: the server resolves the session
            status, d2 = _post(web, "/api/resume", {"job_id": d["job_id"],
                                                     "openai_api_key": "sk-test"})
            assert status == 202, d2
            job2 = self._wait(web, d2["job_id"])
        assert job2["state"] == "completed"
        assert job2["result"]["resume_plan"]["stages_skipped"] == ["verification"]


class TestLocalClientRefusedConnection:
    """The local client retries only a connection that was never made
    (the server never saw the request); an ambiguous drop stays
    un-retried (test_local_client_streaming keeps that invariant)."""

    def test_refused_connection_is_retried_then_succeeds(self, monkeypatch):
        import time as _time
        import requests
        from deep_research_tool.tests.test_local_client_streaming import (
            FakeResponse, complete_sse, make_client)
        monkeypatch.setattr(_time, "sleep", lambda d: None)
        refused = requests.ConnectionError(
            "HTTPConnectionPool(host='127.0.0.1', port=11434): Max retries exceeded "
            "(Caused by NewConnectionError(': Failed to establish a new connection: "
            "[Errno 111] Connection refused'))")
        client, session = make_client(monkeypatch, [refused, FakeResponse(complete_sse())])
        assert client.generate("plan").content == "ok"
        assert len(session.calls) == 2

    def test_chain_detection(self):
        import requests
        from deep_research_tool.api.local_client import LocalLLMClient
        inner = ConnectionRefusedError(111, "Connection refused")
        outer = requests.ConnectionError("boom")
        outer.__cause__ = inner
        assert LocalLLMClient._connection_never_established(outer)
        assert not LocalLLMClient._connection_never_established(
            requests.ConnectionError("disconnect"))
