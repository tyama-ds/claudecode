"""Web UI API for the research engines: the catalog endpoint, request
validation (400 with the field before a job is queued) and one whole job
on the mock engine with the tool wired to the fake LLM / search."""
import json
import time
import urllib.error
import urllib.request

import pytest

import deep_research_tool.main as main_mod
from deep_research_tool.main import DeepResearchTool
from deep_research_tool.tests.test_concurrency_visibility import _post, web  # noqa: F401
from deep_research_tool.tests.test_e2e_pipeline import FakeLLM, FakeSearchClient
from deep_research_tool.webui.server import FieldError, build_config_kwargs, engine_request_error


@pytest.fixture(autouse=True)
def _no_engine_venvs(tmp_path, monkeypatch):
    """The catalog must not depend on this machine's engine virtualenvs."""
    from deep_research_tool.engines import venv as venv_mod
    monkeypatch.setenv("DRT_ENGINES_VENV_DIR", str(tmp_path / "no-venvs"))
    for engine_id in venv_mod.ENGINE_SPECS:
        monkeypatch.delenv(venv_mod.env_var_name(engine_id), raising=False)
    venv_mod.clear_check_cache()
    yield
    venv_mod.clear_check_cache()


def _get(server, path):
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


BASE = {"query": "炭素繊維市場", "provider": "openai", "openai_api_key": "sk-test",
        "plan_review": False, "run_mode": "research_only", "enable_verification": False}


class TestEngineCatalogApi:
    def test_get_and_post_catalog(self, web):
        status, d = _get(web, "/api/engines")
        assert status == 200 and d["default"] == "builtin"
        ids = [e["id"] for e in d["engines"]]
        assert ids == ["builtin", "gpt-researcher", "open-deep-research"]
        assert d["engines"][1]["health"] == "needs_setup"      # implemented, no venv here
        assert d["engines"][1]["setup_command"].endswith("setup_engine_venv.py gpt-researcher")
        assert d["engines"][1]["options_schema"]["properties"]["report_type"]["enum"]
        assert d["engines"][2]["health"] == "needs_setup"
        assert d["engines"][2]["options_schema"]["properties"]["max_react_tool_calls"]["default"] == 5
        status, d = _get(web, "/api/engines?mocks=1")
        assert "mock-fast" in [e["id"] for e in d["engines"]]
        status, d = _post(web, "/api/engines", {"mocks": True, "engine_python": {"gpt-researcher": "/nope"}})
        assert status == 200 and "mock-fail" in [e["id"] for e in d["engines"]]
        status, d = _post(web, "/api/engines", {"mocks": False, "engine_python": "junk"})
        assert status == 200 and len(d["engines"]) == 3


class TestEngineRequestValidation:
    def test_build_config_kwargs(self):
        out = build_config_kwargs({"research_engine": "mock-fast", "engine_max_time_seconds": "300",
                                   "engine_python": {"gpt-researcher": "/x/py", "open-deep-research": ""},
                                   "searx_url": "http://s:8080", "engine_options": {"a": 1},
                                   "engine_embedding_model": "emb"})
        assert out == {"research_engine": "mock-fast", "engine_max_time_seconds": 300,
                       "engine_python": {"gpt-researcher": "/x/py"}, "searx_url": "http://s:8080",
                       "engine_options": {"a": 1}, "engine_embedding_model": "emb"}
        assert build_config_kwargs({"research_engine": ""}) == {}
        for bad, field in [({"research_engine": "zzz"}, "research_engine"),
                           ({"engine_max_time_seconds": 1}, "engine_max_time_seconds"),
                           ({"searx_url": "ftp://x"}, "searx_url"),
                           ({"engine_python": "x"}, "engine_python"),
                           ({"engine_options": 3}, "engine_options")]:
            with pytest.raises(FieldError) as ei:
                build_config_kwargs(bad)
            assert ei.value.field == field

    def test_engine_request_error(self):
        assert engine_request_error({}) is None
        assert engine_request_error({"research_engine": "mock-fast"}) is None
        msg, field = engine_request_error({"research_engine": "open-deep-research"})
        assert field == "research_engine" and "Python 環境がありません" in msg
        msg, field = engine_request_error({"research_engine": "gpt-researcher"})
        assert field == "research_engine" and "Python 環境がありません" in msg
        msg, _ = engine_request_error({"research_engine": "gpt-researcher",
                                       "engine_python": {"gpt-researcher": "/no/such/python"}})
        assert "/no/such/python" in msg
        msg, _ = engine_request_error({"research_engine": "mock-fast", "local_documents": ["a.pdf"]})
        assert "手元の資料" in msg
        msg, _ = engine_request_error({"research_engine": "mock-fast", "source_mode": "hybrid"})
        assert "hybrid" in msg

    def test_api_rejects_before_queueing(self, web):
        for extra, field in [({"research_engine": "gpt-researcher"}, "research_engine"),
                             ({"research_engine": "open-deep-research"}, "research_engine"),
                             ({"research_engine": "zzz"}, "research_engine"),
                             ({"research_engine": "mock-fast", "local_documents": ["x"]}, "research_engine"),
                             ({"research_engine": "mock-fast", "source_mode": "local"}, "research_engine"),
                             ({"research_engine": "mock-fast", "engine_max_time_seconds": 3},
                              "engine_max_time_seconds")]:
            status, d = _post(web, "/api/research", {**BASE, **extra})
            assert status == 400, d
            assert d["field"] == field, d
        status, d = _get(web, "/api/jobs")
        assert status == 200 and not d.get("jobs")


class TestEngineJob:
    def test_mock_engine_job_runs_through_the_web_path(self, web, monkeypatch):
        """The server's /api/research → JobManager → DeepResearchTool.run
        with the request's config (engine, options) and the fakes in
        place of the LLM / search clients."""
        def offline_tool(config):
            tool = DeepResearchTool.__new__(DeepResearchTool)
            tool.config = config
            tool.llm_client = FakeLLM()
            tool.stage_llm_clients = {}
            tool.search_client = FakeSearchClient()
            tool.researcher = None
            tool.verifier = None
            tool.report_generator = None
            tool.deep_think_processor = None
            return tool

        monkeypatch.setattr(main_mod, "DeepResearchTool", offline_tool)
        status, d = _post(web, "/api/research", {
            **BASE, "research_engine": "mock-fast", "engine_options": {"speed_factor": 0.01},
            "plan_review": True, "auto_figures": False})
        assert status == 202, d
        job_id = d["job_id"]
        rec = None
        for _ in range(240):
            status, rec = _get(web, f"/api/status?job_id={job_id}")
            if rec.get("state") in ("completed", "error", "cancelled"):
                break
            time.sleep(0.5)
        assert rec["state"] == "completed", rec.get("error") or rec.get("message")
        res = rec["result"]
        assert res["research_engine"] == "mock-fast"
        assert res["engine"]["ingest"]["sources"] == 3 and res["engine"]["name"] == "Mock Fast"
        assert res["status"]["verification"] == "skipped"
        ps = rec["params_summary"]
        assert ps["research_engine"] == "mock-fast" and ps["plan_review"] is False
        assert "openai_api_key" not in json.dumps(ps) or ps.get("openai_api_key") in (None, "***", "")
        status, hist = _get(web, "/api/history")
        rows = hist.get("records") or hist.get("history") or hist
        assert any(r.get("job_id") == job_id for r in rows)
