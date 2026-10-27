"""Research engines: the registry / catalog, per-engine Python resolution,
the LLM profile, the JSONL worker driver, the EngineResearcher adapter on
the mock engines, and the whole DeepResearchTool pipeline on an engine
result (research only / verify / resume). No network, no paid API."""
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from deep_research_tool.config import create_config
from deep_research_tool.engines import (
    BUILTIN_ID, EXTERNAL_ENGINES, EngineUnavailable, check_engine_selectable,
    engine_catalog, engine_display_name, get_engine, is_external, known_engine_ids)
from deep_research_tool.engines import venv as venv_mod
from deep_research_tool.engines.adapter import EngineResearcher
from deep_research_tool.engines.jsonl_worker import (
    parse_worker_line, redact_secrets, result_from_worker_payload, run_jsonl_worker)
from deep_research_tool.engines.llm_profile import (
    describe_profile, llm_run_config, proxy_env, search_run_config)
from deep_research_tool.engines.runner_core.engine import CancelledByUser, RunContext
from deep_research_tool.engines.runner_core.models import (RunInput, RunRequest,
                                                           RunResult)
from deep_research_tool.research.researcher import ResearchState
from deep_research_tool.tests.test_e2e_pipeline import FakeLLM, FakeSearchClient
from deep_research_tool.tests.test_finalization_integration import build_tool
from deep_research_tool.utils.cancellation import RunCancelled

FAST = {"speed_factor": 0.01}


class FakeSearch:
    def __init__(self, fail=()):
        self.fail = set(fail)

    def get_page_content(self, url):
        if url in self.fail:
            raise ConnectionError("boom")
        return SimpleNamespace(url=url, title="T:" + url, text_content="本文 " * 120, metadata={})


# ----------------------------------------------------------------------
# registry / catalog
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_engine_venvs(tmp_path, monkeypatch):
    """Catalog answers must not depend on whether this machine has the
    engine virtualenvs: point the default location at an empty folder."""
    monkeypatch.setenv("DRT_ENGINES_VENV_DIR", str(tmp_path / "no-venvs"))
    for engine_id in venv_mod.ENGINE_SPECS:
        monkeypatch.delenv(venv_mod.env_var_name(engine_id), raising=False)
    venv_mod.clear_check_cache()
    yield
    venv_mod.clear_check_cache()


class TestRegistry:
    def test_known_ids_and_names(self):
        ids = known_engine_ids()
        assert ids[:3] == ["builtin", "gpt-researcher", "open-deep-research"]
        assert "mock-fast" in ids and "mock-fast" not in known_engine_ids(include_mocks=False)
        assert engine_display_name("builtin").startswith("内蔵")
        assert engine_display_name("gpt-researcher") == "GPT Researcher"
        assert engine_display_name("mock-fail") == "Mock Fail"
        assert engine_display_name("zzz") == "zzz"
        assert is_external("mock-fast") and not is_external(BUILTIN_ID) and not is_external("")

    def test_catalog_entries(self):
        cat = engine_catalog()
        assert [e["id"] for e in cat] == ["builtin", "gpt-researcher", "open-deep-research"]
        assert cat[0]["health"] == "available" and cat[0]["kind"] == "builtin"
        for e in cat[1:]:
            assert e["kind"] == "external" and e["flow"] and e["needs"]
            assert e["version"] == EXTERNAL_ENGINES[e["id"]]["version"]
        # GPT Researcher is implemented: without its virtualenv it needs setup
        assert cat[1]["id"] == "gpt-researcher" and cat[1]["health"] == "needs_setup"
        assert "setup_engine_venv gpt-researcher" in cat[1]["reason"]
        assert cat[2]["id"] == "open-deep-research" and cat[2]["health"] == "planned"
        assert "未実装" in cat[2]["reason"]
        with_mocks = engine_catalog(include_mocks=True)
        mocks = [e for e in with_mocks if e["kind"] == "mock"]
        assert {m["id"] for m in mocks} == {"mock-fast", "mock-slow", "mock-fail",
                                            "mock-partial", "mock-timeout", "mock-cancellable"}
        assert all(m["health"] == "available" for m in mocks)

    def test_get_engine_errors_name_the_reason(self):
        assert get_engine("mock-fast").engine_id == "mock-fast"
        with pytest.raises(EngineUnavailable, match="builtin"):
            get_engine("builtin")
        with pytest.raises(EngineUnavailable, match="不明"):
            get_engine("mock-nope")
        with pytest.raises(EngineUnavailable, match="不明な調査エンジン"):
            get_engine("zzz")
        with pytest.raises(EngineUnavailable, match="使えません: .*Python 環境がありません"):
            get_engine("gpt-researcher")
        with pytest.raises(EngineUnavailable, match="まだ使えません"):
            get_engine("open-deep-research")
        assert check_engine_selectable("builtin") is None
        assert check_engine_selectable("mock-fast") is None
        assert "まだ使えません" in check_engine_selectable("open-deep-research")
        assert "Python 環境がありません" in check_engine_selectable("gpt-researcher")

    def test_create_config_validates_engine_fields(self):
        c = create_config(provider="openai", openai_api_key="x", research_engine="mock-fast",
                          engine_options={"seed": 1}, engine_max_time_seconds=120,
                          engine_python={"gpt-researcher": "/x/py", "open-deep-research": ""},
                          searx_url=" http://s:8080 ")
        rc = c.research
        assert rc.research_engine == "mock-fast" and rc.engine_options == {"seed": 1}
        assert rc.engine_max_time_seconds == 120 and rc.searx_url == "http://s:8080"
        assert rc.engine_python == {"gpt-researcher": "/x/py"}
        assert create_config(provider="openai", openai_api_key="x").research.research_engine == "builtin"
        for bad, msg in [(dict(research_engine="nope"), "Unknown research_engine"),
                         (dict(research_engine="mock-fast", source_mode="local"), "'web', 'urls' or 'urls_web' only"),
                         (dict(engine_max_time_seconds=5), "between 60 and 86400"),
                         (dict(engine_max_time_seconds="x"), "integer"),
                         (dict(engine_python="x"), "mapping"),
                         (dict(engine_options=[1]), "mapping")]:
            with pytest.raises(ValueError, match=msg):
                create_config(provider="openai", openai_api_key="x", **bad)


# ----------------------------------------------------------------------
# per-engine Python
# ----------------------------------------------------------------------
class TestVenv:
    def test_resolution_order(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DRT_ENGINES_VENV_DIR", str(tmp_path / "venvs"))
        monkeypatch.delenv("DRT_ENGINE_PYTHON_GPT_RESEARCHER", raising=False)
        assert venv_mod.resolve_engine_python("gpt-researcher") == (None, "none")
        default = venv_mod.default_venv_python("gpt-researcher")
        assert str(default).startswith(str(tmp_path / "venvs" / "gpt-researcher"))
        default.parent.mkdir(parents=True)
        default.write_text("")
        assert venv_mod.resolve_engine_python("gpt-researcher") == (str(default), "default")
        envpy = tmp_path / "envpy"
        envpy.write_text("")
        monkeypatch.setenv("DRT_ENGINE_PYTHON_GPT_RESEARCHER", str(envpy))
        assert venv_mod.resolve_engine_python("gpt-researcher") == (str(envpy), "env")
        cfgpy = tmp_path / "cfgpy"
        cfgpy.write_text("")
        assert venv_mod.resolve_engine_python(
            "gpt-researcher", {"gpt-researcher": str(cfgpy)}) == (str(cfgpy), "config")
        assert venv_mod.resolve_engine_python(
            "gpt-researcher", {"gpt-researcher": str(tmp_path / "missing")})[1] == "config-missing"
        assert venv_mod.env_var_name("open-deep-research") == "DRT_ENGINE_PYTHON_OPEN_DEEP_RESEARCH"
        assert "setup_engine_venv gpt-researcher" in venv_mod.setup_command("gpt-researcher")

    def test_health_and_interpreter_check(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DRT_ENGINES_VENV_DIR", str(tmp_path / "venvs"))
        monkeypatch.delenv("DRT_ENGINE_PYTHON_GPT_RESEARCHER", raising=False)
        h = venv_mod.engine_health("gpt-researcher")
        assert h["health"] == "needs_setup" and "setup_engine_venv" in h["reason"]
        # this interpreter + a module that exists → ok; + an impossible
        # minimum version → unsupported; + a missing module → needs_setup
        spec = dict(venv_mod.ENGINE_SPECS["gpt-researcher"])
        monkeypatch.setitem(venv_mod.ENGINE_SPECS, "gpt-researcher",
                            {**spec, "import": "json", "dist": "pytest", "min_python": (3, 8)})
        venv_mod.clear_check_cache()
        h = venv_mod.engine_health("gpt-researcher", {"gpt-researcher": sys.executable})
        assert h["health"] == "available" and h["python_version"].startswith("3.")
        assert h["package_version"]
        monkeypatch.setitem(venv_mod.ENGINE_SPECS, "gpt-researcher",
                            {**spec, "import": "json", "min_python": (9, 0)})
        venv_mod.clear_check_cache()
        h = venv_mod.engine_health("gpt-researcher", {"gpt-researcher": sys.executable})
        assert h["health"] == "unsupported" and "9.0" in h["reason"]
        monkeypatch.setitem(venv_mod.ENGINE_SPECS, "gpt-researcher",
                            {**spec, "import": "no_such_module_xyz", "min_python": (3, 8)})
        venv_mod.clear_check_cache()
        h = venv_mod.engine_health("gpt-researcher", {"gpt-researcher": sys.executable})
        assert h["health"] == "needs_setup" and "ModuleNotFoundError" in h["reason"]
        # cached: a second call with the same path does not re-run the interpreter
        calls = []
        monkeypatch.setattr(venv_mod.subprocess, "run",
                            lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(AssertionError))
        assert venv_mod.engine_health("gpt-researcher", {"gpt-researcher": sys.executable})["health"] == "needs_setup"
        assert calls == []
        assert venv_mod.check_engine_python("nope", sys.executable)["ok"] is False
        assert "not found" in venv_mod.check_engine_python("gpt-researcher", str(tmp_path / "x"))["error"]


# ----------------------------------------------------------------------
# LLM / search profile
# ----------------------------------------------------------------------
class TestLlmProfile:
    def test_openai_local_anthropic(self):
        c = create_config(provider="openai", openai_api_key="sk-1", model="gpt-5-mini",
                          openai_base_url="https://gw.example/v1")
        p = llm_run_config(c)
        assert (p.api, p.endpoint, p.model, p.api_key, p.embedding_model) == \
            ("openai-compatible", "https://gw.example/v1", "gpt-5-mini", "sk-1", "text-embedding-3-small")
        c = create_config(provider="openai", openai_api_key="sk-1", engine_embedding_model="emb-x")
        assert llm_run_config(c).embedding_model == "emb-x"
        assert llm_run_config(c).endpoint == "https://api.openai.com/v1"
        c = create_config(provider="local", local_base_url="http://localhost:11434",
                          local_backend="ollama", local_model="llama3.1:8b", local_timeout=300)
        p = llm_run_config(c)
        assert p.endpoint == "http://localhost:11434/v1" and p.api_key == "local"
        assert p.embedding_model == "nomic-embed-text" and p.timeout_seconds == 300
        c = create_config(provider="local", local_base_url="http://h:8000/v1",
                          local_backend="openai_compatible", local_api_key="k")
        assert llm_run_config(c).endpoint == "http://h:8000/v1" and llm_run_config(c).api_key == "k"
        c = create_config(provider="anthropic", anthropic_api_key="a-1")
        p = llm_run_config(c)
        assert p.api == "anthropic" and p.endpoint == "https://api.anthropic.com"
        assert p.embedding_model is None
        d = describe_profile(p)
        assert d["has_api_key"] is True and "api_key" not in d
        assert describe_profile(None) == {}

    def test_local_without_url_is_an_error(self):
        c = create_config(provider="local", local_backend="ollama")
        c.api.local_base_url = None
        with pytest.raises(ValueError, match="local_base_url"):
            llm_run_config(c)

    def test_search_and_proxy(self, monkeypatch):
        # create_config falls back to the process environment for proxies
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            monkeypatch.delenv(k, raising=False)
        c = create_config(provider="openai", openai_api_key="x", search_method="duckduckgo")
        s = search_run_config(c)
        assert s.provider == "duckduckgo" and s.endpoint is None
        c = create_config(provider="openai", openai_api_key="x", searx_url="http://s:8080")
        s = search_run_config(c)
        assert s.provider == "searxng" and s.endpoint == "http://s:8080"
        c = create_config(provider="openai", openai_api_key="x", search_method="selenium")
        assert search_run_config(c).provider == "selenium"
        c = create_config(provider="openai", openai_api_key="x", https_proxy="http://p:3128")
        assert proxy_env(c) == {"HTTPS_PROXY": "http://p:3128"}
        assert proxy_env(create_config(provider="openai", openai_api_key="x")) == {}


# ----------------------------------------------------------------------
# JSONL worker driver
# ----------------------------------------------------------------------
WORKER_OK = r"""
import json, sys, time
payload = json.loads(sys.stdin.read())
print("not json", flush=True)
print(json.dumps({"kind": "event", "type": "stage", "payload": {"stage": "planning", "step": 1, "total": 2}}), flush=True)
print(json.dumps({"kind": "event", "type": "log", "payload": "bad payload"}), flush=True)
print(json.dumps({"kind": "result", "report_markdown": "# R\n\n## A\n\n" + payload["query"],
                  "sources": [{"url": "https://x/1", "title": "t"}]}), flush=True)
"""
WORKER_FATAL = r"""
import json, sys
print(json.dumps({"kind": "fatal", "error": "key sk-SECRET123 rejected"}), flush=True)
"""
WORKER_CRASH = r"""
import sys
sys.stderr.write("Traceback: boom sk-SECRET123\n")
sys.exit(3)
"""
WORKER_HANG = r"""
import time
time.sleep(30)
"""


def _request():
    return RunRequest(client_run_id="r1", engine_id="x", input=RunInput(topic="t"))


class TestJsonlWorker:
    def test_parse_and_redact(self):
        assert parse_worker_line("plain") is None
        assert parse_worker_line('{"nokind": 1}') is None
        assert parse_worker_line(' {"kind": "event"} ') == {"kind": "event"}
        assert parse_worker_line("{bad json") is None
        assert redact_secrets("key sk-SECRET123 and short", ["sk-SECRET123", "short", None]) == \
            "key *** and short"
        r = result_from_worker_payload({"kind": "result", "summary": "s"})
        assert isinstance(r, RunResult) and r.summary == "s"

    def _run(self, code, ctx=None, **kw):
        ctx = ctx or RunContext("r1", _request())
        return asyncio.run(run_jsonl_worker(
            ctx, argv=[sys.executable, "-u", "-c", code], env=dict(os.environ),
            payload={"query": "Q"}, cwd=os.getcwd(), secrets=["sk-SECRET123"], **kw)), ctx

    def test_events_and_result(self):
        payload, ctx = self._run(WORKER_OK)
        assert payload["kind"] == "result" and "Q" in payload["report_markdown"]
        assert [e.type for e in ctx.events] == ["stage", "log"]
        assert ctx.events[0].payload["stage"] == "planning" and ctx.events[1].payload == {}

    def test_fatal_and_crash_mask_secrets(self):
        with pytest.raises(RuntimeError) as ei:
            self._run(WORKER_FATAL)
        assert "***" in str(ei.value) and "SECRET123" not in str(ei.value)
        with pytest.raises(RuntimeError) as ei:
            self._run(WORKER_CRASH)
        assert "exit=3" in str(ei.value) and "***" in str(ei.value)
        assert "SECRET123" not in str(ei.value)

    def test_cancel_terminates_the_worker(self):
        ctx = RunContext("r1", _request())
        ctx.request_cancel()
        t = time.monotonic()
        with pytest.raises(CancelledByUser):
            self._run(WORKER_HANG, ctx=ctx, term_grace_seconds=1.0)
        assert time.monotonic() - t < 15


# ----------------------------------------------------------------------
# EngineResearcher on the mock engines
# ----------------------------------------------------------------------
class TestAdapter:
    def _r(self, engine, tmp_path, **kw):
        kw.setdefault("options", FAST)
        kw.setdefault("search_client", FakeSearch())
        msgs = []
        kw.setdefault("progress_callback", lambda m, p: msgs.append((m, p)))
        r = EngineResearcher(get_engine(engine), output_dir=tmp_path, **kw)
        r.msgs = msgs
        return r

    def test_mock_fast_end_to_end(self, tmp_path):
        r = self._r("mock-fast", tmp_path, search_client=FakeSearch(fail={"https://example.org/stats/" + "x"}))
        s = r.conduct_research("炭素繊維市場", requirements="簡潔に")
        assert s.state == ResearchState.COMPLETED and s.requirements == "簡潔に"
        assert list(s.section_contents) == ["1"] and s.completed_sections == ["1"]
        assert "[SOURCE 1]" in s.section_contents["1"]["content"]
        assert len(r.evidence_locker.get_all_evidence()) == 3
        assert r.ingest_stats == {"sources": 3, "fetched": 3, "excerpt_only": 0}
        assert Path(r.engine_artifacts["engine_result"]).exists()
        assert Path(s.saved_artifacts["session"]).exists()
        assert Path(s.saved_artifacts["evidence_json"]).exists()
        assert s.stage == "completed" and s.run_config["research_engine"] == "mock-fast"
        assert s.run_config["engine_metrics"]["sources"] == 3
        assert r.result.metrics.prompt_tokens and r.engine_elapsed is not None
        assert r.get_session() is s and r.get_evidence_locker() is r.evidence_locker
        assert r.expand_section_content() == {"characters_added": 0, "new_sources": 0}
        assert r.current_activity is None and r.resume_plan is None
        texts = [m for m, _ in r.msgs]
        assert texts[0].startswith("Planning (Mock Fast)")
        assert texts[1] == "Researching (Mock Fast): 開始"
        assert any(t.startswith("Researching (Mock Fast): 計画") for t in texts)
        assert any("検索 炭素繊維市場" in t for t in texts)
        assert texts[-1].startswith("Researching (Mock Fast): 完了（1 章")
        pcts = [p for _, p in r.msgs if p is not None]
        assert pcts == sorted(pcts) and pcts[-1] == 99.0 and 10.0 in pcts
        # the activity line while a stage runs
        data = json.loads(Path(r.engine_artifacts["engine_result"]).read_text(encoding="utf-8"))
        assert data["request"]["topic"] == "炭素繊維市場" and "api_key" not in json.dumps(data)

    def test_excerpt_only_sources_are_warned(self, tmp_path):
        r = self._r("mock-partial", tmp_path, fetch_sources=False)
        r.conduct_research("部分")
        assert any("抜粋のみ" in w for w in r.engine_warnings)
        assert any("token" in w for w in r.engine_warnings)      # the engine's own warning
        assert r.ingest_stats["excerpt_only"] == 3

    def test_failure_timeout_and_cancel(self, tmp_path):
        r = self._r("mock-fail", tmp_path)
        with pytest.raises(RuntimeError, match="Mock Fail の実行に失敗しました"):
            r.conduct_research("x")
        assert r.session.state == ResearchState.ERROR and "session" in r.session.saved_artifacts
        assert r.session.error_message.startswith("Mock Fail の実行に失敗")

        r = self._r("mock-timeout", tmp_path, max_time_seconds=1, options={"speed_factor": 0.2})
        with pytest.raises(RuntimeError, match="制限時間 1 秒"):
            r.conduct_research("x")
        assert r.session.state == ResearchState.ERROR

        ev = threading.Event()

        def cancel_check():
            if ev.is_set():
                raise RunCancelled("cancelled by user")

        r = self._r("mock-cancellable", tmp_path, cancel_check=cancel_check,
                    options={"speed_factor": 0.2})
        threading.Timer(0.5, ev.set).start()
        t = time.monotonic()
        with pytest.raises(RunCancelled):
            r.conduct_research("x")
        assert time.monotonic() - t < 8
        assert r.session.state == ResearchState.CANCELLED
        assert r.session.stage == "cancelled" and Path(r.session.saved_artifacts["session"]).exists()

    def test_documents_refused(self, tmp_path):
        r = self._r("mock-fast", tmp_path)
        with pytest.raises(ValueError, match="手元の資料"):
            r.conduct_research("x", additional_documents=[{"path": "a", "title": "a", "content": "b"}])

    def test_resume_reuses_completed_and_reruns_unfinished(self, tmp_path):
        r = self._r("mock-fast", tmp_path, fetch_sources=False)
        s = r.conduct_research("再開")
        r2 = self._r("mock-fast", tmp_path)
        s2 = r2.resume_research(s.saved_artifacts["session"])
        assert s2.session_id == s.session_id and s2.section_contents["1"]["content"]
        assert r2.resume_plan == {"reused": ["1"], "to_process": [], "engine": "mock-fast"}
        assert len(r2.evidence_locker.get_all_evidence()) == 3     # loaded, not refetched
        assert r2.msgs[-1][0].startswith("Reusing completed sections (Mock Fast)")
        # an unfinished run (the engine failed) is run again under the same id
        rf = self._r("mock-fail", tmp_path)
        with pytest.raises(RuntimeError):
            rf.conduct_research("やり直し")
        r3 = self._r("mock-fast", tmp_path)
        s3 = r3.resume_research(rf.session.saved_artifacts["session"])
        assert s3.session_id == rf.session.session_id and s3.state == ResearchState.COMPLETED
        assert s3.query == "やり直し" and r3.resume_plan is None
        assert any("最初から実行" in m for m, _ in r3.msgs)


# ----------------------------------------------------------------------
# the whole tool on an engine result
# ----------------------------------------------------------------------
class TestToolIntegration:
    def _tool(self, tmp_path, **kw):
        kw.setdefault("enable_verification", False)
        return build_tool(tmp_path, FakeLLM(), FakeSearchClient(), research_engine="mock-fast",
                          engine_options=FAST, **kw)

    def test_research_only_run_and_resume(self, tmp_path):
        tool = self._tool(tmp_path)
        msgs = []
        res = tool.run(query="炭素繊維市場", progress_callback=lambda m, p: msgs.append((m, p)))
        assert res["run_status"] == "completed" and res["research_engine"] == "mock-fast"
        eng = res["engine"]
        assert eng["id"] == "mock-fast" and eng["name"] == "Mock Fast" and eng["version"] == "1.0.0"
        assert eng["ingest"] == {"sources": 3, "fetched": 3, "excerpt_only": 0}
        assert eng["metrics"]["searches"] == 2 and eng["llm"]["model"]
        assert eng["llm"]["has_api_key"] is True and "sk-test" not in json.dumps(eng)
        assert Path(res["report_path"]).exists()
        body = Path(res["report_path"]).read_text(encoding="utf-8")
        assert "主要な発見" in body and "120億ドル" in body
        assert res["research_plan"]["title"].startswith("炭素繊維市場 調査レポート")
        assert res["status"]["verification"] == "skipped"
        assert tool.research_engine_id == "mock-fast"
        # the engine's progress landed inside the research slice (< writing)
        engine_pcts = [p for m, p in msgs if m.startswith("Researching (Mock Fast)") and p is not None]
        assert engine_pcts and max(engine_pcts) < 90
        # the Web UI activity line names the engine while it runs
        tool.researcher.current_activity = {"phase": "engine", "engine": "Mock Fast",
                                            "stage": "searching", "stage_label": "検索",
                                            "index": 2, "total": 4}
        tool.progress_plan.enter("research")
        act = tool.current_activity()
        assert act["text"] == "Mock Fast が調査しています: 検索（2/4）"
        assert act["detail"]["engine"] == "Mock Fast"
        # resume: the engine result is reused, later stages skipped
        res2 = tool.run(query="炭素繊維市場", resume_from=res["saved_artifacts"]["session"])
        assert res2["run_status"] == "completed"
        assert res2["resume_plan"]["reused"] == ["1"]
        assert "evidence_review" in res2["resume_plan"]["stages_skipped"]
        assert res2["engine"]["resumed"]["reused"] == ["1"]

    def test_v1_is_forced_and_engine_warnings_join_the_run(self, tmp_path):
        tool = self._tool(tmp_path, report_generator_version="v2")
        res = tool.run(query="炭素繊維市場")
        from deep_research_tool.config import ReportGeneratorVersion
        assert tool.config.report.generator_version == ReportGeneratorVersion.V1
        msgs = [w["message"] for w in res["warnings"]]
        assert any("V1 のレポート生成" in m for m in msgs)
        tool = build_tool(tmp_path / "p", FakeLLM(), FakeSearchClient(),
                          research_engine="mock-partial", engine_options=FAST,
                          enable_verification=False)
        res = tool.run(query="部分")
        assert any(w["source"] == "Engine:mock-partial" and "token" in w["message"]
                   for w in res["warnings"])

    def test_verification_runs_on_engine_output(self, tmp_path):
        tool = self._tool(tmp_path, enable_verification=True, verification_profile="fast")
        res = tool.run(query="炭素繊維市場")
        assert res["run_status"] == "completed"
        assert res["status"]["process"] == "completed"
        assert res["verification_summary"] is not None
        assert Path(res["report_path"]).exists()

    def test_engine_errors_surface_as_run_errors_with_partial_artifacts(self, tmp_path):
        tool = build_tool(tmp_path, FakeLLM(), FakeSearchClient(), research_engine="mock-fail",
                          engine_options=FAST, enable_verification=False)
        with pytest.raises(RuntimeError, match="Mock Fail の実行に失敗"):
            tool.run(query="x")
        assert "session" in tool.partial_artifacts
        tool = build_tool(tmp_path / "s", FakeLLM(), FakeSearchClient(),
                          research_engine="mock-fast", source_mode="web",
                          enable_verification=False)
        tool.config.research.source_mode = type(tool.config.research.source_mode)("hybrid")
        with pytest.raises(ValueError, match="手元の資料"):
            tool.run(query="x")
        tool = build_tool(tmp_path / "u", FakeLLM(), FakeSearchClient(),
                          research_engine="mock-fast", enable_verification=False)
        tool.config.research.research_engine = "open-deep-research"
        with pytest.raises(ValueError, match="まだ使えません"):
            tool.run(query="x")
        tool.config.research.research_engine = "gpt-researcher"
        with pytest.raises(ValueError, match="Python 環境がありません"):
            tool.run(query="x")

    def test_run_config_snapshot_records_the_engine(self, tmp_path):
        tool = self._tool(tmp_path)
        snap = tool._run_config_snapshot()
        assert snap["research_engine"] == "mock-fast" and snap["engine_options"] == FAST
        assert snap["engine_max_time_seconds"] == 1800
        assert tool._engine_summary() is None          # before any run
