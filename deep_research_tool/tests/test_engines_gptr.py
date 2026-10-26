"""GPT Researcher engine (engines/gptr): request validation, the worker
environment (minimal, secrets never inherited, engine settings not
overridable), the query / payload, the Engine on a stand-in worker, the
worker's own logic with a stubbed gpt_researcher module, the registry
entry, and — when the engine's virtualenv exists on this machine — the
real import. No paid API, no network."""
import asyncio
import importlib.util
import json
import os
import sys
import textwrap
import types
from pathlib import Path

import pytest

from deep_research_tool.engines import (EXTERNAL_ENGINES, EngineUnavailable,
                                        engine_catalog, get_engine)
from deep_research_tool.engines import venv as venv_mod
from deep_research_tool.engines.adapter import EngineResearcher
from deep_research_tool.engines.gptr import engine as gptr
from deep_research_tool.engines.jsonl_worker import EngineConfigError
from deep_research_tool.engines.runner_core.engine import RunContext
from deep_research_tool.engines.runner_core.models import (LlmRunConfig, RunInput,
                                                           RunRequest, SearchRunConfig)

WORKER = Path(gptr.__file__).with_name("worker.py")
VENV_PY = venv_mod.default_venv_python("gpt-researcher")


def _llm(**kw):
    base = dict(profile_id="p", api="openai-compatible", endpoint="http://llm.local/v1",
                model="gpt-5-mini", api_key="sk-SECRET-123456", embedding_model="text-embedding-3-small")
    base.update(kw)
    return LlmRunConfig(**base)


def _req(llm=None, search=None, options=None, proxy_env=None, max_searches=None, **input_kw):
    inp = dict(topic="炭素繊維市場", language="ja")
    inp.update(input_kw)
    return RunRequest(client_run_id="r1", engine_id="gpt-researcher", input=RunInput(**inp),
                      options=options or {}, llm=llm if llm is not None else _llm(),
                      search=search if search is not None else SearchRunConfig(provider="duckduckgo"),
                      proxy_env=proxy_env or {}, max_searches=max_searches)


class TestValidation:
    def test_requirements(self):
        gptr.validate_request(_req())                       # duckduckgo, openai-compatible
        gptr.validate_request(_req(search=SearchRunConfig(provider="searxng", endpoint="http://s:8080")))
        cases = [
            (dict(llm=LlmRunConfig(profile_id="p", api="anthropic", endpoint="x", model="m",
                                   embedding_model="e")), "Anthropic"),
            (dict(llm=_llm(endpoint=" ")), "エンドポイント"),
            (dict(llm=_llm(model="")), "モデル名"),
            (dict(llm=_llm(embedding_model=None)), "埋め込みモデル"),
            (dict(search=SearchRunConfig(provider="disabled")), "Web 検索が必要"),
            (dict(search=SearchRunConfig(provider="searxng", endpoint=None)), "searx_url"),
            (dict(options={"report_type": "outline"}), "report_type"),
        ]
        for kw, msg in cases:
            with pytest.raises(EngineConfigError, match=msg):
                gptr.validate_request(_req(**kw))
        req = _req()
        req.llm = None
        with pytest.raises(EngineConfigError, match="LLM の設定"):
            gptr.validate_request(req)

    def test_max_iterations(self):
        assert gptr.resolve_max_iterations(_req()) == 3
        assert gptr.resolve_max_iterations(_req(options={"max_iterations": 7})) == 7
        assert gptr.resolve_max_iterations(_req(options={"max_iterations": "x"})) == 3
        assert gptr.resolve_max_iterations(_req(options={"max_iterations": 0})) == 1
        assert gptr.resolve_max_iterations(_req(options={"max_iterations": 7}, max_searches=2)) == 2
        assert gptr.resolve_max_iterations(_req(max_searches=0)) == 1


class TestWorkerEnv:
    BASE = {"PATH": "/p", "HOME": "/h", "LANG": "ja_JP.UTF-8", "OPENAI_API_KEY": "server-key",
            "TAVILY_API_KEY": "t", "SSL_CERT_FILE": "/ca.pem", "SECRET_THING": "x",
            "NO_PROXY": "localhost", "SYSTEMROOT": "C:\\Windows"}

    def test_minimal_env_with_engine_settings_last(self):
        env, warnings = gptr.build_worker_env(
            _req(proxy_env={"HTTPS_PROXY": "http://proxy:3128", "RETRIEVER": "tavily",
                            "tavily_api_key": "zzz"}),
            base_env=self.BASE)
        assert env["PATH"] == "/p" and env["HOME"] == "/h" and env["LANG"] == "ja_JP.UTF-8"
        assert env["SSL_CERT_FILE"] == "/ca.pem" and env["NO_PROXY"] == "localhost"
        assert env["SYSTEMROOT"] == "C:\\Windows"
        assert "SECRET_THING" not in env and "TAVILY_API_KEY" not in env
        assert env["OPENAI_API_KEY"] == "sk-SECRET-123456"     # the request's, not the server's
        assert env["HTTPS_PROXY"] == "http://proxy:3128" and env["https_proxy"] == "http://proxy:3128"
        assert env["RETRIEVER"] == "duckduckgo" and "SEARX_URL" not in env
        assert env["OPENAI_BASE_URL"] == "http://llm.local/v1"
        assert env["FAST_LLM"] == env["SMART_LLM"] == env["STRATEGIC_LLM"] == "openai:gpt-5-mini"
        assert env["EMBEDDING"] == "openai:text-embedding-3-small"
        # a non-OpenAI endpoint: embeddings are sent as strings, not token ids
        assert json.loads(env["EMBEDDING_KWARGS"]) == {"check_embedding_ctx_length": False}
        env_oa, _ = gptr.build_worker_env(
            _req(llm=_llm(endpoint="https://api.openai.com/v1")), base_env={})
        assert "EMBEDDING_KWARGS" not in env_oa
        assert env["SCRAPER"] == "bs" and env["MAX_ITERATIONS"] == "3"
        assert env["LANGUAGE"] == "japanese" and env["REPORT_FORMAT"] == "markdown"
        assert env["MAX_SEARCH_RESULTS_PER_QUERY"] == "10"
        assert env["PYTHONUNBUFFERED"] == "1" and env["PYTHONIOENCODING"] == "utf-8"
        assert len(warnings) == 2 and all("無視" in w for w in warnings)

    def test_searxng_selenium_and_dummy_key(self):
        env, warnings = gptr.build_worker_env(
            _req(llm=_llm(api_key=None, embedding_endpoint="http://other/v1"),
                 search=SearchRunConfig(provider="searxng", endpoint="http://searx:8080/", max_results=4)),
            base_env={})
        assert env["RETRIEVER"] == "searx" and env["SEARX_URL"] == "http://searx:8080/"
        assert env["OPENAI_API_KEY"] == "dummy-key" and env["MAX_SEARCH_RESULTS_PER_QUERY"] == "4"
        assert env["PATH"] and env["HOME"]
        assert any("別のエンドポイント" in w for w in warnings)
        env, warnings = gptr.build_worker_env(
            _req(search=SearchRunConfig(provider="selenium"), language="en"), base_env={})
        assert env["RETRIEVER"] == "duckduckgo" and env["LANGUAGE"] == "english"
        assert any("Selenium" in w for w in warnings)
        env, _ = gptr.build_worker_env(_req(language="xx"), base_env={})
        assert env["LANGUAGE"] == "xx"

    def test_query_and_payload(self):
        q = gptr.build_query(RunInput(topic=" 炭素繊維 ", objective="経営層向け", instructions="3社比較"))
        assert q == "炭素繊維\n\n調査目的: 経営層向け\n\n追加指示: 3社比較"
        payload, warnings = gptr.build_worker_payload(
            _req(options={"report_type": "detailed_report"}, input_urls=["https://a/1"],
                 documents=[{"name": "x", "text": "y"}]))
        assert payload == {"query": "炭素繊維市場", "report_type": "detailed_report",
                           "language": "ja", "input_urls": ["https://a/1"]}
        assert warnings and "手元の資料" in warnings[0]


STUB_OK = textwrap.dedent(r"""
    import json, os, sys
    payload = json.loads(sys.stdin.read())
    # the stand-in reports what it was given (the test asserts on it)
    print("library noise on stdout", flush=True)
    print(json.dumps({"kind": "event", "type": "stage", "payload": {"stage": "conduct_research"}}), flush=True)
    print(json.dumps({"kind": "event", "type": "log", "payload": {"message": "(report chunk)", "content": "report_chunk"}}), flush=True)
    print(json.dumps({"kind": "event", "type": "source_found", "payload": {"url": "https://s.example/one", "title": "One"}}), flush=True)
    print(json.dumps({"kind": "event", "type": "stage", "payload": {"stage": "write_report"}}), flush=True)
    print(json.dumps({"kind": "result", "output_kind": "report",
        "summary": "要約", "report_markdown": "# " + payload["query"].split("\n")[0] + "\n\n## 市場規模\n\n規模は大きい [1]。\n\n## 出典\n\n[1] One: https://s.example/one\n",
        "claims": [], "sources": [{"url": "https://s.example/one", "title": "One", "excerpt": "e"}],
        "metrics": {"sources": 1, "llm_cost_usd": 0.01, "llm_cost_is_estimate": True},
        "warnings": ["w1"], "raw": {"env": {k: os.environ.get(k) for k in ("RETRIEVER", "OPENAI_BASE_URL", "FAST_LLM", "EMBEDDING", "MAX_ITERATIONS", "LANGUAGE", "SEARX_URL", "TAVILY_API_KEY")}, "payload": payload}}), flush=True)
""")
STUB_BAD_RESULT = 'import json\nprint(json.dumps({"kind": "result", "sources": "not-a-list"}), flush=True)\n'
STUB_NO_SOURCES = textwrap.dedent(r"""
    import json, sys
    sys.stderr.write("Error: https://duckduckgo.com 403 Ratelimit. Failed fetching sources.\n")
    sys.stderr.write("Exception: key sk-SECRET-123456 rejected\n")
    sys.stderr.write("plain line without a marker\n")
    print(json.dumps({"kind": "result", "report_markdown": "I could not gather any source material.", "sources": []}), flush=True)
""")


class FakeSearch:
    def get_page_content(self, url):
        from types import SimpleNamespace
        return SimpleNamespace(url=url, title="T", text_content="本文 " * 120, metadata={})


class TestEngineWithStandInWorker:
    def _engine(self, tmp_path, code):
        w = tmp_path / "stub_worker.py"
        w.write_text(code, encoding="utf-8")
        return gptr.GptResearcherEngine(python_path=sys.executable, worker=str(w))

    def test_run_builds_env_and_returns_result(self, tmp_path):
        eng = self._engine(tmp_path, STUB_OK)
        caps = eng.capabilities()
        assert caps.engine_id == "gpt-researcher" and caps.version == "0.16.0"
        assert "report_type" in caps.options_schema["properties"]
        req = _req(options={"max_iterations": 2, "report_type": "detailed_report"},
                   objective="目的", proxy_env={"HTTPS_PROXY": "http://p:1"})
        ctx = RunContext("r1", req)
        result = asyncio.run(eng.run(ctx))
        assert result.report_markdown.startswith("# 炭素繊維市場")
        assert [s.url for s in result.sources] == ["https://s.example/one"]
        assert result.warnings == ["w1"]
        seen_env = result.raw["env"]
        assert seen_env["RETRIEVER"] == "duckduckgo" and seen_env["FAST_LLM"] == "openai:gpt-5-mini"
        assert seen_env["EMBEDDING"] == "openai:text-embedding-3-small"
        assert seen_env["MAX_ITERATIONS"] == "2" and seen_env["LANGUAGE"] == "japanese"
        assert seen_env["SEARX_URL"] is None and seen_env["TAVILY_API_KEY"] is None
        assert result.raw["payload"]["report_type"] == "detailed_report"
        assert "調査目的: 目的" in result.raw["payload"]["query"]
        types_ = [e.type for e in ctx.events]
        assert types_[0] == "stage" and ctx.events[0].payload["stage"] == "starting_worker"
        assert "source_found" in types_ and types_.count("stage") == 3

    def test_validation_before_any_process(self, tmp_path):
        eng = self._engine(tmp_path, STUB_OK)
        with pytest.raises(EngineConfigError, match="埋め込みモデル"):
            asyncio.run(eng.run(RunContext("r1", _req(llm=_llm(embedding_model="")))))
        eng = gptr.GptResearcherEngine(python_path=str(tmp_path / "nope"), worker=str(tmp_path / "w.py"))
        (tmp_path / "w.py").write_text("", encoding="utf-8")
        with pytest.raises(EngineConfigError, match="Python が見つかりません"):
            asyncio.run(eng.run(RunContext("r1", _req())))
        eng = gptr.GptResearcherEngine(python_path=sys.executable, worker=str(tmp_path / "missing.py"))
        with pytest.raises(EngineConfigError, match="ワーカーが見つかりません"):
            asyncio.run(eng.run(RunContext("r1", _req())))

    def test_malformed_result_is_an_engine_error_without_the_key(self, tmp_path):
        eng = self._engine(tmp_path, STUB_BAD_RESULT)
        with pytest.raises(RuntimeError) as ei:
            asyncio.run(eng.run(RunContext("r1", _req())))
        assert "結果を読み取れませんでした" in str(ei.value) and "SECRET" not in str(ei.value)

    def test_no_sources_is_a_failed_run_with_the_retriever_error(self, tmp_path):
        eng = self._engine(tmp_path, STUB_NO_SOURCES)
        with pytest.raises(RuntimeError) as ei:
            asyncio.run(eng.run(RunContext("r1", _req())))
        msg = str(ei.value)
        assert "情報源を 1 件も取得できませんでした" in msg and "DuckDuckGo" in msg
        assert "403 Ratelimit" in msg and "SECRET-123456" not in msg and "key *** rejected" in msg
        assert "plain line" not in msg
        eng = self._engine(tmp_path, STUB_NO_SOURCES)
        with pytest.raises(RuntimeError, match="SearXNG の URL"):
            asyncio.run(eng.run(RunContext("r1", _req(
                search=SearchRunConfig(provider="searxng", endpoint="http://s:8080")))))

    def test_through_the_adapter(self, tmp_path):
        """EngineResearcher drives the Engine like any other: stand-in
        worker → session with the chapter, evidence, progress messages."""
        eng = self._engine(tmp_path, STUB_OK)
        msgs = []
        r = EngineResearcher(eng, output_dir=tmp_path / "out", search_client=FakeSearch(),
                             progress_callback=lambda m, p: msgs.append((m, p)),
                             llm=_llm(), search=SearchRunConfig(provider="duckduckgo"),
                             options={"max_iterations": 1})
        s = r.conduct_research("炭素繊維市場")
        assert list(s.section_contents) == ["1"]
        assert s.section_contents["1"]["content"] == "規模は大きい [SOURCE 1]。"
        assert r.ingest_stats["sources"] == 1 and len(r.evidence_locker.get_all_evidence()) == 1
        texts = [m for m, _ in msgs]
        assert any("ワーカー起動" in t for t in texts)
        assert any("調査（検索と読み込み）" in t for t in texts)
        assert any("レポート執筆" in t for t in texts)
        assert not any("report chunk" in t for t in texts)
        pcts = [p for _, p in msgs if p is not None]
        assert pcts == sorted(pcts)
        assert "w1" in r.engine_warnings


def _load_worker(monkeypatch, fake_cls):
    """worker.py imported by path with a stub gpt_researcher module."""
    mod = types.ModuleType("gpt_researcher")
    mod.GPTResearcher = fake_cls
    monkeypatch.setitem(sys.modules, "gpt_researcher", mod)
    spec = importlib.util.spec_from_file_location("gptr_worker_under_test", WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


class TestWorkerLogic:
    def test_research_maps_the_library_output(self, monkeypatch, capsys):
        events = []

        class FakeGPTResearcher:
            def __init__(self, **kw):
                self.kw = kw
                self.visited_urls = {"https://b/2", "https://a/1"}

            async def conduct_research(self):
                await self.kw["websocket"].send_json({"type": "logs", "content": "added_source_url",
                                                      "output": "x", "metadata": "https://a/1"})
                await self.kw["websocket"].send_json({"type": "logs", "content": "starting_research",
                                                      "output": "Starting", "metadata": None})
                await self.kw["websocket"].send_json({"type": "images", "output": []})
                await self.kw["websocket"].send_json({"type": "report", "output": "chunk"})
                await self.kw["websocket"].send_json("not a dict")
                await self.kw["log_handler"].on_tool_start("search")
                await self.kw["log_handler"].on_agent_action("plan")
                await self.kw["log_handler"].on_research_step("scrape", {"urls": ["u"]})

            async def write_report(self):
                return "# 題名\n\n最初の段落です。\n\n## 本文\n\nx [1]\n"

            def get_source_urls(self):
                return ["https://a/1", "https://c/3", "", None]

            def get_research_sources(self):
                return [{"url": "https://a/1", "title": "A", "content": "  本文   A  " * 50},
                        {"href": "https://b/2", "raw_content": "B"}, "junk", {"title": "no url"}]

            def get_costs(self):
                return 0.0123

            def get_step_costs(self):
                return [{"step": "plan", "cost": 0.01}]

        worker = _load_worker(monkeypatch, FakeGPTResearcher)
        worker._emit = lambda obj: events.append(obj)
        result = asyncio.run(worker._research({"query": "Q", "report_type": "research_report",
                                               "input_urls": ["https://seed/"]}))
        assert result["kind"] == "result" and result["summary"] == "最初の段落です。"
        assert [s["url"] for s in result["sources"]] == ["https://a/1", "https://b/2", "https://c/3"]
        assert result["sources"][0]["excerpt"].startswith("本文 A 本文 A")
        assert len(result["sources"][0]["excerpt"]) <= 200
        assert result["metrics"]["llm_cost_usd"] == 0.0123 and result["metrics"]["prompt_tokens"] is None
        assert result["raw"]["visited_urls"] == ["https://a/1", "https://b/2"]
        assert result["raw"]["step_costs"][0]["step"] == "plan"
        assert result["claims"] == [] and len(result["warnings"]) == 3
        kinds = [(e["kind"], e.get("type"), e["payload"].get("stage") or e["payload"].get("url")
                  or e["payload"].get("message")) for e in events]
        assert ("event", "source_found", "https://a/1") in kinds
        assert ("event", "stage", "action:plan") in kinds and ("event", "stage", "scrape") in kinds
        assert ("event", "stage", "conduct_research") in kinds and ("event", "stage", "write_report") in kinds
        assert ("event", "log", "Starting") in kinds
        assert not any(e.get("type") == "images" for e in events)
        chunk = [e for e in events if e["payload"].get("content") == "report_chunk"]
        assert len(chunk) == 1
        cost = [e for e in events if e.get("type") == "cost"][0]
        assert cost["payload"] == {"llm_cost_usd": 0.0123, "estimate": True, "search_api_cost_usd": 0.0}
        # the constructor got the engine's arguments
        assert worker.ENGINE_VERSION == "0.16.0"

    def test_helpers(self, monkeypatch):
        worker = _load_worker(monkeypatch, object)
        assert worker._summary_from_report("# T\n\n## H\n\n本文  です\n\n次") == "本文 です"
        assert worker._summary_from_report("") is None
        assert worker._excerpt(None) is None and worker._excerpt("  a   b ") == "a b"
        assert worker.collect_sources([{"url": "u", "title": 3}], ["u", "v"]) == [
            {"url": "u", "title": None, "excerpt": None, "meta": {}},
            {"url": "v", "title": None, "excerpt": None, "meta": {}}]

    def test_main_reports_fatal_on_stdout_and_keeps_library_prints_off_it(self, tmp_path):
        """The real worker entry point in a subprocess: a failing library
        import becomes a fatal JSONL line; nothing else reaches stdout."""
        env = {**os.environ, "PYTHONPATH": str(tmp_path)}
        (tmp_path / "gpt_researcher.py").write_text(
            "print('noise from the library')\nraise ImportError('no deps here')\n", encoding="utf-8")
        import subprocess
        proc = subprocess.run([sys.executable, "-u", str(WORKER)], input='{"query": "q"}',
                              capture_output=True, text=True, env=env, timeout=60)
        lines = [l for l in proc.stdout.splitlines() if l.strip()]
        assert len(lines) == 1 and json.loads(lines[0])["kind"] == "fatal"
        assert "ImportError" in json.loads(lines[0])["error"]
        assert "noise from the library" in proc.stderr and proc.returncode == 1


class TestRegistry:
    def test_catalog_and_get_engine(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DRT_ENGINES_VENV_DIR", str(tmp_path / "none"))
        monkeypatch.delenv("DRT_ENGINE_PYTHON_GPT_RESEARCHER", raising=False)
        entry = [e for e in engine_catalog() if e["id"] == "gpt-researcher"][0]
        assert entry["implemented"] is True and entry["health"] == "needs_setup"
        assert "setup_engine_venv gpt-researcher" in entry["reason"]
        assert entry["options_schema"]["properties"]["max_iterations"]["default"] == 3
        with pytest.raises(EngineUnavailable, match="使えません"):
            get_engine("gpt-researcher")
        # an interpreter that imports the package → the engine is built with it
        spec = dict(venv_mod.ENGINE_SPECS["gpt-researcher"])
        monkeypatch.setitem(venv_mod.ENGINE_SPECS, "gpt-researcher",
                            {**spec, "import": "json", "dist": "pytest", "min_python": (3, 8)})
        venv_mod.clear_check_cache()
        eng = get_engine("gpt-researcher", engine_python={"gpt-researcher": sys.executable})
        assert isinstance(eng, gptr.GptResearcherEngine) and eng.python_path == sys.executable
        venv_mod.clear_check_cache()
        assert EXTERNAL_ENGINES["open-deep-research"]["implemented"] is False


@pytest.mark.skipif(not VENV_PY.is_file(), reason="engine virtualenv not created on this machine")
class TestRealVirtualenv:
    def test_package_imports_with_the_typing_shim(self):
        """The pinned gpt-researcher imports in its own interpreter (the
        0.16.0 typing bug is worked around by the worker's shim)."""
        import subprocess
        code = ("import builtins, typing\n"
                "for n in ('Any','List','Dict','Optional','Tuple','Set','Union','Callable','Iterable','Sequence'):\n"
                "    setattr(builtins, n, getattr(typing, n))\n"
                "from gpt_researcher import GPTResearcher\n"
                "import importlib.metadata as m\nprint(m.version('gpt-researcher'))\n")
        proc = subprocess.run([str(VENV_PY), "-c", code], capture_output=True, text=True, timeout=300)
        if "No module named 'gpt_researcher'" in proc.stderr:
            pytest.skip("engine package not installed in the virtualenv yet")
        assert proc.returncode == 0, proc.stderr[-800:]
        assert proc.stdout.strip() == "0.16.0"
        venv_mod.clear_check_cache()
        assert venv_mod.engine_health("gpt-researcher")["health"] == "available"
