"""Open Deep Research engine (engines/odr): request validation, the worker
environment (minimal, secrets never inherited, the engine's Configuration
fields and hosted-search keys blocked, NO_PROXY for the MCP server), the
configurable / prompt / payload, the in-process MCP search server (the
stateless streamable-HTTP protocol over plain HTTP, both tools, the log
and events), the Engine on a stand-in worker that calls that server from
the subprocess, the worker's own logic with a stubbed graph, the registry
entry, and — when the engine's virtualenv exists on this machine — the
real package: the MCP client it uses against our server, and the whole
graph end to end on a fake OpenAI server. No paid API, no network."""
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from deep_research_tool.engines import (EXTERNAL_ENGINES, EngineUnavailable,
                                        engine_catalog, get_engine)
from deep_research_tool.engines import venv as venv_mod
from deep_research_tool.engines.adapter import EngineResearcher
from deep_research_tool.engines.jsonl_worker import EngineConfigError
from deep_research_tool.engines.odr import engine as odr
from deep_research_tool.engines.odr import mcp_search
from deep_research_tool.engines.odr.mcp_search import (McpSearchServer, SearchBackend,
                                                       SearchTools, backend_for)
from deep_research_tool.engines.runner_core.engine import RunContext
from deep_research_tool.engines.runner_core.models import (LlmRunConfig, RunInput,
                                                           RunRequest, SearchRunConfig)

WORKER = Path(odr.__file__).with_name("worker.py")
VENV_PY = venv_mod.default_venv_python("open-deep-research")
SCRATCH_FAKE_LLM = Path(__file__).with_name("fixtures") / "fake_openai_odr.py"


def _llm(**kw):
    base = dict(profile_id="p", api="openai-compatible", endpoint="http://llm.local/v1",
                model="gpt-5-mini", api_key="sk-SECRET-123456")
    base.update(kw)
    return LlmRunConfig(**base)


def _req(llm=None, search=None, options=None, proxy_env=None, max_searches=None, **input_kw):
    inp = dict(topic="炭素繊維市場", language="ja")
    inp.update(input_kw)
    return RunRequest(client_run_id="r1", engine_id="open-deep-research", input=RunInput(**inp),
                      options=options or {}, llm=llm if llm is not None else _llm(),
                      search=search if search is not None else SearchRunConfig(provider="duckduckgo"),
                      proxy_env=proxy_env or {}, max_searches=max_searches)


class FakeClient:
    """The tool's search client surface the MCP tools use."""

    def __init__(self, fail_fetch=False, results=3):
        self.fail_fetch = fail_fetch
        self.results = results
        self.searches = []
        self.fetched = []

    def search(self, query, max_results=None, **kw):
        self.searches.append((query, max_results))
        n = min(self.results, max_results or self.results)
        return [SimpleNamespace(title=f"T{i} {query}", url=f"https://ex.test/{len(self.searches)}/{i}",
                                snippet=f"snippet {i}") for i in range(n)]

    def get_page_content(self, url, **kw):
        self.fetched.append(url)
        if self.fail_fetch:
            raise RuntimeError("boom")
        return SimpleNamespace(url=url, title=f"Page {url.rsplit('/', 1)[-1]}",
                               text_content="本文 " * 400, metadata={})


def rpc(url, msg, timeout=10):
    data = json.dumps(msg).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        return r.status, r.headers.get("Content-Type"), (json.loads(body) if body else None)


# ----------------------------------------------------------------------
# validation / env / payload
# ----------------------------------------------------------------------

class TestValidation:
    def test_requirements(self):
        odr.validate_request(_req())
        odr.validate_request(_req(llm=_llm(api="anthropic", endpoint="https://api.anthropic.com")))
        odr.validate_request(_req(search=SearchRunConfig(provider="searxng", endpoint="http://s:8080")))
        odr.validate_request(_req(search=SearchRunConfig(provider="selenium")))
        odr.validate_request(_req(search=SearchRunConfig(provider="disabled"),
                                  options={"search_api": "none"}))      # explicit no-search run
        cases = [
            (dict(llm=_llm(endpoint=" ")), "エンドポイント"),
            (dict(llm=_llm(model="")), "モデル名"),
            (dict(search=SearchRunConfig(provider="disabled")), "Web 検索が必要"),
            (dict(search=SearchRunConfig(provider="searxng", endpoint=None)), "searx_url"),
            (dict(options={"search_api": "tavily"}), "有料／ホスト型"),
            (dict(options={"search_api": "exa"}), "未対応"),
        ]
        for kw, needle in cases:
            with pytest.raises(EngineConfigError, match=needle):
                odr.validate_request(_req(**kw))
        req = _req()
        req.llm = None
        with pytest.raises(EngineConfigError, match="LLM の設定"):
            odr.validate_request(req)

    def test_option_bounds_and_max_searches(self):
        assert odr.resolve_max_react_tool_calls(_req()) == 5
        assert odr.resolve_max_react_tool_calls(_req(options={"max_react_tool_calls": 50})) == 20
        assert odr.resolve_max_react_tool_calls(_req(options={"max_react_tool_calls": "x"})) == 5
        assert odr.resolve_max_react_tool_calls(_req(options={"max_react_tool_calls": 8}, max_searches=3)) == 3
        assert odr.resolve_max_react_tool_calls(_req(max_searches=0)) == 1
        c = odr.build_configurable(_req(options={"max_researcher_iterations": 99,
                                                "max_concurrent_research_units": 0}), None)
        assert c["max_researcher_iterations"] == 10 and c["max_concurrent_research_units"] == 1
        payload, _ = odr.build_worker_payload(_req(options={"recursion_limit": 5}), None)
        assert payload["recursion_limit"] == 10


class TestWorkerEnv:
    def test_minimal_env_blocks_engine_fields_and_keys(self):
        base = {"PATH": "/p", "HOME": "/h", "OPENAI_API_KEY": "server-key", "TAVILY_API_KEY": "t",
                "SEARCH_API": "tavily", "RESEARCH_MODEL": "x", "HTTPS_PROXY": "http://inherited:1",
                "SSL_CERT_FILE": "/ca.pem", "NO_PROXY": "intra.example"}
        env, warnings = odr.build_worker_env(
            _req(proxy_env={"HTTPS_PROXY": "http://proxy:3128", "SEARCH_API": "tavily",
                            "TAVILY_API_KEY": "evil", "OPENAI_API_KEY": "evil"}), base_env=base)
        assert env["PATH"] == "/p" and env["HOME"] == "/h" and env["SSL_CERT_FILE"] == "/ca.pem"
        # the server's own key never leaks; the request's key is what the worker gets
        assert env["OPENAI_API_KEY"] == "sk-SECRET-123456"
        assert env["OPENAI_BASE_URL"] == "http://llm.local/v1" and env["OPENAI_API_BASE"] == "http://llm.local/v1"
        assert "TAVILY_API_KEY" not in env and "SEARCH_API" not in env and "RESEARCH_MODEL" not in env
        assert "ANTHROPIC_API_KEY" not in env
        assert env["HTTPS_PROXY"] == "http://proxy:3128" and env["https_proxy"] == "http://proxy:3128"
        assert env["GET_API_KEYS_FROM_CONFIG"] == "false"
        # the MCP server on localhost must not go through the proxy
        for host in ("127.0.0.1", "localhost", "intra.example"):
            assert host in env["NO_PROXY"].split(",") and host in env["no_proxy"].split(",")
        assert sorted(w.split()[2] for w in warnings) == ["OPENAI_API_KEY", "SEARCH_API", "TAVILY_API_KEY"]

    def test_anthropic_env(self):
        env, _ = odr.build_worker_env(
            _req(llm=_llm(api="anthropic", endpoint="https://api.anthropic.com", api_key="ak-1234567")),
            base_env={})
        assert env["ANTHROPIC_API_KEY"] == "ak-1234567"
        assert env["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com" == env["ANTHROPIC_API_URL"]
        assert "OPENAI_API_KEY" not in env
        env, _ = odr.build_worker_env(_req(llm=_llm(api_key=None)), base_env={})
        assert env["OPENAI_API_KEY"] == "dummy-key"

    def test_configurable_prompt_and_payload(self):
        req = _req(options={"max_react_tool_calls": 3}, objective="目的", instructions="指示",
                   input_urls=["https://a.example/x"])
        c = odr.build_configurable(req, "http://127.0.0.1:5555")
        assert c["allow_clarification"] is False and c["search_api"] == "none"
        for k in ("research_model", "summarization_model", "compression_model", "final_report_model"):
            assert c[k] == "openai:gpt-5-mini"
        assert c["mcp_config"] == {"url": "http://127.0.0.1:5555", "tools": ["web_search", "fetch_page"],
                                   "auth_required": False}
        assert "web_search" in c["mcp_prompt"] and "starting URL" in c["mcp_prompt"]
        assert c["max_react_tool_calls"] == 3
        c2 = odr.build_configurable(_req(llm=_llm(api="anthropic", model="claude-x")), None)
        assert c2["research_model"] == "anthropic:claude-x" and "mcp_config" not in c2
        payload, warnings = odr.build_worker_payload(req, "http://127.0.0.1:5555")
        p = payload["prompt"]
        assert p.startswith("Research topic: 炭素繊維市場") and "Objective: 目的" in p and "Additional instructions: 指示" in p
        assert "Start from these URLs" in p and "- https://a.example/x" in p
        assert "Japanese" in p and "### Sources" in p
        assert payload["language"] == "ja" and payload["recursion_limit"] == 50 and warnings == []
        # 指定 URL のみ: fetch_page only, prompt says only those pages
        only = _req(options={"complement_source_urls": False}, input_urls=["https://a.example/x"])
        assert odr.mcp_tools_for(only) == ["fetch_page"]
        payload, _ = odr.build_worker_payload(only, "http://127.0.0.1:1")
        assert payload["configurable"]["mcp_config"]["tools"] == ["fetch_page"]
        assert "Use ONLY these URLs" in payload["prompt"]
        assert "ONLY the pages listed" in payload["configurable"]["mcp_prompt"]
        # the researcher sees only its system prompt: the URLs are listed in mcp_prompt
        assert "  - https://a.example/x" in payload["configurable"]["mcp_prompt"]
        # documents are refused with a warning; search_api none is noted
        req2 = _req(options={"search_api": "none"}, documents=[{"name": "a", "text": "b"}])
        payload, warnings = odr.build_worker_payload(req2, None)
        assert "mcp_config" not in payload["configurable"] and len(warnings) == 2
        assert odr.build_prompt(RunInput(topic="t", language="en")).endswith("'[n] Title: URL'.")
        assert "English" in odr.build_prompt(RunInput(topic="t", language="en"))


# ----------------------------------------------------------------------
# the MCP search server
# ----------------------------------------------------------------------

class TestMcpSearchServer:
    def test_protocol_over_http(self):
        client = FakeClient()
        events = []
        tools = SearchTools(backend_for("duckduckgo", search_client=client), max_results=2,
                            on_event=lambda t, p: events.append((t, p)))
        with McpSearchServer(tools) as srv:
            url = srv.mcp_url
            assert srv.url.startswith("http://127.0.0.1:") and url.endswith("/mcp")
            assert url == f"http://127.0.0.1:{srv.port}/{srv.token}/mcp" and len(srv.token) >= 24
            st, ct, body = rpc(url, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                     "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                                "clientInfo": {"name": "t", "version": "1"}}})
            assert st == 200 and ct == "application/json"
            assert body["result"]["protocolVersion"] == "2025-06-18"
            assert body["result"]["capabilities"] == {"tools": {"listChanged": False}}
            assert body["result"]["serverInfo"]["name"] == mcp_search.SERVER_NAME
            st, _, body = rpc(url, {"jsonrpc": "2.0", "id": 9, "method": "initialize",
                                    "params": {"protocolVersion": "1999-01-01"}})
            assert body["result"]["protocolVersion"] == mcp_search.DEFAULT_PROTOCOL
            st, _, body = rpc(url, {"jsonrpc": "2.0", "method": "notifications/initialized"})
            assert st == 202 and body is None
            st, _, body = rpc(url, {"jsonrpc": "2.0", "id": 2, "method": "ping"})
            assert body["result"] == {}
            st, _, body = rpc(url, {"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
            names = [t["name"] for t in body["result"]["tools"]]
            assert names == ["web_search", "fetch_page"]
            assert body["result"]["tools"][0]["inputSchema"]["required"] == ["queries"]
            st, _, body = rpc(url, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                    "params": {"name": "web_search",
                                               "arguments": {"queries": ["炭素繊維 市場規模", " ", "q2"]}}})
            res = body["result"]
            assert res["isError"] is False and res["content"][0]["type"] == "text"
            text = res["content"][0]["text"]
            assert 'Search results for "炭素繊維 市場規模":' in text and 'Search results for "q2":' in text
            assert "1. T0 炭素繊維 市場規模\n   URL: https://ex.test/1/0\n   snippet 0\n   Page excerpt: 本文" in text
            assert "3. " not in text                         # max_results=2
            assert client.searches == [("炭素繊維 市場規模", 2), ("q2", 2)]
            assert len(client.fetched) == 4                  # excerpts for the top results
            st, _, body = rpc(url, {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                                    "params": {"name": "fetch_page", "arguments": {"url": "https://ex.test/p"}}})
            assert body["result"]["content"][0]["text"].startswith("Page p\nURL: https://ex.test/p\n\n本文")
            st, _, body = rpc(url, {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                                    "params": {"name": "fetch_page", "arguments": {"url": "ftp://x"}}})
            assert "invalid url" in body["result"]["content"][0]["text"]
            st, _, body = rpc(url, {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                                    "params": {"name": "nope", "arguments": {}}})
            assert body["error"]["code"] == -32602
            st, _, body = rpc(url, {"jsonrpc": "2.0", "id": 8, "method": "resources/list"})
            assert body["error"]["code"] == -32601
            # batch: one response per request, notifications dropped
            st, _, body = rpc(url, [{"jsonrpc": "2.0", "id": 10, "method": "ping"},
                                    {"jsonrpc": "2.0", "method": "notifications/x"}])
            assert st == 200 and [b["id"] for b in body] == [10]
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(url, timeout=5)              # GET: no server stream
            assert ei.value.code == 405 and ei.value.headers.get("Allow") == "POST"
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(urllib.request.Request(
                    url, data=b"{not json", headers={"Content-Type": "application/json"}), timeout=5)
            assert ei.value.code == 400
            with pytest.raises(urllib.error.HTTPError) as ei:
                rpc(srv.url + "/other", {"jsonrpc": "2.0", "id": 1, "method": "ping"})
            assert ei.value.code == 404
            # only the run's own worker: no token → 404, foreign Host / browser Origin → 403
            with pytest.raises(urllib.error.HTTPError) as ei:
                rpc(f"http://127.0.0.1:{srv.port}/mcp", {"jsonrpc": "2.0", "id": 1, "method": "ping"})
            assert ei.value.code == 404
            for headers in ({"Host": "evil.example:80"}, {"Origin": "http://evil.example"},
                            {"Origin": "null"}):
                req = urllib.request.Request(url, data=b'{"jsonrpc":"2.0","id":1,"method":"ping"}',
                                             headers={"Content-Type": "application/json", **headers})
                with pytest.raises(urllib.error.HTTPError) as ei:
                    urllib.request.urlopen(req, timeout=5)
                assert ei.value.code == 403, headers
            req = urllib.request.Request(url, data=b'{"jsonrpc":"2.0","id":1,"method":"ping"}',
                                         headers={"Content-Type": "application/json",
                                                  "Origin": f"http://localhost:{srv.port}"})
            with urllib.request.urlopen(req, timeout=5) as r:
                assert r.status == 200
            assert srv.requests_seen["tools/call"] == 4 and srv.requests_seen["initialize"] == 2
        assert srv.requests_seen["ping"] == 3                 # forbidden / unknown paths never dispatched
        log = tools.log
        assert log["searches"] == 2 and log["results"] == 4 and log["fetches"] == 5
        assert log["queries"] == ["炭素繊維 市場規模", "q2"]
        assert set(log["urls"]) == {"https://ex.test/1/0", "https://ex.test/1/1",
                                    "https://ex.test/2/0", "https://ex.test/2/1", "https://ex.test/p"}
        assert [e for e in events if e[0] == "search"] == [("search", {"query": "炭素繊維 市場規模"}),
                                                           ("search", {"query": "q2"})]
        assert sum(1 for e in events if e[0] == "source_found") == 5
        # stopped: the port no longer answers
        with pytest.raises(urllib.error.URLError):
            rpc(url, {"jsonrpc": "2.0", "id": 1, "method": "ping"}, timeout=2)

    def test_tool_errors_are_reported_not_raised(self):
        class Boom(SearchBackend):
            name = "boom"

            def search(self, query, max_results):
                raise RuntimeError("engine down")

            def fetch(self, url):
                raise RuntimeError("no page")
        tools = SearchTools(Boom())
        out = tools.web_search(["a"])
        assert "search failed: RuntimeError: engine down" in out and tools.log["search_errors"] == 1
        assert tools.fetch_page("https://x.test/") == "(could not fetch https://x.test/)"
        assert tools.log["fetch_errors"] == 1
        assert tools.web_search("not a list") == 'Search results for "not a list":\n\n(search failed: RuntimeError: engine down)'
        assert tools.web_search(42) == "(invalid queries: expected a list of strings)"
        assert tools.web_search([]) == "(no valid queries)"
        # failing excerpt fetches leave the results in place
        c = FakeClient(fail_fetch=True)
        t2 = SearchTools(backend_for("selenium", search_client=c), max_results=1)
        out = t2.web_search(["q"])
        assert "1. T0 q\n   URL: https://ex.test/1/0\n   snippet 0" in out and "Page excerpt" not in out
        assert t2.log["fetch_errors"] == 1
        # enabled tools: 指定 URL のみ serves fetch_page only
        srv = McpSearchServer(t2, enabled_tools=["fetch_page"])
        assert [t["name"] for t in srv.handle_message(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]] == ["fetch_page"]
        err = srv.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                  "params": {"name": "web_search", "arguments": {"queries": ["q"]}}})
        assert err["error"]["code"] == -32602
        assert srv.handle_message({"id": 1})["error"]["code"] == -32600

    def test_failed_fetches_and_slow_searches(self, monkeypatch):
        """A page the client could not read comes back as a page titled
        "Error" with metadata.error: it is a failed fetch, not page text.
        A backend search that runs past its bound is answered as timed out."""
        class ErrorPageClient(FakeClient):
            def get_page_content(self, url, **kw):
                return SimpleNamespace(url=url, title="Error", metadata={"error": "Read timed out"},
                                       text_content="Failed to extract content: Read timed out")
        tools = SearchTools(backend_for("duckduckgo", search_client=ErrorPageClient()), max_results=1)
        out = tools.web_search(["q"])
        assert "Page excerpt" not in out and "Failed to extract" not in out
        assert tools.fetch_page("https://x.test/a") == "(could not fetch https://x.test/a)"
        assert tools.log["fetches"] == 0 and tools.log["fetch_errors"] == 2
        assert "Error" not in tools.log["urls"].values()
        assert tools.log["failed_urls"]["https://x.test/a"] == "RuntimeError: Read timed out"

        import threading as _th
        release = _th.Event()

        class Slow(SearchBackend):
            name = "slow"

            def search(self, query, max_results):
                release.wait(5)
                return [{"title": "late", "url": "https://late.test/", "snippet": ""}]

            def fetch(self, url):
                raise RuntimeError("no")
        monkeypatch.setattr(mcp_search, "PER_QUERY_SEARCH_SECONDS", 0.2)
        t = SearchTools(Slow())
        import time as _time
        t0 = _time.monotonic()
        out = t.web_search(["a", "b"])
        assert _time.monotonic() - t0 < 3
        assert out.count("(search timed out after") == 2 and t.log["search_errors"] == 2
        release.set()
        # the overall budget: no new query once it is spent
        class SlowClient(FakeClient):
            def search(self, query, max_results=None, **kw):
                _time.sleep(0.3)
                return super().search(query, max_results, **kw)
        monkeypatch.setattr(mcp_search, "PER_QUERY_SEARCH_SECONDS", 5.0)
        monkeypatch.setattr(mcp_search, "SEARCH_BUDGET_SECONDS", 0.2)
        t2 = SearchTools(backend_for("duckduckgo", search_client=SlowClient()), excerpt_top=0)
        out = t2.web_search(["a", "b"])
        assert out.count("skipped: time budget exhausted") == 1 and t2.log["searches"] == 1

    def test_idle_connections_are_closed(self, monkeypatch):
        import socket
        monkeypatch.setattr(mcp_search._Handler, "timeout", 0.5)
        with McpSearchServer(SearchTools(backend_for("duckduckgo", search_client=FakeClient()))) as srv:
            for first in (b"", b"POST /x/mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 1000\r\n\r\n{"):
                s = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
                try:
                    if first:
                        s.sendall(first)
                    t0 = time.monotonic()
                    data = b"x"
                    while data:
                        data = s.recv(1024)          # the server closes after its idle timeout
                    assert time.monotonic() - t0 < 4
                    assert srv.requests_seen == {}  # nothing was dispatched
                finally:
                    s.close()

    def test_backends(self):
        with pytest.raises(ValueError, match="検索クライアントがありません"):
            backend_for("duckduckgo")
        with pytest.raises(ValueError, match="searx_url"):
            backend_for("searxng", endpoint=" ")
        b = backend_for("searxng", endpoint="http://s:8080/", search_client=FakeClient())
        assert isinstance(b, mcp_search.SearxSearchBackend) and b.endpoint == "http://s:8080"
        items = mcp_search.format_searx_items(
            {"results": [{"url": "https://a", "title": "A", "content": "x  y"}, {"href": "https://b"},
                         "junk", {"title": "no url"}, {"url": "https://c"}]}, 2)
        assert items == [{"title": "A", "url": "https://a", "snippet": "x y"},
                         {"title": "", "url": "https://b", "snippet": ""}]
        assert mcp_search.format_searx_items("nope", 3) == []
        cb = mcp_search.ClientSearchBackend(FakeClient(results=5))
        assert len(cb.search("q", 2)) == 2
        assert cb.search("q", 2)[0]["title"] == "T0 q"


# ----------------------------------------------------------------------
# the Engine on a stand-in worker
# ----------------------------------------------------------------------

STUB_OK = textwrap.dedent(r'''
    import json, os, sys, urllib.request
    emit_fd = os.dup(1); os.dup2(2, 1); sys.stdout = sys.stderr
    emit = os.fdopen(emit_fd, "w", buffering=1)
    payload = json.load(sys.stdin)
    mcp = payload["configurable"].get("mcp_config")
    tools, text = [], ""
    if mcp:
        url = mcp["url"].rstrip("/") + "/mcp"
        def rpc(msg):
            req = urllib.request.Request(url, data=json.dumps(msg).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                b = r.read(); return json.loads(b) if b else None
        rpc({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"stub","version":"0"}}})
        rpc({"jsonrpc":"2.0","method":"notifications/initialized"})
        tools = [t["name"] for t in rpc({"jsonrpc":"2.0","id":2,"method":"tools/list"})["result"]["tools"]]
        if "web_search" in tools:
            res = rpc({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"web_search","arguments":{"queries":["q1"]}}})
            text = res["result"]["content"][0]["text"]
    print("library noise on stdout")
    emit.write(json.dumps({"kind":"event","type":"stage","payload":{"stage":"brief"}}) + "\n")
    emit.write(json.dumps({"kind":"event","type":"stage","payload":{"stage":"final_report"}}) + "\n")
    url_seen = text.split("URL: ")[1].split()[0] if "URL: " in text else "https://ex.test/none"
    body = "# T\n\n## 概要\n\n本文 [1]。\n\n## Sources\n[1] T q1: " + url_seen + "\n"
    emit.write(json.dumps({"kind":"result","output_kind":"report","summary":"s","report_markdown":body,
      "claims":[],"sources":[],"metrics":{},"warnings":["w1"],
      "raw":{"env":{k: os.environ.get(k) for k in ("OPENAI_BASE_URL","OPENAI_API_KEY","NO_PROXY","HTTPS_PROXY","TAVILY_API_KEY","SEARCH_API")},
             "tools":tools,"payload":payload}}) + "\n")
''')

STUB_NO_SEARCH = textwrap.dedent(r'''
    import json, os, sys
    emit_fd = os.dup(1); os.dup2(2, 1); sys.stdout = sys.stderr
    emit = os.fdopen(emit_fd, "w", buffering=1)
    payload = json.load(sys.stdin)
    print("Error loading MCP tools: connection refused", file=sys.stderr)
    emit.write(json.dumps({"kind":"result","output_kind":"report","summary":"s",
      "report_markdown":"# T\n\n出典なしの本文。\n","claims":[],"sources":[],"metrics":{},"warnings":[],"raw":{}}) + "\n")
''')

STUB_FETCH_ONLY = textwrap.dedent(r'''
    import json, os, sys, urllib.request
    emit_fd = os.dup(1); os.dup2(2, 1); sys.stdout = sys.stderr
    emit = os.fdopen(emit_fd, "w", buffering=1)
    payload = json.load(sys.stdin)
    url = payload["configurable"]["mcp_config"]["url"].rstrip("/") + "/mcp"
    def rpc(msg):
        req = urllib.request.Request(url, data=json.dumps(msg).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            b = r.read(); return json.loads(b) if b else None
    for u in ("https://a.example/x", "https://a.example/y"):
        rpc({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"fetch_page","arguments":{"url":u}}})
    emit.write(json.dumps({"kind":"result","output_kind":"report","summary":"s",
      "report_markdown":"# T\n\n## 概要\n\n読めなかった。\n","claims":[],"sources":[],"metrics":{},"warnings":[],"raw":{}}) + "\n")
''')

STUB_FATAL = textwrap.dedent(r'''
    import json, os, sys
    emit_fd = os.dup(1); os.dup2(2, 1); sys.stdout = sys.stderr
    emit = os.fdopen(emit_fd, "w", buffering=1)
    json.load(sys.stdin)
    emit.write(json.dumps({"kind":"fatal","error":"RuntimeError: key sk-SECRET-123456 rejected"}) + "\n")
    sys.exit(1)
''')


class TestEngineWithStandInWorker:
    def _engine(self, tmp_path, code, client=None):
        w = tmp_path / "stub_worker.py"
        w.write_text(code, encoding="utf-8")
        return odr.OpenDeepResearchEngine(python_path=sys.executable, worker=str(w),
                                          search_client=client or FakeClient())

    def test_run_serves_search_over_mcp_and_returns_result(self, tmp_path):
        eng = self._engine(tmp_path, STUB_OK)
        caps = eng.capabilities()
        assert caps.engine_id == "open-deep-research" and caps.version == "0.0.16"
        assert set(caps.options_schema["properties"]) == {
            "max_researcher_iterations", "max_react_tool_calls",
            "max_concurrent_research_units", "recursion_limit"}
        assert caps.token_usage is True and caps.local_files is False
        req = _req(options={"max_react_tool_calls": 2}, objective="目的",
                   proxy_env={"HTTPS_PROXY": "http://p:1", "TAVILY_API_KEY": "x"})
        ctx = RunContext("r1", req)
        result = asyncio.run(eng.run(ctx))
        assert result.report_markdown.startswith("# T")
        # the worker listed no sources: the report's Sources section is used
        assert [s.url for s in result.sources] == ["https://ex.test/1/0"]
        assert result.sources[0].title == "T q1"
        assert result.warnings == ["proxy_env のキー TAVILY_API_KEY は安全のため無視しました", "w1"]
        env = result.raw["env"]
        assert env["OPENAI_BASE_URL"] == "http://llm.local/v1" and env["OPENAI_API_KEY"] == "sk-SECRET-123456"
        assert env["HTTPS_PROXY"] == "http://p:1" and "127.0.0.1" in env["NO_PROXY"]
        assert env["TAVILY_API_KEY"] is None and env["SEARCH_API"] is None
        assert result.raw["tools"] == ["web_search", "fetch_page"]
        conf = result.raw["payload"]["configurable"]
        assert conf["search_api"] == "none" and conf["max_react_tool_calls"] == 2
        assert conf["mcp_config"]["url"].startswith("http://127.0.0.1:")
        assert result.metrics.searches == 1
        log = result.raw["search_log"]
        assert log["backend"] == "duckduckgo" and log["queries"] == ["q1"] and log["fetches"] >= 1
        types_ = [e.type for e in ctx.events]
        assert ctx.events[0].payload["stage"] == "starting_mcp_server"
        assert ctx.events[0].payload["tools"] == ["web_search", "fetch_page"]
        assert ctx.events[1].payload["stage"] == "starting_worker"
        assert "search" in types_ and "source_found" in types_
        assert [e.payload["stage"] for e in ctx.events if e.type == "stage"][-2:] == ["brief", "final_report"]
        assert eng.last_search_log["searches"] == 1

    def test_urls_only_serves_fetch_page_only(self, tmp_path):
        eng = self._engine(tmp_path, STUB_OK)
        req = _req(options={"complement_source_urls": False}, input_urls=["https://a.example/x"])
        ctx = RunContext("r1", req)
        result = asyncio.run(eng.run(ctx))
        assert result.raw["tools"] == ["fetch_page"]
        assert result.raw["payload"]["configurable"]["mcp_config"]["tools"] == ["fetch_page"]
        assert ctx.events[0].payload["tools"] == ["fetch_page"]

    def test_validation_and_missing_pieces_before_any_process(self, tmp_path):
        eng = self._engine(tmp_path, STUB_OK)
        with pytest.raises(EngineConfigError, match="モデル名"):
            asyncio.run(eng.run(RunContext("r1", _req(llm=_llm(model="")))))
        eng = odr.OpenDeepResearchEngine(python_path=str(tmp_path / "nope"), worker=str(tmp_path / "w.py"),
                                         search_client=FakeClient())
        (tmp_path / "w.py").write_text("", encoding="utf-8")
        with pytest.raises(EngineConfigError, match="Python が見つかりません"):
            asyncio.run(eng.run(RunContext("r1", _req())))
        eng = odr.OpenDeepResearchEngine(python_path=sys.executable, worker=str(tmp_path / "missing.py"))
        with pytest.raises(EngineConfigError, match="ワーカーが見つかりません"):
            asyncio.run(eng.run(RunContext("r1", _req())))
        # no search client: the MCP server cannot search → refused before the worker starts
        w = tmp_path / "stub_worker.py"
        w.write_text(STUB_OK, encoding="utf-8")
        eng = odr.OpenDeepResearchEngine(python_path=sys.executable, worker=str(w))
        with pytest.raises(EngineConfigError, match="検索クライアントがありません"):
            asyncio.run(eng.run(RunContext("r1", _req())))

    def test_fatal_and_no_sources_are_engine_errors_without_the_key(self, tmp_path):
        eng = self._engine(tmp_path, STUB_FATAL)
        with pytest.raises(RuntimeError) as ei:
            asyncio.run(eng.run(RunContext("r1", _req())))
        assert "engine worker 失敗" in str(ei.value) and "SECRET-123456" not in str(ei.value)
        eng = self._engine(tmp_path, STUB_NO_SEARCH)
        with pytest.raises(RuntimeError) as ei:
            asyncio.run(eng.run(RunContext("r1", _req())))
        msg = str(ei.value)
        assert "検索ツールを一度も呼ばずに終了し" in msg and "Error loading MCP tools" in msg
        # searches happened but every one came back empty
        eng = self._engine(tmp_path, STUB_NO_SEARCH, client=FakeClient(results=0))
        eng.search_client = FakeClient(results=0)
        with pytest.raises(RuntimeError, match="検索ツールを一度も呼ばずに"):
            asyncio.run(eng.run(RunContext("r1", _req())))

    def test_urls_only_with_every_fetch_failing_names_the_pages(self, tmp_path):
        eng = self._engine(tmp_path, STUB_FETCH_ONLY, client=FakeClient(fail_fetch=True))
        req = _req(options={"complement_source_urls": False},
                   input_urls=["https://a.example/x", "https://a.example/y"])
        with pytest.raises(RuntimeError) as ei:
            asyncio.run(eng.run(RunContext("r1", req)))
        msg = str(ei.value)
        assert "ページの取得を 2 回試みましたが 1 件も読めず" in msg
        assert "https://a.example/x（RuntimeError: boom）" in msg and "MCP 接続" not in msg

    def test_empty_results_message_names_the_search(self, tmp_path):
        """The stub searches (STUB_OK) but the client returns nothing and
        the report cites no URL → the message points at the search."""
        code = STUB_OK.replace('body = "# T\\n\\n## 概要\\n\\n本文 [1]。\\n\\n## Sources\\n[1] T q1: " + url_seen + "\\n"',
                               'body = "# T\\n\\n## 概要\\n\\n本文。\\n"')
        assert "本文。" in code
        eng = self._engine(tmp_path, code, client=FakeClient(results=0))
        with pytest.raises(RuntimeError, match="検索結果が空で.*DuckDuckGo"):
            asyncio.run(eng.run(RunContext("r1", _req())))
        eng = self._engine(tmp_path, code, client=FakeClient(results=0))
        with pytest.raises(RuntimeError, match="SearXNG の URL"):
            asyncio.run(eng.run(RunContext("r1", _req(
                search=SearchRunConfig(provider="searxng", endpoint="http://s:8080")))))

    def test_through_the_adapter(self, tmp_path):
        eng = self._engine(tmp_path, STUB_OK)
        msgs = []
        r = EngineResearcher(eng, output_dir=tmp_path / "out", search_client=FakeClient(),
                             progress_callback=lambda m, p: msgs.append((m, p)),
                             llm=_llm(), search=SearchRunConfig(provider="duckduckgo"),
                             options={"max_react_tool_calls": 1})
        s = r.conduct_research("炭素繊維市場")
        assert list(s.section_contents) == ["1"]
        assert s.section_contents["1"]["content"] == "本文 [SOURCE 1]。"
        assert r.ingest_stats["sources"] == 1 and len(r.evidence_locker.get_all_evidence()) == 1
        texts = [m for m, _ in msgs]
        assert any("検索サーバー起動" in t for t in texts)
        assert any("調査方針の作成" in t for t in texts)
        assert any("最終レポートの執筆" in t for t in texts)
        assert any("検索 q1" in t for t in texts)
        pcts = [p for _, p in msgs if p is not None]
        assert pcts == sorted(pcts)
        assert s.run_config["engine_metrics"]["searches"] == 1


# ----------------------------------------------------------------------
# the worker's own logic (stubbed graph)
# ----------------------------------------------------------------------

def _llm_result(usage):
    """An LLMResult-shaped object: one ChatGeneration whose message carries usage."""
    return SimpleNamespace(generations=[[SimpleNamespace(message=_AI("x", usage))]])


def _load_worker(monkeypatch, updates, llm_usages=()):
    """worker.py imported by path with stub langchain_core / open_deep_research
    modules whose graph streams ``updates``; every entry of ``llm_usages`` is
    reported to the config's callbacks as one LLM call (as LangChain does for
    each model call, subgraphs included)."""
    class HumanMessage:
        def __init__(self, content):
            self.content = content

    class BaseCallbackHandler:
        def __init__(self):
            pass

    class Graph:
        async def astream(self, inputs, config=None, stream_mode=None, subgraphs=False):
            assert stream_mode == "updates" and subgraphs is True
            assert inputs["messages"][0].content.startswith("Research topic")
            handlers = (config or {}).get("callbacks") or []
            assert handlers and all(isinstance(h, BaseCallbackHandler) for h in handlers)
            for usage in llm_usages:
                for h in handlers:
                    h.on_llm_end(_llm_result(usage), run_id=None)
            for u in updates:
                yield u
    msgs = types.ModuleType("langchain_core.messages")
    msgs.HumanMessage = HumanMessage
    cb_base = types.ModuleType("langchain_core.callbacks.base")
    cb_base.BaseCallbackHandler = BaseCallbackHandler
    cbs = types.ModuleType("langchain_core.callbacks")
    cbs.base = cb_base
    core = types.ModuleType("langchain_core")
    core.messages = msgs
    core.callbacks = cbs
    dr = types.ModuleType("open_deep_research.deep_researcher")
    dr.deep_researcher = Graph()
    pkg = types.ModuleType("open_deep_research")
    pkg.deep_researcher = dr
    for name, mod in (("langchain_core", core), ("langchain_core.messages", msgs),
                      ("langchain_core.callbacks", cbs), ("langchain_core.callbacks.base", cb_base),
                      ("open_deep_research", pkg), ("open_deep_research.deep_researcher", dr)):
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location("odr_worker_under_test", WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


class _AI:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage_metadata = usage


class TestWorkerLogic:
    def test_research_maps_the_graph_output(self, monkeypatch):
        report = ("# 市場\n\n本文 [1]。\n\n### Sources\n[1] 経産省: https://meti.example/a\n"
                  "[2] [東レ IR](https://toray.example/ir)\n")
        updates = [
            ((), {"clarify_with_user": {}}),
            ((), {"write_research_brief": {"research_brief": "brief 1", "supervisor_messages": [
                _AI("x", {"input_tokens": 10, "output_tokens": 5})]}}),
            (("research_supervisor:1",), {"supervisor": {"supervisor_messages": [_AI("y", {"input_tokens": 3, "output_tokens": 2})]}}),
            (("research_supervisor:1", "researcher:2"), {"researcher": {"research_brief": "brief 1"}}),
            {"compress_research": {"compressed_research": "c"}},
            ((), {"final_report_generation": {"final_report": report, "messages": [_AI(report)]}}),
        ]
        # usage comes from the per-call callback: the AIMessages carried in
        # the updates (counted twice / not at all by walking chunks) are ignored
        worker = _load_worker(monkeypatch, updates,
                              llm_usages=[{"input_tokens": 10, "output_tokens": 5},
                                          {"input_tokens": 3, "output_tokens": 2},
                                          None, {"input_tokens": "x"}])
        emitted = []
        monkeypatch.setattr(worker, "_emit", lambda obj: emitted.append(obj))
        payload = {"prompt": "Research topic: x", "language": "ja",
                   "configurable": {"search_api": "none"}, "recursion_limit": 30}
        result = asyncio.run(worker._research(payload))
        assert result["raw"]["llm_calls"] == 4
        assert result["kind"] == "result" and result["report_markdown"] == report
        assert [s["url"] for s in result["sources"]] == ["https://meti.example/a", "https://toray.example/ir"]
        assert [s["title"] for s in result["sources"]] == ["経産省", "東レ IR"]
        assert result["metrics"]["prompt_tokens"] == 13 and result["metrics"]["completion_tokens"] == 7
        assert result["metrics"]["total_tokens"] == 20 and result["metrics"]["sources"] == 2
        assert result["raw"]["research_brief"] == "brief 1"
        assert result["raw"]["node_counts"] == {"clarify": 1, "brief": 1, "supervisor": 1,
                                                "researcher": 1, "compress": 1, "final_report": 1}
        assert result["summary"] == "本文 [1]。"
        stages = [e["payload"] for e in emitted if e["type"] == "stage"]
        assert [s["stage"] for s in stages] == ["clarify", "brief", "supervisor", "researcher", "compress", "final_report"]
        assert stages[3]["namespace"] == ["research_supervisor:1", "researcher:2"] and stages[3]["node"] == "researcher"
        logs = [e["payload"]["message"] for e in emitted if e["type"] == "log"]
        assert logs == ["調査方針: brief 1"]               # once, not on every update that carries it
        usage = [e for e in emitted if e["type"] == "token_usage"]
        assert usage == [{"kind": "event", "type": "token_usage",
                          "payload": {"prompt_tokens": 13, "completion_tokens": 7}}]

    def test_without_sources_section_body_urls_and_failures(self, monkeypatch):
        body = "# R\n\n本文 ([出典](https://a.example/p)) と https://b.example/q を参照。\n"
        worker = _load_worker(monkeypatch, [{"final_report_generation": {"final_report": body}}])
        monkeypatch.setattr(worker, "_emit", lambda obj: None)
        res = asyncio.run(worker._research({"prompt": "Research topic: x", "configurable": {}}))
        assert [(s["url"], s["title"]) for s in res["sources"]] == [("https://a.example/p", "出典"),
                                                                    ("https://b.example/q", None)]
        assert any("本文中の URL を出典として扱いました" in w for w in res["warnings"])
        assert any("トークン使用量" in w for w in res["warnings"])
        assert res["metrics"]["prompt_tokens"] is None
        worker = _load_worker(monkeypatch, [{"final_report_generation": {"final_report": "# R\n\n出典なし。\n"}}])
        monkeypatch.setattr(worker, "_emit", lambda obj: None)
        res = asyncio.run(worker._research({"prompt": "Research topic: x", "configurable": {}}))
        assert res["sources"] == [] and any("出典は空です" in w for w in res["warnings"])
        # a content-block list (Responses-API OpenAI models, Anthropic multi-block)
        blocks = [{"type": "text", "text": "# R\n\n本文 https://c.example/z ", "annotations": []},
                  {"type": "reasoning", "summary": []}, {"type": "text", "text": "続き。\n"}]
        worker = _load_worker(monkeypatch, [{"final_report_generation": {"final_report": blocks}}])
        monkeypatch.setattr(worker, "_emit", lambda obj: None)
        res = asyncio.run(worker._research({"prompt": "Research topic: x", "configurable": {}}))
        assert res["report_markdown"] == "# R\n\n本文 https://c.example/z 続き。\n"
        assert [s["url"] for s in res["sources"]] == ["https://c.example/z"]
        worker = _load_worker(monkeypatch, [{"write_research_brief": {"research_brief": "b"}}])
        monkeypatch.setattr(worker, "_emit", lambda obj: None)
        with pytest.raises(RuntimeError, match="final_report が生成されませんでした"):
            asyncio.run(worker._research({"prompt": "Research topic: x", "configurable": {}}))
        worker = _load_worker(monkeypatch, [{"final_report_generation": {
            "final_report": "Error generating final report: Maximum retries exceeded"}}])
        monkeypatch.setattr(worker, "_emit", lambda obj: None)
        with pytest.raises(RuntimeError, match="内部エラー"):
            asyncio.run(worker._research({"prompt": "Research topic: x", "configurable": {}}))

    def test_pure_helpers(self):
        from deep_research_tool.engines.odr.worker import (content_text, extract_usage,
                                                           find_state_value, parse_sources_section,
                                                           urls_in_body, usage_from_llm_result)
        assert content_text("a") == "a" and content_text(None) is None and content_text(3) is None
        assert content_text([{"type": "text", "text": "x", "citations": [{}]}, "y",
                             {"type": "tool_use", "id": "t"}, {"type": "text", "text": "z"}]) == "xyz"
        assert content_text([]) == ""
        assert usage_from_llm_result(_llm_result({"input_tokens": 7, "output_tokens": 2})) == (7, 2)
        assert usage_from_llm_result(SimpleNamespace(generations=[[SimpleNamespace(text="no message")]])) == (0, 0)
        assert usage_from_llm_result(None) == (0, 0)
        md = ("## 出典\n- [1] A: https://a.example/1.\n* 2. B - https://b.example/2\n"
              "[3] https://a.example/1\n\n## 付録\nhttps://c.example/not-a-source\n")
        src = parse_sources_section(md)
        assert [(s["url"], s["title"]) for s in src] == [("https://a.example/1", "A"), ("https://b.example/2", "B")]
        assert parse_sources_section("# no sources\nhttps://x.example/") == []
        assert parse_sources_section(None) == []
        assert [s["url"] for s in urls_in_body("see [x](https://a.example/p) and https://a.example/p, then https://b.example/q.")] == \
            ["https://a.example/p", "https://b.example/q"]
        assert extract_usage({"a": [_AI("x", {"input_tokens": 2, "output_tokens": 1}),
                                    {"usage_metadata": {"input_tokens": "3", "output_tokens": None}}]}) == (5, 1)
        assert extract_usage("str") == (0, 0)
        assert find_state_value({"x": [{"y": {"final_report": "r"}}]}, "final_report") == "r"
        assert find_state_value({"final_report": None}, "final_report") is None

    def test_main_reports_fatal_on_stdout_and_keeps_library_prints_off_it(self, tmp_path):
        """The real worker file, run as a process without the engine
        packages: the import failure becomes a fatal JSONL line, and a
        print() from the library side never reaches the protocol."""
        proc = subprocess.run([sys.executable, str(WORKER)], input=json.dumps(
            {"prompt": "Research topic: x", "configurable": {}, "recursion_limit": 10}),
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PYTHONPATH": str(tmp_path)})
        lines = [l for l in proc.stdout.splitlines() if l.strip()]
        assert len(lines) == 1
        msg = json.loads(lines[0])
        assert msg["kind"] == "fatal" and "ModuleNotFoundError" in msg["error"]
        assert proc.returncode == 1


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------

class TestRegistry:
    def test_catalog_and_get_engine(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DRT_ENGINES_VENV_DIR", str(tmp_path / "none"))
        monkeypatch.delenv("DRT_ENGINE_PYTHON_OPEN_DEEP_RESEARCH", raising=False)
        entry = [e for e in engine_catalog() if e["id"] == "open-deep-research"][0]
        assert entry["implemented"] is True and entry["health"] == "needs_setup"
        assert "setup_engine_venv.py open-deep-research" in entry["reason"]
        assert entry["options_schema"]["properties"]["max_react_tool_calls"]["default"] == 5
        assert "MCP" in entry["needs"] and "埋め込みモデル不要" in entry["description"]
        with pytest.raises(EngineUnavailable, match="使えません"):
            get_engine("open-deep-research")
        spec = dict(venv_mod.ENGINE_SPECS["open-deep-research"])
        monkeypatch.setitem(venv_mod.ENGINE_SPECS, "open-deep-research",
                            {**spec, "import": "json", "dist": "pytest", "min_python": (3, 8)})
        venv_mod.clear_check_cache()
        client = FakeClient()
        eng = get_engine("open-deep-research", engine_python={"open-deep-research": sys.executable},
                         search_client=client)
        assert isinstance(eng, odr.OpenDeepResearchEngine) and eng.python_path == sys.executable
        assert eng.search_client is client
        venv_mod.clear_check_cache()
        assert EXTERNAL_ENGINES["open-deep-research"]["implemented"] is True


# ----------------------------------------------------------------------
# the real package (only where its virtualenv exists)
# ----------------------------------------------------------------------

CLIENT_CHECK = textwrap.dedent('''
    import asyncio, json, sys
    from langchain_mcp_adapters.client import MultiServerMCPClient
    async def main():
        client = MultiServerMCPClient({"server_1": {"url": sys.argv[1].rstrip("/") + "/mcp",
                                                    "headers": None, "transport": "streamable_http"}})
        tools = await client.get_tools()
        by = {t.name: t for t in tools}
        res = await by["web_search"].ainvoke({"queries": ["炭素繊維 市場規模"], "max_results": 2})
        res2 = await by["fetch_page"].ainvoke({"url": "https://ex.test/42"})
        again = await client.get_tools()
        print(json.dumps({"tools": [t.name for t in tools], "again": [t.name for t in again],
                          "search": str(res)[:300], "fetch": str(res2)[:120]}, ensure_ascii=False))
    asyncio.run(main())
''')


@pytest.mark.skipif(not VENV_PY.is_file(), reason="engine virtualenv not created on this machine")
class TestRealVirtualenv:
    def _skip_unless_installed(self):
        proc = subprocess.run([str(VENV_PY), "-c", "import open_deep_research, langchain_mcp_adapters"],
                              capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            pytest.skip("engine package not installed in the virtualenv yet")

    def test_package_imports_and_health(self):
        self._skip_unless_installed()
        proc = subprocess.run([str(VENV_PY), "-c",
                               "from open_deep_research.deep_researcher import deep_researcher\n"
                               "import importlib.metadata as m\nprint(m.version('open-deep-research'))\n"],
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-800:]
        assert proc.stdout.strip() == "0.0.16"
        venv_mod.clear_check_cache()
        assert venv_mod.engine_health("open-deep-research")["health"] == "available"

    def test_the_engines_mcp_client_talks_to_our_server(self, tmp_path):
        """langchain-mcp-adapters (what open-deep-research uses) lists and
        calls our tools over streamable HTTP."""
        self._skip_unless_installed()
        script = tmp_path / "client_check.py"
        script.write_text(CLIENT_CHECK, encoding="utf-8")
        client = FakeClient()
        tools = SearchTools(backend_for("duckduckgo", search_client=client), max_results=2)
        with McpSearchServer(tools) as srv:
            env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path),
                   "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
            proc = subprocess.run([str(VENV_PY), str(script), srv.url], capture_output=True,
                                  text=True, timeout=180, env=env)
        assert proc.returncode == 0, proc.stderr[-1500:]
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        assert out["tools"] == ["web_search", "fetch_page"] == out["again"]
        assert 'Search results for "炭素繊維 市場規模"' in out["search"]
        assert "Page 42" in out["fetch"]
        assert tools.log["searches"] == 1 and tools.log["fetches"] >= 1
        assert srv.requests_seen["tools/call"] == 2

    def test_graph_end_to_end_on_a_fake_llm(self, tmp_path):
        """The whole graph (brief → supervisor → researcher with our MCP
        tools → compress → final report) against a fake OpenAI server."""
        self._skip_unless_installed()
        if not SCRATCH_FAKE_LLM.is_file():
            pytest.skip("fake OpenAI fixture missing")
        spec = importlib.util.spec_from_file_location("fake_openai_odr", SCRATCH_FAKE_LLM)
        fake = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fake)
        httpd = fake.serve(0)
        port = httpd.server_address[1]
        try:
            client = FakeClient(results=3)
            eng = odr.OpenDeepResearchEngine(python_path=str(VENV_PY), search_client=client)
            req = _req(llm=_llm(endpoint=f"http://127.0.0.1:{port}/v1", model="fake-model"),
                       options={"max_react_tool_calls": 4, "max_researcher_iterations": 2,
                                "max_concurrent_research_units": 1},
                       objective="市場規模と主要メーカー")
            req.max_time_seconds = 300
            ctx = RunContext("r1", req)
            result = asyncio.run(eng.run(ctx))
        finally:
            httpd.shutdown()
        assert result.report_markdown.startswith("# 国内の炭素繊維市場の動向")
        assert len(result.sources) >= 2 and all(s.url.startswith("https://ex.test/") for s in result.sources)
        assert result.metrics.searches == 2     # two queries in one call
        # every LLM call counted exactly once (the fake answers 100 / 50 each)
        n_calls = len(fake.CALLS)
        assert result.raw["llm_calls"] == n_calls >= 6
        assert result.metrics.prompt_tokens == 100 * n_calls
        assert result.metrics.completion_tokens == 50 * n_calls
        stages = [e.payload["stage"] for e in ctx.events if e.type == "stage"]
        for st in ("starting_mcp_server", "starting_worker", "brief", "supervisor", "researcher",
                   "compress", "final_report"):
            assert st in stages, stages
        assert [e.payload["query"] for e in ctx.events if e.type == "search"] == ["炭素繊維 市場規模 2025", "炭素繊維 メーカー 戦略"]
        names = [c["tool_calls"] for c in fake.CALLS]
        assert ["ConductResearch"] in names and ["web_search"] in names and ["fetch_page"] in names
        assert result.raw["node_counts"]["final_report"] == 1
