"""
Configured vs MEASURED parallelism is visible from the tool:
- ActivityGauge counts in-flight work per stage (llm / fetch / extract / verify)
- the real LocalLLMClient issues requests in parallel only up to its
  configured cap (default 1), and the gauge records what actually ran
- the job status carries the snapshot; the result carries performance
- /api/llm-parallel-test probes a LOCAL server and reports the real
  server-side concurrency (never a paid API)
"""

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from deep_research_tool.utils.concurrency import (
    ActivityGauge, ContextThreadPoolExecutor, bind_activity_gauge,
    current_activity_gauge, track_activity)


# --------------------------------------------------------------------------
# a threaded fake OpenAI-compatible server that counts in-flight requests
# --------------------------------------------------------------------------

class _State:
    inflight = 0
    peak = 0
    lock = threading.Lock()
    delay = 0.3
    serialize = False            # emulate a server that runs one at a time
    serve_lock = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        if _State.serialize:
            _State.serve_lock.acquire()
        try:
            with _State.lock:
                _State.inflight += 1
                _State.peak = max(_State.peak, _State.inflight)
            time.sleep(_State.delay)
            with _State.lock:
                _State.inflight -= 1
        finally:
            if _State.serialize:
                _State.serve_lock.release()
        body = json.dumps({"id": "x", "object": "chat.completion", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "1"},
             "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1,
                      "total_tokens": 6}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def llm_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _State.peak = 0
    _State.serialize = False
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


class TestActivityGauge:

    def test_tracks_active_peak_total(self):
        g = ActivityGauge()
        with g.track("llm"):
            with g.track("llm"):
                assert g.snapshot()["llm"]["active"] == 2
        snap = g.snapshot()["llm"]
        assert snap == {"active": 0, "peak": 2, "total": 2}

    def test_bound_gauge_is_seen_by_pool_workers(self):
        g = bind_activity_gauge()
        assert current_activity_gauge() is g

        def work():
            with track_activity("fetch"):
                time.sleep(0.05)
        with ContextThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(lambda _: work(), range(4)))
        assert g.snapshot()["fetch"]["peak"] == 4
        assert g.snapshot()["fetch"]["total"] == 4


class TestLocalClientParallelism:

    def _fire(self, client, n=4):
        errors = []

        def one():
            try:
                client.generate("テスト", system_prompt="s")
            except Exception as e:
                errors.append(e)
        t0 = time.time()
        with ContextThreadPoolExecutor(max_workers=n) as ex:
            list(ex.map(lambda _: one(), range(n)))
        assert not errors, errors
        return time.time() - t0

    def test_default_cap_is_one_and_measured_as_one(self, llm_server):
        from deep_research_tool.api import get_client
        gauge = bind_activity_gauge()
        client = get_client(provider="local", api_key="x", model="m",
                            base_url=llm_server, backend="openai_compatible")
        assert client.max_concurrency == 1
        wall = self._fire(client, 4)
        assert _State.peak == 1               # server never saw 2 at once
        assert wall >= 4 * _State.delay * 0.9
        assert gauge.snapshot()["llm"] == {"active": 0, "peak": 1, "total": 4}

    def test_cap_eight_runs_in_parallel_and_is_measured(self, llm_server):
        from deep_research_tool.api import get_client
        gauge = bind_activity_gauge()
        client = get_client(provider="local", api_key="x", model="m",
                            base_url=llm_server, backend="openai_compatible",
                            local_concurrency=8)
        assert client.max_concurrency == 8
        wall = self._fire(client, 4)
        assert _State.peak == 4
        assert wall < 4 * _State.delay          # overlapped
        assert gauge.snapshot()["llm"]["peak"] == 4

    def test_tool_snapshot_reports_configured_and_measured(self, llm_server, tmp_path):
        from deep_research_tool.config import create_config
        from deep_research_tool.main import DeepResearchTool
        cfg = create_config(provider="local", local_base_url=llm_server,
                            local_api_key="x", local_backend="openai_compatible",
                            model="m", local_concurrency=3,
                            output_dir=str(tmp_path))
        tool = DeepResearchTool.__new__(DeepResearchTool)
        tool.config = cfg
        tool.llm_client = tool._create_llm_client()
        tool.activity_gauge = bind_activity_gauge()
        from deep_research_tool.utils.concurrency import RunLimits
        tool.run_limits = RunLimits(8)
        tool.llm_client.concurrency_limiter = tool.run_limits
        self._fire(tool.llm_client, 3)
        snap = tool.concurrency_snapshot()
        assert snap["provider"] == "local"
        assert snap["configured"]["llm"] == 3
        assert snap["configured"]["parallel_max_workers"] == 8
        assert snap["activity"]["llm"]["peak"] == 3
        assert snap["run_permits"]["limit"] == 8 and snap["run_permits"]["peak"] == 3

    def test_serialized_local_run_raises_a_warning(self, tmp_path):
        from deep_research_tool.main import DeepResearchTool
        from deep_research_tool.utils.helpers import ResearchWarnings
        collector = ResearchWarnings.bind()
        tool = DeepResearchTool.__new__(DeepResearchTool)
        tool._warn_if_llm_serialized({
            "provider": "local", "configured": {"llm": 8},
            "activity": {"llm": {"active": 0, "peak": 1, "total": 40}}})
        msgs = [w["message"] for w in collector.to_dict_list()]
        assert any("同時リクエスト上限は 8" in m for m in msgs)
        # a cloud provider or a genuinely parallel run never warns
        collector2 = ResearchWarnings.bind()
        tool._warn_if_llm_serialized({"provider": "openai", "configured": {"llm": 8},
                                      "activity": {"llm": {"peak": 1, "total": 40}}})
        tool._warn_if_llm_serialized({"provider": "local", "configured": {"llm": 8},
                                      "activity": {"llm": {"peak": 4, "total": 40}}})
        assert collector2.to_dict_list() == []
        ResearchWarnings.unbind()


# --------------------------------------------------------------------------
# Web UI: status field + parallel-test endpoint
# --------------------------------------------------------------------------

@pytest.fixture
def web(tmp_path):
    from deep_research_tool.webui.server import JobManager, WebUIHandler
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WebUIHandler)
    httpd.job_manager = JobManager(output_dir=str(tmp_path))
    httpd.output_dir = str(tmp_path)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()


def _post(server, path, payload):
    port = server.server_address[1]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class TestWebVisibility:

    def test_job_status_carries_concurrency_snapshot(self):
        from deep_research_tool.webui.server import ResearchJob
        job = ResearchJob("job-c", "q", params={"query": "q"})
        assert job.to_dict()["concurrency"] is None
        job.concurrency_source = lambda: {"provider": "local",
                                          "configured": {"llm": 4},
                                          "activity": {"llm": {"active": 2, "peak": 3, "total": 9}}}
        assert job.to_dict()["concurrency"]["activity"]["llm"]["peak"] == 3

    def test_parallel_test_endpoint_reports_server_side_concurrency(self, web, llm_server):
        status, d = _post(web, "/api/llm-parallel-test", {
            "local_base_url": llm_server, "local_backend": "openai_compatible",
            "model": "m", "count": 4, "concurrency": 4})
        assert status == 200, d
        assert d["sent"] == 4 and d["succeeded"] == 4
        assert d["client_inflight_peak"] == 4
        assert d["effective_concurrency"] >= 3.2
        assert "並列処理できています" in d["verdict"]

        # client cap 1: the client itself sends one at a time
        status, d = _post(web, "/api/llm-parallel-test", {
            "local_base_url": llm_server, "local_backend": "openai_compatible",
            "model": "m", "count": 4, "concurrency": 1})
        assert status == 200
        assert d["client_inflight_peak"] == 1
        assert "クライアント側で 1 件ずつ" in d["verdict"]

        # client cap 4 but the SERVER serializes: detected from the wall time
        _State.serialize = True
        status, d = _post(web, "/api/llm-parallel-test", {
            "local_base_url": llm_server, "local_backend": "openai_compatible",
            "model": "m", "count": 4, "concurrency": 4})
        assert status == 200
        assert d["client_inflight_peak"] == 4
        assert d["effective_concurrency"] <= 1.3
        assert "サーバーが直列処理" in d["verdict"]

    def test_parallel_test_requires_a_local_url(self, web):
        status, d = _post(web, "/api/llm-parallel-test", {"concurrency": 4})
        assert status == 400 and d["field"] == "set_local_url"
