"""Offline protocol and loopback HTTP checks for local inference transport."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from deep_research_tool.api.base import TokenUsageStats
from deep_research_tool.api.local_client import LocalLLMClient
from deep_research_tool.utils.concurrency import ConcurrencyLimiter, RunLimits


def sse_event(content=None, finish=None, **extra):
    choice = {"index": 0, "delta": {}}
    if content is not None:
        choice["delta"]["content"] = content
    if finish is not None:
        choice["finish_reason"] = finish
    return "data: " + json.dumps({"choices": [choice], **extra}, ensure_ascii=False)


def complete_sse(content="ok", finish="stop"):
    return [sse_event(content), "", sse_event(finish=finish), "", "data: [DONE]", ""]


class FakeResponse:
    def __init__(self, lines=(), *, status=200, content_type="text/event-stream", data=None,
                 on_read=None):
        self.lines = lines
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        self.data = data
        self.closed = False
        self.on_read = on_read

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_lines(self, **kwargs):
        if self.on_read:
            self.on_read()
        for line in self.lines:
            if isinstance(line, Exception):
                raise line
            yield line.encode("utf-8") if isinstance(line, str) else line

    def json(self):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.headers = {}
        self.lock = threading.Lock()

    def post(self, url, **kwargs):
        with self.lock:
            self.calls.append({"url": url, **kwargs})
            outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(monkeypatch, outcomes, *, backend="openai_compatible", **kwargs):
    session = FakeSession(outcomes)
    monkeypatch.setattr(requests, "Session", lambda: session)
    # Explicit inert endpoint and credential: no host configuration is loaded.
    client = LocalLLMClient(model="test-model", backend=backend,
                            base_url="http://unused.invalid/v1" if backend != "ollama"
                            else "http://unused.invalid", api_key="test-only", **kwargs)
    return client, session


def test_sse_assembles_utf8_text_usage_and_finish_reason(monkeypatch):
    response = FakeResponse([
        ": heartbeat", "", "event: message",
        sse_event("日本", model="server-model", id="completion-1"), "",
        sse_event("語の計画"), "", sse_event(finish="stop"), "",
        'data: {"choices": [],',
        'data: "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}}',
        "", "data: [DONE]", "",
    ])
    client, session = make_client(monkeypatch, [response])
    client.token_stats = TokenUsageStats()
    result = client.generate("plan")
    assert result.content == "日本語の計画"
    assert result.model == "server-model"
    assert result.finish_reason == "stop"
    assert result.usage == {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
    assert client.token_stats.total_calls == 1
    assert client.token_stats.total_tokens == 12
    assert result.raw_response["id"] == "completion-1"
    assert session.calls[0]["json"]["stream"] is True
    assert session.calls[0]["json"]["stream_options"] == {"include_usage": True}
    assert session.calls[0]["stream"] is True
    assert session.calls[0]["timeout"] == (10, 600)
    assert response.closed


def test_ollama_assembles_final_stats_and_does_not_count_thinking_as_content(monkeypatch):
    response = FakeResponse([
        json.dumps({"message": {"thinking": "internal", "content": "前半"}, "done": False}),
        json.dumps({"message": {"content": "後半"}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop",
                    "eval_count": 8, "prompt_eval_count": 11}),
    ], content_type="application/x-ndjson")
    client, session = make_client(monkeypatch, [response], backend="ollama", timeout=0.2)
    result = client.generate("plan")
    assert result.content == "前半後半"
    assert result.usage["total_tokens"] == 19
    assert result.finish_reason == "stop"
    assert result.raw_response["done"] is True
    assert session.calls[0]["url"].endswith("/api/chat")
    assert session.calls[0]["timeout"] == (0.2, 0.2)
    assert response.closed


@pytest.mark.parametrize("backend", ["openai_compatible", "ollama"])
def test_complete_json_fallback_for_server_ignoring_stream(monkeypatch, backend):
    if backend == "ollama":
        data = {"message": {"content": "complete"}, "done": True, "done_reason": "stop"}
    else:
        data = {"choices": [{"message": {"content": "complete"}, "finish_reason": "stop"}]}
    response = FakeResponse(content_type="application/json; charset=utf-8", data=data)
    client, session = make_client(monkeypatch, [response], backend=backend)
    assert client.generate("plan").content == "complete"
    assert len(session.calls) == 1
    assert response.closed


@pytest.mark.parametrize("lines", [
    [sse_event('{"valid_json": true}', finish="stop"), ""],
    [sse_event("prefix"), "", "data: [DONE]", ""],
    [sse_event("prefix"), "", "data: {bad json}", ""],
    [sse_event("prefix"), "", 'data: {"error": {"message": "generation failed"}}', ""],
    [sse_event("prefix"), "", requests.exceptions.ChunkedEncodingError("disconnected")],
    [sse_event("prefix"), "", requests.ConnectionError("read timed out while streaming")],
    [sse_event("prefix"), "", requests.ConnectTimeout("timeout after response began")],
])
def test_incomplete_or_failed_sse_never_returns_prefix_or_retries(monkeypatch, lines):
    response = FakeResponse(lines)
    client, session = make_client(monkeypatch, [response])
    client.token_stats = TokenUsageStats()
    with pytest.raises(RuntimeError):
        client.generate("plan")
    assert len(session.calls) == 1
    assert response.closed
    assert client.token_stats.total_calls == 0
    assert client._local_sem.acquire(blocking=False)
    client._local_sem.release()


@pytest.mark.parametrize("lines", [
    ['{"message":{"content":"prefix"},"done":false}'],
    ['{"message":{"content":"prefix"},"done":false}', '{"error":"stopped"}'],
    ['{"message":{"content":"prefix"},"done":false}', '{"done":'],
])
def test_incomplete_ollama_never_returns_prefix_or_retries(monkeypatch, lines):
    response = FakeResponse(lines, content_type="application/x-ndjson")
    client, session = make_client(monkeypatch, [response], backend="ollama")
    with pytest.raises(RuntimeError):
        client.generate("plan")
    assert len(session.calls) == 1
    assert response.closed


@pytest.mark.parametrize("data", [
    {"choices": []},
    {"choices": [{"message": {"content": "prefix"}}]},
    json.JSONDecodeError("incomplete", '{"choices":', 11),
])
def test_incomplete_json_fallback_is_not_retried(monkeypatch, data):
    response = FakeResponse(content_type="application/json", data=data)
    client, session = make_client(monkeypatch, [response])
    with pytest.raises(RuntimeError):
        client.generate("plan")
    assert len(session.calls) == 1
    assert response.closed


def test_length_finish_is_reported_without_unbounded_generation_retry(monkeypatch):
    response = FakeResponse(complete_sse('{"incomplete":', finish="length"))
    client, session = make_client(monkeypatch, [response])
    result = client.generate("plan")
    assert result.finish_reason == "length"
    assert len(session.calls) == 1


@pytest.mark.parametrize("error", [requests.ReadTimeout("idle"), requests.ConnectionError("disconnect")])
def test_ambiguous_transport_failure_does_not_duplicate_inference(monkeypatch, error):
    client, session = make_client(monkeypatch, [error])
    with pytest.raises(RuntimeError, match="not retried"):
        client.generate("plan")
    assert len(session.calls) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 500])
def test_nonretryable_http_status_fails_once_and_closes(monkeypatch, status):
    response = FakeResponse(status=status)
    client, session = make_client(monkeypatch, [response])
    with pytest.raises(RuntimeError, match=str(status)):
        client.generate("plan")
    assert len(session.calls) == 1
    assert response.closed


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_explicit_transient_status_closes_before_backoff_and_retries(monkeypatch, status):
    rejected = FakeResponse(status=status)
    completed = FakeResponse(complete_sse())
    client, session = make_client(monkeypatch, [rejected, completed])
    sleeps = []

    def backoff(delay):
        assert rejected.closed
        assert client._local_sem.acquire(blocking=False)
        client._local_sem.release()
        sleeps.append(delay)

    monkeypatch.setattr(time, "sleep", backoff)
    assert client.generate("plan").content == "ok"
    assert len(session.calls) == 2
    assert len(sleeps) == 1
    assert completed.closed


def test_retry_budget_is_bounded_and_connection_timeout_can_retry(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    completed = FakeResponse(complete_sse())
    client, session = make_client(monkeypatch, [requests.ConnectTimeout("connect")] * 3 + [completed])
    assert client.generate("plan").content == "ok"
    assert len(session.calls) == 4
    responses = [FakeResponse(status=503) for _ in range(4)]
    client, session = make_client(monkeypatch, responses)
    with pytest.raises(RuntimeError, match="after 4 attempts"):
        client.generate("plan")
    assert len(session.calls) == 4
    assert all(response.closed for response in responses)


def test_cancellation_between_chunks_closes_and_releases_permit(monkeypatch):
    cancelled = False

    def chunks():
        nonlocal cancelled
        yield sse_event("prefix")
        yield ""
        cancelled = True
        yield sse_event("must not be returned")

    response = FakeResponse(chunks())
    client, session = make_client(monkeypatch, [response])

    def check():
        if cancelled:
            raise RuntimeError("cancelled test run")

    client.cancel_check = check
    with pytest.raises(RuntimeError, match="cancelled test run"):
        client.generate("plan")
    assert response.closed
    assert len(session.calls) == 1
    assert client._local_sem.acquire(blocking=False)
    client._local_sem.release()


@pytest.mark.parametrize("backend", ["openai_compatible", "ollama"])
@pytest.mark.parametrize("during_stream", [False, True])
def test_cancellation_preserves_run_cancelled_exception_type(monkeypatch, backend, during_stream):
    from deep_research_tool.main import RunCancelled

    cancelled = not during_stream

    def chunks():
        nonlocal cancelled
        if backend == "ollama":
            yield '{"message":{"content":"prefix"},"done":false}'
        else:
            yield sse_event("prefix")
            yield ""
        cancelled = True
        yield ""

    response = FakeResponse(chunks(), content_type=("application/x-ndjson"
                                                    if backend == "ollama" else "text/event-stream"))
    client, session = make_client(monkeypatch, [response], backend=backend)

    def check():
        if cancelled:
            raise RunCancelled("intentional stop")

    client.cancel_check = check
    with pytest.raises(RunCancelled, match="intentional stop"):
        client.generate("plan")
    assert len(session.calls) == int(during_stream)
    assert response.closed is during_stream
    assert client._local_sem.acquire(blocking=False)
    client._local_sem.release()


@pytest.mark.parametrize("use_run_limit", [False, True])
def test_permit_is_held_until_stream_consumption_finishes(monkeypatch, use_run_limit):
    reading = threading.Event()
    release = threading.Event()
    second_started = threading.Event()

    def block_stream():
        reading.set()
        assert release.wait(3)

    first_response = FakeResponse(complete_sse("first"), on_read=block_stream)
    second_response = FakeResponse(complete_sse("second"))
    client, session = make_client(monkeypatch, [first_response, second_response],
                                  **({"max_concurrency": 2} if use_run_limit else {}))
    if use_run_limit:
        client.concurrency_limiter = RunLimits(1, process_limiter=ConcurrencyLimiter(4))

    def second_request():
        second_started.set()
        return client.generate("second")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(client.generate, "first")
        try:
            assert reading.wait(2)
            second = executor.submit(second_request)
            assert second_started.wait(2)
            time.sleep(0.05)
            assert len(session.calls) == 1
            assert not first_response.closed
        finally:
            release.set()
        assert first.result(timeout=2).content == "first"
        assert second.result(timeout=2).content == "second"
    assert first_response.closed and second_response.closed


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), "600"])
def test_timeout_requires_positive_finite_number(monkeypatch, timeout):
    with pytest.raises(ValueError, match="timeout"):
        make_client(monkeypatch, [], timeout=timeout)


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, "2"])
def test_concurrency_requires_positive_integer(monkeypatch, cap):
    with pytest.raises(ValueError, match="concurrency"):
        make_client(monkeypatch, [], max_concurrency=cap)


@contextmanager
def streaming_http_server(protocol, *, stall=False):
    received = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if protocol == "sse"
                             else "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                if stall:
                    time.sleep(0.3)
                    return
                for _ in range(12):
                    if protocol == "sse":
                        payload = (sse_event("計画" * 30) + "\n\n").encode("utf-8")
                    else:
                        payload = (json.dumps({"message": {"content": "計画" * 30}, "done": False},
                                              ensure_ascii=False) + "\n").encode("utf-8")
                    self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.04)
                if protocol == "sse":
                    final = (sse_event(finish="stop") + "\n\ndata: [DONE]\n\n").encode()
                else:
                    final = b'{"message":{"content":""},"done":true,"done_reason":"stop"}\n'
                self.wfile.write(f"{len(final):x}\r\n".encode() + final + b"\r\n0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("protocol", ["sse", "ndjson"])
def test_real_http_stream_can_exceed_idle_timeout_in_total(monkeypatch, protocol):
    # Real requests/urllib3 transport, synthetic loopback server, no model calls.
    session = requests.Session()
    session.trust_env = False
    monkeypatch.setattr(requests, "Session", lambda: session)
    idle_timeout = 0.25
    try:
        with streaming_http_server(protocol) as (base_url, received):
            client = LocalLLMClient(model="test-model", api_key="test-only",
                                    backend="openai_compatible" if protocol == "sse" else "ollama",
                                    base_url=base_url, timeout=idle_timeout)
            started = time.monotonic()
            result = client.generate("synthetic plan")
            assert time.monotonic() - started > idle_timeout
            assert result.content == "計画" * 30 * 12
            assert result.finish_reason == "stop"
            assert len(received) == 1
            assert received[0]["stream"] is True
    finally:
        session.close()


def test_real_http_stalled_stream_times_out_without_second_post(monkeypatch):
    session = requests.Session()
    session.trust_env = False
    monkeypatch.setattr(requests, "Session", lambda: session)
    try:
        with streaming_http_server("sse", stall=True) as (base_url, received):
            client = LocalLLMClient(model="test-model", api_key="test-only",
                                    backend="openai_compatible", base_url=base_url, timeout=0.1)
            with pytest.raises(RuntimeError, match="not retried"):
                client.generate("synthetic plan")
            assert len(received) == 1
    finally:
        session.close()
