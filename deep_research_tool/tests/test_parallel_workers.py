"""
Tests for the application-wide parallelism controls (§ parallel).

Covers: strict validation of parallel_max_workers (no clamping), the
run-scope / process-scope ConcurrencyLimiter (leaf permits, measured
peaks, no permit leaks on exception/timeout), effective_workers, wiring
into Config / create_config / CLI / Web UI / GUI, per-job token-usage
isolation, and the Local LLM auth chain from both UIs down to the
Bearer header.
"""

import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from deep_research_tool.api.base import TokenUsage, TokenUsageStats, \
    get_token_stats, record_usage, reset_token_stats
from deep_research_tool.api.local_client import LocalLLMClient
from deep_research_tool.config import create_config
from deep_research_tool.gui_config import build_gui_config
from deep_research_tool.utils.concurrency import (
    PARALLEL_MAX_WORKERS_HARD_CAP,
    ConcurrencyLimiter,
    RunLimits,
    effective_workers,
    get_process_limiter,
    validate_parallel_max_workers,
)


class TestValidation(unittest.TestCase):

    def test_valid_values(self):
        for v in (1, 8, 16, "8", " 16 "):
            self.assertIsInstance(validate_parallel_max_workers(v), int)
        self.assertEqual(validate_parallel_max_workers(1), 1)
        self.assertEqual(validate_parallel_max_workers(16), 16)

    def test_invalid_values_rejected_not_clamped(self):
        for bad in (0, 17, -1, -8, 100, 8.0, 8.5, True, False,
                    "abc", "", "8.5", None, [8]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                validate_parallel_max_workers(bad)

    def test_create_config_strict(self):
        for bad in (0, 17, True, 2.5):
            with self.assertRaises(ValueError, msg=repr(bad)):
                create_config(provider="openai", openai_api_key="sk-t",
                              parallel_max_workers=bad)
        config = create_config(provider="openai", openai_api_key="sk-t",
                               parallel_max_workers=16)
        self.assertEqual(config.research.parallel_max_workers, 16)
        self.assertEqual(config.validate(), [])

    def test_webui_converter_rejects(self):
        from deep_research_tool.webui.server import build_config_kwargs
        self.assertEqual(
            build_config_kwargs({"parallel_max_workers": "8"})
            ["parallel_max_workers"], 8)
        for bad in ("0", "17", "-3", "8.5", "abc", 8.5, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                build_config_kwargs({"parallel_max_workers": bad})

    def test_cli_rejects_out_of_range(self):
        from click.testing import CliRunner
        from deep_research_tool.cli import cli
        res = CliRunner().invoke(cli, [
            "research", "テーマ", "--openai-key", "sk-t",
            "--parallel-max-workers", "17"], input="n\n")
        self.assertNotEqual(res.exit_code, 0)

    def test_gui_builder_validates(self):
        with self.assertRaises(ValueError):
            build_gui_config({"topic": "t", "provider": "openai",
                              "parallel_max_workers": 0})
        config = build_gui_config({"topic": "t", "provider": "openai",
                                   "parallel_max_workers": 16})
        self.assertEqual(config["parallel_max_workers"], 16)

    def test_effective_workers(self):
        self.assertEqual(effective_workers(8, 4, 10), 4)
        self.assertEqual(effective_workers(2, 8, 10), 2)
        self.assertEqual(effective_workers(8, 8, 3), 3)
        self.assertEqual(effective_workers(None, 5, None), 5)
        self.assertEqual(effective_workers(None, None, None), 1)


class TestConcurrencyLimiter(unittest.TestCase):

    def _hammer(self, limits, n_tasks=20, hold=0.03, pools=1,
                workers_per_pool=8):
        """Run n_tasks leaf operations through `pools` independent
        ThreadPools (simulating mixed stages) and return the peak."""
        def leaf(_):
            with limits.permit():
                time.sleep(hold)

        threads = []
        for _ in range(pools):
            t = threading.Thread(
                target=lambda: list(ThreadPoolExecutor(
                    max_workers=workers_per_pool).map(
                        leaf, range(n_tasks))))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    def test_max_workers_1_is_sequential(self):
        limits = RunLimits(1, process_limiter=ConcurrencyLimiter(16))
        self._hammer(limits, n_tasks=6)
        self.assertEqual(limits.run_limiter.peak, 1)

    def test_max_workers_8_really_parallel(self):
        limits = RunLimits(8, process_limiter=ConcurrencyLimiter(16))
        self._hammer(limits, n_tasks=24)
        self.assertGreaterEqual(limits.run_limiter.peak, 4)
        self.assertLessEqual(limits.run_limiter.peak, 8)

    def test_mixed_stages_never_exceed_run_cap(self):
        """DeepThink + crawl + chunk extraction + figures = several
        INDEPENDENT thread pools; their combined leaf concurrency stays
        within the single run cap."""
        limits = RunLimits(4, process_limiter=ConcurrencyLimiter(16))
        self._hammer(limits, n_tasks=10, pools=4, workers_per_pool=6)
        self.assertLessEqual(limits.run_limiter.peak, 4)
        self.assertGreaterEqual(limits.run_limiter.peak, 2)

    def test_three_jobs_never_exceed_process_cap(self):
        """Web UI: 3 parallel jobs (each with its own run cap) share ONE
        process limiter; total concurrency stays <= 16."""
        process = ConcurrencyLimiter(PARALLEL_MAX_WORKERS_HARD_CAP)
        jobs = [RunLimits(8, process_limiter=process) for _ in range(3)]
        threads = [threading.Thread(
            target=self._hammer, args=(job,), kwargs={"n_tasks": 16})
            for job in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertLessEqual(process.peak, 16)
        self.assertGreaterEqual(process.peak, 8)

    def test_process_limiter_is_shared_by_default(self):
        a, b = RunLimits(4), RunLimits(4)
        self.assertIs(a.process_limiter, b.process_limiter)
        self.assertIs(a.process_limiter, get_process_limiter())

    def test_no_permit_leak_on_exception(self):
        limiter = ConcurrencyLimiter(2)
        for _ in range(5):
            with self.assertRaises(RuntimeError):
                with limiter.permit():
                    raise RuntimeError("boom")
        self.assertEqual(limiter.active, 0)
        # all permits usable again
        with limiter.permit():
            with limiter.permit():
                self.assertEqual(limiter.active, 2)
        self.assertEqual(limiter.active, 0)

    def test_timeout_does_not_leak(self):
        limiter = ConcurrencyLimiter(1)
        release = threading.Event()

        def holder():
            with limiter.permit():
                release.wait(5)

        t = threading.Thread(target=holder)
        t.start()
        time.sleep(0.05)
        with self.assertRaises(TimeoutError):
            with limiter.permit(timeout=0.05):
                pass
        release.set()
        t.join()
        with limiter.permit(timeout=1):      # usable after timeout error
            self.assertEqual(limiter.active, 1)
        self.assertEqual(limiter.active, 0)


class TestTokenIsolation(unittest.TestCase):

    def test_run_local_stats_isolated_per_job(self):
        job_a = SimpleNamespace(token_stats=TokenUsageStats())
        job_b = SimpleNamespace(token_stats=TokenUsageStats())
        record_usage(job_a, TokenUsage(prompt_tokens=10,
                                       completion_tokens=5,
                                       total_tokens=15, model="m"))
        record_usage(job_b, TokenUsage(prompt_tokens=100,
                                       completion_tokens=50,
                                       total_tokens=150, model="m"))
        self.assertEqual(job_a.token_stats.total_tokens, 15)
        self.assertEqual(job_b.token_stats.total_tokens, 150)
        # one job resetting the GLOBAL stats never clears another job's
        reset_token_stats()
        self.assertEqual(get_token_stats().total_tokens, 0)
        self.assertEqual(job_a.token_stats.total_tokens, 15)
        self.assertEqual(job_b.token_stats.total_tokens, 150)

    def test_add_usage_thread_safe(self):
        stats = TokenUsageStats()
        usage = TokenUsage(prompt_tokens=1, completion_tokens=1,
                           total_tokens=2, model="m")
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda _: stats.add_usage(usage), range(400)))
        self.assertEqual(stats.total_calls, 400)
        self.assertEqual(stats.total_tokens, 800)


class FakeSession:
    """Stands in for requests.Session inside LocalLLMClient."""

    def __init__(self, tracker=None, status=200):
        self.headers = {}
        self.tracker = tracker
        self.status = status
        self.calls = 0

    def post(self, url, json=None, timeout=None):
        self.calls += 1
        if self.tracker is not None:
            self.tracker()
        time.sleep(0.02)
        return SimpleNamespace(
            status_code=self.status,
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "ok"},
                                       "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 1,
                                    "completion_tokens": 1,
                                    "total_tokens": 2}},
        )


class TestLocalClientConcurrency(unittest.TestCase):

    def _client(self, session_factory):
        client = LocalLLMClient(model="gpt-oss-20b",
                                backend="openai_compatible",
                                base_url="http://llm.local/v1",
                                api_key="tok-secret")
        client._build_session = session_factory
        client._thread_local = threading.local()
        return client

    def test_llm_leaf_calls_respect_run_limit(self):
        limits = RunLimits(2, process_limiter=ConcurrencyLimiter(16))
        active = {"n": 0, "peak": 0}
        lock = threading.Lock()

        def track():
            with lock:
                active["n"] += 1
                active["peak"] = max(active["peak"], active["n"])
            time.sleep(0.02)
            with lock:
                active["n"] -= 1

        client = self._client(lambda: FakeSession(tracker=track))
        client.concurrency_limiter = limits
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda _: client.generate("hi"), range(12)))
        self.assertLessEqual(limits.run_limiter.peak, 2)

    def test_thread_local_sessions_one_per_thread(self):
        sessions = []

        def factory():
            s = FakeSession()
            sessions.append(s)
            return s

        client = self._client(factory)
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(lambda _: client.generate("hi"), range(8)))
        # one session per WORKER THREAD, not one shared session
        self.assertGreaterEqual(len(sessions), 2)
        self.assertLessEqual(len(sessions), 5)   # 4 workers + main

    def test_retry_with_backoff_on_429(self):
        attempts = {"n": 0}

        class FlakySession(FakeSession):
            def post(self, url, json=None, timeout=None):
                attempts["n"] += 1
                if attempts["n"] < 3:
                    return SimpleNamespace(status_code=429,
                                           raise_for_status=lambda: None,
                                           json=lambda: {})
                return super().post(url, json=json, timeout=timeout)

        client = self._client(lambda: FlakySession())
        with patch("time.sleep"):        # skip the real backoff delays
            response = client.generate("hi")
        self.assertEqual(response.content, "ok")
        self.assertEqual(attempts["n"], 3)

    def test_api_key_never_in_error_text(self):
        class DeadSession(FakeSession):
            def post(self, url, json=None, timeout=None):
                raise RuntimeError("auth failed for Bearer tok-secret")

        client = self._client(lambda: DeadSession())
        with patch("time.sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                client.generate("hi")
        self.assertNotIn("tok-secret", str(ctx.exception))
        self.assertIn("***", str(ctx.exception))


class TestLocalLLMAuthChain(unittest.TestCase):

    def test_web_ui_to_bearer_header(self):
        from deep_research_tool.webui.server import build_config_kwargs
        kwargs = build_config_kwargs({
            "provider": "local",
            "local_base_url": "http://192.168.1.9/v1",
            "local_api_key": "web-token",
            "local_backend": "openai_compatible",
        })
        config = create_config(**kwargs)
        client = LocalLLMClient(
            model=config.api.local_model,
            backend=config.api.local_backend.value,
            base_url=config.api.local_base_url,
            api_key=config.api.get_active_api_key(),
        )
        self.assertEqual(client._session.headers.get("Authorization"),
                         "Bearer web-token")

    def test_gui_to_bearer_header(self):
        gui_config = build_gui_config({
            "topic": "t", "provider": "local",
            "local_api_key": "gui-token",
            "local_model": "gpt-oss-20b",
            "local_base_url": "http://10.0.0.5/v1",
            "local_backend": "openai_compatible",
            "parallel_max_workers": 8,
        })
        # local must NOT be treated as anthropic
        self.assertNotIn("anthropic_api_key", gui_config)
        self.assertEqual(gui_config["local_api_key"], "gui-token")
        self.assertEqual(gui_config["local_backend"], "openai_compatible")

        config = create_config(**{
            k: v for k, v in gui_config.items()
            if k not in ("topic",)})
        self.assertEqual(config.api.get_active_api_key(), "gui-token")
        client = LocalLLMClient(
            model=config.api.local_model,
            backend=config.api.local_backend.value,
            base_url=config.api.local_base_url,
            api_key=config.api.get_active_api_key(),
        )
        self.assertEqual(client._session.headers.get("Authorization"),
                         "Bearer gui-token")

    def test_local_without_api_key_works(self):
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("LOCAL_LLM_API_KEY", None)
            client = LocalLLMClient(model="llama3.1:8b", backend="ollama",
                                    base_url="http://localhost:11434")
            self.assertNotIn("Authorization", client._session.headers)

    def test_gui_local_key_optional_env_fallback(self):
        cfg = build_gui_config({"topic": "t", "provider": "local",
                                "local_api_key": "",
                                "parallel_max_workers": 8})
        # empty key omitted -> LocalLLMClient falls back to
        # LOCAL_LLM_API_KEY env var internally
        self.assertNotIn("local_api_key", cfg)

    def test_web_ui_secret_storage_is_opt_in(self):
        html = (__import__("pathlib").Path(__file__).parent.parent
                / "webui" / "static" / "index.html").read_text(
                    encoding="utf-8")
        # secrets are separated from non-secret settings and stored in
        # sessionStorage by default; localStorage only after explicit opt-in
        self.assertIn("SECRET_FIELDS", html)
        self.assertIn("sessionStorage", html)
        self.assertIn("set_persist_keys", html)
        self.assertIn("migrateLegacySecrets", html)
        # non-secret settings object no longer includes the API keys
        self.assertNotIn("set_openai_key: 'openai_api_key'",
                         html.split("SECRET_FIELDS")[0])


if __name__ == "__main__":
    unittest.main()
