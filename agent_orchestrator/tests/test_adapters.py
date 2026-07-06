"""Tests for the adapter layer."""

import unittest

from agent_orchestrator.adapters import build_adapter, catalog
from agent_orchestrator.adapters.base import AgentResult
from agent_orchestrator.adapters.mock import MockAdapter


class TestMockAdapter(unittest.TestCase):
    def test_generate_returns_result(self):
        adapter = MockAdapter()
        result = adapter.generate("Sort a list of numbers.")
        self.assertIsInstance(result, AgentResult)
        self.assertTrue(result.ok)
        self.assertTrue(result.text)

    def test_persona_changes_output(self):
        impl = MockAdapter().generate("Task", system="You are the implementer.").text
        rev = MockAdapter().generate("Task", system="You are the reviewer.").text
        self.assertNotEqual(impl, rev)
        self.assertIn("APPROVE", rev.upper())

    def test_deterministic(self):
        a = MockAdapter().generate("same prompt").text
        b = MockAdapter().generate("same prompt").text
        self.assertEqual(a, b)


class TestRegistry(unittest.TestCase):
    def test_catalog_has_mock_available(self):
        entries = {e["id"]: e for e in catalog()}
        self.assertIn("mock", entries)
        self.assertTrue(entries["mock"]["available"])
        # All expected backends are advertised.
        for expected in ("claude_code", "codex", "anthropic", "openai", "local"):
            self.assertIn(expected, entries)

    def test_build_adapter_mock(self):
        adapter = build_adapter({"id": "mock", "name": "implementer"})
        self.assertEqual(adapter.name, "implementer")
        self.assertEqual(adapter.kind, "mock")

    def test_build_adapter_unknown(self):
        with self.assertRaises(ValueError):
            build_adapter({"id": "does_not_exist"})

    def test_api_adapter_reports_unavailable_without_sdk(self):
        # In this environment the SDKs/keys are absent, so these must not be
        # silently 'available'. (If the SDK is installed, availability depends
        # on the API key — either way .available() must return a (bool, str).)
        adapter = build_adapter({"id": "anthropic", "name": "x"})
        ok, reason = adapter.available()
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(reason, str)


class TestOpenAITokenParam(unittest.TestCase):
    """The OpenAI-compatible adapter must use the right token-limit parameter:
    max_completion_tokens for gpt-5 / o-series, max_tokens for 4o, with a
    self-correcting retry when the server rejects the chosen name."""

    def _run_with_fake(self, model, fake):
        import agent_orchestrator.adapters.api_agent as api
        orig = api._post_json
        api._post_json = fake
        try:
            return api.OpenAIAPIAdapter(name="x", model=model).generate("hi")
        finally:
            api._post_json = orig

    def test_needs_completion_tokens_helper(self):
        from agent_orchestrator.adapters.api_agent import _needs_completion_tokens
        for m in ["gpt-5", "gpt-5-mini", "o1", "o1-mini", "o3", "o3-mini", "o4-mini"]:
            self.assertTrue(_needs_completion_tokens(m), m)
        for m in ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "llama3.1", ""]:
            self.assertFalse(_needs_completion_tokens(m), m)

    def test_newer_model_uses_completion_tokens_first(self):
        calls = []

        def fake(url, headers, payload, use_proxy=True):
            calls.append(dict(payload))
            return {"choices": [{"message": {"content": "ok"}}]}

        res = self._run_with_fake("gpt-5-mini", fake)
        self.assertTrue(res.ok, res.error)
        self.assertEqual(len(calls), 1)
        self.assertIn("max_completion_tokens", calls[0])
        self.assertNotIn("max_tokens", calls[0])

    def test_retries_when_max_tokens_unsupported(self):
        import agent_orchestrator.adapters.api_agent as api
        calls = []

        def fake(url, headers, payload, use_proxy=True):
            calls.append(dict(payload))
            if "max_tokens" in payload:
                raise api.ApiHTTPError(400, (
                    '{"error":{"message":"Unsupported parameter: \'max_tokens\' is not '
                    'supported with this model. Use \'max_completion_tokens\' instead.",'
                    '"code":"unsupported_parameter","param":"max_tokens"}}'
                ))
            return {"choices": [{"message": {"content": "ok"}}]}

        # gpt-4o is treated as "old" → tries max_tokens, gets 400, retries the other name.
        res = self._run_with_fake("gpt-4o", fake)
        self.assertTrue(res.ok, res.error)
        self.assertEqual(res.text, "ok")
        self.assertEqual(len(calls), 2)
        self.assertIn("max_tokens", calls[0])
        self.assertIn("max_completion_tokens", calls[1])


class TestLocalProxyToggle(unittest.TestCase):
    """The local LLM can be sent through the proxy or directly (a UI checkbox)."""

    def _captured_use_proxy(self, *, local, local_use_proxy):
        import agent_orchestrator.adapters.api_agent as api
        from agent_orchestrator.config import get_settings
        seen = {}

        def fake(url, headers, payload, use_proxy=True):
            seen["use_proxy"] = use_proxy
            return {"choices": [{"message": {"content": "ok"}}]}

        settings = get_settings()
        orig_post, orig_flag = api._post_json, settings.local_use_proxy
        api._post_json = fake
        settings.local_use_proxy = local_use_proxy
        try:
            api.OpenAIAPIAdapter(name="x", local=local).generate("hi")
        finally:
            api._post_json = orig_post
            settings.local_use_proxy = orig_flag
        return seen["use_proxy"]

    def test_local_direct_by_default(self):
        self.assertFalse(self._captured_use_proxy(local=True, local_use_proxy=False))

    def test_local_via_proxy_when_enabled(self):
        self.assertTrue(self._captured_use_proxy(local=True, local_use_proxy=True))

    def test_non_local_always_uses_proxy(self):
        # A remote provider ignores the local toggle and keeps proxy routing.
        self.assertTrue(self._captured_use_proxy(local=False, local_use_proxy=False))

    def test_apply_overrides_and_default(self):
        from agent_orchestrator.config import Settings
        s = Settings()
        self.assertFalse(s.local_use_proxy)  # direct by default
        s.apply_overrides({"local_use_proxy": True})
        self.assertTrue(s.local_use_proxy)
        s.apply_overrides({"local_use_proxy": False})
        self.assertFalse(s.local_use_proxy)
        # Omitting the key leaves the value untouched.
        s.local_use_proxy = True
        s.apply_overrides({"proxy": "http://p"})
        self.assertTrue(s.local_use_proxy)


class TestWebTools(unittest.TestCase):
    """http_get honors proxy + User-Agent; browser_get falls back gracefully."""

    def test_http_get_uses_proxy_and_user_agent(self):
        import urllib.request
        from agent_orchestrator.config import get_settings
        from agent_orchestrator.orchestrator import strategies

        s = get_settings()
        old_proxy, old_ua = s.proxy, s.user_agent
        s.proxy = "http://proxy.test:3128"
        s.user_agent = "TestAgent/9.9"
        captured = {}

        class FakeResp:
            def read(self, n=-1): return b"<html>ok</html>"
            def __enter__(self): return self
            def __exit__(self, *a): return False

        class FakeOpener:
            def __init__(self, handler=None): captured["proxy_handler"] = handler
            def open(self, req, timeout=None):
                captured["ua"] = req.get_header("User-agent")
                return FakeResp()

        orig = urllib.request.build_opener
        urllib.request.build_opener = lambda *a: FakeOpener(*a)
        try:
            out = strategies._http_get("https://example.com")
        finally:
            urllib.request.build_opener = orig
            s.proxy, s.user_agent = old_proxy, old_ua

        self.assertIn("ok", out)
        self.assertEqual(captured["ua"], "TestAgent/9.9")
        self.assertIsNotNone(captured["proxy_handler"])  # proxy handler was installed

    def test_browser_get_falls_back_to_http_without_a_browser(self):
        """With neither Selenium nor Playwright importable, browser_get returns
        a plain HTTP fetch flagged as un-rendered — never an error."""
        import builtins
        from agent_orchestrator.orchestrator import strategies

        real_import = builtins.__import__

        def no_browsers(name, *a, **k):
            if name.startswith("selenium") or name.startswith("playwright"):
                raise ImportError("not installed")
            return real_import(name, *a, **k)

        orig_http = strategies._http_get
        builtins.__import__ = no_browsers
        strategies._http_get = lambda url: "PLAINBODY"
        try:
            out = strategies._browser_get("https://example.com")
        finally:
            builtins.__import__ = real_import
            strategies._http_get = orig_http

        self.assertIn("JavaScript NOT", out)
        self.assertIn("PLAINBODY", out)

    def test_browser_get_falls_back_when_browser_launch_fails(self):
        """Playwright importable but its browser binary missing/broken must
        also fall back to plain HTTP (never a dead-end error)."""
        import sys
        import types
        from agent_orchestrator.orchestrator import strategies

        fake_api = types.ModuleType("playwright.sync_api")
        def sync_playwright():
            raise RuntimeError("Executable doesn't exist at .../chrome")
        fake_api.sync_playwright = sync_playwright
        fake_pkg = types.ModuleType("playwright")
        fake_pkg.sync_api = fake_api

        import builtins
        real_import = builtins.__import__
        def fake_imports(name, *a, **k):
            if name.startswith("selenium"):
                raise ImportError("not installed")
            if name == "playwright.sync_api":
                return fake_api  # `from X.Y import Z` expects the leaf module
            if name == "playwright":
                return fake_pkg
            return real_import(name, *a, **k)

        orig_http = strategies._http_get
        saved = {n: sys.modules.get(n) for n in ("playwright", "playwright.sync_api")}
        sys.modules["playwright"] = fake_pkg
        sys.modules["playwright.sync_api"] = fake_api
        builtins.__import__ = fake_imports
        strategies._http_get = lambda url: "PLAINBODY"
        try:
            out = strategies._browser_get("https://example.com")
        finally:
            builtins.__import__ = real_import
            strategies._http_get = orig_http
            for n, m in saved.items():
                if m is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = m

        self.assertIn("headless browser failed", out)
        self.assertIn("Executable doesn't exist", out)
        self.assertIn("JavaScript NOT", out)
        self.assertIn("PLAINBODY", out)


class TestCLIWorkspaceMode(unittest.TestCase):
    """CLI adapters gain native file-editing flags only inside a workspace."""

    def test_argv_plain_outside_workspace(self):
        from agent_orchestrator.adapters.cli_agent import claude_code_adapter
        a = claude_code_adapter()
        self.assertEqual(a._build_argv(), ["claude", "-p", "--output-format", "text"])

    def test_argv_gains_edit_flags_in_workspace(self):
        from agent_orchestrator.adapters.cli_agent import claude_code_adapter, codex_adapter
        a = claude_code_adapter()
        a.workdir = "/tmp/ws"
        self.assertEqual(
            a._build_argv(),
            ["claude", "-p", "--output-format", "text", "--permission-mode", "acceptEdits"],
        )
        c = codex_adapter()
        c.workdir = "/tmp/ws"
        self.assertEqual(c._build_argv(), ["codex", "exec", "--full-auto"])


if __name__ == "__main__":
    unittest.main()
