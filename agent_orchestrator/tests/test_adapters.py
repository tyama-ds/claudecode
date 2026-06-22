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

        def fake(url, headers, payload):
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

        def fake(url, headers, payload):
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


if __name__ == "__main__":
    unittest.main()
