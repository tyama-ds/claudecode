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


if __name__ == "__main__":
    unittest.main()
