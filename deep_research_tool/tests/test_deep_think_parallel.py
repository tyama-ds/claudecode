"""
Tests for parallel DeepThink section processing.

DeepThink previously processed sections strictly one by one (each with
several serial LLM calls). Sections now run concurrently, with one
processor instance per section (the processor's validator/metrics are
stateful and not thread-safe).
"""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from deep_research_tool.config import create_config
from deep_research_tool.main import DeepResearchTool


def make_result(conclusion):
    return SimpleNamespace(
        processed_content=conclusion,
        is_valid=True,
        overall_confidence=0.9,
        metrics_summary={},
        consistency_result=None,
    )


class SlowProcessor:
    """Fake DeepThinkProcessor: records concurrency while sleeping."""

    lock = threading.Lock()
    active = 0
    max_active = 0
    instances = 0

    def __init__(self, llm_client=None, config=None, language="ja"):
        with SlowProcessor.lock:
            SlowProcessor.instances += 1

    def process(self, content, source_texts=None, **kwargs):
        with SlowProcessor.lock:
            SlowProcessor.active += 1
            SlowProcessor.max_active = max(SlowProcessor.max_active,
                                           SlowProcessor.active)
        time.sleep(0.15)
        with SlowProcessor.lock:
            SlowProcessor.active -= 1
        return make_result(f"結論: {content[:12]}")

    @classmethod
    def reset(cls):
        cls.active = 0
        cls.max_active = 0
        cls.instances = 0


def build_tool(**kw):
    config = create_config(provider="openai", openai_api_key="sk-test",
                           deep_think=True, **kw)
    tool = DeepResearchTool.__new__(DeepResearchTool)
    tool.config = config
    tool.llm_client = MagicMock()
    tool.stage_llm_clients = {}
    tool.search_client = MagicMock()
    tool._thinking_config = MagicMock()
    tool.deep_think_processor = SlowProcessor()
    return tool


def make_session(n_sections):
    session = MagicMock()
    session.section_contents = {
        str(i): {"title": f"章{i}", "content": f"セクション{i}の本文です。" * 5}
        for i in range(1, n_sections + 1)
    }
    return session


def make_locker():
    locker = MagicMock()
    ev = MagicMock()
    ev.id = "EV-1"
    ev.content_excerpt = "エビデンス本文"
    locker.get_all_evidence.return_value = [ev]
    return locker


class TestDeepThinkParallel(unittest.TestCase):

    def setUp(self):
        SlowProcessor.reset()

    def _run(self, tool, n=4):
        session = make_session(n)
        with patch("deep_research_tool.main.DeepThinkProcessor",
                   new=SlowProcessor):
            t0 = time.time()
            session, results = tool._apply_deep_think(
                session=session, evidence_locker=make_locker())
            elapsed = time.time() - t0
        return session, results, elapsed

    def test_sections_run_concurrently(self):
        tool = build_tool(deep_think_max_workers=4)
        session, results, elapsed = self._run(tool, n=4)

        # 4 sections x 0.15s: sequential would take >= 0.6s
        self.assertLess(elapsed, 0.45, f"not parallel: {elapsed:.2f}s")
        self.assertGreaterEqual(SlowProcessor.max_active, 2)
        # every section got processed and its conclusion appended
        self.assertEqual(len(results), 4)
        for sid, sdata in session.section_contents.items():
            self.assertIn("結論:", sdata["content"])
            self.assertIn("deep_think", sdata)

    def test_one_processor_instance_per_section(self):
        """The processor is stateful: parallel sections must never share
        one instance."""
        tool = build_tool(deep_think_max_workers=4)
        self._run(tool, n=4)
        # 1 from tool init + 4 per-section instances
        self.assertGreaterEqual(SlowProcessor.instances, 5)

    def test_single_worker_stays_sequential(self):
        tool = build_tool(deep_think_max_workers=1)
        _, results, elapsed = self._run(tool, n=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(SlowProcessor.max_active, 1)   # truly sequential

    def test_failing_section_does_not_kill_others(self):
        tool = build_tool(deep_think_max_workers=4)

        class FlakyProcessor(SlowProcessor):
            def process(self, content, source_texts=None, **kwargs):
                if "セクション2" in content:
                    raise RuntimeError("LLM error")
                return super().process(content, source_texts)

        session = make_session(3)
        with patch("deep_research_tool.main.DeepThinkProcessor",
                   new=FlakyProcessor):
            session, results = tool._apply_deep_think(
                session=session, evidence_locker=make_locker())

        self.assertEqual(set(results), {"1", "3"})     # 2 failed, others OK
        # the failed section keeps its original text
        self.assertNotIn("結論:", session.section_contents["2"]["content"])

    def test_config_wiring(self):
        config = create_config(provider="openai", openai_api_key="sk-test",
                               deep_think=True, deep_think_max_workers=6)
        self.assertEqual(config.deep_think.max_workers, 6)
        from deep_research_tool.webui.server import (
            _CONFIG_PARAM_MAP, build_config_kwargs)
        self.assertIn("deep_think_max_workers", _CONFIG_PARAM_MAP)
        kwargs = build_config_kwargs({"deep_think_max_workers": 6})
        self.assertEqual(kwargs["deep_think_max_workers"], 6)


if __name__ == "__main__":
    unittest.main()
