"""End-to-end engine test using the in-memory session manager and mock agents."""

import unittest

from agent_orchestrator.adapters import build_adapter
from agent_orchestrator.orchestrator import SessionManager, run_session
from agent_orchestrator.orchestrator.strategies import get_strategy


class TestEngine(unittest.TestCase):
    def _run(self, strategy_name):
        strategy = get_strategy(strategy_name)
        agents = {key: build_adapter({"id": "mock", "name": key}) for key, _ in strategy.roles}
        manager = SessionManager()
        session = manager.create("Reverse a string.", strategy_name, 2, agents)
        q = session.bus.subscribe()
        run_session(session)  # synchronous

        events = []
        while not q.empty():
            item = q.get_nowait()
            if not session.bus.is_closed_sentinel(item):
                events.append(item)
        return session, events

    def test_run_session_completes(self):
        session, events = self._run("implementer_reviewer")
        self.assertEqual(session.status, "done")
        end = [e for e in events if e.type == "session_end"]
        self.assertTrue(end)
        self.assertEqual(end[-1].data["status"], "done")
        self.assertIn("result", end[-1].data)

    def test_session_registered_and_retrievable(self):
        manager = SessionManager()
        strategy = get_strategy("planner_executor")
        agents = {key: build_adapter({"id": "mock", "name": key}) for key, _ in strategy.roles}
        session = manager.create("task", "planner_executor", 1, agents)
        self.assertIs(manager.get(session.id), session)
        self.assertIsNone(manager.get("nonexistent"))


if __name__ == "__main__":
    unittest.main()
