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

    def test_auto_loop_reworks_until_pass(self):
        """FAIL from the evaluator triggers an automatic rework with its
        feedback; PASS ends the loop; the original task is restored."""
        from agent_orchestrator.adapters.base import AgentAdapter

        class Evaluator(AgentAdapter):
            kind = "fixed"
            def __init__(self):
                super().__init__("eval", display_name="Judge")
                self.supports_history = False
                self.calls = 0
                self.prompts = []
            def _generate(self, prompt, system, history):
                self.calls += 1
                self.prompts.append(prompt)
                return "FAIL\nAdd error handling." if self.calls == 1 else "PASS\nSolid."

        strategy = get_strategy("implementer_reviewer")
        agents = {key: build_adapter({"id": "mock", "name": key}) for key, _ in strategy.roles}
        manager = SessionManager()
        session = manager.create("Reverse a string.", "implementer_reviewer", 1, agents)
        ev = Evaluator()
        session.loop_iters = 3
        session.loop_evaluator = ev
        run_session(session)

        self.assertEqual(session.status, "done")
        self.assertEqual(ev.calls, 2)                      # FAIL then PASS
        loops = [e.data for e in session.bus.history if e.type == "loop"]
        self.assertEqual([d["verdict"] for d in loops], ["fail", "pass"])
        self.assertIn("Add error handling.", loops[0]["feedback"])
        # the second strategy run saw the feedback in its task
        self.assertIn("AUTO-LOOP", ev.prompts[1])
        # two full strategy runs -> two result events
        self.assertEqual(sum(1 for e in session.bus.history if e.type == "result"), 2)
        # evaluator turns are visible in the stream
        eval_turns = [e for e in session.bus.history
                      if e.type == "turn_end" and e.data["role"] == "evaluator"]
        self.assertEqual(len(eval_turns), 2)
        self.assertEqual(session.task, "Reverse a string.")  # restored after the loop

    def test_auto_loop_gives_up_at_cap(self):
        from agent_orchestrator.adapters.base import AgentAdapter

        class AlwaysFail(AgentAdapter):
            kind = "fixed"
            def __init__(self):
                super().__init__("eval", display_name="Judge")
                self.supports_history = False
            def _generate(self, prompt, system, history):
                return "FAIL\nStill not good enough."

        strategy = get_strategy("implementer_reviewer")
        agents = {key: build_adapter({"id": "mock", "name": key}) for key, _ in strategy.roles}
        manager = SessionManager()
        session = manager.create("t", "implementer_reviewer", 1, agents)
        session.loop_iters = 2
        session.loop_evaluator = AlwaysFail()
        run_session(session)
        self.assertEqual(session.status, "done")           # still delivers
        loops = [e.data["verdict"] for e in session.bus.history if e.type == "loop"]
        self.assertEqual(loops, ["fail", "giveup"])

    def test_session_registered_and_retrievable(self):
        manager = SessionManager()
        strategy = get_strategy("planner_executor")
        agents = {key: build_adapter({"id": "mock", "name": key}) for key, _ in strategy.roles}
        session = manager.create("task", "planner_executor", 1, agents)
        self.assertIs(manager.get(session.id), session)
        self.assertIsNone(manager.get("nonexistent"))


if __name__ == "__main__":
    unittest.main()
