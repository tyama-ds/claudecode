"""Tests for the collaboration strategies (run against mock agents)."""

import unittest

from agent_orchestrator.adapters.base import AgentAdapter
from agent_orchestrator.adapters.mock import MockAdapter
from agent_orchestrator.orchestrator.session import Session
from agent_orchestrator.orchestrator.strategies import STRATEGIES, get_strategy


class _CapturingAdapter(AgentAdapter):
    """Records the prompt + history each turn received, for assertions."""

    kind = "capture"

    def __init__(self, name, supports_history):
        super().__init__(name, display_name=name)
        self.supports_history = supports_history
        self.calls = []

    def _generate(self, prompt, system, history):
        self.calls.append((prompt, history))
        return f"{self.display_name} contributes."


def _custom_session(a, b):
    s = Session(id="h", task="Topic.", strategy="custom", rounds=1,
                agents={"agent_1": a, "agent_2": b})
    s.role_order = ["agent_1", "agent_2"]
    return s


def _session(strategy_name, rounds=2):
    strategy = get_strategy(strategy_name)
    agents = {key: MockAdapter(name=key, display_name=key) for key, _ in strategy.roles}
    return Session(
        id="test",
        task="Write a function that sorts a list.",
        strategy=strategy_name,
        rounds=rounds,
        agents=agents,
    )


class TestStrategies(unittest.TestCase):
    def test_all_strategies_registered(self):
        self.assertEqual(
            set(STRATEGIES),
            {"implementer_reviewer", "debate_consensus", "planner_executor",
             "round_robin", "panel_judge", "custom"},
        )

    def test_scratchpad_absorbs_note_lines(self):
        s = _session("round_robin", rounds=1)
        added = s.absorb_notes(
            "Agent A",
            "Here is my take.\nNOTE: use a set for O(1) membership.\n"
            "- NOTE: open question — persist sessions to disk?",
        )
        self.assertEqual(added, 2)
        view = s.scratchpad_view()
        self.assertEqual(view[0]["author"], "Agent A")
        self.assertIn("set", view[0]["text"])
        # Re-stating an identical note does not duplicate it.
        self.assertEqual(s.absorb_notes("Agent B", "NOTE: use a set for O(1) membership."), 0)

    def test_custom_strategy_with_overrides(self):
        strategy = get_strategy("custom")
        agents = {
            "agent_1": MockAdapter(name="agent_1", display_name="One"),
            "agent_2": MockAdapter(name="agent_2", display_name="Two"),
        }
        session = Session(id="t", task="Pick a sort.", strategy="custom", rounds=1, agents=agents)
        session.role_order = ["agent_1", "agent_2"]
        session.personas = {"agent_1": "You are the reviewer."}  # per-role persona override
        result = strategy.run(session)
        self.assertTrue(result)
        roles_seen = [t.role for t in session.transcript]
        self.assertIn("closer", roles_seen)
        # The persona override routed agent_1 into the mock's reviewer behaviour.
        first = next(t for t in session.transcript if t.role == "agent_1")
        self.assertIn("APPROVE", first.content.upper())

    def test_history_passed_when_supported(self):
        a = _CapturingAdapter("A", supports_history=True)
        b = _CapturingAdapter("B", supports_history=True)
        get_strategy("custom").run(_custom_session(a, b))
        # B speaks after A, so B's turn must receive a structured history.
        prompt_b, hist_b = b.calls[0]
        self.assertIsInstance(hist_b, list)
        self.assertTrue(hist_b, "expected a non-empty history")
        self.assertTrue(any("A [agent_1]" in m.content for m in hist_b))
        self.assertNotIn("Conversation so far", prompt_b)  # lean prompt

    def test_history_falls_back_when_unsupported(self):
        a = _CapturingAdapter("A", supports_history=False)
        b = _CapturingAdapter("B", supports_history=False)
        get_strategy("custom").run(_custom_session(a, b))
        prompt_b, hist_b = b.calls[0]
        self.assertFalse(hist_b)  # no native history (empty)
        self.assertIn("Conversation so far", prompt_b)  # transcript embedded instead

    def test_each_strategy_produces_transcript_and_result(self):
        for name in (n for n in STRATEGIES if n != "custom"):  # custom needs runtime roles
            with self.subTest(strategy=name):
                session = _session(name, rounds=2)
                result = get_strategy(name).run(session)
                self.assertTrue(result, "expected a non-empty deliverable")
                self.assertTrue(session.transcript, "expected recorded turns")
                # A result event was emitted on the bus.
                types = [e.type for e in session.bus.history]
                self.assertIn("result", types)
                self.assertIn("turn_end", types)

    def test_rounds_drive_turn_count(self):
        # Implementer+Reviewer: 2 turns per round (no early approval from mock,
        # whose reviewer text contains 'approve' -> may break early). Use the
        # planner_executor strategy which has predictable turn counts instead.
        session = _session("debate_consensus", rounds=3)
        get_strategy("debate_consensus").run(session)
        # openings (2) + (rounds-1)*2 rebuttals + 1 synthesis
        roles_seen = [t.role for t in session.transcript]
        self.assertIn("synthesizer", roles_seen)
        self.assertGreaterEqual(len(session.transcript), 5)


if __name__ == "__main__":
    unittest.main()
