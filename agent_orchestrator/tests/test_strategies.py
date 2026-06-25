"""Tests for the collaboration strategies (run against mock agents)."""

import os
import tempfile
import unittest

from agent_orchestrator.adapters.base import AgentAdapter
from agent_orchestrator.adapters.mock import MockAdapter
from agent_orchestrator.orchestrator.session import Session
from agent_orchestrator.orchestrator.strategies import (
    STRATEGIES,
    AgentTurnError,
    get_strategy,
)


class _FixedAdapter(AgentAdapter):
    """Returns a fixed body each turn (used to feed <FILE> blocks)."""

    kind = "fixed"

    def __init__(self, name, body):
        super().__init__(name, display_name=name)
        self.supports_history = False
        self._body = body

    def _generate(self, prompt, system, history):
        return self._body


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
    session = Session(
        id="test",
        task="Write a function that sorts a list.",
        strategy=strategy_name,
        rounds=rounds,
        agents=agents,
    )
    if strategy_name == "workspace_build":
        session.workspace = tempfile.mkdtemp()
    return session


class TestStrategies(unittest.TestCase):
    def test_all_strategies_registered(self):
        self.assertEqual(
            set(STRATEGIES),
            {"implementer_reviewer", "debate_consensus", "planner_executor",
             "round_robin", "panel_judge", "custom", "doc_authoring", "code_authoring",
             "workspace_build", "conductor_team"},
        )

    def test_conductor_directive_parsers(self):
        from agent_orchestrator.orchestrator.strategies import (
            _parse_assignments, _parse_assessments, _conductor_done,
        )
        text = (
            "@worker_1 [OK]: solid work\n"
            "@worker_2 [WARN]: ignored the assignment\n"
            "@worker_1: write the parser\n"
            "@worker_2: write the tests\n"
            "@stranger: ignore me\n"
            "VERDICT: DONE\n"
        )
        workers = ["worker_1", "worker_2"]
        self.assertEqual(
            _parse_assignments(text, workers),
            {"worker_1": "write the parser", "worker_2": "write the tests"},
        )
        self.assertEqual(
            _parse_assessments(text, workers),
            [("worker_1", "OK", "solid work"),
             ("worker_2", "WARN", "ignored the assignment")],
        )
        self.assertTrue(_conductor_done(text))
        self.assertFalse(_conductor_done("VERDICT: CONTINUE\n@worker_1: keep going"))

    def _conductor_session(self, conductor_body, rounds=2, workers=2):
        agents = {
            "conductor": _FixedAdapter("Maestro", conductor_body),
            "reviewer": _FixedAdapter("Critic", "Looks reasonable; minor gaps."),
        }
        order = ["conductor"]
        for i in range(1, workers + 1):
            agents[f"worker_{i}"] = _FixedAdapter(f"W{i}", f"worker {i} output")
            order.append(f"worker_{i}")
        order.append("reviewer")
        s = Session(id="ct", task="Build a thing.", strategy="conductor_team",
                    rounds=rounds, agents=agents)
        s.role_order = order
        return s

    def test_conductor_team_runs_and_tracks_workers(self):
        body = ("@worker_1 [OK]: good\n@worker_2 [WARN]: nothing delivered\n"
                "@worker_1: do part A\n@worker_2: do part B\nVERDICT: CONTINUE")
        session = self._conductor_session(body, rounds=2, workers=2)
        result = get_strategy("conductor_team").run(session)
        self.assertTrue(result)
        statuses = [e.data["status"] for e in session.bus.history if e.type == "worker_status"]
        for expected in ("assigned", "delivered", "warned", "ok"):
            self.assertIn(expected, statuses)
        self.assertIn("result", [e.type for e in session.bus.history])

    def test_conductor_done_stops_early(self):
        body = ("@worker_1 [OK]: done\n@worker_2 [OK]: done\n"
                "@worker_1: x\n@worker_2: y\nVERDICT: DONE")
        session = self._conductor_session(body, rounds=4, workers=2)
        get_strategy("conductor_team").run(session)
        # round 1 assigns+works; round 2 conductor says DONE -> no round-2 worker turns.
        worker_rounds = {t.round for t in session.transcript if t.role.startswith("worker")}
        self.assertEqual(worker_rounds, {1})

    def test_conductor_team_requires_full_team(self):
        agents = {"conductor": _FixedAdapter("c", "x"), "reviewer": _FixedAdapter("r", "x")}
        session = Session(id="ct", task="t", strategy="conductor_team", rounds=1, agents=agents)
        session.role_order = ["conductor", "reviewer"]
        with self.assertRaises(AgentTurnError):
            get_strategy("conductor_team").run(session)

    def test_extract_files(self):
        from agent_orchestrator.orchestrator.strategies import _extract_files
        files = _extract_files('x <FILE path="a/b.py">\nprint(1)\n</FILE> y')
        self.assertEqual(files, [("a/b.py", "print(1)\n")])

    def test_workspace_build_writes_files(self):
        d = tempfile.mkdtemp()
        impl = _FixedAdapter("impl", '<FILE path="hello.py">\nprint("hi")\n</FILE>')
        rev = _FixedAdapter("rev", "Looks good. APPROVE")
        session = Session(id="w", task="say hi", strategy="workspace_build", rounds=1,
                          agents={"implementer": impl, "reviewer": rev}, workspace=d)
        result = get_strategy("workspace_build").run(session)
        self.assertTrue(os.path.isfile(os.path.join(d, "hello.py")))
        self.assertEqual(session.workspace_files["hello.py"].status, "created")
        self.assertIn("workspace_edit", [e.type for e in session.bus.history])
        self.assertIn("hello.py", result)

    def test_workspace_build_requires_workspace(self):
        impl = _FixedAdapter("i", "")
        session = Session(id="w", task="t", strategy="workspace_build", rounds=1,
                          agents={"implementer": impl, "reviewer": impl})
        with self.assertRaises(AgentTurnError):
            get_strategy("workspace_build").run(session)

    def test_workspace_rejects_path_escape(self):
        from agent_orchestrator.orchestrator.strategies import _apply_workspace_edits
        d = tempfile.mkdtemp()
        impl = _FixedAdapter("i", "")
        session = Session(id="w", task="t", strategy="workspace_build", rounds=1,
                          agents={"implementer": impl, "reviewer": impl}, workspace=d)
        with self.assertRaises(AgentTurnError):
            _apply_workspace_edits(session, "implementer", 1, '<FILE path="../evil.py">x</FILE>')
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(d), "evil.py")))

    def test_extract_artifact(self):
        from agent_orchestrator.orchestrator.strategies import _extract_artifact
        self.assertEqual(_extract_artifact("pre <ARTIFACT>\nhello\n</ARTIFACT> post"), "hello")
        self.assertEqual(_extract_artifact("```python\nprint(1)\n```"), "print(1)")
        self.assertIsNone(_extract_artifact("just prose, no block"))

    def test_authoring_builds_artifact(self):
        for name in ("doc_authoring", "code_authoring"):
            with self.subTest(strategy=name):
                session = _session(name, rounds=1)
                result = get_strategy(name).run(session)
                self.assertTrue(session.artifact, "expected a non-empty artifact")
                self.assertEqual(result, session.artifact)  # deliverable is the artifact
                self.assertTrue(session.artifact_versions)
                self.assertIn("artifact", [e.type for e in session.bus.history])

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
        # custom and conductor_team need runtime-supplied roles.
        for name in (n for n in STRATEGIES if n not in ("custom", "conductor_team")):
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
