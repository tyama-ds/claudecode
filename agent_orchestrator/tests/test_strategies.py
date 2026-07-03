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
             "workspace_build", "conductor_team", "org_team"},
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

    def test_load_references_filters_and_truncates(self):
        from agent_orchestrator.orchestrator.strategies import load_references
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, ".git"))
        os.makedirs(os.path.join(d, "sub"))
        with open(os.path.join(d, "a.txt"), "w") as fh:
            fh.write("hello reference")
        with open(os.path.join(d, "sub", "b.py"), "w") as fh:
            fh.write("print(1)\n")
        with open(os.path.join(d, ".secret"), "w") as fh:  # dotfile -> skipped
            fh.write("nope")
        with open(os.path.join(d, ".git", "config"), "w") as fh:  # in .git -> skipped
            fh.write("nope")
        with open(os.path.join(d, "bin.dat"), "wb") as fh:  # binary -> skipped
            fh.write(b"\x00\x01\x02data")
        with open(os.path.join(d, "big.txt"), "w") as fh:
            fh.write("x" * 100)
        refs = load_references(d, max_file_bytes=10)
        paths = {p for p, _ in refs}
        self.assertIn("a.txt", paths)
        self.assertIn(os.path.join("sub", "b.py"), paths)
        self.assertNotIn(".secret", paths)
        self.assertNotIn("bin.dat", paths)
        self.assertFalse(any(p.startswith(".git") for p in paths))
        big = dict(refs)["big.txt"]
        self.assertIn("truncated", big)
        self.assertLessEqual(len(big), 10 + len("\n… (truncated)"))

    def test_load_references_respects_max_files(self):
        from agent_orchestrator.orchestrator.strategies import load_references
        d = tempfile.mkdtemp()
        for i in range(10):
            with open(os.path.join(d, f"f{i}.txt"), "w") as fh:
                fh.write("data")
        self.assertEqual(len(load_references(d, max_files=3)), 3)

    def test_reference_block_injected_into_system(self):
        from agent_orchestrator.orchestrator.strategies import _scratchpad_system
        session = Session(id="r", task="t", strategy="round_robin", rounds=1, agents={})
        session.reference_dir = "/some/dir"
        session.references = [("notes.md", "IMPORTANT CONTEXT")]
        sysprompt = _scratchpad_system(session, "You are an agent.")
        self.assertIn("REFERENCE FILES", sysprompt)
        self.assertIn("notes.md", sysprompt)
        self.assertIn("IMPORTANT CONTEXT", sysprompt)
        # No reference dir -> no reference block.
        empty = Session(id="r2", task="t", strategy="round_robin", rounds=1, agents={})
        self.assertNotIn("REFERENCE FILES", _scratchpad_system(empty, "You are an agent."))

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

    def test_workspace_build_has_design_phase(self):
        """The first two turns are a round-0 design consultation, and <FILE>
        blocks emitted during it are NOT written to disk."""
        d = tempfile.mkdtemp()
        impl = _FixedAdapter("impl", '<FILE path="hello.py">\nprint("hi")\n</FILE>')
        rev = _FixedAdapter("rev", "Plan looks fine. APPROVE")
        session = Session(id="w", task="say hi", strategy="workspace_build", rounds=1,
                          agents={"implementer": impl, "reviewer": rev}, workspace=d)
        get_strategy("workspace_build").run(session)
        self.assertEqual([t.round for t in session.transcript[:2]], [0, 0])
        self.assertEqual([t.role for t in session.transcript[:2]],
                         ["implementer", "reviewer"])
        # the file lands in round 1 (build), not round 0 (consult)
        self.assertEqual(session.workspace_files["hello.py"].round, 1)

    def test_workspace_build_reviewer_can_fix_directly(self):
        d = tempfile.mkdtemp()
        impl = _FixedAdapter("impl", '<FILE path="hello.py">\nprint("hi")\n</FILE>')
        rev = _FixedAdapter(
            "rev", 'Nit fixed myself. <FILE path="hello.py">\nprint("hello")\n</FILE> APPROVE')
        session = Session(id="w", task="say hi", strategy="workspace_build", rounds=1,
                          agents={"implementer": impl, "reviewer": rev}, workspace=d)
        get_strategy("workspace_build").run(session)
        with open(os.path.join(d, "hello.py")) as fh:
            self.assertEqual(fh.read(), 'print("hello")\n')
        self.assertEqual(session.workspace_files["hello.py"].role, "reviewer")

    def test_workspace_build_multiple_reviewers_all_must_approve(self):
        """With several reviewers, the loop only stops early when every one of
        them approves; each reviewer gets design + review turns."""
        d = tempfile.mkdtemp()
        agents = {
            "implementer": _FixedAdapter("impl", '<FILE path="a.py">\nx = 1\n</FILE>'),
            "reviewer_1": _FixedAdapter("rev1", "Fine by me. APPROVE"),
            "reviewer_2": _FixedAdapter("rev2", "Not there yet. REQUEST CHANGES"),
        }
        session = Session(id="mr", task="t", strategy="workspace_build", rounds=2,
                          agents=agents, workspace=d)
        session.role_order = ["implementer", "reviewer_1", "reviewer_2"]
        get_strategy("workspace_build").run(session)
        # 3 design turns + 2 full build rounds of (impl + 2 reviews) = 9 turns
        self.assertEqual(len(session.transcript), 9)
        # reviewers run in parallel, so compare round-0 roles as a set
        roles_r0 = {t.role for t in session.transcript if t.round == 0}
        self.assertEqual(roles_r0, {"implementer", "reviewer_1", "reviewer_2"})
        msgs = [e.data["message"] for e in session.bus.history if e.type == "status"]
        self.assertTrue(any("1/2 reviewer(s) approved" in m for m in msgs))

        # unanimous approval stops after round 1
        agents["reviewer_2"] = _FixedAdapter("rev2", "Great. APPROVE")
        session2 = Session(id="mr2", task="t", strategy="workspace_build", rounds=2,
                           agents=agents, workspace=tempfile.mkdtemp())
        session2.role_order = ["implementer", "reviewer_1", "reviewer_2"]
        get_strategy("workspace_build").run(session2)
        self.assertEqual(len(session2.transcript), 6)  # design 3 + one build round

    def test_workspace_build_turns_carry_actions(self):
        d = tempfile.mkdtemp()
        impl = _FixedAdapter("impl", '<FILE path="a.py">\nx = 1\n</FILE>')
        rev = _FixedAdapter("rev", "APPROVE")
        session = Session(id="act", task="t", strategy="workspace_build", rounds=1,
                          agents={"implementer": impl, "reviewer": rev}, workspace=d)
        get_strategy("workspace_build").run(session)
        actions = [(e.data["round"], e.data["action"])
                   for e in session.bus.history if e.type == "turn_start"]
        self.assertEqual(actions, [(0, "design"), (0, "design"),
                                   (1, "implement"), (1, "review")])

    def test_org_team_hierarchy_runs_and_reports(self):
        """manager -> 2 conductors -> workers: delegation flows down level by
        level and reports come back up; top VERDICT: DONE stops early."""
        agents = {
            "manager": _FixedAdapter(
                "Boss",
                "@conductor_1 [OK]: good\n@conductor_2 [OK]: good\n"
                "@conductor_1: build the backend\n@conductor_2: build the frontend\n"
                "VERDICT: DONE"),
            "conductor_1": _FixedAdapter("C1", "@worker_1: write the API"),
            "conductor_2": _FixedAdapter("C2", "@worker_2: write the UI"),
            "worker_1": _FixedAdapter("W1", "api done"),
            "worker_2": _FixedAdapter("W2", "ui done"),
        }
        s = Session(id="org", task="Build an app.", strategy="org_team", rounds=3,
                    agents=agents)
        s.role_order = list(agents)
        s.supervisors = {"conductor_1": "manager", "conductor_2": "manager",
                         "worker_1": "conductor_1", "worker_2": "conductor_2"}
        result = get_strategy("org_team").run(s)
        self.assertTrue(result)
        roles = [t.role for t in s.transcript]
        self.assertEqual(roles[0], "manager")                    # top assigns first
        self.assertIn("worker_1", roles)
        self.assertIn("worker_2", roles)
        # round 2's manager turn sees DONE -> early stop (round 1 ran in full)
        self.assertLessEqual(max(t.round for t in s.transcript), 2)
        ws = [e.data for e in s.bus.history if e.type == "worker_status"]
        self.assertTrue(any(d.get("by") == "conductor_1" and d["worker"] == "worker_1"
                            and d["status"] == "assigned" for d in ws))
        self.assertTrue(any(d.get("by") == "manager" and d["worker"] == "conductor_1"
                            for d in ws))

    def test_org_team_rejects_bad_hierarchy(self):
        agents = {"a": _FixedAdapter("A", "x"), "b": _FixedAdapter("B", "y")}
        s = Session(id="bad", task="t", strategy="org_team", rounds=1, agents=agents)
        s.role_order = ["a", "b"]
        s.supervisors = {"a": "b", "b": "a"}  # cycle -> no top manager
        with self.assertRaises(AgentTurnError):
            get_strategy("org_team").run(s)

    def test_unlimited_rounds_run_until_done(self):
        """rounds=0 loops past the old cap until the conductor declares DONE."""
        class DoneAtRound(AgentTurnError):
            pass

        class CountingConductor(AgentAdapter):
            kind = "fixed"
            def __init__(self):
                super().__init__("cond", display_name="Cond")
                self.supports_history = False
                self.calls = 0
            def _generate(self, prompt, system, history):
                self.calls += 1
                verdict = "VERDICT: DONE" if self.calls >= 12 else "VERDICT: CONTINUE"
                return f"@worker_1 [OK]: fine\n@worker_1: keep going\n{verdict}"

        agents = {"conductor": CountingConductor(),
                  "worker_1": _FixedAdapter("W1", "done bit"),
                  "reviewer": _FixedAdapter("R", "ok")}
        s = Session(id="unl", task="t", strategy="conductor_team", rounds=0, agents=agents)
        s.role_order = ["conductor", "worker_1", "reviewer"]
        get_strategy("conductor_team").run(s)
        self.assertGreater(max(t.round for t in s.transcript), 8)  # beyond old cap

    def test_finish_request_wraps_up_gracefully(self):
        """finish_requested stops the loop but still produces a deliverable."""
        class FinishAfterFirstRound(AgentAdapter):
            kind = "fixed"
            def __init__(self, session_ref):
                super().__init__("impl", display_name="impl")
                self.supports_history = False
                self.session_ref = session_ref
                self.calls = 0
            def _generate(self, prompt, system, history):
                self.calls += 1
                if self.calls == 1:          # design turn: plan only
                    return "Plan: one file."
                # implement turn: deliver, then the human presses Finish
                self.session_ref[0].finish_requested = True
                return '<FILE path="f.py">\nx = 1\n</FILE>'

        ref = [None]
        impl = FinishAfterFirstRound(ref)
        rev = _FixedAdapter("rev", "REQUEST CHANGES")  # would never approve
        s = Session(id="fin", task="t", strategy="workspace_build", rounds=0,
                    agents={"implementer": impl, "reviewer": rev},
                    workspace=tempfile.mkdtemp())
        ref[0] = s
        result = get_strategy("workspace_build").run(s)
        self.assertIn("f.py", result)                       # deliverable produced
        self.assertLessEqual(max(t.round for t in s.transcript), 1)
        self.assertIn("result", [e.type for e in s.bus.history])

    def test_implementer_continue_gets_extra_turns(self):
        class ContinuingImpl(AgentAdapter):
            kind = "fixed"
            def __init__(self):
                super().__init__("impl", display_name="impl")
                self.supports_history = False
                self.calls = 0
            def _generate(self, prompt, system, history):
                self.calls += 1
                if self.calls == 1:          # design turn: plan only
                    return "Plan: two files."
                if self.calls == 2:
                    return '<FILE path="one.py">\na = 1\n</FILE>\nCONTINUE'
                return '<FILE path="two.py">\nb = 2\n</FILE>\nAll done.'

        impl = ContinuingImpl()
        rev = _FixedAdapter("rev", "APPROVE")
        d = tempfile.mkdtemp()
        s = Session(id="cont", task="t", strategy="workspace_build", rounds=1,
                    agents={"implementer": impl, "reviewer": rev}, workspace=d)
        get_strategy("workspace_build").run(s)
        self.assertTrue(os.path.isfile(os.path.join(d, "one.py")))
        self.assertTrue(os.path.isfile(os.path.join(d, "two.py")))
        impl_turns = [t for t in s.transcript if t.role == "implementer" and t.round == 1]
        self.assertEqual(len(impl_turns), 2)  # initial + one continuation

    def test_tool_calls_execute_and_feed_back(self):
        from agent_orchestrator.orchestrator.strategies import _run_turn

        class ToolUser(AgentAdapter):
            kind = "fixed"
            def __init__(self):
                super().__init__("tu", display_name="tu")
                self.supports_history = False
                self.prompts = []
            def _generate(self, prompt, system, history):
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    return 'Let me check. <TOOL name="read_file">notes.txt</TOOL>'
                return "Final answer using the file."

        d = tempfile.mkdtemp()
        with open(os.path.join(d, "notes.txt"), "w") as fh:
            fh.write("SECRET-42")
        agent = ToolUser()
        s = Session(id="tool", task="t", strategy="custom", rounds=1,
                    agents={"agent_1": agent}, workspace=d)
        s.tools = ["read_file"]
        out = _run_turn(s, "agent_1", "You are an agent.", "Answer.", 1)
        self.assertEqual(out, "Final answer using the file.")
        self.assertIn("SECRET-42", agent.prompts[1])         # result fed back
        self.assertIn("tool_use", [e.type for e in s.bus.history])

    def test_workspace_context_included_for_existing_files(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "existing.py"), "w") as fh:
            fh.write("legacy = True\n")
        impl = _CapturingAdapter("impl", supports_history=False)
        rev = _FixedAdapter("rev", "APPROVE")
        s = Session(id="ctx", task="t", strategy="workspace_build", rounds=1,
                    agents={"implementer": impl, "reviewer": rev}, workspace=d)
        get_strategy("workspace_build").run(s)
        design_prompt = impl.calls[0][0]
        self.assertIn("EXISTING WORKSPACE FILES", design_prompt)
        self.assertIn("legacy = True", design_prompt)

    def test_snapshot_and_detect_native_edits(self):
        """Files changed directly on disk (a CLI editing natively) are detected
        by diffing tree snapshots and attributed to the acting role."""
        from agent_orchestrator.orchestrator.strategies import (
            _detect_native_edits, _snapshot_workspace,
        )
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "keep.txt"), "w") as fh:
            fh.write("old\n")
        session = Session(id="n", task="t", strategy="workspace_build", rounds=1,
                          agents={"implementer": _FixedAdapter("i", "")}, workspace=d)
        baseline = _snapshot_workspace(d)
        # simulate a native CLI turn: modify, create, and delete files on disk
        with open(os.path.join(d, "keep.txt"), "w") as fh:
            fh.write("new\n")
        with open(os.path.join(d, "made.py"), "w") as fh:
            fh.write("x = 1\n")
        changed, after = _detect_native_edits(session, "implementer", 2, baseline)
        self.assertEqual(changed, 2)
        self.assertEqual(session.workspace_files["keep.txt"].status, "modified")
        self.assertEqual(session.workspace_files["made.py"].status, "created")
        os.remove(os.path.join(d, "made.py"))
        changed, _ = _detect_native_edits(session, "implementer", 3, after)
        self.assertEqual(changed, 1)
        self.assertEqual(session.workspace_files["made.py"].status, "deleted")

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
        # custom, conductor_team, and org_team need runtime-supplied roles.
        for name in (n for n in STRATEGIES if n not in ("custom", "conductor_team", "org_team")):
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
