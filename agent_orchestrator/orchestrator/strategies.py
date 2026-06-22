"""Collaboration strategies.

A strategy choreographs how the agents take turns. Each declares the *roles* it
needs (which the UI maps to a backend each) and implements :meth:`run`, driving
the session turn by turn and emitting events as it goes.

Conversation context is shared two ways, chosen per backend:

- **Native history** — backends that set ``supports_history`` (CLI / API) receive
  the running conversation as a structured ``history`` (a list of messages), with
  each agent's own turns as ``assistant`` and the others' as ``user``.
- **Text fallback** — backends that can't use history (e.g. the mock) instead get
  the transcript embedded as text in the prompt. The orchestrator picks
  automatically, so strategies only ever build a short per-turn *instruction*.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from ..adapters.base import Message
from .session import Session, Turn


class StopRequested(Exception):
    """Raised internally when the user asks to stop a running session."""


class AgentTurnError(Exception):
    """Raised when an agent turn fails, aborting the collaboration."""


def _scratchpad_system(session: Session, system: str) -> str:
    """Append the shared-scratchpad protocol + current contents to a system prompt."""
    return (
        f"{system}\n\n"
        "SHARED SCRATCHPAD — a team blackboard visible to every agent. To record a "
        "durable fact, decision, or open question for the others, add a line beginning "
        "with 'NOTE:' anywhere in your reply.\n"
        f"Current scratchpad:\n{session.render_scratchpad()}"
    )


def _build_history(session: Session, role: str) -> List[Message]:
    """Build a structured conversation history for ``role`` from the transcript.

    The first message is the task (a ``user`` turn, which also satisfies APIs that
    require the conversation to open with a user message). The agent's own prior
    turns are ``assistant``; everyone else's are ``user``, prefixed with who spoke.
    Returns ``[]`` before any turn has happened.
    """
    prior = [t for t in session.transcript if t.ok]
    if not prior:
        return []
    msgs = [Message(role="user", content=f"Discussion topic / task:\n{session.task}")]
    for t in prior:
        if t.role == role:
            msgs.append(Message(role="assistant", content=t.content))
        else:
            msgs.append(Message(role="user", content=f"{t.agent} [{t.role}]: {t.content}"))
    return msgs


def _render_transcript(session: Session) -> str:
    """Render the transcript as plain text (used for the no-history fallback)."""
    return "\n\n".join(
        f"{t.agent} [{t.role}]: {t.content}" for t in session.transcript if t.ok
    )


_ARTIFACT_RE = re.compile(r"<ARTIFACT>\s*(.*?)\s*</ARTIFACT>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```[\w.+-]*\n(.*?)```", re.DOTALL)


def _extract_artifact(text: str) -> Optional[str]:
    """Pull the artifact body out of an editing turn.

    Prefers an explicit ``<ARTIFACT>…</ARTIFACT>`` block; falls back to the first
    fenced code block. Returns ``None`` if neither is present.
    """
    m = _ARTIFACT_RE.search(text)
    if m:
        return m.group(1).strip()
    fm = _FENCE_RE.search(text)
    if fm:
        return fm.group(1).strip()
    return None


def _update_artifact(session: Session, role: str, rnd: int, text: str) -> None:
    """Save a new artifact version from an editing turn and notify the UI."""
    art = _extract_artifact(text)
    if art is None:  # editing roles should always produce content
        art = text.strip()
    name = session.agents[role].display_name
    version = session.set_artifact(art, name, role, rnd)
    session.emit("artifact", content=art, version=version,
                 author=name, role=role, round=rnd)


def _run_turn(session: Session, role: str, system: str, instruction: str, rnd: int) -> str:
    """Execute one agent turn: emit events, record it, return its text.

    ``instruction`` is the short per-turn directive; the conversation context is
    supplied automatically — as ``history`` when the backend supports it, or
    embedded in the prompt as a fallback when it doesn't.

    Raises :class:`StopRequested` on a stop request and :class:`AgentTurnError`
    if the backend fails.
    """
    if session.should_stop():
        raise StopRequested()

    adapter = session.agents[role]
    session.emit("turn_start", agent=adapter.display_name, role=role, round=rnd)

    # A per-role persona override from the UI wins over the strategy's default.
    system = session.personas.get(role) or system

    history = _build_history(session, role)
    use_history = bool(history) and getattr(adapter, "supports_history", True)
    if use_history:
        prompt = instruction
        hist: Optional[List[Message]] = history
    else:
        rendered = _render_transcript(session)
        prompt = f"Conversation so far:\n{rendered}\n\n{instruction}" if rendered else instruction
        hist = None

    result = adapter.generate(
        prompt, system=_scratchpad_system(session, system), history=hist
    )

    turn = Turn(
        agent=adapter.display_name,
        role=role,
        round=rnd,
        content=result.text if result.ok else (result.error or "unknown error"),
        ok=result.ok,
        error=result.error,
        duration=result.duration,
    )
    session.record(turn)
    session.emit(
        "turn_end",
        agent=adapter.display_name,
        role=role,
        round=rnd,
        content=turn.content,
        ok=result.ok,
        error=result.error,
        duration=round(result.duration, 2),
        via=("history" if use_history else "prompt"),
    )
    if not result.ok:
        raise AgentTurnError(f"{adapter.display_name} ({role}) failed: {result.error}")

    # Pull any NOTE: lines onto the shared blackboard and notify the UI.
    if session.absorb_notes(adapter.display_name, result.text):
        session.emit("scratchpad", notes=session.scratchpad_view())
    return result.text


class Strategy(ABC):
    name: str = "base"
    description: str = ""
    #: (role_key, human_label) pairs the UI renders one backend-picker for.
    roles: List[Tuple[str, str]] = []
    #: Default system prompt per role key (shown in the UI, overridable).
    personas: Dict[str, str] = {}
    #: True for the user-defined strategy whose roles come from the request.
    custom: bool = False
    default_rounds: int = 2

    @abstractmethod
    def run(self, session: Session) -> str:
        """Drive the collaboration; return the final deliverable text."""

    def finish(self, session: Session, result: str) -> str:
        session.emit("result", content=result)
        return result


class ImplementerReviewer(Strategy):
    name = "implementer_reviewer"
    description = "One agent implements, the other reviews; iterate to convergence."
    roles = [("implementer", "Implementer"), ("reviewer", "Reviewer")]

    IMPL_SYS = (
        "You are the IMPLEMENTER in a two-agent coding collaboration. "
        "Produce a concrete, working implementation for the task. If the conversation "
        "contains review feedback, revise your implementation to address every point. "
        "Return the full updated solution, with code where appropriate."
    )
    REVIEW_SYS = (
        "You are the REVIEWER in a two-agent coding collaboration. Critically review "
        "the implementer's latest solution for correctness, edge cases, clarity, and "
        "tests. Be specific and actionable. End with a clear verdict: APPROVE or "
        "REQUEST CHANGES."
    )
    personas = {"implementer": IMPL_SYS, "reviewer": REVIEW_SYS}

    def run(self, session: Session) -> str:
        impl = ""
        for rnd in range(1, session.rounds + 1):
            impl = _run_turn(
                session, "implementer", self.IMPL_SYS,
                f"Task:\n{session.task}\n\nProvide a complete implementation. If there is "
                f"review feedback in the conversation, revise to address every point.",
                rnd,
            )
            review = _run_turn(
                session, "reviewer", self.REVIEW_SYS,
                f"Task:\n{session.task}\n\nReview the implementer's latest solution and end "
                f"with a verdict (APPROVE or REQUEST CHANGES).",
                rnd,
            )
            if "APPROVE" in review.upper() and "REQUEST CHANGES" not in review.upper():
                session.emit("status", message=f"Reviewer approved in round {rnd}.")
                break
        return self.finish(session, impl)


class DebateConsensus(Strategy):
    name = "debate_consensus"
    description = "Both agents argue distinct positions, then converge on a synthesis."
    roles = [("agent_a", "Debater A"), ("agent_b", "Debater B")]

    A_SYS = (
        "You are DEBATER A. Take a clear position on the task and argue for it with "
        "concrete reasoning. Engage directly with your opponent's points when given."
    )
    B_SYS = (
        "You are DEBATER B. Take a position that meaningfully challenges Debater A, "
        "and argue for it with concrete reasoning. Engage directly with A's points."
    )
    SYNTH_SYS = (
        "You are the SYNTHESIZER. Given both debaters' arguments, produce a balanced "
        "final answer: state the best-supported conclusion and briefly note the key "
        "trade-offs that decided it."
    )
    personas = {"agent_a": A_SYS, "agent_b": B_SYS}

    def run(self, session: Session) -> str:
        _run_turn(session, "agent_a", self.A_SYS,
                  f"Task:\n{session.task}\n\nOpen with your position.", 1)
        _run_turn(session, "agent_b", self.B_SYS,
                  f"Task:\n{session.task}\n\nRespond to Debater A with your opposing position.", 1)
        for rnd in range(2, session.rounds + 1):
            _run_turn(session, "agent_a", self.A_SYS,
                      f"Task:\n{session.task}\n\nRebut the latest opposing point and refine "
                      f"your position.", rnd)
            _run_turn(session, "agent_b", self.B_SYS,
                      f"Task:\n{session.task}\n\nRebut the latest point and refine your "
                      f"position.", rnd)

        session.agents["synthesizer"] = session.agents["agent_a"]
        result = _run_turn(
            session, "synthesizer", self.SYNTH_SYS,
            f"Task:\n{session.task}\n\nSynthesize the final answer from the debate above.",
            session.rounds,
        )
        return self.finish(session, result)


class PlannerExecutor(Strategy):
    name = "planner_executor"
    description = "One agent plans, the other executes; the planner adjusts each round."
    roles = [("planner", "Planner"), ("executor", "Executor")]

    PLAN_SYS = (
        "You are the PLANNER. Break the task into a concise, ordered, actionable plan. "
        "If given an execution report, assess progress and give targeted adjustments "
        "for the next round (or state the work is complete)."
    )
    EXEC_SYS = (
        "You are the EXECUTOR. Carry out the planner's plan as concretely as possible, "
        "producing real artifacts/code. Report what you completed and anything blocking."
    )
    personas = {"planner": PLAN_SYS, "executor": EXEC_SYS}

    def run(self, session: Session) -> str:
        _run_turn(session, "planner", self.PLAN_SYS,
                  f"Task:\n{session.task}\n\nProduce a concise, ordered, actionable plan.", 1)
        execution = ""
        for rnd in range(1, session.rounds + 1):
            execution = _run_turn(
                session, "executor", self.EXEC_SYS,
                f"Task:\n{session.task}\n\nExecute the current plan; report what you "
                f"completed and any blockers.",
                rnd,
            )
            if rnd < session.rounds:
                plan = _run_turn(
                    session, "planner", self.PLAN_SYS,
                    f"Task:\n{session.task}\n\nAssess the latest execution report and give "
                    f"targeted adjustments — or reply COMPLETE if the work is done.",
                    rnd + 1,
                )
                if "COMPLETE" in plan.upper():
                    session.emit("status", message=f"Planner marked work complete in round {rnd}.")
                    break
        return self.finish(session, execution)


class RoundRobin(Strategy):
    name = "round_robin"
    description = (
        "Several agents discuss the task in an open round-table — each sees the whole "
        "conversation and addresses the others — then close with a shared conclusion."
    )
    roles = [("agent_a", "Agent A"), ("agent_b", "Agent B"), ("agent_c", "Agent C")]
    default_rounds = 2

    CLOSE_SYS = (
        "You are the FACILITATOR closing an open discussion among several AI agents. "
        "Summarise the conclusion the group reached and the concrete next steps, fairly "
        "reflecting the different contributions."
    )
    _ROUND_PERSONA = (
        "You are one of several AI agents in an open round-table discussion. Read the "
        "conversation so far and add your next contribution: build on others' points, "
        "agree or push back with reasons, ask a pointed question, or propose concrete "
        "progress. Address the others directly and keep your message focused."
    )
    personas = {"agent_a": _ROUND_PERSONA, "agent_b": _ROUND_PERSONA, "agent_c": _ROUND_PERSONA}

    def _sys(self, me: str, others: str) -> str:
        return (
            f"You are {me}, one of several AI agents in an open round-table discussion. "
            f"The other participants are: {others}. Add your next contribution: build on "
            f"what others said, agree or push back with reasons, ask a pointed question, "
            f"or propose concrete progress. Keep it focused — don't restate the thread."
        )

    def run(self, session: Session) -> str:
        order = [key for key, _ in self.roles]
        names = {r: session.agents[r].display_name for r in order}
        opened = False
        for rnd in range(1, session.rounds + 1):
            for role in order:
                me = names[role]
                others = ", ".join(n for r, n in names.items() if r != role)
                directive = ("Open the discussion with your initial take." if not opened
                             else "Add your next contribution to the discussion.")
                _run_turn(session, role, self._sys(me, others),
                          f"Topic:\n{session.task}\n\nYou are {me}. {directive}", rnd)
                opened = True

        session.agents["closer"] = session.agents[order[0]]
        closing = _run_turn(
            session, "closer", self.CLOSE_SYS,
            f"Topic:\n{session.task}\n\nClose the discussion: state the group's conclusion "
            f"and the next steps.",
            session.rounds,
        )
        return self.finish(session, closing)


class PanelJudge(Strategy):
    name = "panel_judge"
    description = (
        "Three agents present competing positions on the task; a fourth agent acts as "
        "an impartial judge, evaluates them, and delivers the verdict."
    )
    roles = [
        ("agent_a", "Contender A"),
        ("agent_b", "Contender B"),
        ("agent_c", "Contender C"),
        ("judge", "Judge"),
    ]
    default_rounds = 1

    JUDGE_SYS = (
        "You are an impartial JUDGE. Evaluate each contender's position against the task "
        "on correctness, completeness, and risk. Then deliver a clear verdict: name the "
        "strongest approach (or a synthesis of the best parts), justify it concisely, and "
        "state the recommended next step."
    )
    _PANEL_PERSONA = (
        "You are presenting your own position or solution to a panel that a judge will "
        "evaluate. Make the strongest concrete case for your approach; where useful, point "
        "out weaknesses in the others' positions. Be specific — the judge rewards substance."
    )
    personas = {
        "agent_a": _PANEL_PERSONA, "agent_b": _PANEL_PERSONA, "agent_c": _PANEL_PERSONA,
        "judge": JUDGE_SYS,
    }

    def _sys(self, me: str, others: str) -> str:
        return (
            f"You are {me}, presenting your own position or solution to a panel that a "
            f"judge will evaluate. The other contenders are: {others}. Make the strongest "
            f"concrete case for your approach; where useful, point out weaknesses in the "
            f"others' positions. Be specific — the judge rewards substance over rhetoric."
        )

    def run(self, session: Session) -> str:
        order = ["agent_a", "agent_b", "agent_c"]
        names = {r: session.agents[r].display_name for r in order}
        opened = False
        for rnd in range(1, session.rounds + 1):
            for role in order:
                me = names[role]
                others = ", ".join(n for r, n in names.items() if r != role)
                directive = ("Present your position." if not opened
                             else "Sharpen or defend your position against the others.")
                _run_turn(session, role, self._sys(me, others),
                          f"Task:\n{session.task}\n\nYou are {me}. {directive}", rnd)
                opened = True

        verdict = _run_turn(
            session, "judge", self.JUDGE_SYS,
            f"Task:\n{session.task}\n\nEvaluate the contenders' positions above and deliver "
            f"your verdict.",
            session.rounds,
        )
        return self.finish(session, verdict)


class CustomStrategy(Strategy):
    name = "custom"
    description = (
        "Build your own panel — choose a backend, model, and persona for each "
        "participant. They discuss the task in an open round-table, then close with a "
        "shared conclusion."
    )
    roles = []          # supplied per run via the request (session.role_order)
    custom = True
    default_rounds = 2

    CLOSE_SYS = (
        "You are the FACILITATOR closing the discussion. Summarise the conclusion the "
        "participants reached and the concrete next steps, fairly reflecting their input."
    )

    def _fallback_sys(self, me: str, others: str) -> str:
        return (
            f"You are {me}, a participant in a collaborative discussion. The others are: "
            f"{others}. Read the conversation so far and add a focused, useful contribution."
        )

    def run(self, session: Session) -> str:
        order = session.role_order or [k for k, _ in self.roles]
        if not order:
            raise AgentTurnError("custom strategy needs at least one participant")
        names = {r: session.agents[r].display_name for r in order}
        opened = False
        for rnd in range(1, session.rounds + 1):
            for role in order:
                me = names[role]
                others = ", ".join(n for r, n in names.items() if r != role) or "(none)"
                directive = ("Open the discussion with your initial take." if not opened
                             else "Add your next contribution.")
                _run_turn(session, role, self._fallback_sys(me, others),
                          f"Topic:\n{session.task}\n\nYou are {me}. {directive}", rnd)
                opened = True

        session.agents["closer"] = session.agents[order[0]]
        closing = _run_turn(
            session, "closer", self.CLOSE_SYS,
            f"Topic:\n{session.task}\n\nClose the discussion: state the conclusion and the "
            f"next steps.",
            session.rounds,
        )
        return self.finish(session, closing)


class DocAuthoring(Strategy):
    name = "doc_authoring"
    description = (
        "Co-write a document: a writer drafts and revises a shared artifact while an "
        "editor critiques each version, until it's approved."
    )
    roles = [("writer", "Writer"), ("editor", "Editor")]

    WRITER_SYS = (
        "You are the WRITER. Produce and iteratively improve a single document for the "
        "task. When you write or revise it, output the COMPLETE current version wrapped in "
        "<ARTIFACT> and </ARTIFACT> tags (put nothing else inside the tags). Address the "
        "editor's feedback in each revision."
    )
    EDITOR_SYS = (
        "You are the EDITOR. Critique the writer's latest document for structure, clarity, "
        "accuracy, and completeness — be specific and actionable. Do NOT rewrite it or "
        "output an <ARTIFACT> block; give feedback only. End with APPROVE or REQUEST CHANGES."
    )
    personas = {"writer": WRITER_SYS, "editor": EDITOR_SYS}

    def run(self, session: Session) -> str:
        for rnd in range(1, session.rounds + 1):
            text = _run_turn(
                session, "writer", self.WRITER_SYS,
                f"Task:\n{session.task}\n\nWrite or revise the document, addressing any "
                f"editor feedback. Output the full document inside <ARTIFACT> tags.",
                rnd,
            )
            _update_artifact(session, "writer", rnd, text)
            review = _run_turn(
                session, "editor", self.EDITOR_SYS,
                f"Task:\n{session.task}\n\nReview the writer's latest document and end with "
                f"APPROVE or REQUEST CHANGES.",
                rnd,
            )
            if "APPROVE" in review.upper() and "REQUEST CHANGES" not in review.upper():
                session.emit("status", message=f"Editor approved in round {rnd}.")
                break
        return self.finish(session, session.artifact)


class CodeAuthoring(Strategy):
    name = "code_authoring"
    description = (
        "Co-build code: an implementer writes and revises a single code artifact while a "
        "reviewer critiques each version, until it's approved."
    )
    roles = [("implementer", "Implementer"), ("reviewer", "Reviewer")]

    IMPL_SYS = (
        "You are the IMPLEMENTER. Build and iteratively improve a single code file for the "
        "task. When you write or revise it, output the COMPLETE current file wrapped in "
        "<ARTIFACT> and </ARTIFACT> tags (put nothing else inside the tags). Address the "
        "reviewer's feedback in each revision."
    )
    REVIEW_SYS = (
        "You are the REVIEWER. Review the implementer's latest code for correctness, edge "
        "cases, tests, and clarity — be specific. Do NOT output an <ARTIFACT> block; give "
        "feedback only. End with APPROVE or REQUEST CHANGES."
    )
    personas = {"implementer": IMPL_SYS, "reviewer": REVIEW_SYS}

    def run(self, session: Session) -> str:
        for rnd in range(1, session.rounds + 1):
            text = _run_turn(
                session, "implementer", self.IMPL_SYS,
                f"Task:\n{session.task}\n\nWrite or revise the code, addressing any reviewer "
                f"feedback. Output the full file inside <ARTIFACT> tags.",
                rnd,
            )
            _update_artifact(session, "implementer", rnd, text)
            review = _run_turn(
                session, "reviewer", self.REVIEW_SYS,
                f"Task:\n{session.task}\n\nReview the implementer's latest code and end with "
                f"APPROVE or REQUEST CHANGES.",
                rnd,
            )
            if "APPROVE" in review.upper() and "REQUEST CHANGES" not in review.upper():
                session.emit("status", message=f"Reviewer approved in round {rnd}.")
                break
        return self.finish(session, session.artifact)


STRATEGIES = {
    s.name: s
    for s in (
        ImplementerReviewer(),
        DebateConsensus(),
        PlannerExecutor(),
        RoundRobin(),
        PanelJudge(),
        DocAuthoring(),
        CodeAuthoring(),
        CustomStrategy(),
    )
}


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy: {name!r}")
    return STRATEGIES[name]


def strategy_metadata() -> List[dict]:
    """Describe strategies for the UI (roles to fill, default personas, rounds)."""
    return [
        {
            "name": s.name,
            "description": s.description,
            "roles": [
                {"key": k, "label": label, "system": s.personas.get(k, "")}
                for k, label in s.roles
            ],
            "default_rounds": s.default_rounds,
            "custom": s.custom,
        }
        for s in STRATEGIES.values()
    ]
