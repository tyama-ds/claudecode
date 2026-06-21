"""Collaboration strategies.

A strategy choreographs how the agents take turns. Each declares the *roles* it
needs (which the UI maps to a backend each) and implements :meth:`run`, driving
the session turn by turn and emitting events as it goes.

Three strategies ship today:

- :class:`ImplementerReviewer` — one builds, the other reviews, repeat.
- :class:`DebateConsensus` — both argue, then converge on a synthesis.
- :class:`PlannerExecutor` — one plans, the other executes, the planner adjusts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from .session import Session, Turn


class StopRequested(Exception):
    """Raised internally when the user asks to stop a running session."""


class AgentTurnError(Exception):
    """Raised when an agent turn fails, aborting the collaboration."""


def _run_turn(session: Session, role: str, system: str, prompt: str, rnd: int) -> str:
    """Execute one agent turn: emit events, record it, return its text.

    Raises :class:`StopRequested` if a stop was requested, and
    :class:`AgentTurnError` if the agent backend failed.
    """
    if session.should_stop():
        raise StopRequested()

    adapter = session.agents[role]
    session.emit("turn_start", agent=adapter.display_name, role=role, round=rnd)
    result = adapter.generate(prompt, system=system)

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
    )
    if not result.ok:
        raise AgentTurnError(f"{adapter.display_name} ({role}) failed: {result.error}")
    return result.text


class Strategy(ABC):
    name: str = "base"
    description: str = ""
    #: (role_key, human_label) pairs the UI renders one backend-picker for.
    roles: List[Tuple[str, str]] = []
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
        "Produce a concrete, working implementation for the task. If you are given "
        "review feedback, revise your implementation to address every point. "
        "Return the full updated solution, with code where appropriate."
    )
    REVIEW_SYS = (
        "You are the REVIEWER in a two-agent coding collaboration. Critically review "
        "the implementer's latest solution for correctness, edge cases, clarity, and "
        "tests. Be specific and actionable. End with a clear verdict: APPROVE or "
        "REQUEST CHANGES."
    )

    def run(self, session: Session) -> str:
        impl = ""
        review = ""
        for rnd in range(1, session.rounds + 1):
            impl_prompt = f"Task:\n{session.task}"
            if review:
                impl_prompt += f"\n\nReviewer feedback to address:\n{review}"
            impl = _run_turn(session, "implementer", self.IMPL_SYS, impl_prompt, rnd)

            review_prompt = (
                f"Task:\n{session.task}\n\nImplementer's current solution:\n{impl}\n\n"
                f"Review it."
            )
            review = _run_turn(session, "reviewer", self.REVIEW_SYS, review_prompt, rnd)

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

    def run(self, session: Session) -> str:
        a = _run_turn(session, "agent_a", self.A_SYS, f"Task:\n{session.task}\n\nOpening argument.", 1)
        b = _run_turn(session, "agent_b", self.B_SYS,
                      f"Task:\n{session.task}\n\nDebater A said:\n{a}\n\nOpening rebuttal.", 1)

        for rnd in range(2, session.rounds + 1):
            a = _run_turn(
                session, "agent_a", self.A_SYS,
                f"Task:\n{session.task}\n\nOpponent's last point:\n{b}\n\nRebut and refine.",
                rnd,
            )
            b = _run_turn(
                session, "agent_b", self.B_SYS,
                f"Task:\n{session.task}\n\nOpponent's last point:\n{a}\n\nRebut and refine.",
                rnd,
            )

        synth_prompt = (
            f"Task:\n{session.task}\n\nDebater A's final position:\n{a}\n\n"
            f"Debater B's final position:\n{b}\n\nSynthesize the final answer."
        )
        # The synthesizer reuses agent_a's backend in the synthesizer role.
        session.agents["synthesizer"] = session.agents["agent_a"]
        result = _run_turn(session, "synthesizer", self.SYNTH_SYS, synth_prompt, session.rounds)
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

    def run(self, session: Session) -> str:
        plan = _run_turn(session, "planner", self.PLAN_SYS, f"Task:\n{session.task}", 1)
        execution = ""
        for rnd in range(1, session.rounds + 1):
            exec_prompt = f"Task:\n{session.task}\n\nPlan to execute:\n{plan}"
            if execution:
                exec_prompt += f"\n\nYour previous execution:\n{execution}"
            execution = _run_turn(session, "executor", self.EXEC_SYS, exec_prompt, rnd)

            if rnd < session.rounds:
                plan = _run_turn(
                    session, "planner", self.PLAN_SYS,
                    f"Task:\n{session.task}\n\nExecution report:\n{execution}\n\n"
                    f"Assess and adjust the plan.",
                    rnd + 1,
                )
                if "COMPLETE" in plan.upper():
                    session.emit("status", message=f"Planner marked work complete in round {rnd}.")
                    break
        return self.finish(session, execution)


# -- registry --------------------------------------------------------------

STRATEGIES = {
    s.name: s
    for s in (ImplementerReviewer(), DebateConsensus(), PlannerExecutor())
}


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy: {name!r}")
    return STRATEGIES[name]


def strategy_metadata() -> List[dict]:
    """Describe strategies for the UI (roles to fill, default rounds)."""
    return [
        {
            "name": s.name,
            "description": s.description,
            "roles": [{"key": k, "label": label} for k, label in s.roles],
            "default_rounds": s.default_rounds,
        }
        for s in STRATEGIES.values()
    ]
