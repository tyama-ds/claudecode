"""The orchestration engine: runs a session's strategy to completion."""

from __future__ import annotations

import threading

from .session import Session
from .strategies import AgentTurnError, StopRequested, get_strategy

_EVAL_SYS = (
    "You are a strict, independent EVALUATOR. You judge whether a team's deliverable "
    "truly accomplishes the task — checking substance, completeness, and quality, not "
    "presentation. You are immune to confident wording; claims without evidence do "
    "not count. Reply with 'PASS' as the very first line only when the deliverable is "
    "genuinely acceptable; otherwise reply 'FAIL' on the first line followed by "
    "concrete, prioritised, actionable feedback for the rework."
)


def _evaluate_iteration(session: Session, result: str, iteration: int):
    """One evaluator turn over the current deliverable.

    Returns ``(passed, feedback)``. Evaluator failures count as PASS so a broken
    evaluator can never trap the loop.
    """
    adapter = session.loop_evaluator
    session.emit("turn_start", agent=adapter.display_name, role="evaluator",
                 round=iteration, action="review")
    ws = ""
    if session.workspace_files:
        files = ", ".join(sorted(session.workspace_files))
        ws = f"\n\nFiles changed in the shared workspace: {files}"
    res = adapter.generate(
        f"Task:\n{session.task}\n\nDeliverable produced by the team "
        f"(iteration {iteration}):\n{result[:6000]}{ws}\n\n"
        f"Evaluate it strictly. First line: PASS or FAIL. If FAIL, follow with "
        f"the concrete gaps the team must close.",
        system=_EVAL_SYS,
    )
    content = res.text if res.ok else (res.error or "evaluator error")
    session.emit("turn_end", agent=adapter.display_name, role="evaluator",
                 round=iteration, content=content, ok=res.ok, error=res.error,
                 duration=round(res.duration, 2), action="review")
    if not res.ok:
        session.emit("status", message="Evaluator failed — accepting the result as-is.")
        return True, ""
    first, _, rest = res.text.strip().partition("\n")
    passed = first.strip().upper().startswith("PASS")
    return passed, rest.strip() or first.strip()


def _auto_loop(session: Session, strategy, result: str) -> str:
    """Evaluate-and-rework loop: rerun the strategy with the evaluator's
    feedback (fresh transcript, same workspace) until PASS, the iteration cap,
    or a human Stop/Finish."""
    if not session.loop_iters or session.loop_evaluator is None:
        return result
    base_task = session.task
    try:
        for iteration in range(1, session.loop_iters + 1):
            if session.stop_requested or session.finish_requested:
                break
            passed, feedback = _evaluate_iteration(session, result, iteration)
            if passed:
                session.emit("loop", iteration=iteration, max=session.loop_iters,
                             verdict="pass")
                session.emit("status",
                             message=f"Auto-loop: evaluator PASSED the result in "
                                     f"iteration {iteration}.")
                return result
            if iteration >= session.loop_iters:
                session.emit("loop", iteration=iteration, max=session.loop_iters,
                             verdict="giveup", feedback=feedback[:500])
                session.emit("status",
                             message=f"Auto-loop: iteration cap ({session.loop_iters}) "
                                     f"reached — returning the latest result.")
                return result
            session.emit("loop", iteration=iteration + 1, max=session.loop_iters,
                         verdict="fail", feedback=feedback[:500])
            session.emit("status",
                         message=f"Auto-loop: evaluator FAILED iteration {iteration} — "
                                 f"reworking ({iteration + 1}/{session.loop_iters}).")
            # Fresh context for the rework; the workspace stays as-is on disk.
            session.transcript.clear()
            session.scratchpad.clear()
            session.ws_baseline = None
            session.task = (
                f"{base_task}\n\n[AUTO-LOOP — attempt {iteration + 1} of "
                f"{session.loop_iters}; the previous attempt FAILED evaluation]\n"
                f"Evaluator feedback to address:\n{feedback}\n\n"
                f"[PREVIOUS RESULT]\n{result[:3000]}"
            )
            result = strategy.run(session)
        return result
    finally:
        session.task = base_task


def run_session(session: Session) -> None:
    """Run a session's strategy synchronously (blocking).

    Emits ``session_start`` / ``session_end`` around the strategy (plus the
    auto-loop evaluate/rework cycle when enabled), translates failures into
    ``error`` events, and always closes the event bus so streaming clients
    disconnect cleanly.
    """
    strategy = get_strategy(session.strategy)
    session.status = "running"
    session.emit(
        "session_start",
        session_id=session.id,
        task=session.task,
        strategy=session.strategy,
        rounds=session.rounds,
        agents={role: a.display_name for role, a in session.agents.items()},
        workspace=session.workspace,
        workspace_created=session.workspace_created,
        references=len(session.references),
        reference_dir=session.reference_dir,
        supervisors=session.supervisors,
        tools=session.tools,
        rounds_unlimited=(session.rounds == 0),
        loop_iters=session.loop_iters,
    )
    try:
        result = strategy.run(session)
        result = _auto_loop(session, strategy, result)
        session.status = "done"
        session.result = result
        session.emit("session_end", status="done", result=result)
    except StopRequested:
        session.status = "stopped"
        session.emit("session_end", status="stopped")
    except AgentTurnError as exc:
        session.status = "error"
        session.emit("error", message=str(exc))
        session.emit("session_end", status="error")
    except Exception as exc:  # noqa: BLE001 - report any unexpected failure
        session.status = "error"
        session.emit("error", message=f"{type(exc).__name__}: {exc}")
        session.emit("session_end", status="error")
    finally:
        session.bus.close()


def start_session(session: Session) -> threading.Thread:
    """Run :func:`run_session` in a daemon thread and return it."""
    thread = threading.Thread(target=run_session, args=(session,), daemon=True)
    thread.start()
    return thread
