"""The orchestration engine: runs a session's strategy to completion."""

from __future__ import annotations

import threading

from .session import Session
from .strategies import AgentTurnError, StopRequested, get_strategy


def run_session(session: Session) -> None:
    """Run a session's strategy synchronously (blocking).

    Emits ``session_start`` / ``session_end`` around the strategy, translates
    failures into ``error`` events, and always closes the event bus so streaming
    clients disconnect cleanly.
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
        workspace_git=session.workspace_git,
        references=len(session.references),
        reference_dir=session.reference_dir,
    )
    try:
        result = strategy.run(session)
        session.status = "done"
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
