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

import difflib
import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from ..adapters.base import Message
from .session import Session, Turn, WorkspaceFile


class StopRequested(Exception):
    """Raised internally when the user asks to stop a running session."""


class AgentTurnError(Exception):
    """Raised when an agent turn fails, aborting the collaboration."""


def _reference_block(session: Session) -> str:
    """Render the read-only reference files loaded for the session, if any."""
    if not session.references:
        return ""
    parts = [
        f"REFERENCE FILES — read-only context loaded from {session.reference_dir}. "
        "Consult these to inform your work; you cannot edit them."
    ]
    for rel, content in session.references:
        parts.append(f"----- {rel} -----\n{content}")
    return "\n\n".join(parts)


def _scratchpad_system(session: Session, system: str) -> str:
    """Append reference files + the shared-scratchpad protocol to a system prompt."""
    refs = _reference_block(session)
    ref_part = f"\n\n{refs}" if refs else ""
    return (
        f"{system}{ref_part}\n\n"
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


# -- workspace (real files on disk) ---------------------------------------

# A file block an editing agent emits:  <FILE path="src/foo.py">…contents…</FILE>
_FILE_RE = re.compile(r'<FILE\s+path="([^"]+)"\s*>\n?(.*?)</FILE>', re.DOTALL | re.IGNORECASE)


def _extract_files(text: str) -> List[Tuple[str, str]]:
    """Pull ``(relative_path, contents)`` pairs out of an editing turn."""
    return [(m.group(1).strip(), m.group(2)) for m in _FILE_RE.finditer(text)]


def _safe_join(root: str, rel: str) -> str:
    """Resolve ``rel`` under ``root``, refusing paths that escape the workspace."""
    root_real = os.path.realpath(root)
    full = os.path.realpath(os.path.join(root_real, rel))
    if full != root_real and not full.startswith(root_real + os.sep):
        raise AgentTurnError(f"refusing to write outside the workspace: {rel!r}")
    return full


def _record_edit(session: Session, role: str, rnd: int, rel: str,
                 old: str, new: str, status: str) -> None:
    """Record one file change (diff + stats) on the session and notify the UI."""
    name = session.agents[role].display_name
    diff = "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{rel}", tofile=f"b/{rel}"))
    adds = sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    dels = sum(1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))
    wf = WorkspaceFile(path=rel, content=new, status=status,
                       additions=adds, deletions=dels, diff=diff,
                       author=name, role=role, round=rnd)
    session.record_workspace_file(wf)
    session.emit("workspace_edit", path=rel, status=status, additions=adds,
                 deletions=dels, diff=diff, author=name, role=role, round=rnd)


def _apply_workspace_edits(session: Session, role: str, rnd: int, text: str) -> int:
    """Write the agent's ``<FILE>`` blocks to disk, recording per-file diffs.

    Returns the number of files written. Never stages or commits — changes stay
    in the working tree for the user to review.
    """
    root = session.workspace
    written = 0
    for rel, body in _extract_files(text):
        new_content = body.strip("\n") + "\n" if body.strip() else ""
        full = _safe_join(root, rel)
        existed = os.path.isfile(full)
        old = ""
        if existed:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                old = fh.read()
        os.makedirs(os.path.dirname(full) or root, exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(new_content)
        _record_edit(session, role, rnd, rel, old, new_content,
                     "modified" if existed else "created")
        written += 1
    return written


# -- native workspace edits (a CLI agent editing files with its own tools) --

_WS_MAX_FILE_BYTES = 256 * 1024
_WS_MAX_FILES = 2000


def _snapshot_workspace(root: str) -> Dict[str, str]:
    """Map ``relpath -> content`` for every small text file under ``root``.

    Binary/huge/unreadable files are skipped, as are dependency/VCS dirs, so a
    snapshot stays cheap even in a real project tree.
    """
    snap: Dict[str, str] = {}
    if not os.path.isdir(root):
        return snap
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in sorted(dirnames) if d not in _REF_SKIP_DIRS]
        for fn in sorted(filenames):
            if len(snap) >= _WS_MAX_FILES:
                return snap
            full = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(full) > _WS_MAX_FILE_BYTES:
                    continue
                with open(full, "r", encoding="utf-8") as fh:
                    content = fh.read()
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            snap[os.path.relpath(full, root).replace(os.sep, "/")] = content
    return snap


def _detect_native_edits(session: Session, role: str, rnd: int,
                         baseline: Dict[str, str]) -> Tuple[int, Dict[str, str]]:
    """Diff the tree against a pre-turn snapshot; record changes made directly
    on disk (a CLI agent editing with its own tools).

    Changes already recorded via ``<FILE>`` blocks are skipped (same content).
    Returns ``(changed_count, new_snapshot)`` so the caller can roll forward.
    """
    after = _snapshot_workspace(session.workspace)
    changed = 0
    for rel, content in after.items():
        if baseline.get(rel) == content:
            continue
        wf = session.workspace_files.get(rel)
        if wf is not None and wf.content == content:
            continue  # already recorded this turn (e.g. via a <FILE> block)
        _record_edit(session, role, rnd, rel, baseline.get(rel, ""), content,
                     "modified" if rel in baseline else "created")
        changed += 1
    for rel, old in baseline.items():
        if rel in after:
            continue
        wf = session.workspace_files.get(rel)
        if wf is not None and wf.status == "deleted":
            continue
        _record_edit(session, role, rnd, rel, old, "", "deleted")
        changed += 1
    return changed, after


def _workspace_summary(session: Session) -> str:
    """Render the current workspace diffs for a reviewer's prompt."""
    if not session.workspace_files:
        return "(no files changed yet)"
    parts = []
    for path, wf in session.workspace_files.items():
        parts.append(f"### {path} ({wf.status}, +{wf.additions}/-{wf.deletions})\n{wf.diff}")
    return "\n\n".join(parts)


# -- read-only reference material -----------------------------------------

# Directories and file kinds never worth loading as reference context.
_REF_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", "dist", "build", ".idea", ".vscode", ".tox", ".next",
}
_REF_MAX_FILES = 40
_REF_MAX_FILE_BYTES = 16 * 1024
_REF_MAX_TOTAL_BYTES = 120 * 1024


def load_references(root: str, *, max_files: int = _REF_MAX_FILES,
                    max_file_bytes: int = _REF_MAX_FILE_BYTES,
                    max_total_bytes: int = _REF_MAX_TOTAL_BYTES) -> List[Tuple[str, str]]:
    """Load small text files under ``root`` as ``(relpath, content)`` pairs.

    Skips VCS/build directories, dotfiles, and binary files; truncates any file
    over ``max_file_bytes`` and stops once the file-count or total-size budget is
    reached, so the reference block stays bounded regardless of directory size.
    """
    root_real = os.path.realpath(root)
    out: List[Tuple[str, str]] = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(root_real):
        dirnames[:] = sorted(
            d for d in dirnames if d not in _REF_SKIP_DIRS and not d.startswith(".")
        )
        for fn in sorted(filenames):
            if len(out) >= max_files or total >= max_total_bytes:
                return out
            if fn.startswith("."):
                continue
            full = os.path.join(dirpath, fn)
            try:
                with open(full, "rb") as fh:
                    raw = fh.read(max_file_bytes + 1)
            except OSError:
                continue
            head = raw[:max_file_bytes]
            if b"\x00" in head:  # binary
                continue
            text = head.decode("utf-8", errors="replace")
            if len(raw) > max_file_bytes:
                text += "\n… (truncated)"
            rel = os.path.relpath(full, root_real)
            out.append((rel, text))
            total += len(text.encode("utf-8"))
    return out


# -- conductor / team directives ------------------------------------------

# The conductor speaks to the team in a small line protocol:
#   @worker_1: <assignment>           — assign a subtask
#   @worker_2 [WARN]: <what failed>   — assess a worker (OK or WARN)
#   VERDICT: DONE                     — the task is complete
_ASSIGN_RE = re.compile(r"^[ \t]*@(\w+)[ \t]*:[ \t]*(.+)$", re.MULTILINE)
_ASSESS_RE = re.compile(r"@(\w+)[ \t]*\[[ \t]*(OK|WARN)[ \t]*\][ \t]*:[ \t]*([^\n]*)", re.IGNORECASE)
_DONE_RE = re.compile(r"VERDICT[ \t]*:[ \t]*DONE", re.IGNORECASE)


def _parse_assignments(text: str, workers: List[str]) -> Dict[str, str]:
    """Pull ``@worker_key: instruction`` lines, keeping only real worker keys.

    Assessment lines (``@worker [OK]: …``) are skipped — their ``[..]`` breaks the
    bare ``@key:`` shape, so the regex never matches them.
    """
    out: Dict[str, str] = {}
    for m in _ASSIGN_RE.finditer(text):
        if m.group(1) in workers:
            out[m.group(1)] = m.group(2).strip()
    return out


def _parse_assessments(text: str, workers: List[str]) -> List[Tuple[str, str, str]]:
    """Pull ``(worker_key, "OK"|"WARN", note)`` assessments from a conductor turn."""
    out: List[Tuple[str, str, str]] = []
    for m in _ASSESS_RE.finditer(text):
        if m.group(1) in workers:
            out.append((m.group(1), m.group(2).upper(), m.group(3).strip()))
    return out


def _conductor_done(text: str) -> bool:
    """Whether the conductor declared the task complete."""
    return bool(_DONE_RE.search(text) or re.search(r"^[ \t]*DONE[ \t]*$", text, re.MULTILINE))



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
    #: True when the role set (count/keys) is supplied per run via the request,
    #: rather than fixed by :attr:`roles` (e.g. the conductor's variable team).
    dynamic_roles: bool = False
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


class WorkspaceBuild(Strategy):
    name = "workspace_build"
    description = (
        "Pair-build in a real working directory: the agents first discuss and agree "
        "on a design, then the implementer builds while the reviewer critiques each "
        "diff (applying small fixes directly), until approved."
    )
    roles = [("implementer", "Implementer"), ("reviewer", "Reviewer")]

    IMPL_SYS = (
        "You are the IMPLEMENTER, pair-building software with a reviewer in a real "
        "project workspace. First the two of you agree on a design; then you implement "
        "it, addressing the reviewer's feedback each round. When implementing, follow "
        "the editing instructions given in each turn. Use forward-slash relative paths "
        "inside the workspace; never absolute paths or `..`."
    )
    REVIEW_SYS = (
        "You are the REVIEWER, pair-building software with an implementer in a real "
        "project workspace. First the two of you agree on a design. Then, each round, "
        "review the diff of the implementer's changes for correctness, completeness, "
        "edge cases, and clarity — be specific and actionable. You may apply small, "
        "uncontroversial fixes (typos, obvious bugs) yourself using the editing "
        "instructions given in the turn; leave substantial changes to the implementer "
        "as feedback. End every review with APPROVE or REQUEST CHANGES."
    )
    personas = {"implementer": IMPL_SYS, "reviewer": REVIEW_SYS}

    @staticmethod
    def _edit_how(session: Session, role: str) -> str:
        """Per-role editing instructions: native tools (CLI in the workspace) or
        the <FILE> protocol (everything else)."""
        if getattr(session.agents[role], "workdir", None):
            return (
                "You are running INSIDE the workspace directory: create and edit the "
                "files directly with your own file tools. Do not print <FILE> blocks; "
                "after editing, summarise what you changed and why."
            )
        return (
            "Output EACH file you create or change IN FULL as a block: a line "
            '`<FILE path="relative/path.ext">`, then the complete new file contents, '
            "then a line `</FILE>`."
        )

    def run(self, session: Session) -> str:
        if not session.workspace:
            raise AgentTurnError("workspace_build requires a workspace directory")
        os.makedirs(session.workspace, exist_ok=True)
        baseline = _snapshot_workspace(session.workspace)

        # Phase 1 — design consultation (round 0): agree on an approach first.
        session.emit("status",
                     message="Design phase: the agents discuss and agree on an approach "
                             "before writing any code.")
        _run_turn(
            session, "implementer", self.IMPL_SYS,
            f"Task:\n{session.task}\n\nBefore any code is written, propose a concise "
            f"implementation plan: the files you would create, what each is responsible "
            f"for, and the key design decisions. Ask the reviewer about anything you are "
            f"unsure of. Do NOT create, edit, or output any files yet.",
            0,
        )
        _, baseline = _detect_native_edits(session, "implementer", 0, baseline)
        _run_turn(
            session, "reviewer", self.REVIEW_SYS,
            f"Task:\n{session.task}\n\nDiscuss the implementer's proposed plan: point "
            f"out risks, missing pieces, and simpler alternatives, answer their "
            f"questions, then state the agreed design as a short bullet list. Do NOT "
            f"create, edit, or output any files yet.",
            0,
        )
        _, baseline = _detect_native_edits(session, "reviewer", 0, baseline)

        # Phase 2 — build loop: implement, review (with direct small fixes), repeat.
        for rnd in range(1, session.rounds + 1):
            text = _run_turn(
                session, "implementer", self.IMPL_SYS,
                f"Task:\n{session.task}\n\nImplement it now, following the design you "
                f"both agreed on and addressing any reviewer feedback. "
                f"{self._edit_how(session, 'implementer')}",
                rnd,
            )
            applied = _apply_workspace_edits(session, "implementer", rnd, text)
            native, baseline = _detect_native_edits(session, "implementer", rnd, baseline)
            if applied + native == 0:
                session.emit("status", message=f"No file changes in round {rnd}.")

            review = _run_turn(
                session, "reviewer", self.REVIEW_SYS,
                f"Task:\n{session.task}\n\nCurrent changes:\n\n{_workspace_summary(session)}\n\n"
                f"Review them. If you spot a small, uncontroversial fix, you may apply it "
                f"yourself: {self._edit_how(session, 'reviewer')} "
                f"End with APPROVE or REQUEST CHANGES.",
                rnd,
            )
            r_applied = _apply_workspace_edits(session, "reviewer", rnd, review)
            r_native, baseline = _detect_native_edits(session, "reviewer", rnd, baseline)
            if r_applied + r_native:
                session.emit("status",
                             message=f"Reviewer applied {r_applied + r_native} direct "
                                     f"fix(es) in round {rnd}.")
            if "APPROVE" in review.upper() and "REQUEST CHANGES" not in review.upper():
                session.emit("status", message=f"Reviewer approved in round {rnd}.")
                break
        return self.finish(session, self._summary(session))

    @staticmethod
    def _summary(session: Session) -> str:
        files = session.workspace_files
        if not files:
            return "No files were changed."
        lines = [f"Changed {len(files)} file(s) in {session.workspace}:"]
        for path, wf in files.items():
            lines.append(f"  {wf.status:8} {path}  (+{wf.additions}/-{wf.deletions})")
        return "\n".join(lines)


class ConductorTeam(Strategy):
    name = "conductor_team"
    description = (
        "A conductor assigns subtasks to a team of workers, a reviewer checks each "
        "worker's output, and the conductor evaluates the team each round — calling out "
        "anyone who didn't deliver — until the task is done."
    )
    roles = []          # dynamic: conductor + worker_1..N + reviewer (from the request)
    dynamic_roles = True
    default_rounds = 3

    CONDUCTOR_SYS = (
        "You are the CONDUCTOR leading a team of workers to accomplish one shared task. "
        "You decompose the task, assign concrete subtasks, hold workers accountable — "
        "naming and pushing anyone who doesn't deliver — and integrate their work into the "
        "result. Be specific and demanding but fair. Always use the EXACT worker keys "
        "(e.g. worker_1) when assigning or assessing."
    )
    REVIEWER_SYS = (
        "You are the REVIEWER. You inspect one worker's output at a time against the "
        "assignment the conductor gave them, and report back to the conductor. Be specific: "
        "did they fulfil the assignment? Flag gaps, errors, and anyone who under-delivered. "
        "Keep it short and actionable."
    )

    @staticmethod
    def _worker_sys(me: str, conductor: str) -> str:
        return (
            f"You are {me}, a WORKER on a team led by the conductor {conductor}. Carry out the "
            f"specific assignment the conductor gives you — thoroughly and concretely. Stay "
            f"focused on your assignment and produce real output, not a plan to do it later."
        )

    personas = {"conductor": CONDUCTOR_SYS, "reviewer": REVIEWER_SYS}

    def run(self, session: Session) -> str:
        order = session.role_order or list(session.agents)
        workers = [r for r in order if r.startswith("worker")]
        if "conductor" not in session.agents or "reviewer" not in session.agents or not workers:
            raise AgentTurnError(
                "conductor_team needs a conductor, a reviewer, and at least one worker"
            )
        names = {r: session.agents[r].display_name for r in session.agents}
        cname = names["conductor"]
        team = ", ".join(f"{names[w]} ({w})" for w in workers)

        for w in workers:  # seed the roster panel
            session.emit("worker_status", round=0, worker=w, name=names[w],
                         status="idle", note="awaiting assignment")

        for rnd in range(1, session.rounds + 1):
            # 1. Conductor: assess the prior round (round >= 2) and assign this round.
            instruction = self._conductor_instruction(session, workers, team, rnd)
            ctext = _run_turn(session, "conductor", self.CONDUCTOR_SYS, instruction, rnd)

            for w, verdict, note in _parse_assessments(ctext, workers):
                session.emit("worker_status", round=rnd, worker=w, name=names[w],
                             status=("ok" if verdict == "OK" else "warned"), note=note)

            assignments = _parse_assignments(ctext, workers)
            for w in workers:
                instr = assignments.get(w)
                session.emit("worker_status", round=rnd, worker=w, name=names[w],
                             status=("assigned" if instr else "idle"),
                             note=(instr or "no assignment from the conductor"))

            if rnd > 1 and _conductor_done(ctext):
                session.emit("status", message=f"Conductor declared the work DONE in round {rnd}.")
                break

            # 2. Each worker carries out its assignment.
            for w in workers:
                instr = assignments.get(w) or (
                    "You were not given a specific assignment. Contribute the single most "
                    "useful next step toward the task."
                )
                _run_turn(
                    session, w, self._worker_sys(names[w], cname),
                    f"Task:\n{session.task}\n\nThe conductor ({cname}) assigned you:\n{instr}\n\n"
                    f"Complete your assignment concretely now.",
                    rnd,
                )
                session.emit("worker_status", round=rnd, worker=w, name=names[w],
                             status="delivered", note="")

            # 3. Reviewer inspects each worker individually and reports to the conductor.
            for w in workers:
                _run_turn(
                    session, "reviewer", self.REVIEWER_SYS,
                    f"Task:\n{session.task}\n\nReview {names[w]} ({w})'s latest output against "
                    f"the assignment they were given this round. Did they fulfil it? Note "
                    f"quality, gaps, and whether they pulled their weight. Address your report "
                    f"to the conductor ({cname}).",
                    rnd,
                )

        # Conductor consolidates the team's work into the final deliverable.
        final = _run_turn(
            session, "conductor", self.CONDUCTOR_SYS,
            f"Task:\n{session.task}\n\nThe collaboration is complete. Consolidate the team's "
            f"work into the final deliverable, integrating the workers' contributions and the "
            f"reviewer's feedback into one coherent result.",
            session.rounds,
        )
        return self.finish(session, final)

    def _conductor_instruction(self, session: Session, workers: List[str],
                               team: str, rnd: int) -> str:
        keys = ", ".join(workers)
        if rnd == 1:
            return (
                f"Task:\n{session.task}\n\nYou lead this team: {team}. Break the task into "
                f"concrete subtasks and assign one to each worker. Write each assignment on its "
                f"own line as '@worker_key: instruction', using the exact keys: {keys}."
            )
        return (
            f"Task:\n{session.task}\n\nYour team: {team}. Review the previous round — each "
            f"worker's output and the reviewer's reports. FIRST assess every worker, one per "
            f"line, as '@worker_key [OK]: reason' or '@worker_key [WARN]: what they failed to "
            f"deliver' — call out anyone who slacked or ignored their assignment. THEN reassign "
            f"with '@worker_key: instruction' lines (keys: {keys}). FINALLY write 'VERDICT: "
            f"DONE' if the task is fully complete and the reviewer is satisfied, otherwise "
            f"'VERDICT: CONTINUE'."
        )


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
        WorkspaceBuild(),
        ConductorTeam(),
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
            "dynamic_roles": s.dynamic_roles,
        }
        for s in STRATEGIES.values()
    ]
