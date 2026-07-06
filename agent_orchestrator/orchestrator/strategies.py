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
import subprocess
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
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


# Acceptance criteria a manager defines in round 1: "CRITERION: <statement>"
_CRITERION_RE = re.compile(r"^\s*(?:[-*]\s*)?CRITERION\s*:\s*(.+?)\s*$",
                           re.IGNORECASE | re.MULTILINE)


def _extract_criteria(text: str) -> List[str]:
    return [m.group(1).strip() for m in _CRITERION_RE.finditer(text)]


_AUDITOR_SYS = (
    "You are the completion AUDITOR. A manager just declared the task DONE; your job "
    "is to try to PROVE THEM WRONG. Check the team's actual outputs in the "
    "conversation against every acceptance criterion, hunting for unmet criteria, "
    "missing pieces, unverified claims, and quality gaps. Managers' assertions are "
    "not evidence — only concrete output is. Be strict: when in doubt, reject."
)


def _challenge_done(session: Session, challenger: str, criteria: List[str],
                    rnd: int) -> bool:
    """Adversarial DONE gate: a declared DONE only stands when the challenger
    audits the work and replies CONFIRM DONE."""
    crit_block = ("\n".join(f"- {c}" for c in criteria) if criteria
                  else "(no explicit criteria were recorded — derive them from the task)")
    text = _run_turn(
        session, challenger, _AUDITOR_SYS,
        f"Task:\n{session.task}\n\nThe manager declared the work DONE. Audit it "
        f"adversarially against the acceptance criteria:\n{crit_block}\n\n"
        f"If — and only if — every criterion is genuinely, verifiably met, reply "
        f"with the single line 'CONFIRM DONE'. Otherwise reply 'REJECT' followed "
        f"by the concrete gaps the team must close.",
        rnd, action="review",
    )
    up = text.upper()
    return "CONFIRM DONE" in up and "REJECT" not in up


def _conductor_done(text: str) -> bool:
    """Whether the conductor declared the task complete."""
    return bool(_DONE_RE.search(text) or re.search(r"^[ \t]*DONE[ \t]*$", text, re.MULTILINE))



# -- rounds ------------------------------------------------------------------

# Safety cap when the user picks "no round limit" (rounds == 0).
_MAX_UNLIMITED_ROUNDS = 100


def _rounds_iter(session: Session):
    """Round numbers 1..N; ``rounds == 0`` means unlimited (safety-capped)."""
    limit = session.rounds if session.rounds > 0 else _MAX_UNLIMITED_ROUNDS
    return range(1, limit + 1)


def _wrap_up(session: Session, rnd: int) -> bool:
    """True when a human pressed Finish — the strategy should wrap up now."""
    if session.finish_requested:
        session.emit("status",
                     message=f"Finish requested by the user in round {rnd} — wrapping up.")
        return True
    return False


# -- agent tools (orchestrator-provided, opt-in per session) -------------------

# An agent calls a tool by writing:  <TOOL name="run">pytest -q</TOOL>
_TOOL_RE = re.compile(r'<TOOL\s+name="([^"]+)"\s*>\n?(.*?)</TOOL>', re.DOTALL | re.IGNORECASE)
_TOOL_MAX_OUTPUT = 8_000
_TOOL_MAX_STEPS = 3  # tool->result follow-up turns per agent turn

_TOOL_DOCS = {
    "list_files": "list_files — input ignored; lists every file in the workspace.",
    "read_file": "read_file — input: a workspace-relative path; returns the file's contents.",
    "write_file": ("write_file — input: FIRST line is a workspace-relative path, every "
                   "following line is the complete new file content; writes the file."),
    "run": "run — input: a shell command; runs in the workspace (60s timeout), returns output.",
    "http_get": "http_get — input: an http(s) URL; fetches it and returns the text body.",
}


def _tools_system(session: Session) -> str:
    """The system-prompt suffix documenting the tools this session enables."""
    docs = [_TOOL_DOCS[t] for t in session.tools if t in _TOOL_DOCS]
    if not docs:
        return ""
    return (
        "\n\n[TOOLS] You may use tools by writing, anywhere in a reply, a block like:\n"
        '<TOOL name="tool_name">input</TOOL>\n'
        "The orchestrator executes every call and sends you the results so you can "
        "continue. Available tools:\n" + "\n".join(f"- {d}" for d in docs)
    )


def _tool_root(session: Session) -> str:
    return session.workspace or session.reference_dir or os.getcwd()


def _exec_tool(session: Session, name: str, arg: str, role: str, rnd: int) -> str:
    """Execute one tool call; always returns text (errors included)."""
    root = _tool_root(session)
    try:
        if name == "list_files":
            files = sorted(_snapshot_workspace(root))
            return "\n".join(files) if files else "(no files)"
        if name == "read_file":
            full = _safe_join(root, arg.strip())
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()[:_TOOL_MAX_OUTPUT]
        if name == "write_file":
            if not session.workspace:
                return "write_file requires a workspace directory for this session"
            rel, _, content = arg.partition("\n")
            rel = rel.strip()
            if not rel:
                return "write_file: missing path on the first line"
            full = _safe_join(session.workspace, rel)
            existed = os.path.isfile(full)
            old = ""
            if existed:
                with open(full, "r", encoding="utf-8", errors="replace") as fh:
                    old = fh.read()
            os.makedirs(os.path.dirname(full) or session.workspace, exist_ok=True)
            new_content = content if content.endswith("\n") or not content else content + "\n"
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(new_content)
            _record_edit(session, role, rnd, rel, old, new_content,
                         "modified" if existed else "created")
            return f"wrote {rel} ({len(new_content)} bytes)"
        if name == "run":
            proc = subprocess.run(arg, shell=True, cwd=root, capture_output=True,
                                  text=True, timeout=60)
            out = ((proc.stdout or "") + (proc.stderr or ""))[:_TOOL_MAX_OUTPUT]
            return f"(exit {proc.returncode})\n{out}"
        if name == "http_get":
            import urllib.request
            with urllib.request.urlopen(arg.strip(), timeout=30) as resp:
                return resp.read(_TOOL_MAX_OUTPUT).decode("utf-8", errors="replace")
        return f"unknown tool: {name}"
    except Exception as exc:  # noqa: BLE001 - reported to the agent, not fatal
        return f"tool error: {type(exc).__name__}: {exc}"


def _run_tool_calls(session: Session, role: str, rnd: int, text: str) -> Optional[str]:
    """Execute the enabled tool calls in ``text``; return a results block, or
    ``None`` when there is nothing to run."""
    calls = [(m.group(1).strip(), m.group(2).strip()) for m in _TOOL_RE.finditer(text)]
    calls = [(n, a) for n, a in calls if n in session.tools]
    if not calls:
        return None
    blocks = []
    for name, arg in calls:
        out = _exec_tool(session, name, arg, role, rnd)
        session.emit("tool_use", role=role, agent=session.agents[role].display_name,
                     tool=name, input=arg[:200], output=out[:400], round=rnd)
        blocks.append(f'<TOOL_RESULT name="{name}">\n{out}\n</TOOL_RESULT>')
    return "\n".join(blocks)


# -- parallel turns ------------------------------------------------------------

def _run_turns_parallel(session: Session, specs: List[tuple]) -> Dict[str, str]:
    """Run several agent turns concurrently (one thread each).

    ``specs``: ``(role, system, instruction, rnd, action)`` tuples — distinct
    roles. Same-backend roles parallelise fine (each role has its own adapter).
    Returns ``{role: text}``; the first turn failure is re-raised after all
    turns settle.
    """
    results: Dict[str, str] = {}
    errors: List[Exception] = []

    def one(spec):
        role, system, instruction, rnd, action = spec
        try:
            results[role] = _run_turn(session, role, system, instruction, rnd, action)
        except (StopRequested, AgentTurnError) as exc:
            errors.append(exc)

    if len(specs) == 1:
        one(specs[0])
    else:
        with ThreadPoolExecutor(max_workers=min(8, len(specs))) as pool:
            list(pool.map(one, specs))
    if errors:
        raise errors[0]
    return results


# Serialises generic workspace bookkeeping across parallel turns.
_WS_LOCK = threading.Lock()

_WS_LISTING_CAP = 200


def _workspace_listing(root: str) -> str:
    """A cheap paths-only listing (contents stay out of the context window)."""
    names = sorted(_snapshot_workspace(root))
    lines = names[:_WS_LISTING_CAP]
    if len(names) > _WS_LISTING_CAP:
        lines.append(f"... and {len(names) - _WS_LISTING_CAP} more")
    return "\n".join(lines) if lines else "(empty)"


def _workspace_system(session: Session) -> str:
    """System-prompt suffix for strategies that share a generic workspace.

    Deliberately context-frugal: only the file *listing* is injected — agents
    read and write files individually instead of receiving everything.
    """
    if not session.workspace or session.strategy == "workspace_build":
        return ""  # workspace_build runs its own richer workspace protocol
    return (
        "\n\n[SHARED WORKSPACE] The team shares a real working directory; work on "
        "files INDIVIDUALLY to keep context small — do not ask for everything at "
        "once. Current files:\n" + _workspace_listing(session.workspace) + "\n"
        "Read a file with the read_file tool (or directly, if you are a CLI running "
        "inside the directory). Write or change a file by emitting "
        '`<FILE path="relative/path.ext">full new content</FILE>` or via the '
        "write_file tool — one file at a time, always the complete content. Never "
        "use absolute paths or `..`."
    )


def _absorb_generic_edits(session: Session, role: str, rnd: int, text: str) -> None:
    """Apply <FILE> blocks and pick up native edits for any strategy that has a
    workspace (workspace_build does this itself with richer semantics)."""
    if not session.workspace or session.strategy == "workspace_build":
        return
    with _WS_LOCK:
        if session.ws_baseline is None:
            session.ws_baseline = _snapshot_workspace(session.workspace)
        _apply_workspace_edits(session, role, rnd, text)
        _, session.ws_baseline = _detect_native_edits(session, role, rnd,
                                                      session.ws_baseline)


def _run_turn(session: Session, role: str, system: str, instruction: str, rnd: int,
              action: str = "") -> str:
    """One agent turn, plus tool follow-ups when the session enables tools.

    If the reply contains ``<TOOL>`` calls, they are executed and the agent
    immediately gets another turn with the results (up to a small cap); the
    last reply is returned. When the session has a generic shared workspace,
    ``<FILE>`` blocks and native file edits are absorbed after each reply.
    """
    text = _run_single_turn(session, role, system, instruction, rnd, action)
    if session.tools:
        for _ in range(_TOOL_MAX_STEPS):
            results_block = _run_tool_calls(session, role, rnd, text)
            if results_block is None:
                break
            text = _run_single_turn(
                session, role, system,
                f"Tool results:\n{results_block}\n\nContinue your turn using these "
                f"results. You may call tools again, or give your final answer.",
                rnd, action,
            )
    _absorb_generic_edits(session, role, rnd, text)
    return text


def _run_single_turn(session: Session, role: str, system: str, instruction: str, rnd: int,
                     action: str = "") -> str:
    """Execute one agent turn: emit events, record it, return its text.

    ``instruction`` is the short per-turn directive; the conversation context is
    supplied automatically — as ``history`` when the backend supports it, or
    embedded in the prompt as a fallback when it doesn't. ``action`` is a short
    machine-friendly label of what the agent is doing right now ("design",
    "implement", "review", …) surfaced live on the UI's agent board.

    Raises :class:`StopRequested` on a stop request and :class:`AgentTurnError`
    if the backend fails.
    """
    if session.should_stop():
        raise StopRequested()

    adapter = session.agents[role]
    session.emit("turn_start", agent=adapter.display_name, role=role, round=rnd,
                 action=action)

    # A per-role persona override from the UI wins over the strategy's default;
    # the workspace and tools contracts (when enabled) are appended either way.
    system = ((session.personas.get(role) or system)
              + _workspace_system(session) + _tools_system(session))

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
        action=action,
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
    #: True when ``rounds == 0`` (no limit; run until DONE / a human finishes)
    #: makes sense for this strategy.
    supports_unlimited: bool = False
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
        "Co-build in a real working directory: the team first discusses and agrees "
        "on a design, then the implementer builds while one or more reviewers "
        "critique each diff (applying small fixes directly), until all approve."
    )
    # Default pair for headless runs/tests; the UI supplies a dynamic role list
    # (implementer + reviewer_1..N) so several reviewers can co-create.
    roles = [("implementer", "Implementer"), ("reviewer", "Reviewer")]
    dynamic_roles = True
    supports_unlimited = True

    IMPL_SYS = (
        "You are the IMPLEMENTER, co-building software with one or more reviewers in "
        "a real project workspace. First the team agrees on a design; then you "
        "implement it, addressing the reviewers' feedback each round. When "
        "implementing, follow the editing instructions given in each turn. Use "
        "forward-slash relative paths inside the workspace; never absolute paths "
        "or `..`."
    )
    REVIEW_SYS = (
        "You are a REVIEWER (possibly one of several), co-building software with an "
        "implementer in a real project workspace. First the team agrees on a design. "
        "Then, each round, review the diff of the implementer's changes for "
        "correctness, completeness, edge cases, and clarity — be specific and "
        "actionable, and don't just repeat what other reviewers already said. You may "
        "apply small, uncontroversial fixes (typos, obvious bugs) yourself using the "
        "editing instructions given in the turn; leave substantial changes to the "
        "implementer as feedback. End every review with APPROVE or REQUEST CHANGES."
    )
    personas = {"implementer": IMPL_SYS, "reviewer": REVIEW_SYS}

    @staticmethod
    def _reviewers(session: Session) -> List[str]:
        order = session.role_order or sorted(session.agents)
        return [r for r in order if r != "implementer" and r in session.agents]

    @staticmethod
    def _edit_how(session: Session, role: str) -> str:
        """Per-role editing instructions: native tools (CLI in the workspace) or
        the <FILE> protocol (everything else)."""
        if getattr(session.agents[role], "workdir", None):
            return (
                "You are running INSIDE the workspace directory: read any existing "
                "file and create/edit files directly with your own file tools. Do not "
                "print <FILE> blocks; after editing, summarise what you changed and why."
            )
        return (
            "Output EACH file you create or change IN FULL as a block: a line "
            '`<FILE path="relative/path.ext">`, then the complete new file contents, '
            "then a line `</FILE>`. Work in small increments so no reply is cut off "
            "by output limits: touch at most ~3 files per reply, each written in "
            "full. If more work remains afterwards, end your reply with a single "
            "line reading exactly CONTINUE and you will immediately get another "
            "turn before review."
        )

    @staticmethod
    def _existing_files_block(session: Session) -> str:
        """Existing workspace contents, so agents work WITH the local project."""
        refs = load_references(session.workspace)
        if not refs:
            return ""
        parts = [f"### {path}\n```\n{content}\n```" for path, content in refs]
        return ("\n\n[EXISTING WORKSPACE FILES — this is the project you are "
                "working on; build on and modify these rather than starting over]\n"
                + "\n\n".join(parts))

    # Extra implementer turns granted within one round when it ends with CONTINUE.
    _MAX_CONTINUES = 4

    # Automated verification: run the session's test command in the workspace
    # and stream the outcome; the text summary is fed back to the agents.
    _TEST_TIMEOUT = 120
    _TEST_MAX_OUTPUT = 4_000

    @classmethod
    def _run_tests(cls, session: Session, rnd: int) -> str:
        """Run ``session.test_command`` in the workspace; returns a prompt block
        (empty string when no command is configured)."""
        cmd = (session.test_command or "").strip()
        if not cmd:
            return ""
        try:
            proc = subprocess.run(cmd, shell=True, cwd=session.workspace,
                                  capture_output=True, text=True,
                                  timeout=cls._TEST_TIMEOUT)
            output = ((proc.stdout or "") + (proc.stderr or ""))[-cls._TEST_MAX_OUTPUT:]
            ok, code = proc.returncode == 0, proc.returncode
        except subprocess.TimeoutExpired:
            output, ok, code = f"(timed out after {cls._TEST_TIMEOUT}s)", False, -1
        except OSError as exc:
            output, ok, code = f"(could not run: {exc})", False, -1
        session.emit("test_result", round=rnd, command=cmd, ok=ok, exit=code,
                     output=output[-1_500:])
        verdict = "PASSED" if ok else f"FAILED (exit {code})"
        return (f"\n\n[AUTOMATED TEST RUN — `{cmd}` {verdict}]\n{output}\n"
                + ("" if ok else "Tests are failing: fixing them takes priority, and "
                                 "reviewers must NOT approve while they fail."))

    def run(self, session: Session) -> str:
        if not session.workspace:
            raise AgentTurnError("workspace_build requires a workspace directory")
        if "implementer" not in session.agents:
            raise AgentTurnError("workspace_build needs an 'implementer' role")
        reviewers = self._reviewers(session)
        if not reviewers:
            raise AgentTurnError("workspace_build needs at least one reviewer")
        os.makedirs(session.workspace, exist_ok=True)
        baseline = _snapshot_workspace(session.workspace)
        existing = self._existing_files_block(session)

        # Phase 1 — design consultation (round 0): agree on an approach first.
        session.emit("status",
                     message="Design phase: the team discusses and agrees on an "
                             "approach before writing any code.")
        _run_turn(
            session, "implementer", self.IMPL_SYS,
            f"Task:\n{session.task}{existing}\n\nBefore any code is written, propose "
            f"a concise implementation plan: the files you would create or modify, "
            f"what each is responsible for, and the key design decisions. Ask the "
            f"reviewers about anything you are unsure of. Do NOT create, edit, or "
            f"output any files yet.",
            0, action="design",
        )
        _, baseline = _detect_native_edits(session, "implementer", 0, baseline)
        design_specs = [(
            rv, self.REVIEW_SYS,
            f"Task:\n{session.task}{existing}\n\nDiscuss the implementer's proposed "
            f"plan: point out risks, missing pieces, and simpler alternatives, answer "
            f"their questions, then state the design you'd agree to as a short bullet "
            f"list. Do NOT create, edit, or output any files yet.",
            0, "design",
        ) for rv in reviewers]
        _run_turns_parallel(session, design_specs)
        for rv in reviewers:
            _, baseline = _detect_native_edits(session, rv, 0, baseline)

        # Phase 2 — build loop: implement (incrementally), auto-run the tests,
        # then every reviewer weighs in concurrently (with direct small fixes);
        # repeat until all reviewers approve, the rounds run out, or a human
        # presses Finish.
        test_block = ""
        for rnd in _rounds_iter(session):
            if _wrap_up(session, rnd):
                break
            text = _run_turn(
                session, "implementer", self.IMPL_SYS,
                f"Task:\n{session.task}{existing if rnd == 1 else ''}{test_block}\n\n"
                f"Implement it now, following the design the team agreed on and "
                f"addressing all reviewer feedback. "
                f"{self._edit_how(session, 'implementer')}",
                rnd, action="implement",
            )
            applied = _apply_workspace_edits(session, "implementer", rnd, text)
            native, baseline = _detect_native_edits(session, "implementer", rnd, baseline)

            # Incremental mode: the implementer asked to keep going this round.
            extra = 0
            while (extra < self._MAX_CONTINUES and not session.finish_requested
                   and re.search(r"^\s*CONTINUE\s*$", text, re.MULTILINE)):
                extra += 1
                text = _run_turn(
                    session, "implementer", self.IMPL_SYS,
                    f"Continue implementing exactly where you left off (same round; "
                    f"continuation {extra}/{self._MAX_CONTINUES}). "
                    f"{self._edit_how(session, 'implementer')}",
                    rnd, action="implement",
                )
                applied += _apply_workspace_edits(session, "implementer", rnd, text)
                n2, baseline = _detect_native_edits(session, "implementer", rnd, baseline)
                native += n2
            if applied + native == 0:
                session.emit("status", message=f"No file changes in round {rnd}.")

            # Automated verification: run the configured test command and share
            # the outcome with the whole team.
            test_block = self._run_tests(session, rnd)

            review_specs = [(
                rv, self.REVIEW_SYS,
                f"Task:\n{session.task}\n\nCurrent changes:\n\n"
                f"{_workspace_summary(session)}{test_block}\n\n"
                f"Review them. If you spot a small, uncontroversial fix, you may "
                f"apply it yourself: {self._edit_how(session, rv)} "
                f"End with APPROVE or REQUEST CHANGES.",
                rnd, "review",
            ) for rv in reviewers]
            reviews = _run_turns_parallel(session, review_specs)

            approvals = 0
            fixes = 0
            for rv in reviewers:
                review = reviews.get(rv, "")
                r_applied = _apply_workspace_edits(session, rv, rnd, review)
                r_native, baseline = _detect_native_edits(session, rv, rnd, baseline)
                if r_applied + r_native:
                    fixes += r_applied + r_native
                    session.emit("status",
                                 message=f"{session.agents[rv].display_name} ({rv}) "
                                         f"applied {r_applied + r_native} direct "
                                         f"fix(es) in round {rnd}.")
                if "APPROVE" in review.upper() and "REQUEST CHANGES" not in review.upper():
                    approvals += 1
            if fixes:  # reviewers touched files -> re-verify before moving on
                test_block = self._run_tests(session, rnd)
            if approvals == len(reviewers):
                session.emit("status",
                             message=f"All {len(reviewers)} reviewer(s) approved in "
                                     f"round {rnd}.")
                break
            session.emit("status",
                         message=f"{approvals}/{len(reviewers)} reviewer(s) "
                                 f"approved in round {rnd} — continuing.")
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
    supports_unlimited = True
    default_rounds = 3

    CONDUCTOR_SYS = (
        "You are the CONDUCTOR leading a team of workers to accomplish one shared task. "
        "You decompose the task, assign concrete subtasks, hold workers accountable — "
        "naming and pushing anyone who doesn't deliver — and integrate their work into the "
        "result. Be specific and demanding but fair. Always use the EXACT worker keys "
        "(e.g. worker_1) when assigning or assessing. You are STRICT about completion: "
        "workers' claims are not evidence, only their actual output is; declaring DONE "
        "prematurely is a serious failure on your part, and every DONE you declare is "
        "audited adversarially."
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

        criteria: List[str] = []
        last_rnd = 1
        for rnd in _rounds_iter(session):
            last_rnd = rnd
            if _wrap_up(session, rnd):
                break
            # 1. Conductor: assess the prior round (round >= 2) and assign this round.
            instruction = self._conductor_instruction(session, workers, team, rnd)
            ctext = _run_turn(session, "conductor", self.CONDUCTOR_SYS, instruction, rnd,
                              action="assign")
            if rnd == 1:
                criteria = _extract_criteria(ctext)
                if criteria:
                    session.emit("status",
                                 message=f"{len(criteria)} acceptance criteria defined.")

            for w, verdict, note in _parse_assessments(ctext, workers):
                session.emit("worker_status", round=rnd, worker=w, name=names[w],
                             status=("ok" if verdict == "OK" else "warned"), note=note)

            assignments = _parse_assignments(ctext, workers)
            for w in workers:
                instr = assignments.get(w)
                session.emit("worker_status", round=rnd, worker=w, name=names[w],
                             status=("assigned" if instr else "idle"),
                             note=(instr or "no assignment from the conductor"))

            if rnd > 1 and rnd >= session.min_rounds and _conductor_done(ctext):
                # A declared DONE must survive an adversarial audit to stand.
                if _challenge_done(session, "reviewer", criteria, rnd):
                    session.emit("status",
                                 message=f"DONE declared in round {rnd} and CONFIRMED "
                                         f"by the auditor.")
                    break
                session.emit("status",
                             message=f"DONE declared in round {rnd} but REJECTED by "
                                     f"the auditor — work continues.")
                continue  # let the conductor react to the rejection immediately

            # 2. All workers carry out their assignments IN PARALLEL (also fine
            # when several workers use the same backend — separate adapters).
            def worker_spec(w):
                instr = assignments.get(w) or (
                    "You were not given a specific assignment. Contribute the single most "
                    "useful next step toward the task."
                )
                return (w, self._worker_sys(names[w], cname),
                        f"Task:\n{session.task}\n\nThe conductor ({cname}) assigned you:\n"
                        f"{instr}\n\nComplete your assignment concretely now.",
                        rnd, "implement")
            _run_turns_parallel(session, [worker_spec(w) for w in workers])
            for w in workers:
                session.emit("worker_status", round=rnd, worker=w, name=names[w],
                             status="delivered", note="")

            # 3. Reviewer inspects each worker individually and reports to the
            # conductor (one reviewer agent — necessarily sequential).
            for w in workers:
                _run_turn(
                    session, "reviewer", self.REVIEWER_SYS,
                    f"Task:\n{session.task}\n\nReview {names[w]} ({w})'s latest output against "
                    f"the assignment they were given this round. Did they fulfil it? Note "
                    f"quality, gaps, and whether they pulled their weight. Address your report "
                    f"to the conductor ({cname}).",
                    rnd, action="review",
                )

        # Conductor consolidates the team's work into the final deliverable.
        final = _run_turn(
            session, "conductor", self.CONDUCTOR_SYS,
            f"Task:\n{session.task}\n\nThe collaboration is complete. Consolidate the team's "
            f"work into the final deliverable, integrating the workers' contributions and the "
            f"reviewer's feedback into one coherent result.",
            last_rnd, action="integrate",
        )
        return self.finish(session, final)

    def _conductor_instruction(self, session: Session, workers: List[str],
                               team: str, rnd: int) -> str:
        keys = ", ".join(workers)
        if rnd == 1:
            return (
                f"Task:\n{session.task}\n\nYou lead this team: {team}. FIRST define the "
                f"acceptance criteria for the whole task — 3 to 7 measurable statements, "
                f"each on its own line as 'CRITERION: <statement>'; the task is only DONE "
                f"when every one is verifiably met. THEN break the task into concrete "
                f"subtasks and assign one to each worker, each on its own line as "
                f"'@worker_key: instruction', using the exact keys: {keys}."
            )
        if session.min_rounds and rnd < session.min_rounds:
            verdict = (f"Declaring DONE is NOT allowed before round {session.min_rounds}; "
                       f"write 'VERDICT: CONTINUE'.")
        else:
            verdict = (
                "FINALLY: only if EVERY acceptance criterion is verifiably met by the "
                "actual outputs (a worker saying so is not evidence), write one line "
                "'✔ <criterion> — <evidence>' per criterion and then 'VERDICT: DONE'; "
                "otherwise write 'VERDICT: CONTINUE'. A DONE will be audited "
                "adversarially, and a rejected DONE reflects badly on you."
            )
        return (
            f"Task:\n{session.task}\n\nYour team: {team}. Review the previous round — each "
            f"worker's output and the reviewer's reports. FIRST assess every worker, one per "
            f"line, as '@worker_key [OK]: reason' or '@worker_key [WARN]: what they failed to "
            f"deliver' — call out anyone who slacked or ignored their assignment. THEN reassign "
            f"with '@worker_key: instruction' lines (keys: {keys}). {verdict}"
        )


def _org_levels(order: List[str], sup: Dict[str, str], agents: Dict) -> List[List[str]]:
    """Group roles by depth in the chain of command (top level first).

    Raises :class:`AgentTurnError` on cycles or supervisors that don't exist.
    """
    depth: Dict[str, int] = {}

    def d(r: str, trail: frozenset) -> int:
        if r in depth:
            return depth[r]
        p = sup.get(r)
        if not p:
            depth[r] = 0
        else:
            if p not in agents or p in trail:
                raise AgentTurnError(f"invalid supervisor chain at {r!r}")
            depth[r] = d(p, trail | {r}) + 1
        return depth[r]

    for r in order:
        d(r, frozenset({r}))
    levels: Dict[int, List[str]] = {}
    for r in order:
        levels.setdefault(depth[r], []).append(r)
    return [levels[k] for k in sorted(levels)]


class OrgTeam(Strategy):
    name = "org_team"
    description = (
        "A custom chain of command you design yourself: a top manager delegates "
        "through mid-level managers to workers; each level assigns in parallel, "
        "reports flow back up, and the top manager declares when it's done."
    )
    roles = []          # dynamic: any hierarchy, defined by the request's supervisors map
    dynamic_roles = True
    supports_unlimited = True
    default_rounds = 3

    @staticmethod
    def _mgr_sys(me: str, role: str, reports: List[str], sup_name: Optional[str]) -> str:
        upward = (f"You answer to {sup_name} and report your unit's integrated progress "
                  f"upward." if sup_name else
                  "You are the TOP manager: when the whole task is truly complete, write "
                  "'VERDICT: DONE' on its own line, otherwise 'VERDICT: CONTINUE'. You are "
                  "STRICT about completion — reports' claims are not evidence, only actual "
                  "output is, and every DONE you declare is audited adversarially; a "
                  "premature DONE is a serious failure on your part.")
        return (
            f"You are {me} ({role}), a MANAGER in a chain of command collaborating on one "
            f"shared task. Your direct reports: {', '.join(reports)}. Delegate by writing "
            f"each assignment on its own line as '@role_key: instruction' using those EXACT "
            f"keys; assess prior work with '@role_key [OK]: reason' or '@role_key [WARN]: "
            f"what they failed to deliver' lines. Be specific and demanding but fair. "
            f"{upward}"
        )

    @staticmethod
    def _leaf_sys(me: str, role: str, sup_name: str) -> str:
        return (
            f"You are {me} ({role}), a WORKER reporting to {sup_name}. Carry out the exact "
            f"assignment you are given — thoroughly and concretely, producing real output, "
            f"not a plan to do it later."
        )

    def run(self, session: Session) -> str:
        order = session.role_order or list(session.agents)
        sup = {k: v for k, v in (session.supervisors or {}).items() if v}
        tops = [r for r in order if not sup.get(r)]
        if len(tops) != 1:
            raise AgentTurnError(
                "org_team needs exactly one top manager (one role without a supervisor); "
                f"got {len(tops)}"
            )
        top = tops[0]
        children = {r: [c for c in order if sup.get(c) == r] for r in order}
        if not children[top]:
            raise AgentTurnError("the top manager needs at least one direct report")
        levels = _org_levels(order, sup, session.agents)
        names = {r: session.agents[r].display_name for r in order}
        assignments: Dict[str, Optional[str]] = {}

        for r in order:  # seed the roster
            if r != top:
                session.emit("worker_status", round=0, worker=r, name=names[r],
                             status="idle", note="awaiting assignment", by=sup.get(r, top))

        def process_manager_text(mgr: str, text: str, rnd: int) -> None:
            keys = children[mgr]
            for w, verdict, note in _parse_assessments(text, keys):
                session.emit("worker_status", round=rnd, worker=w, name=names[w],
                             status=("ok" if verdict == "OK" else "warned"),
                             note=note, by=mgr)
            asg = _parse_assignments(text, keys)
            for w in keys:
                assignments[w] = asg.get(w)
                session.emit("worker_status", round=rnd, worker=w, name=names[w],
                             status=("assigned" if asg.get(w) else "idle"),
                             note=(asg.get(w) or "no assignment this round"), by=mgr)

        def own_objective(r: str) -> str:
            return assignments.get(r) or (
                "No explicit assignment was recorded; infer your unit's most useful "
                "objective from the conversation and proceed."
            )

        criteria: List[str] = []
        last_rnd = 1
        for rnd in _rounds_iter(session):
            last_rnd = rnd
            if _wrap_up(session, rnd):
                break

            # 1. Top manager: assess previous round, (re)assign, maybe declare DONE.
            keys = ", ".join(children[top])
            if rnd == 1:
                tinstr = (f"Task:\n{session.task}\n\nFIRST define the acceptance "
                          f"criteria for the whole task — 3 to 7 measurable statements, "
                          f"each on its own line as 'CRITERION: <statement>'; the task "
                          f"is only DONE when every one is verifiably met. THEN break "
                          f"the task down and assign one concrete objective to each "
                          f"direct report, one per line as '@role_key: instruction' "
                          f"(exact keys: {keys}).")
            else:
                if session.min_rounds and rnd < session.min_rounds:
                    verdict = (f"Declaring DONE is NOT allowed before round "
                               f"{session.min_rounds}; write 'VERDICT: CONTINUE'.")
                else:
                    verdict = ("FINALLY: only if EVERY acceptance criterion is "
                               "verifiably met by actual output (a report saying so is "
                               "not evidence), write one line '✔ <criterion> — "
                               "<evidence>' per criterion and then 'VERDICT: DONE'; "
                               "otherwise 'VERDICT: CONTINUE'. A DONE is audited "
                               "adversarially.")
                tinstr = (f"Task:\n{session.task}\n\nReview the previous round (your "
                          f"reports' integrated summaries). FIRST assess each direct "
                          f"report ('@role_key [OK|WARN]: reason'). THEN reassign with "
                          f"'@role_key: instruction' lines (keys: {keys}). {verdict}")
            ttext = _run_turn(session, top, self._mgr_sys(names[top], top, children[top], None),
                              tinstr, rnd, action="assign")
            if rnd == 1:
                criteria = _extract_criteria(ttext)
                if criteria:
                    session.emit("status",
                                 message=f"{len(criteria)} acceptance criteria defined.")
            process_manager_text(top, ttext, rnd)
            if rnd > 1 and rnd >= session.min_rounds and _conductor_done(ttext):
                challenger = children[top][0]  # a direct report audits the DONE
                if _challenge_done(session, challenger, criteria, rnd):
                    session.emit("status",
                                 message=f"DONE declared by {names[top]} in round {rnd} "
                                         f"and CONFIRMED by the auditor.")
                    break
                session.emit("status",
                             message=f"DONE declared in round {rnd} but REJECTED by "
                                     f"the auditor — work continues.")
                continue

            # 2. Delegation flows down, level by level; whole levels run in parallel.
            for level in levels[1:]:
                mids = [r for r in level if children[r]]
                leaves = [r for r in level if not children[r]]
                if mids:
                    specs = [(m, self._mgr_sys(names[m], m, children[m], names[sup[m]]),
                              f"Task:\n{session.task}\n\nYour manager ({names[sup[m]]}) "
                              f"assigned your unit:\n{own_objective(m)}\n\nDecompose it and "
                              f"assign each of your reports, one per line as "
                              f"'@role_key: instruction' (exact keys: "
                              f"{', '.join(children[m])}). Optionally assess their previous "
                              f"round first with '@role_key [OK|WARN]: reason' lines.",
                              rnd, "assign") for m in mids]
                    results = _run_turns_parallel(session, specs)
                    for m in mids:
                        process_manager_text(m, results.get(m, ""), rnd)
                if leaves:
                    specs = [(w, self._leaf_sys(names[w], w, names[sup[w]]),
                              f"Task:\n{session.task}\n\nYour assignment from "
                              f"{names[sup[w]]}:\n{own_objective(w)}\n\nComplete it "
                              f"concretely now.",
                              rnd, "implement") for w in leaves]
                    _run_turns_parallel(session, specs)
                    for w in leaves:
                        session.emit("worker_status", round=rnd, worker=w, name=names[w],
                                     status="delivered", note="", by=sup[w])

            # 3. Reports flow back up: deepest managers first, each integrating
            # their unit's work for their own manager (parallel within a level).
            for level in reversed(levels[1:]):
                mids = [r for r in level if children[r]]
                if not mids:
                    continue
                specs = [(m, self._mgr_sys(names[m], m, children[m], names[sup[m]]),
                          f"Integrate what your reports ({', '.join(children[m])}) produced "
                          f"this round against your unit's objective. Report to "
                          f"{names[sup[m]]} concisely: status, the integrated result so "
                          f"far, and remaining gaps.",
                          rnd, "review") for m in mids]
                _run_turns_parallel(session, specs)
                for m in mids:
                    session.emit("worker_status", round=rnd, worker=m, name=names[m],
                                 status="delivered", note="reported upward", by=sup[m])

        # Top manager consolidates everything into the final deliverable.
        final = _run_turn(
            session, top, self._mgr_sys(names[top], top, children[top], None),
            f"Task:\n{session.task}\n\nThe collaboration is complete. Consolidate your "
            f"organisation's work into the final deliverable — one coherent, complete "
            f"result integrating every unit's contribution.",
            last_rnd, action="integrate",
        )
        return self.finish(session, final)


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
        OrgTeam(),
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
            "supports_unlimited": s.supports_unlimited,
        }
        for s in STRATEGIES.values()
    ]
