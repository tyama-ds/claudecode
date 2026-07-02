# Agent Orchestrator

> 🇯🇵 日本語の使い方ガイド: [README.ja.md](README.ja.md)

Make two coding agents — **Codex** (OpenAI `codex` CLI) and **Claude Code**
(Anthropic `claude` CLI) — collaborate on a task, watch it happen live in a
browser, and extend the cast with **local LLMs** (Ollama / LM Studio) or the
hosted **Claude / GPT APIs**.

```
┌──────────────┐     events (SSE)     ┌──────────────────────────┐
│   Web UI     │ ◀──────────────────  │  Orchestrator engine     │
│ (browser)    │  ──────────────────▶ │  + collaboration strategy│
└──────────────┘     run / stop       └────────────┬─────────────┘
                                                    │ uniform interface
                         ┌──────────────────────────┼───────────────────────────┐
                         ▼              ▼            ▼            ▼               ▼
                     Claude Code     Codex      Claude (API)  GPT (API)     Local LLM
                       (CLI)         (CLI)      anthropic SDK  openai SDK   OpenAI-compat
```

## Why it runs anywhere

The orchestration core **and** the web server use only the Python standard
library — no framework, no build step, no native binary (so nothing for an
antivirus to flag, and no `pip install` to get started). Open it in a browser
and go. Even the hosted-API adapters are SDK-free — they call the providers'
REST endpoints directly via the standard library (`urllib`), so there are no
compiled dependencies anywhere and nothing to install: for the API backends,
an API key is all you need. This keeps the whole tool friendly to locked-down
or audited machines (no `.exe`, no installer, no compiled wheels).

Requires **Python 3.10+**.

## Quick start

```bash
# From the repository root:
python -m agent_orchestrator serve
# → open http://127.0.0.1:8765/
```

Then: type a task, pick a **strategy**, assign a backend to each **role**, and
press **Run collaboration**. Turns stream into the transcript as they happen.

> With nothing installed, the **Mock (offline)** backend works immediately, and
> if the `claude` CLI is on your PATH the **Claude Code (CLI)** backend works
> too. Install `anthropic` / `openai` (see below) to enable the API backends.

## Collaboration strategies

| Strategy | Roles | What happens |
|---|---|---|
| **Implementer + Reviewer** | implementer, reviewer | One builds, the other reviews; repeats until approval or rounds run out. |
| **Debate / Consensus** | debater A, debater B | Both argue distinct positions, then a synthesis turn converges on a final answer. |
| **Planner + Executor** | planner, executor | One plans, the other executes; the planner adjusts each round. |
| **Round-robin (free dialogue)** | agent A, B, C | Several agents discuss openly — each sees the whole thread and addresses the others — then close with a shared conclusion. |
| **Panel + Judge** | contender A, B, C, judge | Three agents argue competing positions; an impartial judge evaluates them and delivers the verdict. |
| **Doc authoring** | writer, editor | Co-write a document: the writer drafts/revises a shared **artifact**, the editor critiques each version, until approved. |
| **Code authoring** | implementer, reviewer | Co-build code: the implementer writes/revises a single code **artifact**, the reviewer critiques each version, until approved. |
| **Workspace build** | implementer, reviewers (1–3) | **Co-build in a real working directory**: the team first discusses and agrees on a design, then the implementer builds while one or more reviewers critique each diff (applying small fixes directly), until **all** approve. |
| **Conductor team** | conductor, workers (2–4), reviewer | A **conductor** splits the task and assigns each worker a subtask; a reviewer checks each worker's output and reports back; the conductor evaluates the team every round — **calling out anyone who didn't deliver** — and reassigns until the work is done. |
| **Custom** | your own (2–5) | Define each participant from scratch — backend, model, and persona — then they discuss and close with a conclusion. |

The authoring strategies build a shared **Artifact** — one evolving document or
code file, shown in its own tab of the feed with **version
history, a Preview/Diff toggle, Copy, and Download**. Editing agents output the
full updated artifact in `<ARTIFACT>…</ARTIFACT>` tags; reviewers give feedback
only.

**Workspace build** goes one step further: the team **co-builds in a real
directory on disk**. It opens with a **design consultation** (round 0) — the
implementer proposes a plan, each reviewer challenges it, and they agree on a
design before any code is written. Then the build loop runs: the implementer
creates/edits files, every reviewer critiques the diff (each may **apply small
fixes directly**), and the round only closes early when **all reviewers
approve**. You can add up to three reviewers, each with its own backend, model,
and persona — e.g. a Claude Code implementer reviewed by both Codex and GPT.

Files are edited two ways, chosen automatically per backend:

- **Native (CLI backends)** — Claude Code and Codex are launched *inside* the
  workspace with their own file tools enabled (`--permission-mode acceptEdits` /
  `--full-auto`), so they genuinely edit the files themselves. The orchestrator
  snapshots the tree around each turn and turns whatever changed into per-file
  unified diffs for the UI.
- **`<FILE>` protocol (API backends and anything else)** — the agent emits each
  changed file in full as `<FILE path="…">…</FILE>`; the orchestrator writes it,
  confined to the workspace root (`..` and absolute paths are refused).

Either way, every change lands in the **Workspace tab** with a file list and
colorized diffs, and stays in the working tree for you to review and commit —
the orchestrator never stages or commits. Set the **workspace directory** in the
UI (blank = the server's launch directory), and tick **Create the directory if
it doesn't exist** to have the orchestrator `mkdir` a fresh folder to build in
(no git involved).

**Reference directory** (any strategy, optional): point it at a local folder and
the orchestrator loads its text files as **read-only context** every agent can
consult — useful for handing the team a spec, an existing codebase, or example
data to work from. It's bounded on purpose (skips `.git`/`node_modules`/binaries,
caps at 40 files · 16 KB each · 120 KB total, truncating the rest) so prompts stay
manageable. The files are never edited; to have agents modify files, use the
Workspace above.

**Conductor team** models a boss-and-team workflow: the conductor speaks to the
team in a small line protocol (`@worker_1: <subtask>` to assign,
`@worker_2 [WARN]: <what's missing>` to call someone out, `VERDICT: DONE` to
finish). A **Team** tab in the feed shows each worker's live status —
*assigned → delivered → approved ✓ / called out ⚠* — so you can see at a glance
who's pulling their weight. The conductor consolidates the team's work into the
final deliverable. The number of workers (2–4) is set in the UI.

For **every role** you can independently choose the **backend, model, and
persona** (system prompt) right in the UI — mix Claude Code, Codex, GPT, and
local models in a single run, and override any role's instructions.

All strategies share a **scratchpad** — a team blackboard any agent can write to
by adding `NOTE:` lines. It appears pinned above the transcript so the shared
state of the collaboration is always visible.

**Conversation context** is shared natively where possible: backends that
support it (the CLI and API adapters) receive the running conversation as a
structured message **history** — each agent's own turns as `assistant`, the
others' as `user`. Backends that can't use history fall back automatically to
the transcript embedded in the prompt. Each turn is tagged `ctx: history` or
`ctx: prompt` so you can see which path was used.

**The console** keeps the feed organized in tabs — *Transcript / Artifact /
Workspace / Team* — with a dot badge when a background tab updates. During a
run, a **live agent board** docks on the right showing, for every participant,
what it is doing *right now* — designing, implementing, or reviewing, with a
pulsing working state, round number, and elapsed seconds. Above it, an animated
**interaction graph** (pure SVG, no dependencies) shows who is talking to whom:
the active agent glows, dots flow along the edges — implementer → workspace
while coding, workspace → reviewer while reviewing, conductor → worker on an
assignment — coloured green for approvals and red for change requests /
call-outs, with a running caption underneath (e.g. *"Codex → Claude Code ·
requests changes"*). Quality of
life built in: a **session history** (▤) that can reopen any past or running
session with its full transcript replayed; one-click **Export** of the whole
transcript as Markdown; form values (task, strategy, directories) that survive a
reload; smart autoscroll with a *↓ Latest* pill instead of yanking you to the
bottom; a live elapsed-seconds counter on in-progress turns; `Ctrl+Enter` to
run; an **English / 日本語** toggle; and an auto / light / dark theme that follows
your OS by default.

## Backends (adapters)

| Backend | Needs |
|---|---|
| Mock (offline) | nothing — deterministic, for demos/tests |
| Claude Code (CLI) | the `claude` CLI on PATH |
| Codex (CLI) | the `codex` CLI on PATH |
| Claude (API) | `ANTHROPIC_API_KEY` — no install (stdlib HTTP) |
| GPT (API) | `OPENAI_API_KEY` — no install (stdlib HTTP) |
| Local LLM | a local OpenAI-compatible server (e.g. Ollama) — no install |

Optional config goes in a `.env` file (see `.env.example`). There are **no
packages to install** — only an API key for each hosted backend you want.

You can also enter API keys, models, base URLs, and an HTTP(S) **proxy** in the
in-app **Settings** (⚙) — used whenever the matching environment variable isn't
set, and held in memory for the local server only (never displayed back). Point
any provider's base URL at any OpenAI-compatible endpoint.

Local-LLM calls **connect directly by default** (the endpoint is usually on
localhost). If you need them routed through the proxy too, tick **Send via
proxy** under *Local LLM* in Settings (or set `LOCAL_LLM_USE_PROXY=1`).

## Headless usage

```bash
# List what's available in this environment:
python -m agent_orchestrator agents

# Run a collaboration in the terminal:
python -m agent_orchestrator run \
    --task "Write a function that de-duplicates a list, preserving order." \
    --strategy implementer_reviewer --rounds 2 \
    --agent implementer=claude_code --agent reviewer=codex
```

## Extending it

Add a new participant by implementing one method:

```python
from agent_orchestrator.adapters.base import AgentAdapter

class MyAdapter(AgentAdapter):
    kind = "my_backend"
    def _generate(self, prompt, system, history):
        return my_llm_call(system, history, prompt)
```

Register it in `agent_orchestrator/adapters/__init__.py` (`_BUILDERS` +
`_CATALOG_META`) and it appears in the UI automatically. New collaboration
patterns are subclasses of `Strategy` in `orchestrator/strategies.py`.

## Tests

```bash
python -m unittest discover -s agent_orchestrator/tests -v
```

## Layout

```
agent_orchestrator/
├── config.py            # enums + settings (env-driven)
├── cli.py / __main__.py # `serve`, `run`, `agents`
├── adapters/            # mock / cli_agent / api_agent + registry
├── orchestrator/        # events, session, strategies, engine
├── server/              # stdlib HTTP + SSE server
│   └── static/          # index.html, style.css, app.js
└── tests/               # unittest suite (runs on mock adapters)
```
