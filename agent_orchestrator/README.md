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
and go. The hosted-API adapters use the official `anthropic` / `openai` SDKs and
are imported lazily, so they are only needed if you actually pick them.

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

## Backends (adapters)

| Backend | Needs |
|---|---|
| Mock (offline) | nothing — deterministic, for demos/tests |
| Claude Code (CLI) | the `claude` CLI on PATH |
| Codex (CLI) | the `codex` CLI on PATH |
| Claude (API) | `pip install anthropic` + `ANTHROPIC_API_KEY` |
| GPT (API) | `pip install openai` + `OPENAI_API_KEY` |
| Local LLM | `pip install openai` + a local OpenAI-compatible server (e.g. Ollama) |

Optional config goes in a `.env` file (see `.env.example`).

```bash
pip install -r agent_orchestrator/requirements.txt   # only for the API backends
```

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
