"""Command-line interface.

Usage:
    python -m agent_orchestrator serve [--host H] [--port P] [--open]
    python -m agent_orchestrator run --task "..." [--strategy S] [--rounds N]
        [--agent role=adapter_id ...]
    python -m agent_orchestrator agents     # list selectable backends
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict

from .adapters import build_adapter, catalog
from .orchestrator import SessionManager, start_session, strategy_metadata
from .orchestrator.events import EventBus
from .orchestrator.strategies import get_strategy


def _parse_agents(pairs) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--agent expects role=adapter_id, got {pair!r}")
        role, _, adapter_id = pair.partition("=")
        out[role.strip()] = adapter_id.strip()
    return out


def cmd_serve(args) -> int:
    from .server import run
    run(host=args.host, port=args.port, open_browser=args.open)
    return 0


def cmd_agents(args) -> int:
    print("Selectable agent backends:\n")
    for entry in catalog():
        mark = "✓" if entry["available"] else "✗"
        line = f"  {mark} {entry['id']:<12} {entry['label']}"
        if not entry["available"]:
            line += f"   ({entry['reason']})"
        print(line)
    print("\nStrategies:\n")
    for st in strategy_metadata():
        roles = ", ".join(r["key"] for r in st["roles"])
        print(f"  • {st['name']:<22} roles: {roles}")
    return 0


def cmd_run(args) -> int:
    try:
        strategy = get_strategy(args.strategy)
    except ValueError as exc:
        raise SystemExit(str(exc))

    chosen = _parse_agents(args.agent)
    agents = {}
    for role_key, _label in strategy.roles:
        adapter_id = chosen.get(role_key, "mock")
        adapter = build_adapter({"id": adapter_id, "name": role_key})
        ok, reason = adapter.available()
        if not ok:
            raise SystemExit(f"agent for role '{role_key}' ({adapter_id}) unavailable: {reason}")
        agents[role_key] = adapter

    manager = SessionManager()
    session = manager.create(args.task, args.strategy, args.rounds, agents)

    q = session.bus.subscribe()
    start_session(session)

    # Drain events to the terminal as they happen.
    while True:
        item = q.get()
        if EventBus.is_closed_sentinel(item):
            break
        _print_event(item)
    return 0 if session.status == "done" else 1


def _print_event(event) -> None:
    t = event.type
    d = event.data
    if t == "session_start":
        print(f"\n=== {d['strategy']} · {d['rounds']} rounds ===")
    elif t == "turn_start":
        print(f"\n--- {d['agent']} [{d['role']}] · round {d['round']} ---")
    elif t == "turn_end":
        if d.get("ok"):
            print(d["content"])
        else:
            print(f"[FAILED] {d.get('error')}")
    elif t == "status":
        print(f"(status) {d['message']}")
    elif t == "result":
        print("\n========== FINAL DELIVERABLE ==========")
        print(d["content"])
    elif t == "error":
        print(f"\n[ERROR] {d['message']}", file=sys.stderr)
    elif t == "session_end":
        print(f"\n=== session {d['status']} ===")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent_orchestrator", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="run the web UI (default)")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--open", action="store_true", help="open a browser window")
    p_serve.set_defaults(func=cmd_serve)

    p_run = sub.add_parser("run", help="run a collaboration headlessly")
    p_run.add_argument("--task", required=True)
    p_run.add_argument("--strategy", default="implementer_reviewer")
    p_run.add_argument("--rounds", type=int, default=2)
    p_run.add_argument("--agent", action="append", metavar="ROLE=ID",
                       help="assign a backend to a role (repeatable)")
    p_run.set_defaults(func=cmd_run)

    p_agents = sub.add_parser("agents", help="list available backends and strategies")
    p_agents.set_defaults(func=cmd_agents)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        # Default to serving the UI.
        return cmd_serve(argparse.Namespace(host=None, port=None, open=False))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
