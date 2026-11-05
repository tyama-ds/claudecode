"""Offline, reproducible packet preparation and recorded-agent answer import."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from factcheck.recorded import (
    CONDITIONS, DEFAULT_SEED, grade_recorded, prepare_check, prepare_fill, read_json,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="公開packetと収集済みLLM回答を使う予備実験。APIへの通信は行いません。")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-fill", "prepare-check", "grade"):
        command = commands.add_parser(name)
        command.add_argument("--reports", required=True, type=Path)
        command.add_argument("--gold", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if name == "prepare-fill":
            command.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
            command.add_argument("--condition-prompts", type=Path, help="任意の条件別プロンプトJSON。条件名→文字列。")
        else:
            command.add_argument("--fills", nargs="+", required=True, type=Path)
            command.add_argument("--fill-packet", type=Path, required=name == "grade", help="prepare-checkでも指定を推奨。省略時は既定packetを再生成。")
        if name == "prepare-check":
            command.add_argument("--seed", type=int, default=DEFAULT_SEED)
        if name == "grade":
            command.add_argument("--check-packet", required=True, type=Path)
            command.add_argument("--checks", nargs="+", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-fill":
            result = prepare_fill(args.reports, args.gold, args.output, conditions=args.conditions,
                                  condition_prompts=read_json(args.condition_prompts) if args.condition_prompts else None)
        elif args.command == "prepare-check":
            result = prepare_check(args.reports, args.gold, args.fills, args.output,
                                   fill_packet=args.fill_packet, seed=args.seed)
        else:
            result = grade_recorded(args.reports, args.gold, args.fill_packet, args.fills,
                                    args.check_packet, args.checks, args.output)
        overview = {"saved": str(args.output.resolve())}
        if args.command == "grade":
            overview.update({"run_mode": result["run_mode"], "status": result["status"], "counts": result["counts"]})
        else:
            overview.update({"packet_sha256": result["packet_sha256"], "stage": result["stage"], "tasks": len(result["items"]), "conditions": result["conditions"]})
        print(json.dumps(overview, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, TypeError) as exc:
        print(f"Recorded pilot rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
