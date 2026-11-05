"""Build claim packets and apply independent preservation judgments, without LLM calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import replication as rep
from factcheck.claim_guard import build_block_packet, build_guard_packet, gate_repair, preview_repair, schema_contract, sha256_json


def _inputs(args):
    paths = {key: Path(getattr(args, key)) for key in ("original", "candidate", "ledger")}
    raw = {key: path.read_bytes() for key, path in paths.items()}
    records = {key: rep._read_json(path) for key, path in paths.items()}
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in raw.items()}
    return records, raw, hashes


def _artifacts(output, raw, packet):
    return {output / "original.frozen.json": raw["original"],
            output / "candidate.raw.json": raw["candidate"], output / "ledger.raw.json": raw["ledger"],
            output / "guard_packet.json": rep._json_bytes(packet),
            output / "candidate.applied.json": rep._json_bytes(packet["candidate_applied"]),
            output / "input_freeze.json": rep._json_bytes({"schema_version": 1,
                "input_file_sha256": packet["input_file_sha256"], "input_sha256": packet["input_sha256"],
                "guard_packet_sha256": sha256_json(packet)})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    blocks = commands.add_parser("blocks", help="Write deterministic original block packet")
    blocks.add_argument("--original", type=Path, required=True)
    blocks.add_argument("--output", type=Path, required=True)
    describe = commands.add_parser("describe", help="Write the frozen worker schema contract")
    describe.add_argument("--output", type=Path, required=True)
    for command in ("prepare", "preview", "apply"):
        sub = commands.add_parser(command, help="Freeze guard inputs" if command == "prepare" else "Filter candidate once through independent guard")
        for field in ("original", "candidate", "ledger"):
            sub.add_argument(f"--{field}", type=Path, required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
        if command in {"apply", "preview"}:
            sub.add_argument("--guard-packet", type=Path, required=True)
            sub.add_argument("--guard", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "describe":
            contract = schema_contract()
            rep._write_immutable({args.output: rep._json_bytes(contract)})
            result = {"output": str(args.output), "contract_sha256": sha256_json(contract)}
        elif args.command == "blocks":
            packet = build_block_packet(rep._read_json(args.original))
            rep._write_immutable({args.output: rep._json_bytes(packet)})
            result = {"output": str(args.output), "block_count": len(packet["blocks"]),
                      "block_packet_sha256": sha256_json(packet), "original_sha256": packet["original_sha256"]}
        else:
            records, raw, file_hashes = _inputs(args)
            packet = build_guard_packet(**records, input_file_sha256=file_hashes)
            outputs = _artifacts(args.output_dir, raw, packet)
            if args.command == "prepare":
                result = {"output_dir": str(args.output_dir), "guard_packet_sha256": sha256_json(packet),
                          "input_sha256": packet["input_sha256"], "input_file_sha256": file_hashes}
            else:
                frozen_packet = rep._read_json(args.guard_packet)
                if packet != frozen_packet:
                    raise ValueError("Frozen guard packet/raw input hashes do not match current inputs")
                guard = rep._read_json(args.guard)
                operation = preview_repair if args.command == "preview" else gate_repair
                gated = operation(**records, guard=guard, guard_packet=frozen_packet)
                if args.command == "preview":
                    outputs.update({args.output_dir / "guard.preview.raw.json": args.guard.read_bytes(),
                        args.output_dir / "gate.preview.json": rep._json_bytes(gated),
                        args.output_dir / "selected_report.json": rep._json_bytes({key: gated["selected_applied"][key] for key in ("text", "sources")}),
                        args.output_dir / "selection.json": rep._json_bytes(gated["selection"])})
                else:
                    outputs.update({args.output_dir / "guard.raw.json": args.guard.read_bytes(),
                        args.output_dir / "gate.decisions.json": rep._json_bytes(gated),
                        args.output_dir / "repair.effective.json": rep._json_bytes(gated["effective_repair"]),
                        args.output_dir / "final.applied.json": rep._json_bytes(gated["final_applied"]),
                        args.output_dir / "final_report.json": rep._json_bytes({key: gated["final_applied"][key] for key in ("text", "sources")}),
                        args.output_dir / "summary.json": rep._json_bytes(gated["summary"])})
                result = gated["summary"]
            rep._write_immutable(outputs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, TypeError, UnicodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
