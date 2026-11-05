"""Opt into conservative ledger support-label normalization; retain all raw inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import replication as rep
from factcheck import claim_guard_v5 as base
from factcheck.ledger_consistency import normalize_ledger, schema_contract, verify_normalization


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    describe = sub.add_parser("describe")
    describe.add_argument("--output", type=Path, required=True)
    normalize = sub.add_parser("normalize")
    for key in ("original", "candidate", "ledger"):
        normalize.add_argument(f"--{key}", type=Path, required=True)
    normalize.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "describe":
            contract = schema_contract()
            rep._write_immutable({args.output: rep._json_bytes(contract)})
            result = {"output": str(args.output), "contract_sha256": base.sha256_json(contract)}
        else:
            paths = {key: getattr(args, key) for key in ("original", "candidate", "ledger")}
            raw = {key: path.read_bytes() for key, path in paths.items()}
            records = {key: rep._read_json(path) for key, path in paths.items()}
            normalized = normalize_ledger(**records)
            verified = verify_normalization(records["original"], records["candidate"], records["ledger"],
                                            normalized["ledger"], normalized["audit"])
            ledger_bytes = rep._json_bytes(normalized["ledger"])
            original_hashes = {key: hashlib.sha256(value).hexdigest() for key, value in raw.items()}
            normalized_hashes = {**original_hashes, "ledger": hashlib.sha256(ledger_bytes).hexdigest()}
            packet = base.build_guard_packet(records["original"], records["candidate"], normalized["ledger"],
                                             input_file_sha256=normalized_hashes)
            freeze = {"schema_version": 1, "raw_input_file_sha256": original_hashes,
                      "normalized_input_file_sha256": normalized_hashes,
                      "normalization_audit_sha256": base.sha256_json(normalized["audit"]),
                      "guard_packet_sha256": base.sha256_json(packet), "verification": verified}
            outputs = {
                args.output_dir / "original.frozen.json": raw["original"],
                args.output_dir / "candidate.raw.json": raw["candidate"],
                args.output_dir / "ledger.raw.json": raw["ledger"],
                args.output_dir / "ledger.normalized.json": ledger_bytes,
                args.output_dir / "normalization.audit.json": rep._json_bytes(normalized["audit"]),
                args.output_dir / "normalization.freeze.json": rep._json_bytes(freeze),
                args.output_dir / "guard_packet.json": rep._json_bytes(packet),
            }
            rep._write_immutable(outputs)
            result = {"output_dir": str(args.output_dir), **normalized["audit"]["counts"],
                      "before_validation": normalized["audit"]["before_validation"], **verified}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, TypeError, UnicodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
