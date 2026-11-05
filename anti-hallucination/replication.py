"""Validate and summarize the preregistered three-report Deep Research trial.

This module never calls a model, fetches evidence, changes raw records, or assigns
semantic labels. The independently supplied labels are provisional AI judgments.
Run ``python replication.py --help`` for the two artifact-producing commands.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import html
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import quote as urlquote

from factcheck.repair import apply_repair


FIXED_FILES = (
    "protocol.json", "repair_instructions.txt", "reference_instructions.txt",
    "post_instructions.txt", "generation_prompts.json",
)
RAW_FILES = (
    "input/original_report.json", "repair/repair.raw.json",
    "evaluation/baseline.raw.json",
)
AFTER_STATUSES = {"resolved", "partially_resolved", "unresolved", "unverifiable", "not_an_original_problem"}
PROBLEM_STATUSES = {"confirmed_problem", "unresolved"}
PATCH_STATUSES = {"acceptable", "regression", "unnecessary_change", "unverifiable"}
CONTROL_STATUSES = {"preserved", "omitted", "corrupted", "unverifiable"}
ISSUE_TAGS = {"factual", "citation", "condition", "generalisation", "arithmetic"}
NEW_TAGS = {"factual", "citation", "condition", "generalisation", "meaning_drift", "information_loss"}


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _array(value, label):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _string(value, label, *, nonempty=False):
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(f"{label} must be {'a nonempty string' if nonempty else 'a string'}")
    return value


def _boolean(value, label):
    if type(value) is not bool:
        raise ValueError(f"{label} must be true or false (JSON boolean)")
    return value


def _enum(value, choices, label):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{label} must be one of {sorted(choices)}")
    return value


def _strings(value, label):
    for i, item in enumerate(_array(value, label)):
        _string(item, f"{label}[{i}]")
    return value


def _tags(value, allowed, label):
    for tag in _strings(value, label):
        _enum(tag, allowed, label)


def _quote(value, text, label, *, allow_empty=False):
    _string(value, label)
    if not value and allow_empty:
        return
    if not value or value not in text:
        raise ValueError(f"{label} must be a nonempty exact substring of its report")


def _rows_by_id(value, key, label):
    rows = {}
    for i, row in enumerate(_array(value, label)):
        _object(row, f"{label}[{i}]")
        identity = _string(row.get(key), f"{label}[{i}].{key}", nonempty=True)
        if identity in rows:
            raise ValueError(f"duplicate {key}: {identity}")
        rows[identity] = row
    return rows


def _id_refs(value, known, label):
    values = _strings(value, label)
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate IDs")
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"{label} contains unknown IDs: {sorted(unknown)}")
    return values


def _all_ids(actual, expected, label):
    if set(actual) != set(expected):
        raise ValueError(f"{label}: missing IDs {sorted(set(expected) - set(actual))}; "
                         f"unknown IDs {sorted(set(actual) - set(expected))}")


def _execution(record, label, *, input_files=False):
    execution = _object(record.get("execution"), f"{label}.execution")
    times = []
    for key in ("started_utc", "finished_utc"):
        value = _string(execution.get(key), f"{label}.execution.{key}", nonempty=True)
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"{label}.execution.{key} must be an ISO datetime") from None
        if timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
            raise ValueError(f"{label}.execution.{key} must specify UTC")
        times.append(timestamp)
    if times[1] < times[0]:
        raise ValueError(f"{label}.execution finishes before it starts")
    calls = execution.get("web_calls")
    if type(calls) is not int or calls < 0 or calls > 25:
        raise ValueError(f"{label}.execution.web_calls must be an integer from 0 through 25")
    if input_files:
        _strings(execution.get("input_files"), f"{label}.execution.input_files")
    return (times[1] - times[0]).total_seconds()


def _read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-JSON constant {value} in {path}")
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=reject_constant, object_pairs_hook=unique_keys)


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_immutable(outputs):
    """Preflight the whole batch before exclusively creating absent artifacts."""
    for path, payload in outputs.items():
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite different existing output: {path}")
    for path, payload in outputs.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                stream.write(payload)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"Output appeared with different content: {path}") from None


def validate_preregistration(root):
    root = Path(root)
    prereg = _object(_read_json(root / "preregistration.json"), "preregistration")
    if prereg.get("generation_started") is not False:
        raise ValueError("preregistration.generation_started must be false in the frozen record")
    hashes = _object(prereg.get("sha256"), "preregistration.sha256")
    for name in FIXED_FILES:
        if hashes.get(name) != _sha(root / name):
            raise ValueError(f"Preregistration SHA256 mismatch: {name}")
    protocol = _object(_read_json(root / "protocol.json"), "protocol")
    topics = _rows_by_id(protocol.get("topics"), "report_id", "protocol.topics")
    if set(topics) != {"R1", "R2", "R3"}:
        raise ValueError("This trial must retain the preregistered R1, R2 and R3")
    return protocol, {**{name: _sha(root / name) for name in FIXED_FILES},
                      "preregistration.json": _sha(root / "preregistration.json")}


def _baseline_freeze(folder):
    """Verify the operator's independent baseline freeze when one was recorded."""
    path = folder / "evaluation/baseline.freeze.json"
    if not path.exists():
        return None
    record = _object(_read_json(path), "baseline.freeze")
    frozen_at = _string(record.get("frozen_at_utc"), "baseline.freeze.frozen_at_utc", nonempty=True)
    try:
        timestamp = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("baseline.freeze.frozen_at_utc must be an ISO UTC datetime") from None
    if timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
        raise ValueError("baseline.freeze.frozen_at_utc must specify UTC")
    expected = record.get("sha256")
    if expected != _sha(folder / "evaluation/baseline.raw.json"):
        raise ValueError("Independent baseline.freeze SHA256 mismatch")
    return {"frozen_at_utc": frozen_at, "sha256": expected, "record_sha256": _sha(path)}


def _verify_existing_freeze(folder, fixed_hashes):
    path = folder / "evaluation/input_freeze.json"
    if not path.exists():
        return
    record = _object(_read_json(path), "input_freeze")
    if record.get("fixed_instruction_sha256") != fixed_hashes:
        raise ValueError("Frozen instruction/preregistration hashes changed")
    raw_hashes = _object(record.get("input_sha256"), "input_freeze.input_sha256")
    if set(raw_hashes) != set(RAW_FILES):
        raise ValueError("input_freeze must cover every required raw input")
    for name, expected in raw_hashes.items():
        path = folder / name
        if not path.exists() or _sha(path) != expected:
            raise ValueError(f"Frozen input missing or changed: {name}")
    if record.get("independent_baseline_freeze") != _baseline_freeze(folder):
        raise ValueError("Independent baseline freeze record changed")


def validate_original(original):
    _object(original, "original")
    _string(original.get("text"), "original.text", nonempty=True)
    # The common repair applicator validates and preserves the full source list.
    apply_repair(original, {"patches": [], "added_sources": []})


def validate_repair(original, repair):
    _object(repair, "repair")
    _execution(repair, "repair", input_files=True)
    model_id = repair["execution"].get("model_id")
    if model_id is not None:
        _string(model_id, "repair.execution.model_id")
    text = original["text"]
    findings = _rows_by_id(repair.get("findings"), "finding_id", "repair.findings")
    for identity, row in findings.items():
        label = f"repair.findings.{identity}"
        _quote(row.get("quote"), text, f"{label}.quote")
        _enum(row.get("status"), PROBLEM_STATUSES, f"{label}.status")
        _tags(row.get("tags"), ISSUE_TAGS, f"{label}.tags")
        _string(row.get("diagnosis"), f"{label}.diagnosis", nonempty=True)
        for i, evidence in enumerate(_array(row.get("evidence"), f"{label}.evidence")):
            _object(evidence, f"{label}.evidence[{i}]")
            for key in ("url", "locator", "finding"):
                _string(evidence.get(key), f"{label}.evidence[{i}].{key}")
    for key in ("checked_controls", "unresolved"):
        for i, row in enumerate(_array(repair.get(key), f"repair.{key}")):
            _object(row, f"repair.{key}[{i}]")
            _quote(row.get("quote"), text, f"repair.{key}[{i}].quote")
            _string(row.get("reason"), f"repair.{key}[{i}].reason")
            if key == "checked_controls":
                _strings(row.get("source_urls"), f"repair.{key}[{i}].source_urls")
    patches = _rows_by_id(repair.get("patches"), "patch_id", "repair.patches")
    for identity, row in patches.items():
        _id_refs(row.get("finding_ids"), findings, f"repair.patches.{identity}.finding_ids")
        _enum(row.get("action"), {"replace", "qualify", "relink"}, f"repair.patches.{identity}.action")
    _strings(repair.get("limitations"), "repair.limitations")
    applied = apply_repair(original, repair)
    original_ids = [int(str(s["citation_id"])) for s in original["sources"]
                    if str(s["citation_id"]).isdigit()]
    max_id = max(original_ids, default=0)
    for source in repair["added_sources"]:
        identity = source["citation_id"]
        if not str(identity).isdigit() or int(identity) <= max_id:
            raise ValueError("Every added citation_id must be a number above the original maximum")
    return applied


def validate_baseline(original, baseline):
    _object(baseline, "baseline")
    _execution(baseline, "baseline", input_files=True)
    if _boolean(baseline.get("human_adjudicated"), "baseline.human_adjudicated"):
        raise ValueError("The fixed trial records human_adjudicated=false")
    text = original["text"]
    issues = _rows_by_id(baseline.get("issues"), "issue_id", "baseline.issues")
    for identity, row in issues.items():
        label = f"baseline.issues.{identity}"
        _quote(row.get("quote"), text, f"{label}.quote")
        _enum(row.get("status"), PROBLEM_STATUSES, f"{label}.status")
        _tags(row.get("tags"), ISSUE_TAGS, f"{label}.tags")
        for key in ("reason", "evidence", "minimum_resolution"):
            _string(row.get(key), f"{label}.{key}")
        _strings(row.get("source_urls"), f"{label}.source_urls")
    inventory = _rows_by_id(baseline.get("inventory"), "item_id", "baseline.inventory")
    for identity, row in inventory.items():
        label = f"baseline.inventory.{identity}"
        _quote(row.get("quote"), text, f"{label}.quote")
        _enum(row.get("kind"), {"numeric", "citation", "generalisation", "proposal"}, f"{label}.kind")
        _enum(row.get("status"), {"supported", "confirmed_problem", "unresolved", "proposal"}, f"{label}.status")
        _id_refs(row.get("issue_ids"), issues, f"{label}.issue_ids")
        for key in ("description", "evidence"):
            _string(row.get(key), f"{label}.{key}")
        for i, occurrence in enumerate(_array(row.get("occurrences"), f"{label}.occurrences")):
            _object(occurrence, f"{label}.occurrences[{i}]")
            _quote(occurrence.get("quote"), text, f"{label}.occurrences[{i}].quote")
        _strings(row.get("source_urls"), f"{label}.source_urls")
    controls = _rows_by_id(baseline.get("controls"), "control_id", "baseline.controls")
    for identity, row in controls.items():
        label = f"baseline.controls.{identity}"
        _quote(row.get("quote"), text, f"{label}.quote")
        for key in ("description", "reason"):
            _string(row.get(key), f"{label}.{key}")
        _strings(row.get("source_urls"), f"{label}.source_urls")
    _strings(baseline.get("limitations"), "baseline.limitations")


def validate_post(original, revised, repair, baseline, post):
    _object(post, "post")
    _execution(post, "post")
    if _boolean(post.get("human_adjudicated"), "post.human_adjudicated"):
        raise ValueError("The fixed trial records human_adjudicated=false")
    text = revised["text"]
    findings = {row["finding_id"]: row for row in repair["findings"]}
    baseline_issues = {row["issue_id"]: row for row in baseline["issues"]}
    issues = _rows_by_id(post.get("issues"), "issue_id", "post.issues")
    _all_ids(issues, baseline_issues, "post.issues")
    for identity, row in issues.items():
        label = f"post.issues.{identity}"
        _enum(row.get("baseline_status"), PROBLEM_STATUSES, f"{label}.baseline_status")
        if row["baseline_status"] != baseline_issues[identity]["status"]:
            raise ValueError(f"{label}.baseline_status changed the frozen baseline")
        detected = _boolean(row.get("detected"), f"{label}.detected")
        refs = _id_refs(row.get("finding_ids"), findings, f"{label}.finding_ids")
        if detected and not refs:
            raise ValueError(f"{label}: detected=true requires existing finding_ids")
        _string(row.get("detection_reason"), f"{label}.detection_reason", nonempty=True)
        status = _enum(row.get("after_status"), AFTER_STATUSES, f"{label}.after_status")
        _quote(row.get("after_quote"), text, f"{label}.after_quote", allow_empty=True)
        removal = _boolean(row.get("resolved_by_removal"), f"{label}.resolved_by_removal")
        if removal and status not in {"resolved", "partially_resolved"}:
            raise ValueError(f"{label}: resolved_by_removal requires a resolution status")
        _string(row.get("reason"), f"{label}.reason", nonempty=True)
        _strings(row.get("source_urls"), f"{label}.source_urls")
    reviews = _rows_by_id(post.get("patch_reviews"), "patch_id", "post.patch_reviews")
    _all_ids(reviews, {row["patch_id"] for row in repair["patches"]}, "post.patch_reviews")
    problem_ids = set()
    for identity, row in reviews.items():
        label = f"post.patch_reviews.{identity}"
        _enum(row.get("status"), PATCH_STATUSES, f"{label}.status")
        _string(row.get("reason"), f"{label}.reason", nonempty=True)
        problems = _rows_by_id(row.get("new_problems"), "problem_id", f"{label}.new_problems")
        for problem_id, problem in problems.items():
            if problem_id in problem_ids:
                raise ValueError(f"duplicate problem_id across patches: {problem_id}")
            problem_ids.add(problem_id)
            name = f"{label}.new_problems.{problem_id}"
            _tags(problem.get("tags"), NEW_TAGS, f"{name}.tags")
            _enum(problem.get("status"), PROBLEM_STATUSES, f"{name}.status")
            _quote(problem.get("quote"), text, f"{name}.quote")
            _string(problem.get("reason"), f"{name}.reason", nonempty=True)
            _strings(problem.get("source_urls"), f"{name}.source_urls")
    controls = _rows_by_id(post.get("controls"), "control_id", "post.controls")
    _all_ids(controls, {row["control_id"] for row in baseline["controls"]}, "post.controls")
    for identity, row in controls.items():
        label = f"post.controls.{identity}"
        status = _enum(row.get("status"), CONTROL_STATUSES, f"{label}.status")
        _quote(row.get("after_quote"), text, f"{label}.after_quote", allow_empty=status in {"omitted", "unverifiable"})
        _string(row.get("reason"), f"{label}.reason", nonempty=True)
    false_findings = _rows_by_id(post.get("false_findings"), "finding_id", "post.false_findings")
    _id_refs(list(false_findings), findings, "post.false_findings")
    for identity, row in false_findings.items():
        _enum(row.get("status"), {"false_alarm", "unverifiable"}, f"post.false_findings.{identity}.status")
        _string(row.get("reason"), f"post.false_findings.{identity}.reason", nonempty=True)
    for i, row in enumerate(_array(post.get("posthoc_original_findings"), "post.posthoc_original_findings")):
        label = f"post.posthoc_original_findings[{i}]"
        _object(row, label)
        _quote(row.get("quote"), original["text"], f"{label}.quote")
        _tags(row.get("tags"), ISSUE_TAGS, f"{label}.tags")
        _enum(row.get("status"), PROBLEM_STATUSES, f"{label}.status")
        present = _boolean(row.get("still_present_after"), f"{label}.still_present_after")
        _quote(row.get("after_quote"), text, f"{label}.after_quote", allow_empty=not present)
        _string(row.get("reason"), f"{label}.reason", nonempty=True)
        _strings(row.get("source_urls"), f"{label}.source_urls")
    _strings(post.get("limitations"), "post.limitations")


def _prepared_outputs(root, report_id, fixed_hashes):
    folder = root / report_id
    independent_baseline_freeze = _baseline_freeze(folder)
    _verify_existing_freeze(folder, fixed_hashes)
    original, repair, baseline = [_read_json(folder / name) for name in RAW_FILES]
    validate_original(original)
    applied = validate_repair(original, repair)
    validate_baseline(original, baseline)
    revised = {**copy.deepcopy(original), "text": applied["text"], "sources": applied["sources"]}
    raw_hashes = {name: _sha(folder / name) for name in RAW_FILES}
    freeze = {
        "schema_version": 1, "report_id": report_id,
        "fixed_instruction_sha256": fixed_hashes, "input_sha256": raw_hashes,
        "baseline_fixed_before_post_packet": True,
        "independent_baseline_freeze": independent_baseline_freeze,
        "baseline_execution_finished_utc": baseline["execution"]["finished_utc"],
        "temporal_attestation": "Hashes fix the baseline before packet creation; chronology of independent generation is recorded by the experiment operator, not inferred from filesystem timestamps.",
    }
    # Allowlist literal changes: never leak patch reasons, actions, or citations
    # supplied as justification into the independent evaluator's post packet.
    literal_keys = ("patch_id", "before", "after", "start_char", "end_char", "after_start_char", "after_end_char")
    packet = {
        "report_id": report_id, "original": original, "revised": revised,
        "literal_changes": [{key: patch[key] for key in literal_keys} for patch in applied["patches"]],
        "writer_findings": copy.deepcopy(repair["findings"]),
        "input_sha256": raw_hashes, "fixed_instruction_sha256": fixed_hashes,
    }
    outputs = {
        folder / "repair/applied.json": _json_bytes(applied),
        folder / "repair/revised_report.json": _json_bytes(revised),
        folder / "repair/revised_report.txt": applied["text"].encode("utf-8"),
        folder / "evaluation/input_freeze.json": _json_bytes(freeze),
        folder / "evaluation/post_packet.json": _json_bytes(packet),
    }
    return outputs, original, repair, baseline, revised, applied


def prepare_post(root, report_id):
    root = Path(root).resolve()
    protocol, hashes = validate_preregistration(root)
    if report_id not in {row["report_id"] for row in protocol["topics"]}:
        raise ValueError(f"Unknown report ID: {report_id}")
    outputs, *_ = _prepared_outputs(root, report_id, hashes)
    _write_immutable(outputs)
    return {"report_id": report_id, "post_packet": str(root / report_id / "evaluation/post_packet.json"),
            "files": [str(path) for path in outputs]}


def _counts(rows, key, choices):
    counts = Counter(row[key] for row in rows)
    return {name: counts[name] for name in sorted(choices)}


def summarize_complete(original, revised, repair, baseline, post):
    confirmed = [row for row in post["issues"] if row["baseline_status"] == "confirmed_problem"]
    unresolved = [row for row in post["issues"] if row["baseline_status"] == "unresolved"]
    excluded = [row for row in confirmed if row["after_status"] == "not_an_original_problem"]
    eligible = [row for row in confirmed if row["after_status"] != "not_an_original_problem"]
    detected = sum(row["detected"] for row in eligible)
    resolved = sum(row["after_status"] == "resolved" for row in eligible)
    problems = [problem for row in post["patch_reviews"] for problem in row["new_problems"]]
    return {
        "baseline_confirmed_problems": len(confirmed), "baseline_unresolved": len(unresolved),
        "confirmed_excluded_not_original_problem": len(excluded),
        "excluded_issue_ids": [row["issue_id"] for row in excluded],
        "confirmed_denominator": len(eligible), "confirmed_detected": detected,
        "confirmed_undetected": len(eligible) - detected,
        "detection_rate": detected / len(eligible) if eligible else None,
        "confirmed_resolved": resolved, "confirmed_not_resolved": len(eligible) - resolved,
        "resolution_rate": resolved / len(eligible) if eligible else None,
        "confirmed_after_status": _counts(eligible, "after_status", AFTER_STATUSES - {"not_an_original_problem"}),
        "baseline_unresolved_after_status": _counts(unresolved, "after_status", AFTER_STATUSES),
        "confirmed_resolved_by_removal": sum(row["resolved_by_removal"] for row in eligible),
        "patch_reviews": _counts(post["patch_reviews"], "status", PATCH_STATUSES),
        "new_problems": _counts(problems, "status", PROBLEM_STATUSES),
        "new_problem_tags": {status: {tag: sum(p["status"] == status and tag in p["tags"] for p in problems)
                                     for tag in sorted(NEW_TAGS)} for status in sorted(PROBLEM_STATUSES)},
        "new_problem_tags_are_nonadditive": True,
        "controls": _counts(post["controls"], "status", CONTROL_STATUSES),
        "false_findings": _counts(post["false_findings"], "status", {"false_alarm", "unverifiable"}),
        "posthoc_original_findings": _counts(post["posthoc_original_findings"], "status", PROBLEM_STATUSES),
        "changed_locations": len(repair["patches"]),
        "original_characters": len(original["text"]), "revised_characters": len(revised["text"]),
        "character_delta": len(revised["text"]) - len(original["text"]),
        "stage_web_calls": {stage: data["execution"]["web_calls"] for stage, data in
                            (("repair", repair), ("baseline", baseline), ("post", post))},
        "stage_elapsed_seconds": {stage: _execution(data, stage) for stage, data in
                                  (("repair", repair), ("baseline", baseline), ("post", post))},
    }


def _artifact_link(path, output):
    return urlquote(os.path.relpath(path, output).replace(os.sep, "/"), safe="/.:_-~")


def _report_record(root, topic, fixed_hashes):
    identity = topic["report_id"]
    folder = root / identity
    record = {"report_id": identity, "theme": topic["theme"], "status": "incomplete",
              "acquired": False, "generation_failed": False, "missing": [], "metrics": None,
              "baseline_counts": None, "artifacts": {}, "raw_sha256": {}, "validation_errors": []}
    payloads = {}
    try:
        _baseline_freeze(folder)
        _verify_existing_freeze(folder, fixed_hashes)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        record["validation_errors"].append(str(exc))
    baseline_freeze_path = folder / "evaluation/baseline.freeze.json"
    if baseline_freeze_path.exists():
        record["artifacts"]["evaluation/baseline.freeze.json"] = str(baseline_freeze_path)
    for name in (*RAW_FILES, "evaluation/post.raw.json"):
        path = folder / name
        if not path.exists():
            record["missing"].append(name)
            continue
        record["artifacts"][name] = str(path)
        record["raw_sha256"][name] = _sha(path)
        try:
            payloads[name] = _read_json(path)
        except (ValueError, OSError) as exc:
            record["validation_errors"].append(f"{name}: {exc}")
    original = payloads.get(RAW_FILES[0])
    repair = payloads.get(RAW_FILES[1])
    baseline = payloads.get(RAW_FILES[2])
    post = payloads.get("evaluation/post.raw.json")
    valid_original = False
    if original is not None:
        try:
            validate_original(original)
            valid_original = True
            record["acquired"] = True
        except (ValueError, TypeError, KeyError) as exc:
            record["validation_errors"].append(f"original: {exc}")
    if valid_original:
        if baseline is not None:
            try:
                validate_baseline(original, baseline)
                record["baseline_counts"] = _counts(baseline["issues"], "status", PROBLEM_STATUSES)
            except (ValueError, TypeError, KeyError) as exc:
                record["validation_errors"].append(f"baseline: {exc}")
        if repair is not None:
            try:
                validate_repair(original, repair)
            except (ValueError, TypeError, KeyError) as exc:
                record["validation_errors"].append(f"repair: {exc}")
    # A failure marker is an operator record; absence alone never proves failure.
    failure = folder / "input/generation_failure.json"
    if failure.exists():
        try:
            data = _object(_read_json(failure), "generation_failure")
            _string(data.get("reason"), "generation_failure.reason", nonempty=True)
            record["generation_failed"] = True
            record["generation_failure"] = data
            record["artifacts"]["input/generation_failure.json"] = str(failure)
            if record["acquired"]:
                raise ValueError("Generation failure marker and acquired original coexist")
        except (ValueError, TypeError, KeyError) as exc:
            record["validation_errors"].append(f"generation_failure: {exc}")
    if all(value is not None for value in (original, repair, baseline)) and not record["validation_errors"]:
        try:
            expected, _, _, _, revised, applied = _prepared_outputs(root, identity, fixed_hashes)
            for path, data in expected.items():
                relative = path.relative_to(folder).as_posix()
                if not path.exists():
                    record["missing"].append(relative)
                elif path.read_bytes() != data:
                    raise ValueError(f"Prepared artifact or frozen input changed: {relative}")
                else:
                    record["artifacts"][relative] = str(path)
            payloads["revised"] = revised
            payloads["applied"] = applied
            if post is not None:
                validate_post(original, revised, repair, baseline, post)
            if not record["missing"]:
                record["metrics"] = summarize_complete(original, revised, repair, baseline, post)
                record["status"] = "complete"
        except (ValueError, OSError, TypeError, KeyError) as exc:
            record["validation_errors"].append(str(exc))
    if record["validation_errors"]:
        record["status"] = "invalid"
    elif record["generation_failed"]:
        record["status"] = "generation_failed"
    return record, payloads


def _aggregate(records):
    complete = [row["metrics"] for row in records if row["status"] == "complete"]
    aggregate = {
        "planned_reports": len(records), "acquired_reports": sum(row["acquired"] for row in records),
        "generation_failed_reports": sum(row["generation_failed"] for row in records),
        "complete_reports": len(complete),
        "incomplete_reports": sum(row["status"] == "incomplete" for row in records),
        "invalid_reports": sum(row["status"] == "invalid" for row in records),
        "not_complete_reports": sum(row["status"] != "complete" for row in records),
        "complete_reports_only": None,
    }
    if not complete:
        return aggregate
    scalar_keys = ("baseline_confirmed_problems", "baseline_unresolved", "confirmed_excluded_not_original_problem",
                   "confirmed_denominator", "confirmed_detected", "confirmed_undetected", "confirmed_resolved",
                   "confirmed_not_resolved", "confirmed_resolved_by_removal", "changed_locations",
                   "original_characters", "revised_characters", "character_delta")
    totals = {key: sum(row[key] for row in complete) for key in scalar_keys}
    for key in ("confirmed_after_status", "baseline_unresolved_after_status", "patch_reviews", "new_problems",
                "controls", "false_findings", "posthoc_original_findings", "stage_web_calls", "stage_elapsed_seconds"):
        totals[key] = {subkey: sum(row[key][subkey] for row in complete) for subkey in complete[0][key]}
    denominator = totals["confirmed_denominator"]
    totals["detection_rate"] = totals["confirmed_detected"] / denominator if denominator else None
    totals["resolution_rate"] = totals["confirmed_resolved"] / denominator if denominator else None
    aggregate["complete_reports_only"] = totals
    return aggregate


def _html_report(summary, payloads_by_id, output):
    def esc(value):
        return html.escape(str(value))
    def block(value):
        return "<pre>" + esc(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"
    def rows(title, data):
        return "<h3>" + esc(title) + "</h3>" + ("<p>未取得・未完了</p>" if data is None else
                "<p>0件（取得済み）</p>" if not data else "".join("<details><summary>" +
                esc(next((row[key] for key in ("issue_id", "patch_id", "control_id", "finding_id", "item_id") if key in row), title)) +
                "</summary>" + block(row) + "</details>" for row in data))
    parts = ["<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>",
             "<title>Deep Research 3テーマ再現試験</title><style>body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.6;color:#152238}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6fa;padding:1rem}a{color:#075ba8}section{border-top:2px solid #bdcadb;margin-top:2rem}details{margin:.5rem 0}.comparison{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.diff{width:100%;overflow:auto}.diff pre{white-space:pre-wrap}.diff_add,ins{background:#d2f7d7}.diff_sub,del{background:#ffdddd}.status{font-weight:bold}table{border-collapse:collapse}td,th{padding:.35rem;border:1px solid #b9c6d6}@media(max-width:900px){.comparison{grid-template-columns:1fr}}</style>",
             "<h1>Deep Research 3テーマ再現試験</h1><p>AIによる暫定参照評価。人の正解ラベルではありません。予定3本を常に表示し、欠測は未完了として扱います。未確定・新問題・不要変更・誤検出・事後追加は別集計です。タグ件数は重複し得るため合算しません。</p>",
             "<h2>文書別の結果</h2><table><thead><tr><th>テーマ</th><th>確認済み問題</th><th>発見</th><th>解消</th><th>未発見</th><th>未解消</th><th>新問題</th><th>不要変更</th></tr></thead><tbody>"]
    for record in summary["reports"]:
        metrics = record["metrics"]
        values = ([metrics[key] for key in ("confirmed_denominator", "confirmed_detected", "confirmed_resolved",
                                           "confirmed_undetected", "confirmed_not_resolved")] +
                  [metrics["new_problems"]["confirmed_problem"], metrics["patch_reviews"]["unnecessary_change"]]
                  if metrics else ["未完了"] * 7)
        parts.append(f"<tr><td><a href='#{esc(record['report_id'])}'>{esc(record['report_id'])}: {esc(record['theme'])}</a></td>" +
                     "".join(f"<td>{esc(value)}</td>" for value in values) + "</tr>")
    parts.extend(["</tbody></table><p>未発見と未解消は重なり得ます。新問題は意味上の問題数、不要変更は変更箇所数であり、単純に足し合わせません。未確定の原文問題は上表の母数に含めません。</p>",
                  "<details><summary>取得状況と全集計</summary>", block(summary["aggregate"]), "</details>"])
    for record in summary["reports"]:
        identity = record["report_id"]
        payloads = payloads_by_id[identity]
        parts.extend([f"<section id='{esc(identity)}'><h2>{esc(identity)} — {esc(record['theme'])}</h2>",
                      f"<p class='status'>状態: {esc(record['status'])}</p>", block({key: record[key] for key in
                        ("missing", "validation_errors", "baseline_counts", "metrics")}), "<h3>保存資料</h3><ul>"])
        for label, path in record["artifacts"].items():
            parts.append(f"<li><a href='{esc(_artifact_link(path, output))}'>{esc(label)}</a></li>")
        parts.append("</ul>")
        original = payloads.get(RAW_FILES[0])
        revised = payloads.get("revised")
        if isinstance(original, dict) and isinstance(original.get("text"), str):
            if original.get("chat_url"):
                parts.append(f"<p><a href='{esc(original['chat_url'])}'>実際に生成したDeep Researchチャット</a></p>")
            parts.append("<details><summary>原文と修正文の全文を並べて見る</summary><div class='comparison'><div><h3>原文</h3><pre>" + esc(original["text"]) + "</pre></div><div><h3>修正文</h3>")
            parts.append("<pre>" + esc(revised["text"]) + "</pre>" if revised else "<p>未取得・未完了</p>")
            parts.append("</div></div></details>")
            if revised:
                diff = list(difflib.unified_diff(original["text"].splitlines(), revised["text"].splitlines(), fromfile="original", tofile="revised", lineterm=""))
                parts.append("<h3>全文差分</h3><div class='diff'><pre>" + esc("\n".join(diff) if diff else "変更なし") + "</pre></div>")
            parts.append("<h3>原文出典</h3>" + block(original.get("sources")))
            if revised:
                parts.append("<h3>修正文出典</h3>" + block(revised["sources"]))
        baseline = payloads.get(RAW_FILES[2])
        repair = payloads.get(RAW_FILES[1])
        post = payloads.get("evaluation/post.raw.json")
        for title, data, key in (("Baseline全inventory", baseline, "inventory"), ("Baseline全issues", baseline, "issues"),
                                 ("Baseline全controls", baseline, "controls"), ("修正者の全findings", repair, "findings"),
                                 ("全patches（理由を含む監査記録）", repair, "patches"), ("事後全issues", post, "issues"),
                                 ("事後全patch_reviews / 新問題", post, "patch_reviews"), ("事後全controls", post, "controls"),
                                 ("誤検出・検証不能", post, "false_findings"), ("事後追加の原文問題（主母数外）", post, "posthoc_original_findings")):
            value = data.get(key) if isinstance(data, dict) else None
            # Invalid records remain visible without allowing malformed shapes
            # to break the whole three-report rendering.
            parts.append(rows(title, value) if isinstance(value, list) and all(isinstance(row, dict) for row in value)
                         else "<h3>" + esc(title) + "</h3>" + block(value))
        parts.append("</section>")
    parts.append("</html>")
    return "\n".join(parts).encode("utf-8")


def build_report(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    protocol, hashes = validate_preregistration(root)
    records, payloads = [], {}
    for topic in protocol["topics"]:
        record, data = _report_record(root, topic, hashes)
        records.append(record)
        payloads[topic["report_id"]] = data
    summary = {
        "schema_version": 1, "experiment_id": protocol["id"],
        "human_adjudicated": False, "fixed_instruction_sha256": hashes,
        "aggregate": _aggregate(records), "reports": records,
        "interpretation": ["Incomplete or invalid reports have null metrics, never zero-filled outcomes.",
                           "Aggregate outcome counts cover complete reports only; planned denominator remains three.",
                           "Only baseline confirmed problems enter the main denominator; not_an_original_problem is explicitly excluded.",
                           "Unresolved baseline issues and posthoc original findings never enter the main detection denominator.",
                           "New problems, unnecessary changes, false findings and posthoc findings are separate, nonadditive measures.",
                           "Zero eligible confirmed problems yields null detection and resolution rates.",
                           "Input hashes demonstrate frozen content, not evaluator independence or semantic correctness."] + protocol.get("limits", []),
    }
    outputs = {output / "summary.json": _json_bytes(summary), output / "index.html": _html_report(summary, payloads, output)}
    _write_immutable(outputs)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-post", help="validate frozen inputs and make the independent post packet")
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--report", choices=("R1", "R2", "R3"), required=True)
    report = subparsers.add_parser("report", help="validate all planned reports and write immutable HTML/JSON summaries")
    report.add_argument("--root", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare_post(args.root, args.report) if args.command == "prepare-post" else build_report(args.root, args.output)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(2, f"Validation failed: {exc}\n")
    print(json.dumps(result if args.command == "prepare-post" else result["aggregate"], ensure_ascii=False, indent=2))
    return 0 if args.command == "prepare-post" or not result["aggregate"]["invalid_reports"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
