"""Prepare frozen evaluator packets and report the preregistered paired trial.

No model calls, evidence retrieval, or semantic judgments are performed here.
Raw records are never rewritten; derived files are immutable and reproducible.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import html
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import replication as rep


FIXED_FILES = (
    "protocol.json", "instructions/repair_v1.txt", "instructions/repair_v2.txt",
    "instructions/reference.txt", "instructions/post.txt",
)
GUARD_FIXED_FILES = ("instructions/guard.txt", "guard_contract.json")
VARIANTS = ("current", "improved")
OUTCOMES = ("current_only_resolved", "improved_only_resolved", "both_resolved", "neither_resolved")
PREPARED_FILES = ("freeze.json", "applied.json", "revised_report.json", "post_packet.json")
VALIDATION_EXCEPTIONS = (ValueError, OSError, TypeError, KeyError)


def _relative(root, name):
    """Resolve a portable relative path without permitting root escapes."""
    rep._string(name, "relative path", nonempty=True)
    path = PurePosixPath(name)
    if ("\\" in name or ":" in name or path.is_absolute() or
            any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError(f"Unsafe relative path: {name}")
    result = (root / Path(*path.parts)).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Relative path escapes trial root: {name}")
    return result


def validate_preregistration(root):
    root = Path(root).resolve()
    prereg = rep._object(rep._read_json(root / "preregistration.json"), "preregistration")
    if prereg.get("generation_started") is not False:
        raise ValueError("preregistration.generation_started must be false in the frozen record")
    hashes = rep._object(prereg.get("sha256"), "preregistration.sha256")
    missing = set(FIXED_FILES) - set(hashes)
    if missing:
        raise ValueError(f"Preregistration must freeze required instructions: {sorted(missing)}")
    for name, expected in hashes.items():
        if name == "preregistration.json":
            raise ValueError("Preregistration cannot contain its own SHA256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Invalid preregistration SHA256: {name}")
        if rep._sha(_relative(root, name)) != expected:
            raise ValueError(f"Preregistration SHA256 mismatch: {name}")
    protocol = rep._object(rep._read_json(root / "protocol.json"), "protocol")
    rep._string(protocol.get("id"), "protocol.id", nonempty=True)
    version = protocol.get("schema_version", 1)
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("protocol.schema_version must be 1 or 2")
    documents = rep._rows_by_id(protocol.get("documents"), "id", "protocol.documents")
    if version == 2:
        planned = rep._strings(protocol.get("planned_document_ids"), "protocol.planned_document_ids")
        if (not planned or len(planned) != len(set(planned)) or
                any(not re.fullmatch(r"[RN][1-9][0-9]*", identity) for identity in planned)):
            raise ValueError("protocol.planned_document_ids must contain unique nonempty stable R/N IDs")
        rep._all_ids(documents, planned, "protocol.documents versus planned_document_ids")
        missing = set(GUARD_FIXED_FILES) - set(hashes)
        if missing:
            raise ValueError(f"Preregistration must freeze guard inputs: {sorted(missing)}")
    elif set(documents) != {"R1", "R2", "R3", "N1", "N2"}:
        raise ValueError("The paired trial must retain all planned R1, R2, R3, N1 and N2 documents")
    mapping = rep._object(protocol.get("arm_mapping"), "protocol.arm_mapping")
    if set(mapping) - set(documents):
        raise ValueError("protocol.arm_mapping contains an unknown document")
    paths = set()
    for identity, doc in documents.items():
        cohort = "regression" if identity.startswith("R") else "new"
        if doc.get("cohort") != cohort:
            raise ValueError(f"Wrong cohort for {identity}")
        folder = _relative(root, doc.get("path"))
        if folder in paths or any(folder in p.parents or p in folder.parents for p in paths):
            raise ValueError("Document paths must be distinct and cannot nest")
        paths.add(folder)
        arms = rep._strings(doc.get("arms"), f"{identity}.arms")
        expected_arms = set(VARIANTS) if cohort == "regression" else {"A", "B"}
        if len(arms) != 2 or set(arms) != expected_arms:
            raise ValueError(f"Wrong arms for {identity}: expected {sorted(expected_arms)}")
        assigned = mapping.get(identity, {arm: arm for arm in arms} if cohort == "regression" else None)
        rep._object(assigned, f"protocol.arm_mapping.{identity}")
        if set(assigned) != expected_arms or set(assigned.values()) != set(VARIANTS):
            raise ValueError(f"Arm mapping must assign current and improved once each for {identity}")
        if cohort == "regression" and assigned != {arm: arm for arm in VARIANTS}:
            raise ValueError("Regression arms must retain their named variants")
    if version == 2 or "evaluation_arm_mapping" in protocol:
        evaluation = rep._object(protocol.get("evaluation_arm_mapping"), "protocol.evaluation_arm_mapping")
        regressions = {identity for identity, doc in documents.items() if doc["cohort"] == "regression"}
        rep._all_ids(evaluation, regressions, "protocol.evaluation_arm_mapping")
        for identity, assigned in evaluation.items():
            rep._object(assigned, f"protocol.evaluation_arm_mapping.{identity}")
            if set(assigned) != {"A", "B"} or set(assigned.values()) != set(VARIANTS):
                raise ValueError(f"Evaluation mapping must assign A/B to current/improved once each for {identity}")
        if version == 2:
            assignments = [evaluation[identity]["A"] for identity in regressions]
            new_assignments = [mapping[identity]["A"] for identity, doc in documents.items() if doc["cohort"] == "new"]
            for cohort, values in (("regression", assignments), ("new", new_assignments)):
                if abs(values.count("current") - values.count("improved")) > 1:
                    raise ValueError(f"Evaluation A/B mapping must be balanced within the {cohort} cohort")
    return protocol, {**hashes, "preregistration.json": rep._sha(root / "preregistration.json")}


def _arm_mapping(protocol, doc):
    return protocol["arm_mapping"].get(doc["id"], {arm: arm for arm in doc["arms"]})


def _required_baseline_freeze(folder):
    result = rep._baseline_freeze(folder)
    if result is None:
        raise ValueError("Independent baseline.freeze.json is required before prepare")
    return result


def _prepared_outputs(root, doc, arm, hashes):
    folder = _relative(root, doc["path"])
    arm_folder = folder / "arms" / arm
    independent = _required_baseline_freeze(folder)
    original_path = folder / "input/original_report.json"
    baseline_path = folder / "evaluation/baseline.raw.json"
    repair_path = arm_folder / "repair.raw.json"
    original, baseline, repair = [rep._read_json(path) for path in (original_path, baseline_path, repair_path)]
    rep.validate_original(original)
    rep.validate_baseline(original, baseline)
    applied = rep.validate_repair(original, repair)
    revised = {**copy.deepcopy(original), "text": applied["text"], "sources": applied["sources"]}
    raw_hashes = {"input/original_report.json": rep._sha(original_path),
                  "evaluation/baseline.raw.json": rep._sha(baseline_path),
                  f"arms/{arm}/repair.raw.json": rep._sha(repair_path)}
    freeze = {
        "schema_version": 1, "document_id": doc["id"], "arm": arm,
        "fixed_instruction_sha256": hashes, "input_sha256": raw_hashes,
        "independent_baseline_freeze": independent,
        "baseline_fixed_before_post_packet": True,
        "baseline_execution_finished_utc": baseline["execution"]["finished_utc"],
        "temporal_attestation": "Hashes fix content before packet creation; they do not independently prove generation chronology or evaluator independence.",
    }
    literal_keys = ("patch_id", "before", "after", "start_char", "end_char", "after_start_char", "after_end_char")
    packet = {
        "document_id": doc["id"], "original": original, "revised": revised,
        "literal_changes": [{key: patch[key] for key in literal_keys} for patch in applied["patches"]],
        "writer_findings": copy.deepcopy(repair["findings"]),
        "baseline_sha256": raw_hashes["evaluation/baseline.raw.json"],
        # Logical keys avoid revealing variant names through paths or instruction files.
        "input_sha256": {"original": rep._sha(original_path), "baseline": rep._sha(baseline_path),
                         "repair": rep._sha(repair_path)},
    }
    outputs = {arm_folder / name: rep._json_bytes(value) for name, value in (
        ("freeze.json", freeze), ("applied.json", applied),
        ("revised_report.json", revised), ("post_packet.json", packet),
    )}
    return outputs, original, repair, baseline, revised


def prepare(root, doc_id, arm):
    root = Path(root).resolve()
    protocol, hashes = validate_preregistration(root)
    documents = {doc["id"]: doc for doc in protocol["documents"]}
    if doc_id not in documents:
        raise ValueError(f"Unknown document ID: {doc_id}")
    doc = documents[doc_id]
    if arm not in doc["arms"]:
        raise ValueError(f"Unknown arm for {doc_id}: {arm}")
    outputs, *_ = _prepared_outputs(root, doc, arm, hashes)
    rep._write_immutable(outputs)
    return {"document_id": doc_id, "arm": arm,
            "post_packet": str(_relative(root, doc["path"]) / "arms" / arm / "post_packet.json"),
            "files": [str(path) for path in outputs]}


def _document(protocol, doc_id):
    for document in protocol["documents"]:
        if document["id"] == doc_id:
            return document
    raise ValueError(f"Unknown document ID: {doc_id}")


def _evaluation_mapping(protocol, doc):
    if doc["cohort"] == "new":
        return {"A": "A", "B": "B"}
    mapping = protocol.get("evaluation_arm_mapping", {}).get(doc["id"])
    if mapping is None:
        raise ValueError(f"Frozen evaluation_arm_mapping required for joint evaluation of {doc['id']}")
    return mapping


def _joint_outputs(root, protocol, doc, hashes):
    """Reconstruct a blinded joint packet only from intact, already prepared arms."""
    folder = _relative(root, doc["path"])
    mapping = _evaluation_mapping(protocol, doc)
    packages, inputs, packet_hashes = {}, {}, {}
    for code in ("A", "B"):
        arm = mapping[code]
        outputs, original, repair, baseline, revised = _prepared_outputs(root, doc, arm, hashes)
        for path, expected in outputs.items():
            if not path.exists() or path.read_bytes() != expected:
                raise ValueError(f"Prepared artifact missing or changed for coded arm {code}: {path.name}")
        packet_path = folder / "arms" / arm / "post_packet.json"
        prepared = rep._read_json(packet_path)
        packet_hashes[code] = rep._sha(packet_path)
        packages[code] = {key: copy.deepcopy(prepared[key]) for key in
                          ("revised", "literal_changes", "writer_findings", "input_sha256")}
        packages[code]["input_sha256"]["post_packet"] = packet_hashes[code]
        inputs[code] = (original, revised, repair, baseline)
    if inputs["A"][0] != inputs["B"][0] or inputs["A"][3] != inputs["B"][3]:
        raise ValueError("Both coded arms must share the same original and frozen baseline")
    independent = _required_baseline_freeze(folder)
    packet = {
        "schema_version": 1, "document_id": doc["id"],
        "original": inputs["A"][0], "baseline": inputs["A"][3], "arms": packages,
        "baseline_sha256": independent["sha256"],
        "input_sha256": {"original": rep._sha(folder / "input/original_report.json"),
                         "baseline": independent["sha256"], "baseline_freeze": independent["record_sha256"]},
    }
    packet_bytes = rep._json_bytes(packet)
    provenance = {
        "schema_version": 1, "document_id": doc["id"],
        "evaluation_arm_mapping": mapping, "fixed_instruction_sha256": hashes,
        "input_sha256": packet["input_sha256"], "prepared_packet_sha256": packet_hashes,
        "paired_post_packet_sha256": hashlib.sha256(packet_bytes).hexdigest(),
    }
    outputs = {folder / "evaluation/paired_post_packet.json": packet_bytes,
               folder / "evaluation/paired_post.freeze.json": rep._json_bytes(provenance)}
    return outputs, inputs, mapping


def prepare_post(root, doc_id):
    """Freeze both masked arms together with their shared independent baseline."""
    root = Path(root).resolve()
    protocol, hashes = validate_preregistration(root)
    doc = _document(protocol, doc_id)
    outputs, _, _ = _joint_outputs(root, protocol, doc, hashes)
    rep._write_immutable(outputs)
    return {"document_id": doc_id,
            "paired_post_packet": str(_relative(root, doc["path"]) / "evaluation/paired_post_packet.json"),
            "files": [str(path) for path in outputs]}


def _post_ingest_outputs(root, protocol, doc, hashes, response=None):
    """Build the complete immutable extraction batch without writing any file."""
    folder = _relative(root, doc["path"])
    prepared_outputs, inputs, mapping = _joint_outputs(root, protocol, doc, hashes)
    for path, expected in prepared_outputs.items():
        if not path.exists() or path.read_bytes() != expected:
            raise ValueError(f"Joint post packet provenance missing or changed: {path.name}")
    raw_path = folder / "evaluation/paired_post.raw.json"
    source = Path(response).resolve() if response is not None else raw_path
    raw_bytes = source.read_bytes()
    post = rep._object(rep._read_json(source), "paired_post")
    if set(post) != {"execution", "human_adjudicated", "arms", "cross_arm_notes"}:
        raise ValueError("paired_post must contain exactly execution, human_adjudicated, arms and cross_arm_notes")
    rep._execution(post, "paired_post")
    if rep._boolean(post.get("human_adjudicated"), "paired_post.human_adjudicated"):
        raise ValueError("The fixed trial records human_adjudicated=false")
    arms = rep._object(post.get("arms"), "paired_post.arms")
    rep._all_ids(arms, {"A", "B"}, "paired_post.arms")
    rep._strings(post.get("cross_arm_notes"), "paired_post.cross_arm_notes")
    outputs = {raw_path: raw_bytes}
    arm_hashes = {}
    for code in ("A", "B"):
        rep.validate_post(*inputs[code], arms[code])
        payload = rep._json_bytes(arms[code])
        outputs[folder / "arms" / mapping[code] / "post.raw.json"] = payload
        arm_hashes[code] = hashlib.sha256(payload).hexdigest()
    audit = {
        "schema_version": 1, "document_id": doc["id"], "human_adjudicated": False,
        "paired_post_raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "paired_post_packet_sha256": rep._sha(folder / "evaluation/paired_post_packet.json"),
        "paired_post_freeze_sha256": rep._sha(folder / "evaluation/paired_post.freeze.json"),
        "evaluation_arm_mapping": mapping, "arm_post_sha256": arm_hashes,
        "extraction": "Exact arm objects; no labels added and no judgments rewritten.",
    }
    outputs[folder / "evaluation/paired_post.ingest.json"] = rep._json_bytes(audit)
    return outputs


def ingest_post(root, doc_id, response=None):
    """Validate both judgments before immutably saving their exact arm objects."""
    root = Path(root).resolve()
    protocol, hashes = validate_preregistration(root)
    outputs = _post_ingest_outputs(root, protocol, _document(protocol, doc_id), hashes, response)
    rep._write_immutable(outputs)
    return {"document_id": doc_id, "files": [str(path) for path in outputs]}


def _read_available(path, label, record, payloads):
    if not path.exists():
        record["missing"].append(label)
        return None
    record["artifacts"][label] = str(path)
    try:
        record["raw_sha256"][label] = rep._sha(path)
        value = rep._read_json(path)
        payloads[label] = value
        return value
    except VALIDATION_EXCEPTIONS as exc:
        record["validation_errors"].append(f"{label}: {exc}")
        return None


def _secondary_metrics(original, revised, repair, baseline, post):
    metrics = rep.summarize_complete(original, revised, repair, baseline, post)
    # Individual exclusions may disagree. Only paired_primary supplies a main
    # denominator or rate; never expose competing one-sided primary estimates.
    for key in list(metrics):
        if key.startswith("confirmed_") or key in {"detection_rate", "resolution_rate", "excluded_issue_ids"}:
            del metrics[key]
    return metrics


def _pair(baseline, current_post, improved_post):
    posts = {variant: {row["issue_id"]: row for row in post["issues"]}
             for variant, post in (("current", current_post), ("improved", improved_post))}
    confirmed = [row["issue_id"] for row in baseline["issues"] if row["status"] == "confirmed_problem"]
    primary = {"baseline_confirmed_issue_ids": confirmed, "eligible_issue_ids": [],
               "excluded_issue_ids": [], "exclusion_disagreements": [], "agreed_exclusion_issue_ids": [],
               **{outcome: [] for outcome in OUTCOMES}, "issues": []}
    unresolved = []
    for issue in baseline["issues"]:
        identity = issue["issue_id"]
        rows = {variant: posts[variant][identity] for variant in VARIANTS}
        detail = {"issue_id": identity, "baseline_status": issue["status"],
                  **{variant: copy.deepcopy(rows[variant]) for variant in VARIANTS}}
        if issue["status"] != "confirmed_problem":
            unresolved.append(detail)
            continue
        exclusions = {variant: rows[variant]["after_status"] == "not_an_original_problem" for variant in VARIANTS}
        detail["included"] = not any(exclusions.values())
        if any(exclusions.values()):
            primary["excluded_issue_ids"].append(identity)
            if exclusions["current"] != exclusions["improved"]:
                primary["exclusion_disagreements"].append({"issue_id": identity,
                    "current": rows["current"]["after_status"], "improved": rows["improved"]["after_status"]})
            else:
                primary["agreed_exclusion_issue_ids"].append(identity)
        else:
            primary["eligible_issue_ids"].append(identity)
            current = rows["current"]["after_status"] == "resolved"
            improved = rows["improved"]["after_status"] == "resolved"
            outcome = ("both_resolved" if current and improved else "current_only_resolved" if current
                       else "improved_only_resolved" if improved else "neither_resolved")
            detail["outcome"] = outcome
            primary[outcome].append(identity)
        primary["issues"].append(detail)
    primary["denominator"] = len(primary["eligible_issue_ids"])
    for variant in VARIANTS:
        eligible = [posts[variant][identity] for identity in primary["eligible_issue_ids"]]
        detected = sum(row["detected"] for row in eligible)
        resolved = sum(row["after_status"] == "resolved" for row in eligible)
        primary[variant] = {"detected": detected, "resolved": resolved,
            "detection_rate": detected / len(eligible) if eligible else None,
            "resolution_rate": resolved / len(eligible) if eligible else None,
            "resolved_by_removal": sum(row["resolved_by_removal"] for row in eligible),
            "after_status": rep._counts(eligible, "after_status", rep.AFTER_STATUSES - {"not_an_original_problem"})}
    return {"primary": primary, "baseline_unresolved": unresolved}


def _document_record(root, protocol, doc, hashes):
    folder = _relative(root, doc["path"])
    record = {"id": doc["id"], "cohort": doc["cohort"], "path": doc["path"],
              "status": "incomplete", "missing": [], "validation_errors": [],
              "baseline_counts": None, "arms": {}, "paired": None,
              "artifacts": {}, "raw_sha256": {}}
    payloads = {"arms": {}}
    original = _read_available(folder / "input/original_report.json", "input/original_report.json", record, payloads)
    baseline = _read_available(folder / "evaluation/baseline.raw.json", "evaluation/baseline.raw.json", record, payloads)
    freeze_path = folder / "evaluation/baseline.freeze.json"
    if freeze_path.exists():
        record["artifacts"]["evaluation/baseline.freeze.json"] = str(freeze_path)
        try:
            _required_baseline_freeze(folder)
        except VALIDATION_EXCEPTIONS as exc:
            record["validation_errors"].append(str(exc))
    else:
        record["missing"].append("evaluation/baseline.freeze.json")
    joint_names = ("paired_post_packet.json", "paired_post.freeze.json", "paired_post.raw.json", "paired_post.ingest.json")
    if any((folder / "evaluation" / name).exists() for name in joint_names):
        for name in joint_names:
            path = folder / "evaluation" / name
            label = "evaluation/" + name
            if path.exists():
                record["artifacts"][label] = str(path)
                record["raw_sha256"][label] = rep._sha(path)
            else:
                record["missing"].append(label)
        if (folder / "evaluation/paired_post.ingest.json").exists():
            try:
                outputs = _post_ingest_outputs(root, protocol, doc, hashes)
                for path, expected in outputs.items():
                    if not path.exists() or path.read_bytes() != expected:
                        raise ValueError(f"Joint post extraction missing or changed: {path.name}")
            except VALIDATION_EXCEPTIONS as exc:
                record["validation_errors"].append(str(exc))
    try:
        if original is not None:
            rep.validate_original(original)
            if baseline is not None:
                rep.validate_baseline(original, baseline)
                record["baseline_counts"] = rep._counts(baseline["issues"], "status", rep.PROBLEM_STATUSES)
    except VALIDATION_EXCEPTIONS as exc:
        record["validation_errors"].append(str(exc))
    mapping = _arm_mapping(protocol, doc)
    complete_posts = {}
    for arm in doc["arms"]:
        arm_folder = folder / "arms" / arm
        arm_record = {"arm": arm, "variant": mapping[arm], "status": "incomplete", "metrics": None,
                      "missing": [], "validation_errors": [], "artifacts": {}, "raw_sha256": {}}
        data = {}
        repair = _read_available(arm_folder / "repair.raw.json", "repair.raw.json", arm_record, data)
        post = _read_available(arm_folder / "post.raw.json", "post.raw.json", arm_record, data)
        for name in PREPARED_FILES:
            path = arm_folder / name
            if path.exists():
                arm_record["artifacts"][name] = str(path)
            else:
                arm_record["missing"].append(name)
        if record["validation_errors"]:
            arm_record["validation_errors"].append("Shared original/baseline validation failed")
        elif original is not None and baseline is not None and repair is not None:
            try:
                # Validate available repair even when its mandatory baseline
                # freeze or derived outputs have yet to be recorded.
                applied = rep.validate_repair(original, repair)
                revised = {**copy.deepcopy(original), "text": applied["text"], "sources": applied["sources"]}
                data["revised"] = revised
                if freeze_path.exists():
                    outputs, *_ = _prepared_outputs(root, doc, arm, hashes)
                    for path, expected in outputs.items():
                        if path.exists() and path.read_bytes() != expected:
                            raise ValueError(f"Frozen input or derived artifact missing or changed: {path.name}")
                if post is not None:
                    rep.validate_post(original, revised, repair, baseline, post)
                if not arm_record["missing"] and not record["missing"] and not arm_record["validation_errors"]:
                    arm_record["status"] = "complete"
                    arm_record["metrics"] = _secondary_metrics(original, revised, repair, baseline, post)
                    complete_posts[mapping[arm]] = post
            except VALIDATION_EXCEPTIONS as exc:
                arm_record["validation_errors"].append(str(exc))
        if arm_record["validation_errors"]:
            arm_record["status"] = "invalid"
            arm_record["metrics"] = None
        record["arms"][arm] = arm_record
        payloads["arms"][arm] = data
    if record["validation_errors"] or any(arm["status"] == "invalid" for arm in record["arms"].values()):
        record["status"] = "invalid"
    elif set(complete_posts) == set(VARIANTS):
        record["status"] = "complete"
        record["paired"] = _pair(baseline, complete_posts["current"], complete_posts["improved"])
    return record, payloads


def _aggregate(records):
    complete = [row for row in records if row["status"] == "complete"]
    aggregate = {"planned_documents": len(records), "planned_arms": sum(len(row["arms"]) for row in records),
                 "complete_pairs": len(complete), "invalid_documents": sum(row["status"] == "invalid" for row in records),
                 "incomplete_documents": sum(row["status"] == "incomplete" for row in records), "paired_primary": None}
    if not complete:
        return aggregate
    primary = {"denominator": sum(row["paired"]["primary"]["denominator"] for row in complete)}
    for field in (*OUTCOMES, "baseline_confirmed_issue_ids", "eligible_issue_ids", "excluded_issue_ids", "agreed_exclusion_issue_ids"):
        primary[field] = [{"document_id": row["id"], "issue_id": identity} for row in complete
                          for identity in row["paired"]["primary"][field]]
    primary["exclusion_disagreements"] = [{"document_id": row["id"], **issue} for row in complete
                                         for issue in row["paired"]["primary"]["exclusion_disagreements"]]
    for variant in VARIANTS:
        counts = {key: sum(row["paired"]["primary"][variant][key] for row in complete)
                  for key in ("detected", "resolved", "resolved_by_removal")}
        counts["detection_rate"] = counts["detected"] / primary["denominator"] if primary["denominator"] else None
        counts["resolution_rate"] = counts["resolved"] / primary["denominator"] if primary["denominator"] else None
        primary[variant] = counts
    aggregate["paired_primary"] = primary
    return aggregate


def _html_report(summary, payloads, output):
    esc = lambda value: html.escape(str(value), quote=True)
    def block(value):
        return "<p>未取得・未完了</p>" if value is None else "<pre>" + esc(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"
    def detail(title, value):
        return "<details><summary>" + esc(title) + "</summary>" + block(value) + "</details>"
    def shown(value):
        return "—" if value is None else esc(value)
    parts = ["<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>現行手順・改善版 ペア比較</title><style>body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.65;color:#152333}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #bbc7d4;padding:.45rem;text-align:left}th{background:#edf2f8}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:1rem}details{margin:.6rem 0}summary{cursor:pointer;font-weight:600}.comparison{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.error{color:#a21c24}section{border-top:2px solid #afbccb;margin-top:2rem}@media(max-width:800px){.comparison{grid-template-columns:1fr}body{margin:1rem}}</style>",
        "<h1>現行手順・改善版 ペア比較</h1>",
        "<p>主解析は凍結baselineで確認済みの同一問題IDに限定。片腕でも原文問題ではないと評価されたIDは両腕から除外し、判定不一致を表示します。未完了は欠測（—）です。部分解消は解消数に含みません。</p>",
        "<p>全評価は人間未裁定のAI暫定評価です。回帰セットと新規セットは別集計し、一般化や独立した再現性の証明とは扱いません。</p>",
        "<p><a href='summary.json'>summary.json</a></p>"]
    for cohort, title in (("regression", "回帰セット"), ("new", "新規セット")):
        agg = summary["cohorts"][cohort]
        parts.append(f"<h2>{title}</h2><p>完了ペア {agg['complete_pairs']} / 計画 {agg['planned_documents']}文書</p>")
        parts.append("<table><thead><tr><th>文書</th><th>状態</th><th>共通母数</th><th>現行解消</th><th>改善解消</th><th>現行のみ</th><th>改善のみ</th><th>両方解消</th><th>両方未解消</th><th>除外不一致</th></tr></thead><tbody>")
        for doc in summary["documents"]:
            if doc["cohort"] != cohort:
                continue
            primary = doc["paired"]["primary"] if doc["paired"] else None
            values = [primary["denominator"], primary["current"]["resolved"], primary["improved"]["resolved"],
                      *[len(primary[key]) for key in OUTCOMES], len(primary["exclusion_disagreements"])] if primary else [None] * 8
            state = {"complete": "完了", "incomplete": "未完了", "invalid": "検証エラー"}[doc["status"]]
            parts.append(f"<tr><td><a href='#{esc(doc['id'])}'>{esc(doc['id'])}</a></td><td>{state}</td>" + "".join("<td>" + shown(value) + "</td>" for value in values) + "</tr>")
        parts.append("</tbody></table>" + detail("セット集計（完了ペアのみ）", agg))
    parts.append(detail("解釈・制約", summary["interpretation"]))
    for doc in summary["documents"]:
        data = payloads[doc["id"]]
        original = data.get("input/original_report.json")
        baseline = data.get("evaluation/baseline.raw.json")
        parts.append(f"<section id='{esc(doc['id'])}'><h2>{esc(doc['id'])} — {esc(doc['cohort'])}</h2>")
        parts.append(detail("ペア比較・全主問題ID・未確定baseline", doc["paired"]))
        parts.append(detail("共通入力の欠測・検証エラー", {"missing": doc["missing"], "errors": doc["validation_errors"]}))
        parts.append("<p>共通記録: " + " · ".join(f"<a href='{esc(rep._artifact_link(Path(path), output))}'>{esc(name)}</a>" for name, path in doc["artifacts"].items()) + "</p>")
        for field in ("inventory", "issues", "controls"):
            parts.append(detail("Baseline 全" + field, baseline.get(field) if isinstance(baseline, dict) else None))
        for arm, arm_record in doc["arms"].items():
            arm_data = data["arms"][arm]
            revised = arm_data.get("revised")
            repair, post = arm_data.get("repair.raw.json"), arm_data.get("post.raw.json")
            parts.append(f"<h3>{esc(arm)} — {esc(arm_record['variant'])}</h3>")
            parts.append(detail("状態・欠測・検証エラー・副次集計", {key: arm_record[key] for key in ("status", "missing", "validation_errors", "metrics")}))
            parts.append("<p>記録: " + " · ".join(f"<a href='{esc(rep._artifact_link(Path(path), output))}'>{esc(name)}</a>" for name, path in arm_record["artifacts"].items()) + "</p>")
            if isinstance(original, dict) and isinstance(original.get("text"), str):
                parts.append("<details><summary>原文・修正文 全文</summary><div class='comparison'><div><h4>原文</h4><pre>" + esc(original["text"]) + "</pre></div><div><h4>修正文</h4>" + ("<pre>" + esc(revised["text"]) + "</pre>" if revised else "<p>未取得・未完了</p>") + "</div></div></details>")
                if revised:
                    diff = "\n".join(difflib.unified_diff(original["text"].splitlines(), revised["text"].splitlines(), fromfile="original", tofile="revised", lineterm=""))
                    parts.append("<details><summary>全文差分</summary><pre>" + esc(diff or "変更なし") + "</pre></details>")
                parts.append(detail("原文出典", original.get("sources")))
                parts.append(detail("修正文出典", revised.get("sources") if revised else None))
            for title, source, field in (
                ("修正者 全findings", repair, "findings"), ("全patches（理由を含む監査記録）", repair, "patches"),
                ("修正者 checked_controls", repair, "checked_controls"), ("修正者 未確定事項", repair, "unresolved"),
                ("事後 全issues", post, "issues"), ("全patch_reviews・新問題", post, "patch_reviews"),
                ("全false_findings・誤指摘", post, "false_findings"), ("全controls", post, "controls"),
                ("事後追加の原文問題（主母数外）", post, "posthoc_original_findings"),
            ):
                parts.append(detail(title, source.get(field) if isinstance(source, dict) else None))
        parts.append("</section>")
    parts.append("</html>")
    return "\n".join(parts).encode("utf-8")


def build_report(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    protocol, hashes = validate_preregistration(root)
    documents, payloads = [], {}
    for doc in protocol["documents"]:
        record, data = _document_record(root, protocol, doc, hashes)
        documents.append(record)
        payloads[doc["id"]] = data
    summary = {
        "schema_version": 1, "experiment_id": protocol["id"], "human_adjudicated": False,
        "fixed_instruction_sha256": hashes, "documents": documents,
        "cohorts": {cohort: _aggregate([doc for doc in documents if doc["cohort"] == cohort]) for cohort in ("regression", "new")},
        "interpretation": [
            "All planned documents and arms are retained. Missing or invalid outcomes are null, not zero.",
            "Primary paired estimates use only common frozen baseline confirmed_problem IDs and complete pairs.",
            "An ID marked not_an_original_problem by either arm is excluded from both arms; disagreements remain explicit.",
            "Baseline unresolved issues, posthoc original findings, new problems, unnecessary changes, false findings and controls remain separate.",
            "Only resolved counts as resolved. Partially resolved, unresolved and unverifiable do not.",
            "Zero eligible problems yields null detection and resolution rates.",
            "Regression and new cohorts are reported separately; regression current responses may be reused as declared in the protocol.",
            "Hashes demonstrate content integrity, not evaluator independence, external preregistration, or semantic correctness.",
        ] + protocol.get("limits", []),
    }
    rep._write_immutable({output / "summary.json": rep._json_bytes(summary),
                          output / "index.html": _html_report(summary, payloads, output)})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="validate and freeze one arm before independent post evaluation")
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--doc", required=True)
    prepare_parser.add_argument("--arm", required=True)
    joint_parser = commands.add_parser("prepare-post", help="freeze a joint coded A/B evaluator packet")
    joint_parser.add_argument("--root", type=Path, required=True)
    joint_parser.add_argument("--doc", required=True)
    ingest_parser = commands.add_parser("ingest-post", help="validate and immutably extract both joint post judgments")
    ingest_parser.add_argument("--root", type=Path, required=True)
    ingest_parser.add_argument("--doc", required=True)
    ingest_parser.add_argument("--response", type=Path, help="raw joint response; defaults to evaluation/paired_post.raw.json")
    report_parser = commands.add_parser("report", help="write immutable paired HTML and JSON reports")
    report_parser.add_argument("--root", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.root, args.doc, args.arm)
        elif args.command == "prepare-post":
            result = prepare_post(args.root, args.doc)
        elif args.command == "ingest-post":
            result = ingest_post(args.root, args.doc, args.response)
        else:
            result = build_report(args.root, args.output)
    except VALIDATION_EXCEPTIONS as exc:
        parser.exit(2, f"Validation failed: {exc}\n")
    print(json.dumps(result["cohorts"] if args.command == "report" else result, ensure_ascii=False, indent=2))
    return int(args.command == "report" and any(doc["status"] == "invalid" for doc in result["documents"]))


if __name__ == "__main__":
    raise SystemExit(main())
