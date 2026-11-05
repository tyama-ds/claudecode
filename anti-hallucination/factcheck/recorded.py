"""Import explicitly recorded, blinded agent batches without invoking any API.

Packet hashes bind the public tasks and prompts. They are integrity checks, not
proof that an external solver followed the blinding protocol. Each batch is one
condition, not a separately identified model or an independent API observation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import re

from .dataset import item_context, validate_dataset
from .experiment import (
    CI_WARNING, EVIDENCE_WARNING, _check_metrics, _fill_metrics, _make_pool,
    _now, _pipeline_metrics, _write_csv, _write_json,
)
from .providers import (
    PROMPT_VERSION, _COMMON_INSTRUCTIONS, _STAGE_INSTRUCTIONS, _json_loads,
    _public_payload, _schema, _validate_answer,
)
from .scoring import score_answer


PACKET_VERSION = "recorded-packet-v1"
ANSWER_VERSION = "recorded-answers-v1"
MODEL_IDENTITY = "runtime inherited, exact API model unverified"
CONDITIONS = ("baseline", "evidence_first")
DEFAULT_SEED = 20261101
DEFAULT_PROMPTS = {
    "baseline": "Answer each task directly using the supplied sources and the required schema. Give a concise reason and exact evidence quotes.",
    "evidence_first": "For each task, first locate exact relevant source excerpts, then verify the subject, period, units and public comparison rules before deciding. Return the evidence and a concise conclusion; do not expose chain-of-thought. Abstain or mark insufficient if the supplied evidence does not resolve the task.",
}
BATCH_INSTRUCTIONS = (
    "Process every task in items exactly once. Apply only the condition prompt assigned by the operator. "
    "Use only the supplied public payloads; do not access hidden answer keys, other conditions' answers, "
    "or external sources. Do not infer labels from task order or identifiers. Each task has its own payload. "
    "Return one JSON answer file matching answer_file_schema, with the exact packet_sha256, assigned "
    "condition and every task_id. Do not omit tasks or fabricate missing answers. "
    "The operator supplies provenance identifying the actual collection session."
)
_RESOURCE_KEYS = ("input_tokens", "output_tokens", "latency_seconds", "mean_latency_seconds", "cost", "known_cost", "cost_unknown_calls")


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_artifact(path):
    data = Path(path).read_bytes()
    if len(data) > 20 * 1024 * 1024:
        raise ValueError("JSON file exceeds the 20 MB limit")
    try:
        value = _json_loads(data.decode("utf-8-sig"))
    except UnicodeDecodeError:
        raise ValueError("JSON files must be UTF-8") from None
    if not isinstance(value, dict):
        raise ValueError("JSON file must contain an object")
    return value, data


def read_json(path) -> dict:
    return _read_artifact(path)[0]


def _dataset(reports, gold):
    return validate_dataset(read_json(reports), read_json(gold))


def _condition_config(conditions=None, prompts=None):
    chosen = list(CONDITIONS if conditions is None else conditions)
    if not chosen or any(not isinstance(condition, str) for condition in chosen) or len(set(chosen)) != len(chosen) or any(condition not in CONDITIONS for condition in chosen):
        raise ValueError("conditions must be unique baseline and/or evidence_first")
    chosen.sort(key=CONDITIONS.index)
    prompts = {} if prompts is None else prompts
    if not isinstance(prompts, dict) or set(prompts) - set(chosen):
        raise ValueError("condition_prompts must map selected condition names to strings")
    result = {condition: prompts.get(condition, DEFAULT_PROMPTS[condition]) for condition in chosen}
    if any(not isinstance(text, str) or not text.strip() for text in result.values()):
        raise ValueError("each condition prompt must be a non-empty string")
    return chosen, result


def _answer_file_schema(stage, conditions):
    provenance = {
        "type": "object", "additionalProperties": False,
        "required": ["agent_id", "collection_method", "model_identity"],
        "properties": {
            "agent_id": {"type": "string"},
            "collection_method": {"type": "string", "enum": ["codex_subagent"]},
            "model_identity": {"type": "string", "enum": [MODEL_IDENTITY]},
            "recorded_at": {"type": "string"}, "notes": {"type": "string"},
        },
    }
    properties = {
        "schema_version": {"type": "string", "enum": [ANSWER_VERSION]},
        "stage": {"type": "string", "enum": [stage]},
        "packet_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
        "condition": {"type": "string", "enum": list(conditions)},
        "provenance": provenance,
        "answers": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["task_id", "answer"],
            "properties": {"task_id": {"type": "string"}, "answer": _schema(stage)},
        }},
    }
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


def _packet_base(stage, conditions, prompts):
    return {
        "schema_version": PACKET_VERSION, "stage": stage,
        "hash_algorithm": "sha256-canonical-json-v1",
        "provider_prompt_version": PROMPT_VERSION,
        "conditions": conditions, "condition_prompts": prompts,
        "common_instructions": _COMMON_INSTRUCTIONS + "\n\n" + _STAGE_INSTRUCTIONS[stage] + "\n\n" + BATCH_INSTRUCTIONS,
        "response_schema": _schema(stage),
        "answer_file_schema": _answer_file_schema(stage, conditions),
    }


def _seal(packet):
    return {**packet, "packet_sha256": _sha(packet)}


def _assert_packet_hash(packet, stage):
    if packet.get("stage") != stage or packet.get("schema_version") != PACKET_VERSION:
        raise ValueError(f"expected a {stage} {PACKET_VERSION} packet")
    digest = packet.get("packet_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("packet_sha256 must be a lowercase SHA256 digest")
    if digest != _sha({key: value for key, value in packet.items() if key != "packet_sha256"}):
        raise ValueError("packet content hash mismatch; the packet was modified")


def _fill_packet(dataset, conditions=None, prompts=None):
    conditions, prompts = _condition_config(conditions, prompts)
    report_map = {report["id"]: report for report in dataset["reports"]}
    packet = _packet_base("fill", conditions, prompts)
    packet["items"] = []
    for item in dataset["items"]:
        payload = _public_payload(item_context(item, report_map[item["report_id"]]), "fill")
        packet["items"].append({"task_id": "fill_" + _sha(payload)[:24], "payload": payload})
    return _seal(packet)


def _validate_fill_packet(packet, dataset):
    _assert_packet_hash(packet, "fill")
    expected = _fill_packet(dataset, packet.get("conditions"), packet.get("condition_prompts"))
    if packet != expected:
        raise ValueError("fill packet does not match the supplied dataset, prompts or supported packet format")
    return packet


def _save_packet(packet, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never silently overwrite an issued packet and its immutable commitment.
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return packet


def prepare_fill(reports, gold, output, *, conditions=None, condition_prompts=None):
    """Save only public tasks, public comparison rules and fixed instructions."""
    return _save_packet(_fill_packet(_dataset(reports, gold), conditions, condition_prompts), output)


def _answer_batches(paths, packet):
    expected_tasks = {task["task_id"] for task in packet["items"]}
    required = {"schema_version", "stage", "packet_sha256", "condition", "provenance", "answers"}
    batches = {}
    for path in paths:
        batch, raw_bytes = _read_artifact(path)
        if set(batch) != required:
            raise ValueError("answer file has missing or additional envelope fields")
        if batch["schema_version"] != ANSWER_VERSION or batch["stage"] != packet["stage"]:
            raise ValueError("answer file schema version or stage mismatch")
        if batch["packet_sha256"] != packet["packet_sha256"]:
            raise ValueError("answer file packet_sha256 mismatch")
        condition = batch["condition"]
        if not isinstance(condition, str) or condition not in packet["conditions"] or condition in batches:
            raise ValueError("unknown or duplicate answer condition")
        provenance = batch["provenance"]
        fields = {"agent_id", "collection_method", "model_identity"}
        if (not isinstance(provenance, dict) or fields - set(provenance)
                or set(provenance) - fields - {"recorded_at", "notes"}
                or any(not isinstance(value, str) or not value.strip() for value in provenance.values())):
            raise ValueError("invalid answer provenance")
        if provenance["collection_method"] != "codex_subagent" or provenance["model_identity"] != MODEL_IDENTITY:
            raise ValueError("this importer requires declared codex_subagent collection and unverified exact model identity")
        if not isinstance(batch["answers"], list):
            raise ValueError("answers must be an array")
        rows = {}
        for row in batch["answers"]:
            if not isinstance(row, dict) or set(row) != {"task_id", "answer"}:
                raise ValueError("each answer row must contain only task_id and answer")
            task_id = row["task_id"]
            if not isinstance(task_id, str) or task_id not in expected_tasks or task_id in rows:
                raise ValueError("unknown or duplicate task_id")
            _validate_answer(row["answer"], packet["stage"])
            rows[task_id] = copy.deepcopy(row["answer"])
        if set(rows) != expected_tasks:
            raise ValueError("missing task answers; omissions are never filled automatically")
        batches[condition] = {
            "file": Path(path), "file_sha256": hashlib.sha256(raw_bytes).hexdigest(), "raw_bytes": raw_bytes,
            "document": batch, "answers": rows, "provenance": provenance,
        }
    if set(batches) != set(packet["conditions"]):
        raise ValueError("one answer file for every packet condition is required")
    return batches


def _unknown_resources():
    return {**{field: None for field in _RESOURCE_KEYS}, "resource_measurement_status": "not_observed"}


def _fill_rows(dataset, packet, batches):
    item_map = {item["id"]: item for item in dataset["items"]}
    report_map = {report["id"]: report for report in dataset["reports"]}
    rows = []
    for condition in packet["conditions"]:
        for task in packet["items"]:
            item = item_map[task["payload"]["item_id"]]
            answer = batches[condition]["answers"][task["task_id"]]
            score = score_answer(item, answer, report_map[item["report_id"]])
            if score["outcome"] == "error":
                raise ValueError("recorded fill answer could not be numerically scored")
            rows.append({
                "id": f"fill-{condition}-{task['task_id']}", "model_id": condition, "condition": condition,
                "item_id": item["id"], "report_id": item["report_id"], "repeat": 1,
                "task_id": task["task_id"], "answer": copy.deepcopy(answer), **score,
                "call_id": f"fill-{condition}-{task['task_id']}", "error": None, **_unknown_resources(),
            })
    return rows


def _pool(dataset, fills):
    warnings = []
    pool = _make_pool(dataset["items"], fills, warnings)
    if warnings:
        raise ValueError("a recorded or supplied candidate could not be included in the common pool")
    for candidate in pool:
        for origin in candidate["origins"]:
            if origin["type"] == "historical":
                # Existing core labels every supplied candidate historical. Keep
                # explicit synthetic provenance honest in this separate importer.
                declared = origin.get("declared_origin")
                origin["type"] = declared or "provided"
    return pool


def _check_packet(dataset, fill_packet, batches, pool, seed):
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("shuffle seed must be an integer from 0 to 2^63-1")
    conditions = fill_packet["conditions"]
    packet = _packet_base("check", conditions, fill_packet["condition_prompts"])
    packet.update({
        "fill_packet_sha256": fill_packet["packet_sha256"], "shuffle_seed": seed,
        "fill_answer_file_sha256": {condition: batches[condition]["file_sha256"] for condition in conditions},
        "items": [],
    })
    item_map = {item["id"]: item for item in dataset["items"]}
    report_map = {report["id"]: report for report in dataset["reports"]}
    task_map = {}
    for candidate in pool:
        item = item_map[candidate["item_id"]]
        payload = _public_payload({
            **item_context(item, report_map[item["report_id"]]),
            "candidate": {"value": candidate["value"], "unit": candidate["unit"]},
        }, "check")
        task_id = "check_" + _sha({"seed": seed, "payload": payload})[:24]
        if task_id in task_map:
            raise ValueError("duplicate opaque task identifier")
        packet["items"].append({"task_id": task_id, "payload": payload})
        task_map[task_id] = candidate
    # The private pool inserts supported controls first. A public shuffle seed
    # makes that insertion order reversible, even with opaque task identifiers.
    # Canonicalize by the public-payload hash before shuffling so neither forward
    # nor inverse ordering carries the private control/provenance sequence.
    packet["items"].sort(key=lambda task: task["task_id"])
    random.Random(seed).shuffle(packet["items"])
    return _seal(packet), task_map


def prepare_check(reports, gold, fills, output, *, fill_packet=None, seed=DEFAULT_SEED):
    """Validate complete fill batches; save a shuffled, unlabeled common pool."""
    dataset = _dataset(reports, gold)
    packet = _validate_fill_packet(read_json(fill_packet), dataset) if fill_packet else _fill_packet(dataset)
    batches = _answer_batches(fills, packet)
    rows = _fill_rows(dataset, packet, batches)
    check_packet, _ = _check_packet(dataset, packet, batches, _pool(dataset, rows), seed)
    return _save_packet(check_packet, output)


def _metric_rows(function, models, population, rows):
    # Core aggregation requires numeric latency; this scratch value is never
    # persisted or reported. Every unobserved resource field is replaced by null.
    scratch = [{**row, "latency_seconds": 0.0} for row in rows]
    return [{**row, **_unknown_resources()} for row in function(models, population, 1, scratch)]


def _import_log(stage, packet, batches, task, row):
    condition = row["condition"]
    return {
        **row, "stage": stage, "record_type": "imported_batch_answer",
        "batch_id": f"{stage}-{condition}",
        "request": {
            "packet_sha256": packet["packet_sha256"], "task_id": task["task_id"],
            "instructions": packet["common_instructions"] + "\n\n" + packet["condition_prompts"][condition],
            "payload": task["payload"],
        },
        "provenance": batches[condition]["provenance"],
        "raw_batch_file": f"raw.{stage}.{condition}.json",
        "raw_batch_file_sha256": batches[condition]["file_sha256"],
        "raw_text": json.dumps(row["answer"], ensure_ascii=False, allow_nan=False),
        "raw_text_representation": "decoded answer object; original complete batch bytes are preserved in raw_batch_file",
    }


def grade_recorded(reports, gold, fill_packet, fills, check_packet, checks, output):
    """Strictly validate all artifacts before saving a UI-compatible complete run."""
    reports_data, reports_bytes = _read_artifact(reports)
    gold_data, gold_bytes = _read_artifact(gold)
    fill_data, fill_bytes = _read_artifact(fill_packet)
    check_data, check_bytes = _read_artifact(check_packet)
    dataset = validate_dataset(reports_data, gold_data)
    fill_public = _validate_fill_packet(fill_data, dataset)
    fill_batches = _answer_batches(fills, fill_public)
    fill_rows = _fill_rows(dataset, fill_public, fill_batches)
    pool = _pool(dataset, fill_rows)
    check_public = check_data
    _assert_packet_hash(check_public, "check")
    expected, task_map = _check_packet(dataset, fill_public, fill_batches, pool, check_public.get("shuffle_seed"))
    if check_public != expected:
        raise ValueError("check packet does not match the original fill artifacts, common pool or supported format")
    check_batches = _answer_batches(checks, check_public)
    item_map = {item["id"]: item for item in dataset["items"]}
    report_map = {report["id"]: report for report in dataset["reports"]}
    check_rows = []
    for condition in check_public["conditions"]:
        for task in check_public["items"]:
            candidate = task_map[task["task_id"]]
            item = item_map[candidate["item_id"]]
            answer = check_batches[condition]["answers"][task["task_id"]]
            evidence = score_answer(item, {
                "status": "answered", "value": candidate["value"], "unit": candidate["unit"],
                "evidence": answer["evidence"],
            }, report_map[item["report_id"]])
            check_rows.append({
                "id": f"check-{condition}-{task['task_id']}", "model_id": condition, "condition": condition,
                "item_id": item["id"], "report_id": item["report_id"], "repeat": 1,
                "task_id": task["task_id"], "candidate_id": candidate["id"],
                "truth": candidate["truth"], "verdict": answer["verdict"], "answer": copy.deepcopy(answer),
                "evidence_valid": evidence["evidence_valid"], "evidence_note": evidence["evidence_note"],
                "call_id": f"check-{condition}-{task['task_id']}", "error": None, **_unknown_resources(),
            })
    conditions = fill_public["conditions"]
    models = [{"id": condition, "provider": "recorded_agent", "model": MODEL_IDENTITY} for condition in conditions]
    output = Path(output)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", output.name):
        raise ValueError("output directory name must use ASCII letters, digits, hyphens or underscores for UI compatibility")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be new or empty; recorded runs are not overwritten")
    count = len(fill_rows) + len(check_rows)
    result = {
        "run_id": output.name, "status": "completed", "demo": False,
        "run_mode": "recorded_agent_pilot", "conditions_same_model": True,
        "model_identity": MODEL_IDENTITY, "created_at": _now(), "finished_at": _now(),
        "warnings": [
            "同じ継承実行環境で収集したLLM回答のプロンプト条件比較です。APIモデル間の性能比較ではありません。",
            "正確なAPIモデルID・トークン利用量・費用・生成待ち時間は未観測のため null です。",
            "各条件は一括回答1回です。calls.jsonlの行は取り込んだ回答であり、個別API呼び出しの観測ではありません。",
            "公開packetのhashは内容の一致を確認します。solverが外部情報を見なかったことを暗号学的に証明するものではありません。",
            "共通候補には正しい対照値と提供された誤り候補を含みます。候補の誤り率は実運用の誤り率ではありません。",
            EVIDENCE_WARNING, CI_WARNING,
        ],
        "conditions": [{"id": condition, "prompt": fill_public["condition_prompts"][condition]} for condition in conditions],
        "counts": {
            "reports": len(dataset["reports"]), "items": len(dataset["items"]), "models": len(conditions),
            "conditions": len(conditions), "repeats": 1,
            "fill_requested": len(fill_rows), "fill_completed": len(fill_rows), "fill_not_run": 0,
            "candidates": len(pool), "candidate_pool_complete": True,
            "check_requested": len(check_rows), "check_completed": len(check_rows), "check_not_run": 0,
            "calls_completed": None, "calls_requested": None, "calls_requested_final": False,
            "imported_answers": count, "recorded_batches": len(conditions) * 2, "errors": 0,
        },
        "fill_metrics": _metric_rows(_fill_metrics, models, dataset["items"], fill_rows),
        "check_metrics": _metric_rows(_check_metrics, models, pool, check_rows),
        "pipeline_metrics": _pipeline_metrics(models, dataset["items"], 1, fill_rows, check_rows),
        "fills": fill_rows, "candidates": pool, "checks": check_rows, "errors": [],
        "resources": _unknown_resources(),
    }
    provenance = {
        "collection_mode": "recorded_agent_pilot", "model_identity": MODEL_IDENTITY,
        "conditions_same_model": True, "packet_hash_algorithm": "sha256-canonical-json-v1",
        "fill_packet_sha256": fill_public["packet_sha256"], "check_packet_sha256": check_public["packet_sha256"],
        "provider_prompt_version": fill_public["provider_prompt_version"],
        "condition_prompts": fill_public["condition_prompts"],
        "common_instructions": {"fill": fill_public["common_instructions"], "check": check_public["common_instructions"]},
        "input_files": [], "batches": [],
        "recorded_at_is_import_time": True,
    }
    source_files = [("reports.json", reports, reports_bytes), ("gold.json", gold, gold_bytes),
                    ("fill_packet.json", fill_packet, fill_bytes), ("check_packet.json", check_packet, check_bytes)]
    for stage, batches in (("fill", fill_batches), ("check", check_batches)):
        for condition in conditions:
            batch = batches[condition]
            filename = f"raw.{stage}.{condition}.json"
            source_files.append((filename, batch["file"], batch["raw_bytes"]))
            provenance["batches"].append({
                "stage": stage, "condition": condition, "raw_file": filename,
                "raw_file_sha256": batch["file_sha256"], "provenance": batch["provenance"],
            })
    # Preserve the exact byte sequences that were validated, even if a caller
    # edits an original file while this import is running.
    copies = []
    for filename, path, content in source_files:
        copies.append((filename, content))
        provenance["input_files"].append({"filename": filename, "source_path": str(Path(path).resolve()), "sha256": hashlib.sha256(content).hexdigest()})
    result["provenance"] = provenance
    logs = []
    for stage, packet, batches, rows in (("fill", fill_public, fill_batches, fill_rows), ("check", check_public, check_batches, check_rows)):
        public_tasks = {task["task_id"]: task for task in packet["items"]}
        logs.extend(_import_log(stage, packet, batches, public_tasks[row["task_id"]], row) for row in rows)
    output.mkdir(parents=True, exist_ok=True)
    for filename, content in copies:
        (output / filename).write_bytes(content)
    _write_json(output / "models.json", {"models": models, "repeats": 1, "run_mode": "recorded_agent_pilot", "conditions_same_model": True})
    _write_json(output / "provenance.json", provenance)
    with (output / "calls.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in logs:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    for name in ("fills", "candidates", "checks", "fill_metrics", "check_metrics", "pipeline_metrics", "errors"):
        _write_csv(output / f"{name}.csv", result[name])
    _write_json(output / "summary.json", result)
    return result
