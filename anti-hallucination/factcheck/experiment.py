"""Reproducible local pilot runner. No gold data is sent to model calls.

Rates use requested tasks as their denominator, including errors. In a partial
run, not_run_rate is reported separately. Evidence validity is a literal quote
and source check, never a claim that the quote entails the candidate.
"""

from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable
import uuid

from .dataset import item_context, validate_dataset
from .providers import call_model, validate_models
from .scoring import candidate_truth, score_answer


EVIDENCE_WARNING = (
    "evidence_valid は、指定出典と引用文字列の存在を確認するだけです。"
    "引用が主張を意味的に裏付けることは保証しません。"
)
CI_WARNING = (
    "この試作は信頼区間・統計的有意差を算出しません。"
    "同じ文書の問題や反復を独立標本とは見なせません。"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _redact(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {key: _redact(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, str):
        stripped = value.lstrip()
        if stripped and stripped[0] in "=+-@":
            try:
                numeric = Decimal(stripped).is_finite()
            except InvalidOperation:
                numeric = False
            if not numeric:
                return "'" + value
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: _csv_cell(value) for key, value in row.items()} for row in rows)


def _numeric_key(value: Any) -> Decimal | None:
    try:
        result = Decimal(str(value))
        return result if result.is_finite() else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def _resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    known_costs = [row["cost"] for row in rows if row.get("cost") is not None]
    latency = sum(row.get("latency_seconds", 0.0) for row in rows)
    return {
        "latency_seconds": latency,
        "mean_latency_seconds": latency / len(rows) if rows else None,
        "cost": sum(known_costs) if rows and len(known_costs) == len(rows) else None,
        "known_cost": sum(known_costs),
        "cost_unknown_calls": len(rows) - len(known_costs),
        "input_tokens": sum(row["input_tokens"] for row in rows)
        if rows and all(row.get("input_tokens") is not None for row in rows) else None,
        "output_tokens": sum(row["output_tokens"] for row in rows)
        if rows and all(row.get("output_tokens") is not None for row in rows) else None,
    }


def _call_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in (
        "call_id", "latency_seconds", "input_tokens", "output_tokens", "cost", "error"
    )}


def _make_pool(
    items: list[dict[str, Any]], fills: list[dict[str, Any]], warnings: list[str]
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    index: dict[tuple[str, Decimal, str], dict[str, Any]] = {}
    item_map = {item["id"]: item for item in items}

    def add(item: dict[str, Any], value: Any, unit: Any, origin: dict[str, Any]) -> str | None:
        number = _numeric_key(value)
        if number is None or not isinstance(unit, str):
            return None
        key = (item["id"], number, unit)
        if key not in index:
            candidate = {"value": str(value), "unit": unit}
            try:
                truth = candidate_truth(item, candidate)
            except ValueError:
                return None
            row = {
                "id": f"candidate-{len(pool) + 1:06d}",
                "item_id": item["id"], "report_id": item["report_id"],
                **candidate, "truth": truth, "origins": [],
            }
            index[key] = row
            pool.append(row)
        index[key]["origins"].append(origin)
        return index[key]["id"]

    for item in items:
        if add(item, item["value"], item["unit"], {"type": "supported_control"}) is None:
            raise ValueError(f"正解対照候補を作成できません: {item['id']}")
        for offset, candidate in enumerate(item.get("candidates", [])):
            declared_origin = candidate.get("origin")
            origin_type = declared_origin if declared_origin in ("historical", "synthetic_perturbation") else "provided_candidate"
            origin = {"type": origin_type, "index": offset,
                      "declared_origin": declared_origin,
                      "declared_verdict": candidate.get("historical_verdict")}
            if add(item, candidate["value"], candidate.get("unit", item["unit"]), origin) is None:
                warnings.append(f"不正な既存候補を除外しました: {item['id']} / {offset}")
    for fill in fills:
        fill["candidate_id"] = None
        answer = fill["answer"]
        if fill["outcome"] not in ("correct", "incorrect") or answer.get("status") != "answered":
            continue
        fill["candidate_id"] = add(
            item_map[fill["item_id"]], answer.get("value"), answer.get("unit"),
            {"type": "generated", "model_id": fill["model_id"],
             "repeat": fill["repeat"], "fill_id": fill["id"]},
        )
        if fill["candidate_id"] is None:
            warnings.append(f"判定候補にできない生成回答を除外しました: {fill['id']}")
    return pool


def _fill_metrics(models: list[dict[str, Any]], items: list[dict[str, Any]], repeats: int,
                  fills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    n = len(items) * repeats
    for model in models:
        rows = [row for row in fills if row["model_id"] == model["id"]]
        counts = {name: sum(row["outcome"] == name for row in rows)
                  for name in ("correct", "incorrect", "abstain", "error")}
        result.append({
            "model_id": model["id"], "n": n, "n_completed": len(rows),
            "n_not_run": n - len(rows),
            **{f"n_{name}": count for name, count in counts.items()},
            **{f"{name}_rate": _rate(count, n) for name, count in counts.items()},
            "not_run_rate": _rate(n - len(rows), n),
            "evidence_valid_rate": _rate(sum(bool(row["evidence_valid"]) for row in rows), n),
            **_resources(rows),
        })
    return result


def _check_metrics(models: list[dict[str, Any]], candidates: list[dict[str, Any]], repeats: int,
                   checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    n = len(candidates) * repeats
    n_supported = sum(candidate["truth"] == "supported" for candidate in candidates) * repeats
    n_refuted = n - n_supported
    for model in models:
        rows = [row for row in checks if row["model_id"] == model["id"]]
        matrix = {
            truth: {verdict: sum(row["truth"] == truth and row["verdict"] == verdict for row in rows)
                    for verdict in ("supported", "refuted", "insufficient", "error")}
            for truth in ("supported", "refuted")
        }
        matrix["supported"]["not_run"] = n_supported - sum(matrix["supported"].values())
        matrix["refuted"]["not_run"] = n_refuted - sum(matrix["refuted"].values())
        passed = matrix["supported"]["supported"] + matrix["refuted"]["supported"]
        abstained = sum(matrix[truth]["insufficient"] for truth in matrix)
        errors = sum(matrix[truth]["error"] for truth in matrix)
        result.append({
            "model_id": model["id"], "n": n, "n_completed": len(rows),
            "n_not_run": n - len(rows), "n_supported": n_supported, "n_refuted": n_refuted,
            "n_passed": passed, "n_abstain": abstained, "n_error": errors,
            "error_detection_rate": _rate(matrix["refuted"]["refuted"], n_refuted),
            "false_flag_rate": _rate(matrix["supported"]["refuted"], n_supported),
            "wrong_pass_rate": _rate(matrix["refuted"]["supported"], n_refuted),
            "correct_pass_rate": _rate(matrix["supported"]["supported"], n_supported),
            "abstain_rate": _rate(abstained, n), "error_rate": _rate(errors, n),
            "passed_error_rate": _rate(matrix["refuted"]["supported"], passed),
            "pass_rate": _rate(passed, n), "not_run_rate": _rate(n - len(rows), n),
            "evidence_valid_rate": _rate(sum(bool(row["evidence_valid"]) for row in rows), n),
            "confusion_matrix": matrix, **_resources(rows),
        })
    return result


def _pipeline_metrics(models: list[dict[str, Any]], items: list[dict[str, Any]], repeats: int,
                      fills: list[dict[str, Any]], checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fill_map = {(row["model_id"], row["item_id"], row["repeat"]): row for row in fills}
    check_map = {(row["model_id"], row["candidate_id"], row["repeat"]): row for row in checks}
    n = len(items) * repeats
    results = []
    for generator in models:
        for checker in models:
            counts = {name: 0 for name in ("correct_provided", "incorrect_provided", "not_provided", "not_run")}
            for item in items:
                for repeat in range(1, repeats + 1):
                    fill = fill_map.get((generator["id"], item["id"], repeat))
                    if fill is None:
                        counts["not_run"] += 1
                    elif fill["outcome"] in ("abstain", "error") or fill.get("candidate_id") is None:
                        counts["not_provided"] += 1
                    else:
                        check = check_map.get((checker["id"], fill["candidate_id"], repeat))
                        if check is None:
                            counts["not_run"] += 1
                        elif check["verdict"] != "supported":
                            counts["not_provided"] += 1
                        elif fill["outcome"] == "correct":
                            counts["correct_provided"] += 1
                        else:
                            counts["incorrect_provided"] += 1
            results.append({
                "generator_id": generator["id"], "checker_id": checker["id"], "n": n,
                "n_completed": n - counts["not_run"],
                **{f"n_{name}": count for name, count in counts.items()},
                **{f"{name}_rate": _rate(count, n) for name, count in counts.items()},
            })
    return results


class _Cancelled(Exception):
    pass


def run_experiment(
    reports: dict[str, Any], gold: dict[str, Any], config: dict[str, Any], output_dir: Path,
    progress: Callable[[dict[str, Any]], Any] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Validate first, then execute and save a complete or explicitly partial run.

    Bad input/credentials raise ValueError before output or model calls. Provider
    failures remain individual error observations; they do not erase a trial.
    All cost fields are in provider-configured currency (normally USD); missing
    prices/usage produce null, never a fabricated zero price.
    """
    dataset = validate_dataset(copy.deepcopy(reports), copy.deepcopy(gold))
    settings = validate_models(copy.deepcopy(config), check_credentials=True)
    models, repeats = settings["models"], settings.get("repeats", 1)
    items, report_list = dataset["items"], dataset["reports"]
    report_map = {report["id"]: report for report in report_list}
    item_map = {item["id"]: item for item in items}
    contexts = {item["id"]: item_context(item, report_map[item["report_id"]]) for item in items}
    secrets = sorted({os.environ[model["api_key_env"]] for model in models
                      if model.get("api_key_env") and os.environ.get(model["api_key_env"])}, key=len, reverse=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in ("summary.json", "calls.jsonl", "gold.json")):
        raise ValueError("出力先に既存の実験があります。新しいディレクトリを指定してください。")
    created_at = _now()
    demo_count = sum(model["provider"] == "demo" for model in models)
    warnings = [EVIDENCE_WARNING, CI_WARNING,
                "cost が null の場合、価格または利用量が不明です。既知の費用のみ known_cost に集計します。",
                "判定候補群には正解対照が含まれます。候補群の誤り比率は実運用の誤り比率ではありません。"]
    if demo_count:
        warnings.append("デモには意図的に作った合成応答を使用しています。実モデルの性能を示す実験結果ではありません。")
    if 0 < demo_count < len(models):
        warnings.append("デモモデルと実APIモデルが混在しています。両者の結果を性能比較に用いないでください。")
    result: dict[str, Any] = {
        "run_id": f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}",
        "status": "completed", "demo": bool(demo_count),
        "run_mode": "demo" if demo_count == len(models) else "mixed" if demo_count else "live",
        "created_at": created_at, "warnings": warnings, "counts": {},
        "fill_metrics": [], "check_metrics": [], "pipeline_metrics": [],
        "fills": [], "candidates": [], "checks": [], "errors": [],
    }
    _write_json(output_dir / "reports.json", _redact({"reports": report_list}, secrets))
    _write_json(output_dir / "gold.json", _redact({"items": items}, secrets))
    _write_json(output_dir / "models.json", _redact(settings, secrets))
    fill_requested = len(models) * len(items) * repeats
    call_number = 0
    progress_failed = False

    def notify(stage: str, completed: int, total: int, message: str) -> None:
        nonlocal progress_failed
        if progress and not progress_failed:
            try:
                progress({"stage": stage, "completed": completed, "total": total, "message": message})
            except Exception:
                progress_failed = True
                warnings.append("進捗コールバックが失敗しました。実験と結果保存は継続しました。")

    def stop_if_cancelled() -> None:
        if cancel and cancel():
            raise _Cancelled()

    with (output_dir / "calls.jsonl").open("w", encoding="utf-8") as log:
        def invoke(model: dict[str, Any], stage: str, payload: dict[str, Any], item: dict[str, Any],
                   repeat: int, candidate_id: str | None = None) -> dict[str, Any]:
            nonlocal call_number
            call_number += 1
            started, timer = _now(), time.perf_counter()
            try:
                response = call_model(copy.deepcopy(model), stage, copy.deepcopy(payload))
                if not isinstance(response, dict) or not isinstance(response.get("answer"), dict):
                    raise ValueError("モデルの応答形式が不正です。")
                response = dict(response)
                answer = response["answer"]
                if not response.get("error"):
                    valid = answer.get("status") in ("answered", "abstain") if stage == "fill" else answer.get("verdict") in ("supported", "refuted", "insufficient")
                    if not valid:
                        response["error"] = "モデル応答の status/verdict が不正です。"
            except Exception as exc:
                response = {"answer": {"status": "error"}, "raw_text": "", "error": str(exc),
                            "input_tokens": None, "output_tokens": None, "cost": None}
            elapsed = response.get("latency_seconds")
            if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                elapsed = time.perf_counter() - timer
            record = _redact({
                **response, "call_id": f"call-{call_number:06d}", "stage": stage,
                "model_id": model["id"], "item_id": item["id"], "report_id": item["report_id"],
                "repeat": repeat, "candidate_id": candidate_id,
                "started_at": started, "finished_at": _now(), "latency_seconds": elapsed,
                "request": response.get("request") or {"stage": stage, "payload": copy.deepcopy(payload)},
            }, secrets)
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            os.fsync(log.fileno())
            if record.get("error"):
                result["errors"].append({key: record.get(key) for key in
                                         ("call_id", "stage", "model_id", "item_id", "repeat", "error")})
            return record

        notify("fill", 0, fill_requested, "穴埋め回答を生成します。")
        try:
            for model in models:
                for item in items:
                    for repeat in range(1, repeats + 1):
                        stop_if_cancelled()
                        call = invoke(model, "fill", contexts[item["id"]], item, repeat)
                        answer = call["answer"]
                        if call.get("error"):
                            score = {"outcome": "error", "normalized_value": None,
                                     "evidence_valid": False, "evidence_note": "モデル呼び出しまたは応答形式のエラー"}
                        else:
                            score = score_answer(item, answer, report_map[item["report_id"]])
                        row = {"id": f"fill-{len(result['fills']) + 1:06d}", "model_id": model["id"],
                               "item_id": item["id"], "report_id": item["report_id"], "repeat": repeat,
                               "answer": answer, **score, **_call_fields(call)}
                        result["fills"].append(row)
                        if score["outcome"] == "error" and not call.get("error"):
                            result["errors"].append({"stage": "fill_score", "model_id": model["id"],
                                                     "item_id": item["id"], "repeat": repeat,
                                                     "error": score.get("evidence_note", "数値回答を採点できません。")})
                        notify("fill", len(result["fills"]), fill_requested, f"穴埋め: {model['id']} / {item['id']}")
        except _Cancelled:
            result["status"] = "cancelled"
        except Exception as exc:
            result["status"] = "failed"
            result["errors"].append({"stage": "fill_runner", "error": str(exc)})

        try:
            result["candidates"] = _make_pool(items, result["fills"], warnings)
        except Exception as exc:
            result["status"] = "failed"
            result["errors"].append({"stage": "pool", "error": str(exc)})
        check_requested = len(models) * len(result["candidates"]) * repeats
        if result["status"] == "completed":
            notify("check", 0, check_requested, "全判定モデルに同じ候補群を渡します。")
            try:
                for model in models:
                    for candidate in result["candidates"]:
                        item = item_map[candidate["item_id"]]
                        payload = {**copy.deepcopy(contexts[item["id"]]),
                                   "candidate": {"value": candidate["value"], "unit": candidate["unit"]}}
                        for repeat in range(1, repeats + 1):
                            stop_if_cancelled()
                            call = invoke(model, "check", payload, item, repeat, candidate["id"])
                            answer = call["answer"]
                            verdict = "error" if call.get("error") else answer["verdict"]
                            evidence = score_answer(item, {
                                "status": "answered", "value": candidate["value"], "unit": candidate["unit"],
                                "evidence": answer.get("evidence", []),
                            }, report_map[item["report_id"]]) if verdict != "error" else {
                                "evidence_valid": False, "evidence_note": "モデル呼び出しまたは応答形式のエラー"}
                            result["checks"].append({
                                "id": f"check-{len(result['checks']) + 1:06d}", "model_id": model["id"],
                                "candidate_id": candidate["id"], "item_id": item["id"],
                                "report_id": item["report_id"], "repeat": repeat,
                                "truth": candidate["truth"], "verdict": verdict, "answer": answer,
                                "evidence_valid": evidence["evidence_valid"], "evidence_note": evidence["evidence_note"],
                                **_call_fields(call),
                            })
                            notify("check", len(result["checks"]), check_requested,
                                   f"判定: {model['id']} / {candidate['id']}")
            except _Cancelled:
                result["status"] = "cancelled"
            except Exception as exc:
                result["status"] = "failed"
                result["errors"].append({"stage": "check_runner", "error": str(exc)})

    if result["status"] != "completed":
        warnings.append("未完了の部分結果です。未実行は not_run として残し、完了実験の成績として扱いません。")
    pool_complete = len(result["fills"]) == fill_requested
    if not pool_complete:
        warnings.append("生成が未完了のため候補群も暫定です。判定の予定件数は現在までに確定した候補群に対する件数です。")
    result["counts"] = {
        "reports": len(report_list), "items": len(items), "models": len(models), "repeats": repeats,
        "fill_requested": fill_requested, "fill_completed": len(result["fills"]),
        "fill_not_run": fill_requested - len(result["fills"]), "candidates": len(result["candidates"]),
        "candidate_pool_complete": pool_complete,
        "check_requested": check_requested, "check_completed": len(result["checks"]),
        "check_not_run": check_requested - len(result["checks"]),
        "calls_completed": call_number, "calls_requested": fill_requested + check_requested,
        "calls_requested_final": pool_complete, "errors": len(result["errors"]),
    }
    result["fill_metrics"] = _fill_metrics(models, items, repeats, result["fills"])
    result["check_metrics"] = _check_metrics(models, result["candidates"], repeats, result["checks"])
    result["pipeline_metrics"] = _pipeline_metrics(models, items, repeats, result["fills"], result["checks"])
    result["finished_at"] = _now()
    result["resources"] = _resources(result["fills"] + result["checks"])
    result = _redact(result, secrets)
    for name in ("fills", "candidates", "checks", "fill_metrics", "check_metrics", "pipeline_metrics", "errors"):
        _write_csv(output_dir / f"{name}.csv", result[name])
    _write_json(output_dir / "summary.json", result)
    notify(result["status"], call_number, fill_requested + check_requested, "実験結果を保存しました。")
    return result
