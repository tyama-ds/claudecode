"""Immutable report separating raw pipelines from normalized-ledger diagnostics.

No searches, semantic judgments, retries or experimental mutations occur here.
All planned documents remain visible. A failed raw pipeline never inherits the
result or safety counts of its normalized diagnostic comparator.
"""
from __future__ import annotations

import argparse
import copy
import difflib
from datetime import datetime, timezone
from pathlib import Path

import paired_trial as paired
import replication as rep
from citation_report import _validate_cases
from factcheck.claim_guard_v5 import sha256_json
from omission_report import STYLE, _count, _table, esc, paragraphs, source_link


COHORTS = {"R1": "regression", "R2": "regression", "N1": "new", "N2": "new"}
VARIANTS = (("current", "診断対照：同じ整合化台帳＋v6"), ("improved", "改善：整合化＋v7"))
REVIEW_LISTS = ("verdict", "design", "regression_notes", "new_notes", "recommendations", "limitations")


def _exact(left, right):
    return sha256_json(left) == sha256_json(right)


def _complete_document(doc):
    if doc.get("status") != "complete" or doc.get("missing") or doc.get("validation_errors") or not doc.get("paired"):
        raise ValueError("Completed diagnostic pair required: " + doc["id"])
    primary = doc["paired"]["primary"]
    denominator = _count(primary.get("denominator"), "primary denominator")
    eligible = primary.get("eligible_issue_ids")
    if not isinstance(eligible, list) or len(set(eligible)) != denominator or len(eligible) != denominator:
        raise ValueError("Common primary denominator mismatch")
    if {arm.get("variant") for arm in doc["arms"].values()} != {"current", "improved"} or len(doc["arms"]) != 2:
        raise ValueError("Both distinct diagnostic variants are required")
    for arm in doc["arms"].values():
        if arm.get("status") != "complete" or arm.get("missing") or arm.get("validation_errors") or not arm.get("metrics"):
            raise ValueError("Complete diagnostic arm required")
        counts = primary[arm["variant"]]
        for field in ("detected", "resolved", "resolved_by_removal"):
            if _count(counts.get(field), field) > denominator:
                raise ValueError("Primary count exceeds denominator")
        after = counts.get("after_status", {})
        if sum(_count(after.get(key), key) for key in rep.AFTER_STATUSES - {"not_an_original_problem"}) != denominator:
            raise ValueError("Primary status counts disagree")
        if after["resolved"] != counts["resolved"]:
            raise ValueError("Only fully resolved issues count as resolved")
        metrics = arm["metrics"]
        for group, choices in (("patch_reviews", rep.PATCH_STATUSES), ("new_problems", rep.PROBLEM_STATUSES),
                               ("controls", rep.CONTROL_STATUSES), ("posthoc_original_findings", rep.PROBLEM_STATUSES),
                               ("false_findings", {"false_alarm", "unverifiable"})):
            for key in choices:
                _count(metrics.get(group, {}).get(key), group + "/" + key)
        if sum(metrics["patch_reviews"].values()) != _count(metrics.get("changed_locations"), "changed_locations"):
            raise ValueError("Applied patch count differs from patch reviews")


def compose_summary(protocol, diagnostic, preflights, terminal_records=None):
    """Pure projection: main raw v6 vs normalized v7, diagnostic kept separate."""
    terminal_records = terminal_records or {}
    if protocol.get("gate_versions") != {"current": "v6", "improved": "v7"}:
        raise ValueError("Expected v6/v7 protocol")
    if diagnostic.get("experiment_id") != protocol["id"]:
        raise ValueError("Diagnostic experiment differs from protocol")
    specs = {spec["id"]: spec for spec in protocol["documents"]}
    rows = diagnostic.get("documents", [])
    if (len(specs) != 4 or len(protocol["documents"]) != 4 or set(specs) != set(COHORTS)
            or len(rows) != 4 or {row["id"] for row in rows} != set(COHORTS)):
        raise ValueError("Every planned R1/R2/N1/N2 document must remain visible")
    if set(terminal_records) - set(COHORTS) or set(preflights) - set(COHORTS):
        raise ValueError("Unknown document record")
    documents = []
    for doc in rows:
        identity = doc["id"]
        if doc["cohort"] != COHORTS[identity] or specs[identity]["cohort"] != COHORTS[identity]:
            raise ValueError("Regression and fresh cohorts must remain distinct")
        pf = preflights.get(identity)
        terminal = terminal_records.get(identity)
        complete = doc["status"] == "complete"
        if not complete and terminal is None:
            raise ValueError("Incomplete diagnostic requires explicit terminal record: " + identity)
        if complete:
            _complete_document(doc)
            if terminal is not None:
                raise ValueError("Completed diagnostic cannot also be terminal")
        if pf is not None:
            raw_valid = rep._boolean(pf.get("raw_v6", {}).get("valid"), "raw_v6.valid")
            norm = pf.get("normalization", {})
            if norm.get("status") not in {"complete", "failed"}:
                raise ValueError("Explicit normalization status required")
            aliases = rep._boolean(pf.get("current_aliases_normalized_v6"), "current alias")
            missing = rep._boolean(pf.get("raw_current_outcomes_missing"), "raw current missing")
            if pf.get("failure_is_zero_outcome") is not False:
                raise ValueError("Failed pipelines cannot be zero outcomes")
            if raw_valid:
                if norm.get("status") == "failed" and not complete:
                    if aliases or missing:
                        raise ValueError("Failed normalization cannot alias raw current")
                elif norm.get("status") != "complete" or norm.get("changed") is not False or aliases is not True or missing:
                    raise ValueError("Raw valid current may alias only an identical unchanged normalized ledger")
            elif aliases or not missing or not isinstance(pf["raw_v6"].get("error"), str) or not pf["raw_v6"]["error"]:
                raise ValueError("Raw invalid current must remain a failure with missing outcomes")
            if complete and norm.get("status") != "complete":
                raise ValueError("Complete diagnostic cannot use failed normalization")
        else:
            if complete:
                raise ValueError("Completed pair requires verified pipeline preflight")
            raw_valid = None
            norm = None

        diagnostic_copy = copy.deepcopy(doc)
        # No partial outcome becomes a final trial result on terminal documents.
        if not complete:
            diagnostic_copy["paired"] = None
            for arm in diagnostic_copy["arms"].values():
                arm["metrics"] = None
                arm["status"] = "terminal_incomplete"
        main_arms = {}
        for variant in ("current", "improved"):
            source = next((arm for arm in doc["arms"].values() if arm.get("variant") == variant), None)
            available = complete and (variant == "improved" or raw_valid is True)
            main_arms[variant] = {
                "variant": variant,
                "status": "complete" if available else "terminal_failed" if variant == "current" and raw_valid is False else "terminal_incomplete",
                "metrics": copy.deepcopy(source["metrics"]) if available else None,
                "result_source": ("normalized_v7" if variant == "improved" else "unchanged_ledger_v6_exact_alias") if available else None,
            }
        main = {"id": identity, "cohort": doc["cohort"],
                "status": "complete" if complete and raw_valid is True else "terminal_failed",
                "arms": main_arms, "paired": copy.deepcopy(doc["paired"]) if complete and raw_valid is True else None}
        documents.append({"id": identity, "cohort": doc["cohort"], "path": specs[identity]["path"],
                          "theme": specs[identity].get("theme", identity),
                          "main": main, "diagnostic": diagnostic_copy,
                          "pipeline_preflight": copy.deepcopy(pf), "normalization": copy.deepcopy(norm),
                          "terminal_failure": copy.deepcopy(terminal)})
    cohorts = {}
    for cohort in ("regression", "new"):
        selected = [doc for doc in documents if doc["cohort"] == cohort]
        main = paired._aggregate([doc["main"] for doc in selected])
        main.update(terminal_failed_documents=sum(doc["main"]["status"] != "complete" for doc in selected),
                    completion_by_pipeline={variant: sum(doc["main"]["arms"][variant]["status"] == "complete" for doc in selected)
                                            for variant in ("current", "improved")})
        cohorts[cohort] = {"main": main, "diagnostic": paired._aggregate([doc["diagnostic"] for doc in selected])}
    return {"schema_version": 1, "experiment_id": protocol["id"], "human_adjudicated": False,
            "main_comparison": "raw_v6_vs_normalized_v7",
            "diagnostic_comparison": "same_normalized_ledger_v6_vs_v7",
            "all_diagnostic_pairs_complete": all(doc["diagnostic"]["status"] == "complete" for doc in documents),
            "all_main_pairs_complete": all(doc["main"]["status"] == "complete" for doc in documents),
            "whole_trial_complete": all(doc["main"]["status"] == "complete" for doc in documents),
            "publication_accounts_for_all_planned_documents": True,
            "documents": documents, "cohorts": cohorts,
            "interpretation": [
                "Main pipelines use raw v6 versus normalization plus v7. A failed raw current has null outcomes, never normalized v6 or unchanged original output.",
                "Diagnostic results use exactly one normalized ledger for both gates. They isolate gate behavior and do not replace a failed main pipeline.",
                "Both gates receive the common expanded uncertainty-localization review instruction. Raw v6 is not an exact reproduction of the previous prompt configuration; historical-run differences are not attributed solely to the new gate.",
                "Raw-valid current is aliased only after normalization is verified unchanged. Aggregate paired effects include only complete main pairs.",
                "All planned failures remain visible. Zero harms with no delivered changes do not establish safe useful correction.",
                "Primary resolved, partially resolved, posthoc original findings, false findings, new problems, unnecessary changes and controls remain separate.",
                "Uncertainty holds are not confirmed errors. Additional review calls are recorded and shared joint calls are counted once.",
                "AI provisional assessment; hashes establish integrity, not truth or reviewer independence.",
            ] + protocol.get("limits", [])}


def validate_terminal(root, folder, identity):
    """Validate an explicit predeclared terminal record, never infer a success."""
    path = folder / "terminal_failure.json"
    record = rep._read_json(path)
    if record.get("schema_version") != 1 or type(record.get("schema_version")) is not int or record.get("document_id") != identity:
        raise ValueError("Terminal record schema/document mismatch")
    for field in ("stage", "reason", "recorded_at_utc"):
        rep._string(record.get(field), "terminal." + field, nonempty=True)
    if record.get("semantic_retry") is not False:
        raise ValueError("Terminal record cannot authorize a semantic retry")
    hashes = rep._object(record.get("input_file_sha256"), "terminal input hashes")
    if not hashes:
        raise ValueError("Terminal record needs bound input artifacts")
    actual = {item.relative_to(folder).as_posix() for item in folder.rglob("*") if item.is_file() and item != path}
    if set(hashes) != actual:
        raise ValueError("Terminal record must bind the complete current artifact set")
    for name, expected in hashes.items():
        if rep._sha(paired._relative(folder, name)) != expected:
            raise ValueError("Terminal input artifact changed: " + name)
    if not (folder / "input/original_report.json").is_file():
        raise ValueError("Terminal document original is required")
    return record


def declare_terminal(root, identity, stage, reason):
    """Freeze an operator-declared terminal failure without inventing outcomes.

    This records a declared stopping cause and bound artifacts. It is not a new
    semantic judgment or an automatic proof of the declared cause. Normalization
    failures are additionally reproduced by load_trial before publication.
    """
    import uncertainty_trial_ops as ops
    root = Path(root).resolve()
    protocol, _ = ops.runtime_check(root)
    spec = paired._document(protocol, identity)
    folder = paired._relative(root, spec["path"])
    _, hashes = paired.validate_preregistration(root)
    doc, _ = paired._document_record(root, protocol, spec, hashes)
    if doc["status"] == "complete":
        raise ValueError("Cannot declare a completed diagnostic pair terminal")
    if stage not in {"writer_input", "normalization", "patch_guard", "final_review", "post_evaluation", "orchestration"}:
        raise ValueError("Unknown terminal stage")
    rep._string(reason, "terminal reason", nonempty=True)
    rep.validate_original(rep._read_json(folder / "input/original_report.json"))
    target = folder / "terminal_failure.json"
    timestamp = rep._read_json(target)["recorded_at_utc"] if target.exists() else datetime.now(timezone.utc).isoformat()
    record = {"schema_version": 1, "document_id": identity, "stage": stage, "reason": reason,
              "recorded_at_utc": timestamp, "semantic_retry": False,
              "declaration_scope": "Operator-declared terminal failure; bound artifacts, not new semantic scoring or independently established cause.",
              "input_file_sha256": {path.relative_to(folder).as_posix(): rep._sha(path)
                  for path in sorted(folder.rglob("*")) if path.is_file() and path != target}}
    rep._write_immutable({target: rep._json_bytes(record)})
    validate_terminal(root, folder, identity)
    return {"terminal_record": str(target), "comparative_outcomes": None}


def _verify_file_hashes(folder, hashes):
    for name, expected in hashes.items():
        if rep._sha(paired._relative(folder, name)) != expected:
            raise ValueError("Source artifact changed: " + name)


def _load_completed_record(root, spec, doc, results, folder):
    original = rep._read_json(folder / "input/original_report.json")
    record = {"folder": folder, "spec": spec, "original": original,
              "candidate": rep._read_json(folder / "shared/candidate.raw.json"),
              "baseline": rep._read_json(folder / "evaluation/baseline.raw.json"),
              "guard": rep._read_json(folder / "shared/guard.patch.raw.json"),
              "post": rep._read_json(folder / "evaluation/paired_post.raw.json"),
              "normalization_audit": rep._read_json(folder / "shared/normalization.audit.json"),
              "arms": {}, "round0": rep._read_json(folder / "shared/reviews/round0/review.raw.json"),
              "round1": None}
    round1 = folder / "shared/reviews/round1/review.raw.json"
    if round1.exists():
        record["round1"] = rep._read_json(round1)
    for arm, data in doc["arms"].items():
        af = folder / "arms" / arm
        gate = rep._read_json(af / "gated/gate.decisions.json")
        revised = rep._read_json(af / "revised_report.json")
        if not _exact(gate, results[arm]):
            raise ValueError("Stored gate differs from replay")
        if any(not _exact(revised[key], gate["final_applied"][key]) for key in ("text", "sources")):
            raise ValueError("Scored report differs from actual final gate output")
        if (af / "repair.raw.json").read_bytes() != (af / "gated/repair.effective.json").read_bytes():
            raise ValueError("Scored repair differs from effective repair")
        record["arms"][data["variant"]] = {"arm": arm, "folder": af, "gate": gate, "revised": revised,
            "preview": rep._read_json(af / "preview/gate.preview.json"),
            "post": rep._read_json(af / "post.raw.json"), "linkage": rep._read_json(af / "gate_linkage.json")}
    record["costs"] = _review_costs(record)
    return record


def _review_costs(record):
    initial = _count(record["round0"]["execution"]["web_calls"], "initial full review calls")
    extra = _count(record["round1"]["execution"]["web_calls"], "additional shared review calls") if record["round1"] else 0
    if initial > 3 or extra > 3:
        raise ValueError("Frozen full review budget exceeded")
    per_variant = {}
    for variant, arm in record["arms"].items():
        count = len(arm["gate"]["review_history"])
        if count not in {1, 2}:
            raise ValueError("Expected one or two actual full reviews")
        participated = count == 2
        if participated and record["round1"] is None:
            raise ValueError("Recovery review history has no raw round1 response")
        link = arm["linkage"]
        if (link["initial_final_review_web_calls"] != initial
                or link["additional_rereview_participated"] is not participated
                or link["additional_joint_rereview_web_calls"] != extra):
            raise ValueError("Full review cost linkage mismatch")
        per_variant[variant] = {"review_count": count, "additional_review_participated": participated,
                               "additional_shared_web_calls": extra if participated else 0}
    if record["round1"] is not None and not any(row["additional_review_participated"] for row in per_variant.values()):
        raise ValueError("Unused extra full review must not be counted as recovery")
    return {"initial_shared_web_calls": initial, "additional_shared_web_calls": extra,
            "shared_total_web_calls": initial + extra, "per_variant": per_variant}


def load_trial(root, output, *, allow_incomplete=False):
    import uncertainty_trial_ops as ops
    root, output = Path(root).resolve(), Path(output).resolve()
    protocol, runtime = ops.runtime_check(root)
    if output.is_relative_to(root):
        raise ValueError("Publication output must be outside trial root")
    diagnostic = rep._read_json(output / "diagnostic/summary.json")
    rows = {row["id"]: row for row in diagnostic.get("documents", [])}
    _, prereg_hashes = paired.validate_preregistration(root)
    preflights, terminals, records, texts = {}, {}, {}, {}
    for spec in protocol["documents"]:
        identity = spec["id"]
        if identity not in rows:
            raise ValueError("Diagnostic output omitted planned document")
        folder = paired._relative(root, spec["path"])
        original = rep._read_json(folder / "input/original_report.json")
        rep.validate_original(original)
        if spec["cohort"] == "new" and not (folder / "input/deep_research_completed.png").is_file():
            raise ValueError("Fresh Deep Research screenshot required")
        derived, _ = paired._document_record(root, protocol, spec, prereg_hashes)
        if not _exact(derived, rows[identity]):
            raise ValueError("Diagnostic summary differs from current source records: " + identity)
        complete = derived["status"] == "complete"
        if complete and (folder / "terminal_failure.json").exists():
            raise ValueError("Previously declared terminal document cannot later become a completed diagnostic: " + identity)
        if not complete:
            if not allow_incomplete:
                raise ValueError("All four diagnostic pairs required; explicit --allow-incomplete needs terminal records")
            terminals[identity] = validate_terminal(root, folder, identity)
        pfpath = folder / "shared/pipeline.preflight.json"
        if pfpath.exists():
            preflight = rep._read_json(pfpath)
            if preflight.get("normalization", {}).get("status") == "failed":
                if complete:
                    raise ValueError("Failed normalization cannot have a complete pair")
                try:
                    ops._normalization_outputs(folder)
                except (ValueError, TypeError, KeyError) as error:
                    if str(error) != preflight["normalization"]["error"]:
                        raise ValueError("Normalization failure is not reproducible") from error
                    raw_records, _, raw_hashes = ops._raw_inputs(folder)
                    if not _exact(raw_hashes, preflight["input_file_sha256"]) or not _exact(ops._validation_status(raw_records), preflight["raw_v6"]):
                        raise ValueError("Failed normalization input/raw validation changed")
                else:
                    raise ValueError("Recorded failed normalization now succeeds")
            else:
                outputs, *_ = ops._normalization_outputs(folder)
                for path, expected in outputs.items():
                    if not path.is_file() or path.read_bytes() != expected:
                        raise ValueError("Normalization/preflight replay differs: " + str(path))
            preflights[identity] = preflight
        if complete:
            results, verified_folder = ops.verify_pair(root, identity)
            if folder != verified_folder:
                raise ValueError("Operator returned different document folder")
            record = _load_completed_record(root, spec, derived, results, folder)
            records[identity] = record
            texts[identity] = {"original": original["text"], **{variant: arm["revised"]["text"] for variant, arm in record["arms"].items()}}
        else:
            records[identity] = {"folder": folder, "spec": spec, "original": original, "terminal": terminals[identity]}
    summary = compose_summary(protocol, diagnostic, preflights, terminals)
    review = rep._read_json(root / "review.json")
    _validate_cases(review, texts)
    return root, output, protocol, runtime, diagnostic, summary, records, review


def _pre(value):
    return "<pre>" + esc(rep._json_bytes(value).decode("utf-8")) + "</pre>"


def _link(path, output, label):
    return "<a href='" + esc(rep._artifact_link(Path(path), output)) + "'>" + esc(label) + "</a>"


def _comparison_tables(summary, cohort):
    docs = [doc for doc in summary["documents"] if doc["cohort"] == cohort]
    main_rows, diagnostic_rows, safety_rows = [], [], []
    for doc in docs:
        for variant, label in (("current", "現行：raw＋v6"), ("improved", "改善：整合化＋v7")):
            arm = doc["main"]["arms"][variant]
            metric = arm["metrics"]
            main_rows.append([doc["id"], label, arm["status"], metric["changed_locations"] if metric else "—",
                              metric["patch_reviews"]["regression"] if metric else "—",
                              metric["patch_reviews"]["unnecessary_change"] if metric else "—"])
        diag = doc["diagnostic"]
        if diag["paired"] is None:
            diagnostic_rows.append([doc["id"], "診断比較未完了", "—", "—", "—", "—", "—"])
            continue
        primary = diag["paired"]["primary"]
        for variant, label in VARIANTS:
            count = primary[variant]
            metric = next(arm["metrics"] for arm in diag["arms"].values() if arm["variant"] == variant)
            diagnostic_rows.append([doc["id"], label, primary["denominator"], count["detected"], count["resolved"],
                                    count["after_status"]["partially_resolved"], primary["denominator"] - count["resolved"]])
            safety_rows.append([doc["id"], label, metric["changed_locations"], metric["patch_reviews"]["acceptable"],
                                metric["patch_reviews"]["regression"], metric["patch_reviews"]["unnecessary_change"],
                                metric["new_problems"]["confirmed_problem"], metric["false_findings"]["false_alarm"]])
    return ("<section id='" + cohort + "'><h2>" + ("既知の回帰テスト" if cohort == "regression" else "新規Deep Research") + "</h2>"
        "<h3>主比較：rawを入力する現行の全工程と、整合化を含む改善工程</h3>"
        + _table(["文書", "工程", "状態", "最終採用", "改悪", "不要変更"], main_rows)
        + "<p>—は未観測です。入力失敗は原文への復帰でも0件の改悪でもありません。主比較の問題解消の集計は両工程が完了したペアに限ります。</p>"
        + "<details><summary>主比較の共通問題集計・工程完了数</summary>" + _pre(summary["cohorts"][cohort]["main"]) + "</details>"
        + "<h3>診断比較：両方式に同じ整合化済み台帳を渡した場合</h3>"
        + _table(["文書", "方式", "共通問題", "発見", "完全解消", "部分解消", "残存"], diagnostic_rows)
        + _table(["文書", "方式", "最終採用", "acceptable", "改悪", "不要変更", "新問題", "誤指摘"], safety_rows)
        + "<p>この診断対照は、失敗したraw現行工程の結果を代替しません。採用した修正数と、完全に解消した問題数は別です。部分解消は残存に含み、事後発見問題は主母数へ加算しません。無変更時の新問題0は安全な訂正の実証ではありません。</p></section>")


def _process_html(records):
    rows, details = [], []
    for identity, record in records.items():
        if "terminal" in record:
            details.append("<h3>" + esc(identity) + "：診断未完了</h3>" + _pre(record["terminal"]))
            continue
        for variant, label in VARIANTS:
            arm = record["arms"][variant]
            summary = arm["gate"]["summary"]
            costs = record["costs"]["per_variant"][variant]
            rows.append([identity, label, arm["preview"]["summary"]["accepted_patch_count"],
                         summary["accepted_patch_count"], summary.get("semantic_recovery_rounds", 0),
                         summary.get("recovery_kind", "v6 failure-only"), costs["review_count"],
                         costs["additional_review_participated"]])
        details.append("<details><summary>" + esc(identity) + "：全費用・台帳変更・全文判定と除外</summary>"
                       + _pre({"shared_writer": record["candidate"]["execution"]["web_calls"],
                               "patch_guard": record["guard"]["execution"]["web_calls"],
                               "baseline": record["baseline"]["execution"]["web_calls"],
                               "post": record["post"]["execution"]["web_calls"], "full_reviews": record["costs"]})
                       + _pre(record["normalization_audit"]))
        for variant, label in VARIANTS:
            gate = record["arms"][variant]["gate"]
            details.append("<h4>" + esc(label) + "</h4>" + _pre({key: gate.get(key) for key in
                ("summary", "review_history", "exclusion_trail", "mechanical_diagnostics")}))
        details.append("</details>")
    return ("<section id='process'><h2>保留・採用・追加確認</h2>"
            + _table(["文書", "診断方式", "初期候補", "最終採用", "除外後確認の準備回数", "回復理由", "実際の全文判定数", "追加確認へ参加"], rows)
            + "<p>共同の追加確認呼出は全体で1回分として数え、両方式が参加しても二重計上しません。確認不能として保留した修正は、誤りが確定した修正とは数えません。回帰では過去のwriter・baselineやguardを再利用した場合があり、実行記録を参照してください。</p>"
            + "".join(details) + "</section>")


def _candidates_html(records):
    parts = ["<section id='candidates'><h2>全修正候補と保留理由（診断比較）</h2>"]
    for identity, record in records.items():
        if "terminal" in record:
            continue
        for patch in record["candidate"]["patches"]:
            parts.append("<details><summary>" + esc(identity + " / " + patch["patch_id"]) + "</summary>"
                         + "<strong>修正前</strong><pre>" + esc(patch["before"]) + "</pre><strong>提案</strong><pre>" + esc(patch["after"]) + "</pre>"
                         + "<p>依存する修正ID：" + esc(", ".join(patch.get("depends_on_patch_ids", [])) or "なし") + "</p>")
            for variant, label in VARIANTS:
                decision = next(row for row in record["arms"][variant]["gate"]["decisions"] if row["patch_id"] == patch["patch_id"])
                parts.append("<h4>" + esc(label) + "</h4>" + _pre(decision))
            parts.append("</details>")
    return "".join(parts) + "</section>"


def _cases_html(review):
    parts = ["<section id='cases'><h2>実際の最終本文の比較例</h2><p>中央列は常に同じ整合化済み台帳を用いたv6の診断結果です。raw現行が失敗した文書の納品結果ではありません。</p>"]
    for case in review["cases"]:
        parts.append("<article class='card'><h3>" + esc(case["document_id"] + "：" + case["title"]) + "</h3><div class='triple'>")
        for variant, label in (("original", "原文"), *VARIANTS):
            parts.append("<div><strong>" + esc(label) + "</strong><div class='quote'>" + esc(case[variant]) + "</div></div>")
        parts.append("</div>" + paragraphs(case["assessment"]) + "<ul>")
        parts.extend("<li>" + source_link(source["url"], source["title"]) + "</li>" for source in case["sources"])
        parts.append("</ul><p>記録ID：" + esc(", ".join(case["record_ids"])) + "</p></article>")
    return "".join(parts) + "</section>"


def _report_html(root, output, protocol, summary, records, review):
    parts = ["<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>" + esc(protocol["report_title"]) + "</title><style>" + STYLE + "</style></head><body><main>",
        "<div class='eyebrow'>" + esc(protocol.get("report_date", "")) + " · AI暫定評価 · 主比較と診断比較を分離</div>",
        "<h1>" + esc(protocol["report_heading"]) + "</h1>",
        "<div class='notice'><strong>主比較はraw＋v6と整合化＋v7。診断比較は両方とも整合化済み台帳です。</strong><p>失敗したraw現行工程を、診断対照の成功・採用数・安全性で置き換えません。未観測はnull／—のままです。</p></div>",
        "<nav><a href='#new'>新規</a><a href='#regression'>回帰</a><a href='#process'>保留と確認</a><a href='#candidates'>全候補</a><a href='#cases'>具体例</a><a href='#records'>原資料</a></nav>",
        "<h2>" + esc(review["headline"]) + "</h2>" + paragraphs(review["verdict"]),
        "<h2>固定した設計</h2>" + paragraphs(review["design"])]
    for cohort in ("new", "regression"):
        parts.extend([_comparison_tables(summary, cohort), paragraphs(review[cohort + "_notes"])])
    parts.extend([_process_html(records), _candidates_html(records), _cases_html(review),
                  paragraphs(review.get("guard_notes", []) if isinstance(review.get("guard_notes", []), list) else [review["guard_notes"]]),
                  "<h2>次の判断</h2>" + paragraphs(review["recommendations"]),
                  "<h2>限界</h2>" + paragraphs(review["limitations"]) + paragraphs(summary["interpretation"]),
                  "<section id='records'><h2>全資料・入力・出典・監査</h2><p>"])
    for name in ("summary.json", "index.html", "publication_manifest.json", "diagnostic/index.html", "diagnostic/summary.json"):
        parts.append(_link(output / name, output, name) + " · ")
    parts.append("</p><details><summary>凍結指示・全raw・ハッシュ記録</summary><ul>")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "publication_manifest.json" and path.suffix.lower() in {".json", ".txt", ".html", ".png", ".py", ".md"}:
            parts.append("<li>" + _link(path, output, path.relative_to(root).as_posix()) + "</li>")
    parts.append("</ul></details>")
    for identity, record in records.items():
        parts.append("<details><summary>" + esc(identity) + "：原文出典・生成画面</summary><ul>")
        for source in record["original"]["sources"]:
            parts.append("<li>" + source_link(source["url"], source.get("title", source["url"])) + "</li>")
        if record["original"].get("chat_url"):
            parts.append("<li>" + source_link(record["original"]["chat_url"], "Deep Research元チャット") + "</li>")
        parts.append("</ul>")
        shot = record["folder"] / "input/deep_research_completed.png"
        if shot.exists():
            parts.append("<img loading='lazy' src='" + esc(rep._artifact_link(shot, output)) + "' alt='" + esc(identity) + " 原文生成画面'>")
        parts.append("</details>")
    return ("\n".join(parts) + "</section></main></body></html>").encode("utf-8")


def _index_html(output, protocol, summary, records):
    parts = ["<!doctype html><html lang='ja'><head><meta charset='utf-8'><title>全記録</title><style>" + STYLE + "</style></head><body><main>",
             "<h1>全予定文書・主比較と診断比較</h1><p>" + _link(output / "report.html", output, "比較レポート") + "</p>",
             _pre(summary)]
    for identity, record in records.items():
        parts.append("<h2>" + esc(identity) + "</h2><details><summary>原文</summary>" + _pre(record["original"]) + "</details>")
        if "terminal" in record:
            parts.append(_pre(record["terminal"]))
            continue
        for variant, label in VARIANTS:
            arm = record["arms"][variant]
            delta = "\n".join(difflib.unified_diff(record["original"]["text"].splitlines(), arm["revised"]["text"].splitlines(), fromfile="original", tofile=variant, lineterm=""))
            parts.append("<details><summary>" + esc(label) + "：最終全文・差分・全採点</summary>" + _pre(arm["revised"])
                         + "<pre>" + esc(delta or "変更なし") + "</pre>" + _pre(arm["post"]) + "</details>")
    return ("\n".join(parts) + "</main></body></html>").encode("utf-8")


def publish(root, output, *, allow_incomplete=False):
    root, output, protocol, runtime, diagnostic, summary, records, review = load_trial(root, output, allow_incomplete=allow_incomplete)
    manifest = {"schema_version": 1, "experiment_id": protocol["id"], "preregistration_sha256": rep._sha(root / "preregistration.json"),
                "runtime_sha256": runtime, "diagnostic_summary_sha256": rep._sha(output / "diagnostic/summary.json"),
                "review_sha256": rep._sha(root / "review.json"), "summary_sha256": sha256_json(summary),
                "allow_incomplete": allow_incomplete, "all_diagnostic_pairs_complete": summary["all_diagnostic_pairs_complete"],
                "all_main_pairs_complete": summary["all_main_pairs_complete"],
                "publication_accounts_for_all_planned_documents": True,
                "documents": [{"id": identity, "path": record["spec"]["path"],
                    "source_file_sha256": {path.relative_to(record["folder"]).as_posix(): rep._sha(path)
                        for path in sorted(record["folder"].rglob("*")) if path.is_file()}}
                    for identity, record in records.items()]}
    for record in manifest["documents"]:
        _verify_file_hashes(paired._relative(root, record["path"]), record["source_file_sha256"])
    outputs = {output / "summary.json": rep._json_bytes(summary),
               output / "report.html": _report_html(root, output, protocol, summary, records, review),
               output / "index.html": _index_html(output, protocol, summary, records),
               output / "review.json": (root / "review.json").read_bytes(),
               output / "publication_manifest.json": rep._json_bytes(manifest),
               root / "publication_manifest.json": rep._json_bytes(manifest)}
    rep._write_immutable(outputs)
    return {"report": str(output / "report.html"), "summary": str(output / "summary.json"),
            "all_diagnostic_pairs_complete": summary["all_diagnostic_pairs_complete"],
            "all_main_pairs_complete": summary["all_main_pairs_complete"],
            "publication_accounts_for_all_planned_documents": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--declare-terminal", metavar="DOCUMENT_ID")
    parser.add_argument("--stage")
    parser.add_argument("--reason")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Require a bound explicit terminal_failure.json for every incomplete diagnostic; retain null outcomes")
    args = parser.parse_args(argv)
    try:
        if args.declare_terminal:
            if not args.stage or not args.reason:
                parser.error("--declare-terminal requires --stage and --reason")
            result = declare_terminal(args.root, args.declare_terminal, args.stage, args.reason)
        else:
            if args.output is None:
                parser.error("--output is required for publication")
            result = publish(args.root, args.output, allow_incomplete=args.allow_incomplete)
        print(rep._json_bytes(result).decode("utf-8"), end="")
    except (ValueError, OSError, TypeError, KeyError, UnicodeError) as error:
        parser.exit(2, f"Publication not generated: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
