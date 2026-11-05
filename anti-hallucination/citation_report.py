"""Read-only completed-input review for the frozen v5/v6 citation trial.

No evidence is retrieved, outcomes invented, or trial inputs changed. All four
paired outcomes and exact editorial quotes are required before the immutable
report.html and review.json outputs can be written.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import paired_trial as paired
import replication as rep
from omission_report import STYLE, _count, _table, esc, paragraphs, source_link


VARIANTS = (("current", "現行 v5"), ("improved", "改善 v6"))
COHORTS = {"R1": "regression", "R2": "regression", "N1": "new", "N2": "new"}
REVIEW_LISTS = ("verdict", "design", "regression_notes", "new_notes", "recommendations", "limitations")
FILTER_SCRIPT = """
const controls=['candidate-search','cohort-filter','status-filter'].map(id=>document.getElementById(id));
const rows=[...document.querySelectorAll('.candidate-row')];
function filterCandidates(){let count=0;const [q,c,s]=controls.map(x=>x.value.toLocaleLowerCase());
for(const row of rows){const shown=(!q||row.textContent.toLocaleLowerCase().includes(q))&&(!c||row.dataset.cohort===c)&&(!s||row.dataset.status===s);row.hidden=!shown;if(shown)count++;}
document.getElementById('filter-count').textContent=count+' / '+rows.length+' 候補を表示';}
for(const control of controls)control.addEventListener('input',filterCandidates);filterCandidates();
"""


def _require_complete_summary(protocol, summary):
    """Missing, invalid, and unfinished records are never interpreted as zero."""
    if summary.get("experiment_id") != protocol["id"]:
        raise ValueError("Summary experiment does not match protocol")
    specs = {row["id"]: row for row in protocol["documents"]}
    if len(protocol["documents"]) != 4 or set(specs) != set(COHORTS):
        raise ValueError("All four planned R1/R2/N1/N2 documents are required")
    rows = summary.get("documents", [])
    if len(rows) != 4 or {row["id"] for row in rows} != set(COHORTS):
        raise ValueError("Complete paired summary required for all four planned documents")
    for row in rows:
        identity = row["id"]
        if row.get("cohort") != COHORTS[identity] or specs[identity]["cohort"] != COHORTS[identity]:
            raise ValueError("Historical and new cohorts must remain separate: " + identity)
        if (row.get("status") != "complete" or row.get("missing") or row.get("validation_errors") or
                not row.get("paired")):
            raise ValueError("Complete paired summary required: " + identity)
        mapping = paired._arm_mapping(protocol, specs[identity])
        if set(row.get("arms", {})) != set(mapping):
            raise ValueError("Complete arm mapping required: " + identity)
        primary = row["paired"]["primary"]
        denominator = _count(primary.get("denominator"), identity + "/denominator")
        eligible = primary.get("eligible_issue_ids")
        if not isinstance(eligible, list) or len(eligible) != len(set(eligible)) or len(eligible) != denominator:
            raise ValueError("Common denominator mismatch: " + identity)
        for arm, data in row["arms"].items():
            if (data.get("status") != "complete" or data.get("missing") or data.get("validation_errors") or
                    not data.get("metrics") or data.get("variant") != mapping[arm]):
                raise ValueError(f"Complete arm required: {identity}/{arm}")
            counts = primary[data["variant"]]
            for key in ("detected", "resolved", "resolved_by_removal"):
                if _count(counts.get(key), identity + "/" + key) > denominator:
                    raise ValueError("Count exceeds common denominator: " + identity)
            after = counts.get("after_status", {})
            total = sum(_count(after.get(key), identity + "/after_status/" + key)
                        for key in rep.AFTER_STATUSES - {"not_an_original_problem"})
            if total != denominator or after["resolved"] != counts["resolved"]:
                raise ValueError("Common issue status counts disagree: " + identity)
            metrics = data["metrics"]
            for group, choices in (("patch_reviews", rep.PATCH_STATUSES), ("new_problems", rep.PROBLEM_STATUSES),
                                   ("controls", rep.CONTROL_STATUSES),
                                   ("posthoc_original_findings", rep.PROBLEM_STATUSES)):
                for key in choices:
                    _count(metrics.get(group, {}).get(key), identity + "/" + group + "/" + key)
            for key in ("false_alarm", "unverifiable"):
                _count(metrics.get("false_findings", {}).get(key), identity + "/false_findings/" + key)
            changed = _count(metrics.get("changed_locations"), identity + "/changed_locations")
            if sum(metrics["patch_reviews"].values()) != changed:
                raise ValueError("Delivered patch counts disagree: " + identity)
    return {row["id"]: row for row in rows}


def _validate_cases(review, texts):
    rep._string(review.get("headline"), "review.headline", nonempty=True)
    for field in REVIEW_LISTS:
        rep._strings(review.get(field), "review." + field)
    for case in rep._array(review.get("cases"), "review.cases"):
        identity = case["document_id"]
        if identity not in texts:
            raise ValueError("Case must refer to a completed document: " + identity)
        for field in ("title", "category"):
            rep._string(case.get(field), "case." + field, nonempty=True)
        for field in ("assessment", "record_ids"):
            rep._strings(case.get(field), "case." + field)
        for variant in ("original", "current", "improved"):
            quote = case.get(variant)
            if not isinstance(quote, str) or not quote.strip() or quote not in texts[identity][variant]:
                raise ValueError(f"Case {case['title']}: {variant} quote is not exact")
        for source in rep._array(case.get("sources"), "case.sources"):
            source_link(source["url"], source["title"])
    notes = review.get("guard_notes", [])
    rep._strings([notes] if isinstance(notes, str) else notes, "review.guard_notes")


def _ids(decisions):
    return {row["patch_id"] for row in decisions if row["accepted"]}


def _review_costs(record):
    """Count a shared initial review once; distinguish a real second review."""
    initial = record["round0"]["execution"]["web_calls"]
    _count(initial, "initial final review Web calls")
    if initial > 3:
        raise ValueError("Initial full review exceeds frozen budget")
    history = record["arms"]["improved"]["gate"]["review_history"]
    if not 1 <= len(history) <= 2:
        raise ValueError("Final v6 gate requires one or two recorded full reviews")
    extra_record = record["round1"]
    real_retries = len(history) - 1
    if bool(extra_record is not None) != bool(real_retries):
        raise ValueError("Second full-review artifact/history mismatch")
    extra = 0 if extra_record is None else _count(extra_record["execution"]["web_calls"], "v6 additional Web calls")
    if extra > 3:
        raise ValueError("Additional v6 full review exceeds frozen budget")
    for variant, arm in record["arms"].items():
        linkage = arm["linkage"]
        if linkage["initial_final_review_web_calls"] != initial:
            raise ValueError("Initial review cost linkage mismatch")
        if linkage["additional_v6_rereview_web_calls"] != (extra if variant == "improved" else 0):
            raise ValueError("Additional review cost linkage mismatch")
    return {"initial_web_calls": initial, "additional_web_calls": extra,
            "actual_rereviews": real_retries,
            "removal_rounds": _count(record["arms"]["improved"]["gate"]["summary"]["semantic_recovery_rounds"], "removal rounds")}


def load_completed(root, output):
    if not (output / "summary.json").is_file():
        raise ValueError("Complete paired summary.json is required before report generation")
    # Import only after the cheap completeness guard; this module can be tested
    # while orchestration is still being implemented, without touching a trial.
    import citation_trial_ops as ops
    protocol, hashes = paired.validate_preregistration(root)
    if protocol.get("gate_versions") != {"current": "v5", "improved": "v6"}:
        raise ValueError("Expected frozen current v5 / improved v6 assignment")
    summary = rep._read_json(output / "summary.json")
    documents = _require_complete_summary(protocol, summary)
    if not (output / "index.html").is_file():
        raise ValueError("Completed paired index.html is required")
    manifest_path = root / "operator_manifest.all.json"
    manifest = rep._read_json(manifest_path)
    if manifest.get("cohort") != "all" or manifest.get("whole_trial_complete") is not True or manifest.get("planned_unrun_document_ids"):
        raise ValueError("Complete all-document operator manifest required")
    if (output / "operator_manifest.json").read_bytes() != manifest_path.read_bytes():
        raise ValueError("Output operator manifest differs from frozen all-document manifest")
    if manifest["preregistration_sha256"] != rep._sha(root / "preregistration.json"):
        raise ValueError("Operator manifest preregistration mismatch")
    manifests = {row["id"]: row for row in manifest["documents"]}
    if len(manifest["documents"]) != 4 or set(manifests) != set(documents):
        raise ValueError("Operator manifest must cover all four completed documents")
    records, texts = {}, {}
    for spec in protocol["documents"]:
        identity = spec["id"]
        # Replays all source identity, frozen metadata, selection/hash, recovery,
        # and effective/scored-output checks without writing any artifacts.
        results, folder = ops.verify_pair(root, identity)
        derived, _ = paired._document_record(root, protocol, spec, hashes)
        if derived != documents[identity]:
            raise ValueError("Saved paired summary differs from verified artifacts: " + identity)
        manifest_row = manifests[identity]
        for field in ("candidate_identity", "ledger_identity", "patch_judgment_identity", "findings_identity",
                      "gate_replay_verified", "effective_equals_scored", "joint_audit_verified"):
            if manifest_row.get(field) is not True:
                raise ValueError(f"Operator integrity check incomplete: {identity}/{field}")
        for name, expected in manifest_row["input_and_output_sha256"].items():
            if rep._sha(paired._relative(folder, name)) != expected:
                raise ValueError(f"Operator artifact changed: {identity}/{name}")
        original = rep._read_json(folder / "input/original_report.json")
        record = {"folder": folder, "spec": spec, "original": original, "manifest": manifest_row,
                  "candidate": rep._read_json(folder / "shared/candidate.raw.json"),
                  "guard": rep._read_json(folder / "shared/guard.patch.raw.json"),
                  "baseline": rep._read_json(folder / "evaluation/baseline.raw.json"), "arms": {},
                  "joint_post": rep._read_json(folder / "evaluation/paired_post.raw.json"),
                  "detection_consistency": rep._read_json(folder / "evaluation/detection_consistency.json"),
                  "round0": rep._read_json(folder / "shared/reviews/round0/review.raw.json"),
                  "round1": None}
        round1 = folder / "shared/reviews/round1/review.raw.json"
        if round1.is_file():
            record["round1"] = rep._read_json(round1)
        texts[identity] = {"original": original["text"]}
        for arm, data in documents[identity]["arms"].items():
            arm_folder = folder / "arms" / arm
            preview = rep._read_json(arm_folder / "preview/gate.preview.json")
            gate = rep._read_json(arm_folder / "gated/gate.decisions.json")
            revised = rep._read_json(arm_folder / "revised_report.json")
            if gate != results[arm] or gate["summary"] != manifest_row["gate_summary"][arm]:
                raise ValueError("Gate result differs from verified replay or manifest")
            if any(revised[field] != gate["final_applied"][field] for field in ("text", "sources")):
                raise ValueError("Scored revised report differs from final gate output")
            accepted, eligible = _ids(gate["decisions"]), _ids(preview["decisions"])
            if (not accepted.issubset(eligible) or len(accepted) != gate["summary"]["accepted_patch_count"] or
                    len(eligible) != preview["summary"]["accepted_patch_count"]):
                raise ValueError("Final and preview selection counts disagree")
            record["arms"][data["variant"]] = {"arm": arm, "folder": arm_folder,
                "preview": preview, "gate": gate, "revised": revised,
                "post": rep._read_json(arm_folder / "post.raw.json"),
                "linkage": rep._read_json(arm_folder / "gate_linkage.json"),
                "eligible": eligible, "accepted": accepted}
            texts[identity][data["variant"]] = revised["text"]
        if spec["cohort"] == "new":
            screenshot = folder / "input/deep_research_completed.png"
            if not screenshot.is_file():
                raise ValueError("Actual Deep Research completion screenshot missing: " + identity)
            record["screenshot"] = screenshot
        record["costs"] = _review_costs(record)
        if record["detection_consistency"] != manifest_row["detection_consistency"]:
            raise ValueError("Detection consistency differs from operator manifest")
        expected_web = {"shared_writer": record["candidate"]["execution"]["web_calls"],
            "shared_patch_guard": record["guard"]["execution"]["web_calls"],
            "joint_final_review": record["costs"]["initial_web_calls"],
            "additional_v6_rereview": record["costs"]["additional_web_calls"],
            "baseline": record["baseline"]["execution"]["web_calls"],
            "joint_post": record["joint_post"]["execution"]["web_calls"]}
        if manifest_row["web_calls"] != expected_web:
            raise ValueError("Execution costs differ from operator manifest")
        records[identity] = record
    review = rep._read_json(root / "review.json")
    _validate_cases(review, texts)
    return protocol, summary, documents, records, review


def _cohort_html(cohort, protocol, documents, records):
    title = "既知2本：同一の過去候補による回帰比較" if cohort == "regression" else "新規2本：実Deep Researchの最終結果"
    parts = [f"<section id='{cohort}'><h2>{title}</h2>"]
    if cohort == "regression":
        parts.append("<div class='notice'>既知の原文・候補・台帳・パッチ判断・原文baselineを再利用した開発回帰です。新規文書の検証と混合せず、独立した一般化検証とは扱いません。</div>")
    rows, safety = [], []
    for spec in protocol["documents"]:
        if spec["cohort"] != cohort:
            continue
        identity, primary = spec["id"], documents[spec["id"]]["paired"]["primary"]
        for variant, label in VARIANTS:
            metrics = next(arm["metrics"] for arm in documents[identity]["arms"].values() if arm["variant"] == variant)
            counts = primary[variant]
            rows.append([identity, label, primary["denominator"], counts["detected"], counts["resolved"],
                counts["after_status"]["partially_resolved"], primary["denominator"] - counts["resolved"],
                counts["after_status"]["unverifiable"]])
            patches = metrics["patch_reviews"]
            safety.append([identity, label, metrics["changed_locations"], patches["acceptable"],
                patches["unnecessary_change"], patches["regression"], metrics["new_problems"]["confirmed_problem"],
                patches["unverifiable"], metrics["controls"]["omitted"], metrics["controls"]["corrupted"], metrics["controls"]["unverifiable"],
                metrics["false_findings"]["false_alarm"]])
    parts.append(_table(["文書", "ゲート", "共通baseline母数", "発見", "完全解消", "部分解消", "残存（部分を含む）", "残存の検証不能"], rows))
    parts.append("<p class='small'>完全解消はresolvedだけ。部分解消・未解消・検証不能は残存に含め、部分解消を別列でも示します。共有findingsの発見差は評価者の不整合であり、ゲートの発見改善ではありません。</p>")
    parts.append(_table(["文書", "ゲート", "最終採用", "妥当と評価された適用変更 (acceptable)", "不要", "改悪 regression", "確認済み新問題", "変更の検証不能", "対照脱落", "対照改悪", "対照の検証不能", "誤指摘"], safety))
    parts.append("<p class='small'>acceptableは採用変更への暫定評価です。それだけでは原文問題を解消した有用な訂正と同義ではありません。有用性は上表の共通問題解消と、下記の主母数外の確認を合わせて読みます。採用変更と確認済み問題IDの対応を自動で推定せず、対応が具体例で確認されないacceptableを有用訂正数へ加えません。改悪・新問題・対照・不要変更は重なり得るため合算しません。</p>")
    secondary = []
    for spec in protocol["documents"]:
        if spec["cohort"] != cohort:
            continue
        identity, record = spec["id"], records[spec["id"]]
        for variant, label in VARIANTS:
            post = record["arms"][variant]["post"]
            uncertain = [row for row in post["issues"] if row["baseline_status"] == "unresolved"]
            posthoc = [row for row in post["posthoc_original_findings"] if row["status"] == "confirmed_problem"]
            secondary.append([identity, label, len(uncertain),
                sum(row["after_status"] == "resolved" for row in uncertain),
                sum(row["after_status"] == "partially_resolved" for row in uncertain),
                sum(row["after_status"] == "not_an_original_problem" for row in uncertain),
                len(posthoc), sum(not row["still_present_after"] for row in posthoc),
                sum(row["still_present_after"] for row in posthoc),
                sum(row["status"] == "unresolved" for row in post["posthoc_original_findings"])])
    parts.append("<h3>主母数外：原文時点の未確定と事後発見</h3>" + _table(["文書", "ゲート", "baseline未確定", "未確定から解消", "未確定から部分", "原文問題ではない", "事後確認問題", "事後確認・残存なし", "事後確認・残存", "事後未確定"], secondary))
    parts.append("<p class='small'>事後の「残存なし」は保存されたstill_present_after=falseの件数で、主比較の完全解消数へ足しません。原文時点で未確定だった問題も、後付けで共通母数へ加えません。</p></section>")
    return "\n".join(parts)


def _candidate_rows(records):
    rows = []
    for identity, record in records.items():
        decisions = {variant: {row["patch_id"]: row for row in arm["gate"]["decisions"]}
                     for variant, arm in record["arms"].items()}
        for patch in record["candidate"]["patches"]:
            patch_id = patch["patch_id"]
            states = {variant: {"eligible": patch_id in arm["eligible"], "accepted": patch_id in arm["accepted"],
                                "reasons": decisions[variant][patch_id]["reasons"]}
                      for variant, arm in record["arms"].items()}
            trail = [row for row in record["arms"]["improved"]["gate"]["exclusion_trail"] if row["patch_id"] == patch_id]
            rows.append({"document_id": identity, "cohort": record["spec"]["cohort"], "patch_id": patch_id,
                "before": patch["before"], "after": patch["after"],
                "depends_on_patch_ids": patch.get("depends_on_patch_ids", []), "states": states, "trail": trail})
    return rows


def _candidates_html(rows):
    parts = ["<section id='candidates'><h2>全候補：完全な変更前後と除外理由</h2>",
        "<p>候補の変更後は提案テキストです。適格性は初回全文確認前、最終採用は確認後の納品本文への適用を表します。</p>",
        "<div class='filters'><label>語句検索<input type='search' id='candidate-search' placeholder='ID・引用・境界・除外理由'></label><label>資料<select id='cohort-filter'><option value=''>すべて</option><option value='regression'>既知回帰</option><option value='new'>新規</option></select></label><label>最終採用差<select id='status-filter'><option value=''>すべて</option><option value='changed'>差あり</option><option value='same'>差なし</option></select></label><span id='filter-count' role='status' aria-live='polite'></span></div>",
        "<div class='scroll'><table><thead><tr><th>文書 / 候補</th><th>状態</th><th>引用・理由・機械的境界</th></tr></thead><tbody>"]
    for row in rows:
        states = row["states"]
        changed = states["current"]["accepted"] != states["improved"]["accepted"]
        parts.append(f"<tr class='candidate-row' data-cohort='{esc(row['cohort'])}' data-status='{'changed' if changed else 'same'}'><td>{esc(row['document_id'])} / {esc(row['patch_id'])}</td><td>")
        for variant, label in VARIANTS:
            state = states[variant]
            parts.append(f"<p>{esc(label)}<br>適格 {'○' if state['eligible'] else '×'} / 最終採用 {'○' if state['accepted'] else '×'}</p>")
        parts.append("</td><td><details><summary>完全な候補と全理由を表示</summary><div class='double'>")
        for field, label in (("before", "変更前（完全一致）"), ("after", "候補の変更後（完全一致）")):
            parts.append(f"<div><strong>{label}</strong><div class='quote'>{esc(row[field])}</div></div>")
        parts.append("</div><p>依存ID：" + esc(", ".join(row["depends_on_patch_ids"]) or "なし") + "</p>")
        for variant, label in VARIANTS:
            parts.append(f"<strong>{esc(label)}の最終判定理由</strong>" + paragraphs(states[variant]["reasons"] or ["棄却理由なし"]))
        parts.append("<strong>v6除外経路・境界の座標と引用</strong><pre>" + esc(rep._json_bytes(row["trail"]).decode("utf-8")) + "</pre></details></td></tr>")
    parts.append("</tbody></table></div></section>")
    return "\n".join(parts)


def _process_html(documents, records):
    total_retries = sum(record["costs"]["actual_rereviews"] for record in records.values())
    parts = ["<section id='process'><h2>機械的除外・全文確認・最終採用</h2>"]
    if total_retries == 0:
        parts.append("<div class='notice'>実データでの追加全文再確認は0回です。今回の実験は局所回復分岐の実証にはなりません。合成テストによる分岐確認と、実文書で観測した機械的除外を区別します。</div>")
    else:
        parts.append(f"<p>実データで追加全文再確認を {total_retries} 回実施しました。下記の最終結果と追加費用を伴う、最大1回の局所除外・再確認です。</p>")
    rows = []
    for identity, record in records.items():
        for variant, label in VARIANTS:
            arm = record["arms"][variant]
            gate = arm["gate"]
            trail = gate.get("exclusion_trail", [])
            ids = lambda phase: ", ".join(sorted({row["patch_id"] for row in trail if row["phase"] == phase})) or "なし"
            rows.append([identity, label, len(record["candidate"]["patches"]), len(arm["eligible"]),
                ids("mechanical") if variant == "improved" else "対象外", ids("dependency") if variant == "improved" else "下記理由参照",
                len(arm["accepted"]), "なし" if gate["summary"]["final_review_passed"] else "全文留保",
                ", ".join(sorted(arm["accepted"])) or "なし"])
    parts.append(_table(["文書", "ゲート", "共有候補", "初回確認前の適格", "機械的除外ID", "依存留保ID", "最終採用", "全文留保", "最終採用ID"], rows))
    cost_rows = []
    for identity, record in records.items():
        cost = record["costs"]
        reuse = record["spec"]["cohort"] == "regression"
        cost_rows.append([identity, "過去処理を再利用" if reuse else "新規処理",
            record["candidate"]["execution"]["web_calls"], record["guard"]["execution"]["web_calls"],
            cost["initial_web_calls"], cost["removal_rounds"], cost["actual_rereviews"], cost["additional_web_calls"],
            record["baseline"]["execution"]["web_calls"], record["joint_post"]["execution"]["web_calls"]])
    parts.append("<h3>呼出実績：共有確認とv6追加確認を分離</h3>" + _table(["文書", "作成・guard・baseline", "共有作成", "共有patch guard", "共同初回全文確認", "v6局所除外回数", "実追加全文確認", "v6追加Web", "原文baseline", "共同事後評価"], cost_rows))
    parts.append("<p class='small'>数値は各workerの実行記録です。共同初回全文確認は1文書につき一度だけ集計し、両群へ二重計上しません。既知回帰の作成・guard・baselineは過去費用であり、今回の新規消費とは分離しています。v6の追加全文確認は別枠最大3 Web呼出で、同じ計算量の比較ではありません。局所除外回数と、実際に新しい全文確認を受けた回数は異なる場合があります。</p>")
    for identity, record in records.items():
        doc = documents[identity]
        primary = doc["paired"]["primary"]
        parts.append(f"<details><summary>{esc(identity)}：全確認履歴・除外・副次評価</summary>")
        extra = {"common_baseline_excluded": primary["excluded_issue_ids"],
            "exclusion_disagreements": primary["exclusion_disagreements"],
            "baseline_unresolved_separate_from_primary": doc["paired"]["baseline_unresolved"],
            "detection_consistency": record["detection_consistency"], "arms": {}}
        for variant, _ in VARIANTS:
            arm = record["arms"][variant]
            extra["arms"][variant] = {"initial_preview_selection": arm["preview"]["selection"],
                "final_summary": arm["gate"]["summary"], "final_decisions": arm["gate"]["decisions"],
                "review_history": arm["gate"].get("review_history", [record["round0"]]),
                "mechanical_diagnostics": arm["gate"].get("mechanical_diagnostics", []),
                "exclusion_trail": arm["gate"].get("exclusion_trail", []),
                "posthoc_original_findings_separate_from_primary": arm["post"]["posthoc_original_findings"],
                "controls": arm["post"]["controls"], "false_findings": arm["post"]["false_findings"]}
        parts.append("<p>原文の未確定問題と事後発見は主母数へ追加せず、以下に独立して掲載します。発見判定の差は共有findingsに対する評価者の不整合です。</p><pre>" + esc(rep._json_bytes(extra).decode("utf-8")) + "</pre></details>")
    parts.append("</section>")
    return "\n".join(parts)


def render_review(root: Path, output: Path):
    root, output = Path(root).resolve(), Path(output).resolve()
    protocol, summary, documents, records, review = load_completed(root, output)

    def artifact(path, label):
        if not path.is_file():
            raise ValueError("Linked artifact is missing: " + str(path))
        return f"<a href='{esc(rep._artifact_link(path, output))}'>{esc(label)}</a>"

    parts = ["<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{esc(protocol['report_title'])}</title><style>{STYLE}</style></head><body><main>",
        f"<div class='eyebrow'>{esc(protocol['report_date'])} · 既知2本＋新規2本 · 共有候補 v5 / v6 · AI暫定評価</div><h1>{esc(protocol['report_heading'])}</h1>",
        "<nav><a href='#new'>新規結果</a><a href='#regression'>既知回帰</a><a href='#process'>除外と再確認</a><a href='#candidates'>全候補</a><a href='#cases'>具体例</a><a href='#records'>原資料</a></nav>",
        f"<div class='lead'><h3>{esc(review['headline'])}</h3>{paragraphs(review['verdict'])}</div>",
        "<h2>固定した条件</h2>", paragraphs(review["design"])]
    for cohort in ("new", "regression"):
        parts.extend([_cohort_html(cohort, protocol, documents, records), paragraphs(review[cohort + "_notes"])])
    parts.extend([_process_html(documents, records), _candidates_html(_candidate_rows(records)),
        "<section id='cases'><h2>原文・現行・改善の具体例</h2><p class='small'>引用は原文と最終採点本文への完全一致を検査済みです。同一性の検査であり、意味や出典の正しさの証明ではありません。</p>"])
    for case in review["cases"]:
        parts.append(f"<article class='card'><span class='badge'>{esc(case['document_id'])} · {esc(case['category'])}</span><h3>{esc(case['title'])}</h3><p>記録ID：{esc(', '.join(case['record_ids']))}</p><div class='triple'>")
        for variant, label in (("original", "原文"), *VARIANTS):
            parts.append(f"<div><strong>{esc(label)}</strong><div class='quote'>{esc(case[variant])}</div></div>")
        parts.extend(["</div>", paragraphs(case["assessment"]), "<ul>"])
        parts.extend("<li>" + source_link(source["url"], source["title"]) + "</li>" for source in case["sources"])
        parts.append("</ul></article>")
    parts.append("</section>")
    notes = review.get("guard_notes", [])
    parts.extend([paragraphs([notes] if isinstance(notes, str) else notes), "<h2>次の判断</h2>",
        paragraphs(review["recommendations"]), "<h2>限界</h2>", paragraphs(review["limitations"]),
        "<section id='records'><h2>全資料・凍結・来歴</h2><p>", artifact(output / "index.html", "全問題ID・全文差分・完全な台帳"),
        " · ", artifact(output / "summary.json", "機械集計JSON"), " · ", artifact(output / "operator_manifest.json", "検証済み整合manifest"), "</p>"])
    # Every available raw, protocol, provenance, worker packet, freeze and code
    # snapshot remains directly inspectable; nothing is silently summarized away.
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".txt", ".py", ".png", ".md"})
    parts.append("<details><summary>全raw・指示・プロトコル・来歴・ハッシュ記録</summary><ul>")
    parts.extend("<li>" + artifact(path, path.relative_to(root).as_posix()) + "</li>" for path in paths)
    parts.append("</ul></details>")
    for identity, record in records.items():
        if record["original"].get("chat_url"):
            parts.append("<p>" + esc(identity) + "：" + source_link(record["original"]["chat_url"], "Deep Research元チャット") + "</p>")
        if "screenshot" in record:
            parts.append(f"<details><summary>{esc(identity)} 実Deep Research完成画面</summary><img loading='lazy' src='{esc(rep._artifact_link(record['screenshot'], output))}' alt='{esc(identity)} 実Deep Research完成画面'></details>")
    parts.append("</section><p class='small'>AIの評価は暫定です。ハッシュと構造の検証は入力・出力の同一性を示し、事実の正しさ・評価の独立性・完全な資料取得を保証しません。欠測は0として扱いません。</p>")
    parts.append(f"</main><script>{FILTER_SCRIPT}</script></body></html>")
    rep._write_immutable({output / "report.html": "\n".join(parts).encode("utf-8"),
                          output / "review.json": (root / "review.json").read_bytes()})
    return {"report": str(output / "report.html"), "complete_documents": len(documents),
            "candidate_count": len(_candidate_rows(records)), "exact_three_way_cases": len(review["cases"]),
            "actual_v6_rereviews": sum(record["costs"]["actual_rereviews"] for record in records.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = render_review(args.root, args.output)
    except (ValueError, OSError, KeyError, TypeError, UnicodeError) as error:
        parser.exit(2, f"Report not generated: {error}\n")
    print(rep._json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
