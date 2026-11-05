"""Read-only, completed-input report for the shared-candidate v3/v5 trial.

Run after omission_v5_trial_ops manifest and the editorial review.json exist.
Only output/report.html and output/review.json are written, immutably. Stage A
is always a preview replay, never an application or a new outcome assessment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
from pathlib import Path
from urllib.parse import urlsplit

import paired_trial as paired
import replication as rep


VARIANTS = (("current", "現行 v3"), ("improved", "改善 v5"))
REFERENCE_LABELS = {
    "valid_correction": "妥当な修正", "unnecessary_change": "不要な変更",
    "harmful": "有害な変更", "unverifiable": "検証不能", "unlabeled": "参照ラベルなし",
}
REVIEW_LISTS = ("verdict", "design", "stage_a_notes", "new_notes", "recommendations", "limitations")


def esc(value):
    return html.escape(str(value), quote=True)


def paragraphs(values):
    return "".join(f"<p>{esc(value)}</p>" for value in values)


def source_link(url, title):
    """Escape all text, and make only ordinary HTTP(S) source URLs clickable."""
    if not isinstance(url, str) or any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise ValueError("Source URL must be a printable HTTP(S) URL")
    parsed = urlsplit(url)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Source URL must be an HTTP(S) URL without credentials")
    return f"<a class='source-url' href='{esc(url)}' rel='noreferrer'>{esc(title)}</a>"


def _count(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"Missing or invalid count: {label}")
    return value


def _require_complete_summary(protocol, summary):
    """Do not convert missing, incomplete or invalid outcomes into zeroes."""
    if summary.get("experiment_id") != protocol["id"]:
        raise ValueError("Summary experiment does not match protocol")
    specs = {row["id"]: row for row in protocol["documents"]}
    records = summary.get("documents", [])
    if len(records) != len(specs) or {row["id"] for row in records} != set(specs):
        raise ValueError("Complete paired summary required for every planned document")
    for record in records:
        identity = record["id"]
        if (record.get("status") != "complete" or record.get("missing") or
                record.get("validation_errors") or not record.get("paired")):
            raise ValueError(f"Complete paired summary required: {identity}")
        if record.get("cohort") != "new" or specs[identity]["cohort"] != "new":
            raise ValueError("Fresh outcomes must be the new cohort only")
        mapping = protocol["arm_mapping"][identity]
        arms = record.get("arms", {})
        if set(arms) != set(mapping) or set(mapping.values()) != {"current", "improved"}:
            raise ValueError(f"Incomplete arm mapping: {identity}")
        primary = record["paired"]["primary"]
        denominator = _count(primary["denominator"], identity + "/denominator")
        if denominator != len(primary["eligible_issue_ids"]):
            raise ValueError(f"Common denominator mismatch: {identity}")
        for arm, data in arms.items():
            if (data.get("status") != "complete" or data.get("missing") or
                    data.get("validation_errors") or not data.get("metrics") or data.get("variant") != mapping[arm]):
                raise ValueError(f"Complete arm required: {identity}/{arm}")
            variant = data["variant"]
            for key in ("detected", "resolved", "resolved_by_removal"):
                if _count(primary[variant][key], identity + "/" + key) > denominator:
                    raise ValueError(f"Count exceeds common denominator: {identity}/{key}")
            metrics = data["metrics"]
            for group, key in (("new_problems", "confirmed_problem"),
                               ("patch_reviews", "unnecessary_change"), ("false_findings", "false_alarm")):
                _count(metrics[group][key], identity + "/" + group)
            _count(metrics["changed_locations"], identity + "/changed_locations")
    return {record["id"]: record for record in records}


def _validate_cases(review, texts):
    rep._string(review.get("headline"), "review.headline", nonempty=True)
    for field in REVIEW_LISTS:
        rep._strings(review.get(field), "review." + field)
    if not isinstance(review.get("cases"), list):
        raise ValueError("review.cases must be a list")
    for case in review["cases"]:
        identity = case["document_id"]
        if identity not in texts:
            raise ValueError("Case must refer to a completed new document: " + identity)
        for field in ("title", "category"):
            rep._string(case.get(field), "case." + field, nonempty=True)
        for field in ("assessment", "record_ids"):
            rep._strings(case.get(field), "case." + field)
        for variant in ("original", "current", "improved"):
            quote = case.get(variant)
            if not isinstance(quote, str) or not quote.strip() or quote not in texts[identity][variant]:
                raise ValueError(f"Case {case['title']}: {variant} quote is not exact")
        for source in case.get("sources", []):
            source_link(source["url"], source["title"])
    notes = review.get("guard_notes", [])
    rep._strings([notes] if isinstance(notes, str) else notes, "review.guard_notes")


def _workspace_path(root, name):
    # Frozen trial references are workspace-relative, not working-directory-relative.
    return paired._relative(root.parent.parent, name)


def _stage_records(root, stage):
    freeze_path = root / "stage_a/input_freeze.json"
    if stage.get("input_freeze_sha256") != rep._sha(freeze_path):
        raise ValueError("Stage A input freeze changed")
    freeze = rep._read_json(freeze_path)
    frozen = {}
    for identity, files in freeze["documents"].items():
        frozen[identity] = {}
        for name, entry in files.items():
            path = _workspace_path(root, entry["path"])
            if rep._sha(path) != entry["sha256"]:
                raise ValueError("Frozen Stage A source changed: " + str(path))
            frozen[identity][name] = rep._read_json(path)
    for entry in freeze["references"].values():
        if rep._sha(_workspace_path(root, entry["path"])) != entry["sha256"]:
            raise ValueError("Frozen Stage A reference labels changed")
    rows = stage["rows"]
    if len({(row["document_id"], row["patch_id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate Stage A candidate")
    for doc in stage["documents"]:
        identity = doc["document_id"]
        selected = [row for row in rows if row["document_id"] == identity]
        patches = {row["patch_id"]: row for row in frozen[identity]["candidate"]["patches"]}
        if doc["candidate_count"] != len(selected) or {row["patch_id"] for row in selected} != set(patches):
            raise ValueError("Stage A candidate coverage mismatch: " + identity)
        for row in selected:
            if any(row[key] != patches[row["patch_id"]][key] for key in ("before", "after")):
                raise ValueError("Stage A before/after quote changed")
        for version in ("v3", "v5"):
            preview = rep._read_json(root / "stage_a" / identity / (version + ".preview.json"))
            decisions = {row["patch_id"]: row for row in preview["decisions"]}
            if doc[version + "_preview_eligible"] != sum(row[version]["eligible"] for row in selected):
                raise ValueError("Stage A document count mismatch")
            for row in selected:
                decision = decisions[row["patch_id"]]
                if (type(row[version]["eligible"]) is not bool or
                        row[version] != {"eligible": decision["accepted"], "reasons": decision["reasons"]}):
                    raise ValueError("Stage A preview decision mismatch")
    if {row["document_id"] for row in rows} != set(frozen):
        raise ValueError("Stage A document coverage mismatch")
    for status in REFERENCE_LABELS:
        selected = [row for row in rows if row["reference_status"] == status]
        expected = {"denominator": len(selected)}
        for version in ("v3", "v5"):
            eligible = sum(row[version]["eligible"] for row in selected)
            expected[version] = {"eligible": eligible, "rejected": len(selected) - eligible}
        if stage["counts"][status] != expected:
            raise ValueError("Stage A reference-group count mismatch: " + status)
    if sum(group["denominator"] for group in stage["counts"].values()) != len(rows):
        raise ValueError("Unknown Stage A reference status")
    return freeze


def load_completed(root, output):
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise ValueError("Complete paired summary.json is required before report generation")
    protocol, _ = paired.validate_preregistration(root)
    summary = rep._read_json(summary_path)
    documents = _require_complete_summary(protocol, summary)
    if not (output / "index.html").is_file():
        raise ValueError("Completed paired index.html is required")
    manifest = rep._read_json(root / "operator_manifest.json")
    if (output / "operator_manifest.json").read_bytes() != (root / "operator_manifest.json").read_bytes():
        raise ValueError("Output operator manifest differs from frozen trial manifest")
    if manifest["preregistration_sha256"] != rep._sha(root / "preregistration.json"):
        raise ValueError("Operator manifest preregistration mismatch")
    manifests = {doc["id"]: doc for doc in manifest["documents"]}
    if set(manifests) != set(documents):
        raise ValueError("Operator manifest must cover all completed documents")
    records, texts = {}, {}
    for spec in protocol["documents"]:
        identity = spec["id"]
        folder = paired._relative(root, spec["path"])
        row = manifests[identity]
        for field in ("candidate_identity", "ledger_identity", "patch_judgment_identity", "findings_identity",
                      "gate_replay_verified", "effective_equals_scored", "joint_audit_verified"):
            if row.get(field) is not True:
                raise ValueError(f"Operator integrity check incomplete: {identity}/{field}")
        for name, expected in row["input_and_output_sha256"].items():
            if rep._sha(paired._relative(folder, name)) != expected:
                raise ValueError(f"Operator artifact changed: {identity}/{name}")
        original = rep._read_json(folder / "input/original_report.json")
        candidate = rep._read_json(folder / "shared/candidate.raw.json")
        guard = rep._read_json(folder / "shared/guard.patch.raw.json")
        final_review = rep._read_json(folder / "shared/final_review.raw.json")
        screenshot = folder / "input/deep_research_completed.png"
        if not screenshot.is_file():
            raise ValueError("Deep Research completion screenshot missing: " + str(screenshot))
        record = {"folder": folder, "spec": spec, "original": original, "candidate": candidate,
                  "guard": guard, "final_review": final_review, "arms": {}, "manifest": row,
                  "screenshot": screenshot}
        texts[identity] = {"original": original["text"]}
        for arm, data in documents[identity]["arms"].items():
            arm_folder = folder / "arms" / arm
            preview = rep._read_json(arm_folder / "preview/gate.preview.json")
            gate = rep._read_json(arm_folder / "gated/gate.decisions.json")
            revised = rep._read_json(arm_folder / "revised_report.json")
            if any(revised[field] != gate["final_applied"][field] for field in ("text", "sources")):
                raise ValueError("Scored revised report differs from final gate output")
            if preview["selection"] != gate["selection"]:
                raise ValueError("Preview selection changed before final application")
            accepted = {row["patch_id"] for row in gate["decisions"] if row["accepted"]}
            eligible = {row["patch_id"] for row in preview["decisions"] if row["accepted"]}
            if (not accepted.issubset(eligible) or gate["summary"]["accepted_patch_count"] != len(accepted) or
                    preview["summary"]["accepted_patch_count"] != len(eligible)):
                raise ValueError("Gate final/preview counts disagree")
            if gate["effective_repair"]["findings"] != candidate["findings"]:
                raise ValueError("Shared findings identity changed")
            if gate["summary"] != row["gate_summary"][arm]:
                raise ValueError("Manifest gate summary mismatch")
            record["arms"][data["variant"]] = {"arm": arm, "folder": arm_folder, "preview": preview,
                "gate": gate, "revised": revised, "eligible": eligible, "accepted": accepted}
            texts[identity][data["variant"]] = revised["text"]
        records[identity] = record
    stage = rep._read_json(root / "stage_a/summary.json")
    stage_freeze = _stage_records(root, stage)
    prior_root = _workspace_path(root, protocol["development_revision"]["previous_trial"])
    prior = rep._read_json(prior_root / "stage_a/summary.json")
    if (prior.get("input_freeze_sha256") != stage["input_freeze_sha256"] or
            rep._sha(prior_root / "preregistration.json") != protocol["development_revision"]["previous_preregistration_sha256"]):
        raise ValueError("Prior v4 replay provenance mismatch")
    review = rep._read_json(root / "review.json")
    _validate_cases(review, texts)
    return protocol, summary, documents, records, stage, stage_freeze, prior_root, prior, review


STYLE = """
:root{color-scheme:light;--ink:#1c2b40;--muted:#526278;--line:#dbe3ec;--blue:#165f9c;--paper:#fff}
*{box-sizing:border-box}body{margin:0;background:#eef3f8;color:var(--ink);font-family:system-ui,'Yu Gothic',sans-serif;line-height:1.8}
main{max-width:1320px;margin:auto;padding:36px 28px 80px}h1{font-size:clamp(1.7rem,4vw,2.7rem);line-height:1.4;max-width:1000px}h2{font-size:1.55rem;margin:3rem 0 1rem}h3{font-size:1.13rem;margin:.3rem 0 1rem}p{margin:.7rem 0}a{color:var(--blue);text-underline-offset:3px}nav{display:flex;gap:18px;flex-wrap:wrap;margin:24px 0}.eyebrow{color:#3c587a;letter-spacing:.07em;font-size:.85rem}.lead,.card{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:24px;margin:20px 0}.lead{border-top:5px solid var(--blue)}.notice{background:#fff5dc;border-left:4px solid #c8921b;padding:16px 20px;margin:20px 0}.small{color:var(--muted);font-size:.88rem}.badge{display:inline-block;border-radius:5px;background:#e6eef8;padding:2px 9px;font-size:.83rem;margin-bottom:8px}.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%;background:var(--paper);font-size:.9rem}th,td{padding:11px 13px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#e4edf7;font-weight:650}td.num{text-align:right;white-space:nowrap}tbody tr:last-child td{border-bottom:0}.triple{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}.double{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.quote,pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f9;padding:15px;border-left:3px solid #9bb1cc;font-family:inherit;font-size:.92rem}.filters{display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin:18px 0}.filters label{font-size:.88rem}.filters input,.filters select{display:block;padding:9px 12px;border:1px solid #aebdd1;border-radius:5px;background:white;color:var(--ink);min-width:190px;max-width:100%}.filters input{width:360px}details{margin:12px 0}summary{cursor:pointer;font-weight:600}details[open]>summary{margin-bottom:14px}li{margin:.35rem 0}.records{word-break:break-word}.candidate-row td:last-child{width:65%}img{display:block;max-width:100%;height:auto;border:1px solid var(--line);margin:12px 0}figure{margin:25px 0}figcaption{font-size:.9rem;color:var(--muted)}[hidden]{display:none!important}@media(max-width:900px){.triple,.double{grid-template-columns:1fr}main{padding:24px 14px 60px}.lead,.card{padding:18px}.candidate-row td:last-child{min-width:360px}}@media print{body{background:white}main{max-width:none;padding:0}.filters,nav{display:none}details{break-inside:avoid}a{color:inherit}.scroll{overflow:visible}.triple{grid-template-columns:1fr}.card{break-inside:avoid}}
"""

FILTER_SCRIPT = """
const query=document.getElementById('candidate-search');
const reference=document.getElementById('reference-filter');
const changed=document.getElementById('changed-filter');
const rows=[...document.querySelectorAll('.candidate-row')];
function filterCandidates(){
 const term=query.value.toLocaleLowerCase();let count=0;
 for(const row of rows){const visible=(!term||row.textContent.toLocaleLowerCase().includes(term))&&(!reference.value||row.dataset.reference===reference.value)&&(!changed.value||row.dataset.changed===changed.value);row.hidden=!visible;if(visible)count++;}
 document.getElementById('filter-count').textContent=count+' / '+rows.length+' 候補を表示';
}
for(const control of [query,reference,changed])control.addEventListener('input',filterCandidates);
filterCandidates();
"""


def _table(headers, rows):
    return ("<div class='scroll'><table><thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers) +
            "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for row in rows) +
            "</tbody></table></div>")


def _stage_html(stage, prior):
    counts = stage["counts"]
    parts = ["<section id='stage-a'><h2>Stage A：既知候補の規則プレビュー</h2>",
        "<div class='notice'>これは最終本文の評価前の適格性比較です。v5の最終適用・最終納品・問題解消を示しません。既知の開発用データを再利用しています。</div>"]
    rows = []
    for status, label in REFERENCE_LABELS.items():
        group = counts[status]
        rows.append([label, group["denominator"], group["v3"]["eligible"], group["v3"]["rejected"],
                     group["v5"]["eligible"], group["v5"]["rejected"]])
    parts.append(_table(["凍結した参照ラベル", "候補数", "v3 適格", "v3 棄却", "v5 適格", "v5 棄却"], rows))
    total = len(stage["rows"])
    unknown = counts["unlabeled"]["denominator"]
    parts.append(f"<p>参照ラベルは {total - unknown} / {total} 候補をカバーし、残る {unknown} 候補はこの監査のラベルがありません。全候補の採用精度は算出しません。有害・検証不能ラベルの母数が0の場合、その検出性能は評価できません。</p>")
    rows = []
    for doc in stage["documents"]:
        covered = sum(row["document_id"] == doc["document_id"] and row["reference_status"] != "unlabeled" for row in stage["rows"])
        rows.append([doc["document_id"], doc["candidate_count"], covered, doc["candidate_count"] - covered,
                     doc["v3_preview_eligible"], doc["v5_preview_eligible"], doc["historical_v3_final_accepted"]])
    parts.append(_table(["旧文書ID", "全候補", "参照あり", "参照なし", "v3 適格", "v5 適格", "歴史的v3 最終採用"], rows))
    historic_veto = sum(doc["v3_preview_eligible"] - doc["historical_v3_final_accepted"] for doc in stage["documents"])
    parts.append(f"<p class='small'>旧文書IDは新規文書IDと別の資料です。歴史的v3の最終評価で留保されたプレビュー適格候補は {historic_veto} 件。規則による棄却と最終評価の留保を混同しません。</p>")
    before = sum(doc["v3_preview_eligible"] for doc in prior["documents"])
    after = sum(doc["v4_preview_eligible"] for doc in prior["documents"])
    parts.append(f"<h3>先行v4の失敗を含む開発履歴</h3><p>先行v4の同一入力プレビューは v3 {before} 件 → v4 {after} 件で、適格性が変わった候補は {len(prior['changed_eligibility'])} 件でした。v5はこの結果を見た後、自由記述の出典位置の完全一致を外し、正規化した出典URLの一致と他の条件を保つ変更を加えています。Stage Aの結果は独立した検証ではありません。</p>")
    parts.append("<h3>全候補：原文・変更案・判定理由</h3><p class='small'>以下の変更案はプレビュー用の候補です。最終納品本文ではありません。</p><div class='filters'><label>語句検索<input id='candidate-search' type='search' placeholder='文書ID・候補ID・原文・理由'></label><label>参照ラベル<select id='reference-filter'><option value=''>すべて</option>")
    parts.extend(f"<option value='{esc(status)}'>{esc(label)}</option>" for status, label in REFERENCE_LABELS.items())
    parts.append("</select></label><label>適格性の変化<select id='changed-filter'><option value=''>すべて</option><option value='yes'>変化あり</option><option value='no'>変化なし</option></select></label><span id='filter-count' role='status' aria-live='polite'></span></div><div class='scroll'><table><thead><tr><th>旧文書 / 候補</th><th>参照判定</th><th>適格性・完全な引用・理由</th></tr></thead><tbody>")
    for row in stage["rows"]:
        changed = row["v3"]["eligible"] != row["v5"]["eligible"]
        parts.append(f"<tr class='candidate-row' data-reference='{esc(row['reference_status'])}' data-changed='{'yes' if changed else 'no'}'><td>{esc(row['document_id'])} / {esc(row['patch_id'])}</td><td>{esc(REFERENCE_LABELS[row['reference_status']])}<br>{esc(row['reference_case_id'] or '—')}</td><td>")
        states = [f"{version}：{'適格' if row[version]['eligible'] else '棄却'}" for version in ("v3", "v5")]
        parts.append(f"<details><summary>{esc(' → '.join(states))}{' · 変化あり' if changed else ''}</summary><div class='double'><div><strong>変更前（完全一致）</strong><div class='quote'>{esc(row['before'])}</div></div><div><strong>候補の変更後（完全一致）</strong><div class='quote'>{esc(row['after'])}</div></div>")
        for version in ("v3", "v5"):
            parts.append(f"<strong>{version} の理由</strong>" + paragraphs(row[version]["reasons"] or ["規則上の棄却理由なし（最終評価前）"]))
        parts.append("<strong>凍結参照ラベルの理由</strong>" + paragraphs([row["reference_reason"] or "この候補への参照ラベルはありません。"])+ "</details></td></tr>")
    parts.append("</tbody></table></div></section>")
    return "\n".join(parts)


def _new_html(protocol, documents, records, artifact):
    parts = [f"<section id='new'><h2>新規{len(documents)}本：共通の確認済み問題を比較</h2>",
        "<p>原文・修正候補・主張台帳・パッチ判断・指摘は両群で共有です。発見は共有候補の性質であり、ゲートの改善効果ではありません。検出判定の不一致は評価者の不整合として示します。</p>",
        "<div class='scroll'><table><thead><tr><th>文書</th><th>共通母数</th><th>共有発見 / 未発見</th><th>ゲート</th><th>解消</th><th>残存</th><th>確認済み新問題</th><th>不要変更</th><th>誤指摘</th><th>変更箇所</th></tr></thead><tbody>"]
    for spec in protocol["documents"]:
        identity = spec["id"]
        doc, record = documents[identity], records[identity]
        primary = doc["paired"]["primary"]
        denominator = primary["denominator"]
        consistency = record["manifest"]["detection_consistency"]
        detected = primary["current"]["detected"]
        equal = consistency["consistent"] and detected == primary["improved"]["detected"]
        detection = f"{detected} / {denominator - detected}" if equal else "判定不一致（下記）"
        for index, (variant, label) in enumerate(VARIANTS):
            arm = next(row for row in doc["arms"].values() if row["variant"] == variant)
            metrics = arm["metrics"]
            parts.append("<tr>")
            if index == 0:
                parts.append(f"<td rowspan='2'><a href='#new-{esc(identity)}'>{esc(identity)}</a><br>{esc(spec['theme'])}</td><td class='num' rowspan='2'>{denominator}</td><td rowspan='2'>{esc(detection)}</td>")
            values = [label, primary[variant]["resolved"], denominator - primary[variant]["resolved"],
                      metrics["new_problems"]["confirmed_problem"], metrics["patch_reviews"]["unnecessary_change"],
                      metrics["false_findings"]["false_alarm"], metrics["changed_locations"]]
            parts.append("".join(f"<td class='num'>{esc(value)}</td>" for value in values) + "</tr>")
    parts.append("</tbody></table></div><p class='small'>共通母数は凍結baselineのconfirmed_problemから、いずれかの群がnot_an_original_problemとしたIDを両群から除いた数です。「解消」はresolvedのみ。部分解消・未解消・検証不能は残存に含みます。新問題・不要変更・誤指摘は重なり得るため合算しません。</p>")
    for spec in protocol["documents"]:
        identity = spec["id"]
        doc, record = documents[identity], records[identity]
        primary = doc["paired"]["primary"]
        parts.append(f"<article class='card' id='new-{esc(identity)}'><h3>{esc(identity)}：{esc(spec['theme'])}</h3>")
        parts.append(f"<p>共通母数からの除外 {len(primary['excluded_issue_ids'])} 件、除外判定の不一致 {len(primary['exclusion_disagreements'])} 件。現行のみ解消 {len(primary['current_only_resolved'])} 件、改善のみ解消 {len(primary['improved_only_resolved'])} 件、両群解消 {len(primary['both_resolved'])} 件、両群未解消 {len(primary['neither_resolved'])} 件。</p>")
        parts.append(f"<p class='small'>解消のうち削除によるもの：現行 {primary['current']['resolved_by_removal']} 件、改善 {primary['improved']['resolved_by_removal']} 件。原文評価で未確定だった問題は主表の母数から分け、下の副次評価記録に掲載しています。</p>")
        conditions = []
        for variant, label in VARIANTS:
            arm = record["arms"][variant]
            final = record["final_review"]["reviews"][arm["arm"]]
            conditions.append([label + " / " + arm["arm"], len(arm["eligible"]), len(arm["accepted"]),
                               len(arm["eligible"] - arm["accepted"]), final["status"], final["coverage_complete"]])
        parts.append(_table(["ゲート / 匿名群", "最終評価前の適格", "最終採用", "最終評価で留保", "最終評価", "評価範囲完全の申告"], conditions))
        improved, current = record["arms"]["improved"], record["arms"]["current"]
        extra = improved["accepted"] - current["accepted"]
        extra_preview = improved["eligible"] - current["eligible"]
        decisions = {row["patch_id"]: row for row in improved["gate"]["decisions"]}
        reviews = {row["patch_id"]: row for row in record["guard"]["patch_reviews"]}
        types = Counter(decisions[patch_id].get("eligibility_route") or "未記録" for patch_id in extra)
        content_types = Counter(addition["content_kind"] for patch_id in extra for addition in reviews[patch_id]["additions"])
        parts.append(f"<p>v5のみ適格 {len(extra_preview)} 件、v5のみ最終採用 {len(extra)} 件。追加採用ID：{esc(', '.join(sorted(extra)) or 'なし')}。</p>")
        parts.append("<p>追加採用の経路：" + esc("、".join(f"{kind} {count}件" for kind, count in sorted(types.items())) or "なし") + "。追加部分の種類：" + esc("、".join(f"{kind} {count}箇所" for kind, count in sorted(content_types.items())) or "なし") + "。</p>")
        for variant, label in VARIANTS:
            arm = record["arms"][variant]
            final = record["final_review"]["reviews"][arm["arm"]]
            parts.append(f"<details><summary>{esc(label)}：最終評価と採用ID</summary><p>適格ID：{esc(', '.join(sorted(arm['eligible'])) or 'なし')}<br>最終採用ID：{esc(', '.join(sorted(arm['accepted'])) or 'なし')}</p><p>{esc(final['reason'])}</p>{paragraphs(arm['gate']['summary']['final_review_blockers'])}</details>")
        consistent = record["manifest"]["detection_consistency"]
        parts.append("<details><summary>発見判定・除外・副次評価の記録</summary>")
        parts.append(f"<p>現行の発見 {primary['current']['detected']} / 未発見 {primary['denominator'] - primary['current']['detected']}、改善の発見 {primary['improved']['detected']} / 未発見 {primary['denominator'] - primary['improved']['detected']}。検出判定の整合：{esc(consistent['consistent'])}。</p>")
        if not consistent["consistent"]:
            parts.append("<div class='notice'>同じ指摘に対する評価者の判定不一致です。ゲートによる発見性能差とは扱いません。</div>")
        parts.append("<pre>" + esc(rep._json_bytes({"detection_consistency": consistent,
            "excluded_issue_ids": primary["excluded_issue_ids"], "exclusion_disagreements": primary["exclusion_disagreements"],
            "baseline_unresolved": doc["paired"]["baseline_unresolved"],
            "arm_secondary_metrics": {row["variant"]: row["metrics"] for row in doc["arms"].values()}}).decode("utf-8")) + "</pre></details>")
        parts.append(f"<p>{artifact(Path('index.html'), '全問題ID・全文・全文差分', relative_output=True, fragment=identity)} · {artifact(record['folder'] / 'shared/candidate.raw.json', '共有候補')} · {artifact(record['folder'] / 'shared/guard.patch.raw.json', '共有パッチ判断')} · {artifact(record['folder'] / 'shared/final_review.raw.json', '最終本文判断')}</p></article>")
    parts.append("</section>")
    return "\n".join(parts)


def render_review(root: Path, output: Path):
    root, output = Path(root).resolve(), Path(output).resolve()
    protocol, summary, documents, records, stage, stage_freeze, prior_root, prior, review = load_completed(root, output)

    def artifact(path, label, *, relative_output=False, fragment=None):
        path = output / path if relative_output else path
        if not path.is_file():
            raise ValueError("Linked artifact is missing: " + str(path))
        href = rep._artifact_link(path, output) + ("#" + fragment if fragment else "")
        return f"<a href='{esc(href)}'>{esc(label)}</a>"

    title = protocol["report_title"]
    parts = ["<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{esc(title)}</title><style>{STYLE}</style></head><body><main>",
        f"<div class='eyebrow'>{esc(protocol['report_date'])} · 共有候補 v3 / v5 · AIによる暫定評価</div><h1>{esc(protocol['report_heading'])}</h1>",
        "<nav aria-label='レポート内の移動'><a href='#new'>新規文書の最終結果</a><a href='#stage-a'>Stage Aプレビュー</a><a href='#cases'>原文と両修正案</a><a href='#records'>原資料</a></nav>",
        f"<div class='lead'><h3>{esc(review['headline'])}</h3>{paragraphs(review['verdict'])}</div>",
        "<h2>固定した実験条件</h2>", paragraphs(review["design"]),
        _new_html(protocol, documents, records, artifact), paragraphs(review["new_notes"]),
        _stage_html(stage, prior), paragraphs(review["stage_a_notes"]),
        "<section id='cases'><h2>原文と両修正案：具体例</h2><p class='small'>3列の引用は凍結原文と実際に採点された両群の修正文に完全一致することを生成時に検査しています。この検査が保証するのは引用の同一性です。評価の妥当性や出典の正しさの証明ではありません。</p>"]
    for case in review["cases"]:
        record = records[case["document_id"]]
        parts.append(f"<article class='card'><span class='badge'>{esc(case['document_id'])} · {esc(case['category'])}</span><h3>{esc(case['title'])}</h3><p class='small'>記録ID：{esc(', '.join(case['record_ids']))}</p><div class='triple'>")
        for variant, label in (("original", "原文"), *VARIANTS):
            path = record["folder"] / "input/original_report.json" if variant == "original" else record["arms"][variant]["folder"] / "revised_report.json"
            parts.append(f"<div><strong>{artifact(path, label)}</strong><div class='quote'>{esc(case[variant])}</div></div>")
        parts.extend(["</div>", paragraphs(case["assessment"]), "<ul>"])
        parts.extend("<li>" + source_link(source["url"], source["title"]) + "</li>" for source in case.get("sources", []))
        parts.append("</ul></article>")
    parts.append("</section>")
    if review.get("guard_notes"):
        notes = review["guard_notes"]
        parts.extend(["<h2>ゲート判断の補足</h2>", paragraphs([notes] if isinstance(notes, str) else notes)])
    parts.extend(["<h2>次の判断</h2>", paragraphs(review["recommendations"]), "<h2>限界</h2>", paragraphs(review["limitations"]),
        "<section id='records'><h2>原文・修正文・評価記録</h2><p class='records'>",
        artifact(output / "index.html", "全問題ID・全文差分の索引"), " · ", artifact(output / "summary.json", "機械集計JSON"), "</p>"])
    links = [(root / "protocol.json", "固定計画"), (root / "preregistration.json", "事前固定ハッシュ"),
        (root / "stage_a_protocol.json", "Stage A計画"), (root / "stage_a/summary.json", "v5 Stage A結果"),
        (root / "stage_a/input_freeze.json", "Stage A入力凍結"), (prior_root / "preregistration.json", "先行v4事前固定"),
        (prior_root / "stage_a/summary.json", "先行v4の失敗結果"), (root / "operator_manifest.json", "整合確認・実行記録"),
        (root / "generation_prompts.json", "固定した生成プロンプト"), (root / "review.json", "本文の編集レビュー")]
    links += [(root / name, label) for name, label in (
        ("method_review.json", "独立した最終方法レビュー"),
        ("stage_a_method_review.json", "既知候補の独立方法レビュー"),
        ("editorial_spotchecks.json", "親担当の補足確認"),
        ("generation_completed.json", "原文取得記録"),
        ("generation_ui_recovery.json", "調査画面の表示復旧記録"),
        ("execution_notes.json", "実行上の注記")) if (root / name).exists()]
    links += [(root / "instructions" / name, label) for name, label in (("repair_v1.txt", "共有修正者指示"),
        ("guard.txt", "共有パッチ判断指示"), ("final_review.txt", "最終本文判断指示"), ("reference.txt", "原文評価指示"), ("post.txt", "比較評価指示"))]
    parts.append("<ul>" + "".join("<li>" + artifact(path, label) + "</li>" for path, label in links) + "</ul>")
    parts.append("<details><summary>Stage Aの凍結原資料と参照ラベル</summary><ul>")
    for identity, entries in stage_freeze["documents"].items():
        for name, entry in entries.items():
            parts.append("<li>" + artifact(_workspace_path(root, entry["path"]), "旧 " + identity + " / " + name) + "</li>")
        for version in ("v3", "v5"):
            parts.append("<li>" + artifact(root / "stage_a" / identity / (version + ".preview.json"), "旧 " + identity + " / " + version + "プレビュー") + "</li>")
    parts.extend("<li>" + artifact(_workspace_path(root, entry["path"]), "参照 " + name) + "</li>" for name, entry in stage_freeze["references"].items())
    parts.append("</ul></details>")
    for identity, record in records.items():
        folder = record["folder"]
        parts.append(f"<h3>{esc(identity)}：{esc(record['spec']['theme'])}</h3><ul>")
        if record["original"].get("chat_url"):
            parts.append("<li>" + source_link(record["original"]["chat_url"], "Deep Researchの元チャット") + "</li>")
        raw = [("input/original_report.json", "原文JSON"), ("input/original_report.txt", "原文テキスト"),
            ("shared/candidate.raw.json", "共有修正候補"), ("shared/ledger.raw.json", "共有主張台帳"),
            ("shared/guard.patch.raw.json", "共有パッチ判断"), ("shared/final_review.raw.json", "共同最終本文判断"),
            ("evaluation/baseline.raw.json", "原文だけの評価"), ("evaluation/paired_post.raw.json", "共同比較評価"),
            ("evaluation/detection_consistency.json", "発見判定の整合")]
        parts.extend("<li>" + artifact(folder / name, label) + "</li>" for name, label in raw)
        for variant, label in VARIANTS:
            arm = record["arms"][variant]
            for name, suffix in (("revised_report.txt", "修正文"), ("revised_report.json", "修正文JSON"),
                ("preview/gate.preview.json", "最終評価前プレビュー"), ("gated/gate.decisions.json", "最終ゲート記録"),
                ("gate_linkage.json", "採点との対応"), ("post.raw.json", "群別評価")):
                parts.append("<li>" + artifact(arm["folder"] / name, label + " / " + suffix) + "</li>")
        parts.append("</ul>")
    parts.append("<details><summary>新規Deep Researchの完成画面</summary>")
    for identity, record in records.items():
        parts.append(f"<figure><figcaption>{esc(identity)}：{esc(record['spec']['theme'])}</figcaption><img loading='lazy' src='{esc(rep._artifact_link(record['screenshot'], output))}' alt='{esc(identity)}のDeep Research完成画面'></figure>")
    parts.append("</details></section><p class='small'>表の数値は保存済み評価・ゲート出力から生成。参照ラベルと実質判断はAIによる暫定評価です。ハッシュと引用一致は同一性を示し、意味的な正しさや評価者の独立性を証明しません。</p>")
    parts.append(f"</main><script>{FILTER_SCRIPT}</script></body></html>")
    payload = "\n".join(parts).encode("utf-8")
    rep._write_immutable({output / "report.html": payload, output / "review.json": rep._json_bytes(review)})
    return {"report": str(output / "report.html"), "complete_documents": len(documents),
            "stage_a_candidates": len(stage["rows"]), "exact_three_way_cases": len(review["cases"])}


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
