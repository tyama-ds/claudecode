"""Render a regenerable, read-only Japanese inspector of claim-guard records.

No model calls, evidence retrieval, outcome files, or semantic rejudgments.
Missing records remain visibly pending; recorded counts are operational only.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from urllib.parse import urlsplit

import replication as rep


CHECKS = {"quantity": "数量・単位", "metric": "指標・定義", "population": "対象・母集団",
          "time": "時点・期間", "comparison": "比較条件", "modality": "断定・推奨の強さ", "citation": "引用・出典"}
LABELS = {"supported": "根拠あり（AI判定）", "problem": "問題あり（AI判定）",
          "uncertain": "不確か（AI判定）", "nonfactual": "非事実的記述（AI判定）",
          "contradicted": "根拠と矛盾（AI判定）", "missing_material": "重要条件の欠落（AI判定）",
          "unknown": "不明（AI判定）", "not_applicable": "対象外（AI判定）",
          "pending": "未取得・未完了", "invalid": "形式不明・要確認",
          "accepted": "採用（ゲート）", "rejected": "保留・不採用（ゲート）",
          "preserved": "原文の情報を保持", "corrected_with_evidence": "根拠に基づき訂正",
          "removed_unsupported": "根拠なしとして削除", "lost": "情報欠落", "necessary": "必要",
          "optional": "任意の補足", "unnecessary": "不要", "full_section": "該当節を閲覧",
          "partial": "一部のみ閲覧", "unavailable": "閲覧不可", "pass": "通過（AI判定）",
          "fail": "不通過（AI判定）", "factual": "事実", "citation": "引用", "nonfactual_unit": "非事実"}
FILES = {"blocks": "input/block_packet.json", "ledger": "ledger.raw.json", "candidate": "candidate.raw.json",
         "gate": "gated/gate.decisions.json", "summary": "gated/summary.json", "guard": "gated/guard.raw.json",
         "final": "gated/final_report.json"}


def _e(value):
    return html.escape(str(value), quote=True)


def _obj(value):
    return value if isinstance(value, dict) else {}


def _rows(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _text(value, fallback="未取得・不明"):
    return _e(value) if value is not None and value != "" else fallback


def _quote(value):
    if value is None:
        return "<div class='quote pending'>未取得・不明</div>"
    return "<div class='quote'>" + (_e(value) if value != "" else "（空文字・削除）") + "</div>"


def _badge(status, allowed=None):
    if status is None:
        key = "pending"
    elif not isinstance(status, str) or (allowed is not None and status not in allowed):
        key = "invalid"
    else:
        key = status if status in LABELS else "invalid"
    return f"<span class='badge status-{key}'>{_e(LABELS[key])}</span>"


def _safe_source_url(value):
    """Permit only absolute HTTP(S), keeping the original URL byte-for-byte."""
    if not isinstance(value, str) or not value or "\\" in value or any(c.isspace() or ord(c) < 32 for c in value):
        return False
    try:
        parsed = urlsplit(value)
        return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname) and parsed.port != 0
    except ValueError:
        return False


def _source(value):
    if _safe_source_url(value):
        return f"<a class='source-url' href='{_e(value)}' target='_blank' rel='noopener noreferrer'>{_e(value)}</a>"
    return f"<span class='unsafe-url'>{_text(value)}（無効なURL・リンク無効）</span>"


def _path(root, value):
    if not isinstance(value, str) or "\\" in value or ":" in value or any(p in {"", ".", ".."} for p in value.split("/")):
        raise ValueError("Document path must be a safe relative path")
    path = PurePosixPath(value)
    resolved = (root / Path(*path.parts)).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root):
        raise ValueError("Document path escapes the trial root")
    return resolved


def _read(path):
    if not path.exists():
        return None, "未取得・未完了"
    try:
        value = rep._read_json(path)
        if not isinstance(value, dict):
            raise ValueError("JSONの最上位がオブジェクトではありません")
        return value, None
    except (ValueError, OSError, UnicodeError) as error:
        return None, "読み取りエラー: " + str(error)


def _load_document(root, protocol, doc):
    folder = _path(root, doc.get("path"))
    mapping = protocol.get("arm_mapping", {}).get(doc["id"], {"current": "current", "improved": "improved"})
    arms = [arm for arm, variant in mapping.items() if variant == "improved"]
    if len(arms) != 1 or arms[0] not in doc.get("arms", []):
        raise ValueError(f"Exactly one improved arm is required for {doc['id']}")
    arm = arms[0]
    if arm not in {"A", "B", "current", "improved"}:
        raise ValueError("Unknown arm path")
    paths = {key: (folder if key == "blocks" else folder / "arms" / arm) / name for key, name in FILES.items()}
    records, errors = {}, {}
    for key, path in paths.items():
        records[key], error = _read(path)
        if error:
            errors[key] = error
    return {"doc": doc, "arm": arm, "paths": paths, "records": records, "errors": errors}


def _refs(values, evidence):
    if not isinstance(values, list):
        return "<span class='pending'>参照未取得</span>"
    if not values:
        return "参照なし（記録値）"
    parts = []
    for value in values:
        matches = evidence.get(str(value), [])
        if not matches:
            parts.append(f"<span class='pending'>{_e(value)}（対応する根拠記録なし）</span>")
        else:
            parts.extend(f"<a href='#{anchor}'>{_e(value)}{'（重複ID）' if len(matches) > 1 else ''}</a>" for anchor in matches)
    return " · ".join(parts)


def _evidence(records, index):
    entries, lookup = [], {}
    for origin in ("ledger", "guard"):
        for row in _rows(_obj(records[origin]).get("evidence")):
            # Identical repeated evidence is one item; conflicting IDs stay visible.
            if any(existing == row for _, existing in entries):
                continue
            anchor = f"evidence-{index}-{len(entries)}"
            entries.append((anchor, row))
            lookup.setdefault(str(row.get("evidence_id")), []).append(anchor)
    parts = ["<details class='evidence'><summary>根拠資料・閲覧状態（AIの申告）</summary>"]
    if not entries:
        parts.append("<p class='pending'>根拠資料の記録なし／未取得。根拠が存在しないという意味ではありません。</p>")
    for anchor, row in entries:
        primary = "一次資料（AI申告）" if row.get("primary") is True else "二次資料（AI申告）" if row.get("primary") is False else "一次・二次の区分不明"
        parts.append(f"<article id='{anchor}' class='evidence-item'><h4>{_text(row.get('evidence_id'))}</h4>"
                     f"<p>{_source(row.get('url'))}</p><p>{_e(primary)} · {_badge(row.get('access'))}</p>"
                     f"<p><strong>参照箇所：</strong>{_text(row.get('locator'))}</p><p>{_text(row.get('summary'))}</p></article>")
    parts.append("</details>")
    return "".join(parts), lookup


def _claims(records, index, evidence):
    ledger, packet = _obj(records["ledger"]), _obj(records["blocks"])
    blocks = {str(row.get("block_id")): row for row in _rows(packet.get("blocks"))}
    parts = ["<h3>原文の主張台帳</h3>"]
    claims = ledger.get("claims")
    if not isinstance(claims, list):
        return "".join(parts) + "<p class='pending'>主張台帳は未取得・未完了です。主張数・解決数を0件として扱いません。</p>"
    if not claims:
        parts.append("<p>台帳に主張は登録されていません（記録値）。抽出の完全性は別途確認が必要です。</p>")
    for number, claim in enumerate(_rows(claims)):
        status = claim.get("status")
        status = status if isinstance(status, str) and status in {"supported", "problem", "uncertain", "nonfactual"} else "pending" if status is None else "invalid"
        block = blocks.get(str(claim.get("block_id")), {})
        quote, block_text = claim.get("quote"), block.get("text")
        exact = isinstance(quote, str) and bool(quote) and isinstance(block_text, str) and quote in block_text
        parts.append(f"<article class='card claim filter-item' data-kind='claim' data-status='{status}'>"
                     f"<header><h4>{_text(claim.get('claim_id'))} <small>{_text(claim.get('block_id'))}</small></h4>{_badge(status)}</header>"
                     + _quote(quote) + f"<p class='small'>{'原文ブロック内の完全一致引用' if exact else '引用一致を確認できません（原文未取得または不一致）'}</p>"
                     f"<p>{_text(claim.get('reason'))}</p><div class='two'><div><strong>レポートが述べる対象条件</strong><p>{_text(claim.get('report_scope'))}</p></div>"
                     f"<div><strong>根拠資料が対象とする条件</strong><p>{_text(claim.get('source_scope'))}</p></div></div>")
        parts.append("<details><summary>文脈：見出し・表の条件・前後の原文</summary>")
        context = claim.get("context_block_ids")
        if not isinstance(context, list) or not context:
            parts.append("<p class='pending'>文脈ブロック未取得・未指定</p>")
        for block_id in context if isinstance(context, list) else []:
            parts.append(f"<h5>{_e(block_id)}</h5>" + _quote(blocks.get(str(block_id), {}).get("text")))
        parts.append("</details><div class='scroll'><table><caption>7項目の照合（すべてAI判定）</caption><thead><tr><th>項目</th><th>判定</th><th>理由</th><th>根拠ID</th></tr></thead><tbody>")
        checks = _obj(claim.get("checks"))
        for key, label in CHECKS.items():
            check = _obj(checks.get(key))
            parts.append(f"<tr><th>{_e(label)}</th><td>{_badge(check.get('status'), {'supported', 'contradicted', 'missing_material', 'unknown', 'not_applicable'})}</td>"
                         f"<td>{_text(check.get('rationale'))}</td><td>{_refs(check.get('evidence_ids'), evidence)}</td></tr>")
        parts.append("</tbody></table></div></article>")
    coverage = {str(row.get("block_id")): row for row in _rows(ledger.get("blocks"))}
    parts.append("<details><summary>原文の全ブロックと台帳への割当</summary>")
    for block_id, block in blocks.items():
        record = coverage.get(block_id, {})
        parts.append(f"<h4>{_e(block_id)}</h4>" + _quote(block.get("text"))
                     + f"<p>主張ID: {_text(', '.join(map(str, record['claim_ids'])) if isinstance(record.get('claim_ids'), list) else None)}"
                     f" · 非事実的記述の理由: {_text(record.get('nonfactual_reason'), '記録なし')}</p>")
    parts.append("</details>")
    return "".join(parts)


def _units(review, evidence):
    parts = ["<h5>原文の情報単位と保持状況（AI判定）</h5>"]
    units = review.get("units")
    if not isinstance(units, list):
        parts.append("<p class='pending'>情報単位の評価は未取得・未完了です。</p>")
    elif not units:
        parts.append("<p>情報単位の登録なし（記録値）</p>")
    for unit in _rows(units):
        parts.append(f"<div class='unit'><h6>{_text(unit.get('unit_id'))} · {_badge(unit.get('status'))} · {_badge(unit.get('disposition'))}</h6>"
                     f"<p>主張ID: {_text(', '.join(map(str, unit['claim_ids'])) if isinstance(unit.get('claim_ids'), list) else None)}"
                     f" · 種類: {_text(unit.get('content_kind', 'factual'))}</p><div class='two'><div><strong>変更前</strong>{_quote(unit.get('before_quote'))}</div>"
                     f"<div><strong>変更後</strong>{_quote(unit.get('after_quote'))}</div></div>"
                     f"<p>根拠評価: {_text(unit.get('reason'))}</p><p>保持・訂正の理由: {_text(unit.get('rationale'))}</p>"
                     f"<p>{_refs(unit.get('evidence_ids'), evidence)}</p></div>")
    parts.append("<h5>追加された情報・引用（AI判定）</h5>")
    additions = review.get("additions")
    if not isinstance(additions, list):
        parts.append("<p class='pending'>追加情報の評価は未取得・未完了です。</p>")
    elif not additions:
        parts.append("<p>追加情報なしとの申告（記録値。網羅性の証明ではありません）</p>")
    for addition in _rows(additions):
        parts.append("<div class='unit'>" + _badge(addition.get("status")) + _quote(addition.get("after_quote"))
                     + f"<p>種類: {_text(addition.get('content_kind'))} · {_text(addition.get('reason'))}</p>"
                     f"<div class='two'><p>レポートの対象条件: {_text(addition.get('report_scope'))}</p>"
                     f"<p>根拠の対象条件: {_text(addition.get('source_scope'))}</p></div><p>{_refs(addition.get('evidence_ids'), evidence)}</p></div>")
    return "".join(parts)


def _patches(records, evidence):
    candidate, gate, guard = (_obj(records[key]) for key in ("candidate", "gate", "guard"))
    decisions = {str(row.get("patch_id")): row for row in _rows(gate.get("decisions"))}
    reviews = {str(row.get("patch_id")): row for row in _rows(guard.get("patch_reviews"))}
    parts = ["<h3>修正候補とゲート判断</h3><p class='small'>不採用は原文を残す操作です。問題解消や誤修正の防止を意味しません。採用も事実の正しさを保証しません。</p>"]
    patches = candidate.get("patches")
    if not isinstance(patches, list):
        return "".join(parts) + "<p class='pending'>修正候補は未取得・未完了です。</p>"
    if not patches:
        parts.append("<p>修正候補なし（記録値）。問題がない／全て解決したという意味ではありません。</p>")
    for patch in _rows(patches):
        identity = str(patch.get("patch_id"))
        decision, review = decisions.get(identity, {}), reviews.get(identity, {})
        accepted = decision.get("accepted")
        state = "accepted" if accepted is True else "rejected" if accepted is False else "pending"
        parts.append(f"<article class='card patch filter-item' data-kind='patch' data-status='{state}'><header><h4>{_e(identity)}</h4>{_badge(state)}</header>"
                     f"<p>主張ID: {_text(', '.join(map(str, patch['claim_ids'])) if isinstance(patch.get('claim_ids'), list) else None)}"
                     f" · 操作: {_text(patch.get('action'))}</p><div class='two'><div><strong>候補の変更前</strong>{_quote(patch.get('before'))}</div>"
                     f"<div><strong>候補の変更後（不採用なら反映されません）</strong>{_quote(patch.get('after'))}</div></div>"
                     f"<p>修正者の理由: {_text(patch.get('reason'))}</p><p>必要性（AI判定）: {_badge(review.get('necessity'))}</p>"
                     f"<p>独立評価者の理由: {_text(review.get('rationale', decision.get('review_rationale')))}</p>")
        reasons = decision.get("reasons")
        if not isinstance(reasons, list):
            parts.append("<p class='pending'>ゲート判断・理由は未取得・未完了です。</p>")
        elif reasons:
            parts.append("<strong>ゲートの不採用理由</strong><ul>" + "".join(f"<li>{_e(reason)}</li>" for reason in reasons) + "</ul>")
        else:
            parts.append("<p>ゲートの阻止条件なし（記録値。正しさの保証ではありません）</p>")
        coverage = review.get("coverage_complete")
        parts.append(f"<p>情報単位・追加情報の網羅性申告: {'完全とのAI申告' if coverage is True else '不完全とのAI申告' if coverage is False else '未取得・不明'}</p>"
                     f"<p>{_text(review.get('coverage_reason'))}</p><p>必要性の根拠: {_refs(review.get('evidence_ids'), evidence)}</p>"
                     + _units(review, evidence) + "</article>")
    final = _obj(guard.get("final_review"))
    parts.append("<h4>採用候補を組み合わせた全文の独立確認（AI判定）</h4>" + _badge(final.get("status"))
                 + f"<p>{_text(final.get('reason'))}</p><details><summary>ゲート適用後の全文</summary>" + _quote(_obj(records["final"]).get("text")) + "</details>")
    return "".join(parts)


CSS = """
:root{color-scheme:light;--ink:#203047;--muted:#627084;--line:#d8e0ea;--blue:#225681}*{box-sizing:border-box}
body{margin:0;background:#f4f6f9;color:var(--ink);font:15px/1.7 system-ui,'Yu Gothic',sans-serif}main{max-width:1340px;margin:auto;padding:32px 24px 70px}
a{color:var(--blue);overflow-wrap:anywhere}h1{font-size:2rem;line-height:1.4;margin:.4rem 0 1rem}h2{font-size:1.55rem;margin:0}h3{margin:2rem 0 .8rem}h4{font-size:1.05rem;margin:.2rem 0}h5,h6{font-size:.95rem;margin:1rem 0 .4rem}p{margin:.6rem 0}
.eyebrow{font-size:.8rem;letter-spacing:.1em;color:var(--muted)}.lead,.filters,.document,.overview{background:#fff;border:1px solid var(--line);border-radius:12px;padding:22px;margin:18px 0}.lead{border-left:5px solid var(--blue)}.small,small{font-size:.85rem;color:var(--muted)}
.filters{position:sticky;top:0;z-index:1;display:flex;align-items:end;gap:14px;flex-wrap:wrap;box-shadow:0 3px 12px #243d5710}.filters label{display:grid;gap:5px;font-size:.8rem;font-weight:600}.filters input,.filters select{font:inherit;padding:9px;border:1px solid #adbbcc;border-radius:6px;background:white}.filters input{min-width:220px}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}.metric{padding:14px;background:#eff3f8;border-radius:8px}.metric strong{display:block;font-size:1.35rem}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}.two>div,.two>p{min-width:0}.card{border:1px solid var(--line);border-radius:9px;padding:18px;margin:16px 0}.card header{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}.quote{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f8;border-left:3px solid #bac9d9;padding:12px;margin:.5rem 0;font-size:.93rem}.unit{border-left:3px solid #cbd7e4;padding:4px 14px;margin:16px 0}
.badge{display:inline-block;border-radius:4px;background:#e8edf4;padding:2px 7px;font-size:.78rem;margin:2px 0;white-space:normal}.status-problem,.status-contradicted,.status-lost,.status-fail{background:#fce4e1;color:#8b302c}.status-uncertain,.status-unknown,.status-pending,.status-rejected,.status-missing_material{background:#fff0cd;color:#75541d}.status-supported,.status-accepted,.status-preserved{background:#e3eef7;color:#235174}.pending,.unsafe-url{color:#80551a}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:.85rem}th,td{border:1px solid var(--line);padding:9px;text-align:left;vertical-align:top}th{background:#f0f4f8}caption{text-align:left;font-weight:600;margin:.6rem 0}details{margin:14px 0}summary{cursor:pointer;font-weight:600}.evidence-item{border-top:1px solid var(--line);padding:15px 0}.raw-links{display:flex;gap:8px 18px;flex-wrap:wrap;list-style:none;padding:0}.raw-links li{font-size:.82rem}.document{scroll-margin-top:120px}.evidence-item{scroll-margin-top:120px}[hidden]{display:none!important}.empty-filter{display:none;color:var(--muted)}.no-matches .empty-filter{display:block}
@media(max-width:760px){main{padding:20px 12px 45px}.two{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}.document{padding:16px}.filters{position:static}.filters label{flex:1}h1{font-size:1.55rem}}@media print{.filters{display:none}details{display:block}.document{break-before:page}.card{break-inside:avoid}}
"""

SCRIPT = """
(() => {
  const controls = ['document-filter', 'claim-filter', 'patch-filter', 'text-filter'].map(id => document.getElementById(id));
  function applyFilters() {
    const [doc, claim, patch, search] = controls.map(control => control.value);
    const needle = search.trim().toLocaleLowerCase();
    document.querySelectorAll('.document').forEach(section => {
      section.hidden = doc !== 'all' && section.dataset.document !== doc;
      let shown = 0;
      section.querySelectorAll('.filter-item').forEach(item => {
        const status = item.dataset.kind === 'claim' ? claim : patch;
        item.hidden = (status !== 'all' && item.dataset.status !== status) || (needle && !item.textContent.toLocaleLowerCase().includes(needle));
        if (!item.hidden) shown++;
      });
      section.classList.toggle('no-matches', shown === 0);
    });
    document.getElementById('filter-feedback').textContent = '表示条件を適用中。非表示の記録も保存済みHTMLに含まれます。';
  }
  controls.forEach(control => control.addEventListener('input', applyFilters));
})();
"""


def render_report(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    target = output if output.suffix.lower() == ".html" else output / "claim_guard.html"
    protocol = rep._read_json(root / "protocol.json")
    documents = rep._rows_by_id(protocol.get("documents"), "id", "protocol.documents")
    data = [_load_document(root, protocol, doc) for doc in documents.values()]
    link = lambda path: _e(rep._artifact_link(path, target.parent))
    title = str(protocol.get("report_title", "主張台帳と情報保持の検証")) + "｜Claim guard 記録"
    parts = ["<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             f"<title>{_e(title)}</title><style>{CSS}</style></head><body><main><div class='eyebrow'>CLAIM GUARD · READ-ONLY INSPECTOR</div>",
             f"<h1>{_e(title)}</h1><p>{_text(protocol.get('report_date'))} · 計画文書 {len(data)}本 · 改善手順の監査記録</p>",
             "<p><a href='report.html'>比較実験のレビュー</a> · <a href='index.html'>ペア比較・全評価記録</a> · "
             f"<a href='{link(root / 'protocol.json')}'>固定計画</a></p>",
             "<div class='lead'><strong>AIの意味判定と、機械的な採否を分けて読む</strong><p>主張の正誤、対象条件、根拠の解釈、修正の必要性、情報単位の網羅性はAIの暫定判定です。unknown（不明）や未取得を、支持・解消として扱いません。</p>"
             "<p>ゲートは記録された条件に基づく採否です。<strong>不採用件数は成功件数ではありません。</strong>有効な訂正が止まる場合もあります。問題解消・誤修正防止の比較は別の事後評価で確認します。ここではbaseline、gold、事後評価を読みません。</p></div>",
             "<section class='overview'><h2>文書ごとの取得状況</h2><ul>"]
    for index, item in enumerate(data):
        doc = item["doc"]
        state = f"未取得・読み取り不可 {len(item['errors'])}ファイル" if item["errors"] else "必要な表示用ファイルを取得済み（意味的妥当性は未保証）"
        parts.append(f"<li><a href='#document-{index}'>{_e(doc['id'])} {_text(doc.get('theme'))}</a> — {_e(state)}</li>")
    parts.append("</ul></section><div class='filters' role='search'><label>文書<select id='document-filter'><option value='all'>全ての文書</option>")
    for index, item in enumerate(data):
        parts.append(f"<option value='{index}'>{_e(item['doc']['id'])}</option>")
    parts.append("</select></label>")
    for kind, label, states in (("claim", "主張の状態", ("supported", "problem", "uncertain", "nonfactual", "pending", "invalid")),
                                ("patch", "ゲートの採否", ("accepted", "rejected", "pending"))):
        parts.append(f"<label>{label}<select id='{kind}-filter'><option value='all'>全て</option>" + "".join(
            f"<option value='{state}'>{_e(LABELS[state])}</option>" for state in states) + "</select></label>")
    parts.append("<label>本文・理由・IDを検索<input id='text-filter' type='search' placeholder='例：対象、数量、C001'></label><span id='filter-feedback' class='small' aria-live='polite'>表示条件はこの画面内だけに適用されます。</span></div><noscript><p>JavaScriptが無効のため、フィルターを使わず全記録を表示します。</p></noscript>")
    for index, item in enumerate(data):
        doc, records = item["doc"], item["records"]
        summary = _obj(records["summary"])
        evidence_html, evidence = _evidence(records, index)
        cohort = "回帰セット" if doc.get("cohort") == "regression" else "新規セット"
        parts.append(f"<section class='document' id='document-{index}' data-document='{index}'><div class='eyebrow'>{_e(cohort)} · 改善手順 / {_e(item['arm'])}</div><h2>{_e(doc['id'])} {_text(doc.get('theme'))}</h2><div class='metrics'>")
        for key, label in (("claim_count", "台帳の主張数"), ("candidate_patch_count", "修正候補"), ("accepted_patch_count", "ゲート採用"), ("rejected_patch_count", "ゲート不採用")):
            value = summary.get(key)
            shown = str(value) if type(value) is int and value >= 0 else "—（未取得）"
            parts.append(f"<div class='metric'>{label}<strong>{_e(shown)}</strong><span class='small'>操作・記録件数</span></div>")
        parts.append("</div><details><summary>表示に使用した原記録</summary><ul class='raw-links'>")
        for key, path in item["paths"].items():
            state = item["errors"].get(key)
            label = str(path.relative_to(root)).replace("\\", "/")
            parts.append(f"<li><a href='{link(path)}'>{_e(label)}</a>" + (f" — <span class='pending'>{_e(state)}</span>" if state else " — 取得済み") + "</li>")
        parts.append("</ul></details><p class='empty-filter'>現在の条件に一致する主張・修正候補はありません。未取得状況も確認してください。</p>")
        parts.extend([_claims(records, index, evidence), _patches(records, evidence), evidence_html, "</section>"])
    parts.append("<p class='small'>読み取り専用の表示です。原記録の編集、根拠の追加取得、AI判定の書き換えは行いません。全ブロックの割当や完全一致引用が成立しても、原子的主張の抽出漏れや意味の誤判定がないことは保証されません。</p></main>"
                 + f"<script>{SCRIPT}</script></body></html>")
    target.parent.mkdir(parents=True, exist_ok=True)
    # This inspector is regenerated as pending artifacts arrive; raw records are never changed.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write("\n".join(parts))
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"output": str(target), "planned_documents": len(data),
            "documents_with_pending_or_unreadable_records": sum(bool(item["errors"]) for item in data)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="HTML file or directory for claim_guard.html")
    args = parser.parse_args(argv)
    try:
        result = render_report(args.root, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
