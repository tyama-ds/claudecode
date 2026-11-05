"""Render a static Japanese review of checks on an unchanged research report.

Usage: python natural_report.py --input results.json --output report.html
Input metrics are produced by factcheck.natural_eval.evaluate_claims. This
renderer does not perform model calls or create experimental observations.
"""

from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path
from urllib.parse import urlsplit


UNKNOWN = "未計測"
TRUTH_LABELS = {"correct": "正しい", "incorrect": "誤り", "unresolved": "未確定"}
VERDICT_LABELS = {
    "supported": "支持", "refuted": "反証", "insufficient": "根拠不足・保留",
    "not_extracted": "抽出漏れ", "unjudged": "未判定",
}
CATEGORY_LABELS = {
    "detected_error": "発見した誤り", "missed_error": "見逃し",
    "false_refutation": "誤指摘（誤反証）", "abstained_on_error": "誤りに対する保留",
    "correct_supported": "正しい主張を支持", "correct_abstained": "正しい主張を保留",
}


def _text(value) -> str:
    if value is None or value == "":
        return UNKNOWN
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _e(value) -> str:
    return escape(_text(value), quote=True)


def _percent(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return UNKNOWN
    return f"{value * 100:.1f}%"


def _object(value, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _source_link(value) -> str:
    """Only a supplied, absolute HTTPS source URL becomes a link."""
    if not isinstance(value, str) or not value.strip():
        return _e(None)
    try:
        parsed = urlsplit(value)
        allowed = (parsed.scheme == "https" and parsed.hostname
                   and parsed.username is None and parsed.password is None
                   and not any(char.isspace() for char in value))
    except ValueError:
        allowed = False
    if not allowed:
        return _e(value)
    return f'<a href="{escape(value, quote=True)}" target="_blank" rel="noopener noreferrer">{escape(value)}</a>'


def _table(headers: list[str], rows: list[list[str]], css: str = "") -> str:
    # Callers supply escaped cell HTML; header text is always escaped here.
    head = "".join(f"<th>{_e(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table class="{css}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_report(data: dict) -> str:
    """Return escaped standalone HTML; missing optional measurements stay unknown."""
    if not isinstance(data, dict):
        raise ValueError("input must be an object")
    metrics = _object(data.get("metrics"), "metrics")
    counts = _object(metrics.get("counts"), "metrics.counts")
    truths = _object(counts.get("truth"), "metrics.counts.truth")
    errors = _object(metrics.get("error_detection"), "metrics.error_detection")
    correct = _object(metrics.get("correct_claims"), "metrics.correct_claims")
    coverage = _object(metrics.get("coverage"), "metrics.coverage")
    gate = _object(metrics.get("support_only_gate"), "metrics.support_only_gate")
    no_check = _object(metrics.get("no_check"), "metrics.no_check")
    provenance = _object(data.get("provenance"), "provenance")
    claims = data.get("claims", [])
    limitations = data.get("limitations", [])
    if not isinstance(claims, list) or any(not isinstance(row, dict) for row in claims):
        raise ValueError("claims must be an array of objects")
    if not isinstance(limitations, list):
        raise ValueError("limitations must be an array")

    title = data.get("title") or "原文レポートのファクトチェック実験"
    zero_errors = errors.get("reference_errors") == 0
    zero_correct = correct.get("reference_correct") == 0
    no_error_rate = "未評価（原文の誤りが0件）"
    error_rate = lambda key: no_error_rate if zero_errors else _percent(errors.get(key))
    false_rate = "未評価（正しい主張が0件）" if zero_correct else _percent(correct.get("false_refutation_rate"))

    parts = [
        '<p class="eyebrow">FACT-CHECK · UNEDITED REPORT</p>',
        f'<h1>{_e(title)}</h1>',
        f'<p class="lead">{_e(data.get("conclusion"))}</p>',
        '<div class="note">原文の主張と、別工程で作成した独立評価を照合します。未確定は正答・誤答のどちらにも含めません。'
        '引用の支持関係や正解ラベルの品質は、この集計だけでは保証されません。</div>',
        '<div class="cards">' + "".join(
            f'<div class="card"><span>{_e(label)}</span><strong>{_e(counts.get(key))}</strong></div>'
            for key, label in (("claims", "元の対象主張"), ("extracted", "抽出できた主張"), ("judged", "判定できた主張"))
        ) + '</div>',
        '<h2>独立評価による原文の内訳</h2>',
        _table(["正しい", "誤り", "未確定"], [[_e(truths.get(key)) for key in ("correct", "incorrect", "unresolved")]]),
    ]
    if zero_errors:
        parts.append('<p class="warning">原文に確認済みの誤りが0件のため、誤りの検出・削減効果は未評価です。全問正解や改善成功とは扱いません。</p>')

    review_cases = data.get("review_cases", [])
    if review_cases:
        parts.extend([
            '<h2>原文で実際に起きたこと</h2>',
            '<p>数値の正誤、引用先の適合、条件や一般化の根拠不足を区別します。未確定の主張を確認済みの誤りには数えません。</p>',
            _table(["区分", "原文の例", "自動チェック", "独立レビュー・残った問題"], [
                [_e(row.get("category")), f'<blockquote>{_e(row.get("quote"))}</blockquote>' if row.get("quote") else '該当例なし（集計結果）',
                 _e(row.get("checker")), _e(row.get("review")) + '<div class="source">' + _source_link(row.get("source_url")) + '</div>']
                for row in review_cases
            ]),
        ])
    secondary = data.get("secondary_measurements", [])
    if secondary:
        parts.extend(['<h2>補助集計：引用と判定の完了状況</h2>',
                      _table(["指標", "結果", "解釈"], [[_e(row.get(key)) for key in ("name", "value", "meaning")] for row in secondary])])
    citation_review = data.get("citation_review") or {}
    groups = citation_review.get("wrong_source_groups", [])
    if groups:
        parts.extend([
            '<h2>確認できた引用先の取り違え</h2>',
            '<p>同じ誤引用が複数の数値にかかるため、引用箇所と主張数を分けています。原文中の番号は保存した原文の出典一覧に対応します。</p>',
            _table(["箇所", "原文", "問題と対応すべき出典", "チェック記録"], [
                [_e(row.get("group_id")), '<blockquote>' + _e(row.get("quote")) + '</blockquote>',
                 _e(row.get("reason")) + '<br>原文の引用番号：' + _e(row.get("citation_ids"))
                 + '<br>対応する既存の引用番号：' + _e(row.get("appropriate_source_citation_ids")),
                 _e("指摘あり" if row.get("checker_detected") else "指摘なし")]
                for row in groups
            ]),
        ])

    parts.extend([
        '<h2>チェックで発見した誤り・見逃し・誤指摘</h2>',
        '<p>「発見」は誤った主張への反証です。見逃しは支持・抽出漏れ・未判定を含み、誤りに対する保留は別に示します。'
        '正しい主張への根拠不足判定は、誤反証と区別します。</p>',
        _table(["項目", "件数", "割合", "割合の母数"], [
            [_e("発見した誤り"), _e(errors.get("detected")), _e(error_rate("detection_rate")), _e("原文の確認済みの誤り")],
            [_e("見逃し"), _e(errors.get("missed")), _e(error_rate("miss_rate")), _e("原文の確認済みの誤り")],
            [_e("誤りに対する保留"), _e(errors.get("abstained")), _e(error_rate("abstention_rate")), _e("原文の確認済みの誤り")],
            [_e("正しい主張への誤反証"), _e(correct.get("false_refutations")), _e(false_rate), _e("原文の正しい主張")],
            [_e("正しい主張への根拠不足判定"), _e(correct.get("insufficient")), _e(_percent(correct.get("insufficient_rate"))), _e("原文の正しい主張")],
        ]),
        _table(["見逃しの内訳", "誤って支持", "抽出漏れ", "未判定"], [[
            _e("誤りの件数"), _e(errors.get("missed_supported")), _e(errors.get("missed_not_extracted")), _e(errors.get("missed_unjudged")),
        ]]),
        f'<p class="small">主張抽出の再現率：{_e(_percent(coverage.get("extraction_recall")))} ／ '
        f'元の全主張に対する判定率：{_e(_percent(coverage.get("judged_fraction")))}</p>',
        '<h2>チェック前後の比較：支持された主張だけを提供する場合</h2>',
        '<p>以下は「支持だけ提供し、反証・根拠不足・抽出漏れ・未判定は保留する」方針の比較です。'
        '原文を自動修正した結果ではありません。誤りを止める効果と、正しい情報を失う影響を同時に確認します。</p>',
    ])
    before_counts = _object(no_check.get("provided_by_truth"), "metrics.no_check.provided_by_truth")
    after_counts = _object(gate.get("provided_by_truth"), "metrics.support_only_gate.provided_by_truth")
    parts.append(_table(["条件", "提供する全主張", "正しい主張", "誤った主張", "未確定の主張", "保留"], [
        [_e("チェックなし"), _e(no_check.get("provided"))] + [_e(before_counts.get(key)) for key in ("correct", "incorrect", "unresolved")] + [_e(0 if no_check.get("provided") is not None else None)],
        [_e("支持だけ提供"), _e(gate.get("provided"))] + [_e(after_counts.get(key)) for key in ("correct", "incorrect", "unresolved")] + [_e(gate.get("withheld"))],
    ]))
    reduction = no_error_rate if zero_errors else _percent(gate.get("known_error_reduction_rate"))
    parts.append(_table(["確認済み誤りの提供削減率", "元の正しい主張の保持率", "全主張の提供率"], [[
        _e(reduction), _e(_percent(gate.get("correct_retention_rate"))), _e(_percent(gate.get("provided_fraction"))),
    ]]))
    parts.append('<p class="small">未確定の提供が残る場合、確認済み誤りの割合を全体の誤り率とは解釈できません。全件を保留すれば誤りの提供も0になりますが、正しい主張の保持率も併せて評価します。</p>')

    parts.extend(['<h2>原文の主張・独立評価・自動判定の全件一覧</h2>',
                  '<p>引用された原文は編集せず表示します。根拠・理由欄は評価者が入力した記録であり、この画面がその内容を自動検証したものではありません。</p>'])
    claim_rows = []
    for row in claims:
        claim_rows.append([
            _e(row.get("claim_id")),
            f'<strong>今回の判定対象：{_e(row.get("description"))}</strong><blockquote>{_e(row.get("quote"))}</blockquote>',
            _e(TRUTH_LABELS.get(row.get("truth"), row.get("truth"))),
            _e(VERDICT_LABELS.get(row.get("verdict"), row.get("verdict"))),
            _e(CATEGORY_LABELS.get(row.get("category"), row.get("category"))),
            f'<div class="evidence">{_e(row.get("evidence"))}</div><div class="source">{_source_link(row.get("source_url"))}</div>',
            _e(row.get("rationale")),
        ])
    if claim_rows:
        parts.append(_table(["主張ID", "原文", "独立評価", "自動判定", "結果分類", "評価の根拠・出典", "判定理由・レビュー"], claim_rows, "claims"))
    else:
        parts.append(f'<p>{UNKNOWN}</p>')
    if isinstance(counts.get("claims"), int) and len(claims) != counts["claims"]:
        parts.append(f'<p class="warning">全件表示の確認が必要です。集計対象 {_e(counts["claims"])} 件に対し、原文付き明細は {_e(len(claims))} 件です。</p>')

    parts.append('<h2>独立評価と実行の記録</h2>')
    parts.append(_table(["項目", "記録"], [[_e(key), _e(value)] for key, value in provenance.items()]) if provenance else f'<p>{UNKNOWN}</p>')
    parts.append('<h2>この結果から言える範囲</h2>')
    parts.append('<ul>' + ''.join(f'<li>{_e(value)}</li>' for value in limitations) + '</ul>' if limitations else f'<p>{UNKNOWN}</p>')
    next_steps = data.get("next_steps", [])
    if next_steps:
        parts.append('<h2>この結果を受けて次に変えること</h2>')
        parts.append('<ol>' + ''.join(f'<li>{_e(value)}</li>' for value in next_steps) + '</ol>')
    parts.append('<footer>保存された入力JSONと集計に基づく表示です。このHTMLの生成時には、調査・モデル呼び出し・再判定を行っていません。</footer>')

    style = """
body{margin:0;background:#f3f5f6;color:#1d303b;font:15px/1.8 'Segoe UI','Yu Gothic',Meiryo,sans-serif}
main{max-width:1320px;margin:28px auto;padding:36px 42px;background:#fff;border:1px solid #dce4e8;border-radius:12px}
h1{font-size:28px;line-height:1.5;margin:10px 0 18px}h2{font-size:20px;line-height:1.6;border-top:1px solid #dce4e8;padding-top:26px;margin-top:32px}
.eyebrow{font-size:11px;letter-spacing:2px;color:#436775}.lead{font-size:18px;font-weight:600;white-space:pre-wrap}
.note,.warning{padding:14px 18px;border-left:4px solid #28718a;background:#eaf4f8}.warning{background:#fff5df;border-color:#b77910}
.cards{display:flex;gap:16px;margin:24px 0}.card{flex:1;padding:16px 20px;background:#f4f7f8;border-radius:8px}.card span{display:block;color:#526875}.card strong{display:block;font-size:27px}
.table-wrap{overflow-x:auto;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:11px 12px;border-bottom:1px solid #dce4e8;text-align:left;vertical-align:top}th{background:#edf3f5;font-weight:600}td{white-space:pre-wrap;overflow-wrap:anywhere}
.claims{min-width:1150px;table-layout:fixed}.claims th:nth-child(1){width:6%}.claims th:nth-child(2){width:21%}.claims th:nth-child(3){width:7%}.claims th:nth-child(4){width:8%}.claims th:nth-child(5){width:10%}.claims th:nth-child(6){width:25%}.claims th:nth-child(7){width:23%}
blockquote{margin:0;padding:0 0 0 10px;border-left:3px solid #c4d5dd}.source{font-size:12px;margin-top:8px;overflow-wrap:anywhere}a{color:#096986}.small,footer{font-size:12px;color:#526875}footer{margin-top:34px;border-top:1px solid #dce4e8;padding-top:16px}li{margin:8px 0}
@media(max-width:700px){main{margin:0;border:0;border-radius:0;padding:22px 16px}.cards{gap:8px}.card{padding:12px 10px}.card strong{font-size:23px}h1{font-size:23px}}
@media print{body{background:white}main{margin:0;padding:0;max-width:none;border:0}h2{break-after:avoid}.table-wrap{overflow:visible}.claims{min-width:0;font-size:9px}th,td{padding:5px}thead{display:table-header-group}a{color:inherit}blockquote{padding-left:5px}}
"""
    return ('<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_e(title)}</title><style>{style}</style></head><body><main>'
            + '\n'.join(parts) + '</main></body></html>')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8-sig"))
    html = render_report(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
