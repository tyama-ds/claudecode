"""Attach a reviewed interpretation without altering grading fields; export HTML."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from urllib.parse import urlsplit

from app import read_json


def percentage(value):
    return "未計測" if value is None else f"{value * 100:.1f}%"


def table(rows, columns):
    head = "".join(f"<th>{escape(label)}</th>" for _, label, _ in columns)
    body = "".join("<tr>" + "".join(
        f"<td>{escape(str(formatter(row.get(key))))}</td>"
        for key, _, formatter in columns
    ) + "</tr>" for row in rows)
    return f"<div class='table'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def write_report(run_dir: Path, review_path: Path):
    summary_path = run_dir / "summary.json"
    result = read_json(summary_path)
    review = read_json(review_path)
    if result.get("run_mode") != "recorded_agent_pilot":
        raise ValueError("この報告書は記録済みLLM応答の予備実験専用です。")
    if not (run_dir / "grading_summary.json").exists():
        (run_dir / "grading_summary.json").write_bytes(summary_path.read_bytes())
    result["review"] = review
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = result["counts"]
    parts = [f"<p class='eyebrow'>NUMERIC FACT-CHECK PILOT · 2026-11-01</p><h1>{escape(review['title'])}</h1>",
             f"<p class='lead'>{escape(review['conclusion'])}</p>",
             f"<p>{counts['reports']}文書・{counts['items']}問・2プロンプト条件・1反復。公開一次資料から再構成した試験用レポートです。</p>",
             "<div class='note'>同じLLM実行環境の別プロンプトを比較しています。異なるAPIモデルの比較ではありません。"
             "合成した誤候補は過去のAIの実績ではありません。費用・トークン数・API応答時間は未計測です。</div>",
             "<h2>穴埋め</h2>"]
    parts.append(table(result["fill_metrics"], [
        ("model_id", "条件", str), ("n", "問題数", str), ("n_correct", "正答", str),
        ("n_incorrect", "誤答", str), ("n_abstain", "保留", str),
        ("correct_rate", "正答率", percentage), ("evidence_valid_rate", "引用一致率", percentage)]))
    parts.append("<h2>共通候補のファクトチェック</h2>")
    parts.append(table(result["check_metrics"], [
        ("model_id", "条件", str), ("n", "候補数", str),
        ("error_detection_rate", "誤り検出率", percentage),
        ("false_flag_rate", "正答の誤指摘率", percentage),
        ("abstain_rate", "保留率", percentage),
        ("passed_error_rate", "通過中の誤り率", percentage)]))
    parts.append("<h2>チェック前後の効果</h2><p>分母は元の全問題数。支持された回答だけを通し、数値の修正は行っていません。</p>")
    baseline = [dict(generator_id=row["model_id"], checker_id="チェックなし",
                     correct_provided_rate=row["correct_rate"], incorrect_provided_rate=row["incorrect_rate"],
                     not_provided_rate=row["abstain_rate"] + row["error_rate"])
                for row in result["fill_metrics"]]
    parts.append(table(baseline + result["pipeline_metrics"], [
        ("generator_id", "穴埋め条件", str), ("checker_id", "判定条件", str),
        ("correct_provided_rate", "正しい回答の提供率", percentage),
        ("incorrect_provided_rate", "誤った回答の提供率", percentage),
        ("not_provided_rate", "未提供率", percentage)]))
    for section in review.get("sections", []):
        parts.append(f"<h2>{escape(section['title'])}</h2>")
        parts.extend(f"<p>{escape(text)}</p>" for text in section.get("paragraphs", []))
        if section.get("bullets"):
            parts.append("<ul>" + "".join(f"<li>{escape(text)}</li>" for text in section["bullets"]) + "</ul>")
    parts.append("<h2>資料・参照</h2><ul>")
    for source in review.get("sources", []):
        url = source["url"]
        if urlsplit(url).scheme != "https":
            raise ValueError("参照URLはHTTPSで指定してください。")
        parts.append(f"<li><a href='{escape(url, quote=True)}'>{escape(source['title'])}</a></li>")
    parts.append("</ul><footer>数値・集計の根拠は同じフォルダーの summary.json、CSV、記録済み回答に保存。"
                 "正解作成とレビューはAIによる資料照合を含み、専門家による正解ラベルの承認は未実施です。</footer>")
    html = """<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>数値ファクトチェック実験のレビュー</title><style>
body{font:15px/1.9 'Segoe UI','Yu Gothic',Meiryo,sans-serif;color:#19343a;background:#f2f5f1;margin:0}
main{max-width:1060px;margin:32px auto;background:white;padding:44px 50px;border-radius:12px}
h1{font-size:28px;line-height:1.5}h2{font-size:20px;margin-top:36px;border-top:1px solid #dce4df;padding-top:22px}
.eyebrow{font-size:11px;letter-spacing:2px;color:#167467}.lead{font-size:18px;font-weight:600}.note{background:#eaf4ef;padding:15px 20px;border-left:4px solid #197d68}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:11px 12px;border-bottom:1px solid #dce4df;text-align:right;white-space:nowrap}th{background:#f2f6f2}th:first-child,td:first-child{text-align:left}.table{overflow-x:auto}li{margin:7px 0}a{color:#086b62}footer{margin-top:35px;font-size:12px;color:#637675}
@media(max-width:700px){main{margin:0;padding:24px 18px}h1{font-size:23px}}@media print{body{background:white}main{margin:0;padding:0}h2{break-after:avoid}table{font-size:10px}}
</style><main>""" + "\n".join(parts) + "</main></html>"
    target = run_dir / "report.html"
    target.write_text(html, encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    print(write_report(args.run, args.review).resolve())
