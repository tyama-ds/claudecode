"""Render an editorial review alongside the immutable replication audit."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from replication import _artifact_link, _json_bytes, _write_immutable


def render_review(root: Path, output: Path) -> None:
    root, output = root.resolve(), output.resolve()
    review = json.loads((root / "review.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    for case in review["cases"]:
        if case["report_id"] not in {"R1", "R2", "R3"}:
            raise ValueError("Unknown report ID in review example")
        folder = root / case["report_id"]
        for label, name in (("before", "input/original_report.json"), ("after", "repair/revised_report.json")):
            report = json.loads((folder / name).read_text(encoding="utf-8"))
            if not case[label] or case[label] not in report["text"]:
                raise ValueError(f"Review {case['title']}: {label} is not an exact report excerpt")
    e = lambda value: html.escape(str(value))
    link = lambda path: e(_artifact_link(path, output))
    paragraphs = lambda values: "".join(f"<p>{e(value)}</p>" for value in values)
    parts = ["<!doctype html><html lang='ja'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width, initial-scale=1'>",
             "<title>未知のDeep Research 3本：固定手順の再現試験</title>",
             "<style>body{margin:0;background:#f4f6fa;color:#18273b;font-family:system-ui,'Yu Gothic',sans-serif;line-height:1.75}main{max-width:1180px;margin:auto;padding:40px 24px}h1{font-size:2rem;line-height:1.4}h2{margin-top:2.5rem}h3{margin:0 0 .8rem}a{color:#135ea3}p{max-width:80em}.badge{display:inline-block;background:#e0e9f5;padding:3px 10px;border-radius:5px;font-size:.85rem}.lead{background:#fff;border-left:5px solid #365e8b;padding:20px 24px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;background:white;font-size:.93rem}th,td{padding:12px;border-bottom:1px solid #d9e1eb;text-align:left;vertical-align:top}th{background:#e9eff6}td.num{text-align:right;white-space:nowrap}.card{background:white;padding:24px;margin:20px 0;border:1px solid #d9e1eb;border-radius:8px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.quote{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.95rem;padding:16px;background:#f5f7fb;border-left:3px solid #a2b3c7}.small{font-size:.88rem;color:#4d5d72}.good{color:#176841}.warn{color:#93431f}img{max-width:100%;height:auto;border:1px solid #d9e1eb}summary{cursor:pointer}li{margin:.4rem 0}@media(max-width:750px){.pair{grid-template-columns:1fr}main{padding:24px 14px}h1{font-size:1.5rem}}</style></head><body><main>",
             "<span class='badge'>2026-11-01 · 3テーマ × 各1本 · 修正1回 · AI暫定評価</span>",
             "<h1>未知のDeep Researchレポートで<br>発見・修正の効果は再現するか</h1>",
             f"<div class='lead'><h3>{e(review['headline'])}</h3>{paragraphs(review['verdict'])}</div>",
             "<h2>実験条件</h2>", paragraphs(review["design"]),
             "<h2>文書別の比較</h2><div class='scroll'><table><thead><tr><th>文書</th><th>確認済み問題</th><th>原文の未確定</th><th>発見</th><th>解消</th><th>見逃し</th><th>残った問題</th><th>新問題</th><th>不要変更</th></tr></thead><tbody>"]
    for record in summary["reports"]:
        m = record["metrics"]
        values = ([m[k] for k in ("confirmed_denominator", "baseline_unresolved", "confirmed_detected", "confirmed_resolved", "confirmed_undetected", "confirmed_not_resolved")] +
                  [m["new_problems"]["confirmed_problem"], m["patch_reviews"]["unnecessary_change"]] if m else ["未完了"] * 8)
        parts.append(f"<tr><td><a href='index.html#{e(record['report_id'])}'>{e(record['report_id'])}: {e(record['theme'])}</a></td>" +
                     "".join(f"<td class='num'>{e(value)}</td>" for value in values) + "</tr>")
    parts.extend(["</tbody></table></div>", paragraphs(review["metric_notes"]),
                  "<h2>具体的に何が変わったか</h2>"])
    for case in review["cases"]:
        parts.extend([f"<article class='card'><span class='badge'>{e(case['report_id'])} · {e(case['category'])}</span><h3>{e(case['title'])}</h3>",
                      f"<p class='small'>監査記録：{e(', '.join(case['record_ids']))}</p>",
                      f"<div class='pair'><div><strong>原文</strong><div class='quote'>{e(case['before'])}</div></div>",
                      f"<div><strong>修正後</strong><div class='quote'>{e(case['after'])}</div></div></div>",
                      paragraphs(case["assessment"]), "<ul>"])
        for source in case.get("sources", []):
            parts.append(f"<li><a href='{e(source['url'])}'>{e(source['title'])}</a></li>")
        parts.append("</ul></article>")
    parts.extend(["<h2>前回との比較とレビュー</h2>", paragraphs(review["comparison"]),
                  "<h2>この結果から進める改良</h2><ol>"])
    parts.extend(f"<li>{e(value)}</li>" for value in review["next_steps"])
    parts.extend(["</ol><h2>限界と未確定</h2>", paragraphs(review["limitations"]),
                  "<h2>原文・修正文・証跡</h2><p><a href='index.html'>全差分・全評価台帳・すべての問題と変更を開く</a> · <a href='summary.json'>集計JSON</a></p><ul>"])
    for rid in ("R1", "R2", "R3"):
        f = root / rid
        original = json.loads((f / "input/original_report.json").read_text(encoding="utf-8"))
        parts.append(f"<li>{e(rid)}：<a href='{e(original['chat_url'])}'>実際のDeep Research</a> · "
                     f"<a href='{link(f / 'input/original_report.txt')}'>未修正原文</a> · "
                     f"<a href='{link(f / 'repair/revised_report.txt')}'>1回修正版</a> · "
                     f"<a href='{link(f / 'generation_metadata.json')}'>生成記録</a></li>")
    parts.extend([f"</ul><p><a href='{link(root / 'protocol.json')}'>生成前に固定した計画</a> · "
                  f"<a href='{link(root / 'repair_instructions.txt')}'>共通修正指示</a> · "
                  f"<a href='{link(root / 'preregistration.json')}'>事前固定ハッシュ</a> · "
                  f"<a href='{link(root / 'review.json')}'>レビュー記録</a></p>",
                  f"<details><summary>実際に完成したDeep Researchの画面</summary><img src='{link(root / 'R1/deep_research_completed.png')}' alt='協働ロボットのDeep Research完成画面'></details>",
                  "<p class='small'>この報告書の数値は保存済みの独立評価から機械集計しています。修正後の評価を理由に本文を再修正した結果ではありません。</p></main></body></html>"])
    _write_immutable({output / "report.html": "\n".join(parts).encode("utf-8"), output / "review.json": _json_bytes(review)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_review(args.root, args.output)
