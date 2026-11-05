"""Readable review of a frozen paired trial; audit numbers come from summary.json."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re

from replication import _artifact_link, _json_bytes, _write_immutable


def render_review(root: Path, output: Path) -> None:
    root, output = root.resolve(), output.resolve()
    review = json.loads((root / "review.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    docs = {d["id"]: d for d in summary["documents"]}
    specs = {d["id"]: d for d in protocol["documents"]}
    for case in review["cases"]:
        record = docs[case["document_id"]]
        folder = root / specs[case["document_id"]]["path"]
        paths = {"original": folder / "input/original_report.json"}
        for arm, data in record["arms"].items():
            paths[data["variant"]] = folder / "arms" / arm / "revised_report.json"
        for key in ("original", "current", "improved"):
            text = json.loads(paths[key].read_text(encoding="utf-8"))["text"]
            if not case[key] or case[key] not in text:
                raise ValueError(f"Case {case['title']}: {key} quote is not exact")
    e = lambda v: html.escape(str(v), quote=True)
    link = lambda p: e(_artifact_link(p, output))
    paragraphs = lambda values: "".join(f"<p>{e(v)}</p>" for v in values)
    counts = {cohort: sum(d["cohort"] == cohort for d in protocol["documents"])
              for cohort in ("regression", "new")}
    title = protocol.get("report_title", protocol.get("title", "修正手順の改善：回帰テストと新規レポートでの比較"))
    heading = protocol.get("report_heading", protocol.get("title", protocol.get("report_title")))
    date = protocol.get("report_date", protocol.get("date"))
    if date is None:
        suffix = re.search(r"(\d{4})(\d{2})(\d{2})$", protocol["id"])
        date = "-".join(suffix.groups()) if suffix else "日付未指定"
    passes = protocol.get("design", {}).get("passes", 1)
    parts = ["<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
             f"<title>{e(title)}</title>",
             "<style>body{margin:0;background:#f4f6fa;color:#18273b;font-family:system-ui,'Yu Gothic',sans-serif;line-height:1.75}main{max-width:1260px;margin:auto;padding:36px 24px}h1{font-size:2rem;line-height:1.4}h2{margin-top:2.5rem}h3{margin:0 0 .8rem}a{color:#135ea3}.badge{display:inline-block;background:#e0e9f5;padding:3px 10px;border-radius:5px;font-size:.85rem}.lead{background:white;border-left:5px solid #365e8b;padding:20px 24px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;background:white;font-size:.92rem}th,td{padding:10px;border-bottom:1px solid #d9e1eb;text-align:left;vertical-align:top}th{background:#e9eff6}.num{white-space:nowrap;text-align:right}.card{background:white;padding:22px;margin:20px 0;border:1px solid #d9e1eb;border-radius:8px}.triple{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}.quote{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.93rem;padding:14px;background:#f5f7fb;border-left:3px solid #a2b3c7}.small{font-size:.88rem;color:#4d5d72}img{max-width:100%;height:auto}li{margin:.4rem 0}summary{cursor:pointer}@media(max-width:850px){.triple{grid-template-columns:1fr}main{padding:24px 14px}h1{font-size:1.5rem}}</style></head><body><main>",
             f"<span class='badge'>{e(date)} · 旧{counts['regression']}本＋新規{counts['new']}本 · {e(passes)}回修正 · AI暫定評価</span>",
             f"<h1>{e(heading)}</h1>" if heading else "<h1>修正手順を改善すると、<br>見逃しと不要な変更は減るか</h1>",
             f"<div class='lead'><h3>{e(review['headline'])}</h3>{paragraphs(review['verdict'])}</div>",
             "<h2>固定した改善と実験条件</h2>", paragraphs(review["changes"]), paragraphs(review["design"])]
    for cohort, title in (("new", f"新しい{counts['new']}本：現行版と改善版の比較"), ("regression", f"既知の{counts['regression']}本：回帰テスト")):
        parts.append(f"<h2>{title}</h2><div class='scroll'><table><thead><tr><th>文書</th><th>共通問題数</th><th>手順</th><th>発見</th><th>解消</th><th>残存</th><th>新問題</th><th>不要変更</th><th>誤指摘</th><th>変更箇所</th></tr></thead><tbody>")
        for doc in summary["documents"]:
            if doc["cohort"] != cohort:
                continue
            pair = doc["paired"]["primary"] if doc["paired"] else None
            for variant, label in (("current", "現行"), ("improved", "改善")):
                arm = next(v for v in doc["arms"].values() if v["variant"] == variant)
                m = arm["metrics"]
                values = ([pair["denominator"], label, pair[variant]["detected"], pair[variant]["resolved"], pair["denominator"] - pair[variant]["resolved"], m["new_problems"]["confirmed_problem"], m["patch_reviews"]["unnecessary_change"], m["false_findings"]["false_alarm"], m["changed_locations"]] if pair and m else ["—", label] + ["—"] * 7)
                parts.append(f"<tr><td><a href='index.html#{e(doc['id'])}'>{e(doc['id'])}</a> {e(specs[doc['id']]['theme'])}</td>" + "".join(f"<td class='num'>{e(v)}</td>" for v in values) + "</tr>")
        parts.append("</tbody></table></div>")
        parts.append(paragraphs(review[cohort + "_notes"]))
    parts.extend(["<h2>回帰テストの個別判定</h2><div class='scroll'><table><tr><th>確認対象</th><th>結果</th><th>根拠</th></tr>"])
    for gate in review["gates"]:
        parts.append(f"<tr><td>{e(gate['target'])}</td><td>{e(gate['result'])}</td><td>{e(gate['reason'])}</td></tr>")
    parts.extend(["</table></div><h2>原文と両修正案：具体例</h2>"])
    for case in review["cases"]:
        parts.append(f"<article class='card'><span class='badge'>{e(case['document_id'])} · {e(case['category'])}</span><h3>{e(case['title'])}</h3><p class='small'>{e(', '.join(case['record_ids']))}</p><div class='triple'>")
        for key, label in (("original", "原文"), ("current", "現行版の修正後"), ("improved", "改善版の修正後")):
            parts.append(f"<div><strong>{label}</strong><div class='quote'>{e(case[key])}</div></div>")
        parts.extend(["</div>", paragraphs(case["assessment"]), "<ul>"])
        for source in case.get("sources", []):
            parts.append(f"<li><a href='{e(source['url'])}'>{e(source['title'])}</a></li>")
        parts.append("</ul></article>")
    if "guard_notes" in review or "guard_stats" in review:
        parts.append("<h2>Claim guardの補足記録</h2>")
        notes = review.get("guard_notes", [])
        if isinstance(notes, str):
            parts.append(paragraphs([notes]))
        elif isinstance(notes, list) and all(isinstance(note, str) for note in notes):
            parts.append(paragraphs(notes))
        elif notes is not None:
            parts.append("<pre>" + e(json.dumps(notes, ensure_ascii=False, indent=2)) + "</pre>")
        if "guard_stats" in review:
            parts.append("<pre>" + e(json.dumps(review["guard_stats"], ensure_ascii=False, indent=2)) + "</pre>")
    parts.extend(["<h2>誤修正・不要変更と評価の不確実性</h2>", paragraphs(review["safety"]),
                  "<h2>レビューと次の判断</h2>", paragraphs(review["recommendations"]),
                  "<h2>限界</h2>", paragraphs(review["limitations"]),
                  "<h2>原文・修正文・全評価記録</h2><p><a href='index.html'>全問題ID・全文差分・評価台帳</a> · <a href='summary.json'>機械集計JSON</a></p><ul>"])
    for doc in summary["documents"]:
        folder = root / specs[doc["id"]]["path"]
        original = json.loads((folder / "input/original_report.json").read_text(encoding="utf-8"))
        links = [f"<a href='{e(original['chat_url'])}'>Deep Research</a>", f"<a href='{link(folder / 'input/original_report.txt')}'>原文</a>"]
        for arm, data in doc["arms"].items():
            label = "現行版" if data["variant"] == "current" else "改善版"
            links.append(f"<a href='{link(folder / 'arms' / arm / 'revised_report.txt')}'>{label}修正文</a>")
        parts.append(f"<li>{e(doc['id'])}: " + " · ".join(links) + "</li>")
    parts.extend(["</ul><p>" + " · ".join(f"<a href='{link(root / path)}'>{label}</a>" for path, label in (("protocol.json", "固定計画"), ("preregistration.json", "事前固定ハッシュ"), ("instructions/repair_v1.txt", "現行手順"), ("instructions/repair_v2.txt", "改善手順"), ("review.json", "レビュー記録"))) + "</p>",
                  "<p>" + " · ".join(f"<a href='{link(root / path)}'>{label}</a>" for path, label in (("method_review.json", "独立した方法レビュー"), ("editorial_spotchecks.json", "親担当の補足確認"), ("execution_notes.json", "取得遅延・形式訂正の記録"), ("operator_manifest.json", "固定順序・ファイル整合"))) + "</p>",
                  ])
    screenshots = [(root / doc["path"] / "deep_research_completed.png", doc.get("theme", doc["id"]))
                   for doc in protocol["documents"] if doc["cohort"] == "new"]
    screenshots = [(path, theme) for path, theme in screenshots if path.exists()]
    if screenshots:
        parts.append("<details><summary>新しいDeep Researchの完成画面</summary>" + "".join(
            f"<img src='{link(path)}' alt='{e(theme)}のDeep Research完成画面'>" for path, theme in screenshots) + "</details>")
    parts.append("<p class='small'>数値は保存済み評価から集計。結果を見て修正指示や本文を再調整していません。誤指摘・不要変更・新問題は重なる場合があり、合算しません。</p></main></body></html>")
    _write_immutable({output / "report.html": "\n".join(parts).encode("utf-8"), output / "review.json": _json_bytes(review)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_review(args.root, args.output)
