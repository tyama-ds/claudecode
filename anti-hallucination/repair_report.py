"""Render saved repair/audit observations. This command makes no model calls."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from difflib import HtmlDiff
from html import escape
import hashlib
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlsplit


LABELS = {
    "resolved":"解消", "partially_resolved":"一部解消", "unresolved":"未解消・未確定",
    "unverifiable":"検証不能", "not_an_original_problem":"元から問題とは判定せず",
    "confirmed_problem":"確認済みの不備", "not_a_problem":"問題なし",
    "acceptable":"受容可能", "regression":"誤修正・新たな問題", "unnecessary_change":"不要な変更",
    "preserved":"保持", "omitted":"情報の脱落", "corrupted":"値・意味を損なった", "unchanged":"逐字不変",
    "citation":"引用対応", "condition":"対象・条件", "generalisation":"一般化", "other":"その他",
}


def e(value):
    return escape("未記録" if value is None else str(value), quote=True)


def label(value):
    return e(LABELS.get(value,value))


def links(urls):
    output=[]
    for value in urls or []:
        p=urlsplit(value)
        if p.scheme!="https" or not p.hostname or p.username or p.password or any(c.isspace() for c in value):
            output.append(e(value)); continue
        output.append(f'<a href="{e(value)}" target="_blank" rel="noopener noreferrer">{e(p.hostname)}</a>')
    return " · ".join(output)


def table(headers, rows, css=""):
    return '<div class="scroll"><table class="'+e(css)+'"><thead><tr>'+''.join('<th>'+e(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def page(title, content):
    style="""
body{margin:0;background:#f2f5f7;color:#18303b;font:15px/1.8 'Segoe UI','Yu Gothic',Meiryo,sans-serif}main{max-width:1250px;margin:24px auto;padding:32px;background:#fff;border:1px solid #dce5ea;border-radius:12px}h1{font-size:29px;line-height:1.5}h2{font-size:21px;margin-top:34px;border-top:1px solid #dce5ea;padding-top:22px}h3{font-size:17px}.lead{font-size:18px;font-weight:600}.note{padding:14px 18px;background:#eaf3f7;border-left:4px solid #2d728c}.warning{background:#fff4dd;border-left:4px solid #a96d00;padding:14px 18px}.small{font-size:12px;color:#526977}nav{font-size:14px;margin-bottom:26px}a{color:#076b89;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:12px;border-bottom:1px solid #dce5ea;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#edf3f5}td{white-space:pre-wrap}.scroll{overflow:auto}blockquote{margin:4px 0;padding-left:12px;border-left:3px solid #c4d8e2;white-space:pre-wrap}details{margin:12px 0;padding:14px;background:#f7f9fa}summary{cursor:pointer;font-weight:600}.pair{display:grid;grid-template-columns:1fr 1fr;gap:22px}.pair>div{min-width:0}.report-text{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.85 'Yu Gothic',Meiryo,sans-serif}.diff{font-family:monospace;font-size:11px;white-space:pre}.diff_header{background:#e4edf2}.diff_add{background:#d9f1df}.diff_sub{background:#ffe1dc}.diff_chg{background:#fff0ba}.diff_next{display:none}.stats{display:flex;flex-wrap:wrap;gap:12px}.stat{flex:1;min-width:150px;padding:14px 20px;background:#f0f5f7;border-radius:8px}.stat strong{font-size:25px;display:block}li{margin:8px 0}@media(max-width:760px){main{margin:0;padding:18px;border:0}.pair{grid-template-columns:1fr}h1{font-size:24px}}@media print{main{margin:0;border:0;padding:0}.scroll{overflow:visible}details{break-inside:avoid}}
"""
    return '<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+e(title)+'</title><style>'+style+'</style></head><body><main>'+content+'</main></body></html>'


def render_text(packet, title):
    source_rows=[]
    for source in packet["sources"]:
        source_rows.append([e(source["citation_id"]),e(source.get("title")),links([source["url"]])])
    return page(title,'<nav><a href="report.html">比較・レビューへ戻る</a></nav><h1>'+e(title)+'</h1><p class="note">本文は保存されたテキストのまま表示。単独行の番号は末尾の出典一覧に対応します。</p><div class="report-text">'+e(packet["text"])+'</div><h2>出典一覧</h2>'+table(["番号","タイトル","リンク"],source_rows))


def render_run(folder: Path, output: Path):
    read=lambda name:json.loads((folder/name).read_text(encoding="utf-8-sig"))
    original=read("original_report.json")
    revised=read("revised_report.v1.json")
    changes=read("applied.v1.json")["patches"]
    fixed=read("fixed_targets.json")["targets"]
    baseline=read("baseline.independent.raw.json")
    audit=read("post_audit.v1.raw.json")
    packet=read("audit_packet.v1.json")
    repair=read("repair.v1.raw.json")
    review=read("review.json")
    final=read("revised_report.v2.json") if (folder/"revised_report.v2.json").exists() else None
    followup=read("post_audit.v2.raw.json") if final else None
    for freeze_name in ("pre_repair_freeze.json","repair_v1_freeze.json"):
        for name, expected in read(freeze_name)["files"].items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected, name
    assert hashlib.sha256((folder/'post_audit.v1.raw.json').read_bytes()).hexdigest()==read('post_audit_v1_freeze.json')['sha256']
    if final:
        for name, expected in read('repair_v2_freeze.json')['files'].items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected,name
        assert {r['target_id'] for r in followup['checks']}=={'P30','P14'}
        assert len(followup['checks'])==2 and len(followup['patch_reviews'])==2
        assert {r['patch_id'] for r in followup['patch_reviews']}=={r['patch_id'] for r in read('applied.v2.json')['patches']}
        for row in followup['checks']:
            assert row['status'] in {'resolved','unresolved','unverifiable'}
            assert row['after_quote'] in final['text']
        for row in followup['patch_reviews']:
            assert row['status'] in {'acceptable','regression','unverifiable'}
        assert final['sources']==revised['sources']
        assert hashlib.sha256((folder/'post_audit.v2.raw.json').read_bytes()).hexdigest()==read('post_audit_v2_freeze.json')['sha256']
        assert re.findall(r'\d+(?:[.,]\d+)*',final['text'])==re.findall(r'\d+(?:[.,]\d+)*',revised['text'])
        # This is a preservation check, not a second semantic audit of all 12.
        assert all(r['after_quote'] in final['text'] for r in audit['targets'])
    target_by_id={r["target_id"]:r for r in audit["targets"]}
    baseline_after={r["issue_id"]:r for r in audit["baseline_issues"]}
    patch_after={r["patch_id"]:r for r in audit["patch_reviews"]}
    assert len(target_by_id)==len(audit["targets"])==len(fixed)
    assert set(target_by_id)=={r["target_id"] for r in fixed}
    assert len(baseline_after)==len(audit["baseline_issues"])==len(baseline["issues"])
    assert set(baseline_after)=={r["issue_id"] for r in baseline["issues"]}
    for row in audit["baseline_issues"]:
        assert set(row.get("overlap_target_ids",[])) <= set(target_by_id), row
        assert len(row.get("overlap_target_ids",[])) == len(set(row.get("overlap_target_ids",[]))), row
    assert len(patch_after)==len(audit["patch_reviews"])==len(changes)
    assert set(patch_after)=={p["patch_id"] for p in changes}
    assert len(audit["numeric_controls"])==len(packet["numeric_controls"])
    assert {x['claim_id'] for x in audit['numeric_controls']}=={x['claim_id'] for x in packet['numeric_controls']}
    assert len(audit["controls"])==len(baseline["controls"])
    assert {x['control_id'] for x in audit['controls']}=={x['control_id'] for x in baseline['controls']}
    allowed_outcomes={"resolved","partially_resolved","unresolved","unverifiable","not_an_original_problem"}
    for row in audit['targets']+audit['baseline_issues']:
        assert row['after_status'] in allowed_outcomes, row
        assert isinstance(row['resolved_by_removal'],bool), row
    for row in audit['controls']+audit['numeric_controls']:
        assert row['status'] in {"preserved","omitted","corrupted","unverifiable"}, row
    for row in audit['patch_reviews']:
        assert row['status'] in {"acceptable","regression","unverifiable","unnecessary_change"}, row
    for rows in (audit["targets"],audit["baseline_issues"],audit["controls"],audit["numeric_controls"]):
        for row in rows:
            if row.get("after_quote"):
                assert row["after_quote"] in revised["text"], row
    for row in audit["patch_reviews"]:
        for finding in row.get("new_problems",[]):
            assert finding["quote"] in revised["text"]
    for row in audit["remaining_findings"]:
        assert row["quote"] in revised["text"]

    def outcome_counts(rows): return dict(Counter(r["after_status"] for r in rows))
    groups={kind:[target_by_id[r["target_id"]] for r in fixed if r["type"]==kind] for kind in ["citation","condition","generalisation"]}
    additions=[r for r in audit["baseline_issues"] if not r.get("overlap_target_ids")]
    patch_counts=dict(Counter(r["status"] for r in audit["patch_reviews"]))
    numeric_counts=Counter(r["status"] for r in audit["numeric_controls"])
    control_counts=Counter(r["status"] for r in audit["controls"])
    numeric_counts["unchanged"]=len(packet["mechanically_unchanged_numeric_controls"])
    stats={
        "fixed_target_outcomes":{kind:outcome_counts(rows) for kind,rows in groups.items()},
        "baseline_all_outcomes":outcome_counts(audit["baseline_issues"]),
        "baseline_additional_outcomes":outcome_counts(additions),
        "baseline_additional_count":len(additions),
        "patch_outcomes":patch_counts,
        "numeric_controls":dict(numeric_counts),
        "baseline_controls":dict(control_counts),
        "resolved_by_removal_fixed_targets":sum(r["resolved_by_removal"] for r in audit['targets']),
        "resolved_by_removal_baseline_additional":sum(r["resolved_by_removal"] for r in additions),
        "numeric_control_total":sum(numeric_counts.values()),
        "original_characters":len(original["text"]), "revised_characters":len(revised["text"]),
        "replaced_original_characters":sum(p["end_char"]-p["start_char"] for p in changes),
        "patch_count":len(changes),"added_source_count":len(repair["added_sources"]),
        "human_adjudicated":False,
    }
    if final:
        stats['followup']={
            'patch_count':2,
            'check_outcomes':dict(Counter(r['status'] for r in followup['checks'])),
            'patch_outcomes':dict(Counter(r['status'] for r in followup['patch_reviews'])),
            'final_characters':len(final['text']),
            'all_12_previous_target_quotes_unchanged':True,
            'source_list_unchanged':True,
            'note':'2か所の限定再検査。v1の全監査の再実行ではない。'
        }
    assert stats["numeric_control_total"]==59
    for example in review.get('examples',[]):
        assert example['before'] in original['text']
        assert example['after'] in revised['text']
    output.mkdir(parents=True,exist_ok=True)
    for path in folder.iterdir():
        if path.is_file() and path.suffix in (".json",".txt",".py"):
            shutil.copy2(path,output/path.name)
    (output/"metrics.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    (output/"original_report.html").write_text(render_text(original,"修正前：Deep Research原文"),encoding="utf-8")
    (output/"revised_report.v1.html").write_text(render_text(revised,"自動修正後 v1：全文と出典"),encoding="utf-8")
    if final:
        (output/"revised_report.v2.html").write_text(render_text(final,"最終自動修正版 v2：全文と出典"),encoding="utf-8")
        final_diff=HtmlDiff(wrapcolumn=70).make_table(original['text'].splitlines(),final['text'].splitlines(),fromdesc='修正前（原文）',todesc='最終自動修正版 v2',context=True,numlines=2)
        (output/'diff.v2.html').write_text(page('原文と最終修正版の差分','<nav><a href="report.html">比較・レビューへ戻る</a></nav><h1>原文と最終修正版 v2 の差分</h1><div class="scroll">'+final_diff+'</div>'),encoding='utf-8')
    diff=HtmlDiff(wrapcolumn=70).make_table(original["text"].splitlines(),revised["text"].splitlines(),fromdesc="修正前（原文）",todesc="自動修正後 v1",context=True,numlines=2)
    (output/"diff.v1.html").write_text(page("原文と自動修正文の差分",'<nav><a href="report.html">比較・レビューへ戻る</a></nav><h1>原文と自動修正文の差分</h1><div class="scroll">'+diff+'</div>'),encoding="utf-8")
    pieces=['<nav><a href="original_report.html">修正前の原文</a> ｜ <a href="revised_report.v1.html">自動修正後の全文</a> ｜ <a href="diff.v1.html">差分</a> ｜ <a href="post_audit.v1.raw.json">独立監査JSON</a> ｜ <a href="metrics.json">集計</a></nav>',
            '<p class="small">実Deep Research原文 · 初回自動修正＋限定再修正 · 独立した事後監査</p><h1>引用・条件・一般化を自動修正した前後比較</h1>',
            '<p class="lead">'+e(review["conclusion"])+'</p>',
            '<p class="note">同じ1文書の既知問題を使った改修評価です。修正担当には前回チェッカーの記録を渡しましたが、独立監査の結果は渡していません。参照評価はAIによる暫定評価で、人間の正解裁定は未実施です。</p>',
            '<h2>固定した問題の修正前後：初回 v1</h2>',
            '<p>引用の誤接続と、条件・一般化が未確定だった主張を別集計します。後者を数値の虚偽と数えず、互いに同じ文章へかかる項目を合算した削減率も出しません。</p>']
    outcome_order=["resolved","partially_resolved","unresolved","unverifiable","not_an_original_problem"]
    pieces.append(table(["固定評価軸","修正前の対象数"]+[LABELS[x] for x in outcome_order],[[label(kind),e(len(rows))]+[e(Counter(r["after_status"] for r in rows)[s]) for s in outcome_order] for kind,rows in groups.items()]))
    if final:
        pieces.insert(0,'<nav class="note"><strong><a href="revised_report.v2.html">最終修正版 v2 を読む</a></strong> ｜ <a href="diff.v2.html">原文との最終差分</a> ｜ <a href="post_audit.v2.raw.json">限定再検査の記録</a></nav>')
        pieces.append('<h2>初回で生じた副作用と限定再修正</h2><p>初回の失敗を保存したまま、元の価値判断を弱めた1箇所と、補助レビューで曖昧とされた表現1箇所だけを再修正しました。以下はその2箇所の再検査です。</p>')
        pieces.append(table(['対象','初回での指摘','再修正後の原文','再検査'],[[e(r['target_id']), '分析者の価値判断を不要に弱めた' if r['target_id']=='P30' else '評価条件の掛かり先が曖昧という疑義', '<blockquote>'+e(r['after_quote'])+'</blockquote>',label(r['status'])+'\n'+e(r['reason'])+'\n'+links(r.get('source_urls'))] for r in followup['checks']]))
        pieces.append('<p class="small">既知12項目の修正後引用はv2にも逐字残り、出典一覧・数値も維持されています。12項目すべてを再度意味評価したという計数にはしません。</p>')
    pieces.extend(['<h2>初回 v1 の誤修正と情報の保持</h2>', table(["項目","結果","解釈"],[
        ["自動修正箇所",e(len(changes)),"各箇所の追加事実・意味変化を別担当が監査"],
        ["修正の判定",e(' / '.join(LABELS.get(k,k)+': '+str(v) for k,v in patch_counts.items())),"修正が正解かを修正者自身の採点で決めていない"],
        ["元の数値情報59主張",e(' / '.join(LABELS.get(k,k)+': '+str(v) for k,v in numeric_counts.items())),"不変は変更箇所と重ならない機械的対照。保持は変更された文脈の意味を別担当が確認。元ラベルも暫定。"],
        ["独立監査で選んだ保持対照",e(' / '.join(LABELS.get(k,k)+': '+str(v) for k,v in control_counts.items())),"元の正しい説明・制約・提案を不必要に失っていないかを検査。数値59主張とは別の対象。"],
        ["削除による問題解消",e(f"固定対象 {stats['resolved_by_removal_fixed_targets']}件 / 独立追加 {stats['resolved_by_removal_baseline_additional']}件"),"削除や情報の喪失を、情報を保った修正と区別する。"],
        ["修正対象の文字数",e(f"{stats['replaced_original_characters']} / {stats['original_characters']}文字"),"引用の差し替えだけでも、位置を一意にするため周辺文がpatch範囲に含まれる。意味変更率ではない。"],
    ])])
    pieces.append('<h2>具体的な変更とレビュー</h2>')
    for case in review.get("examples",[]):
        pieces.append('<h3>'+e(case["title"])+'</h3><div class="pair"><div><strong>修正前</strong><blockquote>'+e(case["before"])+'</blockquote></div><div><strong>修正後</strong><blockquote>'+e(case["after"])+'</blockquote></div></div><p>'+e(case["review"])+'</p>'+links(case.get("source_urls")))
    pieces.append('<h2>固定12項目の全件監査</h2>')
    pieces.append(table(["ID・評価軸","原文の問題","修正後","独立監査"],[[e(x['target_id'])+' / '+label(x['type']),'<blockquote>'+e(x['quote'])+'</blockquote>', '<blockquote>'+e(target_by_id[x['target_id']].get('after_quote'))+'</blockquote>', label(target_by_id[x['target_id']]['after_status'])+'\n'+e(target_by_id[x['target_id']]['reason'])+'\n'+links(target_by_id[x['target_id']].get('source_urls'))] for x in fixed]))
    pieces.extend(['<h2>一般化を含めた独立監査</h2>','<p>修正文を見ずに抽出された候補を全件表示します。固定12項目との重複を明示し、同じ不備を別の発見として加算しません。</p>'])
    pieces.append(table(["ID・修正前の評価","原文","修正後の評価","重複する固定項目／理由"],[[e(x['issue_id'])+' / '+label(x['status']),'<blockquote>'+e(x['quote'])+'</blockquote>',label(baseline_after[x['issue_id']]['after_status']), e(', '.join(baseline_after[x['issue_id']].get('overlap_target_ids',[])) or '独立監査で追加')+'\n'+e(baseline_after[x['issue_id']]['reason'])+'\n'+links(baseline_after[x['issue_id']].get('source_urls'))] for x in baseline['issues']]))
    pieces.append('<h2>全修正箇所：変更と誤修正の検査</h2>')
    for patch in changes:
        a=patch_after[patch['patch_id']]
        problems=''.join('<p class="warning">'+e(v['type'])+'：'+e(v['reason'])+'<blockquote>'+e(v['quote'])+'</blockquote>'+links(v.get('source_urls'))+'</p>' for v in a.get('new_problems',[]))
        pieces.append('<details><summary>'+e(patch['patch_id'])+' · '+label(a['status'])+'</summary><div class="pair"><div><strong>修正前</strong><blockquote>'+e(patch['before'])+'</blockquote></div><div><strong>修正後</strong><blockquote>'+e(patch['after'])+'</blockquote></div></div><p>'+e(a['reason'])+'</p>'+problems+'</details>')
    control_before={r['control_id']:r for r in baseline['controls']}
    pieces.append('<h2>独立した保持対照の全件確認</h2>')
    pieces.append(table(["ID","維持すべき原文","判定","理由"],[[e(r['control_id']),e(control_before[r['control_id']]['quote']),label(r['status']),e(r['reason'])] for r in audit['controls']]))
    pieces.append('<h2>初回 v1 に残った問題・未確定事項</h2>')
    remaining=audit['remaining_findings']
    pieces.append(table(["修正後の箇所","状態","理由"],[[e(r['quote']),label(r['status']),e(r['reason'])+'\n'+links(r.get('source_urls'))] for r in remaining]) if remaining else '<p>この監査範囲で追加の残存問題は記録されませんでした。全文の完全性を保証する結果ではありません。</p>')
    if final:
        pieces.append('<p class="note">上記P30は最終版v2で再修正済みです。限定再検査の残存指摘：'+e(len(followup['remaining_findings']))+'件。未知の問題が全文に存在しないことを保証する値ではありません。</p>')
    pieces.append('<h2>レビューと次の検証</h2><ul>'+''.join('<li>'+e(v)+'</li>' for v in review['limitations']+review['next_steps'])+'</ul>')
    pieces.append('<h2>実行記録</h2>'+table(["工程","記録"],[["修正",e(json.dumps(repair['execution'],ensure_ascii=False))],["修正前の独立監査",e(json.dumps(baseline['execution'],ensure_ascii=False))],["修正後の独立監査",e(json.dumps(audit['execution'],ensure_ascii=False))],["モデルID・API費・人のレビュー時間","未計測。実行はこの環境の別LLMエージェント。APIサービスへのモデル別実験ではない。"]]))
    pieces.append('<p class="small">本文の自動修正案、適用結果、原文ハッシュ、監査記録を保存しています。このHTMLの生成は保存済みデータの再表示であり、新たなモデル実験ではありません。</p>')
    (output/"report.html").write_text(page("自動修正の前後比較",'\n'.join(pieces)),encoding="utf-8")
    return stats


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(render_run(args.input,args.output),ensure_ascii=False,indent=2))
