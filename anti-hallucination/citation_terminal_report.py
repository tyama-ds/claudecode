"""Opt-in publication of three scored pairs and one declared terminal input failure.

This reporting-only extension does not modify the frozen runtime or authorize a
worker retry. Missing data alone can never be promoted into a terminal failure.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
from pathlib import Path

import citation_report as report
import citation_trial_ops as ops
import paired_trial as paired
import replication as rep
from factcheck import claim_guard_v5 as v5, claim_guard_v6 as v6

COMPLETED_IDS = ('R1', 'R2', 'N1')
FAILED_ID = 'N2'
INPUT_PATHS = {'original': 'input/original_report.json',
               'candidate': 'staging/candidate.raw.json', 'ledger': 'staging/ledger.raw.json'}
REFUSAL_PATH = 'staging/writer_validation_refusal.json'
EXTENSION_FILES = ('citation_terminal_report.py', 'tests/test_citation_terminal_report.py')
FAILURE_REASON = ('Frozen shared candidate/ledger validation failed before guard or gate processing. '
    'The no-semantic-retry rule is retained; the original report and writer responses are preserved. '
    'No replacement report, corrected candidate, guard judgment or paired outcome is substituted.')


def _trial(root):
    root = Path(root).resolve()
    protocol, runtime = ops.runtime_check(root)
    if protocol.get('planned_document_ids') != ['R1', 'R2', 'N1', 'N2']:
        raise ValueError('This opt-in extension requires exactly the frozen R1/R2/N1/N2 plan')
    if {row['id']: row['cohort'] for row in protocol['documents']} != report.COHORTS:
        raise ValueError('Frozen cohort identity changed')
    return root, protocol, runtime


def _extension_hashes():
    base = Path(__file__).resolve().parent
    return {name: rep._sha(base / name) for name in EXTENSION_FILES}


def _require_no_downstream(folder):
    for directory in ('shared', 'arms'):
        if any(path.is_file() for path in (folder / directory).rglob('*')):
            raise ValueError('Failed document has comparative downstream output: ' + directory)
    allowed_evaluation = {'baseline.raw.json', 'baseline.freeze.json',
                          'baseline.raw.initial.json', 'baseline.raw.format_repair.json'}
    if any(path.is_file() and path.name not in allowed_evaluation
           for path in (folder / 'evaluation').rglob('*')):
        raise ValueError('Failed document has a non-baseline evaluation artifact')
    canonical = {(folder / name).resolve() for name in INPUT_PATHS.values()}
    for path in folder.rglob('*'):
        if (path.is_file() and ('candidate' in path.name.lower() or 'ledger' in path.name.lower())
                and path.resolve() not in canonical):
            raise ValueError('Failed writer input has an additional or corrected candidate/ledger: ' + str(path))


def _support_observations(ledger):
    """Describe raw support/access inconsistencies without changing any label."""
    evidence = {row['evidence_id']: row for row in ledger['evidence']}
    observations = []
    for claim in ledger['claims']:
        for name, check in claim['checks'].items():
            if check['status'] != 'supported':
                continue
            refs = check['evidence_ids']
            supporting = [evidence[identity] for identity in refs if identity in evidence]
            accessible = any(row.get('access') == 'full_section' and
                             (name == 'citation' or row.get('primary') is True) for row in supporting)
            if not accessible:
                observations.append({'claim_id': claim['claim_id'], 'check': name,
                    'recorded_status': check['status'], 'evidence_ids': copy.deepcopy(refs),
                    'recorded_evidence_access': [{'evidence_id': row['evidence_id'],
                        'access': row['access'], 'primary': row['primary']} for row in supporting],
                    'interpretation': 'Frozen structural support/access invariant is not satisfied; no semantic relabeling performed.'})
    return observations


def _failure_state(root, protocol):
    spec = paired._document(protocol, FAILED_ID)
    folder = paired._relative(root, spec['path'])
    _require_no_downstream(folder)
    paths = {key: paired._relative(folder, name) for key, name in INPUT_PATHS.items()}
    # Missing, malformed or unrelated candidate errors are not this declared
    # ledger-validation terminal path. Both existing raw responses are required.
    records = {key: rep._read_json(path) for key, path in paths.items()}
    rep.validate_original(records['original'])
    rep.validate_repair(records['original'], records['candidate'])
    ops.execution_budget(records['candidate'], 18, 'failed shared writer')
    hashes = {key: rep._sha(path) for key, path in paths.items()}
    errors = {}
    for name, core in [('v5', v5), ('v6', v6)]:
        try:
            core.build_guard_packet(**records, input_file_sha256=hashes)
        except ValueError as error:
            errors[name] = {'exception_type': type(error).__name__, 'message': str(error)}
        else:
            raise ValueError('Writer inputs validate; cannot declare terminal input failure')
    if errors['v5'] != errors['v6']:
        raise ValueError('Frozen validators do not reproduce the same shared input failure')
    if not errors['v5']['message'].startswith('ledger.'):
        raise ValueError('Failure is outside this ledger-validation publication extension')
    return folder, paths, records, hashes, errors


def _declaration_outputs(root):
    root, protocol, runtime = _trial(root)
    folder, paths, records, hashes, errors = _failure_state(root, protocol)
    refusal_path = folder / REFUSAL_PATH
    refusal = rep._read_json(refusal_path)
    if (refusal.get('record_type') != 'semantic_retry_declined' or
            refusal.get('format_repair_performed') is not False or refusal.get('candidate_rewritten') is not False or
            type(refusal.get('additional_web_calls')) is not int or refusal['additional_web_calls'] != 0):
        raise ValueError('Writer refusal transcript must preserve the declined semantic retry without a rewritten candidate')
    for field in ('source', 'writer_statement', 'interpretation'):
        rep._string(refusal.get(field), 'writer_refusal.' + field, nonempty=True)
    target = root / 'terminal_failure.N2.json'
    terminal = {'schema_version': 1, 'document_id': FAILED_ID, 'cohort': 'new',
        'status': 'terminal_failed', 'failed_stage': 'shared_candidate_ledger_validation',
        'declared_at_utc': ops._timestamp(target, 'declared_at_utc'),
        'preregistration_sha256': rep._sha(root / 'preregistration.json'), 'frozen_runtime_sha256': runtime,
        'input_paths': {key: path.relative_to(root).as_posix() for key, path in paths.items()},
        'input_file_sha256': hashes, 'reproduced_validator_errors': errors,
        'writer_refusal': {'path': refusal_path.relative_to(root).as_posix(), 'raw_sha256': rep._sha(refusal_path),
            'record_type': refusal['record_type'],
            'provenance': 'Operator transcript of the writer statement; not human adjudication or a new semantic label.'},
        'attempted_input_counts': {'candidate_patches': len(records['candidate']['patches']),
            'ledger_claims': len(records['ledger']['claims']), 'ledger_blocks': len(records['ledger']['blocks'])},
        'structural_observations_not_semantic_reassessment': _support_observations(records['ledger']),
        'reason': FAILURE_REASON, 'semantic_retry_authorized': False,
        'candidate_correction_performed': False, 'comparative_downstream_present': False,
        'gate_veto_observed': False, 'paired_outcomes': None,
        'baseline_policy': 'Original-only baseline may finish and is displayed separately; it supplies no paired outcomes.'}
    amendment_path = root / 'publication_amendment.json'
    amendment = {'schema_version': 1, 'recorded_at_utc': ops._timestamp(amendment_path, 'recorded_at_utc'),
        'amendment_type': 'reporting_only_terminal_input_failure',
        'preregistration_sha256': rep._sha(root / 'preregistration.json'),
        'frozen_runtime_sha256': runtime, 'reporting_extension_sha256': _extension_hashes(),
        'terminal_failure_sha256': v5.sha256_json(terminal),
        'terminal_failure_file_sha256': hashlib.sha256(rep._json_bytes(terminal)).hexdigest(),
        'completed_pair_ids_required': list(COMPLETED_IDS), 'terminal_failed_document_ids': [FAILED_ID],
        'planned_document_ids': protocol['planned_document_ids'],
        'scoring_amended': False, 'experiment_retries_authorized': False,
        'frozen_protocol_runtime_instructions_and_raw_modified': False,
        'scope': 'Publish all four planned documents: three unchanged scored pairs and N2 as an explicit terminal failed input with null paired/arm metrics. Missing or invalid unrelated records remain blocking errors.',
        'interpretation': 'Reporting was extended after an observed writer-input failure. This is not a pre-specified successful experiment or an additional experimental arm; it changes no scored labels, denominator rules or gate outputs.',
        'new_cohort_coverage': {'planned': 2, 'paired_comparisons_required': 1, 'terminal_failed': 1},
        'preservation': 'Frozen citation_report.py still requires all four complete pairs and is not modified or used to bypass its completeness guard.'}
    return {target: rep._json_bytes(terminal), amendment_path: rep._json_bytes(amendment)}, terminal, amendment


def declare_terminal(root):
    outputs, terminal, _ = _declaration_outputs(root)
    rep._write_immutable(outputs)
    return {'terminal_record': str(Path(root).resolve() / 'terminal_failure.N2.json'),
            'status': terminal['status'], 'validator_error': terminal['reproduced_validator_errors']['v5'],
            'published': False}


def verify_terminal(root):
    outputs, terminal, amendment = _declaration_outputs(root)
    ops._intact(outputs, 'Terminal failure declaration/publication amendment')
    return terminal, amendment


def _completed(root, protocol):
    """Replay and ingest only already-produced, independently scored pairs."""
    _, hashes = paired.validate_preregistration(root)
    prepared, results_by_id = {}, {}
    for identity in COMPLETED_IDS:
        results, folder = ops.verify_pair(root, identity)
        post = rep._read_json(folder / 'evaluation/paired_post.raw.json')
        ops.execution_budget(post, 25, 'joint post')
        prepared.update(paired._post_ingest_outputs(root, protocol, paired._document(protocol, identity), hashes))
        prepared[folder / 'evaluation/detection_consistency.json'] = rep._json_bytes(ops.old._detection_consistency(post))
        results_by_id[identity] = results
    rep._write_immutable(prepared)
    documents, payloads, records, texts = {}, {}, {}, {}
    for identity in COMPLETED_IDS:
        spec = paired._document(protocol, identity)
        folder = paired._relative(root, spec['path'])
        doc, data = paired._document_record(root, protocol, spec, hashes)
        if doc['status'] != 'complete' or doc['missing'] or doc['validation_errors']:
            raise ValueError('Expected completed pair is missing or invalid: ' + identity)
        original = rep._read_json(folder / 'input/original_report.json')
        record = {'folder': folder, 'spec': spec, 'original': original,
            'candidate': rep._read_json(folder / 'shared/candidate.raw.json'),
            'guard': rep._read_json(folder / 'shared/guard.patch.raw.json'),
            'baseline': rep._read_json(folder / 'evaluation/baseline.raw.json'),
            'joint_post': rep._read_json(folder / 'evaluation/paired_post.raw.json'),
            'detection_consistency': rep._read_json(folder / 'evaluation/detection_consistency.json'),
            'round0': rep._read_json(folder / 'shared/reviews/round0/review.raw.json'),
            'round1': None, 'arms': {}}
        extra = folder / 'shared/reviews/round1/review.raw.json'
        if extra.is_file():
            record['round1'] = rep._read_json(extra)
        texts[identity] = {'original': original['text']}
        for arm, row in doc['arms'].items():
            base = folder / 'arms' / arm
            gate = rep._read_json(base / 'gated/gate.decisions.json')
            preview = rep._read_json(base / 'preview/gate.preview.json')
            revised = rep._read_json(base / 'revised_report.json')
            if gate != results_by_id[identity][arm] or any(revised[key] != gate['final_applied'][key] for key in ('text', 'sources')):
                raise ValueError('Scored final report differs from frozen gate replay')
            if (base / 'repair.raw.json').read_bytes() != (base / 'gated/repair.effective.json').read_bytes():
                raise ValueError('Effective repair differs from the scored repair')
            record['arms'][row['variant']] = {'arm': arm, 'folder': base, 'gate': gate,
                'preview': preview, 'revised': revised, 'post': rep._read_json(base / 'post.raw.json'),
                'linkage': rep._read_json(base / 'gate_linkage.json'),
                'eligible': report._ids(preview['decisions']), 'accepted': report._ids(gate['decisions'])}
            texts[identity][row['variant']] = revised['text']
        record['costs'] = report._review_costs(record)
        if spec['cohort'] == 'new' and not (folder / 'input/deep_research_completed.png').is_file():
            raise ValueError('Actual Deep Research completion screenshot missing: ' + identity)
        documents[identity], payloads[identity], records[identity] = doc, data, record
    return documents, payloads, records, texts


def _failed_document(root, protocol, terminal):
    spec = paired._document(protocol, FAILED_ID)
    folder, paths, raw, _, _ = _failure_state(root, protocol)
    baseline_path = folder / 'evaluation/baseline.raw.json'
    baseline = rep._read_json(baseline_path)
    rep.validate_baseline(raw['original'], baseline)
    ops.execution_budget(baseline, 25, 'terminal original-only baseline')
    frozen = rep._baseline_freeze(folder)
    if frozen is None:
        raise ValueError('Terminal document original-only baseline must be frozen before publication')
    if not (folder / 'input/deep_research_completed.png').is_file():
        raise ValueError('Terminal original Deep Research completion screenshot missing')
    mapping = paired._arm_mapping(protocol, spec)
    doc = {'id': FAILED_ID, 'cohort': 'new', 'path': spec['path'], 'status': 'terminal_failed',
        'paired': None, 'arms': {arm: {'arm': arm, 'variant': variant, 'status': 'terminal_failed',
            'metrics': None, 'comparative_outcome': None} for arm, variant in mapping.items()},
        'baseline_counts': rep._counts(baseline['issues'], 'status', rep.PROBLEM_STATUSES),
        'baseline_use': 'original-only descriptive record; no paired denominator or resolution/detection outcome',
        'terminal_failure': copy.deepcopy(terminal), 'missing': [], 'validation_errors': [],
        'unclassified_missing': False}
    data = {'folder': folder, 'spec': spec, 'original': raw['original'], 'candidate': raw['candidate'],
            'ledger': raw['ledger'], 'baseline': baseline, 'baseline_freeze': frozen, 'paths': paths,
            'writer_refusal': rep._read_json(folder / REFUSAL_PATH)}
    return doc, data


def _summary(protocol, documents, root):
    ordered = [documents[identity] for identity in protocol['planned_document_ids']]
    cohorts = {}
    for cohort in ('regression', 'new'):
        rows = [row for row in ordered if row['cohort'] == cohort]
        aggregate = paired._aggregate(rows)
        aggregate.update(terminal_failed_documents=sum(row['status'] == 'terminal_failed' for row in rows),
                         paired_metrics_scope='completed pairs only; failed documents retained without outcomes')
        cohorts[cohort] = aggregate
    return {'schema_version': 2, 'experiment_id': protocol['id'], 'human_adjudicated': False,
        'publication_mode': 'reporting_only_terminal_input_failure',
        'preregistration_sha256': rep._sha(root / 'preregistration.json'),
        'planned_document_ids': protocol['planned_document_ids'], 'completed_pair_ids': list(COMPLETED_IDS),
        'terminal_failed_document_ids': [FAILED_ID], 'unexpected_missing_document_ids': [],
        'whole_trial_complete': False, 'publication_accounts_for_all_planned_documents': True,
        'documents': ordered, 'cohorts': cohorts,
        'interpretation': ['The reporting-only extension follows a terminal writer-input validation failure; no experimental retry or scoring amendment was authorized.',
            'All four planned documents remain visible. N2 contributes no paired outcome, not zero failures, zero benefit, or a successful gate veto.',
            'Only one of two planned new documents completed a paired comparison. Its results do not estimate the missing N2 effect.',
            'Primary scoring and common-baseline exclusion rules remain exactly those of the frozen paired trial.',
            'Historical known-data regression is separate from fresh-document evidence.',
            'N2 original-only baseline is descriptive and never supplies paired detection, resolution, safety or gate metrics.'] + protocol['limits']}


def _manifest(root, summary, records, failed, terminal, amendment):
    documents = []
    for identity in summary['planned_document_ids']:
        record = failed if identity == FAILED_ID else records[identity]
        folder = record['folder']
        names = sorted(path.relative_to(folder).as_posix() for path in folder.rglob('*') if path.is_file())
        row = {'id': identity, 'cohort': record['spec']['cohort'],
            'status': 'terminal_failed' if identity == FAILED_ID else 'complete',
            'input_and_output_sha256': {name: rep._sha(folder / name) for name in names}}
        if identity == FAILED_ID:
            row.update(gate_summary=None, candidate_identity=None, ledger_identity=None,
                patch_judgment_identity=None, findings_identity=None, gate_replay_verified=None,
                effective_equals_scored=None, joint_audit_verified=None,
                writer_input_failure_reproduced=True, no_comparative_downstream_verified=True,
                web_calls={'shared_writer': record['candidate']['execution']['web_calls'],
                    'shared_patch_guard': None, 'joint_final_review': None, 'additional_v6_rereview': None,
                    'baseline': record['baseline']['execution']['web_calls'], 'joint_post': None},
                paired_outcomes=None, unrun_cost_fields='null denotes stage not executed, not a measured zero')
        else:
            row.update(gate_summary={variant: arm['gate']['summary'] for variant, arm in record['arms'].items()},
                candidate_identity=True, ledger_identity=True, patch_judgment_identity=True, findings_identity=True,
                gate_replay_verified=True, effective_equals_scored=True, joint_audit_verified=True,
                web_calls={'shared_writer': record['candidate']['execution']['web_calls'],
                    'shared_patch_guard': record['guard']['execution']['web_calls'],
                    'joint_final_review': record['costs']['initial_web_calls'],
                    'additional_v6_rereview': record['costs']['additional_web_calls'],
                    'baseline': record['baseline']['execution']['web_calls'],
                    'joint_post': record['joint_post']['execution']['web_calls']},
                detection_consistency=record['detection_consistency'],
                costs_reused_from_history=record['spec']['cohort'] == 'regression')
        documents.append(row)
    return {'schema_version': 1, 'publication_mode': summary['publication_mode'],
        'preregistration_sha256': summary['preregistration_sha256'],
        'publication_amendment_sha256': rep._sha(root / 'publication_amendment.json'),
        'terminal_failure_sha256': rep._sha(root / 'terminal_failure.N2.json'),
        'reporting_extension_sha256': amendment['reporting_extension_sha256'],
        'frozen_runtime_sha256': terminal['frozen_runtime_sha256'],
        'editorial_records_sha256': {path.name: rep._sha(path) for path in
            [root / 'review.json', *sorted(root.glob('*spotcheck*.json'))]},
        'editorial_cost_policy': 'Root editorial spotchecks are separate from frozen experimental worker budgets and are not merged into those stage counts.',
        'completed_pair_ids': list(COMPLETED_IDS), 'terminal_failed_document_ids': [FAILED_ID],
        'whole_trial_complete': False, 'publication_accounts_for_all_planned_documents': True,
        'documents': documents, 'limits': summary['interpretation']}


def _link(path, output, label):
    return f"<a href='{report.esc(rep._artifact_link(Path(path), output))}'>{report.esc(label)}</a>"


def _pre(value):
    return '<pre>' + report.esc(rep._json_bytes(value).decode('utf-8')) + '</pre>'


def _banner():
    return "<div class='notice'><strong>新規2本中1本のみ比較完了。N2は入力検証で終了しました。</strong><p>計画4本は既知回帰2本＋新規2本。R1/R2/N1の3ペアを採点し、N2の比較結果は欠測（null）として明示します。N2は成功した全文留保でも、効果・有害変更が0件という結果でもありません。入力を意味上修復する再試行や文書の差替えは行っていません。</p></div>"


def _failure_html(root, output, failed, terminal):
    rows = [[row['claim_id'], row['check'], row['recorded_status'], ', '.join(row['evidence_ids']) or 'なし',
             str(row['recorded_evidence_access'])] for row in terminal['structural_observations_not_semantic_reassessment']]
    parts = ["<section id='terminal-failure'><h2>N2：入力検証の終了記録（比較結果なし）</h2>",
        '<p>' + report.esc(terminal['reason']) + '</p>',
        '<p>v5/v6に共通の入力検証が同じ例外を再現しました。guard作成・全文確認・ゲート適用・共同事後評価は開始していません。以下は保存されたラベルと根拠アクセス状態の構造的な不一致で、主張の意味を再採点したものではありません。</p>',
        '<p>入力として作成された候補・主張・ブロックの件数です。採用・棄却数や事実誤りの件数ではありません。</p>' + _pre(terminal['attempted_input_counts']),
        _pre(terminal['reproduced_validator_errors']),
        report._table(['claim ID', '軸', '記録ラベル', '根拠ID', '保存された取得状態'], rows),
        '<p>原文だけのbaselineは完了していますが、検出率・解消率・安全性の比較母数には入れません。候補は未採点の提案として保存しています。</p>',
        report._table(['段階', 'Web呼出・記録'], [['失敗した共有作成', failed['candidate']['execution']['web_calls']],
            ['原文のみbaseline', failed['baseline']['execution']['web_calls']], ['patch guard・全文確認・共同事後評価', '未実施（null）']]),
        '<p>' + ' · '.join(_link(path, output, name) for name, path in failed['paths'].items()) + ' · ' +
        _link(failed['folder'] / 'evaluation/baseline.raw.json', output, 'N2 original-only baseline') + '</p>',
        '<p>' + _link(failed['folder'] / REFUSAL_PATH, output, 'writer semantic-retry refusal（operator transcript）') +
        '：作成者が意味を変える修復を行わなかった旨の転記です。人間による事実判定ではありません。</p>' +
        '<details><summary>作成者の拒否記録（原文の転記）</summary>' + _pre(failed['writer_refusal']) + '</details>',
        '<details><summary>N2 原文全文・出典</summary>' + _pre(failed['original']) + '</details>',
        '<details><summary>N2 baseline全inventory・issues・controls（原文のみ）</summary>' + _pre(failed['baseline']) + '</details>',
        '<details><summary>N2 失敗した候補・台帳の全raw</summary>' + _pre(failed['candidate']) + _pre(failed['ledger']) + '</details></section>']
    return '\n'.join(parts)


def _cohort_html(cohort, documents):
    title = '既知回帰：2/2ペア完了' if cohort == 'regression' else '新規：1/2ペア完了、N2は入力検証で終了'
    rows, safety, unchanged = [], [], []
    for doc in documents.values():
        if doc['cohort'] != cohort:
            continue
        if doc['status'] == 'terminal_failed':
            rows.append([doc['id'], '入力検証で終了', '—', '—', '—', '—', '—'])
            safety.append([doc['id'], '比較なし', '—', '—', '—', '—', '—'])
            continue
        primary = doc['paired']['primary']
        if all(arm['metrics']['changed_locations'] == 0 for arm in doc['arms'].values()):
            unchanged.append(doc['id'])
        for variant, label in report.VARIANTS:
            count = primary[variant]
            metric = next(row['metrics'] for row in doc['arms'].values() if row['variant'] == variant)
            rows.append([doc['id'], label, primary['denominator'], count['detected'], count['resolved'],
                         count['after_status']['partially_resolved'], primary['denominator'] - count['resolved']])
            safety.append([doc['id'], label, metric['changed_locations'], metric['patch_reviews']['acceptable'],
                           metric['patch_reviews']['unnecessary_change'], metric['patch_reviews']['regression'],
                           metric['new_problems']['confirmed_problem']])
    unchanged_note = ('<div class="notice">' + report.esc(', '.join(unchanged)) +
        'は両群とも最終的な変更がありません。無変更に伴う新問題0という評価は、安全に有用な訂正を届けたことの実証ではありません。</div>') if unchanged else ''
    return (f"<section id='{cohort}'><h2>{title}</h2>" + unchanged_note +
        report._table(['文書', '状態／ゲート', '共通母数', '発見', '完全解消', '部分解消', '残存'], rows) +
        report._table(['文書', '状態／ゲート', '最終採用', 'acceptable', '不要変更', '改悪', '確認済み新問題'], safety) +
        '<p>—は未観測であり0ではありません。集計は比較完了ペアだけを対象とし、新規N2の失敗を除外して2本完了と称しません。acceptableは有用な訂正数と同義ではありません。新問題・改悪・対照損失は重なり得るため合算しません。完全解消はresolvedのみです。</p></section>')


def _index_html(root, output, protocol, summary, documents, records, failed, terminal):
    parts = ["<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>全記録：{report.esc(protocol['report_title'])}</title><style>{report.STYLE}</style><body><main>",
        '<h1>全4文書の状態・完全な採点台帳</h1>', _banner(),
        '<p>' + _link(output / 'report.html', output, '比較レポート') + ' · ' + _link(output / 'summary.json', output, 'summary.json') + ' · ' +
        _link(output / 'terminal_manifest.json', output, 'terminal manifest') + '</p>',
        _cohort_html('new', documents), _cohort_html('regression', documents),
        '<details><summary>計画・完了・終了の集計</summary>' + _pre(summary['cohorts']) + '</details>',
        _failure_html(root, output, failed, terminal)]
    for identity, record in records.items():
        parts.append(f"<section id='{identity}'><h2>{identity}：比較完了の全記録</h2>" + _pre(documents[identity]['paired']))
        parts.append('<details><summary>原文・出典・baseline全文</summary>' + _pre(record['original']) + _pre(record['baseline']) + '</details>')
        for variant, label in report.VARIANTS:
            arm = record['arms'][variant]
            delta = '\n'.join(difflib.unified_diff(record['original']['text'].splitlines(), arm['revised']['text'].splitlines(),
                                                  fromfile='original', tofile=variant, lineterm=''))
            parts.append('<h3>' + report.esc(label) + '</h3><details><summary>最終全文・出典・差分</summary>' + _pre(arm['revised']) +
                         '<pre>' + report.esc(delta or '変更なし') + '</pre></details>')
            parts.append('<details><summary>全findings・patches・採点・対照・誤指摘・事後発見</summary>' +
                         _pre(rep._read_json(arm['folder'] / 'repair.raw.json')) + _pre(arm['post']) + '</details>')
        parts.append('</section>')
    parts.append('</main></body></html>')
    return '\n'.join(parts).encode('utf-8')


def _report_html(root, output, protocol, summary, documents, records, failed, terminal, review):
    parts = ["<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{report.esc(protocol['report_title'])}</title><style>{report.STYLE}</style></head><body><main>",
        f"<div class='eyebrow'>{report.esc(protocol['report_date'])} · 既知2本＋新規1本の比較完了／新規1本の入力終了 · AI暫定評価</div>",
        f"<h1>{report.esc(protocol['report_heading'])}</h1>", _banner(),
        "<nav><a href='#new'>新規結果</a><a href='#terminal-failure'>N2終了記録</a><a href='#regression'>既知回帰</a><a href='#process'>除外と費用</a><a href='#cases'>具体例</a><a href='#records'>原資料</a></nav>",
        '<h2>' + report.esc(review['headline']) + '</h2>' + report.paragraphs(review['verdict']),
        '<h2>設計と報告上の変更</h2>' + report.paragraphs(review['design']),
        '<p>凍結後に観測したN2の失敗を明示するため、報告機能だけを追加しました。凍結済みの実装・指示・raw・採点基準を変更せず、意味を変える再試行を認めていません。</p>']
    for cohort in ('new', 'regression'):
        parts.extend([_cohort_html(cohort, documents), report.paragraphs(review[cohort + '_notes'])])
    parts.extend([_failure_html(root, output, failed, terminal),
        '<h2>以下のゲート過程と候補一覧は比較完了の3文書だけ</h2>',
        report._process_html({key: documents[key] for key in COMPLETED_IDS}, records),
        report._candidates_html(report._candidate_rows(records)), "<section id='cases'><h2>比較完了文書の完全一致例</h2>"])
    for case in review['cases']:
        parts.append("<article class='card'><h3>" + report.esc(case['document_id'] + '：' + case['title']) + "</h3><div class='triple'>")
        for variant, label in [('original', '原文'), *report.VARIANTS]:
            parts.append('<div><strong>' + report.esc(label) + "</strong><div class='quote'>" + report.esc(case[variant]) + '</div></div>')
        parts.append('</div>' + report.paragraphs(case['assessment']) + '<ul>')
        parts.extend('<li>' + report.source_link(row['url'], row['title']) + '</li>' for row in case['sources'])
        parts.append('</ul><p>記録ID：' + report.esc(', '.join(case['record_ids'])) + '</p></article>')
    guard_notes = review.get('guard_notes', [])
    parts.extend(['</section>', report.paragraphs([guard_notes] if isinstance(guard_notes, str) else guard_notes),
        '<h2>次の判断</h2>', report.paragraphs(review['recommendations']),
        '<h2>限界</h2>', report.paragraphs(review['limitations']), report.paragraphs(summary['interpretation']),
        "<section id='records'><h2>全資料・凍結・失敗raw・報告変更</h2><p>"])
    if list(root.glob('*spotcheck*.json')):
        parts.append('編集時の出典spotcheckは実験workerの凍結予算と別の確認です。以下の独立記録に保存し、上記の実験段階別呼出数に混入させません。 ')
    for name in ('index.html', 'summary.json', 'terminal_manifest.json', 'publication_amendment.json', 'terminal_failure.N2.json'):
        parts.append(_link(output / name, output, name) + ' · ')
    parts.append('</p><details><summary>全raw・出典・指示・プロトコル・ハッシュ記録</summary><ul>')
    for path in sorted(root.rglob('*')):
        if path == root / 'terminal_operator_manifest.json':
            continue
        if path.is_file() and path.suffix.lower() in {'.json', '.txt', '.py', '.png', '.md', '.html'}:
            parts.append('<li>' + _link(path, output, path.relative_to(root).as_posix()) + '</li>')
    parts.append('</ul></details>')
    for identity, record in {**records, FAILED_ID: failed}.items():
        parts.append('<details><summary>' + report.esc(identity) + '：保存された出典・生成記録</summary><ul>')
        sources = list(record['original']['sources'])
        if identity in records:
            for arm in record['arms'].values():
                sources.extend(arm['revised']['sources'])
        seen = set()
        for source in sources:
            if source['url'] not in seen:
                parts.append('<li>' + report.source_link(source['url'], source.get('title') or source['url']) + '</li>')
                seen.add(source['url'])
        if record['original'].get('chat_url'):
            parts.append('<li>' + report.source_link(record['original']['chat_url'], 'Deep Research元チャット') + '</li>')
        parts.append('</ul>')
        screenshot = record['folder'] / 'input/deep_research_completed.png'
        if screenshot.is_file():
            parts.append("<img loading='lazy' src='" + report.esc(rep._artifact_link(screenshot, output)) + "' alt='" + report.esc(identity) + " 実Deep Research完成画面'>")
        parts.append('</details>')
    parts.append('</section>')
    parts.append(f'</main><script>{report.FILTER_SCRIPT}</script></body></html>')
    return '\n'.join(parts).encode('utf-8')


def publish(root, output):
    root, protocol, _ = _trial(root)
    output = Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError('Publication output must be outside the frozen trial root')
    terminal, amendment = verify_terminal(root)
    failed_doc, failed = _failed_document(root, protocol, terminal)
    documents, _, records, texts = _completed(root, protocol)
    review = rep._read_json(root / 'review.json')
    report._validate_cases(review, texts)
    documents[FAILED_ID] = failed_doc
    summary = _summary(protocol, documents, root)
    manifest = _manifest(root, summary, records, failed, terminal, amendment)
    outputs = {output / 'summary.json': rep._json_bytes(summary),
        output / 'terminal_manifest.json': rep._json_bytes(manifest),
        output / 'operator_manifest.json': rep._json_bytes(manifest),
        root / 'terminal_operator_manifest.json': rep._json_bytes(manifest),
        output / 'index.html': _index_html(root, output, protocol, summary, documents, records, failed, terminal),
        output / 'report.html': _report_html(root, output, protocol, summary, documents, records, failed, terminal, review),
        output / 'review.json': (root / 'review.json').read_bytes(),
        output / 'terminal_failure.N2.json': (root / 'terminal_failure.N2.json').read_bytes(),
        output / 'publication_amendment.json': (root / 'publication_amendment.json').read_bytes()}
    rep._write_immutable(outputs)
    return {'report': str(output / 'report.html'), 'index': str(output / 'index.html'),
            'completed_pairs': 3, 'planned_pairs': 4, 'new_completed_pairs': 1, 'new_planned_pairs': 2,
            'terminal_failed_document_ids': [FAILED_ID], 'whole_trial_complete': False,
            'publication_accounts_for_all_planned_documents': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['declare-terminal', 'verify-terminal', 'publish'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'declare-terminal':
            result = declare_terminal(args.root)
        elif args.command == 'verify-terminal':
            result = {'terminal_record': verify_terminal(args.root)[0], 'verified': True}
        else:
            if args.output is None:
                parser.error('--output is required for publish')
            result = publish(args.root, args.output)
    except (ValueError, OSError, TypeError, KeyError, UnicodeError) as error:
        parser.exit(2, f'Publication not generated: {error}\n')
    print(rep._json_bytes(result).decode('utf-8'), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
