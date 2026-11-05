"""Immutable orchestration for two gates sharing one candidate and patch judgment.

This helper never retrieves evidence, supplies semantic judgments or rewrites a
worker response. All timestamps written here are actual operator UTC times.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path

import replication as rep
import paired_trial as paired
from factcheck import claim_guard as v3

DEFAULT_ROOT = Path('pilot/omission_gate_20261102')
DEFAULT_OUTPUT = Path('runs/omission_gate_20261102')
REVIEW_FIELDS = {'status', 'coverage_complete', 'reason', 'accepted_patch_ids',
                 'final_text_sha256_utf8', 'final_sources_sha256'}
GUARD_FIELDS = {'schema_version', 'guard_packet_sha256', 'input_sha256',
                'evidence', 'patch_reviews', 'execution'}


def _timestamp(path, field):
    return (rep._read_json(path)[field] if path.exists()
            else datetime.now(timezone.utc).isoformat())


def _intact(outputs, label):
    for path, payload in outputs.items():
        if not path.exists() or path.read_bytes() != payload:
            raise ValueError(f'{label} missing or changed: {path}')


def runtime_check(root):
    root = Path(root).resolve()
    protocol, _ = paired.validate_preregistration(root)
    prereg = rep._read_json(root / 'preregistration.json')
    runtime = rep._strings(prereg.get('runtime_files'), 'preregistration.runtime_files')
    if not runtime or len(set(runtime)) != len(runtime):
        raise ValueError('Runtime snapshot list must be nonempty and unique')
    checked = {}
    for name in runtime:
        snapshot = paired._relative(root, 'code_snapshots/' + name)
        actual = rep._sha(Path(name))
        if actual != rep._sha(snapshot):
            raise ValueError('Runtime differs from frozen snapshot: ' + name)
        checked[name] = actual
    if protocol.get('gate_versions') != {'current': 'v3', 'improved': 'v4'}:
        raise ValueError('Unexpected frozen gate assignment')
    if (root / 'instructions/repair_v1.txt').read_bytes() != (root / 'instructions/repair_v2.txt').read_bytes():
        raise ValueError('Shared writer instruction identity failed')
    return protocol, checked


def doc_info(root, identity):
    root = Path(root).resolve()
    protocol, hashes = runtime_check(root)
    doc = paired._document(protocol, identity)
    return protocol, doc, paired._relative(root, doc['path']), hashes


def execution_budget(record, limit, label):
    rep._execution(record, label)
    count = record['execution']['web_calls']
    if count > limit:
        raise ValueError(f'{label} exceeds frozen Web call cap: {count}>{limit}')
    return count


def _core(variant):
    return v3 if variant == 'current' else importlib.import_module('factcheck.claim_guard_v4')


def _shared(folder):
    paths = {'original': folder / 'input/original_report.json',
             'candidate': folder / 'shared/candidate.raw.json',
             'ledger': folder / 'shared/ledger.raw.json'}
    records = {key: rep._read_json(path) for key, path in paths.items()}
    raw = {key: path.read_bytes() for key, path in paths.items()}
    hashes = {key: rep._sha(path) for key, path in paths.items()}
    execution_budget(records['candidate'], 18, 'shared candidate')
    packet = v3.build_guard_packet(**records, input_file_sha256=hashes)
    other = _core('improved').build_guard_packet(**records, input_file_sha256=hashes)
    if packet != other:
        raise ValueError('Gate implementations did not construct identical shared inputs')
    return records, raw, hashes, packet


def _guard_inputs(folder, raw, packet):
    base = folder / 'shared/guard_input'
    return {
        base / 'original.frozen.json': raw['original'],
        base / 'candidate.raw.json': raw['candidate'], base / 'ledger.raw.json': raw['ledger'],
        base / 'guard_packet.json': rep._json_bytes(packet),
        base / 'candidate.applied.json': rep._json_bytes(packet['candidate_applied']),
        base / 'input_freeze.json': rep._json_bytes({
            'schema_version': 1, 'input_file_sha256': packet['input_file_sha256'],
            'input_sha256': packet['input_sha256'],
            'guard_packet_sha256': v3.sha256_json(packet)}),
    }


def prepare_shared_guard(root, identity):
    _, _, folder, _ = doc_info(root, identity)
    _, raw, hashes, packet = _shared(folder)
    rep._write_immutable(_guard_inputs(folder, raw, packet))
    return {'document': identity, 'guard_packet': str(folder / 'shared/guard_input/guard_packet.json'),
            'input_file_sha256': hashes, 'guard_packet_sha256': v3.sha256_json(packet)}


def _patch_guard(folder):
    path = folder / 'shared/guard.patch.raw.json'
    guard = rep._read_json(path)
    if set(guard) != GUARD_FIELDS:
        raise ValueError('Shared patch guard must contain the v3 patch fields plus execution, without final_review')
    execution_budget(guard, 4, 'shared patch guard')
    return guard


def _patch_context(candidate, ledger, guard):
    # No gate reasons, gate labels or necessity/acceptance judgments enter the
    # final judge's shared context. Source excerpts remain reviewer attestations.
    patch_fields = ('patch_id', 'before', 'after', 'source_urls', 'claim_ids', 'depends_on_patch_ids')
    claim_fields = ('claim_id', 'block_id', 'quote', 'context_block_ids', 'report_scope', 'source_scope')
    return {
        'candidate_patches': [{key: copy.deepcopy(row[key]) for key in patch_fields if key in row}
                              for row in candidate['patches']],
        'claim_context': [{key: copy.deepcopy(row[key]) for key in claim_fields}
                          for row in ledger['claims']],
        'evidence': {'ledger': copy.deepcopy(ledger['evidence']), 'patch_review': copy.deepcopy(guard['evidence'])},
        'unit_context': [{'patch_id': row['patch_id'],
                          'units': [{key: copy.deepcopy(unit[key]) for key in
                                     ('unit_id', 'claim_ids', 'before_quote', 'after_quote', 'evidence_ids')}
                                    for unit in row['units']],
                          'additions': [{key: copy.deepcopy(unit[key]) for key in
                                         ('after_quote', 'evidence_ids', 'report_scope', 'source_scope') if key in unit}
                                        for unit in row['additions']]}
                         for row in guard['patch_reviews']],
    }


def _preview_outputs(root, identity):
    protocol, doc, folder, runtime = doc_info(root, identity)
    records, raw, hashes, packet = _shared(folder)
    _intact(_guard_inputs(folder, raw, packet), 'Shared guard input freeze')
    guard = _patch_guard(folder)
    mapping = paired._arm_mapping(protocol, doc)
    previews, arms, outputs = {}, {}, {}
    for arm in ('A', 'B'):
        result = _core(mapping[arm]).preview_repair(**records, guard=guard, guard_packet=packet)
        previews[arm] = result
        selected = {**copy.deepcopy(records['original']),
                    **{key: copy.deepcopy(result['selected_applied'][key]) for key in ('text', 'sources')}}
        base = folder / 'arms' / arm / 'preview'
        for name, value in [('gate.preview.json', result), ('selected_report.json', selected),
                            ('selection.json', result['selection'])]:
            outputs[base / name] = rep._json_bytes(value)
        arms[arm] = {'selected_report': selected, 'selection': {
            key: copy.deepcopy(result['selection'][key]) for key in
            ('accepted_patch_ids', 'final_text_sha256_utf8', 'final_sources_sha256')}}
    joint = {'schema_version': 1, 'document_id': identity,
             'original': copy.deepcopy(records['original']),
             'shared_context': _patch_context(records['candidate'], records['ledger'], guard),
             'arms': arms,
             'identical_selections': arms['A'] == arms['B']}
    packet_path = folder / 'shared/final_review_packet.json'
    freeze_path = folder / 'shared/final_review.freeze.json'
    freeze = {
        'schema_version': 1, 'document_id': identity,
        'frozen_at_utc': _timestamp(freeze_path, 'frozen_at_utc'),
        'input_file_sha256': hashes, 'runtime_sha256': runtime,
        'guard_patch_raw_sha256': rep._sha(folder / 'shared/guard.patch.raw.json'),
        'shared_guard_semantic_sha256': v3.sha256_json(guard),
        'preview_sha256': {arm: v3.sha256_json(previews[arm]) for arm in ('A', 'B')},
        'final_review_packet_sha256': v3.sha256_json(joint),
        'preregistration_sha256': rep._sha(Path(root) / 'preregistration.json'),
    }
    outputs[packet_path] = rep._json_bytes(joint)
    outputs[freeze_path] = rep._json_bytes(freeze)
    outputs[folder / 'shared/guard.patch.frozen.json'] = (folder / 'shared/guard.patch.raw.json').read_bytes()
    return outputs, records, hashes, packet, guard, previews, joint, mapping, runtime, folder


def preview_pair(root, identity):
    outputs, _, _, _, _, previews, joint, _, _, folder = _preview_outputs(root, identity)
    rep._write_immutable(outputs)
    return {'document': identity, 'final_review_packet': str(folder / 'shared/final_review_packet.json'),
            'identical_selections': joint['identical_selections'],
            'selection': {arm: preview['selection'] for arm, preview in previews.items()}}


def _validate_final(response, previews, identical):
    if set(response) != {'execution', 'reviews'}:
        raise ValueError('Joint final review must contain exactly execution and reviews')
    count = execution_budget(response, 3, 'joint final review')
    reviews = rep._object(response['reviews'], 'final.reviews')
    rep._all_ids(reviews, {'A', 'B'}, 'final.reviews')
    for arm, review in reviews.items():
        if set(rep._object(review, 'final review ' + arm)) != REVIEW_FIELDS:
            raise ValueError('Final review fields differ from the frozen schema for ' + arm)
        rep._enum(review['status'], {'pass', 'fail'}, 'final.status')
        rep._boolean(review['coverage_complete'], 'final.coverage_complete')
        rep._string(review['reason'], 'final.reason', nonempty=True)
        selection = previews[arm]['selection']
        ids = rep._id_refs(review['accepted_patch_ids'], selection['accepted_patch_ids'], 'final.accepted_patch_ids')
        rep._all_ids(ids, selection['accepted_patch_ids'], 'final.accepted_patch_ids')
        for field in ('final_text_sha256_utf8', 'final_sources_sha256'):
            if review[field] != selection[field]:
                raise ValueError('Final review selection hash mismatch: ' + arm + '/' + field)
    if identical and reviews['A'] != reviews['B']:
        raise ValueError('Identical selections require one identical final-review object reused for A and B')
    return count


def _application_outputs(root, identity):
    prepared = _preview_outputs(root, identity)
    outputs, records, hashes, packet, patch_guard, previews, joint, mapping, runtime, folder = prepared
    _intact(outputs, 'Paired preview freeze')
    response_path = folder / 'shared/final_review.raw.json'
    response = rep._read_json(response_path)
    final_calls = _validate_final(response, previews, joint['identical_selections'])
    patch_calls = execution_budget(patch_guard, 4, 'shared patch guard')
    candidate_calls = execution_budget(records['candidate'], 18, 'shared candidate')
    if patch_calls + final_calls > 7:
        raise ValueError('Combined shared verification budget exceeds 7')
    outputs, results, arm_guards = {}, {}, {}
    for arm in ('A', 'B'):
        guard = copy.deepcopy(patch_guard)
        guard['final_review'] = copy.deepcopy(response['reviews'][arm])
        if mapping[arm] == 'improved':
            guard['final_review']['gate_version'] = 'v4'
        arm_guards[arm] = guard
        result = _core(mapping[arm]).gate_repair(**records, guard=guard, guard_packet=packet)
        results[arm] = result
        if result['selection'] != previews[arm]['selection']:
            raise ValueError('Final application did not retain the frozen preview selection')
        base = folder / 'arms' / arm
        effective = rep._json_bytes(result['effective_repair'])
        outputs[base / 'guard.raw.json'] = rep._json_bytes(guard)
        outputs[base / 'repair.raw.json'] = effective
        outputs[base / 'gated/repair.effective.json'] = effective
        for name, value in [('gate.decisions.json', result), ('final.applied.json', result['final_applied']),
                            ('summary.json', result['summary']), ('final_report.json',
                             {key: result['final_applied'][key] for key in ('text', 'sources')})]:
            outputs[base / 'gated' / name] = rep._json_bytes(value)
        linkage_path = base / 'gate_linkage.json'
        linkage = {
            'recorded_at_utc': _timestamp(linkage_path, 'recorded_at_utc'),
            'gate_version': protocol_version(mapping[arm]), 'input_file_sha256': hashes,
            'runtime_sha256': runtime, 'shared_guard_semantic_sha256': v3.sha256_json(patch_guard),
            'shared_guard_raw_sha256': rep._sha(folder / 'shared/guard.patch.raw.json'),
            'joint_final_review_raw_sha256': rep._sha(response_path),
            'final_review_packet_sha256': rep._sha(folder / 'shared/final_review_packet.json'),
            'final_review_freeze_sha256': rep._sha(folder / 'shared/final_review.freeze.json'),
            'candidate_web_calls': candidate_calls, 'patch_guard_web_calls': patch_calls,
            'joint_final_review_web_calls': final_calls, 'shared_verification_web_calls': patch_calls + final_calls,
            'effective_repair_sha256': hashlib.sha256(effective).hexdigest(),
            'scored_repair_sha256': hashlib.sha256(effective).hexdigest(),
            'effective_equals_scored': True,
        }
        outputs[linkage_path] = rep._json_bytes(linkage)
    semantics = [{key: value for key, value in guard.items() if key != 'final_review'}
                 for guard in arm_guards.values()]
    if semantics != [patch_guard, patch_guard]:
        raise ValueError('Per-arm patch judgments differ from the shared raw judgment')
    if any(result['effective_repair']['findings'] != records['candidate']['findings']
           for result in results.values()):
        raise ValueError('Gate filtering changed writer findings')
    outputs[folder / 'shared/final_review.frozen.json'] = response_path.read_bytes()
    return outputs, results, folder


def protocol_version(variant):
    return {'current': 'v3', 'improved': 'v4'}[variant]


def apply_pair(root, identity):
    outputs, results, _ = _application_outputs(root, identity)
    rep._write_immutable(outputs)
    return {'document': identity, 'arms': {arm: result['summary'] for arm, result in results.items()},
            'candidate_identity': True, 'patch_judgment_identity': True, 'findings_identity': True}


def verify_pair(root, identity):
    outputs, results, folder = _application_outputs(root, identity)
    _intact(outputs, 'Gate application replay')
    return results, folder


def freeze_baseline(root, identity):
    _, _, folder, _ = doc_info(root, identity)
    path = folder / 'evaluation/baseline.raw.json'
    baseline = rep._read_json(path)
    rep.validate_baseline(rep._read_json(folder / 'input/original_report.json'), baseline)
    execution_budget(baseline, 25, 'baseline')
    target = folder / 'evaluation/baseline.freeze.json'
    record = {'frozen_at_utc': _timestamp(target, 'frozen_at_utc'), 'sha256': rep._sha(path)}
    rep._write_immutable({target: rep._json_bytes(record)})
    rep._baseline_freeze(folder)
    return {'document': identity, 'baseline_sha256': record['sha256']}


def prepare_post(root, identity):
    _, folder = verify_pair(root, identity)
    if rep._baseline_freeze(folder) is None:
        raise ValueError('Freeze the original-only baseline before preparing any post packet')
    baseline = rep._read_json(folder / 'evaluation/baseline.raw.json')
    execution_budget(baseline, 25, 'baseline')
    for arm in ('A', 'B'):
        paired.prepare(root, identity, arm)
    return paired.prepare_post(root, identity)


def _detection_consistency(post):
    rows = {arm: {row['issue_id']: row for row in post['arms'][arm]['issues']} for arm in ('A', 'B')}
    disagreements, id_disagreements = [], []
    for identity in rows['A']:
        a, b = rows['A'][identity], rows['B'][identity]
        # The schema stores explicit detected booleans. Do not silently replace
        # conflicting semantic judgments, even when shared findings imply equality.
        if a['detected'] != b['detected']:
            disagreements.append({'issue_id': identity, 'A': a['detected'], 'B': b['detected']})
        if set(a.get('finding_ids', [])) != set(b.get('finding_ids', [])):
            id_disagreements.append({'issue_id': identity, 'A': a.get('finding_ids', []),
                                     'B': b.get('finding_ids', [])})
    return {'same_findings_require_same_detection': True,
            'consistent': not disagreements and not id_disagreements,
            'disagreements': disagreements, 'finding_id_disagreements': id_disagreements,
            'interpretation': 'Any difference is evaluator inconsistency, not a gate detection effect.'}


def ingest_post(root, identity, response=None):
    _, _, folder, _ = doc_info(root, identity)
    verify_pair(root, identity)
    source = Path(response).resolve() if response is not None else folder / 'evaluation/paired_post.raw.json'
    post = rep._read_json(source)
    execution_budget(post, 25, 'joint post')
    result = paired.ingest_post(root, identity, response)
    consistency = _detection_consistency(post)
    rep._write_immutable({folder / 'evaluation/detection_consistency.json': rep._json_bytes(consistency)})
    return {**result, 'detection_consistency': consistency}


def manifest(root, output=DEFAULT_OUTPUT):
    root, output = Path(root).resolve(), Path(output).resolve()
    protocol, hashes = runtime_check(root)
    docs = []
    for doc in protocol['documents']:
        identity = doc['id']
        results, folder = verify_pair(root, identity)
        ingest = ingest_post(root, identity)
        original = rep._read_json(folder / 'input/original_report.json')
        exports = {folder / 'input/original_report.txt': original['text'].encode('utf-8')}
        raw_names = ['input/original_report.json', 'shared/candidate.raw.json', 'shared/ledger.raw.json',
                     'shared/guard.patch.raw.json', 'shared/final_review.raw.json',
                     'evaluation/baseline.raw.json', 'evaluation/baseline.freeze.json',
                     'evaluation/paired_post.raw.json', 'evaluation/paired_post.freeze.json',
                     'evaluation/paired_post.ingest.json', 'evaluation/detection_consistency.json']
        for arm in ('A', 'B'):
            revised = rep._read_json(folder / 'arms' / arm / 'revised_report.json')
            exports[folder / 'arms' / arm / 'revised_report.txt'] = revised['text'].encode('utf-8')
            raw_names += [f'arms/{arm}/repair.raw.json', f'arms/{arm}/guard.raw.json',
                          f'arms/{arm}/gate_linkage.json', f'arms/{arm}/gated/gate.decisions.json']
        rep._write_immutable(exports)
        docs.append({'id': identity, 'input_and_output_sha256': {name: rep._sha(folder / name) for name in raw_names},
                     'gate_summary': {arm: result['summary'] for arm, result in results.items()},
                     'candidate_identity': True, 'ledger_identity': True, 'patch_judgment_identity': True,
                     'findings_identity': True, 'gate_replay_verified': True,
                     'effective_equals_scored': True, 'joint_audit_verified': True,
                     'web_calls': {label: execution_budget(rep._read_json(folder / name), cap, label)
                                   for label, name, cap in
                                   [('shared_writer', 'shared/candidate.raw.json', 18),
                                    ('shared_patch_guard', 'shared/guard.patch.raw.json', 4),
                                    ('joint_final_review', 'shared/final_review.raw.json', 3),
                                    ('baseline', 'evaluation/baseline.raw.json', 25),
                                    ('joint_post', 'evaluation/paired_post.raw.json', 25)]},
                     'detection_consistency': ingest['detection_consistency']})
    target = root / 'operator_manifest.json'
    value = {'created_at_utc': _timestamp(target, 'created_at_utc'), 'runtime_sha256': hashes,
             'preregistration_sha256': rep._sha(root / 'preregistration.json'), 'documents': docs,
             'limits': ['Hashes establish content identity, not correctness or actual worker isolation.',
                        'Web counts are execution attestations; parent logs may provide additional evidence.',
                        'Shared findings fix detection; report evaluator disagreement separately.']}
    rep._write_immutable({target: rep._json_bytes(value)})
    report = paired.build_report(root, output)
    rep._write_immutable({output / 'operator_manifest.json': rep._json_bytes(value)})
    return {'manifest': str(target), 'report': str(output / 'index.html'),
            'document_status': {doc['id']: doc['status'] for doc in report['documents']},
            'detection_consistent': all(doc['detection_consistency']['consistent'] for doc in docs)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare-shared-guard', 'prepare-guard', 'preview-pair',
                                          'apply-pair', 'freeze-baseline', 'prepare-post', 'ingest-post', 'manifest'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--doc', choices=['N1', 'N2'])
    parser.add_argument('--response', type=Path)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        if args.command == 'manifest':
            result = manifest(args.root, args.output)
        elif not args.doc:
            parser.error('--doc is required for document operations')
        elif args.command == 'ingest-post':
            result = ingest_post(args.root, args.doc, args.response)
        else:
            operations = {'prepare-shared-guard': prepare_shared_guard, 'prepare-guard': prepare_shared_guard,
                          'preview-pair': preview_pair, 'apply-pair': apply_pair,
                          'freeze-baseline': freeze_baseline, 'prepare-post': prepare_post}
            result = operations[args.command](args.root, args.doc)
    except (ValueError, OSError, TypeError, KeyError, UnicodeError) as error:
        parser.exit(2, f'Validation failed: {error}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
