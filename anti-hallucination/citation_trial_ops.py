"""Immutable, coded orchestration for shared-candidate v5/v6 comparison.

No retrieval, semantic judgment, candidate rewriting or outcome-dependent tuning
is performed. Review packets bind exact selections without exposing gate labels.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import replication as rep
import paired_trial as paired
import omission_v5_trial_ops as old
from factcheck import claim_guard_v5 as v5, claim_guard_v6 as v6

DEFAULT_ROOT = Path('pilot/citation_recovery_20261102')
DEFAULT_OUTPUT = Path('runs/citation_recovery_20261102')
SEMANTIC_FIELDS = {'status', 'coverage_complete', 'reason', 'patch_localization'}
PLAIN_BINDING = ('accepted_patch_ids', 'final_text_sha256_utf8', 'final_sources_sha256')
_timestamp, _intact, execution_budget = old._timestamp, old._intact, old.execution_budget


def runtime_check(root):
    root = Path(root).resolve()
    protocol, _ = paired.validate_preregistration(root)
    prereg = rep._read_json(root / 'preregistration.json')
    runtime = rep._strings(prereg.get('runtime_files'), 'preregistration.runtime_files')
    if not runtime or len(runtime) != len(set(runtime)):
        raise ValueError('Runtime snapshots must be nonempty and unique')
    checked = {}
    for name in runtime:
        actual = rep._sha(Path(name))
        if actual != rep._sha(paired._relative(root, 'code_snapshots/' + name)):
            raise ValueError('Runtime differs from frozen snapshot: ' + name)
        checked[name] = actual
    if protocol.get('gate_versions') != {'current': 'v5', 'improved': 'v6'}:
        raise ValueError('Unexpected frozen gate assignment')
    if (root / 'instructions/repair_v1.txt').read_bytes() != (root / 'instructions/repair_v2.txt').read_bytes():
        raise ValueError('Shared writer instructions differ')
    return protocol, checked


def doc_info(root, identity):
    root = Path(root).resolve()
    protocol, hashes = runtime_check(root)
    doc = paired._document(protocol, identity)
    return protocol, doc, paired._relative(root, doc['path']), hashes


def _core(variant):
    return {'current': v5, 'improved': v6}[variant]


def _coded_mapping(protocol, doc):
    storage = paired._evaluation_mapping(protocol, doc)
    variants = paired._arm_mapping(protocol, doc)
    return storage, {code: variants[arm] for code, arm in storage.items()}


def _shared(folder):
    paths = {'original': folder / 'input/original_report.json',
             'candidate': folder / 'shared/candidate.raw.json', 'ledger': folder / 'shared/ledger.raw.json'}
    records = {key: rep._read_json(path) for key, path in paths.items()}
    raw = {key: path.read_bytes() for key, path in paths.items()}
    hashes = {key: rep._sha(path) for key, path in paths.items()}
    execution_budget(records['candidate'], 18, 'shared candidate')
    packet = v5.build_guard_packet(**records, input_file_sha256=hashes)
    if packet != v6.build_guard_packet(**records, input_file_sha256=hashes):
        raise ValueError('Shared guard inputs differ between gates')
    return records, raw, hashes, packet


def _require_generation_ready(root, doc):
    if doc['cohort'] != 'new':
        return
    path = Path(root) / 'generation_ready.json'
    record = rep._read_json(path)
    if record.get('preregistration_sha256') != rep._sha(Path(root) / 'preregistration.json'):
        raise ValueError('Generation readiness preregistration mismatch')
    manifest_path = Path(root) / 'operator_manifest.regression.json'
    if record.get('regression_manifest_sha256') != rep._sha(manifest_path):
        raise ValueError('Generation readiness regression manifest mismatch')
    manifest = rep._read_json(manifest_path)
    if manifest.get('cohort') != 'regression' or set(manifest.get('completed_document_ids', [])) != {'R1', 'R2'}:
        raise ValueError('Historical same-candidate final comparison is not complete')


def prepare_shared_guard(root, identity):
    _, doc, folder, _ = doc_info(root, identity)
    _require_generation_ready(root, doc)
    _, raw, hashes, packet = _shared(folder)
    rep._write_immutable(old._guard_inputs(folder, raw, packet))
    return {'document': identity, 'guard_packet': str(folder / 'shared/guard_input/guard_packet.json'),
            'input_file_sha256': hashes, 'guard_packet_sha256': v5.sha256_json(packet)}


def _review_dir(folder, round_number):
    if round_number not in (0, 1):
        raise ValueError('Only review rounds 0 and 1 are allowed')
    return folder / f'shared/reviews/round{round_number}'


def _packet_outputs(root, identity, folder, round_number, records, hashes, guard, previews, runtime):
    arms = {}
    for code, result in previews.items():
        report = {**copy.deepcopy(records['original']),
                  **{key: copy.deepcopy(result['selected_applied'][key]) for key in ('text', 'sources')}}
        arms[code] = {'selected_report': report,
                      'selection': {key: copy.deepcopy(result['selection'][key]) for key in PLAIN_BINDING}}
    packet = {'schema_version': 1, 'document_id': identity, 'original': copy.deepcopy(records['original']),
              'shared_context': old._patch_context(records['candidate'], records['ledger'], guard),
              'arms': arms, 'identical_selections': len(arms) > 1 and len({v5.sha256_json(a) for a in arms.values()}) == 1}
    # Digest covers the complete semantic packet core; the digest field itself is excluded.
    packet['packet_binding_sha256'] = v5.sha256_json(packet)
    base = _review_dir(folder, round_number)
    freeze_path = base / 'packet.freeze.json'
    freeze = {'schema_version': 1, 'document_id': identity, 'round': round_number,
        'frozen_at_utc': _timestamp(freeze_path, 'frozen_at_utc'), 'input_file_sha256': hashes,
        'runtime_sha256': runtime, 'shared_guard_raw_sha256': rep._sha(folder / 'shared/guard.patch.raw.json'),
        'shared_guard_semantic_sha256': v5.sha256_json(guard),
        'preview_sha256': {code: v5.sha256_json(result) for code, result in previews.items()},
        'packet_binding_sha256': packet['packet_binding_sha256'],
        'packet_file_sha256': hashlib.sha256(rep._json_bytes(packet)).hexdigest(),
        'preregistration_sha256': rep._sha(Path(root) / 'preregistration.json')}
    if round_number:
        previous = _review_dir(folder, 0)
        freeze['previous_review_raw_sha256'] = rep._sha(previous / 'review.raw.json')
        freeze['previous_review_freeze_sha256'] = rep._sha(previous / 'review.freeze.json')
    return {base / 'review_packet.json': rep._json_bytes(packet), freeze_path: rep._json_bytes(freeze)}, packet


def _initial_state(root, identity):
    protocol, doc, folder, runtime = doc_info(root, identity)
    _require_generation_ready(root, doc)
    records, raw, hashes, packet = _shared(folder)
    _intact(old._guard_inputs(folder, raw, packet), 'Shared guard input freeze')
    guard = old._patch_guard(folder)
    storage, variants = _coded_mapping(protocol, doc)
    previews, outputs = {}, {}
    for code in ('A', 'B'):
        result = _core(variants[code]).preview_repair(**records, guard=guard, guard_packet=packet)
        previews[code] = result
        base = folder / 'arms' / storage[code] / 'preview'
        for name, value in [('gate.preview.json', result), ('selection.json', result['selection']),
                            ('selected_report.json', {**records['original'], **{k: result['selected_applied'][k] for k in ('text', 'sources')}})]:
            outputs[base / name] = rep._json_bytes(value)
    review_outputs, review_packet = _packet_outputs(root, identity, folder, 0, records, hashes, guard, previews, runtime)
    outputs.update(review_outputs)
    outputs[folder / 'shared/guard.patch.frozen.json'] = (folder / 'shared/guard.patch.raw.json').read_bytes()
    return {'outputs': outputs, 'records': records, 'hashes': hashes, 'packet': packet, 'guard': guard,
            'previews': previews, 'review_packet': review_packet, 'storage': storage, 'variants': variants,
            'runtime': runtime, 'folder': folder, 'doc': doc}


def preview_pair(root, identity):
    state = _initial_state(root, identity)
    rep._write_immutable(state['outputs'])
    return {'document': identity, 'round': 0,
            'review_packet': str(_review_dir(state['folder'], 0) / 'review_packet.json'),
            'identical_selections': state['review_packet']['identical_selections'],
            'selection': {code: value['selection'] for code, value in state['previews'].items()}}


def _validate_semantic(review, selection):
    rep._all_ids(rep._object(review, 'semantic review'), SEMANTIC_FIELDS, 'semantic review')
    # The v6 structural review validator is used for both coded arms; this supplies
    # no judgment and does not require a v6 selection to validate semantic fields.
    v6._validate_review({**copy.deepcopy(selection), **copy.deepcopy(review)}, selection)


def _validate_response(response, packet, previews):
    rep._all_ids(rep._object(response, 'review response'), {'execution', 'packet_binding_sha256', 'reviews'}, 'review response')
    calls = execution_budget(response, 3, 'full review')
    if response['packet_binding_sha256'] != packet['packet_binding_sha256']:
        raise ValueError('Review response packet binding mismatch')
    reviews = rep._object(response['reviews'], 'reviews')
    rep._all_ids(reviews, previews, 'reviews')
    for code, review in reviews.items():
        _validate_semantic(review, previews[code]['selection'])
    if packet['identical_selections'] and len({v5.sha256_json(review) for review in reviews.values()}) != 1:
        raise ValueError('Identical selections require identical semantic judgments')
    return calls


def _bound_review(semantic, selection, variant):
    if variant == 'improved':
        return {**copy.deepcopy(selection), **copy.deepcopy(semantic)}
    # Unchanged v5 has only pass/fail. Unknown maps conservatively to fail while
    # its exact original status and localization remain in the raw review.
    return {**copy.deepcopy(selection), 'status': 'pass' if semantic['status'] == 'pass' else 'fail',
            'coverage_complete': semantic['coverage_complete'], 'reason': semantic['reason']}


def _review_outputs(state, round_number, packet, previews):
    base = _review_dir(state['folder'], round_number)
    response = rep._read_json(base / 'review.raw.json')
    calls = _validate_response(response, packet, previews)
    bound = {code: _bound_review(response['reviews'][code], result['selection'], state['variants'][code])
             for code, result in previews.items()}
    target = base / 'review.freeze.json'
    record = {'schema_version': 1, 'round': round_number, 'frozen_at_utc': _timestamp(target, 'frozen_at_utc'),
        'raw_sha256': rep._sha(base / 'review.raw.json'), 'packet_sha256': rep._sha(base / 'review_packet.json'),
        'packet_binding_sha256': packet['packet_binding_sha256'], 'web_calls': calls,
        'semantic_sha256': {code: v5.sha256_json(value) for code, value in response['reviews'].items()},
        'bound_review_sha256': {code: v5.sha256_json(value) for code, value in bound.items()},
        'adapter': {code: {'original_semantic_status': value['status'],
            'bound_status': bound[code]['status'], 'unknown_to_fail': value['status'] == 'unknown' and state['variants'][code] == 'current'}
            for code, value in response['reviews'].items()}}
    outputs = {target: rep._json_bytes(record), base / 'review.frozen.json': (base / 'review.raw.json').read_bytes(),
               base / 'bound_reviews.json': rep._json_bytes(bound)}
    return outputs, response, bound, calls


def _initial_review(state):
    _intact(state['outputs'], 'Initial selection/packet freeze')
    outputs, response, bound, calls = _review_outputs(state, 0, state['review_packet'], state['previews'])
    _intact(outputs, 'Initial review freeze')
    return response, bound, calls


def _recovery_state(root, identity, state=None):
    state = state or _initial_state(root, identity)
    _, bound, _ = _initial_review(state)
    code = next(key for key, variant in state['variants'].items() if variant == 'improved')
    recovery = v6.prepare_recovery(**state['records'], guard=state['guard'], final_review=bound[code], guard_packet=state['packet'])
    base = _review_dir(state['folder'], 1)
    decision_path = base / 'recovery.prepare.json'
    outputs = {decision_path: rep._json_bytes(recovery)}
    packet = None
    if recovery['recovery_eligible']:
        packet_outputs, packet = _packet_outputs(root, identity, state['folder'], 1, state['records'],
            state['hashes'], state['guard'], {code: recovery}, state['runtime'])
        outputs.update(packet_outputs)
    return state, code, recovery, outputs, packet


def prepare_recovery(root, identity):
    state, code, recovery, outputs, _ = _recovery_state(root, identity)
    rep._write_immutable(outputs)
    return {'document': identity, 'round': 1, 'coded_arm': code, 'recovery_eligible': recovery['recovery_eligible'],
            'review_packet': str(_review_dir(state['folder'], 1) / 'review_packet.json') if recovery['recovery_eligible'] else None}


def freeze_review(root, identity, round_number=0):
    state = _initial_state(root, identity)
    _intact(state['outputs'], 'Initial selection/packet freeze')
    if round_number == 0:
        packet, previews = state['review_packet'], state['previews']
    elif round_number == 1:
        state, code, recovery, expected, packet = _recovery_state(root, identity, state)
        if not recovery['recovery_eligible']:
            raise ValueError('No rereview is allowed after terminal initial judgment')
        _intact(expected, 'Recovery preparation freeze')
        previews = {code: recovery}
    else:
        raise ValueError('Only review rounds 0 and 1 are allowed')
    outputs, response, _, calls = _review_outputs(state, round_number, packet, previews)
    rep._write_immutable(outputs)
    return {'document': identity, 'round': round_number, 'web_calls': calls,
            'semantic_status': {code: row['status'] for code, row in response['reviews'].items()}}


def _application_outputs(root, identity):
    state = _initial_state(root, identity)
    response0, bound0, calls0 = _initial_review(state)
    code = next(key for key, variant in state['variants'].items() if variant == 'improved')
    first = bound0[code]
    bound1, calls1, recovery = None, 0, None
    def reject_extra_review():
        extra = _review_dir(state['folder'], 1)
        if any((extra / name).exists() for name in ('review.raw.json', 'review.freeze.json', 'review_packet.json')):
            raise ValueError('Extra review after terminal initial judgment or no remainder')
    if v6._recoverable(first):
        _, _, recovery, expected, packet1 = _recovery_state(root, identity, state)
        if recovery['recovery_eligible']:
            _intact(expected, 'Recovery preparation freeze')
            review_outputs, _, bound1, calls1 = _review_outputs(state, 1, packet1, {code: recovery})
            _intact(review_outputs, 'Recovery full review freeze')
        else:
            reject_extra_review()
    else:
        reject_extra_review()
    outputs, results = {}, {}
    candidate_calls = execution_budget(state['records']['candidate'], 18, 'candidate')
    patch_calls = execution_budget(state['guard'], 4, 'patch guard')
    for coded, arm in state['storage'].items():
        variant = state['variants'][coded]
        guard = copy.deepcopy(state['guard'])
        if variant == 'current':
            guard['final_review'] = bound0[coded]
        else:
            guard['final_reviews'] = [bound0[coded]] + ([bound1[coded]] if bound1 else [])
        semantic_keys = set(guard) - {'final_review', 'final_reviews'}
        if {key: guard[key] for key in semantic_keys} != state['guard']:
            raise ValueError('Patch judgment semantics changed in arm adapter')
        result = _core(variant).gate_repair(**state['records'], guard=guard, guard_packet=state['packet'])
        expected_selection = recovery['selection'] if variant == 'improved' and recovery else state['previews'][coded]['selection']
        if result['selection'] != expected_selection:
            raise ValueError('Gate replay did not retain the reviewed selection binding')
        if result['effective_repair']['findings'] != state['records']['candidate']['findings']:
            raise ValueError('Gate changed shared findings')
        base = state['folder'] / 'arms' / arm
        effective = rep._json_bytes(result['effective_repair'])
        applied = rep.validate_repair(state['records']['original'], result['effective_repair'])
        if any(applied[key] != result['final_applied'][key] for key in ('text', 'sources')):
            raise ValueError('Effective repair does not reproduce final applied report')
        outputs[base / 'guard.raw.json'] = rep._json_bytes(guard)
        outputs[base / 'repair.raw.json'] = effective
        outputs[base / 'gated/repair.effective.json'] = effective
        for name, value in [('gate.decisions.json', result), ('final.applied.json', result['final_applied']),
                            ('summary.json', result['summary']), ('final_report.json', {k: result['final_applied'][k] for k in ('text', 'sources')})]:
            outputs[base / 'gated' / name] = rep._json_bytes(value)
        target = base / 'gate_linkage.json'
        linkage = {'recorded_at_utc': _timestamp(target, 'recorded_at_utc'), 'gate_version': 'v5' if variant == 'current' else 'v6',
            'input_file_sha256': state['hashes'], 'runtime_sha256': state['runtime'],
            'shared_guard_semantic_sha256': v5.sha256_json(state['guard']),
            'shared_guard_raw_sha256': rep._sha(state['folder'] / 'shared/guard.patch.raw.json'),
            'initial_review_raw_sha256': rep._sha(_review_dir(state['folder'], 0) / 'review.raw.json'),
            'initial_semantic_sha256': v5.sha256_json(response0['reviews'][coded]),
            'initial_semantic_status': response0['reviews'][coded]['status'],
            'candidate_web_calls': candidate_calls, 'patch_guard_web_calls': patch_calls,
            'initial_final_review_web_calls': calls0, 'additional_v6_rereview_web_calls': calls1 if variant == 'improved' else 0,
            'semantic_recovery_rounds': int(bound1 is not None and variant == 'improved'), 'candidate_rewrite_retries': 0,
            'unequal_additional_verification_budget': True,
            'effective_repair_sha256': hashlib.sha256(effective).hexdigest(),
            'scored_repair_sha256': hashlib.sha256(effective).hexdigest(), 'effective_equals_scored': True}
        if bound1 and variant == 'improved':
            linkage['rereview_raw_sha256'] = rep._sha(_review_dir(state['folder'], 1) / 'review.raw.json')
        outputs[target] = rep._json_bytes(linkage)
        results[arm] = result
    return outputs, results, state['folder']


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
    _, doc, folder, _ = doc_info(root, identity)
    path = folder / 'evaluation/baseline.raw.json'
    baseline = rep._read_json(path)
    rep.validate_baseline(rep._read_json(folder / 'input/original_report.json'), baseline)
    execution_budget(baseline, 25, 'baseline')
    if doc['cohort'] == 'regression':
        record = rep._baseline_freeze(folder)
        if record is None:
            raise ValueError('Regression baseline must be frozen with preregistration')
        return {'document': identity, 'baseline_sha256': record['sha256'], 'provenance': 'reused-known-data'}
    target = folder / 'evaluation/baseline.freeze.json'
    record = {'frozen_at_utc': _timestamp(target, 'frozen_at_utc'), 'sha256': rep._sha(path)}
    rep._write_immutable({target: rep._json_bytes(record)})
    rep._baseline_freeze(folder)
    return {'document': identity, 'baseline_sha256': record['sha256'], 'provenance': 'fresh-original-only'}


def prepare_post(root, identity):
    _, doc, _, _ = doc_info(root, identity)
    _, folder = verify_pair(root, identity)
    if rep._baseline_freeze(folder) is None:
        raise ValueError('Freeze original-only baseline before post packet')
    execution_budget(rep._read_json(folder / 'evaluation/baseline.raw.json'), 25, 'baseline')
    for arm in doc['arms']:
        paired.prepare(root, identity, arm)
    return paired.prepare_post(root, identity)


def ingest_post(root, identity, response=None):
    _, folder = verify_pair(root, identity)
    source = Path(response).resolve() if response is not None else folder / 'evaluation/paired_post.raw.json'
    post = rep._read_json(source)
    execution_budget(post, 25, 'joint post')
    result = paired.ingest_post(root, identity, response)
    consistency = old._detection_consistency(post)
    rep._write_immutable({folder / 'evaluation/detection_consistency.json': rep._json_bytes(consistency)})
    return {**result, 'detection_consistency': consistency}


def _selected_docs(protocol, cohort):
    if cohort not in {'all', 'regression', 'new'}:
        raise ValueError('Cohort must be all, regression or new')
    return [doc for doc in protocol['documents'] if cohort == 'all' or doc['cohort'] == cohort]


def manifest(root, output=DEFAULT_OUTPUT, cohort='all'):
    root, output = Path(root).resolve(), Path(output).resolve()
    protocol, hashes = runtime_check(root)
    documents = []
    for doc in _selected_docs(protocol, cohort):
        identity = doc['id']
        results, folder = verify_pair(root, identity)
        ingest = ingest_post(root, identity)
        original = rep._read_json(folder / 'input/original_report.json')
        exports = {folder / 'input/original_report.txt': original['text'].encode('utf-8')}
        for arm in doc['arms']:
            if (folder / f'arms/{arm}/repair.raw.json').read_bytes() != (folder / f'arms/{arm}/gated/repair.effective.json').read_bytes():
                raise ValueError('Effective repair is not the scored repair')
            revised = rep._read_json(folder / f'arms/{arm}/revised_report.json')
            exports[folder / f'arms/{arm}/revised_report.txt'] = revised['text'].encode('utf-8')
        rep._write_immutable(exports)
        names = sorted(path.relative_to(folder).as_posix() for path in folder.rglob('*.json'))
        web = {'shared_writer': execution_budget(rep._read_json(folder / 'shared/candidate.raw.json'), 18, 'writer'),
               'shared_patch_guard': execution_budget(rep._read_json(folder / 'shared/guard.patch.raw.json'), 4, 'guard'),
               'joint_final_review': execution_budget(rep._read_json(_review_dir(folder, 0) / 'review.raw.json'), 3, 'initial full review'),
               'additional_v6_rereview': 0,
               'baseline': execution_budget(rep._read_json(folder / 'evaluation/baseline.raw.json'), 25, 'baseline'),
               'joint_post': execution_budget(rep._read_json(folder / 'evaluation/paired_post.raw.json'), 25, 'joint post')}
        if (_review_dir(folder, 1) / 'review.raw.json').exists():
            web['additional_v6_rereview'] = execution_budget(rep._read_json(_review_dir(folder, 1) / 'review.raw.json'), 3, 'rereview')
        documents.append({'id': identity, 'cohort': doc['cohort'], 'status': 'complete',
            'input_and_output_sha256': {name: rep._sha(folder / name) for name in names},
            'gate_summary': {arm: result['summary'] for arm, result in results.items()},
            'candidate_identity': True, 'ledger_identity': True, 'patch_judgment_identity': True,
            'findings_identity': True, 'gate_replay_verified': True, 'effective_equals_scored': True,
            'joint_audit_verified': True, 'web_calls': web, 'candidate_rewrite_retries': 0,
            'format_only_repairs': len(list(folder.rglob('*.format_repair.json'))),
            'detection_consistency': ingest['detection_consistency']})
    # Validate selected score completeness before creating any report artifact.
    # The unchanged reporter still retains planned-but-unrun documents with null
    # metrics; cohort selection only limits which documents must be complete.
    _, prereg_hashes = paired.validate_preregistration(root)
    statuses = {doc['id']: paired._document_record(root, protocol, doc, prereg_hashes)[0]['status']
                for doc in protocol['documents']}
    if any(statuses[row['id']] != 'complete' for row in documents):
        raise ValueError('Selected cohort has incomplete or invalid scored pairs')
    paired.build_report(root, output)
    target = root / f'operator_manifest.{cohort}.json'
    value = {'created_at_utc': _timestamp(target, 'created_at_utc'), 'cohort': cohort, 'runtime_sha256': hashes,
        'preregistration_sha256': rep._sha(root / 'preregistration.json'), 'documents': documents,
        'completed_document_ids': [row['id'] for row in documents],
        'planned_document_ids': protocol['planned_document_ids'],
        'planned_unrun_document_ids': [identity for identity in protocol['planned_document_ids'] if statuses[identity] != 'complete'],
        'whole_trial_complete': all(status == 'complete' for status in statuses.values()),
        'limits': protocol['limits']}
    outputs = {target: rep._json_bytes(value), output / 'operator_manifest.json': rep._json_bytes(value)}
    if cohort == 'all':
        outputs[root / 'operator_manifest.json'] = rep._json_bytes(value)
    rep._write_immutable(outputs)
    return {'manifest': str(target), 'report': str(output / 'index.html'), 'document_status': statuses,
            'whole_trial_complete': value['whole_trial_complete'],
            'detection_consistent': all(row['detection_consistency']['consistent'] for row in documents)}


def generation_ready(root):
    root = Path(root).resolve()
    protocol, _ = runtime_check(root)
    source = root / 'operator_manifest.regression.json'
    regression = rep._read_json(source)
    expected = {doc['id'] for doc in protocol['documents'] if doc['cohort'] == 'regression'}
    if regression.get('cohort') != 'regression' or set(regression.get('completed_document_ids', [])) != expected:
        raise ValueError('Complete regression cohort manifest required before new generation')
    for identity in sorted(expected):
        verify_pair(root, identity)
        ingest_post(root, identity)
    target = root / 'generation_ready.json'
    record = {'recorded_at_utc': _timestamp(target, 'recorded_at_utc'),
              'preregistration_sha256': rep._sha(root / 'preregistration.json'),
              'regression_manifest_sha256': rep._sha(source), 'regression_completed_document_ids': sorted(expected),
              'new_generation_started': False}
    rep._write_immutable({target: rep._json_bytes(record)})
    return record


def import_response(root, identity, kind, source, initial=None, attestation=None):
    """Immutably import one raw response, optionally preserving one format repair.

    The no-semantic-change statement is an operator attestation, not a code proof.
    Source files should be external worker outputs; existing target raw bytes are
    never replaced, and a second corrected response cannot be imported.
    """
    _, _, folder, _ = doc_info(root, identity)
    names = {'candidate': 'shared/candidate.raw.json', 'ledger': 'shared/ledger.raw.json',
             'guard': 'shared/guard.patch.raw.json', 'review0': 'shared/reviews/round0/review.raw.json',
             'review1': 'shared/reviews/round1/review.raw.json', 'baseline': 'evaluation/baseline.raw.json',
             'post': 'evaluation/paired_post.raw.json'}
    if kind not in names:
        raise ValueError('Unknown response kind')
    source = Path(source).resolve()
    rep._read_json(source)
    target = folder / names[kind]
    outputs = {target: source.read_bytes()}
    if initial is not None:
        if not isinstance(attestation, str) or not attestation.strip():
            raise ValueError('One format-only repair requires an explicit no-semantic-change attestation')
        initial = Path(initial).resolve()
        if initial.read_bytes() == source.read_bytes():
            raise ValueError('Format repair must preserve a distinct original response')
        first = target.with_name(target.stem + '.initial' + target.suffix)
        audit = target.with_name(target.stem + '.format_repair.json')
        outputs[first] = initial.read_bytes()
        outputs[audit] = rep._json_bytes({'repair_count': 1, 'initial_raw_sha256': rep._sha(initial),
            'corrected_raw_sha256': rep._sha(source), 'format_only_no_semantic_change_attestation': attestation,
            'recorded_at_utc': _timestamp(audit, 'recorded_at_utc')})
    rep._write_immutable(outputs)
    return {'document': identity, 'response': str(target), 'format_only_repairs': int(initial is not None)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare-shared-guard', 'preview-pair', 'freeze-review', 'prepare-recovery',
        'apply-pair', 'verify-pair', 'freeze-baseline', 'prepare-post', 'ingest-post', 'manifest', 'generation-ready', 'import-response'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--doc', choices=['R1', 'R2', 'N1', 'N2'])
    parser.add_argument('--round', dest='round_number', type=int, choices=[0, 1], default=0)
    parser.add_argument('--response', type=Path)
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--kind', choices=['candidate', 'ledger', 'guard', 'review0', 'review1', 'baseline', 'post'])
    parser.add_argument('--attestation')
    parser.add_argument('--cohort', choices=['all', 'regression', 'new'], default='all')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        if args.command == 'manifest':
            result = manifest(args.root, args.output, args.cohort)
        elif args.command == 'generation-ready':
            result = generation_ready(args.root)
        elif not args.doc:
            parser.error('--doc is required')
        elif args.command == 'freeze-review':
            result = freeze_review(args.root, args.doc, args.round_number)
        elif args.command == 'ingest-post':
            result = ingest_post(args.root, args.doc, args.response)
        elif args.command == 'import-response':
            if not args.response or not args.kind:
                parser.error('--response and --kind required for import-response')
            result = import_response(args.root, args.doc, args.kind, args.response, args.initial, args.attestation)
        elif args.command == 'verify-pair':
            results, _ = verify_pair(args.root, args.doc)
            result = {'document': args.doc, 'verified': True, 'arms': list(results)}
        else:
            operations = {'prepare-shared-guard': prepare_shared_guard, 'preview-pair': preview_pair,
                'prepare-recovery': prepare_recovery, 'apply-pair': apply_pair, 'freeze-baseline': freeze_baseline,
                'prepare-post': prepare_post}
            result = operations[args.command](args.root, args.doc)
    except (ValueError, OSError, TypeError, KeyError, UnicodeError) as error:
        parser.exit(2, f'Validation failed: {error}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
