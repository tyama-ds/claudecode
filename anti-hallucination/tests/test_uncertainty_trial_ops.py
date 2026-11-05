"""Synthetic orchestration E2E: immutable packets, localization and scored replay."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import uncertainty_trial_ops as ops
import paired_trial as paired
import replication as rep
from factcheck import claim_guard_v6 as v6

_spec = importlib.util.spec_from_file_location('citation_synthetic_fixture', Path(__file__).with_name('test_claim_guard_v6.py'))
fixture_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture_module)


class UncertaintyTrialOpsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'trial'
        self.root.mkdir()
        self.execution = {'started_utc': '2026-11-02T00:00:00Z', 'finished_utc': '2026-11-02T00:01:00Z',
                          'web_calls': 0, 'input_files': []}
        self.protocol = {'schema_version': 2, 'id': 'synthetic-citation-trial',
            'planned_document_ids': ['R1', 'R2', 'N1', 'N2'], 'gate_versions': {'current': 'v6', 'improved': 'v7'},
            'documents': [{'id': f'R{i}', 'cohort': 'regression', 'path': f'regression/R{i}',
                           'arms': ['current', 'improved']} for i in (1, 2)] +
                         [{'id': f'N{i}', 'cohort': 'new', 'path': f'new/N{i}', 'arms': ['A', 'B']} for i in (1, 2)],
            'arm_mapping': {'N1': {'A': 'current', 'B': 'improved'}, 'N2': {'A': 'improved', 'B': 'current'}},
            'evaluation_arm_mapping': {'R1': {'A': 'current', 'B': 'improved'}, 'R2': {'A': 'improved', 'B': 'current'}},
            'limits': ['Synthetic test; no experimental observations.']}
        self.write(self.root / 'protocol.json', self.protocol)
        fixed = list(paired.FIXED_FILES) + list(paired.GUARD_FIXED_FILES)
        for name in fixed:
            if name != 'protocol.json':
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('Synthetic identical instruction', encoding='utf-8')
        runtime = ['uncertainty_trial_ops.py']
        for name in runtime:
            path = self.root / 'code_snapshots' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(Path(name).read_bytes())
            fixed.append('code_snapshots/' + name)
        self.write(self.root / 'preregistration.json', {'generation_started': False,
            'runtime_files': runtime, 'sha256': {name: rep._sha(self.root / name) for name in fixed}})

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rep._json_bytes(value))

    def prepare(self, identity='R1', dependencies=True, mutate_ledger=None):
        folder = self.root / 'regression' / identity
        data = fixture_module.fixture('Alpha old.\nBeta old.\nGamma old.\n',
            [('Alpha old.', 'Alpha new.', []), ('Beta old.', 'Beta new.', []), ('Gamma old.', 'Gamma new.', [])])
        original, candidate, ledger, guard = data
        if dependencies:
            candidate['patches'][1]['depends_on_patch_ids'] = ['P1']
        if mutate_ledger:
            mutate_ledger(ledger)
        for path, record in [('input/original_report.json', original), ('shared/candidate.raw.json', candidate),
                             ('shared/ledger.raw.json', ledger)]:
            self.write(folder / path, record)
        ops.prepare_shared_guard(self.root, identity)
        packet = rep._read_json(folder / 'shared/guard_input/guard_packet.json')
        guard.update(guard_packet_sha256=v6.sha256_json(packet), input_sha256=packet['input_sha256'],
                     execution=copy.deepcopy(self.execution))
        self.write(folder / 'shared/guard.patch.raw.json', guard)
        ops.preview_pair(self.root, identity)
        return folder, data

    def semantic(self, status='pass', ids=(), localization=None, complete=True,
                 uncertain_ids=(), uncertainty_status=None, all_localized=True):
        local_status = uncertainty_status or ('definite' if uncertain_ids else 'none')
        return {'status': status, 'coverage_complete': complete, 'reason': 'Synthetic independent review',
                'patch_localization': {'status': localization or ('definite' if ids else 'none'),
                    'patch_ids': list(ids), 'all_failure_causes_localized': bool(ids), 'reason': 'Synthetic localization'},
                'uncertainty_localization': {'status': local_status, 'patch_ids': list(uncertain_ids),
                    'patch_reasons': {key: 'Synthetic unavailable evidence local to this patch' for key in uncertain_ids},
                    'all_unresolved_concerns_localized': bool(uncertain_ids) and all_localized,
                    'remainder_coverage_complete': bool(uncertain_ids) and all_localized,
                    'no_preexisting_or_nonlocal_concerns': bool(uncertain_ids) and all_localized,
                    'reason': 'Synthetic remainder and interactions were independently reviewed'}}

    def review(self, folder, semantic, round_number=0):
        base = ops._review_dir(folder, round_number)
        packet = rep._read_json(base / 'review_packet.json')
        response = {'execution': copy.deepcopy(self.execution), 'packet_binding_sha256': packet['packet_binding_sha256'],
                    'reviews': {code: copy.deepcopy(semantic) for code in packet['arms']}}
        self.write(base / 'review.raw.json', response)
        return response

    def finish_post(self, identity, folder):
        baseline = {'execution': copy.deepcopy(self.execution), 'human_adjudicated': False,
                    'inventory': [], 'issues': [], 'controls': [], 'limitations': ['Synthetic baseline']}
        self.write(folder / 'evaluation/baseline.raw.json', baseline)
        self.write(folder / 'evaluation/baseline.freeze.json', {'frozen_at_utc': '2026-11-02T00:00:00Z',
                   'sha256': rep._sha(folder / 'evaluation/baseline.raw.json')})
        ops.prepare_post(self.root, identity)
        packet = rep._read_json(folder / 'evaluation/paired_post_packet.json')
        post = {'execution': copy.deepcopy(self.execution), 'human_adjudicated': False, 'arms': {}, 'cross_arm_notes': []}
        for code, arm in packet['arms'].items():
            post['arms'][code] = {'execution': copy.deepcopy(self.execution), 'human_adjudicated': False,
                'issues': [], 'patch_reviews': [{'patch_id': row['patch_id'], 'status': 'acceptable',
                    'reason': 'Synthetic independent judgment', 'new_problems': []} for row in arm['literal_changes']],
                'controls': [], 'false_findings': [], 'posthoc_original_findings': [], 'limitations': ['Synthetic']}
        self.write(folder / 'evaluation/paired_post.raw.json', post)
        ops.ingest_post(self.root, identity)

    def test_localized_failure_closure_new_hash_pass_and_scored_replay(self):
        folder, data = self.prepare()
        original0 = rep._read_json(folder / 'arms/improved/preview/selection.json')
        packet = rep._read_json(ops._review_dir(folder, 0) / 'review_packet.json')
        encoded = rep._json_bytes(packet).decode()
        for hidden in ('gate_version', 'mechanical_diagnostics', 'exclusion_trail', 'necessity', 'baseline'):
            self.assertNotIn(hidden, encoded)
        self.assertTrue(packet['identical_selections'])
        self.review(folder, self.semantic('fail', ['P1']))
        ops.freeze_review(self.root, 'R1', 0)
        self.assertTrue(ops.prepare_recovery(self.root, 'R1')['recovery_eligible'])
        changed = rep._read_json(ops._review_dir(folder, 1) / 'recovery.prepare.json')
        self.assertEqual(changed['eligible_codes'], ['A', 'B'])
        changed = changed['recoveries']['B']
        self.assertEqual(changed['selection']['accepted_patch_ids'], ['P3'])
        self.assertNotEqual(original0['final_text_sha256_utf8'], changed['selection']['final_text_sha256_utf8'])
        self.assertNotEqual(original0['selection_sha256'], changed['selection']['selection_sha256'])
        with self.assertRaises((ValueError, FileNotFoundError)):
            ops.apply_pair(self.root, 'R1')
        self.review(folder, self.semantic(), 1)
        ops.freeze_review(self.root, 'R1', 1)
        ops.apply_pair(self.root, 'R1')
        results, _ = ops.verify_pair(self.root, 'R1')
        self.assertEqual([row['patch_id'] for row in results['current']['effective_repair']['patches']], ['P3'])
        self.assertEqual([row['patch_id'] for row in results['improved']['effective_repair']['patches']], ['P3'])
        self.assertEqual(len(results['improved']['review_history']), 2)
        self.finish_post('R1', folder)
        for arm in ('current', 'improved'):
            self.assertEqual((folder / f'arms/{arm}/repair.raw.json').read_bytes(),
                             (folder / f'arms/{arm}/gated/repair.effective.json').read_bytes())
        before = {path: path.read_bytes() for path in folder.rglob('*.json')}
        ops.apply_pair(self.root, 'R1')
        ops.prepare_post(self.root, 'R1')
        ops.ingest_post(self.root, 'R1')
        self.assertEqual(before, {path: path.read_bytes() for path in folder.rglob('*.json')})

    def test_unlocalized_unknown_full_veto_preserves_semantics_and_only_strips_new_field(self):
        folder, _ = self.prepare()
        self.review(folder, self.semantic('unknown', localization='unknown', complete=False))
        ops.freeze_review(self.root, 'R1')
        self.assertFalse(ops.prepare_recovery(self.root, 'R1')['recovery_eligible'])
        ops.apply_pair(self.root, 'R1')
        results, _ = ops.verify_pair(self.root, 'R1')
        self.assertTrue(all(not row['effective_repair']['patches'] for row in results.values()))
        adapted = rep._read_json(ops._review_dir(folder, 0) / 'bound_reviews.json')
        self.assertEqual(adapted['A']['status'], 'unknown')
        self.assertNotIn('uncertainty_localization', adapted['A'])
        self.assertIn('uncertainty_localization', adapted['B'])
        self.assertEqual(adapted['B']['status'], 'unknown')
        raw = rep._read_json(ops._review_dir(folder, 0) / 'review.raw.json')
        self.assertEqual(raw['reviews']['A']['status'], 'unknown')
        with self.assertRaisesRegex(ValueError, 'No rereview'):
            ops.freeze_review(self.root, 'R1', 1)

    def test_empty_recovery_is_terminal_without_new_reviewer_or_success(self):
        folder, _ = self.prepare()
        self.review(folder, self.semantic('fail', ['P1', 'P3']))
        ops.freeze_review(self.root, 'R1')
        prepared = ops.prepare_recovery(self.root, 'R1')
        self.assertFalse(prepared['recovery_eligible'])
        self.assertIsNone(prepared['review_packet'])
        ops.apply_pair(self.root, 'R1')
        results, _ = ops.verify_pair(self.root, 'R1')
        self.assertEqual(results['improved']['effective_repair']['patches'], [])
        self.assertEqual(results['improved']['recovery_status'], 'no_remainder')
        self.assertFalse(results['improved']['summary']['final_review_passed'])
        self.assertFalse((ops._review_dir(folder, 1) / 'review_packet.json').exists())
        self.write(ops._review_dir(folder, 1) / 'review.raw.json', {'unexpected': 'second review'})
        with self.assertRaisesRegex(ValueError, 'Extra review'):
            ops.verify_pair(self.root, 'R1')

    def test_stale_packet_binding_and_semantic_divergence_are_rejected(self):
        folder, _ = self.prepare()
        raw = self.review(folder, self.semantic())
        raw['packet_binding_sha256'] = '0' * 64
        self.write(ops._review_dir(folder, 0) / 'review.raw.json', raw)
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            ops.freeze_review(self.root, 'R1')
        raw = self.review(folder, self.semantic())
        raw['reviews']['A']['reason'] = 'Different judgment for identical input'
        self.write(ops._review_dir(folder, 0) / 'review.raw.json', raw)
        with self.assertRaisesRegex(ValueError, 'Identical selections'):
            ops.freeze_review(self.root, 'R1')

    def test_changed_raw_or_effective_after_freeze_is_rejected(self):
        folder, _ = self.prepare()
        self.review(folder, self.semantic())
        ops.freeze_review(self.root, 'R1')
        ops.apply_pair(self.root, 'R1')
        path = folder / 'arms/current/repair.raw.json'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'Gate application replay'):
            ops.verify_pair(self.root, 'R1')

    def test_cohort_manifest_preserves_unrun_new_null_outcomes(self):
        for identity in ('R1', 'R2'):
            folder, _ = self.prepare(identity)
            self.review(folder, self.semantic())
            ops.freeze_review(self.root, identity)
            ops.apply_pair(self.root, identity)
            self.finish_post(identity, folder)
        output = Path(self.temp.name) / 'regression-report'
        result = ops.manifest(self.root, output, 'regression')
        self.assertFalse(result['whole_trial_complete'])
        summary = rep._read_json(output / 'diagnostic/summary.json')
        self.assertIn('not the raw current pipeline', summary['comparison_scope'])
        html = (output / 'diagnostic/index.html').read_text(encoding='utf-8')
        self.assertIn('共通整合化台帳でのゲート診断（raw現行手順の結果ではない）', html)
        self.assertNotIn('<th>現行解消</th>', html)
        self.assertIn('<th>整合化v6解消</th>', html)
        self.assertIsNone(summary['cohorts']['new']['paired_primary'])
        self.assertEqual(summary['cohorts']['new']['incomplete_documents'], 2)
        self.assertEqual(summary['cohorts']['regression']['complete_pairs'], 2)
        manifest = rep._read_json(self.root / 'operator_manifest.regression.json')
        self.assertEqual(manifest['planned_unrun_document_ids'], ['N1', 'N2'])
        ready = ops.generation_ready(self.root)
        self.assertEqual(ready['regression_completed_document_ids'], ['R1', 'R2'])
        with self.assertRaises((ValueError, FileNotFoundError)):
            ops.manifest(self.root, Path(self.temp.name) / 'all', 'all')

    def test_new_work_requires_completed_historical_stage(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            ops.prepare_shared_guard(self.root, 'N1')

    def test_localized_unknown_holds_only_v7_patch_and_dependents_then_requires_fresh_pass(self):
        folder, _ = self.prepare()
        self.review(folder, self.semantic('unknown', complete=False, uncertain_ids=['P1']))
        ops.freeze_review(self.root, 'R1')
        response = ops.prepare_recovery(self.root, 'R1')
        self.assertEqual(response['eligible_codes'], ['B'])
        packet = rep._read_json(ops._review_dir(folder, 1) / 'review_packet.json')
        self.assertEqual(set(packet['arms']), {'B'})
        self.assertEqual(packet['arms']['B']['selection']['accepted_patch_ids'], ['P3'])
        with self.assertRaises((ValueError, FileNotFoundError)):
            ops.apply_pair(self.root, 'R1')
        self.review(folder, self.semantic(), 1)
        ops.freeze_review(self.root, 'R1', 1)
        ops.apply_pair(self.root, 'R1')
        results, _ = ops.verify_pair(self.root, 'R1')
        self.assertFalse(results['current']['effective_repair']['patches'])
        self.assertEqual([p['patch_id'] for p in results['improved']['effective_repair']['patches']], ['P3'])
        self.assertEqual(results['improved']['recovery_kind'], 'localized_uncertainty')
        self.assertEqual(results['improved']['review_history'][0]['review']['status'], 'unknown')
        self.assertEqual(len(results['current']['review_history']), 1)
        self.assertEqual(len(results['improved']['review_history']), 2)
        self.finish_post('R1', folder)

    def test_second_unknown_is_terminal_and_does_not_keep_retrying(self):
        folder, _ = self.prepare()
        self.review(folder, self.semantic('unknown', complete=False, uncertain_ids=['P1']))
        ops.freeze_review(self.root, 'R1')
        ops.prepare_recovery(self.root, 'R1')
        self.review(folder, self.semantic('unknown', complete=False, uncertain_ids=['P3']), 1)
        ops.freeze_review(self.root, 'R1', 1)
        ops.apply_pair(self.root, 'R1')
        results, _ = ops.verify_pair(self.root, 'R1')
        self.assertFalse(results['improved']['effective_repair']['patches'])
        self.assertEqual(len(results['improved']['review_history']), 2)
        with self.assertRaisesRegex(ValueError, 'Only review rounds'):
            ops.freeze_review(self.root, 'R1', 2)

    def test_partial_localization_and_missing_attestation_do_not_authorize_recovery(self):
        folder, _ = self.prepare()
        review = self.semantic('unknown', complete=False, uncertain_ids=['P1'], all_localized=False)
        self.review(folder, review)
        ops.freeze_review(self.root, 'R1')
        self.assertFalse(ops.prepare_recovery(self.root, 'R1')['recovery_eligible'])
        ops.apply_pair(self.root, 'R1')
        results, _ = ops.verify_pair(self.root, 'R1')
        self.assertTrue(all(not r['effective_repair']['patches'] for r in results.values()))
        missing = self.semantic('unknown', complete=False, uncertain_ids=['P1'])
        del missing['uncertainty_localization']['remainder_coverage_complete']
        with self.assertRaises(ValueError):
            ops._validate_semantic(missing, results['improved']['selection'])

    def test_round1_cannot_add_an_ineligible_current_arm(self):
        folder, _ = self.prepare()
        self.review(folder, self.semantic('unknown', complete=False, uncertain_ids=['P1']))
        ops.freeze_review(self.root, 'R1')
        ops.prepare_recovery(self.root, 'R1')
        raw = self.review(folder, self.semantic(), 1)
        raw['reviews']['A'] = copy.deepcopy(raw['reviews']['B'])
        self.write(ops._review_dir(folder, 1) / 'review.raw.json', raw)
        with self.assertRaises(ValueError):
            ops.freeze_review(self.root, 'R1', 1)

    def test_raw_failure_and_normalized_success_remain_separate_with_raw_bytes_preserved(self):
        def invalid_supported(ledger):
            ledger['claims'][0]['checks']['population'].update(status='supported', evidence_ids=[])
        folder, data = self.prepare(mutate_ledger=invalid_supported)
        preflight = rep._read_json(folder / 'shared/pipeline.preflight.json')
        self.assertFalse(preflight['raw_v6']['valid'])
        self.assertTrue(preflight['raw_current_outcomes_missing'])
        self.assertFalse(preflight['current_aliases_normalized_v6'])
        self.assertTrue(preflight['normalization']['changed'])
        self.assertEqual((folder / 'shared/ledger.raw.json').read_bytes(), rep._json_bytes(data[2]))
        normalized = rep._read_json(folder / 'shared/ledger.normalized.json')
        self.assertEqual(normalized['claims'][0]['checks']['population']['status'], 'unknown')
        self.assertEqual(data[2]['claims'][0]['checks']['population']['status'], 'supported')
        self.review(folder, self.semantic())
        ops.freeze_review(self.root, 'R1')
        ops.apply_pair(self.root, 'R1')
        results, _ = ops.verify_pair(self.root, 'R1')
        self.assertEqual(len(results['current']['effective_repair']['patches']), 3)
        self.assertFalse(rep._read_json(folder / 'shared/pipeline.preflight.json')['raw_v6']['valid'])

    def test_valid_raw_alias_requires_exact_unchanged_ledger_bytes(self):
        folder, _ = self.prepare()
        preflight = rep._read_json(folder / 'shared/pipeline.preflight.json')
        self.assertTrue(preflight['raw_v6']['valid'])
        self.assertTrue(preflight['current_aliases_normalized_v6'])
        self.assertFalse(preflight['normalization']['changed'])
        self.assertEqual((folder / 'shared/ledger.raw.json').read_bytes(), (folder / 'shared/ledger.normalized.json').read_bytes())
        altered = rep._read_json(folder / 'shared/ledger.normalized.json')
        altered['claims'][0]['reason'] = 'Changed by an attacker'
        self.write(folder / 'shared/ledger.normalized.json', altered)
        with self.assertRaisesRegex(ValueError, 'Normalization'):
            ops.preview_pair(self.root, 'R1')

    def test_unrepairable_ledger_failure_is_saved_without_normalized_output(self):
        def unknown_reference(ledger):
            ledger['claims'][0]['checks']['population'].update(status='supported', evidence_ids=['ABSENT'])
        with self.assertRaisesRegex(ValueError, 'failed and was recorded'):
            self.prepare(mutate_ledger=unknown_reference)
        folder = self.root / 'regression/R1'
        preflight = rep._read_json(folder / 'shared/pipeline.preflight.json')
        self.assertEqual(preflight['normalization']['status'], 'failed')
        self.assertFalse(preflight['raw_v6']['valid'])
        self.assertFalse((folder / 'shared/ledger.normalized.json').exists())
        self.assertFalse((folder / 'shared/guard_input/guard_packet.json').exists())

    def test_raw_valid_changed_normalization_cannot_be_aliased(self):
        folder, _ = self.prepare()
        records, _, _ = ops._raw_inputs(folder)
        altered = ops.ledger_consistency.normalize_ledger(**records)
        altered['ledger']['claims'][0]['reason'] = 'Unexpected transformation'
        with patch.object(ops.ledger_consistency, 'normalize_ledger', return_value=altered):
            with self.assertRaisesRegex(ValueError, 'Raw-valid normalization changed'):
                ops._normalization_outputs(folder)

    def test_shared_round1_cost_is_recorded_once_with_both_arm_participation(self):
        folder, _ = self.prepare()
        self.review(folder, self.semantic('fail', ['P1']))
        ops.freeze_review(self.root, 'R1')
        ops.prepare_recovery(self.root, 'R1')
        raw = self.review(folder, self.semantic(), 1)
        raw['execution']['web_calls'] = 3
        self.write(ops._review_dir(folder, 1) / 'review.raw.json', raw)
        result = ops.freeze_review(self.root, 'R1', 1)
        self.assertEqual(result['web_calls_shared'], 3)
        self.assertEqual(result['participating_codes'], ['A', 'B'])
        ops.apply_pair(self.root, 'R1')
        for arm in ('current', 'improved'):
            link = rep._read_json(folder / f'arms/{arm}/gate_linkage.json')
            self.assertEqual(link['additional_joint_rereview_web_calls'], 3)
            self.assertTrue(link['additional_rereview_participated'])
            self.assertEqual(link['semantic_recovery_rounds'], 1)

    def test_one_format_repair_preserves_initial_and_cannot_be_replaced(self):
        initial = Path(self.temp.name) / 'initial.json'
        initial.write_text('{broken json', encoding='utf-8')
        corrected = Path(self.temp.name) / 'corrected.json'
        self.write(corrected, {'synthetic': 'format repair'})
        result = ops.import_response(self.root, 'R1', 'review0', corrected, initial,
                                     'Only JSON punctuation repaired; no semantic judgment changed.')
        self.assertEqual(result['format_only_repairs'], 1)
        target = Path(result['response'])
        self.assertEqual(target.with_name('review.raw.initial.json').read_bytes(), initial.read_bytes())
        self.write(corrected, {'synthetic': 'second changed result'})
        with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
            ops.import_response(self.root, 'R1', 'review0', corrected, initial, 'Second format repair is disallowed.')


if __name__ == '__main__':
    unittest.main()
