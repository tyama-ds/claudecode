"""Synthetic terminal-publication safeguards; never read experimental outcomes."""
import base64
import copy
import importlib.util
from pathlib import Path
import unittest

import citation_report as frozen_report
import citation_terminal_report as terminal
import citation_trial_ops as ops
import replication as rep
from factcheck import claim_guard_v6 as v6

_spec = importlib.util.spec_from_file_location('terminal_ops_fixtures', Path(__file__).with_name('test_citation_trial_ops.py'))
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jZxkAAAAASUVORK5CYII=')


class TerminalPublicationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CitationTrialOpsTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.output = self.root.parent / 'terminal-publication'
        self.folder = self.root / 'new/N2'
        self.fixture.protocol.update(report_title='Synthetic terminal report', report_heading='Synthetic comparison', report_date='2026-11-02')
        self.fixture.write(self.root / 'protocol.json', self.fixture.protocol)
        prereg = rep._read_json(self.root / 'preregistration.json')
        prereg['sha256']['protocol.json'] = rep._sha(self.root / 'protocol.json')
        self.fixture.write(self.root / 'preregistration.json', prereg)
        original, candidate, ledger, _ = fixtures.fixture_module.fixture()
        self.valid_ledger = copy.deepcopy(ledger)
        ledger['claims'][0]['checks']['quantity'].update(status='supported', evidence_ids=[])
        for key, value in zip(('original', 'candidate', 'ledger'), (original, candidate, ledger)):
            self.fixture.write(self.folder / terminal.INPUT_PATHS[key], value)
        (self.folder / 'input/deep_research_completed.png').write_bytes(PNG)
        self.fixture.write(self.folder / terminal.REFUSAL_PATH, {
            'source': 'Synthetic operator transcript of synthetic writer message', 'record_type': 'semantic_retry_declined',
            'format_repair_performed': False, 'candidate_rewritten': False, 'additional_web_calls': 0,
            'writer_statement': 'Synthetic: changes would alter semantic labels, so no retry was performed.',
            'interpretation': 'Synthetic invalid input record; no factual adjudication.'})
        self.review = {**{field: [] for field in frozen_report.REVIEW_LISTS}, 'headline': 'Synthetic partial publication',
            'cases': [{'document_id': 'R1', 'title': 'Exact synthetic example', 'category': 'Synthetic',
                'original': 'Alpha old.', 'current': 'Alpha new.', 'improved': 'Alpha new.',
                'assessment': ['Synthetic only'], 'record_ids': ['P1'], 'sources': []}]}

    def baseline(self, freeze=True):
        value = {'execution': copy.deepcopy(self.fixture.execution), 'human_adjudicated': False,
                 'inventory': [], 'issues': [], 'controls': [], 'limitations': ['Synthetic original-only baseline']}
        self.fixture.write(self.folder / 'evaluation/baseline.raw.json', value)
        if freeze:
            ops.freeze_baseline(self.root, 'N2')

    def complete_three(self):
        for identity in ('R1', 'R2'):
            folder, _ = self.fixture.prepare(identity, dependencies=False)
            self.fixture.review(folder, self.fixture.semantic())
            ops.freeze_review(self.root, identity)
            ops.apply_pair(self.root, identity)
            self.fixture.finish_post(identity, folder)
        ops.manifest(self.root, self.root.parent / 'regression-report', 'regression')
        ops.generation_ready(self.root)
        folder = self.root / 'new/N1'
        original, candidate, ledger, guard = fixtures.fixture_module.fixture()
        paths = {'original': 'input/original_report.json', 'candidate': 'shared/candidate.raw.json', 'ledger': 'shared/ledger.raw.json'}
        for key, value in zip(paths, (original, candidate, ledger)):
            self.fixture.write(folder / paths[key], value)
        packet = v6.build_guard_packet(original, candidate, ledger,
            input_file_sha256={key: rep._sha(folder / name) for key, name in paths.items()})
        guard.update(guard_packet_sha256=v6.sha256_json(packet), input_sha256=packet['input_sha256'],
                     execution=copy.deepcopy(self.fixture.execution))
        self.fixture.write(folder / 'shared/guard.patch.raw.json', guard)
        (folder / 'input/deep_research_completed.png').write_bytes(PNG)
        ops.prepare_shared_guard(self.root, 'N1')
        ops.preview_pair(self.root, 'N1')
        self.fixture.review(folder, self.fixture.semantic())
        ops.freeze_review(self.root, 'N1')
        ops.apply_pair(self.root, 'N1')
        self.fixture.finish_post('N1', folder)
        self.baseline()
        self.fixture.write(self.root / 'review.json', self.review)

    def test_missing_input_cannot_be_declared_terminal(self):
        (self.folder / 'staging/ledger.raw.json').unlink()
        with self.assertRaises(FileNotFoundError):
            terminal.declare_terminal(self.root)
        self.assertFalse((self.root / 'terminal_failure.N2.json').exists())
        self.assertFalse((self.root / 'publication_amendment.json').exists())

    def test_valid_input_cannot_be_declared_terminal(self):
        self.fixture.write(self.folder / 'staging/ledger.raw.json', self.valid_ledger)
        with self.assertRaisesRegex(ValueError, 'inputs validate'):
            terminal.declare_terminal(self.root)
        self.assertFalse((self.root / 'terminal_failure.N2.json').exists())

    def test_declaration_preserves_raw_and_requires_same_hashes(self):
        raw = {name: (self.folder / path).read_bytes() for name, path in terminal.INPUT_PATHS.items()}
        result = terminal.declare_terminal(self.root)
        self.assertEqual(result['status'], 'terminal_failed')
        record, amendment = terminal.verify_terminal(self.root)
        self.assertEqual(record['reproduced_validator_errors']['v5'], record['reproduced_validator_errors']['v6'])
        self.assertIsNone(record['paired_outcomes'])
        self.assertFalse(record['gate_veto_observed'])
        self.assertFalse(amendment['scoring_amended'])
        self.assertEqual(record['writer_refusal']['raw_sha256'], rep._sha(self.folder / terminal.REFUSAL_PATH))
        self.assertEqual(raw, {name: (self.folder / path).read_bytes() for name, path in terminal.INPUT_PATHS.items()})
        ledger = self.folder / 'staging/ledger.raw.json'
        ledger.write_bytes(ledger.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'declaration/publication amendment'):
            terminal.verify_terminal(self.root)

    def test_extra_guard_or_corrected_input_blocks_terminal_replay(self):
        terminal.declare_terminal(self.root)
        for name in ('shared/guard.patch.raw.json', 'staging/ledger.corrected.json', 'evaluation/paired_post.raw.json'):
            path = self.folder / name
            self.fixture.write(path, {'unexpected': True})
            with self.subTest(name=name), self.assertRaises(ValueError):
                terminal.verify_terminal(self.root)
            path.unlink()

    def test_unfrozen_original_only_baseline_blocks_publication(self):
        terminal.declare_terminal(self.root)
        self.baseline(freeze=False)
        with self.assertRaisesRegex(ValueError, 'baseline must be frozen'):
            terminal.publish(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_no_changes_is_not_presented_as_demonstrated_safe_correction(self):
        metric = {'changed_locations': 0, 'patch_reviews': {'acceptable': 0, 'unnecessary_change': 0, 'regression': 0},
                  'new_problems': {'confirmed_problem': 0}}
        counts = {'detected': 1, 'resolved': 0, 'after_status': {'partially_resolved': 0}}
        document = {'id': 'N1', 'cohort': 'new', 'status': 'complete',
            'paired': {'primary': {'denominator': 1, 'current': counts, 'improved': counts}},
            'arms': {variant: {'variant': variant, 'metrics': metric} for variant in ('current', 'improved')}}
        page = terminal._cohort_html('new', {'N1': document})
        self.assertIn('両群とも最終的な変更がありません', page)
        self.assertIn('安全に有用な訂正を届けたことの実証ではありません', page)

    def test_publish_retains_failure_as_null_not_zero_and_is_immutable(self):
        self.complete_three()
        terminal.declare_terminal(self.root)
        preserved = {path: path.read_bytes() for path in self.folder.rglob('*') if path.is_file()}
        result = terminal.publish(self.root, self.output)
        self.assertEqual((result['completed_pairs'], result['planned_pairs']), (3, 4))
        self.assertFalse(result['whole_trial_complete'])
        summary = rep._read_json(self.output / 'summary.json')
        self.assertEqual(summary['cohorts']['new']['planned_documents'], 2)
        self.assertEqual(summary['cohorts']['new']['complete_pairs'], 1)
        self.assertEqual(summary['cohorts']['new']['terminal_failed_documents'], 1)
        failed = next(row for row in summary['documents'] if row['id'] == 'N2')
        self.assertEqual(failed['status'], 'terminal_failed')
        self.assertIsNone(failed['paired'])
        self.assertTrue(all(row['metrics'] is None and row['comparative_outcome'] is None for row in failed['arms'].values()))
        manifest = rep._read_json(self.output / 'terminal_manifest.json')
        failed_manifest = next(row for row in manifest['documents'] if row['id'] == 'N2')
        self.assertIsNone(failed_manifest['gate_summary'])
        self.assertIsNone(failed_manifest['web_calls']['shared_patch_guard'])
        self.assertIsNone(failed_manifest['web_calls']['joint_post'])
        self.assertEqual(preserved, {path: path.read_bytes() for path in self.folder.rglob('*') if path.is_file()})
        page = (self.output / 'report.html').read_text(encoding='utf-8')
        for phrase in ('新規2本中1本のみ比較完了', '事実誤りの件数ではありません', 'staging/ledger.raw.json',
                       'null', 'publication_amendment.json', '全候補', '共有候補', '原文のみbaseline'):
            self.assertIn(phrase, page)
        original_outputs = {path: path.read_bytes() for path in self.output.iterdir()}
        terminal.publish(self.root, self.output)
        self.assertEqual(original_outputs, {path: path.read_bytes() for path in self.output.iterdir()})
        self.review['headline'] += ' changed'
        self.fixture.write(self.root / 'review.json', self.review)
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            terminal.publish(self.root, self.output)
        self.assertEqual(original_outputs, {path: path.read_bytes() for path in self.output.iterdir()})

    def test_exact_quotes_and_failed_document_cases_are_rejected(self):
        self.complete_three()
        terminal.declare_terminal(self.root)
        invalid = copy.deepcopy(self.review)
        invalid['cases'][0]['original'] = 'Alpha  old.'
        self.fixture.write(self.root / 'review.json', invalid)
        with self.assertRaisesRegex(ValueError, 'quote is not exact'):
            terminal.publish(self.root, self.output)
        self.assertFalse(self.output.exists())
        invalid = copy.deepcopy(self.review)
        invalid['cases'][0]['document_id'] = 'N2'
        self.fixture.write(self.root / 'review.json', invalid)
        with self.assertRaisesRegex(ValueError, 'completed document'):
            terminal.publish(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_unexpected_missing_completed_pair_is_not_reclassified(self):
        self.complete_three()
        terminal.declare_terminal(self.root)
        (self.root / 'new/N1/evaluation/paired_post.raw.json').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            terminal.publish(self.root, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(rep._read_json(self.root / 'terminal_failure.N2.json')['document_id'], 'N2')


if __name__ == '__main__':
    unittest.main()
