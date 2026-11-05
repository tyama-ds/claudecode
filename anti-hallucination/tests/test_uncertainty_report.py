"""Synthetic reporting tests: raw failures never inherit diagnostic success."""
import base64
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import uncertainty_report as report
import uncertainty_trial_ops as ops
import paired_trial as paired
import replication as rep
from factcheck.claim_guard_v5 import sha256_json

_spec = importlib.util.spec_from_file_location('uncertainty_report_fixture', Path(__file__).with_name('test_uncertainty_trial_ops.py'))
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)


class UncertaintyReportTests(unittest.TestCase):
    def review(self):
        return {**{field: [] for field in report.REVIEW_LISTS}, 'headline': 'Synthetic comparison only',
                'cases': [{'document_id': 'R1', 'title': 'Exact final excerpt', 'category': 'Synthetic',
                           'original': 'Beta old.', 'current': 'Beta old.', 'improved': 'Beta new.',
                           'assessment': ['Synthetic paired case'], 'record_ids': ['P2'],
                           'sources': [{'url': 'https://example.org/?a=1&b=2', 'title': 'Synthetic source'}]}]}

    def build_fixture(self):
        fx = fixtures.UncertaintyTrialOpsTests()
        fx.setUp()
        self.addCleanup(fx.doCleanups)
        fx.protocol.update(report_title='Synthetic uncertainty trial', report_heading='Synthetic raw and normalized', report_date='2026-11-02')
        for spec in fx.protocol['documents']:
            spec['theme'] = 'Synthetic ' + spec['id']
        fx.write(fx.root / 'protocol.json', fx.protocol)
        prereg = rep._read_json(fx.root / 'preregistration.json')
        prereg['sha256']['protocol.json'] = rep._sha(fx.root / 'protocol.json')
        fx.write(fx.root / 'preregistration.json', prereg)
        for identity in ('R1', 'R2'):
            def make_inconsistent(ledger):
                ledger['claims'][0]['checks']['population'] = {'status': 'supported', 'rationale': 'Synthetic missing access', 'evidence_ids': []}
            folder, _ = fx.prepare(identity, dependencies=False, mutate_ledger=make_inconsistent if identity == 'R2' else None)
            fx.review(folder, fx.semantic('unknown', complete=False, uncertain_ids=['P1']) if identity == 'R1' else fx.semantic())
            ops.freeze_review(fx.root, identity)
            if identity == 'R1':
                ops.prepare_recovery(fx.root, identity)
                response = fx.review(folder, fx.semantic(), 1)
                response['execution']['web_calls'] = 2
                fx.write(ops._review_dir(folder, 1) / 'review.raw.json', response)
                ops.freeze_review(fx.root, identity, 1)
            ops.apply_pair(fx.root, identity)
            fx.finish_post(identity, folder)
        ops.manifest(fx.root, fx.root.parent / 'regression-output', 'regression')
        ops.generation_ready(fx.root)
        for identity in ('N1', 'N2'):
            folder = fx.root / 'new' / identity
            original, candidate, ledger, guard = fixtures.fixture_module.fixture('Alpha old.\nBeta old.\n',
                [('Alpha old.', 'Alpha new.', []), ('Beta old.', 'Beta new.', [])])
            for path, value in [('input/original_report.json', original), ('shared/candidate.raw.json', candidate), ('shared/ledger.raw.json', ledger)]:
                fx.write(folder / path, value)
            ops.prepare_shared_guard(fx.root, identity)
            packet = rep._read_json(folder / 'shared/guard_input/guard_packet.json')
            guard.update(guard_packet_sha256=sha256_json(packet), input_sha256=packet['input_sha256'], execution=copy.deepcopy(fx.execution))
            fx.write(folder / 'shared/guard.patch.raw.json', guard)
            (folder / 'input/deep_research_completed.png').write_bytes(base64.b64decode(
                'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jZxkAAAAASUVORK5CYII='))
            ops.preview_pair(fx.root, identity)
            fx.review(folder, fx.semantic())
            ops.freeze_review(fx.root, identity)
            ops.apply_pair(fx.root, identity)
            fx.finish_post(identity, folder)
        output = fx.root.parent / 'output'
        ops.manifest(fx.root, output)
        fx.write(fx.root / 'review.json', self.review())
        return fx, output

    def test_raw_failure_keeps_null_main_and_complete_diagnostic(self):
        fx, output = self.build_fixture()
        *_, summary, records, review = report.load_trial(fx.root, output)
        docs = {row['id']: row for row in summary['documents']}
        self.assertTrue(summary['all_diagnostic_pairs_complete'])
        self.assertFalse(summary['all_main_pairs_complete'])
        self.assertFalse(summary['whole_trial_complete'])
        self.assertEqual(len(docs), 4)
        failed = docs['R2']
        self.assertIsNone(failed['main']['paired'])
        self.assertIsNone(failed['main']['arms']['current']['metrics'])
        self.assertEqual(failed['main']['arms']['current']['status'], 'terminal_failed')
        self.assertEqual(failed['main']['arms']['improved']['status'], 'complete')
        self.assertIsNotNone(failed['diagnostic']['paired'])
        self.assertEqual(summary['cohorts']['regression']['main']['complete_pairs'], 1)
        self.assertEqual(summary['cohorts']['regression']['diagnostic']['complete_pairs'], 2)
        self.assertEqual(summary['cohorts']['regression']['main']['completion_by_pipeline'], {'current': 1, 'improved': 2})
        self.assertEqual(docs['R1']['main']['arms']['current']['metrics']['changed_locations'], 0)
        self.assertEqual(docs['R1']['main']['arms']['improved']['metrics']['changed_locations'], 2)

    def test_raw_valid_alias_requires_no_normalization_change(self):
        fx, output = self.build_fixture()
        loaded = report.load_trial(fx.root, output)
        protocol, diagnostic = loaded[2], loaded[4]
        pf = {row['id']: row['pipeline_preflight'] for row in loaded[5]['documents']}
        for change in (lambda p: p['normalization'].update(changed=True),
                       lambda p: p.update(current_aliases_normalized_v6=False),
                       lambda p: p.update(raw_current_outcomes_missing=True),
                       lambda p: p.update(failure_is_zero_outcome=True)):
            invalid = copy.deepcopy(pf)
            change(invalid['R1'])
            with self.subTest(change=change), self.assertRaises(ValueError):
                report.compose_summary(protocol, diagnostic, invalid)

    def test_omitted_document_wrong_cohort_and_noninteger_counts_rejected(self):
        fx, output = self.build_fixture()
        loaded = report.load_trial(fx.root, output)
        protocol, diagnostic = loaded[2], loaded[4]
        pf = {row['id']: row['pipeline_preflight'] for row in loaded[5]['documents']}
        for change in (lambda d: d['documents'].pop(),
                       lambda d: d['documents'][0].update(cohort='new'),
                       lambda d: d['documents'][0]['paired']['primary'].update(denominator=True)):
            invalid = copy.deepcopy(diagnostic)
            change(invalid)
            with self.subTest(change=change), self.assertRaises(ValueError):
                report.compose_summary(protocol, invalid, pf)

    def test_exact_cases_use_diagnostic_text_without_whitespace_repair(self):
        review = self.review()
        texts = {'R1': {'original': 'Beta old.\n', 'current': 'Beta old.\n', 'improved': 'Beta new.\n'}}
        report._validate_cases(review, texts)
        for variant in texts['R1']:
            changed = copy.deepcopy(review)
            changed['cases'][0][variant] += ' '
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, 'quote is not exact'):
                report._validate_cases(changed, texts)
        page = report._cases_html(review)
        self.assertIn('診断対照：同じ整合化台帳＋v6', page)
        self.assertIn('raw現行が失敗した文書の納品結果ではありません', page)

    def test_shared_extra_cost_counted_once_and_nonparticipant_is_not_charged(self):
        fx, output = self.build_fixture()
        record = report.load_trial(fx.root, output)[6]['R1']
        costs = record['costs']
        self.assertEqual(costs['additional_shared_web_calls'], 2)
        self.assertEqual(costs['shared_total_web_calls'], 2)
        self.assertFalse(costs['per_variant']['current']['additional_review_participated'])
        self.assertEqual(costs['per_variant']['current']['additional_shared_web_calls'], 0)
        self.assertEqual(costs['per_variant']['improved']['additional_shared_web_calls'], 2)
        bad = copy.deepcopy(record)
        bad['round1'] = None
        with self.assertRaises(ValueError):
            report._review_costs(bad)

    def test_publish_is_immutable_and_all_diagnostic_labels_are_explicit(self):
        fx, output = self.build_fixture()
        before = {path: path.read_bytes() for path in fx.root.rglob('*') if path.is_file()}
        result = report.publish(fx.root, output)
        self.assertTrue(result['all_diagnostic_pairs_complete'])
        self.assertFalse(result['all_main_pairs_complete'])
        for path, data in before.items():
            self.assertEqual(path.read_bytes(), data)
        page = (output / 'report.html').read_text(encoding='utf-8')
        for text in ('同じ整合化済み台帳', '全修正候補', '主比較', '診断比較', 'terminal_failed',
                     'normalization.audit.json', 'review.raw.json', 'deep_research_completed.png'):
            self.assertIn(text, page)
        self.assertEqual((output / 'review.json').read_bytes(), (fx.root / 'review.json').read_bytes())
        report.publish(fx.root, output)
        self.assertEqual(page, (output / 'report.html').read_text(encoding='utf-8'))
        review = self.review()
        review['headline'] += ' Changed'
        fx.write(fx.root / 'review.json', review)
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            report.publish(fx.root, output)
        self.assertEqual(page, (output / 'report.html').read_text(encoding='utf-8'))

    def test_tampered_raw_or_normalized_input_blocks_publication(self):
        for name in ('shared/candidate.raw.json', 'shared/ledger.normalized.json'):
            with self.subTest(name=name):
                fx, output = self.build_fixture()
                path = fx.root / 'regression/R2' / name
                path.write_bytes(path.read_bytes() + b'\n')
                with self.assertRaises(ValueError):
                    report.publish(fx.root, output)
                self.assertFalse((output / 'report.html').exists())

    def test_terminal_incomplete_needs_explicit_declaration_and_retains_nulls(self):
        fx, _ = self.build_fixture()
        folder = fx.root / 'new/N2'
        # Delete only a temporary synthetic response to model a missing stage.
        (folder / 'evaluation/paired_post.raw.json').unlink()
        output = fx.root.parent / 'incomplete-output'
        ops.build_diagnostic_report(fx.root, output / 'diagnostic')
        with self.assertRaisesRegex(ValueError, 'All four diagnostic pairs'):
            report.publish(fx.root, output)
        with self.assertRaises(FileNotFoundError):
            report.publish(fx.root, output, allow_incomplete=True)
        report.declare_terminal(fx.root, 'N2', 'post_evaluation', 'Synthetic missing response')
        result = report.publish(fx.root, output, allow_incomplete=True)
        self.assertFalse(result['all_diagnostic_pairs_complete'])
        summary = rep._read_json(output / 'summary.json')
        row = next(row for row in summary['documents'] if row['id'] == 'N2')
        self.assertIsNone(row['main']['paired'])
        self.assertIsNone(row['diagnostic']['paired'])
        self.assertTrue(all(arm['metrics'] is None for arm in row['main']['arms'].values()))
        self.assertTrue(all(arm['metrics'] is None for arm in row['diagnostic']['arms'].values()))

    def test_terminal_hashes_and_completed_pair_declaration_fail_closed(self):
        fx, output = self.build_fixture()
        with self.assertRaisesRegex(ValueError, 'completed diagnostic'):
            report.declare_terminal(fx.root, 'R1', 'post_evaluation', 'Synthetic')
        folder = fx.root / 'new/N2'
        (folder / 'evaluation/paired_post.raw.json').unlink()
        report.declare_terminal(fx.root, 'N2', 'post_evaluation', 'Synthetic missing response')
        raw = folder / 'shared/candidate.raw.json'
        raw.write_bytes(raw.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            report.validate_terminal(fx.root, folder, 'N2')
        terminal = rep._read_json(folder / 'terminal_failure.json')
        terminal['input_file_sha256'].pop('shared/candidate.raw.json')
        fx.write(folder / 'terminal_failure.json', terminal)
        with self.assertRaisesRegex(ValueError, 'complete current artifact set'):
            report.validate_terminal(fx.root, folder, 'N2')

    def test_terminal_declaration_cannot_be_overridden_by_restored_stage(self):
        fx, output = self.build_fixture()
        folder = fx.root / 'new/N2'
        response = folder / 'evaluation/paired_post.raw.json'
        raw = response.read_bytes()
        response.unlink()
        report.declare_terminal(fx.root, 'N2', 'post_evaluation', 'Synthetic missing response')
        response.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'Previously declared terminal'):
            report.publish(fx.root, output, allow_incomplete=True)
        self.assertFalse((output / 'report.html').exists())

    def test_sources_and_raw_text_are_escaped(self):
        review = self.review()
        review['cases'][0]['original'] = '<script>bad</script>'
        page = report._cases_html(review)
        self.assertNotIn('<script>bad</script>', page)
        self.assertIn('&lt;script&gt;bad&lt;/script&gt;', page)
        review['cases'][0]['sources'][0]['url'] = 'javascript:bad'
        with self.assertRaises(ValueError):
            report._cases_html(review)


if __name__ == '__main__':
    unittest.main()
