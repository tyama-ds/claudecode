"""Completed-input reporter safeguards using synthetic temporary trials only."""
import base64
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import citation_report as report
import citation_trial_ops as ops
import paired_trial as paired
import replication as rep
from factcheck import claim_guard_v6 as v6


_spec = importlib.util.spec_from_file_location('citation_ops_report_fixture', Path(__file__).with_name('test_citation_trial_ops.py'))
fixture_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture_module)


class CitationReportTests(unittest.TestCase):
    def review(self):
        return {**{field: [] for field in report.REVIEW_LISTS}, 'headline': 'Synthetic editorial review',
            'cases': [{'document_id': 'R1', 'title': 'Exact synthetic quotation', 'category': 'Synthetic',
                'original': 'Alpha old.', 'current': 'Alpha new.', 'improved': 'Alpha new.',
                'assessment': ['Synthetic only'], 'record_ids': ['P1'],
                'sources': [{'url': 'https://example.org/?a=1&b=2', 'title': 'Synthetic source'}]}]}

    def summary(self):
        protocol = {'id': 'synthetic', 'arm_mapping': {}, 'documents': [
            {'id': identity, 'cohort': cohort, 'arms': ['current', 'improved']}
            for identity, cohort in report.COHORTS.items()]}
        metrics = {group: {key: 0 for key in choices} for group, choices in (
            ('patch_reviews', rep.PATCH_STATUSES), ('new_problems', rep.PROBLEM_STATUSES),
            ('controls', rep.CONTROL_STATUSES), ('posthoc_original_findings', rep.PROBLEM_STATUSES),
            ('false_findings', {'false_alarm', 'unverifiable'}))}
        metrics['changed_locations'] = 0
        primary = {'denominator': 1, 'eligible_issue_ids': ['B1']}
        for variant in ('current', 'improved'):
            primary[variant] = {'detected': 1, 'resolved': 0, 'resolved_by_removal': 0,
                'after_status': {'resolved': 0, 'partially_resolved': 1, 'unresolved': 0, 'unverifiable': 0}}
        summary = {'experiment_id': 'synthetic', 'documents': [
            {'id': spec['id'], 'cohort': spec['cohort'], 'status': 'complete', 'missing': [], 'validation_errors': [],
                'paired': {'primary': copy.deepcopy(primary)},
                'arms': {variant: {'variant': variant, 'status': 'complete', 'missing': [], 'validation_errors': [],
                                  'metrics': copy.deepcopy(metrics)} for variant in ('current', 'improved')}}
            for spec in protocol['documents']]}
        return protocol, summary

    def test_every_planned_document_and_every_count_is_required(self):
        protocol, summary = self.summary()
        self.assertEqual(set(report._require_complete_summary(protocol, summary)), set(report.COHORTS))
        changes = [lambda value: value['documents'].pop(),
            lambda value: value['documents'][0].update(status='incomplete'),
            lambda value: value['documents'][0].update(cohort='new'),
            lambda value: value['documents'][0]['arms']['improved']['metrics']['patch_reviews'].pop('unverifiable'),
            lambda value: value['documents'][0]['paired']['primary']['current']['after_status'].update(partially_resolved=0),
            lambda value: value['documents'][0]['arms']['current']['metrics'].update(changed_locations=True)]
        for change in changes:
            invalid = copy.deepcopy(summary)
            change(invalid)
            with self.subTest(change=change), self.assertRaises(ValueError):
                report._require_complete_summary(protocol, invalid)

    def test_exact_three_way_quotes_without_whitespace_normalization(self):
        review = self.review()
        texts = {'R1': {'original': 'Alpha old.\n', 'current': 'Alpha new.\n', 'improved': 'Alpha new.\n'}}
        report._validate_cases(review, texts)
        for variant in texts['R1']:
            changed = copy.deepcopy(review)
            changed['cases'][0][variant] += ' '
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, variant + ' quote is not exact'):
                report._validate_cases(changed, texts)

    def test_missing_summary_blocks_every_write(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with self.assertRaisesRegex(ValueError, 'Complete paired summary.json'):
                report.render_review(base / 'trial', base / 'output')
            self.assertEqual(list(base.rglob('*')), [])

    def test_source_links_reject_active_urls_and_escape_source_text(self):
        linked = report.source_link('https://example.org/?a=1&b=2', '<script> & title')
        self.assertIn('a=1&amp;b=2', linked)
        self.assertIn('&lt;script&gt; &amp; title', linked)
        for url in ('javascript:alert(1)', 'data:text/html,test', 'https://user:pass@example.org/', 'https://example.org/\nX'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                report.source_link(url, 'Synthetic')

    def test_full_candidate_boundaries_and_reasons_are_escaped_and_filterable(self):
        attack = "<script>alert('patch')</script>"
        row = {'document_id': 'R1', 'cohort': 'regression', 'patch_id': 'P1', 'before': attack,
            'after': attack + '\n11本文', 'depends_on_patch_ids': ['P2'],
            'states': {'current': {'eligible': True, 'accepted': False, 'reasons': ['final veto']},
                       'improved': {'eligible': False, 'accepted': True, 'reasons': ['synthetic <reason>']}},
            'trail': [{'phase': 'mechanical', 'reason': '<boundary>', 'details': {'joined_text': '11本文', 'start_char': 9}}]}
        result = report._candidates_html([row])
        self.assertNotIn(attack, result)
        self.assertIn('&lt;script&gt;alert(&#x27;patch&#x27;)&lt;/script&gt;', result)
        self.assertIn("data-cohort='regression' data-status='changed'", result)
        self.assertIn('start_char', result)
        self.assertIn('11本文', result)
        self.assertIn('P2', result)

    def test_extra_review_is_counted_for_v6_only_and_missing_is_not_zero(self):
        record = {'round0': {'execution': {'web_calls': 2}}, 'round1': {'execution': {'web_calls': 1}},
            'arms': {'current': {'linkage': {'initial_final_review_web_calls': 2, 'additional_v6_rereview_web_calls': 0}},
                'improved': {'linkage': {'initial_final_review_web_calls': 2, 'additional_v6_rereview_web_calls': 1},
                    'gate': {'review_history': [{}, {}], 'summary': {'semantic_recovery_rounds': 1}}}}}
        self.assertEqual(report._review_costs(record), {'initial_web_calls': 2, 'additional_web_calls': 1,
            'actual_rereviews': 1, 'removal_rounds': 1})
        missing = copy.deepcopy(record)
        missing['round1'] = None
        with self.assertRaisesRegex(ValueError, 'artifact/history mismatch'):
            report._review_costs(missing)
        missing = copy.deepcopy(record)
        missing['round0']['execution']['web_calls'] = None
        with self.assertRaisesRegex(ValueError, 'invalid count'):
            report._review_costs(missing)

    def build_complete_fixture(self, recovery=False):
        fixture = fixture_module.CitationTrialOpsTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.protocol.update(report_title='Synthetic citation report', report_heading='Synthetic v5/v6', report_date='2026-11-02')
        for spec in fixture.protocol['documents']:
            spec['theme'] = 'Synthetic theme ' + spec['id']
        fixture.write(fixture.root / 'protocol.json', fixture.protocol)
        prereg = rep._read_json(fixture.root / 'preregistration.json')
        prereg['sha256']['protocol.json'] = rep._sha(fixture.root / 'protocol.json')
        fixture.write(fixture.root / 'preregistration.json', prereg)
        for identity in ('R1', 'R2'):
            folder, _ = fixture.prepare(identity, dependencies=False)
            fixture.review(folder, fixture.semantic('fail', ['P1']) if recovery and identity == 'R1' else fixture.semantic())
            ops.freeze_review(fixture.root, identity)
            if recovery and identity == 'R1':
                ops.prepare_recovery(fixture.root, identity)
                raw = fixture.review(folder, fixture.semantic(), 1)
                raw['execution']['web_calls'] = 2
                fixture.write(ops._review_dir(folder, 1) / 'review.raw.json', raw)
                ops.freeze_review(fixture.root, identity, 1)
            ops.apply_pair(fixture.root, identity)
            fixture.finish_post(identity, folder)
        ops.manifest(fixture.root, fixture.root.parent / 'regression-output', 'regression')
        ops.generation_ready(fixture.root)
        for identity in ('N1', 'N2'):
            folder = fixture.root / 'new' / identity
            data = fixture_module.fixture_module.fixture('Alpha old.\nBeta old.\n',
                [('Alpha old.', 'Alpha new.', []), ('Beta old.', 'Beta new.', [])])
            original, candidate, ledger, guard = data
            names = {'original': 'input/original_report.json', 'candidate': 'shared/candidate.raw.json', 'ledger': 'shared/ledger.raw.json'}
            for key, value in zip(names, (original, candidate, ledger)):
                fixture.write(folder / names[key], value)
            packet = v6.build_guard_packet(original, candidate, ledger,
                input_file_sha256={key: rep._sha(folder / name) for key, name in names.items()})
            guard.update(guard_packet_sha256=v6.sha256_json(packet), input_sha256=packet['input_sha256'], execution=copy.deepcopy(fixture.execution))
            fixture.write(folder / 'shared/guard.patch.raw.json', guard)
            # A tiny synthetic PNG belongs only to this temporary fixture.
            (folder / 'input/deep_research_completed.png').write_bytes(base64.b64decode(
                'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jZxkAAAAASUVORK5CYII='))
            ops.prepare_shared_guard(fixture.root, identity)
            ops.preview_pair(fixture.root, identity)
            fixture.review(folder, fixture.semantic())
            ops.freeze_review(fixture.root, identity)
            ops.apply_pair(fixture.root, identity)
            fixture.finish_post(identity, folder)
        output = fixture.root.parent / 'output'
        ops.manifest(fixture.root, output)
        review = self.review()
        if recovery:
            review['cases'][0]['current'] = review['cases'][0]['improved'] = 'Alpha old.'
        fixture.write(fixture.root / 'review.json', review)
        return fixture, output

    def test_completed_read_only_render_is_immutable_and_reports_zero_real_retries(self):
        fixture, output = self.build_complete_fixture()
        before = {path: path.read_bytes() for path in fixture.root.rglob('*') if path.is_file()}
        result = report.render_review(fixture.root, output)
        self.assertEqual(result['complete_documents'], 4)
        self.assertEqual(result['actual_v6_rereviews'], 0)
        self.assertEqual(before, {path: path.read_bytes() for path in fixture.root.rglob('*') if path.is_file()})
        self.assertEqual((output / 'review.json').read_bytes(), (fixture.root / 'review.json').read_bytes())
        page = (output / 'report.html').read_text(encoding='utf-8')
        for expected in ('実データでの追加全文再確認は0回', '局所回復分岐の実証にはなりません',
                         "id='new'", "id='regression'", '妥当と評価された適用変更', 'baseline未確定',
                         '事後確認・残存なし', 'review.raw.json', 'deep_research_completed.png'):
            self.assertIn(expected, page)
        report.render_review(fixture.root, output)
        self.assertEqual((output / 'report.html').read_text(encoding='utf-8'), page)
        changed = self.review()
        changed['headline'] += ' changed'
        fixture.write(fixture.root / 'review.json', changed)
        with self.assertRaisesRegex(ValueError, 'immutable|Immutable|overwrite'):
            report.render_review(fixture.root, output)
        self.assertEqual((output / 'report.html').read_text(encoding='utf-8'), page)

    def test_tampered_frozen_input_blocks_report_publication(self):
        fixture, output = self.build_complete_fixture()
        candidate = fixture.root / 'regression/R1/shared/candidate.raw.json'
        candidate.write_bytes(candidate.read_bytes() + b'\n')
        with self.assertRaises(ValueError):
            report.render_review(fixture.root, output)
        self.assertFalse((output / 'report.html').exists())
        self.assertFalse((output / 'review.json').exists())

    def test_real_second_review_is_separate_from_initial_shared_cost(self):
        fixture, output = self.build_complete_fixture(recovery=True)
        _, _, _, records, _ = report.load_completed(fixture.root, output)
        self.assertEqual(records['R1']['costs'], {'initial_web_calls': 0, 'additional_web_calls': 2,
            'actual_rereviews': 1, 'removal_rounds': 1})
        self.assertEqual(records['R1']['arms']['current']['accepted'], set())
        self.assertEqual(records['R1']['arms']['improved']['accepted'], {'P2', 'P3'})
        result = report.render_review(fixture.root, output)
        self.assertEqual(result['actual_v6_rereviews'], 1)
        page = (output / 'report.html').read_text(encoding='utf-8')
        self.assertIn('追加全文再確認を 1 回実施', page)
        self.assertNotIn('実データでの追加全文再確認は0回', page)


if __name__ == '__main__':
    unittest.main()
