"""Report safeguards with synthetic records; never write experiment outcomes."""
import copy
from pathlib import Path
import tempfile
import unittest

import omission_report as report


class OmissionReportTests(unittest.TestCase):
    def review(self):
        value = {field: [] for field in report.REVIEW_LISTS}
        value.update({"headline": "Synthetic review", "cases": [{"document_id": "N1", "title": "Exact quote",
            "category": "synthetic", "original": "原文\n引用", "current": "現行の引用", "improved": "改善の引用",
            "assessment": ["Synthetic assessment"], "record_ids": ["B01", "P01"],
            "sources": [{"url": "https://example.org/?a=1&b=2", "title": "Source"}]}]})
        return value

    def texts(self):
        return {"N1": {"original": "前。原文\n引用。後。", "current": "前。現行の引用。後。", "improved": "前。改善の引用。後。"}}

    def summary(self):
        protocol = {"id": "synthetic", "documents": [{"id": "N1", "cohort": "new"}],
                    "arm_mapping": {"N1": {"A": "current", "B": "improved"}}}
        primary = {"denominator": 1, "eligible_issue_ids": ["B01"],
                   "current": {"detected": 1, "resolved": 0, "resolved_by_removal": 0},
                   "improved": {"detected": 1, "resolved": 1, "resolved_by_removal": 0}}
        metrics = {"new_problems": {"confirmed_problem": 0}, "patch_reviews": {"unnecessary_change": 0},
                   "false_findings": {"false_alarm": 0}, "changed_locations": 1}
        arms = {arm: {"status": "complete", "missing": [], "validation_errors": [],
                      "variant": variant, "metrics": copy.deepcopy(metrics)}
                for arm, variant in protocol["arm_mapping"]["N1"].items()}
        summary = {"experiment_id": "synthetic", "documents": [{"id": "N1", "cohort": "new", "status": "complete",
            "missing": [], "validation_errors": [], "paired": {"primary": primary}, "arms": arms}]}
        return protocol, summary

    def test_all_three_quotes_must_match_exactly_without_normalizing_whitespace(self):
        review, texts = self.review(), self.texts()
        report._validate_cases(review, texts)
        for variant in ("original", "current", "improved"):
            wrong = copy.deepcopy(review)
            wrong["cases"][0][variant] += " "
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, variant + " quote is not exact"):
                report._validate_cases(wrong, texts)
        review["cases"][0]["original"] = "原文 引用"
        with self.assertRaisesRegex(ValueError, "original quote is not exact"):
            report._validate_cases(review, texts)

    def test_complete_summary_required_and_missing_values_never_become_zero(self):
        protocol, summary = self.summary()
        self.assertEqual(set(report._require_complete_summary(protocol, summary)), {"N1"})
        mutations = [lambda doc: doc.update(status="incomplete"),
                     lambda doc: doc.update(paired=None),
                     lambda doc: doc["arms"]["B"].update(metrics=None),
                     lambda doc: doc["arms"]["B"].update(missing=["post.raw.json"]),
                     lambda doc: doc["paired"]["primary"].update(denominator=2)]
        for change in mutations:
            invalid = copy.deepcopy(summary)
            change(invalid["documents"][0])
            with self.assertRaises(ValueError):
                report._require_complete_summary(protocol, invalid)

    def test_missing_summary_blocks_all_artifact_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            with self.assertRaisesRegex(ValueError, "Complete paired summary.json"):
                report.render_review(base / "trial", base / "output")
            self.assertEqual(list(base.rglob("*")), [])

    def test_source_urls_are_safe_and_html_escaped(self):
        safe = report.source_link("https://example.org/?a=1&b=%22x%22", "<script> & quoted")
        self.assertIn("a=1&amp;b=%22x%22", safe)
        self.assertIn("&lt;script&gt; &amp; quoted", safe)
        for unsafe in ("javascript:alert(1)", "data:text/html,x", "https://example.org/\nattack", "https://user:pass@example.org/"):
            with self.subTest(url=unsafe), self.assertRaises(ValueError):
                report.source_link(unsafe, "source")

    def test_stage_candidates_escape_quotes_and_expose_filter_metadata(self):
        attack = "<script>alert('candidate')</script>"
        row = {"document_id": "R1", "patch_id": "P01", "before": attack, "after": attack + " + correction",
            "reference_status": "valid_correction", "reference_case_id": "H001", "reference_reason": "A & B",
            "v3": {"eligible": False, "reasons": ["hold <reason>"]}, "v5": {"eligible": True, "reasons": []}}
        counts = {status: {"denominator": int(status == "valid_correction"),
            "v3": {"eligible": 0, "rejected": int(status == "valid_correction")},
            "v5": {"eligible": int(status == "valid_correction"), "rejected": 0}} for status in report.REFERENCE_LABELS}
        stage = {"rows": [row], "counts": counts, "documents": [{"document_id": "R1", "candidate_count": 1,
            "v3_preview_eligible": 0, "v5_preview_eligible": 1, "historical_v3_final_accepted": 0}]}
        prior = {"documents": [{"v3_preview_eligible": 0, "v4_preview_eligible": 0}], "changed_eligibility": []}
        output = report._stage_html(stage, prior)
        self.assertNotIn(attack, output)
        self.assertIn("&lt;script&gt;alert(&#x27;candidate&#x27;)&lt;/script&gt;", output)
        self.assertIn("data-reference='valid_correction' data-changed='yes'", output)
        self.assertIn("最終納品本文ではありません", output)
        self.assertIn("候補は 0 件", output)


if __name__ == "__main__":
    unittest.main()
