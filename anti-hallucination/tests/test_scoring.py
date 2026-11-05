"""Meaningful boundaries for numeric scoring and the private/public split."""

import copy
import json
from pathlib import Path
import unittest

from factcheck.dataset import decimal_value, item_context, validate_dataset
from factcheck.scoring import candidate_truth, score_answer


ROOT = Path(__file__).resolve().parents[1]


class DatasetAndScoringTests(unittest.TestCase):
    def setUp(self):
        self.reports = json.loads((ROOT / "samples" / "reports.json").read_text(encoding="utf-8"))
        self.gold = json.loads((ROOT / "samples" / "gold.json").read_text(encoding="utf-8"))
        self.item = self.gold["items"][0]
        self.report = self.reports["reports"][0]

    def answer(self, value="120.5", unit="億円", **changes):
        result = {
            "status": "answered",
            "value": value,
            "unit": unit,
            "evidence": [{"source_id": "s1", "quote": "数値ID=n1; 値=120.5; 単位=億円"}],
        }
        result.update(changes)
        return result

    def test_samples_are_two_reports_six_items_and_do_not_mutate(self):
        before = copy.deepcopy((self.reports, self.gold))
        result = validate_dataset(self.reports, self.gold)
        self.assertEqual(len(result["reports"]), 2)
        self.assertEqual(len(result["items"]), 6)
        self.assertEqual((self.reports, self.gold), before)
        self.assertTrue(all("架空" in row["title"] for row in result["reports"]))

    def test_report_and_source_extra_fields_cannot_leak_gold(self):
        secret = "PRIVATE-GOLD-SENTINEL-456789"
        self.item["private_notes"] = secret
        self.item["candidates"][0]["private_notes"] = secret
        self.report["value"] = secret
        self.report["historical_verdict"] = secret
        self.report["sources"][0]["gold_value"] = secret
        self.report["sources"][0]["metadata"] = {"answer": secret}
        validated = validate_dataset(self.reports, self.gold)
        context = item_context(validated["items"][0], validated["reports"][0])
        self.assertEqual(set(context), {"item_id", "report_title", "report_text", "statement", "unit", "as_of", "sources", "comparison_rules"})
        self.assertNotIn(secret, json.dumps(context))
        self.assertNotIn("gold_value", context["sources"][0])
        self.assertNotIn("metadata", context["sources"][0])
        self.assertNotIn("{{", context["report_text"])
        self.assertNotIn("{{", context["statement"])
        # Evidence naturally includes the answer; its inclusion is intentional.
        self.assertIn("値=120.5", context["sources"][0]["text"])

    def test_public_comparison_rules_do_not_contain_the_answer(self):
        context = item_context(self.item, self.report)
        self.assertEqual(context["comparison_rules"], {
            "absolute_tolerance": "0.05",
            "target_unit": "億円",
            "accepted_units": {"百万円": "0.01", "万円": "0.0001", "円": "0.00000001"},
        })
        self.assertNotIn("value", context["comparison_rules"])
        self.assertNotIn("candidates", context)
        context["comparison_rules"]["accepted_units"]["円"] = "999"
        self.assertEqual(self.item["accepted_units"]["円"], "0.00000001")

    def test_undeclared_and_missing_placeholders_rejected(self):
        for text in (self.report["text"] + " {{unknown}}", self.report["text"].replace("{{n1}}", "120.5"), self.report["text"] + " {{broken"):
            with self.subTest(text=text):
                reports = copy.deepcopy(self.reports)
                reports["reports"][0]["text"] = text
                with self.assertRaises(ValueError):
                    validate_dataset(reports, self.gold)

    def test_mask_text_replaces_all_literal_tokens_but_not_sources(self):
        self.report["text"] = self.report["text"].replace("{{n1}}", "1,205") + " 再掲：1,205 億円。別値：11,205 億円と1,205.1億円。"
        self.item["mask_text"] = "1,205"
        original_source = self.report["sources"][0]["text"]
        validated = validate_dataset(self.reports, self.gold)
        report = validated["reports"][0]
        self.assertEqual(report["text"].count("{{n1}}"), 2)
        self.assertIn("11,205", report["text"])
        self.assertIn("1,205.1", report["text"])
        self.assertEqual(report["sources"][0]["text"], original_source)
        self.assertNotIn("{{n1}}", self.report["text"])

    def test_mask_text_rejects_duplicate_or_partially_masked_values(self):
        for conflict in ("duplicate", "mixed", "missing"):
            reports, gold = copy.deepcopy(self.reports), copy.deepcopy(self.gold)
            gold["items"][0]["mask_text"] = "120.5"
            if conflict == "duplicate":
                gold["items"][1]["mask_text"] = "120.5"
            elif conflict == "mixed":
                reports["reports"][0]["text"] += " 再掲：120.5億円。"
            else:
                reports["reports"][0]["text"] = reports["reports"][0]["text"].replace("{{n1}}", "1120.5")
            with self.subTest(conflict=conflict):
                with self.assertRaisesRegex(ValueError, "explicit placeholders"):
                    validate_dataset(reports, gold)

    def test_mask_text_numeric_format_and_signed_token_boundaries(self):
        for mask in ("1,20", "1e3", "120円", "{{n1}}", " 120 "):
            gold = copy.deepcopy(self.gold)
            gold["items"][0]["mask_text"] = mask
            with self.subTest(mask=mask):
                with self.assertRaises(ValueError):
                    validate_dataset(self.reports, gold)
        self.item["mask_text"] = "-12.5"
        self.report["text"] = self.report["text"].replace("{{n1}}", "-12.5") + " 別値：-112.5、12.5、A-12.5、1e-12.5。"
        text = validate_dataset(self.reports, self.gold)["reports"][0]["text"]
        self.assertEqual(text.count("{{n1}}"), 1)
        self.assertIn("-112.5", text)

    def test_duplicates_and_bad_source_reference_rejected(self):
        cases = []
        reports = copy.deepcopy(self.reports)
        reports["reports"].append(copy.deepcopy(reports["reports"][0]))
        cases.append((reports, self.gold))
        gold = copy.deepcopy(self.gold)
        gold["items"].append(copy.deepcopy(gold["items"][0]))
        cases.append((self.reports, gold))
        reports = copy.deepcopy(self.reports)
        reports["reports"][0]["sources"].append(copy.deepcopy(reports["reports"][0]["sources"][0]))
        cases.append((reports, self.gold))
        gold = copy.deepcopy(self.gold)
        gold["items"][0]["source_ids"] = ["s2"]
        cases.append((self.reports, gold))
        for reports, gold in cases:
            with self.subTest(reports=reports["reports"][0]["id"], items=len(gold["items"])):
                with self.assertRaises(ValueError):
                    validate_dataset(reports, gold)

    def test_invalid_numeric_gold_configuration_rejected(self):
        for key, value in (("value", "NaN"), ("value", "Infinity"), ("tolerance", "-0.1"), ("accepted_units", {"円": "0"}), ("accepted_units", {"億円": "2"}), ("accepted_units", {"円": "-1"})):
            with self.subTest(key=key, value=value):
                gold = copy.deepcopy(self.gold)
                gold["items"][0][key] = value
                with self.assertRaises(ValueError):
                    validate_dataset(self.reports, gold)

    def test_statement_requires_one_answer_slot_and_dates_are_iso(self):
        for statement in ("No answer slot", "{{value}} and {{value}}", "{{value}} and {{other}}"):
            with self.subTest(statement=statement):
                gold = copy.deepcopy(self.gold)
                gold["items"][0]["statement"] = statement
                with self.assertRaises(ValueError):
                    validate_dataset(self.reports, gold)
        self.report["as_of"] = "2024-02-30"
        with self.assertRaises(ValueError):
            validate_dataset(self.reports, self.gold)

    def test_source_publication_date_cannot_exceed_report_cutoff(self):
        source = self.report["sources"][0]
        source["published_at"] = self.report["as_of"]
        validate_dataset(self.reports, self.gold)
        source["published_at"] = "2024-07-01"
        with self.assertRaisesRegex(ValueError, "published after report.as_of"):
            validate_dataset(self.reports, self.gold)
        del source["published_at"]
        validate_dataset(self.reports, self.gold)

    def test_exact_answer_and_unit_conversion(self):
        for value, unit in (("120.5", "億円"), ("12050", "百万円"), ("12050000000", "円")):
            with self.subTest(value=value, unit=unit):
                result = score_answer(self.item, self.answer(value, unit), self.report)
                self.assertEqual(result["outcome"], "correct")
                self.assertTrue(result["evidence_valid"])
                self.assertEqual(candidate_truth(self.item, {"value": value, "unit": unit}), "supported")

    def test_absolute_tolerance_inclusive_boundary(self):
        for value, outcome in (("120.45", "correct"), ("120.55", "correct"), ("120.5500000000000000000000000001", "incorrect"), ("120.4499999999999999999999999999", "incorrect")):
            with self.subTest(value=value):
                self.assertEqual(score_answer(self.item, self.answer(value), self.report)["outcome"], outcome)

    def test_decimal_unit_multiplication_does_not_round_at_default_precision(self):
        item = copy.deepcopy(self.item)
        item.update(value="123456789012345678901234567890.1", tolerance="0", accepted_units={"unit10": "10"})
        answer = self.answer("12345678901234567890123456789.01", "unit10")
        self.assertEqual(score_answer(item, answer, self.report)["outcome"], "correct")

    def test_wrong_and_unknown_unit_are_incorrect(self):
        self.assertEqual(score_answer(self.item, self.answer("118"), self.report)["outcome"], "incorrect")
        wrong_unit = score_answer(self.item, self.answer("120.5", "ドル"), self.report)
        self.assertEqual(wrong_unit["outcome"], "incorrect")
        self.assertIsNone(wrong_unit["normalized_value"])

    def test_percent_fraction_conversion_and_percentage_points_are_distinct(self):
        item = self.gold["items"][2]
        self.assertEqual(candidate_truth(item, {"value": "0.082", "unit": "割合"}), "supported")
        self.assertEqual(candidate_truth(item, {"value": "8.2", "unit": "ポイント"}), "refuted")

    def test_abstain_and_errors_are_not_incorrect_answers(self):
        for answer, outcome in (({"status": "abstain", "value": None}, "abstain"), ({"status": "error"}, "error"), ({"status": "unknown"}, "error"), ({"status": "abstain", "value": "120.5"}, "error")):
            with self.subTest(answer=answer):
                self.assertEqual(score_answer(self.item, answer, self.report)["outcome"], outcome)

    def test_malformed_answers_are_errors_and_not_check_candidates(self):
        for value in (None, True, "NaN", "Infinity", "1,205", "120.5億円", "1e9999999"):
            with self.subTest(value=value):
                self.assertEqual(score_answer(self.item, self.answer(value), self.report)["outcome"], "error")
                with self.assertRaises(ValueError):
                    candidate_truth(self.item, {"value": value, "unit": "億円"})
        with self.assertRaises(ValueError):
            decimal_value("0." + "1" * 1001)

    def test_correct_value_does_not_imply_valid_evidence(self):
        for evidence in ([], [{"source_id": "s2", "quote": "数値ID=n1; 値=120.5; 単位=億円"}], [{"source_id": "s1", "quote": "invented quote"}], [{"source_id": "s1", "quote": ""}], [{"source_id": "s1", "quote": "数値ID=n1; 値=120.5; 単位=億円"}, {"source_id": "s1", "quote": "invented"}]):
            with self.subTest(evidence=evidence):
                result = score_answer(self.item, self.answer(evidence=evidence), self.report)
                self.assertEqual(result["outcome"], "correct")
                self.assertFalse(result["evidence_valid"])
                self.assertIn("semantic entailment is not evaluated", result["evidence_note"])

    def test_exact_quote_check_explicitly_does_not_claim_entailment(self):
        result = score_answer(self.item, self.answer(evidence=[{"source_id": "s1", "quote": "【架空資料】"}]), self.report)
        self.assertTrue(result["evidence_valid"])
        self.assertIn("semantic entailment is not evaluated", result["evidence_note"])


if __name__ == "__main__":
    unittest.main()
