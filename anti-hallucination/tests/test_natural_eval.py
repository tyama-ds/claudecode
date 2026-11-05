"""Whole-report accounting must retain omissions, abstentions, and unknowns."""

import copy
import unittest

from factcheck.natural_eval import evaluate_claims


class NaturalEvaluationTests(unittest.TestCase):
    def test_all_truth_and_outcome_combinations_and_denominators(self):
        gold, checks, extracted = [], [], []
        for truth in ("correct", "incorrect", "unresolved"):
            for state in ("supported", "refuted", "insufficient", "not_extracted", "unjudged"):
                claim_id = truth + "-" + state
                gold.append({"claim_id": claim_id, "truth": truth})
                if state != "not_extracted":
                    extracted.append(claim_id)
                if state in ("supported", "refuted", "insufficient"):
                    checks.append({"claim_id": claim_id, "verdict": state})
        result = evaluate_claims(gold, checks, extracted_claim_ids=extracted)
        self.assertEqual(result["counts"]["claims"], 15)
        self.assertEqual(result["counts"]["extracted"], 12)
        self.assertEqual(result["counts"]["judged"], 9)
        self.assertEqual(result["coverage"]["extraction_recall"], 12 / 15)
        self.assertEqual(result["coverage"]["judged_fraction"], 9 / 15)
        self.assertEqual(result["error_detection"]["detected"], 1)
        self.assertEqual(result["error_detection"]["missed"], 3)
        self.assertEqual(result["error_detection"]["abstained"], 1)
        self.assertEqual(result["error_detection"]["detection_rate"], 1 / 5)
        self.assertEqual(result["error_detection"]["not_detected_including_abstention"], 4)
        self.assertEqual(result["correct_claims"]["false_refutations"], 1)
        self.assertEqual(result["correct_claims"]["insufficient"], 1)
        self.assertEqual(result["correct_claims"]["false_refutation_rate"], 1 / 5)
        self.assertEqual(result["unresolved_claims"], {
            "supported": 1, "refuted": 1, "insufficient": 1, "not_extracted": 1, "unjudged": 1,
        })
        gate = result["support_only_gate"]
        self.assertEqual(gate["provided"], 3)
        self.assertEqual(gate["withheld"], 12)
        self.assertEqual(gate["correct_retention_rate"], 1 / 5)
        self.assertEqual(gate["known_error_reduction_rate"], 4 / 5)
        self.assertEqual(gate["provided_by_truth_per_original_claim"]["incorrect"], 1 / 15)
        self.assertEqual(gate["unresolved_fraction_among_provided"], 1 / 3)
        self.assertEqual(result["no_check"]["provided_by_truth_per_original_claim"]["incorrect"], 5 / 15)

    def test_all_withheld_is_not_detection_or_preserved_correctness(self):
        gold = [{"claim_id": "good", "truth": "correct"}, {"claim_id": "bad", "truth": "incorrect"}]
        checks = [{"claim_id": row["claim_id"], "verdict": "insufficient"} for row in gold]
        result = evaluate_claims(gold, checks, extracted_claim_ids=["good", "bad"])
        self.assertEqual(result["error_detection"]["detection_rate"], 0)
        self.assertEqual(result["error_detection"]["abstained"], 1)
        self.assertEqual(result["correct_claims"]["false_refutations"], 0)
        gate = result["support_only_gate"]
        self.assertEqual(gate["known_error_reduction_rate"], 1)
        self.assertEqual(gate["correct_retention_rate"], 0)
        self.assertEqual(gate["provided_fraction"], 0)
        self.assertIsNone(gate["known_error_fraction_among_provided"])

    def test_no_reference_errors_means_undefined_reduction_not_perfect_score(self):
        result = evaluate_claims(
            [{"claim_id": "c", "truth": "correct"}],
            [{"claim_id": "c", "verdict": "supported"}], extracted_claim_ids=["c"],
        )
        self.assertIsNone(result["error_detection"]["detection_rate"])
        self.assertIsNone(result["support_only_gate"]["known_error_reduction_rate"])
        self.assertEqual(result["support_only_gate"]["correct_retention_rate"], 1)

    def test_unknown_truth_is_never_counted_as_a_correct_abstention(self):
        result = evaluate_claims(
            [{"claim_id": "u", "truth": "unresolved"}],
            [{"claim_id": "u", "verdict": "insufficient"}], extracted_claim_ids=["u"],
        )
        self.assertEqual(result["counts"]["truth"], {"correct": 0, "incorrect": 0, "unresolved": 1})
        self.assertEqual(result["unresolved_claims"]["insufficient"], 1)
        self.assertIsNone(result["correct_claims"]["false_refutation_rate"])
        self.assertNotIn("accuracy", result)

    def test_missing_extraction_and_missing_decision_remain_distinct(self):
        gold = [{"claim_id": key, "truth": "incorrect"} for key in ("not_found", "not_checked")]
        result = evaluate_claims(gold, [], extracted_claim_ids=["not_checked"])
        self.assertEqual(result["error_detection"]["missed_not_extracted"], 1)
        self.assertEqual(result["error_detection"]["missed_unjudged"], 1)
        self.assertEqual(result["error_detection"]["miss_rate"], 1)
        self.assertEqual([row["state"] for row in result["claims"]], ["not_extracted", "unjudged"])

    def test_empty_inventory_has_no_defined_rates(self):
        result = evaluate_claims([], [], extracted_claim_ids=[])
        self.assertEqual(result["counts"]["claims"], 0)
        self.assertIsNone(result["coverage"]["extraction_recall"])
        self.assertIsNone(result["support_only_gate"]["provided_fraction"])

    def test_duplicates_unknown_ids_and_unextracted_decisions_are_rejected(self):
        gold = [{"claim_id": "c", "truth": "correct"}]
        check = {"claim_id": "c", "verdict": "supported"}
        invalid = [
            (gold + gold, [], ["c"]),
            (gold, [check, check], ["c"]),
            (gold, [], ["c", "c"]),
            (gold, [], ["unknown"]),
            (gold, [{"claim_id": "unknown", "verdict": "supported"}], ["c"]),
            (gold, [check], []),
        ]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                evaluate_claims(args[0], args[1], extracted_claim_ids=args[2])

    def test_invalid_shapes_and_labels_are_rejected(self):
        invalid = [
            ({}, [], []), ([], {}, []), ([], [], {}),
            ([None], [], []), ([{"claim_id": " ", "truth": "correct"}], [], []),
            ([{"claim_id": "c", "truth": "supported"}], [], []),
            ([{"claim_id": "c", "truth": "correct"}], [{"claim_id": "c", "verdict": "correct"}], ["c"]),
            ([], [], [None]),
        ]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                evaluate_claims(args[0], args[1], extracted_claim_ids=args[2])

    def test_inputs_are_not_modified_or_shared_with_output(self):
        gold = [{"claim_id": "c", "truth": "correct", "private_notes": ["review"]}]
        checks = [{"claim_id": "c", "verdict": "supported", "evidence": []}]
        extracted = ["c"]
        before = copy.deepcopy((gold, checks, extracted))
        result = evaluate_claims(gold, checks, extracted_claim_ids=extracted)
        result["claims"][0]["truth"] = "incorrect"
        self.assertEqual((gold, checks, extracted), before)
        self.assertNotIn("private_notes", result["claims"][0])


if __name__ == "__main__":
    unittest.main()
