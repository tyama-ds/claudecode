"""Runner tests: shared inputs, complete denominators, persistence and isolation."""

import copy
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from factcheck.experiment import run_experiment


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "run"
        self.reports = {"reports": [{
            "id": "r1", "title": "試験レポート", "text": "A {{n1}} 件、B {{n2}} 件、C {{n3}} 件。",
            "as_of": "2024-03-31", "private_notes": "GOLD-PRIVATE-METADATA",
            "sources": [{"id": "s1", "title": "出典", "text": "n1=10; n2=20; n3=30",
                         "private_notes": "GOLD-PRIVATE-METADATA"}],
        }]}
        self.gold = {"items": [{
            "id": f"n{i}", "report_id": "r1", "placeholder": "{{n" + str(i) + "}}",
            "statement": "対象" + str(i) + "の件数は {{value}} 件。", "value": str(i * 10),
            "unit": "件", "tolerance": "0", "source_ids": ["s1"],
            "private_notes": "GOLD-PRIVATE-METADATA",
            "candidates": [{"value": "9", "unit": "件", "origin": "historical",
                            "historical_verdict": "refuted"}] if i == 1 else [],
        } for i in (1, 2, 3)]}
        self.config = {"models": [
            {"id": "A", "provider": "demo", "model": "careful"},
            {"id": "B", "provider": "demo", "model": "noisy"},
        ], "repeats": 1}
        self.calls = []

    def response(self, answer, payload, stage, error=None, cost=None):
        return {"answer": answer, "raw_text": json.dumps(answer, ensure_ascii=False),
                "input_tokens": 10, "output_tokens": 5, "latency_seconds": 0.1,
                "cost": cost, "error": error,
                "request": {"stage": stage, "payload": copy.deepcopy(payload)}}

    def fill_answer(self, value, status="answered"):
        return {"status": status, "value": value, "unit": "件", "reason": "試験回答",
                "evidence": [{"source_id": "s1", "quote": "n1=10; n2=20; n3=30"}]}

    def checker_answer(self, verdict):
        return {"verdict": verdict, "reason": "試験判定",
                "evidence": [{"source_id": "s1", "quote": "n1=10; n2=20; n3=30"}]}

    def fake_model(self, model, stage, payload):
        self.calls.append((model["id"], stage, copy.deepcopy(payload)))
        if stage == "fill":
            values = {"A": {"n1": "10", "n2": "21", "n3": None},
                      "B": {"n1": "11", "n2": "20", "n3": "error"}}
            value = values[model["id"]][payload["item_id"]]
            if value == "error":
                return self.response({"status": "error"}, payload, stage, error="mock network failure")
            return self.response(self.fill_answer(value, "answered" if value is not None else "abstain"), payload, stage)
        value = payload["candidate"]["value"]
        expected = {"n1": "10", "n2": "20", "n3": "30"}[payload["item_id"]]
        verdict = ("supported" if value == expected else "refuted") if model["id"] == "A" else (
            "insufficient" if value == "9" else "supported")
        return self.response(self.checker_answer(verdict), payload, stage)

    def run_fake(self, **kwargs):
        with patch("factcheck.experiment.call_model", side_effect=self.fake_model):
            return run_experiment(self.reports, self.gold, self.config, self.out, **kwargs)

    def one_item(self):
        self.config["models"] = self.config["models"][:1]
        self.reports["reports"][0]["text"] = "A {{n1}} 件。"
        self.gold["items"] = self.gold["items"][:1]
        self.gold["items"][0]["candidates"] = []

    def test_metrics_keep_wrong_abstained_and_failed_fills_in_denominator(self):
        result = self.run_fake()
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["demo"])
        a, b = result["fill_metrics"]
        self.assertEqual(a["n"], 3)
        self.assertEqual((a["n_correct"], a["n_incorrect"], a["n_abstain"], a["n_error"]), (1, 1, 1, 0))
        self.assertEqual((b["n_correct"], b["n_incorrect"], b["n_abstain"], b["n_error"]), (1, 1, 0, 1))
        self.assertAlmostEqual(a["correct_rate"], 1 / 3)
        self.assertAlmostEqual(b["error_rate"], 1 / 3)
        self.assertIsNone(a["cost"])
        self.assertEqual(a["cost_unknown_calls"], 3)
        self.assertEqual(result["counts"]["fill_completed"], 6)
        self.assertEqual(result["counts"]["check_completed"], 12)
        self.assertEqual(len(result["errors"]), 1)

    def test_all_checkers_get_the_same_pool_with_no_labels_or_model_provenance(self):
        result = self.run_fake()
        payloads = {model: [payload for mid, stage, payload in self.calls if mid == model and stage == "check"]
                    for model in ("A", "B")}
        self.assertEqual(payloads["A"], payloads["B"])
        self.assertEqual(len(payloads["A"]), 6)
        allowed = {"item_id", "report_title", "report_text", "statement", "unit", "as_of", "sources", "comparison_rules"}
        for _, stage, payload in self.calls:
            self.assertEqual(set(payload), allowed | ({"candidate"} if stage == "check" else set()))
            self.assertNotIn("GOLD-PRIVATE-METADATA", json.dumps(payload))
            self.assertEqual(set(payload["comparison_rules"]), {"absolute_tolerance", "target_unit", "accepted_units"})
            if stage == "check":
                self.assertEqual(set(payload["candidate"]), {"value", "unit"})
        ten = next(candidate for candidate in result["candidates"] if candidate["value"] == "10")
        self.assertEqual(len(ten["origins"]), 2)
        self.assertEqual({origin["type"] for origin in ten["origins"]}, {"supported_control", "generated"})

    def test_check_confusion_does_not_count_abstention_as_detection(self):
        result = self.run_fake()
        a, b = result["check_metrics"]
        self.assertEqual((a["n_supported"], a["n_refuted"]), (3, 3))
        self.assertEqual(a["error_detection_rate"], 1)
        self.assertEqual(a["passed_error_rate"], 0)
        self.assertEqual(a["pass_rate"], 0.5)
        self.assertEqual(b["error_detection_rate"], 0)
        self.assertEqual(b["confusion_matrix"]["refuted"]["insufficient"], 1)
        self.assertAlmostEqual(b["passed_error_rate"], 2 / 5)
        self.assertAlmostEqual(b["pass_rate"], 5 / 6)

    def test_candidate_origin_preserves_artificial_and_historical_categories(self):
        self.one_item()
        self.gold["items"][0]["candidates"] = [
            {"value": "9", "unit": "件", "origin": "synthetic_perturbation"},
            {"value": "8", "unit": "件", "origin": "historical", "historical_verdict": "supported"},
            {"value": "7", "unit": "件", "origin": "custom_external_source"},
            {"value": "6", "unit": "件"},
            {"value": "9.0", "unit": "件", "origin": "historical"},
        ]
        result = self.run_fake()
        pool = {candidate["value"]: candidate for candidate in result["candidates"]}
        self.assertEqual(pool["9"]["origins"], [
            {"type": "synthetic_perturbation", "index": 0, "declared_origin": "synthetic_perturbation", "declared_verdict": None},
            {"type": "historical", "index": 4, "declared_origin": "historical", "declared_verdict": None},
        ])
        self.assertEqual(pool["8"]["origins"][0]["type"], "historical")
        self.assertEqual(pool["8"]["origins"][0]["declared_verdict"], "supported")
        self.assertEqual(pool["7"]["origins"][0]["type"], "provided_candidate")
        self.assertEqual(pool["7"]["origins"][0]["declared_origin"], "custom_external_source")
        self.assertEqual(pool["6"]["origins"][0]["type"], "historical")
        self.assertEqual(pool["9"]["truth"], "refuted")
        self.assertEqual(pool["8"]["truth"], "refuted")

    def test_public_comparison_rules_reach_both_stages_without_gold_value(self):
        self.one_item()
        self.gold["items"][0]["tolerance"] = "0.1"
        self.gold["items"][0]["accepted_units"] = {"別単位": "0.01"}
        payloads = []

        def converted(model, stage, payload):
            payloads.append(copy.deepcopy(payload))
            answer = self.fill_answer("1000") if stage == "fill" else self.checker_answer("supported")
            if stage == "fill":
                answer["unit"] = "別単位"
            return self.response(answer, payload, stage)

        with patch("factcheck.experiment.call_model", side_effect=converted):
            result = run_experiment(self.reports, self.gold, self.config, self.out)
        for payload in payloads:
            self.assertEqual(payload["comparison_rules"], {
                "absolute_tolerance": "0.1", "target_unit": "件", "accepted_units": {"別単位": "0.01"}})
            self.assertNotIn("value", payload["comparison_rules"])
        self.assertEqual(result["fill_metrics"][0]["correct_rate"], 1)
        self.assertEqual(result["pipeline_metrics"][0]["correct_provided_rate"], 1)
        self.assertEqual(result["check_metrics"][0]["n_supported"], 2)
        self.assertIsNone(result["check_metrics"][0]["error_detection_rate"])

    def test_checker_errors_and_false_flags_have_distinct_confusion_cells(self):
        self.one_item()
        self.gold["items"][0]["candidates"] = [{"value": "9", "unit": "件", "origin": "historical"}]

        def flagged_or_error(model, stage, payload):
            if stage == "fill":
                return self.response(self.fill_answer("10"), payload, stage)
            if payload["candidate"]["value"] == "9":
                return self.response({"status": "error"}, payload, stage, error="network failure")
            return self.response(self.checker_answer("refuted"), payload, stage)

        with patch("factcheck.experiment.call_model", side_effect=flagged_or_error):
            result = run_experiment(self.reports, self.gold, self.config, self.out)
        check = result["check_metrics"][0]
        self.assertEqual((check["n"], check["n_completed"], check["n_error"]), (2, 2, 1))
        self.assertEqual(check["false_flag_rate"], 1)
        self.assertEqual(check["error_detection_rate"], 0)
        self.assertEqual(check["error_rate"], 0.5)
        self.assertEqual(check["abstain_rate"], 0)
        self.assertEqual(check["confusion_matrix"]["supported"]["refuted"], 1)
        self.assertEqual(check["confusion_matrix"]["refuted"]["error"], 1)
        self.assertEqual(check["pass_rate"], 0)
        self.assertIsNone(check["passed_error_rate"])
        self.assertEqual(result["pipeline_metrics"][0]["not_provided_rate"], 1)

    def test_cross_pipeline_pairs_each_generation_with_each_checker(self):
        result = self.run_fake()
        rows = {(row["generator_id"], row["checker_id"]): row for row in result["pipeline_metrics"]}
        self.assertEqual(set(rows), {("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")})
        for key, row in rows.items():
            self.assertEqual(row["n"], 3)
            self.assertAlmostEqual(row["correct_provided_rate"], 1 / 3)
            self.assertAlmostEqual(row["incorrect_provided_rate"], 0 if key[1] == "A" else 1 / 3)
            self.assertAlmostEqual(row["not_provided_rate"], 2 / 3 if key[1] == "A" else 1 / 3)

    def test_repetition_pairing_uses_matching_checker_repetition(self):
        self.one_item()
        self.config["repeats"] = 2
        counters = {}

        def repeated(model, stage, payload):
            key = (stage, payload.get("candidate", {}).get("value"))
            counters[key] = counters.get(key, 0) + 1
            repetition = counters[key]
            answer = self.fill_answer("10" if repetition == 1 else "11") if stage == "fill" else (
                self.checker_answer("supported" if repetition == 1 else "refuted"))
            return self.response(answer, payload, stage, cost=0.01)

        with patch("factcheck.experiment.call_model", side_effect=repeated):
            result = run_experiment(self.reports, self.gold, self.config, self.out)
        row = result["pipeline_metrics"][0]
        self.assertEqual((row["n_correct_provided"], row["n_incorrect_provided"], row["n_not_provided"]), (1, 0, 1))
        self.assertAlmostEqual(result["resources"]["cost"], 0.06)

    def test_cancel_before_start_retains_requested_denominators(self):
        result = self.run_fake(cancel=lambda: True)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.calls, [])
        self.assertEqual(result["counts"]["fill_requested"], 6)
        self.assertFalse(result["counts"]["candidate_pool_complete"])
        self.assertFalse(result["counts"]["calls_requested_final"])
        self.assertEqual(result["fill_metrics"][0]["not_run_rate"], 1)
        self.assertEqual(result["pipeline_metrics"][0]["not_run_rate"], 1)
        self.assertEqual(result["pipeline_metrics"][0]["not_provided_rate"], 0)
        self.assertIsNone(result["check_metrics"][0]["passed_error_rate"])
        self.assertTrue((self.out / "summary.json").exists())

    def test_cancel_during_checks_does_not_label_pending_work_as_abstention(self):
        self.config["models"] = self.config["models"][:1]
        result = self.run_fake(cancel=lambda: len(self.calls) >= 4)
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(result["counts"]["candidate_pool_complete"])
        self.assertEqual(result["counts"]["check_completed"], 1)
        row = result["pipeline_metrics"][0]
        self.assertEqual((row["n_correct_provided"], row["n_not_provided"], row["n_not_run"]), (1, 1, 1))
        self.assertEqual(result["check_metrics"][0]["n_abstain"], 0)

    def test_logs_are_flushed_before_the_next_call_and_all_exports_exist(self):
        original = self.fake_model

        def checking_log(model, stage, payload):
            existing = (self.out / "calls.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(existing), len(self.calls))
            return original(model, stage, payload)

        with patch("factcheck.experiment.call_model", side_effect=checking_log):
            result = run_experiment(self.reports, self.gold, self.config, self.out)
        records = [json.loads(line) for line in (self.out / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), result["counts"]["calls_completed"])
        self.assertTrue(all("request" in row and "raw_text" in row and "report_id" in row for row in records))
        for name in ("reports.json", "gold.json", "models.json", "summary.json", "fills.csv", "checks.csv",
                     "candidates.csv", "fill_metrics.csv", "check_metrics.csv", "pipeline_metrics.csv", "errors.csv"):
            self.assertTrue((self.out / name).exists(), name)
        saved = json.loads((self.out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, result)

    def test_provider_exception_is_an_error_trial_not_a_dropped_row(self):
        self.one_item()

        def raises(model, stage, payload):
            if stage == "fill":
                raise RuntimeError("connection failed")
            return self.response(self.checker_answer("insufficient"), payload, stage)

        with patch("factcheck.experiment.call_model", side_effect=raises):
            result = run_experiment(self.reports, self.gold, self.config, self.out)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["fill_metrics"][0]["error_rate"], 1)
        self.assertEqual(result["pipeline_metrics"][0]["not_provided_rate"], 1)
        self.assertIsNone(result["check_metrics"][0]["passed_error_rate"])
        self.assertEqual(len(result["candidates"]), 1)

    def test_non_numeric_output_is_an_error_and_never_a_candidate(self):
        self.one_item()

        def malformed(model, stage, payload):
            answer = self.fill_answer("NaN") if stage == "fill" else self.checker_answer("supported")
            return self.response(answer, payload, stage)

        with patch("factcheck.experiment.call_model", side_effect=malformed):
            result = run_experiment(self.reports, self.gold, self.config, self.out)
        self.assertEqual(result["fill_metrics"][0]["error_rate"], 1)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertIsNone(result["fills"][0]["candidate_id"])
        self.assertEqual(result["pipeline_metrics"][0]["not_provided_rate"], 1)

    def test_credential_validation_precedes_all_calls_and_output(self):
        self.config["models"] = [{"id": "live", "provider": "openai_chat", "model": "test-model",
                                  "api_key_env": "FACTCHECK_TEST_MISSING_KEY"}]
        with patch.dict(os.environ, {"FACTCHECK_TEST_MISSING_KEY": ""}), patch("factcheck.experiment.call_model") as model:
            with self.assertRaises(ValueError):
                run_experiment(self.reports, self.gold, self.config, self.out)
            model.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_secrets_are_redacted_from_errors_raw_logs_and_results(self):
        self.one_item()
        self.config["models"][0]["api_key_env"] = "FACTCHECK_TEST_KEY"
        secret = "TEST-SECRET-MUST-NEVER-BE-SAVED"

        def response_with_secret(model, stage, payload):
            answer = self.fill_answer("10") if stage == "fill" else self.checker_answer("supported")
            answer["reason"] = secret
            response = self.response(answer, payload, stage)
            response["raw_text"] = secret
            return response

        with patch.dict(os.environ, {"FACTCHECK_TEST_KEY": secret}), patch("factcheck.experiment.call_model", side_effect=response_with_secret):
            result = run_experiment(self.reports, self.gold, self.config, self.out)
        self.assertNotIn(secret, json.dumps(result))
        for path in self.out.iterdir():
            self.assertNotIn(secret, path.read_text(encoding="utf-8-sig"), path.name)
        self.assertIn("[REDACTED]", (self.out / "calls.jsonl").read_text(encoding="utf-8"))

    def test_csv_formula_cells_are_escaped(self):
        self.one_item()

        def okay(model, stage, payload):
            answer = self.fill_answer("10") if stage == "fill" else self.checker_answer("supported")
            if stage == "fill":
                answer["unit"] = "=HYPERLINK(\"evil\")"
            return self.response(answer, payload, stage)

        with patch("factcheck.experiment.call_model", side_effect=okay):
            run_experiment(self.reports, self.gold, self.config, self.out)
        with (self.out / "candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(rows[1]["unit"].startswith("'="))

    def test_failed_runner_still_saves_partial_summary_and_raw_call(self):
        self.one_item()
        with patch("factcheck.experiment.call_model", side_effect=self.fake_model), patch("factcheck.experiment.score_answer", side_effect=RuntimeError("scoring fault")):
            result = run_experiment(self.reports, self.gold, self.config, self.out)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["counts"]["calls_completed"], 1)
        self.assertEqual(result["counts"]["fill_not_run"], 1)
        self.assertEqual(len((self.out / "calls.jsonl").read_text(encoding="utf-8").splitlines()), 1)
        self.assertEqual(json.loads((self.out / "summary.json").read_text(encoding="utf-8"))["status"], "failed")

    def test_existing_experiment_is_not_overwritten(self):
        self.run_fake()
        old = (self.out / "calls.jsonl").read_bytes()
        with patch("factcheck.experiment.call_model") as model:
            with self.assertRaises(ValueError):
                run_experiment(self.reports, self.gold, self.config, self.out)
            model.assert_not_called()
        self.assertEqual((self.out / "calls.jsonl").read_bytes(), old)

    def test_progress_has_stage_local_totals_and_finishes_after_saving(self):
        events = []

        def progress(event):
            events.append(event)
            if event["stage"] == "completed":
                self.assertTrue((self.out / "summary.json").exists())

        self.run_fake(progress=progress)
        self.assertEqual(events[0]["stage"], "fill")
        self.assertEqual(events[-1]["stage"], "completed")
        self.assertTrue(all(0 <= event["completed"] <= event["total"] for event in events))


if __name__ == "__main__":
    unittest.main()
