import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from factcheck.recorded import (
    ANSWER_VERSION, MODEL_IDENTITY, _check_packet, grade_recorded, prepare_check, prepare_fill, read_json,
)
from factcheck.dataset import validate_dataset


class RecordedPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.reports = self.write("reports.json", {"reports": [{
            "id": "r1", "title": "Public reference", "as_of": "2025-01-01",
            "text": "A: {{n1}} units; B: {{n2}} units.",
            "sources": [{"id": "s1", "title": "Source", "text": "A has 10 units. B has 20 units."}],
        }]})
        self.gold = self.write("gold.json", {"items": [{
            "id": identifier, "report_id": "r1", "placeholder": "{{" + identifier + "}}",
            "statement": entity + " has {{value}} units.", "value": value, "unit": "units", "tolerance": "0",
            "source_ids": ["s1"], "private_note": "ANSWER_KEY_ONLY_SENTINEL",
            "candidates": [{"value": wrong, "unit": "units", "origin": "synthetic_perturbation", "historical_verdict": "refuted"}],
        } for identifier, entity, value, wrong in (("n1", "A", "10", "11"), ("n2", "B", "20", "21"))]})
        self.fill_path = self.root / "fill_packet.json"
        self.fill = prepare_fill(self.reports, self.gold, self.fill_path)

    def write(self, filename, value):
        path = self.root / filename
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def answer_file(self, stage, condition, packet, answers):
        return self.write(f"{stage}.{condition}.json", {
            "schema_version": ANSWER_VERSION, "stage": stage, "packet_sha256": packet["packet_sha256"],
            "condition": condition, "provenance": {
                "agent_id": "/test/" + stage + "_" + condition, "collection_method": "codex_subagent",
                "model_identity": MODEL_IDENTITY,
            }, "answers": answers,
        })

    def fills(self):
        files = []
        for condition in self.fill["conditions"]:
            rows = []
            for task in self.fill["items"]:
                item = task["payload"]["item_id"]
                abstain = condition == "baseline" and item == "n2"
                value = "11" if condition == "baseline" else "10" if item == "n1" else "20"
                rows.append({"task_id": task["task_id"], "answer": {
                    "status": "abstain" if abstain else "answered", "value": None if abstain else value,
                    "unit": "units", "evidence": [] if abstain else [{"source_id": "s1", "quote": "A has 10 units." if item == "n1" else "B has 20 units."}],
                    "reason": "Fixture response.",
                }})
            files.append(self.answer_file("fill", condition, self.fill, rows))
        return files

    def checks(self, packet):
        files = []
        for condition in packet["conditions"]:
            rows = []
            for task in packet["items"]:
                item = task["payload"]["item_id"]
                correct = "10" if item == "n1" else "20"
                verdict = "supported" if condition == "baseline" or task["payload"]["candidate"]["value"] == correct else "refuted"
                rows.append({"task_id": task["task_id"], "answer": {
                    "verdict": verdict, "evidence": [{"source_id": "s1", "quote": "A has 10 units." if item == "n1" else "B has 20 units."}],
                    "reason": "Fixture verdict.",
                }})
            files.append(self.answer_file("check", condition, packet, rows))
        return files

    def prepared(self):
        fills = self.fills()
        check_path = self.root / "check_packet.json"
        packet = prepare_check(self.reports, self.gold, fills, check_path, fill_packet=self.fill_path)
        return fills, check_path, packet, self.checks(packet)

    def test_fill_packet_is_public_deterministic_and_never_overwritten(self):
        duplicate = prepare_fill(self.reports, self.gold, self.root / "fill_again.json")
        self.assertEqual(self.fill, duplicate)
        self.assertNotIn("ANSWER_KEY_ONLY_SENTINEL", self.fill_path.read_text(encoding="utf-8"))
        for task in self.fill["items"]:
            self.assertNotIn("value", task["payload"])
            self.assertNotIn("candidates", task["payload"])
            self.assertNotIn("source_ids", task["payload"])
            self.assertIn("[MASK]", task["payload"]["statement"])
        with self.assertRaises(FileExistsError):
            prepare_fill(self.reports, self.gold, self.fill_path)

    def test_common_pool_is_shuffled_opaque_and_unlabeled(self):
        fills, check_path, packet, _ = self.prepared()
        again = prepare_check(self.reports, self.gold, fills, self.root / "check_again.json", fill_packet=self.fill_path)
        self.assertEqual(packet, again)
        self.assertEqual(len(packet["items"]), 4)
        raw = check_path.read_text(encoding="utf-8")
        for forbidden in ("candidate-000", '"truth"', '"origins"', "ANSWER_KEY_ONLY_SENTINEL"):
            self.assertNotIn(forbidden, raw)
        for task in packet["items"]:
            self.assertRegex(task["task_id"], r"^check_[a-f0-9]{24}$")
            self.assertEqual(set(task["payload"]["candidate"]), {"value", "unit"})

    def test_grade_reuses_metrics_preserves_raw_bytes_and_unknown_resources(self):
        fills, check_path, packet, checks = self.prepared()
        output = self.root / "recorded_run"
        result = grade_recorded(self.reports, self.gold, self.fill_path, fills, check_path, checks, output)
        self.assertEqual(result["run_mode"], "recorded_agent_pilot")
        self.assertFalse(result["demo"])
        self.assertTrue(result["conditions_same_model"])
        self.assertEqual(result["counts"]["recorded_batches"], 4)
        self.assertIsNone(result["counts"]["calls_completed"])
        self.assertEqual(result["counts"]["imported_answers"], 12)
        self.assertEqual(result["fill_metrics"][0]["incorrect_rate"], 0.5)
        self.assertEqual(result["fill_metrics"][0]["abstain_rate"], 0.5)
        self.assertEqual(result["fill_metrics"][1]["correct_rate"], 1)
        self.assertEqual(result["check_metrics"][0]["passed_error_rate"], 0.5)
        self.assertEqual(result["check_metrics"][1]["error_detection_rate"], 1)
        pipeline = next(row for row in result["pipeline_metrics"] if row["generator_id"] == "baseline" and row["checker_id"] == "evidence_first")
        self.assertEqual(pipeline["not_provided_rate"], 1)
        self.assertEqual(pipeline["incorrect_provided_rate"], 0)
        for name in ("fills", "checks", "fill_metrics", "check_metrics"):
            for row in result[name]:
                for key in ("input_tokens", "output_tokens", "cost", "latency_seconds", "known_cost"):
                    self.assertIsNone(row[key], (name, key))
        self.assertTrue(any(origin["type"] == "synthetic_perturbation" for candidate in result["candidates"] for origin in candidate["origins"]))
        self.assertFalse(any(origin["type"] == "historical" for candidate in result["candidates"] for origin in candidate["origins"]))
        self.assertEqual((output / "raw.fill.baseline.json").read_bytes(), fills[0].read_bytes())
        self.assertEqual(result["provenance"]["batches"][0]["raw_file_sha256"], hashlib.sha256(fills[0].read_bytes()).hexdigest())
        self.assertEqual(result["provenance"]["check_packet_sha256"], packet["packet_sha256"])
        self.assertEqual(read_json(output / "summary.json")["run_id"], "recorded_run")
        logs = [json.loads(line) for line in (output / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(logs), 12)
        self.assertTrue(all(row["record_type"] == "imported_batch_answer" for row in logs))
        self.assertIsNone(logs[0]["latency_seconds"])
        for filename in ("fill_metrics.csv", "check_metrics.csv", "pipeline_metrics.csv", "provenance.json", "reports.json", "gold.json"):
            self.assertTrue((output / filename).is_file())
        with self.assertRaises(ValueError):
            grade_recorded(self.reports, self.gold, self.fill_path, fills, check_path, checks, output)

    def test_public_check_order_does_not_depend_on_private_control_insertion_order(self):
        fills, _, packet, _ = self.prepared()
        dataset = validate_dataset(read_json(self.reports), read_json(self.gold))
        batches = {condition: {"file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for condition, path in zip(self.fill["conditions"], fills)}
        private_pool = [{
            "id": f"candidate-{index}", "item_id": task["payload"]["item_id"], "report_id": "r1",
            **task["payload"]["candidate"], "truth": "supported", "origins": [{"type": "supported_control"}],
        } for index, task in enumerate(packet["items"])]
        forward, _ = _check_packet(dataset, self.fill, batches, private_pool, packet["shuffle_seed"])
        reversed_pool = list(reversed(copy.deepcopy(private_pool)))
        for candidate in reversed_pool:
            candidate["truth"] = "refuted"
            candidate["origins"] = [{"type": "synthetic_perturbation"}]
        reverse, _ = _check_packet(dataset, self.fill, batches, reversed_pool, packet["shuffle_seed"])
        self.assertEqual(forward, reverse)
        self.assertEqual(forward, packet)

    def test_rejects_missing_duplicate_unknown_and_cross_packet_answers(self):
        fills = self.fills()
        original = read_json(fills[0])
        mutations = []
        missing = copy.deepcopy(original)
        missing["answers"].pop()
        mutations.append(missing)
        duplicate = copy.deepcopy(original)
        duplicate["answers"].append(copy.deepcopy(duplicate["answers"][0]))
        mutations.append(duplicate)
        unknown = copy.deepcopy(original)
        unknown["answers"][0]["task_id"] = "unknown"
        mutations.append(unknown)
        mismatch = copy.deepcopy(original)
        mismatch["packet_sha256"] = "0" * 64
        mutations.append(mismatch)
        invalid = copy.deepcopy(original)
        invalid["answers"][0]["answer"]["value"] = 11
        mutations.append(invalid)
        for index, value in enumerate(mutations):
            bad = self.write(f"bad_{index}.json", value)
            output = self.root / f"rejected_{index}.json"
            with self.subTest(index=index), self.assertRaises(ValueError):
                prepare_check(self.reports, self.gold, [bad, fills[1]], output, fill_packet=self.fill_path)
            self.assertFalse(output.exists())
        with self.assertRaises(ValueError):
            prepare_check(self.reports, self.gold, [fills[0], fills[0]], self.root / "duplicate_condition.json")

    def test_changed_packet_or_fill_bytes_are_rejected_before_grade_output(self):
        fills, check_path, packet, checks = self.prepared()
        packet["items"][0]["payload"]["candidate"]["value"] = "999"
        altered = self.write("changed_packet.json", packet)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            grade_recorded(self.reports, self.gold, self.fill_path, fills, altered, checks, self.root / "bad_run")
        # Even reformatting a bound original answer batch changes its file hash.
        fills[0].write_bytes(fills[0].read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "original fill artifacts"):
            grade_recorded(self.reports, self.gold, self.fill_path, fills, check_path, checks, self.root / "bad_run")
        self.assertFalse((self.root / "bad_run").exists())

    def test_strict_json_and_extra_provenance_fields_are_rejected(self):
        duplicate = self.root / "duplicate.json"
        duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
        with self.assertRaises(ValueError):
            read_json(duplicate)
        fills = self.fills()
        batch = read_json(fills[0])
        batch["provenance"]["api_key"] = "not-allowed"
        bad = self.write("bad_provenance.json", batch)
        with self.assertRaises(ValueError):
            prepare_check(self.reports, self.gold, [bad, fills[1]], self.root / "rejected.json")

    def test_cli_route_and_custom_condition_prompt(self):
        custom = self.root / "custom_packet.json"
        prompts = self.write("prompts.json", {"baseline": "Custom baseline instruction."})
        cli = Path(__file__).resolve().parents[1] / "pilot.py"
        completed = subprocess.run([
            sys.executable, str(cli), "prepare-fill", "--reports", str(self.reports), "--gold", str(self.gold),
            "--output", str(custom), "--conditions", "baseline", "--condition-prompts", str(prompts),
        ], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        packet = read_json(custom)
        self.assertEqual(packet["conditions"], ["baseline"])
        self.assertEqual(packet["condition_prompts"]["baseline"], "Custom baseline instruction.")


if __name__ == "__main__":
    unittest.main()
