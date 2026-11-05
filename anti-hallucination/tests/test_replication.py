"""Small structural tests; no model calls or research labels are synthesized for the trial."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import replication as rep


def execution(*, repair=False):
    result = {"started_utc": "2026-11-01T00:00:00Z", "finished_utc": "2026-11-01T00:01:00Z",
              "web_calls": 0, "input_files": []}
    if repair:
        result["model_id"] = None
    return result


def records():
    original = {"text": "Alpha. Beta. Proposal.", "sources": []}
    repair = {
        "execution": execution(repair=True),
        "findings": [{"finding_id": "F1", "quote": "Alpha.", "tags": ["factual"],
                      "status": "confirmed_problem", "diagnosis": "Fixture diagnosis.", "evidence": []}],
        "checked_controls": [], "unresolved": [], "limitations": [], "added_sources": [],
        "patches": [{"patch_id": "P1", "finding_ids": ["F1"], "before": "Alpha.", "after": "Gamma.",
                     "reason": "PRIVATE PATCH JUSTIFICATION", "source_urls": [], "action": "replace"}],
    }
    baseline = {
        "execution": execution(), "human_adjudicated": False, "inventory": [], "limitations": [],
        "issues": [{"issue_id": "B1", "quote": "Alpha.", "tags": ["factual"], "status": "confirmed_problem",
                    "reason": "Fixture.", "source_urls": [], "evidence": "Fixture.", "minimum_resolution": "Fixture."},
                   {"issue_id": "B2", "quote": "Beta.", "tags": ["condition"], "status": "unresolved",
                    "reason": "Fixture.", "source_urls": [], "evidence": "Fixture.", "minimum_resolution": "Fixture."}],
        "controls": [{"control_id": "K1", "quote": "Proposal.", "description": "Fixture proposal.",
                      "source_urls": [], "reason": "Fixture."}],
    }
    post = {
        "execution": execution(), "human_adjudicated": False, "limitations": [],
        "issues": [{"issue_id": "B1", "baseline_status": "confirmed_problem", "detected": True,
                    "finding_ids": ["F1"], "detection_reason": "Fixture semantic match.", "after_status": "resolved",
                    "after_quote": "Gamma.", "resolved_by_removal": False, "reason": "Fixture.", "source_urls": []},
                   {"issue_id": "B2", "baseline_status": "unresolved", "detected": False,
                    "finding_ids": [], "detection_reason": "Fixture no match.", "after_status": "unverifiable",
                    "after_quote": "Beta.", "resolved_by_removal": False, "reason": "Fixture.", "source_urls": []}],
        "patch_reviews": [{"patch_id": "P1", "status": "acceptable", "reason": "Fixture.", "new_problems": []}],
        "controls": [{"control_id": "K1", "status": "preserved", "after_quote": "Proposal.", "reason": "Fixture."}],
        "false_findings": [], "posthoc_original_findings": [],
    }
    return original, repair, baseline, post


class ReplicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "trial"
        self.root.mkdir()
        for name in rep.FIXED_FILES:
            (self.root / name).write_text("{}", encoding="utf-8")
        protocol = {"id": "test", "topics": [{"report_id": f"R{i}", "theme": f"Fixture {i}"} for i in (1, 2, 3)]}
        (self.root / "protocol.json").write_bytes(rep._json_bytes(protocol))
        prereg = {"fixed_at_utc": "2026-11-01T00:00:00Z", "generation_started": False,
                  "sha256": {name: rep._sha(self.root / name) for name in rep.FIXED_FILES}}
        (self.root / "preregistration.json").write_bytes(rep._json_bytes(prereg))
        self.original, self.repair, self.baseline, self.post = records()

    def write_inputs(self, *, with_post=False):
        data = list(zip(rep.RAW_FILES, (self.original, self.repair, self.baseline)))
        if with_post:
            data.append(("evaluation/post.raw.json", self.post))
        for name, record in data:
            path = self.root / "R1" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(rep._json_bytes(record))

    def revised(self):
        return rep.validate_repair(self.original, self.repair)

    def validate_post(self):
        rep.validate_post(self.original, self.revised(), self.repair, self.baseline, self.post)

    def test_missing_reports_stay_in_planned_denominator_and_are_not_zero(self):
        summary = rep.build_report(self.root, self.base / "report")
        self.assertEqual(summary["aggregate"]["planned_reports"], 3)
        self.assertEqual(summary["aggregate"]["incomplete_reports"], 3)
        self.assertEqual(summary["aggregate"]["generation_failed_reports"], 0)
        self.assertIsNone(summary["aggregate"]["complete_reports_only"])
        self.assertTrue(all(row["metrics"] is None for row in summary["reports"]))

    def test_confirmed_unresolved_exclusions_and_empty_denominator_are_separate(self):
        self.validate_post()
        metrics = rep.summarize_complete(self.original, self.revised(), self.repair, self.baseline, self.post)
        self.assertEqual(metrics["confirmed_denominator"], 1)
        self.assertEqual(metrics["baseline_unresolved"], 1)
        self.assertEqual(metrics["baseline_unresolved_after_status"]["unverifiable"], 1)
        self.post["issues"][0]["after_status"] = "not_an_original_problem"
        self.validate_post()
        metrics = rep.summarize_complete(self.original, self.revised(), self.repair, self.baseline, self.post)
        self.assertEqual(metrics["confirmed_excluded_not_original_problem"], 1)
        self.assertEqual(metrics["excluded_issue_ids"], ["B1"])
        self.assertEqual(metrics["confirmed_denominator"], 0)
        self.assertIsNone(metrics["detection_rate"])
        self.assertIsNone(metrics["resolution_rate"])

    def test_complete_prepare_report_raw_immutability_and_reason_blinding(self):
        self.write_inputs(with_post=True)
        raw_paths = [self.root / "R1" / name for name in (*rep.RAW_FILES, "evaluation/post.raw.json")]
        before = {path: path.read_bytes() for path in raw_paths}
        original_objects = copy.deepcopy((self.original, self.repair, self.baseline, self.post))
        rep.prepare_post(self.root, "R1")
        rep.prepare_post(self.root, "R1")  # Identical reexecution is safe.
        packet_path = self.root / "R1/evaluation/post_packet.json"
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        self.assertNotIn("PRIVATE PATCH JUSTIFICATION", packet_path.read_text(encoding="utf-8"))
        self.assertEqual(packet["writer_findings"], self.repair["findings"])
        self.assertEqual(packet["revised"]["text"], "Gamma. Beta. Proposal.")
        self.assertEqual(packet["input_sha256"][rep.RAW_FILES[0]], hashlib.sha256(before[raw_paths[0]]).hexdigest())
        summary = rep.build_report(self.root, self.base / "report")
        self.assertEqual(summary["aggregate"]["complete_reports"], 1)
        self.assertEqual(summary["aggregate"]["incomplete_reports"], 2)
        self.assertEqual(summary["reports"][0]["metrics"]["confirmed_detected"], 1)
        self.assertEqual(before, {path: path.read_bytes() for path in raw_paths})
        self.assertEqual(original_objects, (self.original, self.repair, self.baseline, self.post))
        rep.build_report(self.root, self.base / "report")

    def test_unknown_finding_ids_and_incomplete_id_sets_are_rejected(self):
        mutations = [
            lambda p: p["issues"][0].update(finding_ids=["UNKNOWN"]),
            lambda p: p["issues"].pop(),
            lambda p: p["controls"].clear(),
            lambda p: p["patch_reviews"][0].update(patch_id="UNKNOWN"),
            lambda p: p["issues"][0].update(finding_ids=[]),
            lambda p: p["false_findings"].append({"finding_id": "UNKNOWN", "status": "false_alarm", "reason": "x"}),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.post = records()[3]
                mutate(self.post)
                with self.assertRaises(ValueError):
                    self.validate_post()

    def test_real_json_booleans_quotes_and_status_enums(self):
        mutations = [
            lambda p: p["issues"][0].update(detected="false"),
            lambda p: p["issues"][0].update(resolved_by_removal=0),
            lambda p: p.update(human_adjudicated="false"),
            lambda p: p["issues"][0].update(after_quote="Not in the report"),
            lambda p: p["issues"][0].update(after_status="fixed"),
            lambda p: p["issues"][0].update(baseline_status="unresolved"),
            lambda p: p["controls"][0].update(status="okay"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.post = records()[3]
                mutate(self.post)
                with self.assertRaises(ValueError):
                    self.validate_post()

    def test_new_problems_unnecessary_false_findings_and_posthoc_not_combined(self):
        self.post["patch_reviews"][0].update(status="unnecessary_change", new_problems=[{
            "problem_id": "N1", "tags": ["information_loss", "meaning_drift"], "status": "unresolved",
            "quote": "Gamma.", "reason": "Fixture.", "source_urls": [],
        }])
        self.post["false_findings"] = [{"finding_id": "F1", "status": "unverifiable", "reason": "Fixture."}]
        self.post["posthoc_original_findings"] = [{"quote": "Beta.", "tags": ["condition"],
            "status": "confirmed_problem", "reason": "Fixture.", "still_present_after": True,
            "after_quote": "Beta.", "source_urls": []}]
        self.validate_post()
        metrics = rep.summarize_complete(self.original, self.revised(), self.repair, self.baseline, self.post)
        self.assertEqual(metrics["confirmed_denominator"], 1)
        self.assertEqual(metrics["patch_reviews"]["unnecessary_change"], 1)
        self.assertEqual(metrics["new_problems"], {"confirmed_problem": 0, "unresolved": 1})
        self.assertEqual(metrics["false_findings"]["unverifiable"], 1)
        self.assertEqual(metrics["posthoc_original_findings"]["confirmed_problem"], 1)

    def test_frozen_raw_change_and_different_existing_outputs_are_rejected(self):
        self.write_inputs()
        rep.prepare_post(self.root, "R1")
        revised_path = self.root / "R1/repair/revised_report.txt"
        before = revised_path.read_bytes()
        self.baseline["limitations"].append("Changed after freeze")
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Frozen input missing or changed"):
            rep.prepare_post(self.root, "R1")
        self.assertEqual(revised_path.read_bytes(), before)
        summary = rep.build_report(self.root, self.base / "report")
        self.assertEqual(summary["reports"][0]["status"], "invalid")
        self.assertIsNone(summary["reports"][0]["metrics"])

    def test_different_derived_output_is_never_overwritten(self):
        self.write_inputs()
        target = self.root / "R1/repair/revised_report.txt"
        target.write_text("Preexisting different result", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            rep.prepare_post(self.root, "R1")
        self.assertEqual(target.read_text(encoding="utf-8"), "Preexisting different result")
        self.assertFalse((self.root / "R1/evaluation/post_packet.json").exists())

    def test_independent_baseline_freeze_is_verified(self):
        self.write_inputs()
        freeze = self.root / "R1/evaluation/baseline.freeze.json"
        freeze.write_bytes(rep._json_bytes({"frozen_at_utc": "2026-11-01T00:01:00Z",
            "sha256": rep._sha(self.root / "R1/evaluation/baseline.raw.json")}))
        rep.prepare_post(self.root, "R1")
        record = rep._read_json(self.root / "R1/evaluation/input_freeze.json")
        self.assertEqual(record["independent_baseline_freeze"]["record_sha256"], rep._sha(freeze))
        freeze.write_bytes(rep._json_bytes({"frozen_at_utc": "2026-11-01T00:01:00Z", "sha256": "0" * 64}))
        with self.assertRaisesRegex(ValueError, "baseline.freeze SHA256 mismatch"):
            rep.prepare_post(self.root, "R1")

    def test_preregistered_instruction_change_is_rejected(self):
        (self.root / "repair_instructions.txt").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Preregistration SHA256 mismatch"):
            rep.build_report(self.root, self.base / "report")


if __name__ == "__main__":
    unittest.main()
