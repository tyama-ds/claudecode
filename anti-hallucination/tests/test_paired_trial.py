"""Structural paired-trial safeguards using synthetic fixtures only."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import paired_trial as paired
import paired_review
import replication as rep


_fixture_spec = importlib.util.spec_from_file_location(
    "replication_fixtures", Path(__file__).with_name("test_replication.py"))
_fixtures = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixtures)


class PairedTrialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "trial"
        self.root.mkdir()
        self.protocol = {
            "id": "paired-fixture",
            "documents": [
                {"id": f"R{i}", "cohort": "regression", "path": f"regression/R{i}",
                 "arms": ["current", "improved"]} for i in (1, 2, 3)
            ] + [
                {"id": f"N{i}", "cohort": "new", "path": f"new/N{i}",
                 "arms": ["A", "B"]} for i in (1, 2)
            ],
            "arm_mapping": {
                "N1": {"A": "current", "B": "improved"},
                "N2": {"A": "improved", "B": "current"},
            },
            "limits": ["Synthetic fixtures, not experimental observations."],
        }
        self.fixed_files = (
            "protocol.json", "instructions/repair_v1.txt", "instructions/repair_v2.txt",
            "instructions/reference.txt", "instructions/post.txt",
        )
        for name in self.fixed_files[1:]:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Fixed instruction " + name, encoding="utf-8")
        self.write_json(self.root / "protocol.json", self.protocol)
        self.preregister()
        self.original, self.repair, self.baseline, self.post = _fixtures.records()

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rep._json_bytes(value))

    def preregister(self, extra=()):
        self.write_json(self.root / "preregistration.json", {
            "fixed_at_utc": "2026-11-01T00:00:00Z", "generation_started": False,
            "sha256": {name: rep._sha(self.root / name) for name in (*self.fixed_files, *extra)},
        })

    def folder(self, doc_id="R1"):
        document = next(row for row in self.protocol["documents"] if row["id"] == doc_id)
        return self.root / document["path"]

    def write_common(self, doc_id="R1", *, freeze=True):
        folder = self.folder(doc_id)
        self.write_json(folder / "input/original_report.json", self.original)
        self.write_json(folder / "evaluation/baseline.raw.json", self.baseline)
        if freeze:
            self.write_json(folder / "evaluation/baseline.freeze.json", {
                "frozen_at_utc": "2026-11-01T00:01:00Z",
                "sha256": rep._sha(folder / "evaluation/baseline.raw.json"),
            })
        return folder

    def write_arm(self, arm, doc_id="R1", *, post=None, prepare=True, with_post=True):
        folder = self.folder(doc_id) / "arms" / arm
        self.write_json(folder / "repair.raw.json", self.repair)
        if prepare:
            paired.prepare(self.root, doc_id, arm)
        if with_post:
            self.write_json(folder / "post.raw.json", self.post if post is None else post)
        return folder

    def report(self, name="report"):
        return paired.build_report(self.root, self.base / name)

    def document(self, summary, doc_id="R1"):
        return next(row for row in summary["documents"] if row["id"] == doc_id)

    def use_v2(self):
        self.protocol.update(schema_version=2, planned_document_ids=["R1", "R2", "N1", "N2"],
                             evaluation_arm_mapping={"R1": {"A": "current", "B": "improved"},
                                                     "R2": {"A": "improved", "B": "current"}})
        self.protocol["documents"] = [doc for doc in self.protocol["documents"] if doc["id"] != "R3"]
        for name in paired.GUARD_FIXED_FILES:
            (self.root / name).write_text("Frozen guard fixture", encoding="utf-8")
        self.fixed_files += paired.GUARD_FIXED_FILES
        self.write_json(self.root / "protocol.json", self.protocol)
        self.preregister()

    def prepare_joint(self, doc_id="R1"):
        folder = self.write_common(doc_id)
        for arm in next(doc for doc in self.protocol["documents"] if doc["id"] == doc_id)["arms"]:
            self.write_arm(arm, doc_id, with_post=False)
        paired.prepare_post(self.root, doc_id)
        return folder

    def joint_response(self, path=None):
        response = {"execution": copy.deepcopy(self.post["execution"]), "human_adjudicated": False,
                    "arms": {"A": copy.deepcopy(self.post), "B": copy.deepcopy(self.post)},
                    "cross_arm_notes": ["Synthetic paired audit."]}
        if path is not None:
            self.write_json(path, response)
        return response

    def test_v2_retains_explicit_four_document_plan_and_null_missing_pairs(self):
        self.use_v2()
        summary = self.report()
        self.assertEqual([row["id"] for row in summary["documents"]], ["R1", "R2", "N1", "N2"])
        for cohort in ("regression", "new"):
            self.assertEqual(summary["cohorts"][cohort]["planned_documents"], 2)
            self.assertIsNone(summary["cohorts"][cohort]["paired_primary"])
        self.assertTrue(all(row["paired"] is None for row in summary["documents"]))

    def test_v2_rejects_missing_extra_duplicate_and_empty_planned_ids(self):
        self.use_v2()
        valid = copy.deepcopy(self.protocol)
        cases = [
            {"documents": valid["documents"][:-1]},
            {"planned_document_ids": valid["planned_document_ids"][:-1]},
            {"planned_document_ids": valid["planned_document_ids"] + ["N2"]},
            {"planned_document_ids": valid["planned_document_ids"] + [""]},
            {"planned_document_ids": []},
        ]
        for change in cases:
            with self.subTest(change=change):
                self.write_json(self.root / "protocol.json", {**valid, **change})
                self.preregister()
                with self.assertRaisesRegex(ValueError, "planned_document_ids"):
                    paired.validate_preregistration(self.root)

    def test_v2_requires_frozen_guard_instructions_and_contract(self):
        self.use_v2()
        for name in paired.GUARD_FIXED_FILES:
            with self.subTest(name=name):
                self.preregister()
                record = rep._read_json(self.root / "preregistration.json")
                del record["sha256"][name]
                self.write_json(self.root / "preregistration.json", record)
                with self.assertRaisesRegex(ValueError, "guard inputs"):
                    paired.validate_preregistration(self.root)
        self.preregister()
        (self.root / "guard_contract.json").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            paired.validate_preregistration(self.root)

    def test_v2_requires_complete_balanced_evaluation_mapping(self):
        self.use_v2()
        valid = copy.deepcopy(self.protocol)
        mappings = [{"R1": valid["evaluation_arm_mapping"]["R1"]},
                    {"R1": {"A": "current", "B": "improved"},
                     "R2": {"A": "current", "B": "improved"}}]
        for mapping in mappings:
            with self.subTest(mapping=mapping):
                self.write_json(self.root / "protocol.json", {**valid, "evaluation_arm_mapping": mapping})
                self.preregister()
                with self.assertRaises(ValueError):
                    paired.validate_preregistration(self.root)
        valid["arm_mapping"]["N2"] = {"A": "current", "B": "improved"}
        self.write_json(self.root / "protocol.json", valid)
        self.preregister()
        with self.assertRaisesRegex(ValueError, "balanced.*new"):
            paired.validate_preregistration(self.root)

    def test_joint_packet_contains_both_coded_arms_and_no_operator_mapping(self):
        self.use_v2()
        folder = self.prepare_joint("R2")
        packet_path = folder / "evaluation/paired_post_packet.json"
        text = packet_path.read_text(encoding="utf-8")
        packet = rep._read_json(packet_path)
        self.assertEqual(set(packet["arms"]), {"A", "B"})
        self.assertEqual(packet["original"], self.original)
        self.assertEqual(packet["baseline"], self.baseline)
        for forbidden in ("current", "improved", "repair_v1", "repair_v2", "guard", "ledger",
                          "evaluation_arm_mapping", "PRIVATE PATCH JUSTIFICATION", "arms/"):
            self.assertNotIn(forbidden, text)
        for code, arm in (("A", "improved"), ("B", "current")):
            original_packet = rep._read_json(folder / "arms" / arm / "post_packet.json")
            for key in ("revised", "literal_changes", "writer_findings"):
                self.assertEqual(packet["arms"][code][key], original_packet[key])
            self.assertEqual(packet["arms"][code]["input_sha256"]["post_packet"],
                             rep._sha(folder / "arms" / arm / "post_packet.json"))
        before = packet_path.read_bytes()
        paired.prepare_post(self.root, "R2")
        self.assertEqual(packet_path.read_bytes(), before)

    def test_joint_prepare_requires_both_existing_prepared_packets(self):
        self.use_v2()
        folder = self.write_common()
        self.write_arm("current", with_post=False)
        self.write_arm("improved", prepare=False, with_post=False)
        with self.assertRaisesRegex(ValueError, "Prepared artifact missing"):
            paired.prepare_post(self.root, "R1")
        self.assertFalse((folder / "evaluation/paired_post_packet.json").exists())
        self.assertFalse((folder / "evaluation/paired_post.freeze.json").exists())

    def test_joint_ingest_preserves_exact_judgments_and_reversed_regression_mapping(self):
        self.use_v2()
        folder = self.prepare_joint("R2")
        response = self.joint_response()
        response["arms"]["B"]["issues"][0]["after_status"] = "unresolved"
        source = self.base / "response.json"
        self.write_json(source, response)
        paired.ingest_post(self.root, "R2", source)
        for code, arm in (("A", "improved"), ("B", "current")):
            self.assertEqual(rep._read_json(folder / "arms" / arm / "post.raw.json"), response["arms"][code])
        self.assertEqual((folder / "evaluation/paired_post.raw.json").read_bytes(), source.read_bytes())
        audit = rep._read_json(folder / "evaluation/paired_post.ingest.json")
        self.assertEqual(audit["paired_post_raw_sha256"], rep._sha(source))
        self.assertEqual(audit["arm_post_sha256"]["A"], rep._sha(folder / "arms/improved/post.raw.json"))
        paired.ingest_post(self.root, "R2")
        primary = self.document(self.report(), "R2")["paired"]["primary"]
        self.assertEqual(primary["improved_only_resolved"], ["B1"])

    def test_joint_new_arms_keep_literal_codes_before_variant_comparison(self):
        self.use_v2()
        folder = self.prepare_joint("N2")
        response = self.joint_response()
        response["arms"]["B"]["issues"][0]["after_status"] = "unresolved"
        source = self.base / "response.json"
        self.write_json(source, response)
        paired.ingest_post(self.root, "N2", source)
        for code in ("A", "B"):
            self.assertEqual(rep._read_json(folder / "arms" / code / "post.raw.json"), response["arms"][code])
        self.assertEqual(self.document(self.report(), "N2")["paired"]["primary"]["improved_only_resolved"], ["B1"])

    def test_joint_ingest_checks_all_packets_baseline_and_provenance_for_tampering(self):
        self.use_v2()
        folder = self.prepare_joint()
        source = self.base / "response.json"
        self.joint_response(source)
        paths = [folder / name for name in ("evaluation/paired_post_packet.json", "evaluation/paired_post.freeze.json",
                 "evaluation/baseline.raw.json", "evaluation/baseline.freeze.json", "arms/current/post_packet.json",
                 "arms/improved/freeze.json", "arms/improved/repair.raw.json")]
        for path in paths:
            with self.subTest(path=path):
                before = path.read_bytes()
                value = rep._read_json(path)
                value["tampered"] = True
                self.write_json(path, value)
                try:
                    with self.assertRaises(ValueError):
                        paired.ingest_post(self.root, "R1", source)
                    self.assertFalse((folder / "arms/current/post.raw.json").exists())
                    self.assertFalse((folder / "arms/improved/post.raw.json").exists())
                    self.assertFalse((folder / "evaluation/paired_post.raw.json").exists())
                finally:
                    path.write_bytes(before)

    def test_joint_invalid_second_arm_writes_neither_and_leaves_pair_null(self):
        self.use_v2()
        folder = self.prepare_joint()
        response = self.joint_response()
        response["arms"]["B"]["issues"][0]["issue_id"] = "UNKNOWN"
        source = self.base / "response.json"
        self.write_json(source, response)
        with self.assertRaisesRegex(ValueError, "post.issues"):
            paired.ingest_post(self.root, "R1", source)
        for name in ("arms/current/post.raw.json", "arms/improved/post.raw.json", "evaluation/paired_post.raw.json",
                     "evaluation/paired_post.ingest.json"):
            self.assertFalse((folder / name).exists())
        record = self.document(self.report())
        self.assertIsNone(record["paired"])
        self.assertTrue(all(arm["metrics"] is None for arm in record["arms"].values()))

    def test_joint_conflicting_second_output_prevents_first_output(self):
        self.use_v2()
        folder = self.prepare_joint()
        source = self.base / "response.json"
        self.joint_response(source)
        existing = folder / "arms/improved/post.raw.json"
        existing.write_text("prior immutable record", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            paired.ingest_post(self.root, "R1", source)
        self.assertEqual(existing.read_text(encoding="utf-8"), "prior immutable record")
        self.assertFalse((folder / "arms/current/post.raw.json").exists())
        self.assertFalse((folder / "evaluation/paired_post.raw.json").exists())

    def test_joint_audit_and_extracted_judgment_tampering_invalidates_report(self):
        self.use_v2()
        folder = self.prepare_joint()
        source = self.base / "response.json"
        self.joint_response(source)
        paired.ingest_post(self.root, "R1", source)
        for index, name in enumerate(("evaluation/paired_post.raw.json", "evaluation/paired_post.ingest.json",
                                      "arms/current/post.raw.json")):
            path = folder / name
            before = path.read_bytes()
            with self.subTest(name=name):
                value = rep._read_json(path)
                if name.endswith("paired_post.raw.json"):
                    value["cross_arm_notes"].append("changed after ingest")
                elif name.endswith("paired_post.ingest.json"):
                    value["arm_post_sha256"]["A"] = "0" * 64
                else:
                    value["issues"][0]["after_status"] = "unresolved"
                self.write_json(path, value)
                try:
                    record = self.document(self.report(f"changed-ingest-{index}"))
                    self.assertEqual(record["status"], "invalid")
                    self.assertIsNone(record["paired"])
                    self.assertTrue(all(arm["metrics"] is None for arm in record["arms"].values()))
                finally:
                    path.write_bytes(before)

    def test_review_uses_protocol_metadata_and_optional_guard_section(self):
        self.use_v2()
        self.original["chat_url"] = "https://example.test/research"
        self.protocol.update(report_title="Guard comparison", report_heading="Frozen claim guard", report_date="2026-11-02")
        for doc in self.protocol["documents"]:
            doc["theme"] = "Fixture theme"
            self.write_common(doc["id"])
        self.write_json(self.root / "protocol.json", self.protocol)
        self.preregister()
        self.report()
        review = {key: [] for key in ("verdict", "changes", "design", "new_notes", "regression_notes", "gates", "cases",
                                     "safety", "recommendations", "limitations")}
        review.update(headline="Fixture", guard_notes=["Guard notes <escaped>"], guard_stats={"pass": 2})
        self.write_json(self.root / "review.json", review)
        paired_review.render_review(self.root, self.base / "report")
        text = (self.base / "report/report.html").read_text(encoding="utf-8")
        self.assertIn("<title>Guard comparison</title>", text)
        self.assertIn("Frozen claim guard", text)
        self.assertIn("2026-11-02 · 旧2本＋新規2本", text)
        self.assertIn("既知の2本", text)
        self.assertNotIn("既知の3本", text)
        self.assertIn("Guard notes &lt;escaped&gt;", text)

    def test_missing_documents_and_arms_remain_null_and_planned(self):
        summary = self.report()
        self.assertEqual([row["id"] for row in summary["documents"]], ["R1", "R2", "R3", "N1", "N2"])
        for cohort, planned in (("regression", 3), ("new", 2)):
            self.assertEqual(summary["cohorts"][cohort]["planned_documents"], planned)
            self.assertEqual(summary["cohorts"][cohort]["complete_pairs"], 0)
            self.assertIsNone(summary["cohorts"][cohort]["paired_primary"])
        for document in summary["documents"]:
            self.assertEqual(len(document["arms"]), 2)
            self.assertIsNone(document["paired"])
            for arm in document["arms"].values():
                self.assertIsNone(arm["metrics"])

    def test_prepare_requires_preexisting_independent_baseline_freeze(self):
        folder = self.write_common(freeze=False)
        self.write_arm("current", prepare=False, with_post=False)
        with self.assertRaisesRegex(ValueError, "baseline.*freeze|freeze.*baseline"):
            paired.prepare(self.root, "R1", "current")
        self.assertFalse((folder / "arms/current/post_packet.json").exists())
        self.assertFalse((folder / "arms/current/freeze.json").exists())

    def test_pairing_uses_issue_ids_and_exact_resolved_status(self):
        self.original["text"] = "Alpha. Beta. Delta. Epsilon. Zeta. Proposal."
        for identity, quote in (("B3", "Delta."), ("B4", "Epsilon."), ("B5", "Zeta.")):
            issue = copy.deepcopy(self.baseline["issues"][0])
            issue.update(issue_id=identity, quote=quote)
            self.baseline["issues"].append(issue)
            row = copy.deepcopy(self.post["issues"][0])
            row.update(issue_id=identity, after_quote=quote, detected=False, finding_ids=[])
            self.post["issues"].append(row)
        current = copy.deepcopy(self.post)
        improved = copy.deepcopy(self.post)
        current_status = {"B1": "resolved", "B2": "resolved", "B3": "unresolved", "B4": "resolved", "B5": "partially_resolved"}
        improved_status = {"B1": "unresolved", "B2": "resolved", "B3": "resolved", "B4": "resolved", "B5": "unresolved"}
        for row in current["issues"]:
            row["after_status"] = current_status[row["issue_id"]]
        for row in improved["issues"]:
            row["after_status"] = improved_status[row["issue_id"]]
        improved["issues"].reverse()
        self.write_common()
        self.write_arm("current", post=current)
        self.write_arm("improved", post=improved)
        summary = self.report()
        primary = self.document(summary)["paired"]["primary"]
        self.assertEqual(primary["denominator"], 4)
        self.assertEqual(primary["current_only_resolved"], ["B1"])
        self.assertEqual(primary["improved_only_resolved"], ["B3"])
        self.assertEqual(primary["both_resolved"], ["B4"])
        self.assertEqual(primary["neither_resolved"], ["B5"])
        self.assertEqual(summary["cohorts"]["regression"]["complete_pairs"], 1)
        self.assertEqual(summary["cohorts"]["new"]["complete_pairs"], 0)

    def test_new_document_arm_mapping_is_used_in_both_directions(self):
        self.write_common("N2")
        current = copy.deepcopy(self.post)
        current["issues"][0]["after_status"] = "unresolved"
        self.write_arm("A", "N2")
        self.write_arm("B", "N2", post=current)
        summary = self.report()
        primary = self.document(summary, "N2")["paired"]["primary"]
        self.assertEqual(primary["improved_only_resolved"], ["B1"])
        self.assertEqual(primary["current_only_resolved"], [])
        self.assertEqual(summary["cohorts"]["new"]["complete_pairs"], 1)
        self.assertEqual(summary["cohorts"]["regression"]["complete_pairs"], 0)

    def test_missing_one_post_does_not_create_a_paired_zero(self):
        self.write_common()
        self.write_arm("current")
        self.write_arm("improved", with_post=False)
        summary = self.report()
        document = self.document(summary)
        self.assertIsNotNone(document["arms"]["current"]["metrics"])
        self.assertIsNone(document["arms"]["improved"]["metrics"])
        self.assertIsNone(document["paired"])
        self.assertIsNone(summary["cohorts"]["regression"]["paired_primary"])

    def test_asymmetric_exclusion_removes_both_arms_from_common_denominator(self):
        self.write_common()
        current = copy.deepcopy(self.post)
        current["issues"][0]["after_status"] = "not_an_original_problem"
        self.write_arm("current", post=current)
        self.write_arm("improved")
        primary = self.document(self.report())["paired"]["primary"]
        self.assertEqual(primary["denominator"], 0)
        self.assertEqual(primary["excluded_issue_ids"], ["B1"])
        self.assertEqual([row["issue_id"] for row in primary["exclusion_disagreements"]], ["B1"])
        for key in ("current_only_resolved", "improved_only_resolved", "both_resolved", "neither_resolved"):
            self.assertEqual(primary[key], [])

    def test_new_problems_posthoc_unresolved_and_false_findings_stay_separate(self):
        self.post["patch_reviews"][0].update(status="unnecessary_change", new_problems=[{
            "problem_id": "NP1", "status": "confirmed_problem", "tags": ["information_loss"],
            "quote": "Gamma.", "reason": "Fixture only.", "source_urls": [],
        }])
        self.post["posthoc_original_findings"] = [{
            "quote": "Beta.", "status": "confirmed_problem", "tags": ["condition"],
            "reason": "Fixture only.", "still_present_after": True, "after_quote": "Beta.", "source_urls": [],
        }]
        self.post["false_findings"] = [{"finding_id": "F1", "status": "false_alarm", "reason": "Fixture only."}]
        self.write_common()
        self.write_arm("current")
        self.write_arm("improved")
        document = self.document(self.report())
        self.assertEqual(document["paired"]["primary"]["denominator"], 1)
        for arm in document["arms"].values():
            metrics = arm["metrics"]
            self.assertEqual(metrics["baseline_unresolved"], 1)
            self.assertEqual(metrics["new_problems"]["confirmed_problem"], 1)
            self.assertEqual(metrics["posthoc_original_findings"]["confirmed_problem"], 1)
            self.assertEqual(metrics["patch_reviews"]["unnecessary_change"], 1)
            self.assertEqual(metrics["false_findings"]["false_alarm"], 1)

    def test_input_and_freeze_tampering_invalidates_metrics_without_overwrite(self):
        folder = self.write_common()
        arm = self.write_arm("current")
        self.write_arm("improved")
        paths = [folder / "input/original_report.json", folder / "evaluation/baseline.raw.json",
                 folder / "evaluation/baseline.freeze.json", arm / "repair.raw.json", arm / "freeze.json"]
        output_before = (arm / "post_packet.json").read_bytes()
        for index, path in enumerate(paths):
            with self.subTest(path=path.name):
                before = path.read_bytes()
                data = rep._read_json(path)
                data["tampered_fixture_metadata"] = True
                self.write_json(path, data)
                try:
                    with self.assertRaises(ValueError):
                        paired.prepare(self.root, "R1", "current")
                    document = self.document(self.report(f"tampered-{index}"))
                    self.assertEqual(document["arms"]["current"]["status"], "invalid")
                    self.assertIsNone(document["arms"]["current"]["metrics"])
                    self.assertIsNone(document["paired"])
                    self.assertEqual((arm / "post_packet.json").read_bytes(), output_before)
                finally:
                    path.write_bytes(before)

    def test_extra_preregistered_file_is_checked_and_instruction_tampering_rejected(self):
        extra = self.root / "instructions/additional-fixed-note.txt"
        extra.write_text("Fixed extra note", encoding="utf-8")
        self.preregister(extra=("instructions/additional-fixed-note.txt",))
        self.write_common()
        self.write_arm("current")
        extra.write_text("Altered note", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256|hash"):
            paired.prepare(self.root, "R1", "current")
        with self.assertRaises(ValueError):
            self.report()

    def test_prepare_is_blinded_immutable_and_idempotent(self):
        folder = self.write_common()
        arm = self.write_arm("current", prepare=False)
        raw_paths = [folder / "input/original_report.json", folder / "evaluation/baseline.raw.json",
                     folder / "evaluation/baseline.freeze.json", arm / "repair.raw.json", arm / "post.raw.json"]
        raw_before = {path: path.read_bytes() for path in raw_paths}
        paired.prepare(self.root, "R1", "current")
        derived = {name: (arm / name).read_bytes() for name in
                   ("freeze.json", "applied.json", "revised_report.json", "post_packet.json")}
        paired.prepare(self.root, "R1", "current")
        self.assertEqual(raw_before, {path: path.read_bytes() for path in raw_paths})
        self.assertEqual(derived, {name: (arm / name).read_bytes() for name in derived})
        packet_text = (arm / "post_packet.json").read_text(encoding="utf-8")
        packet = json.loads(packet_text)
        self.assertNotIn("PRIVATE PATCH JUSTIFICATION", packet_text)
        for forbidden in ("current", "improved", "repair_v1", "repair_v2", '"arm"', '"variant"'):
            self.assertNotIn(forbidden, packet_text)
        self.assertEqual(packet["writer_findings"], self.repair["findings"])
        self.assertEqual(packet["original"], self.original)
        self.assertEqual(packet["revised"]["text"], "Gamma. Beta. Proposal.")
        self.assertNotIn("reason", packet["literal_changes"][0])

    def test_wrong_post_issue_id_invalidates_arm_and_pair(self):
        self.write_common()
        self.write_arm("current")
        bad = copy.deepcopy(self.post)
        bad["issues"][0]["issue_id"] = "DIFFERENT_ISSUE"
        self.write_arm("improved", post=bad)
        document = self.document(self.report())
        self.assertEqual(document["arms"]["improved"]["status"], "invalid")
        self.assertIsNone(document["arms"]["improved"]["metrics"])
        self.assertIsNone(document["paired"])

    def test_conflicting_derived_artifact_is_not_overwritten(self):
        folder = self.write_common()
        arm = self.write_arm("current", prepare=False, with_post=False)
        target = arm / "revised_report.json"
        target.write_text('{"text":"Prior different output"}', encoding="utf-8")
        before = target.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite|different"):
            paired.prepare(self.root, "R1", "current")
        self.assertEqual(target.read_bytes(), before)
        self.assertFalse((folder / "arms/current/post_packet.json").exists())


if __name__ == "__main__":
    unittest.main()
