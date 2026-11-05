"""Synthetic v5 omission tests; no real candidate records, outcomes or labels."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import claim_guard_v5 as cli
import replication as rep
from factcheck import claim_guard as v3
from factcheck import claim_guard_v5 as v5


# Re-run the complete existing synthetic safety suite through the opt-in engine
# without editing the original tests or any v3 implementation file.
_spec = importlib.util.spec_from_file_location("legacy_synthetic_guard_tests", Path(__file__).with_name("test_claim_guard.py"))
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)


class V5InheritedSafetyTests(legacy.ClaimGuardTests):
    def setUp(self):
        binding = mock.patch.multiple(legacy, cli=cli, build_block_packet=v5.build_block_packet,
                                      build_guard_packet=v5.build_guard_packet, gate_repair=v5.gate_repair,
                                      preview_repair=v5.preview_repair, schema_contract=v5.schema_contract,
                                      validate_ledger=v5.validate_ledger)
        binding.start()
        self.addCleanup(binding.stop)
        super().setUp()


def omission_fixture(kind="factual", check="population"):
    original, candidate, ledger = legacy.fixture()
    candidate["patches"] = candidate["patches"][:1]
    candidate["added_sources"] = candidate["added_sources"][:1]
    claim = ledger["claims"][0]
    claim["checks"]["quantity"]["status"] = "supported"
    claim["checks"][check] = {"status": "missing_material", "rationale": "Synthetic material omission",
                               "evidence_ids": ["E1"]}
    added = "In adults only." if kind == "factual" else "[2]"
    patch = candidate["patches"][0]
    patch["after"] = patch["before"] + " " + added
    guard = legacy.review(original, candidate, ledger)
    row = guard["patch_reviews"][0]
    row["units"][0].update({"status": "supported", "disposition": "preserved",
                            "after_quote": row["units"][0]["before_quote"]})
    row["additions"] = [{"after_quote": added, "content_kind": kind, "claim_ids": ["C1"],
                         "status": "supported", "evidence_ids": ["E1"],
                         "reason": "Synthetic independent necessary omission assessment",
                         "report_scope": "Adult respondents", "source_scope": "Adult respondents"}]
    row["rationale"] = "The existing material omission needs the stated qualification"
    return original, candidate, ledger, guard


def rebind(original, candidate, ledger, guard):
    packet = v5.build_guard_packet(original, candidate, ledger)
    guard.update({"guard_packet_sha256": v5.sha256_json(packet), "input_sha256": packet["input_sha256"]})
    return packet


def finalize(original, candidate, ledger, guard, status="pass", complete=True):
    preview = v5.preview_repair(original, candidate, ledger, guard)
    guard["final_review"] = {**preview["selection"], "status": status, "coverage_complete": complete,
                             "reason": "Synthetic independent final selected-report review"}
    return v5.gate_repair(original, candidate, ledger, guard)


class MaterialOmissionTests(unittest.TestCase):
    def setUp(self):
        self.original, self.candidate, self.ledger, self.guard = omission_fixture()

    def preview(self):
        rebind(self.original, self.candidate, self.ledger, self.guard)
        return v5.preview_repair(self.original, self.candidate, self.ledger, self.guard)

    def decision(self):
        return self.preview()["decisions"][0]

    def test_same_verbatim_inputs_diverge_only_on_eligibility_and_do_not_mutate(self):
        before = copy.deepcopy((self.original, self.candidate, self.ledger, self.guard))
        old = v3.preview_repair(*before[:3], before[3])
        new = v5.preview_repair(self.original, self.candidate, self.ledger, self.guard)
        self.assertEqual((self.original, self.candidate, self.ledger, self.guard), before)
        for key in ("input_sha256", "guard_sha256", "guard_packet_sha256", "candidate_applied"):
            self.assertEqual(old[key], new[key])
        self.assertFalse(old["decisions"][0]["accepted"])
        self.assertTrue(new["decisions"][0]["accepted"])
        self.assertEqual(new["decisions"][0]["eligibility_route"], "material_omission")
        self.assertEqual(new["decisions"][0]["material_omission"]["links"][0]["check"], "population")
        self.assertEqual(new["selected_applied"]["text"], new["candidate_applied"]["text"])
        self.assertEqual(new["gate_version"], "v5")
        self.assertEqual(new["summary"]["gate_version"], "v5")
        self.assertEqual(new["selection"]["gate_version"], "v5")

    def test_all_condition_check_categories_and_citation_qualify(self):
        for check in v5.CHECKS:
            kind = "citation" if check == "citation" else "factual"
            original, candidate, ledger, guard = omission_fixture(kind, check)
            with self.subTest(check=check):
                self.assertTrue(v5.preview_repair(original, candidate, ledger, guard)["decisions"][0]["accepted"])

    def test_optional_unnecessary_uncertain_additions_remain_held(self):
        for necessity in ("optional", "unnecessary", "uncertain"):
            self.guard["patch_reviews"][0]["necessity"] = necessity
            with self.subTest(necessity=necessity):
                self.assertFalse(self.decision()["accepted"])

    def test_unknown_contradicted_and_inaccessible_addition_evidence_held(self):
        for status in ("uncertain", "contradicted"):
            self.guard["patch_reviews"][0]["additions"][0]["status"] = status
            with self.subTest(status=status):
                self.assertFalse(self.decision()["accepted"])
        self.guard["patch_reviews"][0]["additions"][0]["status"] = "supported"
        for access in ("partial", "unavailable"):
            self.guard["evidence"] = [{**self.ledger["evidence"][0], "evidence_id": "G1", "access": access}]
            self.guard["patch_reviews"][0]["additions"][0]["evidence_ids"] = ["G1"]
            with self.subTest(access=access):
                self.assertFalse(self.decision()["accepted"])
        self.guard["patch_reviews"][0]["additions"][0]["evidence_ids"] = ["absent"]
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.preview()

    def test_missing_material_is_required_not_contradiction_or_unknown(self):
        for status in ("unknown", "contradicted", "supported"):
            self.ledger["claims"][0]["checks"]["population"]["status"] = status
            self.ledger["claims"][0]["status"] = "problem" if status == "contradicted" else "uncertain"
            with self.subTest(status=status):
                self.assertFalse(self.decision()["accepted"])

    def test_additions_require_correct_touched_claim_mapping(self):
        for refs in ([], ["C2"], ["C3"], ["C2", "C3"]):
            self.guard["patch_reviews"][0]["additions"][0]["claim_ids"] = refs
            with self.subTest(refs=refs):
                self.assertFalse(self.decision()["accepted"])

    def test_supported_and_untouched_context_do_not_veto_actual_missing_link(self):
        for refs in (["C1", "C2"], ["C1", "C3"], ["C1", "C2", "C3"]):
            self.guard["patch_reviews"][0]["additions"][0]["claim_ids"] = refs
            with self.subTest(refs=refs):
                decision = self.decision()
                self.assertTrue(decision["accepted"])
                self.assertEqual({link["claim_id"] for link in decision["material_omission"]["links"]}, {"C1"})

    def test_factual_and_citation_checks_cannot_cross_authorize(self):
        self.guard["patch_reviews"][0]["additions"][0]["content_kind"] = "citation"
        self.assertFalse(self.decision()["accepted"])
        self.original, self.candidate, self.ledger, self.guard = omission_fixture("factual", "citation")
        self.assertFalse(self.decision()["accepted"])

    def test_all_inserted_nonwhitespace_text_requires_review(self):
        self.candidate["patches"][0]["after"] += " Unreviewed detail."
        decision = self.decision()
        self.assertFalse(decision["accepted"])
        self.assertTrue(any("every inserted" in reason for reason in decision["reasons"]))

    def test_all_material_additions_require_missing_check_links(self):
        self.candidate["patches"][0]["after"] += " [2]"
        self.guard["patch_reviews"][0]["additions"].append({**self.guard["patch_reviews"][0]["additions"][0],
                                                           "after_quote": "[2]", "content_kind": "citation"})
        self.assertFalse(self.decision()["accepted"])
        self.ledger["claims"][0]["checks"]["citation"] = {"status": "missing_material", "rationale": "Missing citation", "evidence_ids": ["E1"]}
        self.assertTrue(self.decision()["accepted"])

    def test_preexisting_quote_does_not_count_as_an_added_unit(self):
        self.guard["patch_reviews"][0]["additions"][0]["after_quote"] = "The count was 5."
        self.assertFalse(self.decision()["accepted"])

    def test_nonfactual_only_addition_cannot_qualify(self):
        self.guard["patch_reviews"][0]["additions"][0]["content_kind"] = "nonfactual"
        self.assertFalse(self.decision()["accepted"])

    def test_loss_or_rewrite_of_original_content_is_held(self):
        # Both reviewed sentence quotes survive; deletion between them still
        # fails the independent insertion-only text check.
        self.candidate["patches"][0]["after"] = self.candidate["patches"][0]["after"].replace("5. The", "5.The")
        self.assertFalse(self.decision()["accepted"])
        self.candidate["patches"][0]["after"] = self.candidate["patches"][0]["before"] + " In adults only."
        self.guard["patch_reviews"][0]["units"][1].update({"disposition": "lost", "after_quote": ""})
        self.assertFalse(self.decision()["accepted"])

    def test_complete_coverage_is_still_required(self):
        self.guard["patch_reviews"][0]["coverage_complete"] = False
        self.assertFalse(self.decision()["accepted"])

    def test_independent_evidence_ids_with_same_document_can_link(self):
        source = self.ledger["evidence"][0]
        self.guard["evidence"] = [{**source, "evidence_id": "G1", "url": "https://EXAMPLE.org:443/data",
                                   "locator": "  Fixture\n table  ", "summary": "Independent summary"}]
        self.guard["patch_reviews"][0]["additions"][0]["evidence_ids"] = ["G1"]
        self.assertTrue(self.decision()["accepted"])
        pairs = self.decision()["material_omission"]["links"][0]["evidence_pairs"]
        self.assertEqual(pairs, [{"addition_evidence_id": "G1", "ledger_evidence_id": "E1"}])

    def test_different_freeform_locators_for_same_document_can_link(self):
        self.ledger["evidence"][0]["locator"] = "p. 3, pp. 5–6"
        self.guard["evidence"] = [{**self.ledger["evidence"][0], "evidence_id": "G1",
                                   "locator": "Independently extracted p. 3 and pp. 5–6",
                                   "summary": "Independent synthetic assessment"}]
        self.guard["patch_reviews"][0]["additions"][0]["evidence_ids"] = ["G1"]
        self.assertTrue(self.decision()["accepted"])

    def test_document_url_matching_is_exact_not_fuzzy(self):
        for change in ({"url": "https://example.org/other"}, {"url": "https://example.org/data?x=1"},
                       {"url": "https://example.org/data#other"}, {"url": "https://example.org:444/data"},
                       {"url": "https://example.org:0/data"}, {"url": "https://other.example.org/data"},
                       {"url": "https://example.org/DATA"}):
            self.guard["evidence"] = [{**self.ledger["evidence"][0], "evidence_id": "G1", **change}]
            self.guard["patch_reviews"][0]["additions"][0]["evidence_ids"] = ["G1"]
            with self.subTest(change=change):
                self.assertFalse(self.decision()["accepted"])

    def test_both_linked_factual_records_need_primary_full_section(self):
        self.ledger["evidence"].append({**self.ledger["evidence"][0], "evidence_id": "E2"})
        self.ledger["claims"][0]["checks"]["population"]["evidence_ids"] = ["E2"]
        for change in ({"primary": False}, {"access": "partial"}, {"access": "unavailable"}):
            self.ledger["evidence"][1] = {**self.ledger["evidence"][0], "evidence_id": "E2", **change}
            with self.subTest(change=change):
                self.assertFalse(self.decision()["accepted"])

    def test_secondary_full_section_can_support_missing_citation(self):
        self.original, self.candidate, self.ledger, self.guard = omission_fixture("citation", "citation")
        self.ledger["evidence"].append({**self.ledger["evidence"][0], "evidence_id": "E2", "primary": False})
        self.ledger["claims"][0]["checks"]["citation"]["evidence_ids"] = ["E2"]
        self.guard["patch_reviews"][0]["additions"][0]["evidence_ids"] = ["E2"]
        self.assertTrue(self.decision()["accepted"])

    def test_rejected_dependencies_still_hold_eligible_omission(self):
        other = legacy.fixture()[1]
        self.candidate["patches"].append(other["patches"][1])
        self.candidate["added_sources"].append(other["added_sources"][1])
        self.candidate["patches"][0]["depends_on_patch_ids"] = ["P2"]
        second = legacy.review(self.original, self.candidate, self.ledger)["patch_reviews"][1]
        second["necessity"] = "optional"
        self.guard["patch_reviews"].append(second)
        decision = self.decision()
        self.assertTrue(decision["material_omission"]["eligible"])
        self.assertFalse(decision["accepted"])
        self.assertTrue(any("dependencies" in reason for reason in decision["reasons"]))

    def test_original_scopes_and_addition_scopes_remain_mandatory(self):
        for field in ("report_scope", "source_scope"):
            guard = copy.deepcopy(self.guard)
            guard["patch_reviews"][0]["additions"][0][field] = ""
            with self.subTest(field=field), self.assertRaises(ValueError):
                v5.preview_repair(self.original, self.candidate, self.ledger, guard)

    def test_old_final_review_cannot_approve_new_selection(self):
        old = v3.preview_repair(self.original, self.candidate, self.ledger, self.guard)
        self.guard["final_review"] = {**old["selection"], "status": "pass", "coverage_complete": True,
                                      "reason": "Old final report reviewed"}
        with self.assertRaisesRegex(ValueError, "gate_version"):
            v5.gate_repair(self.original, self.candidate, self.ledger, self.guard)
        self.guard["final_review"]["gate_version"] = "v5"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            v5.gate_repair(self.original, self.candidate, self.ledger, self.guard)

    def test_new_final_review_binds_version_ids_text_sources_and_can_veto(self):
        result = finalize(self.original, self.candidate, self.ledger, self.guard)
        self.assertTrue(result["decisions"][0]["accepted"])
        for field, value in (("gate_version", "v3"), ("accepted_patch_ids", []),
                             ("final_text_sha256_utf8", "0" * 64), ("final_sources_sha256", "0" * 64)):
            changed = copy.deepcopy(self.guard)
            changed["final_review"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "mismatch"):
                v5.gate_repair(self.original, self.candidate, self.ledger, changed)
        for status, complete in (("fail", True), ("pass", False)):
            result = finalize(self.original, self.candidate, self.ledger, self.guard, status, complete)
            with self.subTest(status=status, complete=complete):
                self.assertEqual(result["final_applied"]["text"], self.original["text"])
                self.assertEqual(result["final_applied"]["sources"], self.original["sources"])
                self.assertEqual(result["summary"]["accepted_patch_count"], 0)

    def test_v5_contract_documents_opt_in_and_final_metadata(self):
        contract = v5.schema_contract()
        self.assertEqual(contract["gate_version"], "v5")
        self.assertIn("gate_version", contract["guard"]["final_review"])
        self.assertTrue(any("material_omission" in rule for rule in contract["gate_rules"]))

    def test_v5_cli_accepts_existing_packet_and_freezes_separate_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = {"original": self.original, "candidate": self.candidate, "ledger": self.ledger}
            paths = {key: root / (key + ".json") for key in records}
            for key, path in paths.items():
                path.write_bytes(rep._json_bytes(records[key]))
            flags = [item for key, path in paths.items() for item in ("--" + key, str(path))]
            prepared, previewed, applied = root / "old-prepare", root / "v5-preview", root / "v5-apply"
            with contextlib.redirect_stdout(io.StringIO()):
                legacy.cli.main(["prepare", *flags, "--output-dir", str(prepared)])
                packet = rep._read_json(prepared / "guard_packet.json")
                self.guard.update({"guard_packet_sha256": v5.sha256_json(packet), "input_sha256": packet["input_sha256"]})
                guard_path = root / "guard.json"
                guard_path.write_bytes(rep._json_bytes(self.guard))
                guard_flags = ["--guard", str(guard_path), "--guard-packet", str(prepared / "guard_packet.json")]
                cli.main(["preview", *flags, *guard_flags, "--output-dir", str(previewed)])
                preview = rep._read_json(previewed / "gate.preview.json")
                self.assertEqual(preview["selection"]["accepted_patch_ids"], ["P1"])
                self.assertFalse((previewed / "repair.effective.json").exists())
                self.guard["final_review"] = {**preview["selection"], "status": "pass", "coverage_complete": True,
                                              "reason": "Synthetic selected report reviewed"}
                guard_path.write_bytes(rep._json_bytes(self.guard))
                cli.main(["apply", *flags, *guard_flags, "--output-dir", str(applied)])
            self.assertEqual(rep._read_json(applied / "summary.json")["gate_version"], "v5")
            self.assertEqual((prepared / "guard_packet.json").read_bytes(), (applied / "guard_packet.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
