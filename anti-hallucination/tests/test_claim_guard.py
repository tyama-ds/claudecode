"""Structural gate tests with explicitly synthetic semantic judgments; no LLM calls."""
import contextlib
import copy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

import claim_guard as cli
import replication as rep
from factcheck.claim_guard import (CHECKS, build_block_packet, build_guard_packet, gate_repair,
                                  preview_repair, schema_contract, sha256_json, validate_ledger)


def fixture():
    original = {"text": "The count was 5. The location was East.\nThe size was 8.\n# Commentary\n",
                "sources": [{"citation_id": 1, "title": "Original", "url": "https://example.org/original"}]}
    evidence = {"evidence_id": "E1", "url": "https://example.org/data", "locator": "Fixture table",
                "summary": "Synthetic primary data for structural tests", "access": "full_section", "primary": True}
    packet = build_block_packet(original)
    claims = []
    for identity, block, quote, status in (("C1", "B0001", "The count was 5.", "problem"),
                                           ("C2", "B0001", "The location was East.", "supported"),
                                           ("C3", "B0002", "The size was 8.", "problem")):
        checks = {key: {"status": "not_applicable", "rationale": "Not applicable to this fixture", "evidence_ids": []} for key in CHECKS}
        checks["quantity" if status == "problem" else "population"] = {
            "status": "contradicted" if status == "problem" else "supported", "rationale": "Fixture finding", "evidence_ids": ["E1"]}
        claims.append({"claim_id": identity, "block_id": block, "quote": quote, "status": status,
                       "reason": "Fixture", "context_block_ids": [block], "report_scope": "Fixture scope",
                       "source_scope": "Fixture evidence scope", "checks": checks})
    ledger = {"schema_version": 1, "original_sha256": sha256_json(original), "block_packet_sha256": sha256_json(packet),
              "evidence": [evidence], "claims": claims,
              "blocks": [{"block_id": "B0001", "claim_ids": ["C1", "C2"], "nonfactual_reason": ""},
                         {"block_id": "B0002", "claim_ids": ["C3"], "nonfactual_reason": ""},
                         {"block_id": "B0003", "claim_ids": [], "nonfactual_reason": "Heading only"}]}
    candidate = {"execution": {"started_utc": "2026-11-01T00:00:00Z", "finished_utc": "2026-11-01T00:01:00Z", "web_calls": 0, "input_files": []},
                 "findings": [{"finding_id": "F1", "quote": "The count was 5.", "status": "confirmed_problem", "tags": ["factual"], "diagnosis": "Fixture", "evidence": []},
                              {"finding_id": "F2", "quote": "The size was 8.", "status": "confirmed_problem", "tags": ["factual"], "diagnosis": "Fixture", "evidence": []}],
                 "checked_controls": [], "unresolved": [], "limitations": [],
                 "added_sources": [{"citation_id": 2, "title": "Data", "url": evidence["url"]},
                                   {"citation_id": "3", "title": "More data", "url": "https://example.org/more"}],
                 "patches": [{"patch_id": "P1", "finding_ids": ["F1"], "claim_ids": ["C1", "C2"],
                              "before": "The count was 5. The location was East.", "after": "The count was 6. The location was East. [2]",
                              "reason": "Fixture", "source_urls": [evidence["url"]], "action": "replace"},
                             {"patch_id": "P2", "finding_ids": ["F2"], "claim_ids": ["C3"],
                              "before": "The size was 8.", "after": "The size was 9. [3]",
                              "reason": "Fixture", "source_urls": ["https://example.org/more"], "action": "replace"}]}
    return original, candidate, ledger


def review(original, candidate, ledger, *, packet=None):
    packet = packet or build_guard_packet(original, candidate, ledger)
    def unit(identity, claims, before, after, status="contradicted", disposition="corrected_with_evidence"):
        return {"unit_id": identity, "claim_ids": claims, "before_quote": before, "after_quote": after,
                "status": status, "content_kind": "factual", "disposition": disposition,
                "reason": "Independent synthetic source assessment", "rationale": "Independent synthetic mapping", "evidence_ids": ["E1"]}
    rows = [{"patch_id": "P1", "units": [unit("U1", ["C1"], "The count was 5.", "The count was 6."),
                                            unit("U2", ["C2"], "The location was East.", "The location was East.", "supported", "preserved")]},
            {"patch_id": "P2", "units": [unit("U3", ["C3"], "The size was 8.", "The size was 9.")]}]
    for row in rows:
        row.update({"necessity": "necessary", "rationale": "Independent evidence confirms an error", "evidence_ids": ["E1"],
                    "coverage_complete": True, "coverage_reason": "Every original assertion and added citation was reviewed; no added factual assertions beyond mapped corrections",
                    "additions": [{"after_quote": "[2]" if row["patch_id"] == "P1" else "[3]", "content_kind": "citation",
                                   "status": "supported", "evidence_ids": ["E1"], "reason": "Synthetic citation supports the corrected assertion",
                                   "report_scope": "Fixture citation", "source_scope": "Fixture evidence"}]})
    rows = [row for row in rows if row["patch_id"] in {patch["patch_id"] for patch in candidate["patches"]}]
    return {"schema_version": 1, "guard_packet_sha256": sha256_json(packet), "input_sha256": packet["input_sha256"],
            "evidence": [], "patch_reviews": rows}


def finalize(original, candidate, ledger, guard, *, packet=None, status="pass", complete=True):
    selected = preview_repair(original, candidate, ledger, guard, guard_packet=packet)
    guard["final_review"] = {**selected["selection"], "status": status, "coverage_complete": complete,
                             "reason": "Independently reviewed actual combined selected report"}
    return gate_repair(original, candidate, ledger, guard, guard_packet=packet)


class ClaimGuardTests(unittest.TestCase):
    def setUp(self):
        self.original, self.candidate, self.ledger = fixture()

    def test_blocks_exact_unicode_offsets_hashes_blank_lines(self):
        original = {"text": "  見出し\r\n\r\nAlpha 😀\n \nBeta", "sources": []}
        packet = build_block_packet(original)
        self.assertEqual([row["block_id"] for row in packet["blocks"]], ["B0001", "B0002", "B0003"])
        for row in packet["blocks"]:
            self.assertEqual(original["text"][row["start_char"]:row["end_char"]], row["text"])
            self.assertEqual(hashlib.sha256(row["text"].encode()).hexdigest(), row["sha256_utf8"])
        self.assertEqual(packet, build_block_packet(original))

    def test_missing_or_duplicate_blocks_fail_closed(self):
        for mode in ("missing", "duplicate"):
            ledger = copy.deepcopy(self.ledger)
            if mode == "missing":
                ledger["blocks"].pop()
            else:
                ledger["blocks"].append(ledger["blocks"][0])
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_ledger(self.original, ledger)

    def test_all_seven_checks_and_nonempty_rationales_required(self):
        for key in CHECKS:
            ledger = copy.deepcopy(self.ledger)
            del ledger["claims"][0]["checks"][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "checks"):
                validate_ledger(self.original, ledger)
        self.ledger["claims"][0]["checks"]["quantity"]["rationale"] = ""
        with self.assertRaises(ValueError):
            validate_ledger(self.original, self.ledger)

    def test_supported_cannot_hide_unknown_contradicted_missing_or_no_checks(self):
        for status in ("unknown", "contradicted", "missing_material", "not_applicable"):
            ledger = copy.deepcopy(self.ledger)
            ledger["claims"][1]["checks"]["population"]["status"] = status
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "supported"):
                validate_ledger(self.original, ledger)

    def test_supported_facts_require_primary_full_section_and_known_refs(self):
        for change in ({"primary": False}, {"access": "partial"}, {"access": "unavailable"}):
            ledger = copy.deepcopy(self.ledger)
            ledger["evidence"][0].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "primary"):
                validate_ledger(self.original, ledger)
        self.ledger["claims"][1]["checks"]["population"]["evidence_ids"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_ledger(self.original, self.ledger)

    def test_secondary_citation_support_is_valid(self):
        self.ledger["evidence"].append({**self.ledger["evidence"][0], "evidence_id": "E2", "primary": False})
        self.ledger["claims"][1]["checks"]["citation"] = {"status": "supported", "rationale": "Actual secondary citation supports this claim", "evidence_ids": ["E2"]}
        self.assertEqual(validate_ledger(self.original, self.ledger)["supported_claim_count"], 1)

    def test_scope_and_context_mandatory_and_quote_in_own_block(self):
        for field, value in (("report_scope", ""), ("source_scope", ""), ("context_block_ids", ["B0002"]),
                             ("quote", "The size was 8.")):
            ledger = copy.deepcopy(self.ledger)
            ledger["claims"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_ledger(self.original, ledger)

    def test_nonfactual_blocks_require_explanation_and_reciprocal_claim_mapping(self):
        self.ledger["blocks"][2]["nonfactual_reason"] = ""
        with self.assertRaises(ValueError):
            validate_ledger(self.original, self.ledger)
        self.ledger = fixture()[2]
        self.ledger["blocks"][0]["claim_ids"].pop()
        with self.assertRaisesRegex(ValueError, "mapped exactly once"):
            validate_ledger(self.original, self.ledger)

    def test_patch_claim_ids_must_cover_all_touched_claims(self):
        self.candidate["patches"][0]["claim_ids"] = ["C1"]
        with self.assertRaisesRegex(ValueError, "missing IDs"):
            build_guard_packet(self.original, self.candidate, self.ledger)

    def test_valid_gate_preserves_original_sources_and_inputs(self):
        originals = copy.deepcopy((self.original, self.candidate, self.ledger))
        guard = review(self.original, self.candidate, self.ledger)
        result = finalize(self.original, self.candidate, self.ledger, guard)
        self.assertEqual(result["summary"]["accepted_patch_count"], 2)
        self.assertEqual(result["final_applied"]["sources"][0], self.original["sources"][0])
        self.assertEqual((self.original, self.candidate, self.ledger), originals)
        self.assertNotIn("claim_ids", result["effective_repair"]["patches"][0])
        self.assertIn("not proof", result["summary"]["limitation"])
        self.assertEqual(result["summary"]["supported_claim_count"], 1)
        self.assertEqual(result["summary"]["uncertain_claim_count"], 0)

    def test_missing_patch_review_or_unmatched_unit_quote_invalid(self):
        for mode in ("patch", "before", "after", "mapping"):
            guard = review(self.original, self.candidate, self.ledger)
            if mode == "patch":
                guard["patch_reviews"].pop()
            elif mode == "mapping":
                guard["patch_reviews"][0]["units"][0]["claim_ids"] = ["C2"]
            else:
                guard["patch_reviews"][0]["units"][0][mode + "_quote"] = "Absent quote"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                preview_repair(self.original, self.candidate, self.ledger, guard)

    def test_lost_valid_or_uncertain_unit_holds_whole_patch(self):
        for status in ("supported", "uncertain"):
            guard = review(self.original, self.candidate, self.ledger)
            guard["patch_reviews"][0]["units"][1].update({"status": status, "after_quote": "", "disposition": "lost"})
            result = finalize(self.original, self.candidate, self.ledger, guard)
            self.assertFalse(result["decisions"][0]["accepted"])
            self.assertIn(self.candidate["patches"][0]["before"], result["final_applied"]["text"])

    def test_relabeling_supported_unit_cannot_authorize_removal(self):
        guard = review(self.original, self.candidate, self.ledger)
        guard["patch_reviews"][0]["units"][1].update({"status": "contradicted", "after_quote": "", "disposition": "removed_unsupported"})
        result = finalize(self.original, self.candidate, self.ledger, guard)
        self.assertFalse(result["decisions"][0]["accepted"])

    def test_corrected_label_cannot_rewrite_supported_or_uncertain_unit(self):
        for status in ("supported", "uncertain"):
            with self.subTest(status=status):
                guard = review(self.original, self.candidate, self.ledger)
                guard["patch_reviews"][0]["units"][0]["status"] = status
                result = finalize(self.original, self.candidate, self.ledger, guard)
                self.assertFalse(result["decisions"][0]["accepted"])
                self.assertIn(self.candidate["patches"][0]["before"], result["final_applied"]["text"])

    def test_optional_unnecessary_uncertain_changes_are_held(self):
        for necessity in ("optional", "unnecessary", "uncertain"):
            guard = review(self.original, self.candidate, self.ledger)
            guard["patch_reviews"][0]["necessity"] = necessity
            result = finalize(self.original, self.candidate, self.ledger, guard)
            self.assertEqual(result["summary"]["rejected_patch_count"], 1)
            self.assertIn("necessity is " + necessity, result["decisions"][0]["reasons"])

    def test_incomplete_coverage_or_unmapped_atoms_holds_patch(self):
        for mode in ("attestation", "missing_atom", "no_atoms"):
            guard = review(self.original, self.candidate, self.ledger)
            row = guard["patch_reviews"][0]
            if mode == "attestation":
                row["coverage_complete"] = False
            elif mode == "missing_atom":
                row["units"].pop()
            else:
                row["units"] = []
            with self.subTest(mode=mode):
                self.assertFalse(finalize(self.original, self.candidate, self.ledger, guard)["decisions"][0]["accepted"])

    def test_preserved_requires_literal_match_and_correction_requires_source(self):
        guard = review(self.original, self.candidate, self.ledger)
        guard["patch_reviews"][0]["units"][1]["after_quote"] = "East."
        self.assertFalse(finalize(self.original, self.candidate, self.ledger, guard)["decisions"][0]["accepted"])
        guard = review(self.original, self.candidate, self.ledger)
        guard["patch_reviews"][0]["units"][0]["evidence_ids"] = []
        self.assertFalse(finalize(self.original, self.candidate, self.ledger, guard)["decisions"][0]["accepted"])

    def test_added_detail_with_only_preserved_units_is_held(self):
        self.candidate["patches"][0]["after"] = self.candidate["patches"][0]["before"] + " Extra detail."
        guard = review(self.original, self.candidate, self.ledger)
        guard["patch_reviews"][0]["additions"] = []
        unit = guard["patch_reviews"][0]["units"][0]
        unit.update({"status": "supported", "disposition": "preserved", "after_quote": unit["before_quote"]})
        result = finalize(self.original, self.candidate, self.ledger, guard)
        self.assertFalse(result["decisions"][0]["accepted"])
        self.assertTrue(any("detail alone" in reason for reason in result["decisions"][0]["reasons"]))

    def test_secondary_full_section_can_support_citation_only_relink(self):
        self.candidate["patches"][1]["action"] = "relink"
        self.ledger["evidence"].append({**self.ledger["evidence"][0], "evidence_id": "E2", "primary": False})
        guard = review(self.original, self.candidate, self.ledger)
        row = guard["patch_reviews"][1]
        row["evidence_ids"] = ["E2"]
        row["units"][0].update({"content_kind": "citation", "evidence_ids": ["E2"]})
        self.assertTrue(finalize(self.original, self.candidate, self.ledger, guard)["decisions"][1]["accepted"])
        row["units"][0]["content_kind"] = "factual"
        self.assertFalse(finalize(self.original, self.candidate, self.ledger, guard)["decisions"][1]["accepted"])

    def test_unused_source_removed_no_renumber_and_cross_patch_source_retained(self):
        guard = review(self.original, self.candidate, self.ledger)
        guard["patch_reviews"][0]["necessity"] = "optional"
        result = finalize(self.original, self.candidate, self.ledger, guard)
        self.assertEqual([source["citation_id"] for source in result["effective_repair"]["added_sources"]], ["3"])
        self.candidate["patches"][0]["after"] += " [3]"
        guard = review(self.original, self.candidate, self.ledger)
        guard["patch_reviews"][1]["necessity"] = "optional"
        result = finalize(self.original, self.candidate, self.ledger, guard)
        self.assertEqual([source["citation_id"] for source in result["effective_repair"]["added_sources"]], [2, "3"])

    def test_source_url_dependency_and_no_source_number_substring_confusion(self):
        self.candidate["patches"][0]["after"] = "The count was 6. The location was East. [23]"
        self.candidate["patches"][0]["source_urls"] = ["https://example.org/data"]
        guard = review(self.original, self.candidate, self.ledger)
        guard["patch_reviews"][0]["additions"][0]["after_quote"] = "[23]"
        guard["patch_reviews"][1]["necessity"] = "optional"
        result = finalize(self.original, self.candidate, self.ledger, guard)
        self.assertEqual([source["citation_id"] for source in result["effective_repair"]["added_sources"]], [2])

    def test_patch_dependencies_cascade_and_unknown_or_cycles_invalid(self):
        self.candidate["patches"][0]["depends_on_patch_ids"] = ["P2"]
        guard = review(self.original, self.candidate, self.ledger)
        guard["patch_reviews"][1]["necessity"] = "optional"
        result = finalize(self.original, self.candidate, self.ledger, guard)
        self.assertEqual(result["summary"]["accepted_patch_count"], 0)
        self.assertTrue(any("dependencies" in reason for reason in result["decisions"][0]["reasons"]))
        self.candidate["patches"][1]["depends_on_patch_ids"] = ["P1"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            build_guard_packet(self.original, self.candidate, self.ledger)
        self.candidate["patches"][0]["depends_on_patch_ids"] = ["P99"]
        with self.assertRaisesRegex(ValueError, "unknown"):
            build_guard_packet(self.original, self.candidate, self.ledger)

    def test_hashes_bind_original_candidate_ledger_and_guard_packet(self):
        packet = build_guard_packet(self.original, self.candidate, self.ledger)
        guard = review(self.original, self.candidate, self.ledger, packet=packet)
        self.candidate["patches"][0]["reason"] = "Tampered candidate metadata"
        with self.assertRaisesRegex(ValueError, "match"):
            preview_repair(self.original, self.candidate, self.ledger, guard, guard_packet=packet)
        self.candidate = fixture()[1]
        guard["guard_packet_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            preview_repair(self.original, self.candidate, self.ledger, guard, guard_packet=packet)
        guard = review(self.original, self.candidate, self.ledger, packet=packet)
        packet["candidate_applied"]["text"] += "tamper"
        with self.assertRaisesRegex(ValueError, "match"):
            preview_repair(self.original, self.candidate, self.ledger, guard, guard_packet=packet)

    def test_final_review_required_and_selection_hashes_ids_exact(self):
        guard = review(self.original, self.candidate, self.ledger)
        with self.assertRaisesRegex(ValueError, "final_review"):
            gate_repair(self.original, self.candidate, self.ledger, guard)
        for field, value in (("accepted_patch_ids", []), ("final_text_sha256_utf8", "0" * 64), ("final_sources_sha256", "0" * 64)):
            finalize(self.original, self.candidate, self.ledger, guard)
            guard["final_review"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "mismatch"):
                gate_repair(self.original, self.candidate, self.ledger, guard)

    def test_failed_or_incomplete_final_review_holds_all_and_keeps_preview(self):
        for status, complete in (("fail", True), ("pass", False)):
            guard = review(self.original, self.candidate, self.ledger)
            result = finalize(self.original, self.candidate, self.ledger, guard, status=status, complete=complete)
            self.assertEqual(result["final_applied"]["text"], self.original["text"])
            self.assertEqual(result["final_applied"]["sources"], self.original["sources"])
            self.assertNotEqual(result["selected_applied"]["text"], self.original["text"])
            self.assertEqual(result["summary"]["accepted_patch_count"], 0)
            self.assertFalse(result["summary"]["final_review_passed"])

    def test_empty_candidate_valid_but_final_combined_review_still_required(self):
        self.candidate["patches"] = []
        guard = review(self.original, self.candidate, self.ledger)
        result = finalize(self.original, self.candidate, self.ledger, guard)
        self.assertEqual(result["final_applied"]["text"], self.original["text"])
        self.assertEqual(result["effective_repair"]["added_sources"], [])
        self.assertEqual(result["decisions"], [])

    def test_cli_raw_freeze_preview_apply_and_immutable_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {key: root / (key + ".json") for key in ("original", "candidate", "ledger")}
            for key, record in zip(paths, (self.original, self.candidate, self.ledger)):
                paths[key].write_bytes(rep._json_bytes(record))
            flags = [item for key, path in paths.items() for item in ("--" + key, str(path))]
            prepared, previewed, output = root / "prepared", root / "preview", root / "output"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["prepare", *flags, "--output-dir", str(prepared)]), 0)
                packet = rep._read_json(prepared / "guard_packet.json")
                guard = review(self.original, self.candidate, self.ledger, packet=packet)
                guard_path = root / "guard.json"
                guard_path.write_bytes(rep._json_bytes(guard))
                guard_flags = ["--guard", str(guard_path), "--guard-packet", str(prepared / "guard_packet.json")]
                self.assertEqual(cli.main(["preview", *flags, *guard_flags, "--output-dir", str(previewed)]), 0)
                self.assertTrue((previewed / "selected_report.json").exists())
                self.assertFalse((previewed / "repair.effective.json").exists())
                finalize(self.original, self.candidate, self.ledger, guard, packet=packet)
                guard_path.write_bytes(rep._json_bytes(guard))
                command = ["apply", *flags, *guard_flags, "--output-dir", str(output)]
                self.assertEqual(cli.main(command), 0)
                self.assertEqual(cli.main(command), 0)
                self.assertEqual((output / "candidate.raw.json").read_bytes(), paths["candidate"].read_bytes())
                self.assertEqual((output / "guard.raw.json").read_bytes(), guard_path.read_bytes())
                (output / "summary.json").write_text("tamper", encoding="utf-8")
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli.main(command)
                # Semantically identical JSON with different raw bytes is also
                # rejected after preparation, even though canonical hashes match.
                paths["original"].write_bytes(paths["original"].read_bytes() + b"\n")
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli.main(["apply", *flags, *guard_flags, "--output-dir", str(root / "tampered")])
                self.assertFalse((root / "tampered").exists())

    def test_schema_contract_describes_final_review_and_scope(self):
        contract = schema_contract()
        self.assertIn("final_review", contract["guard"])
        self.assertIn("report_scope", contract["ledger"]["claims"][0])
        self.assertIn("source_scope", contract["ledger"]["claims"][0])

    def test_additions_required_exact_and_unknown_additions_held(self):
        for mode in ("missing", "quote", "uncertain", "contradicted", "missing_primary", "missing_scope"):
            guard = review(self.original, self.candidate, self.ledger)
            row = guard["patch_reviews"][0]
            if mode == "missing":
                del row["additions"]
            elif mode == "quote":
                row["additions"][0]["after_quote"] = "not present"
            elif mode == "missing_primary":
                row["additions"][0].update({"content_kind": "factual", "evidence_ids": []})
            elif mode == "missing_scope":
                row["additions"][0]["source_scope"] = ""
            else:
                row["additions"][0]["status"] = mode
            with self.subTest(mode=mode):
                if mode in {"missing", "quote", "missing_scope"}:
                    with self.assertRaises(ValueError):
                        preview_repair(self.original, self.candidate, self.ledger, guard)
                else:
                    self.assertFalse(finalize(self.original, self.candidate, self.ledger, guard)["decisions"][0]["accepted"])

    def test_added_facts_require_primary_but_secondary_citations_valid(self):
        self.ledger["evidence"].append({**self.ledger["evidence"][0], "evidence_id": "E2", "primary": False})
        guard = review(self.original, self.candidate, self.ledger)
        addition = guard["patch_reviews"][0]["additions"][0]
        addition["evidence_ids"] = ["E2"]
        self.assertTrue(finalize(self.original, self.candidate, self.ledger, guard)["decisions"][0]["accepted"])
        addition["content_kind"] = "factual"
        self.assertFalse(finalize(self.original, self.candidate, self.ledger, guard)["decisions"][0]["accepted"])
        addition["evidence_ids"] = ["E1"]
        self.assertTrue(finalize(self.original, self.candidate, self.ledger, guard)["decisions"][0]["accepted"])


if __name__ == "__main__":
    unittest.main()
