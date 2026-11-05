"""Synthetic v6 boundary, dependency, recovery and immutable-CLI tests."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import claim_guard_v6 as cli
import replication as rep
from factcheck import claim_guard_v5 as v5
from factcheck import claim_guard_v6 as v6

_spec = importlib.util.spec_from_file_location("v5_synthetic_fixtures", Path(__file__).with_name("test_claim_guard_v5.py"))
v5_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v5_fixtures)


def fixture(text="Alpha old.\nBeta old.\n", specs=None):
    """Build fully synthetic source-backed correction reviews for arbitrary spans."""
    if specs is None:
        specs = [("Alpha old.", "Alpha new.", []), ("Beta old.", "Beta new.", [])]
    original = {"text": text, "sources": [{"citation_id": identity, "title": "Synthetic source",
        "url": f"https://example.org/source/{identity}"} for identity in (1, 8, 11, 23, 2026)]}
    evidence = {"evidence_id": "E", "url": "https://example.org/source/1", "locator": "Synthetic table",
        "summary": "Synthetic data", "access": "full_section", "primary": True}
    blocks = v6.build_block_packet(original)
    ledger = {"schema_version": 1, "original_sha256": v6.sha256_json(original),
        "block_packet_sha256": v6.sha256_json(blocks), "evidence": [evidence], "blocks": [], "claims": []}
    for index, block in enumerate(blocks["blocks"]):
        identity = f"C{index + 1}"
        checks = {key: {"status": "not_applicable", "rationale": "Synthetic", "evidence_ids": []} for key in v6.CHECKS}
        checks["quantity"] = {"status": "contradicted", "rationale": "Synthetic", "evidence_ids": ["E"]}
        ledger["blocks"].append({"block_id": block["block_id"], "claim_ids": [identity], "nonfactual_reason": ""})
        ledger["claims"].append({"claim_id": identity, "block_id": block["block_id"], "quote": block["text"],
            "context_block_ids": [block["block_id"]], "report_scope": "Synthetic", "source_scope": "Synthetic",
            "status": "problem", "reason": "Synthetic", "checks": checks})
    candidate = {"execution": {"started_utc": "2026-11-01T00:00:00Z", "finished_utc": "2026-11-01T00:01:00Z",
        "web_calls": 0, "input_files": []}, "findings": [], "checked_controls": [], "unresolved": [],
        "limitations": [], "added_sources": [], "patches": []}
    reviews = []
    for index, (before, after, citation_quotes) in enumerate(specs):
        identity = f"P{index + 1}"
        start = text.index(before)
        end = start + len(before)
        refs = [f"C{i + 1}" for i, block in enumerate(blocks["blocks"])
                if block["start_char"] < end and start < block["end_char"]]
        candidate["findings"].append({"finding_id": identity, "quote": before, "status": "confirmed_problem",
            "tags": ["factual"], "diagnosis": "Synthetic", "evidence": []})
        candidate["patches"].append({"patch_id": identity, "finding_ids": [identity], "claim_ids": refs,
            "before": before, "after": after, "reason": "Synthetic", "source_urls": [evidence["url"]], "action": "replace"})
        reviews.append({"patch_id": identity, "necessity": "necessary", "rationale": "Synthetic correction",
            "evidence_ids": ["E"], "coverage_complete": True, "coverage_reason": "Every assertion reviewed",
            "units": [{"unit_id": "U", "claim_ids": refs, "before_quote": before, "after_quote": after,
                "status": "contradicted", "content_kind": "factual", "evidence_ids": ["E"],
                "reason": "Synthetic", "rationale": "Synthetic", "disposition": "corrected_with_evidence" if after else "removed_unsupported"}],
            "additions": [{"after_quote": quote, "content_kind": "citation", "claim_ids": refs,
                "status": "supported", "evidence_ids": ["E"], "reason": "Synthetic citation",
                "report_scope": "Synthetic", "source_scope": "Synthetic"} for quote in citation_quotes]})
    packet = v6.build_guard_packet(original, candidate, ledger)
    guard = {"schema_version": 1, "guard_packet_sha256": v6.sha256_json(packet),
        "input_sha256": packet["input_sha256"], "evidence": [], "patch_reviews": reviews}
    return original, candidate, ledger, guard


def rebind(original, candidate, ledger, guard):
    packet = v6.build_guard_packet(original, candidate, ledger)
    guard.update(guard_packet_sha256=v6.sha256_json(packet), input_sha256=packet["input_sha256"])
    return packet


def review(selection, *, status="pass", complete=True, ids=(), localization=None, all_localized=None):
    localization = localization or ("definite" if ids else "none")
    return {**copy.deepcopy(selection), "status": status, "coverage_complete": complete,
        "reason": "Synthetic independent full-report review",
        "patch_localization": {"status": localization, "patch_ids": list(ids),
            "all_failure_causes_localized": bool(ids) if all_localized is None else all_localized,
            "reason": "All defects localized" if ids else "No localized defects"}}


class BoundaryTests(unittest.TestCase):
    def test_unrelated_adjacent_body_edit_survives_citation_rejection(self):
        data = fixture("Aold.Bbody", [("Aold.", "Anew.\n11", ["11"]), ("Bbody", "Xnew", [])])
        result = v6.preview_repair(*data)
        self.assertEqual(result["selection"]["accepted_patch_ids"], ["P2"])
        diagnostic = result["mechanical_diagnostics"][0]["violations"][0]
        self.assertEqual(diagnostic["responsible_patch_ids"], ["P1"])
        self.assertIn({"patch_id": "P2", "same_join_remains_without_patch": True}, diagnostic["counterfactual_attribution"])

    def test_repeated_citation_quote_does_not_promote_ambiguous_numeric_occurrences(self):
        data = fixture("Old.Body", [("Old.", "New.\n11\nValue\n11", ["11"])])
        result = v6.preview_repair(*data)
        self.assertEqual(result["selection"]["accepted_patch_ids"], ["P1"])
        scan = result["mechanical_diagnostics"][0]
        self.assertFalse(scan["violations"])
        self.assertEqual(scan["intent_ambiguities"][0]["occurrence_count"], 2)

    def test_broad_citation_quote_does_not_promote_numeric_table_value(self):
        after = "New.\nValue\n11"
        data = fixture("Old.Body", [("Old.", after, [after])])
        result = v6.preview_repair(*data)
        self.assertEqual(result["selection"]["accepted_patch_ids"], ["P1"])
        self.assertFalse(result["mechanical_diagnostics"][0]["violations"])

    def test_unique_declared_citation_already_joined_inside_replacement_is_screened(self):
        for after, quote in (("New.\n11Body", "\n11"), ("New.\n11Body", "11"),
                             ("Body11\nNew.", "11")):
            with self.subTest(after=after, quote=quote):
                result = v6.preview_repair(*fixture("Old.", [("Old.", after, [quote])]))
                self.assertEqual(result["selection"]["accepted_patch_ids"], [])
                self.assertEqual(result["mechanical_diagnostics"][0]["violations"][0]["responsible_patch_ids"], ["P1"])

    def test_all_v5_material_omission_categories_survive_without_mutating_inputs(self):
        for check in v5.CHECKS:
            with self.subTest(check=check):
                data = v5_fixtures.omission_fixture("citation" if check == "citation" else "factual", check)
                before = copy.deepcopy(data)
                old, new = v5.preview_repair(*data), v6.preview_repair(*data)
                self.assertEqual(old["decisions"], new["decisions"])
                self.assertEqual(new["selection"]["accepted_patch_ids"], ["P1"])
                self.assertEqual(old["selected_applied"], new["selected_applied"])
                self.assertEqual(data, before)

    def test_v5_supported_information_loss_still_holds_entire_patch(self):
        data = v5_fixtures.omission_fixture()
        data[3]["patch_reviews"][0]["units"][1].update(disposition="lost", after_quote="")
        old, new = v5.preview_repair(*data), v6.preview_repair(*data)
        self.assertEqual(old["decisions"], new["decisions"])
        self.assertEqual(new["selection"]["accepted_patch_ids"], [])

    def test_mechanical_rejection_keeps_source_still_referenced_by_survivor(self):
        original, candidate, ledger = v5_fixtures.legacy.fixture()
        candidate["patches"][0]["before"] += "\n"
        candidate["patches"][0]["after"] = candidate["patches"][0]["after"].replace(" [2]", "\n2")
        candidate["patches"][1]["after"] = candidate["patches"][1]["after"].replace("[3]", "[2]")
        candidate["patches"][1]["source_urls"] = candidate["patches"][0]["source_urls"][:]
        guard = v5_fixtures.legacy.review(original, candidate, ledger)
        guard["patch_reviews"][0]["additions"][0]["after_quote"] = "\n2"
        guard["patch_reviews"][1]["additions"][0]["after_quote"] = "[2]"
        result = v6.preview_repair(original, candidate, ledger, guard)
        self.assertEqual(result["selection"]["accepted_patch_ids"], ["P2"])
        self.assertEqual([source["citation_id"] for source in result["selected_applied"]["sources"]], [1, 2])
        self.assertEqual(original["sources"], result["selected_applied"]["sources"][:1])

    def test_known_regression_shapes_generalize_to_body_dates_and_both_edges(self):
        cases = [("Old.Body", "Old.", "New.\n11", "11"),
                 ("Old.9月", "Old.", "New.\n8", "8"),
                 ("Old.2026年", "Old.", "New.\n1\n23", "1\n23"),
                 ("BodyOld.", "Old.", "11\nNew.", "11"),
                 ("2026年Old.", "Old.", "23\nNew.", "23")]
        for text, before, after, quote in cases:
            with self.subTest(text=text):
                data = fixture(text, [(before, after, [quote])])
                result = v6.preview_repair(*data)
                self.assertEqual(result["selection"]["accepted_patch_ids"], [])
                violations = result["mechanical_diagnostics"][0]["violations"]
                self.assertEqual(violations[0]["responsible_patch_ids"], ["P1"])
                self.assertEqual(result["final_applied"]["text"], text)

    def test_ordinary_numbers_ids_years_amounts_and_bracketed_citations_pass(self):
        afters = ["1", "11", "2026", "Amount 11", "ID11", "2026年", "11円", "1.11", "[11]", "New.\n11"]
        for after in afters:
            with self.subTest(after=after):
                citations = ["[11]"] if after == "[11]" else []
                result = v6.preview_repair(*fixture("Old.Body", [("Old.", after, citations)]))
                self.assertEqual(result["selection"]["accepted_patch_ids"], ["P1"])
                self.assertFalse(result["mechanical_diagnostics"][0]["violations"])

    def test_standalone_citation_and_horizontal_padding_with_line_break_pass(self):
        for after in ("New.\n11", "New.\n 11 \t", "New.\r\n11"):
            with self.subTest(after=after):
                result = v6.preview_repair(*fixture("Old.\nBody", [("Old.", after, ["11"])]))
                self.assertEqual(result["selection"]["accepted_patch_ids"], ["P1"])

    def test_existing_bad_original_join_is_not_blame_for_unchanged_token(self):
        data = fixture("Old.\n11Body", [("Old.\n11", "New.\n11", ["11"])])
        result = v6.preview_repair(*data)
        self.assertEqual(result["selection"]["accepted_patch_ids"], ["P1"])

    def test_separator_deletion_blames_deleter_not_unchanged_original_citation(self):
        data = fixture("11\nBody old.", [("11\nBody old.", "11Body new.", [])])
        data[3]["patch_reviews"][0]["units"].append({
            "unit_id": "Citation", "claim_ids": ["C1"], "before_quote": "11", "after_quote": "11",
            "status": "supported", "content_kind": "citation", "evidence_ids": ["E"],
            "reason": "Synthetic original citation", "rationale": "Retained token", "disposition": "preserved"})
        result = v6.preview_repair(*data)
        self.assertEqual(result["selection"]["accepted_patch_ids"], [])
        diagnostic = result["mechanical_diagnostics"][0]["violations"][0]
        self.assertEqual(diagnostic["responsible_patch_ids"], ["P1"])
        self.assertEqual(diagnostic["intent_origin"], "original_standalone_line")

    def test_original_numeric_table_value_matching_source_id_is_not_promoted(self):
        for number in ("1", "11", "2026"):
            with self.subTest(number=number):
                data = fixture(f"{number}\nAmount old.", [(f"{number}\nAmount old.", f"{number}Amount new.", [])])
                result = v6.preview_repair(*data)
                self.assertEqual(result["selection"]["accepted_patch_ids"], ["P1"])
                self.assertFalse(result["mechanical_diagnostics"][0]["violations"])

    def test_dependency_chain_and_diamond_are_closed_but_unrelated_patch_remains(self):
        specs = [("Aold.", "Anew.\n11", ["11"])] + [(f"{letter}old.", f"{letter}new.", []) for letter in "BCDE"]
        data = fixture("Aold.Body\nBold.\nCold.\nDold.\nEold.", specs)
        original, candidate, ledger, guard = data
        for index, refs in ((1, ["P1"]), (2, ["P1"]), (3, ["P2", "P3"])):
            candidate["patches"][index]["depends_on_patch_ids"] = refs
        rebind(*data)
        result = v6.preview_repair(*data)
        self.assertEqual(result["selection"]["accepted_patch_ids"], ["P5"])
        self.assertEqual({row["patch_id"] for row in result["exclusion_trail"] if row["phase"] == "dependency"}, {"P2", "P3", "P4"})

    def test_rejection_exposes_boundary_and_rescreens_to_fixed_point(self):
        data = fixture("Aold.BbodyCbody", [("Aold.", "Anew.\n11", ["11"]),
            ("Bbody", "\nBnew\n23", ["23"]), ("Cbody", "Cnew", [])])
        result = v6.preview_repair(*data)
        # P2 initially separates P1's 11, but its 23 joins P3. Removing P2
        # restores Bbody and exposes P1's previously safe 11.
        self.assertEqual(result["selection"]["accepted_patch_ids"], ["P3"])
        passes = result["mechanical_diagnostics"]
        self.assertEqual(len(passes), 3)
        self.assertEqual(passes[0]["violations"][0]["responsible_patch_ids"], ["P2"])
        self.assertEqual(passes[1]["violations"][0]["responsible_patch_ids"], ["P1"])

    def test_preserves_v5_eligibility_information_and_source_safety(self):
        data = fixture()
        original, candidate, ledger, guard = data
        guard["patch_reviews"][0]["necessity"] = "optional"
        guard["patch_reviews"][1]["units"][0]["disposition"] = "lost"
        old, new = v5.preview_repair(*data), v6.preview_repair(*data)
        self.assertEqual(old["decisions"], new["decisions"])
        self.assertEqual(v5.build_guard_packet(*data[:3]), v6.build_guard_packet(*data[:3]))


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()
        self.preview = v6.preview_repair(*self.data)

    def gate(self, reviews):
        guard = copy.deepcopy(self.data[3])
        guard["final_reviews"] = reviews
        return v6.gate_repair(*self.data[:3], guard)

    def test_full_pass_and_no_input_mutation(self):
        frozen = copy.deepcopy(self.data)
        result = self.gate([review(self.preview["selection"])])
        self.assertTrue(result["summary"]["final_review_passed"])
        self.assertEqual(result["summary"]["accepted_patch_count"], 2)
        self.assertEqual(self.data, frozen)

    def test_localized_removal_requires_fresh_full_pass(self):
        first = review(self.preview["selection"], status="fail", ids=["P1"])
        next_result = v6.prepare_recovery(*self.data, final_review=first)
        self.assertTrue(next_result["recovery_eligible"])
        self.assertEqual(next_result["selection"]["accepted_patch_ids"], ["P2"])
        self.assertEqual(next_result["selection"]["previous_review_sha256"], v6.sha256_json(first))
        self.assertNotEqual(next_result["selection"]["selection_sha256"], self.preview["selection"]["selection_sha256"])
        no_second = self.gate([first])
        self.assertFalse(no_second["summary"]["final_review_passed"])
        self.assertEqual(no_second["final_applied"]["text"], self.data[0]["text"])
        result = self.gate([first, review(next_result["selection"])])
        self.assertTrue(result["summary"]["final_review_passed"])
        self.assertEqual(result["summary"]["accepted_patch_count"], 1)
        self.assertEqual(len(result["review_history"]), 2)

    def test_empty_recovery_remainder_terminates_without_claiming_a_pass(self):
        first = review(self.preview["selection"], status="fail", ids=["P1", "P2"])
        prepared = v6.prepare_recovery(*self.data, final_review=first)
        self.assertFalse(prepared["recovery_eligible"])
        self.assertEqual(prepared["recovery_status"], "no_remainder")
        self.assertEqual(prepared["selection"]["accepted_patch_ids"], [])
        result = self.gate([first])
        self.assertEqual(result["final_applied"]["text"], self.data[0]["text"])
        self.assertFalse(result["summary"]["final_review_passed"])
        self.assertFalse(result["summary"]["final_review_required"])
        self.assertEqual(result["summary"]["recovery_status"], "no_remainder")
        with self.assertRaisesRegex(ValueError, "extra review"):
            self.gate([first, review(prepared["selection"])])

    def test_stale_review_and_changed_guard_cannot_approve_changed_selection(self):
        first = review(self.preview["selection"], status="fail", ids=["P1"])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.gate([first, review(self.preview["selection"])])
        guard = copy.deepcopy(self.data[3])
        guard["patch_reviews"][0]["rationale"] += " changed"
        guard["final_reviews"] = [review(self.preview["selection"])]
        with self.assertRaisesRegex(ValueError, "guard_evaluation_sha256"):
            v6.gate_repair(*self.data[:3], guard)

    def test_source_hash_is_bound_even_when_selected_text_is_unchanged(self):
        passed = review(self.preview["selection"])
        passed["final_sources_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "final_sources_sha256"):
            self.gate([passed])

    def test_failed_incomplete_or_unknown_second_review_keeps_full_veto(self):
        first = review(self.preview["selection"], status="fail", ids=["P1"])
        selection = v6.prepare_recovery(*self.data, final_review=first)["selection"]
        for kwargs in ({"status": "fail", "ids": ["P2"]}, {"complete": False}, {"status": "unknown"}):
            with self.subTest(kwargs=kwargs):
                result = self.gate([first, review(selection, **kwargs)])
                self.assertFalse(result["summary"]["final_review_passed"])
                self.assertEqual(result["summary"]["accepted_patch_count"], 0)
                self.assertEqual(result["final_applied"]["text"], self.data[0]["text"])

    def test_uncertain_incomplete_absent_and_unattested_localization_never_recovers(self):
        for kwargs in ({"status": "unknown", "ids": ["P1"]}, {"status": "fail", "ids": ["P1"], "complete": False},
            {"status": "fail"}, {"status": "fail", "ids": ["P1"], "localization": "ambiguous"},
            {"status": "fail", "ids": ["P1"], "localization": "unknown"},
            {"status": "fail", "ids": ["P1"], "all_localized": False}):
            with self.subTest(kwargs=kwargs):
                first = review(self.preview["selection"], **kwargs)
                result = self.gate([first])
                self.assertEqual(result["summary"]["accepted_patch_count"], 0)
                prepared = v6.prepare_recovery(*self.data, final_review=first)
                self.assertFalse(prepared["recovery_eligible"])

    def test_malformed_unknown_duplicate_and_unselected_localization_rejected(self):
        first = review(self.preview["selection"], status="fail", ids=["P1"])
        for mutation in (lambda r: r.pop("patch_localization"),
                         lambda r: r["patch_localization"].update(patch_ids=["unknown"]),
                         lambda r: r["patch_localization"].update(patch_ids=["P1", "P1"]),
                         lambda r: r["patch_localization"].update(all_failure_causes_localized="true"),
                         lambda r: r.update(status="pass"), lambda r: r.update(unexpected=True)):
            bad = copy.deepcopy(first)
            mutation(bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.gate([bad])

    def test_round_limit_and_extra_review_after_pass_rejected(self):
        passed = review(self.preview["selection"])
        for reviews in ([], [passed, passed], [passed, passed, passed]):
            with self.subTest(count=len(reviews)), self.assertRaises(ValueError):
                self.gate(reviews)

    def test_localized_dependency_closure_and_mechanical_rescreen(self):
        data = fixture("Aold.Bbody\nCold.\nDold.", [("Aold.", "Anew.\n11", ["11"]),
            ("Bbody", "\nBnew", []), ("Cold.", "Cnew.", []), ("Dold.", "Dnew.", [])])
        data[1]["patches"][2]["depends_on_patch_ids"] = ["P2"]
        rebind(*data)
        initial = v6.preview_repair(*data)
        first = review(initial["selection"], status="fail", ids=["P2"])
        next_result = v6.prepare_recovery(*data, final_review=first)
        self.assertEqual(next_result["selection"]["accepted_patch_ids"], ["P4"])
        phases = {row["patch_id"]: row["phase"] for row in next_result["exclusion_trail"]}
        self.assertEqual(phases, {"P2": "localized_final_review", "P3": "dependency", "P1": "mechanical"})

    def test_cli_preview_recover_apply_and_immutable_reexecution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = []
            for name, record in zip(("original", "candidate", "ledger"), self.data[:3]):
                path = root / f"{name}.json"
                path.write_bytes(rep._json_bytes(record))
                args.extend([f"--{name}", str(path)])
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(["prepare", *args, "--output-dir", str(root / "prepare")])
            packet = rep._read_json(root / "prepare/guard_packet.json")
            guard = copy.deepcopy(self.data[3])
            guard["guard_packet_sha256"] = v6.sha256_json(packet)
            guard_path = root / "guard.json"
            guard_path.write_bytes(rep._json_bytes(guard))
            shared = [*args, "--guard-packet", str(root / "prepare/guard_packet.json"), "--guard", str(guard_path)]
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(["preview", *shared, "--output-dir", str(root / "preview")])
            first = review(rep._read_json(root / "preview/selection.json"), status="fail", ids=["P1"])
            first_path = root / "first.json"
            first_path.write_bytes(rep._json_bytes(first))
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(["recover", *shared, "--final-review", str(first_path), "--output-dir", str(root / "recover")])
            guard["final_reviews"] = [first, review(rep._read_json(root / "recover/selection.json"))]
            guard_path.write_bytes(rep._json_bytes(guard))
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(["apply", *shared, "--output-dir", str(root / "apply")])
            self.assertTrue(rep._read_json(root / "apply/summary.json")["final_review_passed"])
            self.assertEqual(len(rep._read_json(root / "apply/review_history.json")), 2)
            # Identical replay is idempotent; changed evidence cannot overwrite.
            guard["final_reviews"][1]["reason"] += " changed"
            guard_path.write_bytes(rep._json_bytes(guard))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main(["apply", *shared, "--output-dir", str(root / "apply")])


if __name__ == "__main__":
    unittest.main()
