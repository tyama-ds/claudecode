"""Synthetic v7 uncertainty isolation, conservative limits and provenance tests."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import claim_guard_v7 as cli
import replication as rep
from factcheck import claim_guard_v6 as v6
from factcheck import claim_guard_v7 as v7

_spec = importlib.util.spec_from_file_location("v6_synthetic_fixtures_for_v7", Path(__file__).with_name("test_claim_guard_v6.py"))
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)
fixture, rebind = fixtures.fixture, fixtures.rebind


def review(selection, **kwargs):
    result = fixtures.review(selection, **kwargs)
    result["uncertainty_localization"] = {"status": "none", "patch_ids": [], "patch_reasons": {},
        "all_unresolved_concerns_localized": False, "remainder_coverage_complete": False,
        "no_preexisting_or_nonlocal_concerns": False, "reason": "No approval-blocking uncertainty"}
    return result


def uncertain(selection, ids=("P1",)):
    result = review(selection, status="unknown", complete=False)
    result["uncertainty_localization"] = {"status": "definite", "patch_ids": list(ids),
        "patch_reasons": {identity: f"Source for {identity} was unavailable" for identity in ids},
        "all_unresolved_concerns_localized": True, "remainder_coverage_complete": True,
        "no_preexisting_or_nonlocal_concerns": True,
        "reason": "All other changes and their interactions were actually reviewed; uncertainty confined to these patches"}
    return result


class UncertaintyRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()
        self.preview = v7.preview_repair(*self.data)

    def gate(self, reviews, data=None):
        data = self.data if data is None else data
        guard = copy.deepcopy(data[3])
        guard["final_reviews"] = reviews
        return v7.gate_repair(*data[:3], guard)

    def test_unknown_localization_prepares_but_does_not_authorize_application(self):
        first = uncertain(self.preview["selection"])
        prepared = v7.prepare_recovery(*self.data, final_review=first)
        self.assertTrue(prepared["recovery_eligible"])
        self.assertEqual(prepared["selection"]["accepted_patch_ids"], ["P2"])
        self.assertEqual(prepared["recovery_kind"], "localized_uncertainty")
        result = self.gate([first])
        self.assertEqual(result["final_applied"]["text"], self.data[0]["text"])
        self.assertFalse(result["summary"]["final_review_passed"])
        self.assertEqual(result["summary"]["localized_uncertainty_hold_rounds"], 1)

    def test_fresh_pass_adopts_only_remainder_and_preserves_raw_unknown(self):
        frozen = copy.deepcopy(self.data)
        first = uncertain(self.preview["selection"])
        prepared = v7.prepare_recovery(*self.data, final_review=first)
        result = self.gate([first, review(prepared["selection"])])
        self.assertEqual(result["final_applied"]["text"], "Alpha old.\nBeta new.\n")
        self.assertTrue(result["summary"]["final_review_passed"])
        self.assertEqual(result["summary"]["accepted_patch_count"], 1)
        self.assertEqual(result["review_history"][0]["review"], first)
        self.assertEqual(result["review_history"][0]["review"]["status"], "unknown")
        self.assertFalse(result["summary"]["uncertainty_hold_is_error_finding"])
        self.assertEqual(result["summary"]["localized_failure_removal_rounds"], 0)
        self.assertEqual(self.data, frozen)

    def test_same_semantic_unknown_v6_vetoes_v7_can_recover(self):
        initial6 = v6.preview_repair(*self.data)
        first7 = uncertain(self.preview["selection"])
        first6 = {**copy.deepcopy(first7), **initial6["selection"]}
        first6.pop("uncertainty_localization")
        old_guard = {**copy.deepcopy(self.data[3]), "final_reviews": [first6]}
        old = v6.gate_repair(*self.data[:3], old_guard)
        next7 = v7.prepare_recovery(*self.data, final_review=first7)
        new = self.gate([first7, review(next7["selection"])])
        self.assertEqual(old["final_applied"]["text"], self.data[0]["text"])
        self.assertEqual(new["summary"]["accepted_patch_count"], 1)
        self.assertEqual(first6["status"], first7["status"])
        self.assertEqual(first6["patch_localization"], first7["patch_localization"])

    def test_all_three_uncertainty_attestations_are_necessary(self):
        for field in ("all_unresolved_concerns_localized", "remainder_coverage_complete", "no_preexisting_or_nonlocal_concerns"):
            with self.subTest(field=field):
                first = uncertain(self.preview["selection"])
                first["uncertainty_localization"][field] = False
                result = self.gate([first])
                self.assertEqual(result["summary"]["accepted_patch_count"], 0)
                self.assertEqual(result["summary"]["localized_uncertainty_hold_rounds"], 0)
                self.assertFalse(v7.prepare_recovery(*self.data, final_review=first)["recovery_eligible"])

    def test_uncertain_ambiguous_or_empty_location_never_authorizes_recovery(self):
        for status in ("ambiguous", "unknown"):
            with self.subTest(status=status):
                first = uncertain(self.preview["selection"])
                first["uncertainty_localization"]["status"] = status
                self.assertEqual(self.gate([first])["summary"]["accepted_patch_count"], 0)
        for first in (review(self.preview["selection"], status="unknown", complete=False),
                      uncertain(self.preview["selection"], ids=[])):
            self.assertEqual(self.gate([first])["summary"]["accepted_patch_count"], 0)

    def test_nonlocal_unknown_cannot_be_hidden_by_complete_overall_coverage(self):
        first = uncertain(self.preview["selection"])
        first["coverage_complete"] = True
        first["uncertainty_localization"]["no_preexisting_or_nonlocal_concerns"] = False
        self.assertEqual(self.gate([first])["summary"]["accepted_patch_count"], 0)

    def test_mixed_failure_and_uncertainty_are_not_recovered(self):
        for status in ("unknown", "fail"):
            first = uncertain(self.preview["selection"])
            first.update(status=status, coverage_complete=True)
            first["patch_localization"] = fixtures.review(self.preview["selection"], status="fail", ids=["P2"])["patch_localization"]
            with self.subTest(status=status):
                self.assertEqual(self.gate([first])["summary"]["accepted_patch_count"], 0)

    def test_existing_complete_failure_recovery_is_preserved_and_separately_counted(self):
        first = review(self.preview["selection"], status="fail", ids=["P1"])
        prepared = v7.prepare_recovery(*self.data, final_review=first)
        result = self.gate([first, review(prepared["selection"])])
        self.assertEqual(result["summary"]["accepted_patch_count"], 1)
        self.assertEqual(result["summary"]["localized_failure_removal_rounds"], 1)
        self.assertEqual(result["summary"]["localized_uncertainty_hold_rounds"], 0)
        self.assertEqual(result["exclusion_trail"][0]["phase"], "localized_final_review")

    def test_incomplete_failure_with_no_uncertainty_localization_still_vetoes(self):
        first = review(self.preview["selection"], status="fail", ids=["P1"], complete=False)
        self.assertFalse(v7._recoverable(first))
        self.assertEqual(self.gate([first])["summary"]["accepted_patch_count"], 0)

    def test_pass_and_incomplete_pass_behavior_are_unchanged(self):
        good = self.gate([review(self.preview["selection"])])
        self.assertEqual(good["summary"]["accepted_patch_count"], 2)
        self.assertEqual(good["summary"]["recovery_kind"], "none")
        self.assertEqual(self.gate([review(self.preview["selection"], complete=False)])["summary"]["accepted_patch_count"], 0)

    def test_pass_cannot_contain_uncertainty_even_when_all_attestations_are_true(self):
        first = uncertain(self.preview["selection"])
        first.update(status="pass", coverage_complete=True)
        with self.assertRaisesRegex(ValueError, "pass requires none uncertainty"):
            self.gate([first])

    def test_exact_per_patch_reasons_are_kept_in_hold_trail(self):
        first = uncertain(self.preview["selection"], ids=["P1", "P2"])
        first["uncertainty_localization"]["patch_reasons"] = {"P1": "PDF unavailable", "P2": "Table scope could not be verified"}
        prepared = v7.prepare_recovery(*self.data, final_review=first)
        trail = {row["patch_id"]: row for row in prepared["exclusion_trail"]}
        for identity, reason in first["uncertainty_localization"]["patch_reasons"].items():
            self.assertEqual(trail[identity]["details"]["patch_reason"], reason)
            self.assertIn(reason, trail[identity]["reason"])
            self.assertEqual(trail[identity]["details"]["review_status"], "unknown")

    def test_malformed_or_incomplete_uncertainty_records_raise(self):
        mutations = [lambda r: r.pop("uncertainty_localization"),
            lambda r: r["uncertainty_localization"].update(patch_ids=["unknown"]),
            lambda r: r["uncertainty_localization"].update(patch_ids=["P1", "P1"]),
            lambda r: r["uncertainty_localization"].update(patch_reasons={}),
            lambda r: r["uncertainty_localization"].update(patch_reasons={"P1": ""}),
            lambda r: r["uncertainty_localization"].update(patch_reasons={"P1": "a", "P2": "b"}),
            lambda r: r["uncertainty_localization"].update(remainder_coverage_complete="true"),
            lambda r: r["uncertainty_localization"].update(status="none"),
            lambda r: r["uncertainty_localization"].update(extra=True),
            lambda r: r.update(extra=True)]
        for index, mutation in enumerate(mutations):
            first = uncertain(self.preview["selection"])
            mutation(first)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.gate([first])

    def test_dependency_chain_diamond_hold_only_dependents(self):
        data = fixture("\n".join(f"{x}old." for x in "ABCDE"),
            [(f"{x}old.", f"{x}new.", []) for x in "ABCDE"])
        for i, dependencies in ((1, ["P1"]), (2, ["P1"]), (3, ["P2", "P3"])):
            data[1]["patches"][i]["depends_on_patch_ids"] = dependencies
        rebind(*data)
        first = uncertain(v7.preview_repair(*data)["selection"])
        prepared = v7.prepare_recovery(*data, final_review=first)
        self.assertEqual(prepared["selection"]["accepted_patch_ids"], ["P5"])
        self.assertEqual({row["patch_id"] for row in prepared["exclusion_trail"] if row["phase"] == "dependency"}, {"P2", "P3", "P4"})
        result = self.gate([first, review(prepared["selection"])], data)
        self.assertEqual(result["final_applied"]["text"], "Aold.\nBold.\nCold.\nDold.\nEnew.")

    def test_unknown_removal_rescreens_newly_exposed_citation_join(self):
        data = fixture("Aold.Bbody\nCold.\nDold.", [("Aold.", "Anew.\n11", ["11"]),
            ("Bbody", "\nBnew", []), ("Cold.", "Cnew.", []), ("Dold.", "Dnew.", [])])
        data[1]["patches"][2]["depends_on_patch_ids"] = ["P2"]
        rebind(*data)
        first = uncertain(v7.preview_repair(*data)["selection"], ids=["P2"])
        prepared = v7.prepare_recovery(*data, final_review=first)
        self.assertEqual(prepared["selection"]["accepted_patch_ids"], ["P4"])
        self.assertEqual({row["patch_id"]: row["phase"] for row in prepared["exclusion_trail"]},
                         {"P2": "localized_uncertainty", "P3": "dependency", "P1": "mechanical"})

    def test_empty_remainder_has_no_claimed_pass_and_rejects_extra_review(self):
        first = uncertain(self.preview["selection"], ids=["P1", "P2"])
        prepared = v7.prepare_recovery(*self.data, final_review=first)
        self.assertEqual(prepared["recovery_status"], "no_remainder")
        self.assertFalse(prepared["recovery_eligible"])
        result = self.gate([first])
        self.assertEqual(result["final_applied"]["text"], self.data[0]["text"])
        self.assertFalse(result["summary"]["final_review_passed"])
        self.assertFalse(result["summary"]["final_review_required"])
        with self.assertRaisesRegex(ValueError, "extra review"):
            self.gate([first, review(prepared["selection"])])

    def test_second_unknown_failure_or_incomplete_cannot_start_another_round(self):
        first = uncertain(self.preview["selection"])
        prepared = v7.prepare_recovery(*self.data, final_review=first)
        seconds = [uncertain(prepared["selection"], ids=["P2"]),
            review(prepared["selection"], status="fail", ids=["P2"]),
            review(prepared["selection"], complete=False)]
        for second in seconds:
            with self.subTest(status=second["status"]):
                result = self.gate([first, second])
                self.assertEqual(result["summary"]["accepted_patch_count"], 0)
                self.assertEqual(result["summary"]["semantic_recovery_rounds"], 1)
                self.assertEqual(len(result["review_history"]), 2)
                with self.assertRaisesRegex(ValueError, "one or two"):
                    self.gate([first, second, review(prepared["selection"])])

    def test_nonrecoverable_unknown_and_initial_pass_reject_extra_review(self):
        for first in (review(self.preview["selection"]), review(self.preview["selection"], status="unknown", complete=False)):
            with self.subTest(status=first["status"]), self.assertRaisesRegex(ValueError, "extra review"):
                self.gate([first, first])

    def test_stale_selection_prior_review_sources_and_guard_are_bound(self):
        first = uncertain(self.preview["selection"])
        prepared = v7.prepare_recovery(*self.data, final_review=first)
        self.assertEqual(prepared["selection"]["previous_review_sha256"], v7.sha256_json(first))
        for field in ("selection_sha256", "final_text_sha256_utf8", "final_sources_sha256", "guard_evaluation_sha256", "previous_review_sha256"):
            second = review(prepared["selection"])
            second[field] = "0" * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "mismatch"):
                self.gate([first, second])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.gate([first, review(self.preview["selection"])])
        changed = copy.deepcopy(self.data)
        changed[3]["patch_reviews"][0]["rationale"] += " altered"
        with self.assertRaisesRegex(ValueError, "guard_evaluation_sha256"):
            self.gate([first], changed)

    def test_v6_bound_review_cannot_authorize_v7(self):
        old = review(v6.preview_repair(*self.data)["selection"])
        with self.assertRaisesRegex(ValueError, "gate_version"):
            self.gate([old])

    def test_mechanical_preview_and_packet_remain_v6_equivalent(self):
        data = fixture("Aold.Body\nBold.", [("Aold.", "Anew.\n11", ["11"]), ("Bold.", "Bnew.", [])])
        old, new = v6.preview_repair(*data), v7.preview_repair(*data)
        for key in ("selected_applied", "decisions", "mechanical_diagnostics", "exclusion_trail"):
            self.assertEqual(old[key], new[key])
        self.assertEqual(v6.build_guard_packet(*data[:3]), v7.build_guard_packet(*data[:3]))
        self.assertEqual(new["selection"]["gate_version"], "v7")
        self.assertNotEqual(old["selection"]["selection_sha256"], new["selection"]["selection_sha256"])

    def test_old_eligibility_rejections_are_not_reenabled_by_recovery(self):
        data = fixture("\n".join(f"{x}old." for x in "ABC"), [(f"{x}old.", f"{x}new.", []) for x in "ABC"])
        data[3]["patch_reviews"][2]["necessity"] = "optional"
        first = uncertain(v7.preview_repair(*data)["selection"], ids=["P1"])
        prepared = v7.prepare_recovery(*data, final_review=first)
        self.assertEqual(prepared["selection"]["accepted_patch_ids"], ["P2"])
        result = self.gate([first, review(prepared["selection"])], data)
        self.assertEqual(result["final_applied"]["text"], "Aold.\nBnew.\nCold.")

    def test_contract_requires_real_remainder_interaction_review_and_single_total_round(self):
        contract = v7.schema_contract()
        uncertainty = contract["guard"]["final_review"]["uncertainty_localization"]
        self.assertIn("actually examined ALL remaining changes", uncertainty["remainder_coverage_complete"])
        self.assertIn("cross-document interactions", uncertainty["remainder_coverage_complete"])
        self.assertIn("never one of each", contract["guard"]["final_reviews"])

    def test_cli_unknown_recover_apply_and_immutable_artifacts(self):
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
            guard["guard_packet_sha256"] = v7.sha256_json(packet)
            guard_path = root / "guard.json"
            guard_path.write_bytes(rep._json_bytes(guard))
            shared = [*args, "--guard-packet", str(root / "prepare/guard_packet.json"), "--guard", str(guard_path)]
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(["preview", *shared, "--output-dir", str(root / "preview")])
            first = uncertain(rep._read_json(root / "preview/selection.json"))
            first_path = root / "first.json"
            first_path.write_bytes(rep._json_bytes(first))
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(["recover", *shared, "--final-review", str(first_path), "--output-dir", str(root / "recover")])
            guard["final_reviews"] = [first, review(rep._read_json(root / "recover/selection.json"))]
            guard_path.write_bytes(rep._json_bytes(guard))
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(["apply", *shared, "--output-dir", str(root / "apply")])
                cli.main(["apply", *shared, "--output-dir", str(root / "apply")])
            summary = rep._read_json(root / "apply/summary.json")
            self.assertEqual(summary["gate_version"], "v7")
            self.assertTrue(summary["final_review_passed"])
            self.assertEqual(rep._read_json(root / "apply/review_history.json")[0]["review"]["status"], "unknown")
            guard["final_reviews"][1]["reason"] += " changed"
            guard_path.write_bytes(rep._json_bytes(guard))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main(["apply", *shared, "--output-dir", str(root / "apply")])


if __name__ == "__main__":
    unittest.main()
