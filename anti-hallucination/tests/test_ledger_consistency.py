"""Synthetic conservative-normalization invariants plus the declared N2 regression."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import ledger_consistency as cli
import replication as rep
from factcheck import claim_guard_v5 as base
from factcheck.ledger_consistency import normalize_ledger, verify_normalization


_spec = importlib.util.spec_from_file_location("ledger_synthetic_fixture", Path(__file__).with_name("test_claim_guard.py"))
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)


class LedgerConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.original, self.candidate, self.ledger = legacy.fixture()

    def run_normalization(self):
        return normalize_ledger(self.original, self.candidate, self.ledger)

    @property
    def claim(self):
        return self.ledger["claims"][1]

    def test_valid_ledger_unchanged_and_inputs_not_mutated(self):
        self.candidate["patches"][1]["depends_on_patch_ids"] = ["P1"]
        before = copy.deepcopy((self.original, self.candidate, self.ledger))
        result = self.run_normalization()
        self.assertEqual(result["ledger"], self.ledger)
        self.assertEqual((self.original, self.candidate, self.ledger), before)
        self.assertFalse(result["audit"]["changed"])
        self.assertEqual(result["audit"]["changes"], [])
        self.assertTrue(result["audit"]["before_validation"]["guard_packet"]["valid"])

    def test_missing_refs_downgrade_axis_and_supported_claim(self):
        self.claim["checks"]["population"]["evidence_ids"] = []
        original = copy.deepcopy(self.ledger)
        result = self.run_normalization()
        normalized = result["ledger"]["claims"][1]
        self.assertEqual(normalized["checks"]["population"]["status"], "unknown")
        self.assertEqual(normalized["status"], "uncertain")
        self.assertEqual(self.ledger, original)
        self.assertEqual(result["audit"]["counts"], {
            "changed_fields": 2, "checks_downgraded": 1, "claims_downgraded": 1, "affected_claims": 1})
        self.assertFalse(result["audit"]["before_validation"]["ledger"]["valid"])
        self.assertTrue(result["audit"]["after_validation"]["guard_packet"]["valid"])

    def test_all_seven_checks_use_their_declared_access_predicate(self):
        for check in base.CHECKS:
            for access in ("partial", "unavailable"):
                with self.subTest(check=check, access=access):
                    o, c, ledger = legacy.fixture()
                    ledger["evidence"][0]["access"] = access
                    claim = ledger["claims"][1]
                    claim["checks"][check] = {"status": "supported", "rationale": "Synthetic", "evidence_ids": ["E1"]}
                    result = normalize_ledger(o, c, ledger)
                    self.assertEqual(result["ledger"]["claims"][1]["checks"][check]["status"], "unknown")
                    self.assertEqual(result["ledger"]["evidence"], ledger["evidence"])

    def test_secondary_full_section_allowed_for_citation_only(self):
        self.ledger["evidence"][0]["primary"] = False
        self.claim["checks"]["citation"] = {"status": "supported", "rationale": "Synthetic", "evidence_ids": ["E1"]}
        result = self.run_normalization()["ledger"]["claims"][1]
        self.assertEqual(result["checks"]["population"]["status"], "unknown")
        self.assertEqual(result["checks"]["citation"]["status"], "supported")

    def test_one_qualifying_ref_among_mixed_evidence_is_sufficient(self):
        self.ledger["evidence"].extend([
            {**self.ledger["evidence"][0], "evidence_id": "E2", "access": "unavailable"},
            {**self.ledger["evidence"][0], "evidence_id": "E3", "primary": False},
        ])
        self.claim["checks"]["population"]["evidence_ids"] = ["E2", "E3", "E1"]
        result = self.run_normalization()
        self.assertFalse(result["audit"]["changed"])

    def test_existing_unknown_top_supported_becomes_uncertain(self):
        self.claim["checks"]["time"] = {"status": "unknown", "rationale": "Not established", "evidence_ids": []}
        result = self.run_normalization()
        self.assertEqual(result["audit"]["counts"]["checks_downgraded"], 0)
        self.assertEqual(result["audit"]["counts"]["claims_downgraded"], 1)
        self.assertEqual(result["ledger"]["claims"][1]["status"], "uncertain")

    def test_supported_mixed_with_contradiction_is_not_repaired(self):
        for status in ("contradicted", "missing_material"):
            with self.subTest(status=status):
                self.claim["checks"]["time"] = {"status": status, "rationale": "Synthetic", "evidence_ids": ["E1"]}
                self.claim["checks"]["population"]["evidence_ids"] = []
                with self.assertRaisesRegex(ValueError, "supported requires at least"):
                    self.run_normalization()

    def test_all_not_applicable_supported_is_not_repaired(self):
        self.claim["checks"]["population"]["status"] = "not_applicable"
        with self.assertRaisesRegex(ValueError, "supported requires at least"):
            self.run_normalization()

    def test_other_check_statuses_and_problem_claim_never_promoted(self):
        claim = self.ledger["claims"][0]
        for check, status in zip(base.CHECKS[1:5], ("unknown", "contradicted", "missing_material", "supported")):
            claim["checks"][check] = {"status": status, "rationale": "Synthetic", "evidence_ids": []}
        result = self.run_normalization()["ledger"]["claims"][0]
        self.assertEqual(result["status"], "problem")
        for check, status in zip(base.CHECKS[1:5], ("unknown", "contradicted", "missing_material", "unknown")):
            self.assertEqual(result["checks"][check]["status"], status)

    def test_nonfactual_supported_check_stays_invalid(self):
        self.claim["status"] = "nonfactual"
        self.claim["checks"]["population"]["evidence_ids"] = []
        with self.assertRaisesRegex(ValueError, "nonfactual requires"):
            self.run_normalization()

    def test_problem_without_contradiction_stays_invalid(self):
        self.claim["status"] = "problem"
        self.claim["checks"]["population"]["evidence_ids"] = []
        with self.assertRaisesRegex(ValueError, "problem requires"):
            self.run_normalization()

    def test_unknown_evidence_id_rejected_not_downgraded(self):
        self.claim["checks"]["population"]["evidence_ids"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.run_normalization()

    def test_missing_or_extra_check_is_not_repaired(self):
        for mode in ("missing", "extra"):
            with self.subTest(mode=mode):
                ledger = copy.deepcopy(self.ledger)
                if mode == "missing":
                    del ledger["claims"][1]["checks"]["time"]
                else:
                    ledger["claims"][1]["checks"]["other"] = copy.deepcopy(ledger["claims"][1]["checks"]["time"])
                with self.assertRaises(ValueError):
                    normalize_ledger(self.original, self.candidate, ledger)

    def test_invalid_evidence_access_type_and_primary_not_repaired(self):
        for field, value in (("access", "snippet"), ("primary", "true"), ("url", "http://example.org/data")):
            with self.subTest(field=field):
                ledger = copy.deepcopy(self.ledger)
                ledger["evidence"][0][field] = value
                with self.assertRaises(ValueError):
                    normalize_ledger(self.original, self.candidate, ledger)

    def test_blank_reason_or_bad_quote_remains_error_even_with_downgrade(self):
        for field, value in (("quote", "Absent text"), ("reason", ""), ("source_scope", "")):
            with self.subTest(field=field):
                ledger = copy.deepcopy(self.ledger)
                ledger["claims"][1][field] = value
                ledger["claims"][1]["checks"]["population"]["evidence_ids"] = []
                with self.assertRaises(ValueError):
                    normalize_ledger(self.original, self.candidate, ledger)

    def test_original_binding_and_block_packet_must_match(self):
        altered = copy.deepcopy(self.original)
        altered["text"] += "Different"
        with self.assertRaisesRegex(ValueError, "original_sha256"):
            normalize_ledger(altered, self.candidate, self.ledger)
        with self.assertRaisesRegex(ValueError, "block packet"):
            normalize_ledger(self.original, self.candidate, self.ledger, block_packet={})

    def test_missing_block_coverage_and_duplicate_claim_rejected(self):
        for mode in ("block", "claim"):
            with self.subTest(mode=mode):
                ledger = copy.deepcopy(self.ledger)
                if mode == "block":
                    ledger["blocks"].pop()
                else:
                    ledger["claims"].append(copy.deepcopy(ledger["claims"][1]))
                with self.assertRaises(ValueError):
                    normalize_ledger(self.original, self.candidate, ledger)

    def test_candidate_mapping_and_dependency_cycle_not_repaired(self):
        for mode in ("mapping", "cycle"):
            with self.subTest(mode=mode):
                candidate = copy.deepcopy(self.candidate)
                if mode == "mapping":
                    candidate["patches"][0]["claim_ids"] = ["C1"]
                else:
                    candidate["patches"][0]["depends_on_patch_ids"] = ["P2"]
                    candidate["patches"][1]["depends_on_patch_ids"] = ["P1"]
                with self.assertRaises(ValueError):
                    normalize_ledger(self.original, candidate, self.ledger)

    def test_only_logged_status_fields_change(self):
        self.claim["checks"]["population"]["evidence_ids"] = []
        result = self.run_normalization()
        replay = copy.deepcopy(self.ledger)
        for change in result["audit"]["changes"]:
            parts = change["json_pointer"].split("/")[1:]
            target = replay
            for part in parts[:-1]:
                target = target[int(part)] if isinstance(target, list) else target[part]
            self.assertEqual(target[parts[-1]], change["before"])
            self.assertIn((change["before"], change["after"]), {("supported", "unknown"), ("supported", "uncertain")})
            target[parts[-1]] = change["after"]
        self.assertEqual(replay, result["ledger"])

    def test_determinism_idempotence_and_exact_replay_verification(self):
        self.claim["checks"]["population"]["evidence_ids"] = []
        result = self.run_normalization()
        self.assertEqual(result, self.run_normalization())
        repeated = normalize_ledger(self.original, self.candidate, result["ledger"])
        self.assertFalse(repeated["audit"]["changed"])
        self.assertEqual(result["ledger"], repeated["ledger"])
        self.assertTrue(verify_normalization(self.original, self.candidate, self.ledger,
                                             result["ledger"], result["audit"])["verified"])

    def test_replay_rejects_changed_inputs_outputs_or_audit(self):
        self.claim["checks"]["population"]["evidence_ids"] = []
        result = self.run_normalization()
        changed_ledger = copy.deepcopy(result["ledger"])
        changed_ledger["claims"][1]["reason"] += " tamper"
        with self.assertRaisesRegex(ValueError, "ledger does not match"):
            verify_normalization(self.original, self.candidate, self.ledger, changed_ledger, result["audit"])
        changed_audit = copy.deepcopy(result["audit"])
        changed_audit["counts"]["checks_downgraded"] = 0
        with self.assertRaisesRegex(ValueError, "audit does not match"):
            verify_normalization(self.original, self.candidate, self.ledger, result["ledger"], changed_audit)
        candidate = copy.deepcopy(self.candidate)
        candidate["patches"][1]["depends_on_patch_ids"] = ["P1"]
        with self.assertRaisesRegex(ValueError, "audit does not match"):
            verify_normalization(self.original, candidate, self.ledger, result["ledger"], result["audit"])

    def test_cli_preserves_bytes_and_immutable_outputs(self):
        self.claim["checks"]["population"]["evidence_ids"] = []
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            inputs = {"original": self.original, "candidate": self.candidate, "ledger": self.ledger}
            args = ["normalize", "--output-dir", str(folder / "out")]
            for key, value in inputs.items():
                path = folder / (key + ".json")
                path.write_bytes(json.dumps(value, ensure_ascii=False, indent=3).encode("utf-8") + b"\r\n")
                args += ["--" + key, str(path)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(args), 0)
                self.assertEqual(cli.main(args), 0)
            for key, target in (("original", "original.frozen.json"), ("candidate", "candidate.raw.json"), ("ledger", "ledger.raw.json")):
                self.assertEqual((folder / (key + ".json")).read_bytes(), (folder / "out" / target).read_bytes())
            freeze = rep._read_json(folder / "out/normalization.freeze.json")
            packet = rep._read_json(folder / "out/guard_packet.json")
            self.assertEqual(packet["input_file_sha256"], freeze["normalized_input_file_sha256"])
            self.assertEqual(freeze["normalized_input_file_sha256"]["ledger"],
                             hashlib.sha256((folder / "out/ledger.normalized.json").read_bytes()).hexdigest())
            (folder / "out/ledger.normalized.json").write_text("{}", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main(args)
            self.assertEqual((folder / "out/ledger.normalized.json").read_text(encoding="utf-8"), "{}")

    def test_replay_rejects_boolean_integer_equality_collision(self):
        self.claim["checks"]["population"]["evidence_ids"] = []
        result = self.run_normalization()
        altered = copy.deepcopy(result["ledger"])
        altered["schema_version"] = True
        with self.assertRaisesRegex(ValueError, "ledger does not match"):
            verify_normalization(self.original, self.candidate, self.ledger, altered, result["audit"])
        altered_audit = copy.deepcopy(result["audit"])
        altered_audit["counts"]["checks_downgraded"] = True
        with self.assertRaisesRegex(ValueError, "audit does not match"):
            verify_normalization(self.original, self.candidate, self.ledger, result["ledger"], altered_audit)

    def test_cli_invalid_structure_writes_no_outputs(self):
        self.claim["checks"]["population"]["evidence_ids"] = ["missing"]
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            args = ["normalize", "--output-dir", str(folder / "out")]
            for key, value in (("original", self.original), ("candidate", self.candidate), ("ledger", self.ledger)):
                path = folder / (key + ".json")
                path.write_bytes(rep._json_bytes(value))
                args += ["--" + key, str(path)]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main(args)
            self.assertFalse((folder / "out").exists())

    def test_prior_electronic_bidding_raw_has_exactly_four_access_downgrades(self):
        folder = Path(__file__).resolve().parents[1] / "pilot/citation_recovery_20261102/new/N2"
        paths = [folder / name for name in ("input/original_report.json", "staging/candidate.raw.json", "staging/ledger.raw.json")]
        if not all(path.is_file() for path in paths):
            self.skipTest("Historical regression inputs not installed")
        raw = [path.read_bytes() for path in paths]
        original, candidate, ledger = [json.loads(data) for data in raw]
        result = normalize_ledger(original, candidate, ledger)
        self.assertEqual({(row["claim_id"], row["check"]) for row in result["audit"]["changes"]},
                         {("C070", "population"), ("C070", "modality"), ("C133", "metric"), ("C133", "population")})
        self.assertEqual(result["audit"]["counts"]["claims_downgraded"], 0)
        self.assertEqual(result["audit"]["output_sha256"]["ledger"],
                         "1736a413a4ff6216f8238904b0fa9ab70ab492b58210984e8d8dca7916da623a")
        self.assertFalse(result["audit"]["before_validation"]["guard_packet"]["valid"])
        self.assertTrue(result["audit"]["after_validation"]["guard_packet"]["valid"])
        self.assertEqual(raw, [path.read_bytes() for path in paths])


if __name__ == "__main__":
    unittest.main()
