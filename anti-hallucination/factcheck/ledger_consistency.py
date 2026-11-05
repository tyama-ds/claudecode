"""Conservative, opt-in normalization of unsupported ledger support labels.

This is an explicit experimental transformation, not a format-only repair or a
new factual assessment. It only lowers support labels. Evidence, quotations,
writer reasons, candidate findings and patches are retained without editing.
The unmodified v5/v6 validator remains the final authority on valid structure.
"""
from __future__ import annotations

import copy
from collections import Counter

import replication as rep
from . import claim_guard_v5 as base


NORMALIZATION_VERSION = "ledger_consistency_v1"
LIMITATION = (
    "Normalization only makes recorded support labels consistent with declared "
    "evidence access. It does not verify retrieval, entailment, factual truth, "
    "necessity or claim completeness. Writer reason/rationale strings remain "
    "unchanged and are not new justifications for the normalized labels."
)


def _validation(operation):
    try:
        result = operation()
    except ValueError as error:
        return {"valid": False, "error": str(error)}
    return {"valid": True, "error": None, "result_sha256": base.sha256_json(result)}


def _evidence_state(refs, evidence):
    return [{"evidence_id": identity, "access": evidence[identity]["access"],
             "primary": evidence[identity]["primary"]} for identity in refs]


def normalize_ledger(original, candidate, ledger, *, block_packet=None):
    """Return ``{ledger, audit}``, rejecting every non-repairable invariant.

    A supported check lacking the exact existing validator's access predicate
    becomes unknown. A supported claim with applicable checks consisting only
    of supported/unknown, at least one unknown, becomes uncertain. No other
    labels change. In particular contradicted/missing_material conflicts and
    all-not-applicable supported claims remain invalid rather than being hidden.

    This function does not mutate any argument. It deterministically binds the
    original, candidate and raw ledger; a fresh call replays the whole operation.
    Success requires the unchanged ledger AND candidate/ledger packet validator.
    """
    expected_packet = base.build_block_packet(original)
    if block_packet is not None and block_packet != expected_packet:
        raise ValueError("block packet does not match original")
    base._version(ledger, "ledger")
    evidence = base._evidence(ledger.get("evidence"), "ledger.evidence")
    # Check all referenced identities and status/schema fields before changing
    # anything. Unknown IDs must not be mistaken for inaccessible evidence.
    claims = rep._rows_by_id(ledger.get("claims"), "claim_id", "ledger.claims")
    for identity, claim in claims.items():
        label = f"ledger.claims.{identity}"
        rep._enum(claim.get("status"), base.CLAIM_STATUSES, f"{label}.status")
        checks = rep._object(claim.get("checks"), f"{label}.checks")
        rep._all_ids(checks, base.CHECKS, f"{label}.checks")
        for check in base.CHECKS:
            item = rep._object(checks[check], f"{label}.checks.{check}")
            rep._enum(item.get("status"), base.CHECK_STATUSES, f"{label}.checks.{check}.status")
            base._refs(item.get("evidence_ids"), evidence, f"{label}.checks.{check}.evidence_ids")

    before = {
        "ledger": _validation(lambda: base.validate_ledger(original, ledger, expected_packet)),
        "guard_packet": _validation(lambda: base.build_guard_packet(original, candidate, ledger)),
    }
    normalized = copy.deepcopy(ledger)
    changes = []
    for index, claim in enumerate(normalized["claims"]):
        for check in base.CHECKS:
            item = claim["checks"][check]
            refs = item["evidence_ids"]
            access = base._accessible(refs, evidence) if check == "citation" else base._primary_accessible(refs, evidence)
            if item["status"] == "supported" and not access:
                item["status"] = "unknown"
                changes.append({
                    "json_pointer": f"/claims/{index}/checks/{check}/status",
                    "claim_id": claim["claim_id"], "field": "check.status", "check": check,
                    "before": "supported", "after": "unknown",
                    "reason_code": "supported_without_required_access",
                    "required_evidence": "full_section" if check == "citation" else "primary_full_section",
                    "evidence_ids": copy.deepcopy(refs), "recorded_evidence_state": _evidence_state(refs, evidence),
                })
        applicable = [claim["checks"][check]["status"] for check in base.CHECKS
                      if claim["checks"][check]["status"] != "not_applicable"]
        if (claim["status"] == "supported" and "unknown" in applicable
                and all(status in {"supported", "unknown"} for status in applicable)):
            claim["status"] = "uncertain"
            changes.append({
                "json_pointer": f"/claims/{index}/status", "claim_id": claim["claim_id"],
                "field": "claim.status", "check": None, "before": "supported", "after": "uncertain",
                "reason_code": "supported_claim_contains_unknown_check",
                "unknown_checks": [check for check in base.CHECKS if claim["checks"][check]["status"] == "unknown"],
            })

    # Do not return an allegedly repaired record if any other invariant fails.
    summary = base.validate_ledger(original, normalized, expected_packet)
    packet = base.build_guard_packet(original, candidate, normalized)
    counts = Counter(change["field"] for change in changes)
    audit = {
        "schema_version": 1, "normalization_version": NORMALIZATION_VERSION,
        "transformation_kind": "conservative_support_label_downgrade_not_format_only",
        "input_sha256": {"original": base.sha256_json(original), "candidate": base.sha256_json(candidate),
                         "ledger": base.sha256_json(ledger)},
        "output_sha256": {"original": base.sha256_json(original), "candidate": base.sha256_json(candidate),
                          "ledger": base.sha256_json(normalized)},
        "before_validation": before,
        "after_validation": {
            "ledger": {"valid": True, "error": None, "result_sha256": base.sha256_json(summary)},
            "guard_packet": {"valid": True, "error": None, "result_sha256": base.sha256_json(packet)},
        },
        "changed": bool(changes), "changes": changes,
        "counts": {"changed_fields": len(changes), "checks_downgraded": counts["check.status"],
                   "claims_downgraded": counts["claim.status"],
                   "affected_claims": len({change["claim_id"] for change in changes})},
        "unchanged_fields": ["original", "candidate", "claim_ids", "evidence", "evidence_ids",
                             "quotes", "blocks", "reason", "rationale", "scopes"],
        "ledger_summary": summary, "limitation": LIMITATION,
    }
    return {"ledger": normalized, "audit": audit}


def verify_normalization(original, candidate, raw_ledger, normalized_ledger, audit, *, block_packet=None):
    """Reject altered outputs or audit trails by exact deterministic replay."""
    expected = normalize_ledger(original, candidate, raw_ledger, block_packet=block_packet)
    if base.sha256_json(expected["ledger"]) != base.sha256_json(normalized_ledger):
        raise ValueError("normalized ledger does not match deterministic replay")
    if base.sha256_json(expected["audit"]) != base.sha256_json(audit):
        raise ValueError("normalization audit does not match deterministic replay")
    return {"verified": True, "normalization_version": NORMALIZATION_VERSION,
            "audit_sha256": base.sha256_json(audit), "normalized_ledger_sha256": base.sha256_json(normalized_ledger)}


def schema_contract():
    return {
        "normalization_version": NORMALIZATION_VERSION,
        "transformation_kind": "conservative_support_label_downgrade_not_format_only",
        "rules": [
            "Only supported check statuses without the existing validator's required access become unknown.",
            "Citation permits any declared full_section evidence; other checks require declared primary full_section evidence.",
            "One qualifying referenced evidence item is sufficient, matching the unmodified validator.",
            "A supported claim becomes uncertain only if applicable checks contain unknown and otherwise only supported/unknown.",
            "No positive label, evidence, claim text, candidate finding, patch or dependency is created or changed.",
            "All other invalid ledger, quotation, mapping or candidate invariants fail; no output is returned as normalized.",
            "Preserve raw bytes and the complete audit; bind original/candidate/raw and normalized ledger hashes.",
            "Replay this function to verify artifacts. Raw current-arm failure remains failure in an experiment.",
        ],
        "limitation": LIMITATION,
    }
