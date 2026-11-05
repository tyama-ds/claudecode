"""Opt-in v7: isolate localized uncertainty, then independently review the remainder.

This module preserves the v6 eligibility and mechanical screening. An unknown
review remains unknown. It may authorize one removal-and-review round only when
the reviewer explicitly accounts for every blocking uncertainty and has actually
reviewed all other changes and their interactions. The original and replacements
are never rewritten. This validates attestations, not their factual truth or the
reviewer's independence.
"""
from __future__ import annotations

import copy

import replication as rep
from . import claim_guard_v6 as v6

CHECKS = v6.CHECKS
GATE_VERSION = "v7"
MAX_RECOVERY_ROUNDS = 1
LIMITATION = v6.LIMITATION + " Localized uncertainty is a hold, not a confirmed error; its scope and remainder coverage require independent semantic judgment."
sha256_json = v6.sha256_json
build_block_packet = v6.build_block_packet
build_guard_packet = v6.build_guard_packet
validate_ledger = v6.validate_ledger
citation_boundary_diagnostics = v6.citation_boundary_diagnostics


def _selection(result, guard, *, round_number=0, previous_review=None):
    v6._selection(result, guard, round_number=round_number, previous_review=previous_review)
    selection = result["selection"]
    selection["gate_version"] = GATE_VERSION
    selection["selection_sha256"] = sha256_json({key: value for key, value in selection.items()
                                                if key != "selection_sha256"})


def _summary(result, candidate, *, required=False, passed=False, blockers=()):
    v6._summary(result, candidate, required=required, passed=passed, blockers=blockers)
    kind = result.get("recovery_kind", "none")
    result["summary"].update(gate_version=GATE_VERSION, limitation=LIMITATION,
        recovery_kind=kind,
        localized_failure_removal_rounds=int(kind == "localized_failure"),
        localized_uncertainty_hold_rounds=int(kind == "localized_uncertainty"),
        uncertainty_held_patch_ids=list(result.get("uncertainty_held_patch_ids", [])),
        uncertainty_hold_is_error_finding=False)


def preview_repair(original, candidate, ledger, guard, *, guard_packet=None):
    result = copy.deepcopy(v6.preview_repair(original, candidate, ledger, guard, guard_packet=guard_packet))
    result["gate_version"] = GATE_VERSION
    result["recovery_kind"] = "none"
    result["uncertainty_held_patch_ids"] = []
    _selection(result, guard)
    _summary(result, candidate)
    return result


def _validate_review(review, selection):
    rep._object(review, "final_review")
    # v6 validates all existing fields and exact bindings. Its strict schema
    # receives a detached view so no raw review or caller-owned object is altered.
    legacy = {key: value for key, value in review.items() if key != "uncertainty_localization"}
    v6._validate_review(legacy, selection)
    label = "final_review.uncertainty_localization"
    local = rep._object(review.get("uncertainty_localization"), label)
    rep._all_ids(local, {"status", "patch_ids", "patch_reasons", "all_unresolved_concerns_localized",
        "remainder_coverage_complete", "no_preexisting_or_nonlocal_concerns", "reason"}, label)
    rep._enum(local["status"], {"none", "definite", "ambiguous", "unknown"}, label + ".status")
    v6.v5._refs(local["patch_ids"], selection["accepted_patch_ids"], label + ".patch_ids")
    reasons = rep._object(local["patch_reasons"], label + ".patch_reasons")
    rep._all_ids(reasons, local["patch_ids"], label + ".patch_reasons")
    for identity, reason in reasons.items():
        v6.v5._reason(reason, label + ".patch_reasons." + identity)
    flags = ("all_unresolved_concerns_localized", "remainder_coverage_complete", "no_preexisting_or_nonlocal_concerns")
    for field in flags:
        rep._boolean(local[field], label + "." + field)
    v6.v5._reason(local["reason"], label + ".reason")
    if local["status"] == "none" and (local["patch_ids"] or any(local[field] for field in flags)):
        raise ValueError("none uncertainty localization requires empty patch_ids/reasons and false attestations")
    if review["status"] == "pass" and local["status"] != "none":
        raise ValueError("pass requires none uncertainty localization")


def _recovery_kind(review):
    uncertain = review["uncertainty_localization"]
    if v6._recoverable(review) and uncertain["status"] == "none":
        return "localized_failure"
    if (review["status"] == "unknown" and review["patch_localization"]["status"] == "none"
            and uncertain["status"] == "definite" and bool(uncertain["patch_ids"])
            and uncertain["all_unresolved_concerns_localized"]
            and uncertain["remainder_coverage_complete"]
            and uncertain["no_preexisting_or_nonlocal_concerns"]):
        return "localized_uncertainty"
    return "none"


def _recoverable(review):
    """Eligibility after structural validation; does not change review status."""
    return _recovery_kind(review) != "none"


def _veto(result, original, candidate, blockers):
    v6._reject(result, v6._accepted(result), "final_veto", "; ".join(blockers))
    result["effective_repair"], result["final_applied"] = v6._effective(original, candidate, set())
    result["recovery_eligible"] = False
    result["recovery_status"] = "veto"
    _summary(result, candidate, required=True, blockers=blockers)
    return result


def _prepare_next(result, original, candidate, guard, review):
    kind = _recovery_kind(review)
    if kind == "none":
        raise ValueError("recovery requested without complete localization attestations")
    result["recovery_kind"] = kind
    if kind == "localized_uncertainty":
        local = review["uncertainty_localization"]
        result["uncertainty_held_patch_ids"] = sorted(local["patch_ids"])
        for identity in local["patch_ids"]:
            v6._reject(result, [identity], "localized_uncertainty",
                "held as unconfirmed, not a confirmed error: " + local["patch_reasons"][identity],
                details={"review_sha256": sha256_json(review), "review_status": review["status"],
                    "patch_reason": local["patch_reasons"][identity], "localization": local})
    else:
        local = review["patch_localization"]
        v6._reject(result, local["patch_ids"], "localized_final_review",
            "independent review localized all failure causes to this patch",
            details={"review_sha256": sha256_json(review), "localization": local})
    v6._screen(result, original, candidate, guard)
    _selection(result, guard, round_number=1, previous_review=sha256_json(review))
    if result["mechanical_veto"]:
        return _veto(result, original, candidate, ["mechanical attribution was ambiguous after localized removal"])
    result["recovery_eligible"] = bool(v6._accepted(result))
    result["recovery_status"] = "awaiting_review" if result["recovery_eligible"] else "no_remainder"
    if not result["recovery_eligible"]:
        result["recovery_terminal_reason"] = "no selected patches remain after localized removal"
    _summary(result, candidate)
    return result


def prepare_recovery(original, candidate, ledger, guard, *, final_review, guard_packet=None):
    """Prepare a provisional remainder after a localized failure or uncertainty.

    Uncertainty may have coverage_complete=false. Recovery instead requires an
    explicit attestation that all remaining changes AND their interactions have
    actually been examined. Listing the other IDs is not adequate coverage.
    This helper never authorizes application; a fresh complete pass is required.
    """
    result = preview_repair(original, candidate, ledger, guard, guard_packet=guard_packet)
    _validate_review(final_review, result["selection"])
    v6._record_review(result, final_review)
    if not _recoverable(final_review) or result["mechanical_veto"]:
        return _veto(result, original, candidate, ["initial review does not permit completely localized failure or uncertainty recovery"])
    return _prepare_next(result, original, candidate, guard, final_review)


def gate_repair(original, candidate, ledger, guard, *, guard_packet=None):
    """Apply only with a complete pass, after at most one total recovery round."""
    result = preview_repair(original, candidate, ledger, guard, guard_packet=guard_packet)
    if "final_review" in guard and "final_reviews" in guard:
        raise ValueError("provide final_review or final_reviews, not both")
    reviews = guard.get("final_reviews", [guard.get("final_review")])
    rep._array(reviews, "guard.final_reviews")
    if not 1 <= len(reviews) <= MAX_RECOVERY_ROUNDS + 1:
        raise ValueError("guard.final_reviews requires one or two reviews")
    first = reviews[0]
    _validate_review(first, result["selection"])
    v6._record_review(result, first)
    if first["status"] == "pass" and first["coverage_complete"] and not result["mechanical_veto"]:
        if len(reviews) != 1:
            raise ValueError("extra review after terminal pass")
        result["recovery_status"] = "not_needed"
        _summary(result, candidate, required=True, passed=True)
        return result
    if not _recoverable(first) or result["mechanical_veto"]:
        if len(reviews) != 1:
            raise ValueError("extra review after nonrecoverable failure or uncertainty")
        return _veto(result, original, candidate, ["final review failed, unknown, incomplete, or not completely localized"])
    _prepare_next(result, original, candidate, guard, first)
    if not result["recovery_eligible"]:
        if len(reviews) != 1:
            raise ValueError("extra review after terminal recovery outcome")
        return result
    if len(reviews) != 2:
        return _veto(result, original, candidate, ["fresh full final re-review is required after localized removal"])
    second = reviews[1]
    _validate_review(second, result["selection"])
    v6._record_review(result, second)
    if second["status"] != "pass" or not second["coverage_complete"] or result["mechanical_veto"]:
        return _veto(result, original, candidate, ["final re-review did not provide a complete pass; recovery limit reached"])
    result["recovery_eligible"] = False
    result["recovery_status"] = "passed"
    _summary(result, candidate, required=True, passed=True)
    return result


def schema_contract():
    contract = copy.deepcopy(v6.schema_contract())
    contract.update(gate_version=GATE_VERSION, limitation=LIMITATION)
    contract["guard"]["final_review"]["uncertainty_localization"] = {
        "status": "none|definite|ambiguous|unknown; definite describes location certainty, not factual correctness",
        "patch_ids": "unique selected IDs containing all approval-blocking uncertainty; nonempty to recover",
        "patch_reasons": "object mapping exactly every patch_id to its nonempty uncertainty reason; {} when no IDs",
        "all_unresolved_concerns_localized": "boolean: ALL approval-blocking unresolved concerns are confined to these selected changes",
        "remainder_coverage_complete": "boolean: reviewer actually examined ALL remaining changes AND their cross-document interactions, including preservation and source support; unmentioned IDs do not constitute coverage",
        "no_preexisting_or_nonlocal_concerns": "boolean: no approval-blocking uncertainty comes from unchanged original text, nonlocal context, or an unidentified cause; unchanged known original problems are not automatically new repair failures",
        "reason": "nonempty scope and coverage explanation; none requires empty IDs/reasons and all three attestations false"}
    contract["guard"]["final_reviews"] = "Alternative to final_review: one or two strict v7 reviews. One total removal/review round may follow either localized failure or localized uncertainty; never one of each. Never provide both fields."
    # Replace v6 recovery-specific wording, retaining its mechanical contract.
    rules = contract["gate_rules"]
    rules = [rule for rule in rules if not rule.startswith(("A complete failed full-report review", "A blinded reviewer"))]
    rules.extend([
        "Retain complete, definite v6 failure recovery only when uncertainty_localization is none. Unknown stays unknown and can recover only with none failure localization, definite nonempty uncertainty IDs, exact per-ID reasons, and all three uncertainty attestations true. Absent, ambiguous, mixed failure/uncertainty, incomplete remainder coverage, or nonlocal uncertainty vetoes the whole selection.",
        "Holding uncertainty is not a confirmed error finding and does not correct or validate that patch. Keep original raw status/reasons. Exclude only identified patches plus transitive dependents and any newly mechanically implicated patches. Never rewrite or re-enable a patch.",
        "Failure and uncertainty recovery share a single maximum round. A fresh independent complete pass of the exact reduced full report is mandatory. Unknown, fail, or incomplete second review causes a terminal veto; no third review or candidate regeneration is allowed.",
        "A blinded reviewer returns status, coverage_complete, reason, patch_localization and uncertainty_localization; the operator attaches exact selection bindings. A v6 comparator may strip only uncertainty_localization for schema compatibility, preserving all original status/coverage/failure judgments. Independence and truth of attestations cannot be established by this code."])
    contract["gate_rules"] = rules
    contract["report_interface"] += " v7 adds recovery_kind, uncertainty_held_patch_ids and separate failure/uncertainty removal-round counts; unknown review status and per-patch reasons are retained in review_history and exclusion_trail."
    return contract
