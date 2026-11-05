"""Compare checks of an unchanged report with an independent claim inventory.

This module does not create reference labels, align claims, or validate evidence.
Those steps must happen independently before calling :func:`evaluate_claims`.
``unresolved`` is an unknown reference truth, not a synonym for a correct
``insufficient`` verdict. No aggregate "accuracy" is inferred for unknown truth.
"""

from __future__ import annotations


TRUTHS = ("correct", "incorrect", "unresolved")
VERDICTS = ("supported", "refuted", "insufficient")
STATES = VERDICTS + ("not_extracted", "unjudged")


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _records(value: object, name: str, label: str, allowed: tuple[str, ...]) -> dict:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    result = {}
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"{name}[{index}] must be an object")
        claim_id = _id(row.get("claim_id"), f"{name}[{index}].claim_id")
        if claim_id in result:
            raise ValueError(f"duplicate claim_id in {name}: {claim_id}")
        if row.get(label) not in allowed:
            raise ValueError(f"{name}[{index}].{label} must be one of {allowed}")
        result[claim_id] = row[label]
    return result


def evaluate_claims(gold: list[dict], checks: list[dict], *, extracted_claim_ids: list[str]) -> dict:
    """Evaluate one checking condition; inputs are not modified.

    ``gold`` is the *complete* independently established inventory of in-scope
    original claims: ``[{"claim_id": "c1", "truth": "correct"}, ...]``.
    Truth is one of ``correct``, ``incorrect``, or ``unresolved``.

    ``checks`` contains completed decisions:
    ``[{"claim_id": "c1", "verdict": "supported"}, ...]``.
    Verdict is one of ``supported``, ``refuted``, or ``insufficient``.
    Metadata may be included but is not scored or copied into the result.

    ``extracted_claim_ids`` explicitly identifies claims found by the checking
    workflow. An extracted claim with no completed decision is ``unjudged``;
    another missing claim is ``not_extracted``. Neither disappears from the
    denominator. Unknown/duplicate IDs and decisions for unextracted claims
    raise ValueError. All three inputs therefore use the same pre-aligned IDs.

    Confirmed errors partition into detected (refuted), missed (supported,
    not_extracted, or unjudged), and abstained (insufficient). The separate
    not_detected count includes missed and abstained errors. A correct claim
    marked insufficient is reported separately from a false refutation: truth
    alone does not establish whether available evidence warrants abstention.

    The hypothetical support-only gate withholds *every* other state, including
    missing decisions. This is not automatic correction or proof that the
    original report was edited. Its error reduction is reported together with
    correct retention, coverage, and unresolved claims that pass the gate.
    """
    reference = _records(gold, "gold", "truth", TRUTHS)
    decisions = _records(checks, "checks", "verdict", VERDICTS)
    if not isinstance(extracted_claim_ids, list):
        raise ValueError("extracted_claim_ids must be an array")
    extracted = set()
    for index, value in enumerate(extracted_claim_ids):
        claim_id = _id(value, f"extracted_claim_ids[{index}]")
        if claim_id in extracted:
            raise ValueError(f"duplicate extracted claim_id: {claim_id}")
        if claim_id not in reference:
            raise ValueError(f"unknown extracted claim_id: {claim_id}")
        extracted.add(claim_id)
    for claim_id in decisions:
        if claim_id not in reference:
            raise ValueError(f"unknown checked claim_id: {claim_id}")
        if claim_id not in extracted:
            raise ValueError(f"checked claim was not extracted: {claim_id}")

    matrix = {truth: {state: 0 for state in STATES} for truth in TRUTHS}
    rows = []
    for claim_id, truth in reference.items():
        state = decisions.get(claim_id, "unjudged" if claim_id in extracted else "not_extracted")
        matrix[truth][state] += 1
        rows.append({
            "claim_id": claim_id,
            "truth": truth,
            "state": state,
            "support_only_provided": state == "supported",
        })

    total = len(reference)
    truth_counts = {truth: sum(matrix[truth].values()) for truth in TRUTHS}
    states = {state: sum(matrix[truth][state] for truth in TRUTHS) for state in STATES}
    errors = matrix["incorrect"]
    correct = matrix["correct"]
    unresolved = matrix["unresolved"]
    missed_errors = errors["supported"] + errors["not_extracted"] + errors["unjudged"]
    incorrect_n = truth_counts["incorrect"]
    correct_n = truth_counts["correct"]
    provided_counts = {truth: matrix[truth]["supported"] for truth in TRUTHS}
    provided = sum(provided_counts.values())

    return {
        "schema_version": 1,
        "counts": {
            "claims": total,
            "truth": truth_counts,
            "extracted": len(extracted),
            "judged": len(decisions),
            "states": states,
        },
        "coverage": {
            "extraction_recall": _rate(len(extracted), total),
            "judged_fraction": _rate(len(decisions), total),
            "asserted_fraction": _rate(states["supported"] + states["refuted"], total),
        },
        "error_detection": {
            "reference_errors": incorrect_n,
            "detected": errors["refuted"],
            "missed": missed_errors,
            "missed_supported": errors["supported"],
            "missed_not_extracted": errors["not_extracted"],
            "missed_unjudged": errors["unjudged"],
            "abstained": errors["insufficient"],
            "not_detected_including_abstention": incorrect_n - errors["refuted"],
            "detection_rate": _rate(errors["refuted"], incorrect_n),
            "miss_rate": _rate(missed_errors, incorrect_n),
            "abstention_rate": _rate(errors["insufficient"], incorrect_n),
        },
        "correct_claims": {
            "reference_correct": correct_n,
            "supported": correct["supported"],
            "false_refutations": correct["refuted"],
            "insufficient": correct["insufficient"],
            "not_extracted": correct["not_extracted"],
            "unjudged": correct["unjudged"],
            "false_refutation_rate": _rate(correct["refuted"], correct_n),
            "insufficient_rate": _rate(correct["insufficient"], correct_n),
        },
        "unresolved_claims": dict(unresolved),
        "no_check": {
            "provided": total,
            "provided_by_truth": dict(truth_counts),
            "provided_by_truth_per_original_claim": {
                truth: _rate(truth_counts[truth], total) for truth in TRUTHS
            },
        },
        "support_only_gate": {
            "policy": "Provide supported claims only; withhold all other states. No rewriting.",
            "provided": provided,
            "withheld": total - provided,
            "provided_fraction": _rate(provided, total),
            "provided_by_truth": provided_counts,
            "provided_by_truth_per_original_claim": {
                truth: _rate(provided_counts[truth], total) for truth in TRUTHS
            },
            "correct_retention_rate": _rate(provided_counts["correct"], correct_n),
            "known_error_reduction_rate": _rate(incorrect_n - provided_counts["incorrect"], incorrect_n),
            "known_error_fraction_among_provided": _rate(provided_counts["incorrect"], provided),
            "unresolved_fraction_among_provided": _rate(provided_counts["unresolved"], provided),
        },
        "matrix": matrix,
        "claims": rows,
        "limitations": [
            "Reference truth and claim alignment are supplied, not independently verified by this scorer.",
            "Correctness of abstention or citation support requires separate evidence assessment.",
            "Unresolved references are neither correct nor incorrect; the known-error fraction is not a complete error estimate.",
            "Gate withholding is not correction; review correct retention and coverage alongside error reduction.",
        ],
    }
