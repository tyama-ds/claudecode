"""Numeric scoring with exact decimals and separately reported citation checks."""

from __future__ import annotations

from decimal import Decimal, DecimalException, localcontext

from .dataset import decimal_value


EVIDENCE_LIMITATION = "Checks source ID and exact quote presence only; semantic entailment is not evaluated."


def _normal_value(item: dict, candidate: dict) -> Decimal | None:
    if not isinstance(candidate, dict):
        raise ValueError("answer/candidate must be an object")
    value = decimal_value(candidate.get("value"), "answer.value")
    unit = candidate.get("unit")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("answer.unit must be a non-empty string")
    if unit == item["unit"]:
        multiplier = Decimal(1)
    elif unit in item.get("accepted_units", {}):
        multiplier = decimal_value(item["accepted_units"][unit], "unit multiplier")
    else:
        return None
    with localcontext() as context:
        context.prec = max(28, len(value.as_tuple().digits) + len(multiplier.as_tuple().digits))
        return value * multiplier


def _within_tolerance(item: dict, value: Decimal) -> bool:
    expected = decimal_value(item["value"], "gold.value")
    tolerance = decimal_value(item.get("tolerance", "0"), "gold.tolerance")
    if tolerance < 0:
        raise ValueError("gold.tolerance cannot be negative")
    # Choose enough precision for subtraction across widely different scales.
    # The input size is bounded by decimal_value().
    values = (expected, value, tolerance)
    lowest_exponent = min(number.as_tuple().exponent for number in values)
    highest_position = max(number.adjusted() for number in values)
    with localcontext() as context:
        context.prec = max(28, highest_position - lowest_exponent + 3)
        return abs(value - expected) <= tolerance


def candidate_truth(item: dict, candidate: dict) -> str:
    """Return a numeric reference label; malformed candidates raise ValueError."""
    try:
        value = _normal_value(item, candidate)
        return "supported" if value is not None and _within_tolerance(item, value) else "refuted"
    except DecimalException as exc:
        raise ValueError("numeric candidate cannot be evaluated") from exc


def _evidence(item: dict, answer: dict, report: dict) -> tuple[bool, str]:
    evidence = answer.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return False, f"No evidence supplied. {EVIDENCE_LIMITATION}"
    source_map = {source["id"]: source for source in report.get("sources", [])}
    permitted = set(item.get("source_ids", []))
    for entry in evidence:
        if not isinstance(entry, dict):
            return False, f"Evidence entry is not an object. {EVIDENCE_LIMITATION}"
        source_id, quote = entry.get("source_id"), entry.get("quote")
        if not isinstance(source_id, str) or source_id not in permitted or source_id not in source_map:
            return False, f"Unknown or unapproved evidence source. {EVIDENCE_LIMITATION}"
        if not isinstance(quote, str) or not quote.strip() or quote not in source_map[source_id]["text"]:
            return False, f"An exact non-empty quote was not found in its source. {EVIDENCE_LIMITATION}"
    return True, f"All cited quotes were found in approved sources. {EVIDENCE_LIMITATION}"


def score_answer(item: dict, answer: dict, report: dict) -> dict:
    """Score answer content, keeping abstention and execution/schema errors apart."""
    result = {"outcome": "error", "normalized_value": None, "evidence_valid": False, "evidence_note": EVIDENCE_LIMITATION}
    if not isinstance(answer, dict):
        result["evidence_note"] = f"Answer is not an object. {EVIDENCE_LIMITATION}"
        return result
    status = answer.get("status")
    if status == "error":
        result["evidence_note"] = f"Model execution or response-schema error. {EVIDENCE_LIMITATION}"
        return result
    if status not in ("answered", "abstain"):
        result["evidence_note"] = f"Unknown answer status. {EVIDENCE_LIMITATION}"
        return result
    result["evidence_valid"], result["evidence_note"] = _evidence(item, answer, report)
    if status == "abstain":
        if answer.get("value") is not None:
            result["evidence_note"] = f"Abstention must not include a numeric answer. {EVIDENCE_LIMITATION}"
            result["evidence_valid"] = False
        else:
            result["outcome"] = "abstain"
        return result
    try:
        value = _normal_value(item, answer)
        if value is None:
            result["outcome"] = "incorrect"
        else:
            result["normalized_value"] = str(value)
            result["outcome"] = "correct" if _within_tolerance(item, value) else "incorrect"
    except (ValueError, DecimalException) as exc:
        result["evidence_note"] = f"Invalid numeric response: {exc}. {EVIDENCE_LIMITATION}"
        result["evidence_valid"] = False
    return result
