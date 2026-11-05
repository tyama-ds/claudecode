"""Dataset validation and the deliberately small public model-input boundary."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from decimal import Decimal, InvalidOperation
import re
from typing import Any


_PLACEHOLDER = re.compile(r"\{\{([^{}]+)\}\}")
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_MASK_NUMBER = re.compile(r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)\Z")
_SOURCE_FIELDS = ("id", "title", "text", "url", "published_at")


def decimal_value(value: Any, label: str = "value") -> Decimal:
    """Read a finite decimal; units and grouping separators belong elsewhere."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{label}: a finite decimal number is required")
    rendered = str(value).strip()
    if not _NUMBER.fullmatch(rendered):
        raise ValueError(f"{label}: a finite decimal number is required")
    try:
        result = Decimal(rendered)
    except InvalidOperation as exc:
        raise ValueError(f"{label}: invalid decimal number") from exc
    if not result.is_finite():
        raise ValueError(f"{label}: a finite decimal number is required")
    # Extremely large exponents are not useful report figures and can exhaust
    # memory during exact subtraction. This is an input bound, not a tolerance.
    if len(result.as_tuple().digits) > 1000 or abs(result.as_tuple().exponent) > 1000:
        raise ValueError(f"{label}: at most 1000 digits and exponent magnitude 1000 are supported")
    return result


def _object(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label}: an object is required")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: a non-empty string is required")
    return value


def _identifier(value: Any, label: str) -> str:
    result = _string(value, label)
    if result != result.strip() or any(char.isspace() or char in "{}" for char in result):
        raise ValueError(f"{label}: IDs cannot contain whitespace or braces")
    return result


def _date(value: Any, label: str) -> str:
    result = _string(value, label)
    try:
        if date.fromisoformat(result).isoformat() != result:
            raise ValueError("non-canonical date")
    except ValueError as exc:
        raise ValueError(f"{label}: use YYYY-MM-DD") from exc
    return result


def _list(value: Any, label: str, *, nonempty: bool = False) -> list:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{label}: {'a non-empty' if nonempty else 'an'} array is required")
    return value


def _placeholders(text: str, label: str) -> set[str]:
    matches = {match.group(0) for match in _PLACEHOLDER.finditer(text)}
    remainder = _PLACEHOLDER.sub("", text)
    if "{{" in remainder or "}}" in remainder:
        raise ValueError(f"{label}: malformed placeholder")
    return matches


def _apply_masks(report: dict, items: list[dict]) -> None:
    """Mask literal numeric tokens, refusing ambiguity rather than guessing."""
    original = report["text"]
    replacements: list[tuple[re.Pattern, str]] = []
    seen_masks: set[str] = set()
    instruction = "Use explicit placeholders in the report text to resolve this ambiguity."
    for item in items:
        if "mask_text" not in item:
            continue
        mask = item["mask_text"]
        if not isinstance(mask, str) or not _MASK_NUMBER.fullmatch(mask):
            raise ValueError(f"item {item['id']}.mask_text: use only a numeric literal with optional sign, decimal, and three-digit comma groups. {instruction}")
        decimal_value(mask.replace(",", ""), f"item {item['id']}.mask_text")
        if mask in seen_masks:
            raise ValueError(f"report {report['id']}: multiple items use mask_text {mask!r}. {instruction}")
        seen_masks.add(mask)
        # Exclude parts of larger numbers, identifiers, scientific notation,
        # ranges and existing placeholders. Unicode prose may border a number.
        pattern = re.compile(r"(?<![A-Za-z0-9_.,+\-{}])" + re.escape(mask) + r"(?![A-Za-z0-9_.,+\-{}])")
        matches = bool(pattern.search(original))
        if item["placeholder"] in original:
            if matches:
                raise ValueError(f"report {report['id']}: placeholder {item['placeholder']} and unmasked {mask!r} both occur. {instruction}")
        elif not matches:
            raise ValueError(f"report {report['id']}: mask_text {mask!r} was not found as a complete numeric token. {instruction}")
        else:
            replacements.append((pattern, item["placeholder"]))
    for pattern, placeholder in replacements:
        report["text"] = pattern.sub(lambda match: placeholder, report["text"])


def validate_dataset(reports: dict, gold: dict) -> dict:
    """Validate and copy inputs; keep local metadata out of item_context()."""
    report_rows = deepcopy(_list(_object(reports, "reports").get("reports"), "reports.reports", nonempty=True))
    item_rows = deepcopy(_list(_object(gold, "gold").get("items"), "gold.items", nonempty=True))
    report_map: dict[str, dict] = {}
    source_ids_by_report: dict[str, set[str]] = {}
    declared_by_report: dict[str, set[str]] = {}

    for index, report in enumerate(report_rows):
        label = f"reports[{index}]"
        _object(report, label)
        report_id = _identifier(report.get("id"), f"{label}.id")
        if report_id in report_map:
            raise ValueError(f"duplicate report ID: {report_id}")
        _string(report.get("title"), f"{label}.title")
        _string(report.get("text"), f"{label}.text")
        _date(report.get("as_of"), f"{label}.as_of")
        report_map[report_id] = report
        declared_by_report[report_id] = set()
        source_ids: set[str] = set()
        for source_index, source in enumerate(_list(report.get("sources"), f"{label}.sources", nonempty=True)):
            source_label = f"{label}.sources[{source_index}]"
            _object(source, source_label)
            source_id = _identifier(source.get("id"), f"{source_label}.id")
            if source_id in source_ids:
                raise ValueError(f"duplicate source ID in {report_id}: {source_id}")
            source_ids.add(source_id)
            _string(source.get("title"), f"{source_label}.title")
            _string(source.get("text"), f"{source_label}.text")
            if "url" in source:
                _string(source["url"], f"{source_label}.url")
            if "published_at" in source:
                published_at = _date(source["published_at"], f"{source_label}.published_at")
                if published_at > report["as_of"]:
                    raise ValueError(f"{source_label}.published_at: source was published after report.as_of ({report['as_of']})")
        source_ids_by_report[report_id] = source_ids

    seen_items: set[str] = set()
    for index, item in enumerate(item_rows):
        label = f"items[{index}]"
        _object(item, label)
        item_id = _identifier(item.get("id"), f"{label}.id")
        if item_id in seen_items:
            raise ValueError(f"duplicate item ID: {item_id}")
        seen_items.add(item_id)
        report_id = _identifier(item.get("report_id"), f"{label}.report_id")
        if report_id not in report_map:
            raise ValueError(f"{label}: unknown report_id {report_id}")
        placeholder = _string(item.get("placeholder"), f"{label}.placeholder")
        if not _PLACEHOLDER.fullmatch(placeholder):
            raise ValueError(f"{label}.placeholder: use a complete {{{{placeholder}}}}")
        if placeholder in declared_by_report[report_id]:
            raise ValueError(f"duplicate placeholder in {report_id}: {placeholder}")
        declared_by_report[report_id].add(placeholder)
        statement = _string(item.get("statement"), f"{label}.statement")
        if statement.count("{{value}}") != 1 or _placeholders(statement, f"{label}.statement") != {"{{value}}"}:
            raise ValueError(f"{label}.statement: include {{{{value}}}} exactly once and no other placeholders")
        item["value"] = str(decimal_value(item.get("value"), f"{label}.value"))
        unit = _string(item.get("unit"), f"{label}.unit")
        tolerance = decimal_value(item.get("tolerance", "0"), f"{label}.tolerance")
        if tolerance < 0:
            raise ValueError(f"{label}.tolerance: cannot be negative")
        item["tolerance"] = str(tolerance)
        alternatives = _object(item.get("accepted_units", {}), f"{label}.accepted_units")
        normalized_units: dict[str, str] = {}
        for alternative, multiplier in alternatives.items():
            _string(alternative, f"{label}.accepted_units key")
            parsed = decimal_value(multiplier, f"{label}.accepted_units[{alternative}]")
            if parsed <= 0 or (alternative == unit and parsed != 1):
                raise ValueError(f"{label}.accepted_units: multipliers must be positive; the gold unit must have multiplier 1")
            normalized_units[alternative] = str(parsed)
        item["accepted_units"] = normalized_units
        sources = _list(item.get("source_ids"), f"{label}.source_ids", nonempty=True)
        seen_sources: set[str] = set()
        for source_id in sources:
            _identifier(source_id, f"{label}.source_ids entry")
            if source_id not in source_ids_by_report[report_id]:
                raise ValueError(f"{label}: source {source_id} is not in report {report_id}")
            if source_id in seen_sources:
                raise ValueError(f"{label}: duplicate source_id {source_id}")
            seen_sources.add(source_id)
        candidates = _list(item.get("candidates", []), f"{label}.candidates")
        for candidate_index, candidate in enumerate(candidates):
            candidate_label = f"{label}.candidates[{candidate_index}]"
            _object(candidate, candidate_label)
            candidate["value"] = str(decimal_value(candidate.get("value"), f"{candidate_label}.value"))
            _string(candidate.get("unit"), f"{candidate_label}.unit")
            candidate["origin"] = _string(candidate.get("origin", "historical"), f"{candidate_label}.origin")
            historical_verdict = candidate.get("historical_verdict")
            if historical_verdict is not None and historical_verdict not in ("supported", "refuted", "insufficient", "error"):
                raise ValueError(f"{candidate_label}.historical_verdict: invalid verdict")
        item["candidates"] = candidates

    for report_id, report in report_map.items():
        _apply_masks(report, [item for item in item_rows if item["report_id"] == report_id])
        actual = _placeholders(report["text"], f"report {report_id}.text")
        declared = declared_by_report[report_id]
        if not declared:
            raise ValueError(f"report {report_id}: no numeric items declared")
        if actual != declared:
            undeclared = sorted(actual - declared)
            missing = sorted(declared - actual)
            raise ValueError(f"report {report_id}: undeclared placeholders={undeclared}; missing placeholders={missing}. Add explicit placeholders or an unambiguous mask_text.")
    return {"reports": report_rows, "items": item_rows}


def item_context(item: dict, report: dict) -> dict:
    """Build model-visible context from an allowlist, never from a gold copy.

    Source text intentionally contains the evidence needed to answer. Additional
    source/report fields, gold answers, and historical verdicts are never sent.
    Comparison tolerances and unit conversions are public evaluation rules.
    This does not semantically redact answers hidden inside prose.
    """
    return {
        "item_id": item["id"],
        "report_title": report["title"],
        "report_text": _PLACEHOLDER.sub("[MASK]", report["text"]),
        "statement": item["statement"].replace("{{value}}", "[MASK]"),
        "unit": item["unit"],
        "as_of": report["as_of"],
        "sources": [{key: source[key] for key in _SOURCE_FIELDS if key in source} for source in report["sources"]],
        "comparison_rules": {
            "absolute_tolerance": item.get("tolerance", "0"),
            "target_unit": item["unit"],
            "accepted_units": dict(item.get("accepted_units", {})),
        },
    }
