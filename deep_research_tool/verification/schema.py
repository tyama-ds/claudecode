"""
Strict, fail-closed validators for every LLM-produced JSON structure.

Contract (mirrors the pipeline specification):
- status: one of the allowed enum strings, NOTHING else (no dicts,
  lists, numbers, null);
- reason: str only;
- supporting_source_ids: list[str] only — nested lists, dicts, ints and
  null elements make the whole verdict invalid;
- source_numbers: list[int] only (bool is not an int here) — ["1"] is
  invalid;
- answered: JSON boolean only — "false", "true", 0, 1 are invalid;
- search_queries: list[str] only.

A validator returns the NORMALIZED value or ``None`` — it never raises.
Callers do a bounded retry on ``None`` and then fail closed
(uncertain / unanswered); nothing is auto-filled and no schema anomaly
may crash the verification run.
"""

from typing import Any, Dict, List, Optional

VALID_CLAIM_STATUSES = ("supported", "unsupported", "contradicted",
                        "uncertain")


def as_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def as_str_list(value: Any) -> Optional[List[str]]:
    """list[str] or None. Rejects nested lists/dicts/numbers/null."""
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    return list(value)


def as_int_list(value: Any) -> Optional[List[int]]:
    """list[int] or None. bool is NOT an int here; "1" is invalid."""
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool)
               for item in value):
        return None
    return list(value)


def as_json_bool(value: Any) -> Optional[bool]:
    """A JSON boolean or None — strings and numbers are anomalies."""
    return value if isinstance(value, bool) else None


def validate_verdict(raw: Any,
                     require_ids_for_supported: bool = True,
                     ) -> Optional[Dict[str, Any]]:
    """One claim-judgement verdict, or None on ANY schema anomaly."""
    if not isinstance(raw, dict):
        return None
    status = raw.get("status")
    if not isinstance(status, str) or status not in VALID_CLAIM_STATUSES:
        return None
    reason = raw.get("reason", "")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        return None
    ids = as_str_list(raw.get("supporting_source_ids"))
    if ids is None:
        return None
    if require_ids_for_supported and status == "supported" and not ids:
        return None
    return {"status": status, "reason": reason,
            "supporting_source_ids": ids}


def validate_extracted_claim(raw: Any) -> Optional[Dict[str, Any]]:
    """One extracted claim; None on anomaly (the claim is dropped, and
    the body sentence keeps its verification via the other claims —
    dropping is safe because extraction is re-runnable, unlike a
    judgement that silently supports)."""
    if not isinstance(raw, dict):
        return None
    text = raw.get("claim")
    if not isinstance(text, str) or not text.strip():
        return None
    importance = raw.get("importance", "important")
    if not isinstance(importance, str):
        importance = "important"
    out: Dict[str, Any] = {"claim": text.strip(), "importance": importance}
    if "source_numbers" in raw:
        numbers = as_int_list(raw.get("source_numbers"))
        # invalid source_numbers -> the FIELD is unusable (treated as
        # unreported); the deterministic body parser is the citation
        # authority anyway, so nothing is guessed from a broken field
        out["source_numbers"] = numbers          # None when anomalous
    return out


def validate_coverage_entry(raw: Any) -> Optional[Dict[str, Any]]:
    """One coverage entry; None on anomaly (fail-closed: unanswered)."""
    if not isinstance(raw, dict):
        return None
    try:
        idx = raw.get("question")
        if isinstance(idx, bool) or not isinstance(idx, (int, str)):
            return None
        idx = int(idx)
    except (TypeError, ValueError):
        return None
    answered = as_json_bool(raw.get("answered"))
    if answered is None:
        return None
    section_id = raw.get("section_id", "")
    if section_id is None:
        section_id = ""
    if not isinstance(section_id, str):
        return None
    missing = raw.get("missing", "")
    if missing is None:
        missing = ""
    if not isinstance(missing, str):
        return None
    queries = as_str_list(raw.get("search_queries"))
    if queries is None:            # e.g. a bare string / nested lists
        return None
    return {"question": idx, "answered": answered,
            "section_id": section_id, "missing": missing,
            "search_queries": queries}
