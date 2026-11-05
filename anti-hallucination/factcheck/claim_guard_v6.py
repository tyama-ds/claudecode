"""Opt-in v6: v5 eligibility, citation-boundary screening and bounded removal.

This module never rewrites a replacement or supplies semantic judgments. Bare
number intent comes from explicit citation review spans, not numeric appearance.
Full independent review is mandatory after every changed nonempty selection.
"""
from __future__ import annotations

import copy
import difflib
import re

import replication as rep
from . import claim_guard_v5 as v5

CHECKS = v5.CHECKS
GATE_VERSION = "v6"
MAX_RECOVERY_ROUNDS = 1
LIMITATION = v5.LIMITATION + " Citation screening covers declared bare numeric citation lines; it does not prove citation intent or semantic correctness."
sha256_json = v5.sha256_json
build_block_packet = v5.build_block_packet
build_guard_packet = v5.build_guard_packet
validate_ledger = v5.validate_ledger

_BREAKS = "\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029"


def _declared_tokens(text, items, field, source_ids):
    """Locate uniquely quoted numeric-only citation annotations.

    Broad citation-unit quotes may contain factual numbers. Only annotations
    made entirely of bare numbered citation lines identify numeric intent.
    Repeated annotation quotes have no supplied offsets, so remain ambiguous.
    """
    tokens, ambiguities = [], []
    for index, item in enumerate(items):
        if item.get("content_kind", "factual") != "citation":
            continue
        quote = item.get(field, "")
        lines = [line for line in quote.splitlines() if line.strip()]
        if not lines or any(re.fullmatch(r"[ \t]*[0-9]+[ \t]*", line) is None for line in lines):
            continue
        locations = list(v5._occurrences(text, quote))
        if len(locations) != 1:
            ambiguities.append({"field": field, "annotation_index": index, "quote": quote,
                "occurrence_count": len(locations), "reason": "citation annotation has no unique text occurrence; intent remains for full semantic review"})
            continue
        offset = locations[0][0]
        tokens.extend((offset + a, offset + b, identity) for a, b, identity in _line_tokens(quote, source_ids))
    return tokens, ambiguities


def _line_tokens(text, source_ids):
    tokens, offset = [], 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip(_BREAKS)
        match = re.fullmatch(r"[ \t]*([0-9]+)[ \t]*", content)
        if match and match.group(1) in source_ids:
            tokens.append((offset + match.start(1), offset + match.end(1), match.group(1)))
        offset += len(line)
    return tokens


def _assemble_provenance(original, applied):
    """Map selected characters to original positions and actual editing owners.

    Equal characters retain their original positions even inside a replacement.
    Edit boundaries preserve ownership of zero-character deletions as well.
    """
    origin, owner, seams, patch_ranges = [], [], {}, {}
    cursor = 0
    for patch in applied["patches"]:
        start, end = patch["start_char"], patch["end_char"]
        origin.extend(range(cursor, start))
        owner.extend([None] * (start - cursor))
        offset = len(origin)
        patch_ranges[patch["patch_id"]] = (offset, offset + len(patch["after"]))
        mapped = [None] * len(patch["after"])
        owned = [patch["patch_id"]] * len(patch["after"])
        for tag, a, b, c, d in difflib.SequenceMatcher(
                None, patch["before"], patch["after"], autojunk=False).get_opcodes():
            if tag == "equal":
                mapped[c:d] = range(start + a, start + b)
                owned[c:d] = [None] * (d - c)
            else:
                for edge in (offset + c, offset + d):
                    seams.setdefault(edge, set()).add(patch["patch_id"])
        origin.extend(mapped)
        owner.extend(owned)
        cursor = end
    origin.extend(range(cursor, len(original["text"])))
    owner.extend([None] * (len(original["text"]) - cursor))
    assert len(origin) == len(applied["text"])
    return origin, owner, seams, patch_ranges


def _boundary_candidates(original, applied, guard):
    """Collect declared-token joins and potential editing owners, without blame."""
    text = applied["text"]
    source_ids = {str(source["citation_id"]) for source in applied["sources"]}
    reviews = {row["patch_id"]: row for row in guard["patch_reviews"]}
    origin, owner, seams, patch_ranges = _assemble_provenance(original, applied)
    inverse = {position: index for index, position in enumerate(origin) if position is not None}
    anchors = {}
    # An original numeric value may equal a source ID. Original anchors require
    # explicit before-quote citation intent too; a bare number alone is ambiguous.
    original_citations, ambiguities = [], []
    for patch in applied["patches"]:
        tokens, uncertain = _declared_tokens(patch["before"], reviews[patch["patch_id"]]["units"], "before_quote", source_ids)
        original_citations.extend((patch["start_char"] + a, patch["start_char"] + b) for a, b, _ in tokens)
        ambiguities.extend({"patch_id": patch["patch_id"], **item} for item in uncertain)
    original_ids = {str(source["citation_id"]) for source in original["sources"]}
    for start, end, identity in _line_tokens(original["text"], original_ids):
        if not any(a <= start and end <= b for a, b in original_citations):
            continue
        mapped = [inverse.get(position) for position in range(start, end)]
        if mapped and None not in mapped and mapped == list(range(mapped[0], mapped[0] + len(mapped))):
            anchors[(mapped[0], mapped[-1] + 1)] = {"citation_id": identity, "intent_origin": "original_standalone_line",
                "anchor_id": f"original:{start}:{end}"}
    for patch in applied["patches"]:
        offset = patch_ranges[patch["patch_id"]][0]
        row = reviews[patch["patch_id"]]
        tokens, uncertain = _declared_tokens(patch["after"], row["units"] + row["additions"], "after_quote", source_ids)
        ambiguities.extend({"patch_id": patch["patch_id"], **item} for item in uncertain)
        for start, end, identity in tokens:
            positions = origin[offset + start:offset + end]
            unchanged = (positions and all(position is not None for position in positions) and
                         positions == list(range(positions[0], positions[0] + len(positions))))
            anchor_id = (f"original:{positions[0]}:{positions[-1] + 1}" if unchanged else
                         f"patch:{patch['patch_id']}:{start}:{end}")
            anchors.setdefault((offset + start, offset + end), {
                "citation_id": identity, "intent_origin": "reviewed_citation_line",
                "citation_patch_id": patch["patch_id"], "anchor_id": anchor_id})
    diagnostics = []
    for (start, end), anchor in sorted(anchors.items()):
        for side, step, edge in (("left", -1, start - 1), ("right", 1, end)):
            neighbor = edge
            while 0 <= neighbor < len(text) and text[neighbor] in " \t":
                neighbor += step
            if not 0 <= neighbor < len(text) or text[neighbor] in _BREAKS:
                continue
            lo, hi = (neighbor, end) if side == "left" else (start, neighbor + 1)
            positions = origin[lo:hi]
            # A verbatim contiguous original join is not a newly caused defect.
            if positions and all(position is not None for position in positions) and positions == list(range(positions[0], positions[0] + len(positions))):
                continue
            implicated = {identity for identity in owner[lo:hi] if identity is not None}
            for boundary in range(lo + 1, hi):
                implicated.update(seams.get(boundary, ()))
            diagnostics.append({"kind": "citation_boundary_join", **anchor, "side": side,
                "start_char": start, "end_char": end, "neighbor_char": neighbor,
                "assembled_quote": text[max(0, lo - 30):min(len(text), hi + 30)],
                "joined_text": text[lo:hi], "responsible_patch_ids": sorted(implicated),
                "attribution": "definite" if implicated else "ambiguous"})
    return diagnostics, ambiguities


def citation_boundary_diagnostics(original, applied, guard):
    """Screen actual assembled text and attribute only counterfactual causes.

    A unique numeric-only citation annotation supplies each bare decimal token
    even if its replacement already joins it to prose. Complete brackets are
    permitted inline. An adjacent body edit is not responsible if restoring
    that patch's original span leaves the same anchor-side join. Counterfactual
    applications are diagnostic only and never replace candidate text.
    """
    diagnostics, _ = _boundary_candidates(original, applied, guard)
    without_patch = {}
    for identity in sorted({identity for item in diagnostics for identity in item["responsible_patch_ids"]}):
        counterfactual = rep.apply_repair(original, {
            "patches": [patch for patch in applied["patches"] if patch["patch_id"] != identity],
            "added_sources": applied["sources"][len(original["sources"]):]})
        candidates, _ = _boundary_candidates(original, counterfactual, guard)
        without_patch[identity] = {(item["anchor_id"], item["side"]) for item in candidates}
    for item in diagnostics:
        key = (item["anchor_id"], item["side"])
        potential = item["responsible_patch_ids"]
        item["counterfactual_attribution"] = [{"patch_id": identity,
            "same_join_remains_without_patch": key in without_patch[identity]} for identity in potential]
        item["responsible_patch_ids"] = [identity for identity in potential if key not in without_patch[identity]]
        item["attribution"] = "definite" if item["responsible_patch_ids"] else "ambiguous"
    return diagnostics


def _effective(original, candidate, accepted):
    effective = copy.deepcopy(candidate)
    effective["patches"] = [copy.deepcopy(patch) for patch in candidate["patches"] if patch["patch_id"] in accepted]
    effective["added_sources"] = v5._retained_sources(candidate, effective["patches"])
    for patch in effective["patches"]:
        patch.pop("claim_ids", None)
        patch.pop("depends_on_patch_ids", None)
    return effective, rep.validate_repair(original, effective)


def _reject(result, identities, phase, reason, *, details=None):
    identities = set(identities)
    for decision in result["decisions"]:
        if decision["accepted"] and decision["patch_id"] in identities:
            decision["accepted"] = False
            decision["reasons"].append(reason)
            entry = {"patch_id": decision["patch_id"], "phase": phase, "reason": reason}
            if details is not None:
                entry["details"] = copy.deepcopy(details)
            result["exclusion_trail"].append(entry)


def _accepted(result):
    return {row["patch_id"] for row in result["decisions"] if row["accepted"]}


def _dependency_closure(result, candidate):
    while True:
        accepted = _accepted(result)
        removed = False
        for patch in candidate["patches"]:
            missing = set(patch.get("depends_on_patch_ids", [])) - accepted
            if patch["patch_id"] in accepted and missing:
                _reject(result, [patch["patch_id"]], "dependency", "rejected patch dependencies: " + ", ".join(sorted(missing)),
                        details={"missing_patch_ids": sorted(missing)})
                removed = True
        if not removed:
            return


def _screen(result, original, candidate, guard):
    # Each nonterminal iteration removes at least one patch; at most N+1 scans.
    while True:
        _dependency_closure(result, candidate)
        effective, applied = _effective(original, candidate, _accepted(result))
        diagnostics = citation_boundary_diagnostics(original, applied, guard)
        _, ambiguities = _boundary_candidates(original, applied, guard)
        iteration = len(result["mechanical_diagnostics"])
        result["mechanical_diagnostics"].append({"iteration": iteration,
            "accepted_patch_ids": sorted(_accepted(result)),
            "text_sha256_utf8": applied["revised_sha256_utf8"],
            "sources_sha256": sha256_json(applied["sources"]), "violations": diagnostics,
            "intent_ambiguities": ambiguities})
        if not diagnostics:
            result["effective_repair"], result["selected_applied"] = effective, applied
            result["final_applied"] = copy.deepcopy(applied)
            return
        implicated = {identity for row in diagnostics for identity in row["responsible_patch_ids"]}
        if any(row["attribution"] != "definite" for row in diagnostics) or not implicated:
            implicated = _accepted(result)
            result["mechanical_veto"] = True
        before = _accepted(result)
        _reject(result, implicated, "mechanical", "new citation token joins neighboring text", details=diagnostics)
        if _accepted(result) == before:
            raise ValueError("citation screening could not remove the responsible selected patches")


def _selection(result, guard, *, round_number=0, previous_review=None):
    selected = result["selected_applied"]
    base_guard = {key: value for key, value in guard.items() if key not in {"final_review", "final_reviews"}}
    selection = {"gate_version": GATE_VERSION, "accepted_patch_ids": sorted(_accepted(result)),
        "final_text_sha256_utf8": selected["revised_sha256_utf8"],
        "final_sources_sha256": sha256_json(selected["sources"]),
        "selection_round": round_number, "previous_review_sha256": previous_review,
        "input_sha256": copy.deepcopy(result["input_sha256"]),
        "guard_packet_sha256": result["guard_packet_sha256"],
        "guard_evaluation_sha256": sha256_json(base_guard),
        "exclusion_trail_sha256": sha256_json(result["exclusion_trail"]),
        "mechanical_diagnostics_sha256": sha256_json(result["mechanical_diagnostics"])}
    selection["selection_sha256"] = sha256_json(selection)
    result["selection"] = selection


def _summary(result, candidate, *, required=False, passed=False, blockers=()):
    accepted = _accepted(result)
    result["summary"].update({"gate_version": GATE_VERSION,
        "accepted_patch_count": len(accepted), "rejected_patch_count": len(candidate["patches"]) - len(accepted),
        "rejected_patches": [{"patch_id": row["patch_id"], "reasons": row["reasons"]}
                             for row in result["decisions"] if not row["accepted"]],
        "retained_added_source_count": len(result["effective_repair"]["added_sources"]),
        "rewrite_retries": 0, "semantic_recovery_rounds": result["selection"]["selection_round"],
        "maximum_semantic_recovery_rounds": MAX_RECOVERY_ROUNDS,
        "recovery_status": result.get("recovery_status", "not_requested"),
        "recovery_terminal_reason": result.get("recovery_terminal_reason", ""),
        "final_review_required": required, "final_review_passed": passed,
        "final_review_blockers": list(blockers), "limitation": LIMITATION})


def preview_repair(original, candidate, ledger, guard, *, guard_packet=None):
    """Return a provisional, mechanically stable selection for fresh review."""
    result = copy.deepcopy(v5.preview_repair(original, candidate, ledger, guard, guard_packet=guard_packet))
    result["gate_version"] = GATE_VERSION
    result["mechanical_diagnostics"], result["review_history"] = [], []
    result["mechanical_veto"] = False
    result["exclusion_trail"] = [{"patch_id": row["patch_id"], "phase": "v5_eligibility",
        "reason": "; ".join(row["reasons"])} for row in result["decisions"] if not row["accepted"]]
    _screen(result, original, candidate, guard)
    _selection(result, guard)
    result["recovery_eligible"] = False
    _summary(result, candidate)
    return result


def _validate_review(review, selection):
    label = "final_review"
    rep._object(review, label)
    fields = set(selection) | {"status", "coverage_complete", "reason", "patch_localization"}
    rep._all_ids(review, fields, label)
    for field, value in selection.items():
        if review.get(field) != value or type(review.get(field)) is not type(value):
            raise ValueError(f"final_review {field} mismatch with selected report")
    rep._enum(review["status"], {"pass", "fail", "unknown"}, "final_review.status")
    rep._boolean(review["coverage_complete"], "final_review.coverage_complete")
    v5._reason(review["reason"], "final_review.reason")
    localization = rep._object(review["patch_localization"], "final_review.patch_localization")
    rep._all_ids(localization, {"status", "patch_ids", "all_failure_causes_localized", "reason"}, "final_review.patch_localization")
    rep._enum(localization["status"], {"none", "definite", "ambiguous", "unknown"}, "final_review.patch_localization.status")
    v5._refs(localization["patch_ids"], selection["accepted_patch_ids"], "final_review.patch_localization.patch_ids")
    rep._boolean(localization["all_failure_causes_localized"], "final_review.patch_localization.all_failure_causes_localized")
    v5._reason(localization["reason"], "final_review.patch_localization.reason")
    if localization["status"] == "none" and (localization["patch_ids"] or localization["all_failure_causes_localized"]):
        raise ValueError("none localization requires empty patch_ids and false all_failure_causes_localized")
    if review["status"] == "pass" and localization["status"] != "none":
        raise ValueError("pass requires none localization")


def _record_review(result, review):
    result["review_history"].append({"selection": copy.deepcopy(result["selection"]),
        "review_sha256": sha256_json(review), "review": copy.deepcopy(review)})


def _recoverable(review):
    localized = review["patch_localization"]
    return (review["status"] == "fail" and review["coverage_complete"] and
            localized["status"] == "definite" and localized["all_failure_causes_localized"] and
            bool(localized["patch_ids"]))


def _veto(result, original, candidate, blockers):
    _reject(result, _accepted(result), "final_veto", "; ".join(blockers))
    result["effective_repair"], result["final_applied"] = _effective(original, candidate, set())
    result["recovery_eligible"] = False
    result["recovery_status"] = "veto"
    _summary(result, candidate, required=True, blockers=blockers)
    return result


def _prepare_next(result, original, candidate, guard, review):
    _reject(result, review["patch_localization"]["patch_ids"], "localized_final_review",
            "independent review localized all failure causes to this patch",
            details={"review_sha256": sha256_json(review), "localization": review["patch_localization"]})
    _screen(result, original, candidate, guard)
    _selection(result, guard, round_number=1, previous_review=sha256_json(review))
    if result["mechanical_veto"]:
        return _veto(result, original, candidate, ["mechanical attribution was ambiguous after localized removal"])
    result["recovery_eligible"] = bool(_accepted(result))
    result["recovery_status"] = "awaiting_review" if result["recovery_eligible"] else "no_remainder"
    if not result["recovery_eligible"]:
        result["recovery_terminal_reason"] = "no selected patches remain after localized removal"
    _summary(result, candidate)
    return result


def prepare_recovery(original, candidate, ledger, guard, *, final_review, guard_packet=None):
    """Consume round-0 failure and prepare the full changed report for review.

    This helper never authorizes an applied repair. A nonlocal, incomplete, or
    unknown failure produces a terminal full veto; malformed records raise.
    """
    result = preview_repair(original, candidate, ledger, guard, guard_packet=guard_packet)
    _validate_review(final_review, result["selection"])
    _record_review(result, final_review)
    if not _recoverable(final_review) or result["mechanical_veto"]:
        return _veto(result, original, candidate, ["initial review does not permit definite complete localized recovery"])
    return _prepare_next(result, original, candidate, guard, final_review)


def gate_repair(original, candidate, ledger, guard, *, guard_packet=None):
    """Apply only after a fresh full pass, with at most one removal/review round."""
    result = preview_repair(original, candidate, ledger, guard, guard_packet=guard_packet)
    if "final_review" in guard and "final_reviews" in guard:
        raise ValueError("provide final_review or final_reviews, not both")
    reviews = guard.get("final_reviews", [guard.get("final_review")])
    rep._array(reviews, "guard.final_reviews")
    if not 1 <= len(reviews) <= MAX_RECOVERY_ROUNDS + 1:
        raise ValueError("guard.final_reviews requires one or two reviews")
    first = reviews[0]
    _validate_review(first, result["selection"])
    _record_review(result, first)
    if first["status"] == "pass" and first["coverage_complete"] and not result["mechanical_veto"]:
        if len(reviews) != 1:
            raise ValueError("extra review after terminal pass")
        result["recovery_status"] = "not_needed"
        _summary(result, candidate, required=True, passed=True)
        return result
    if not _recoverable(first) or result["mechanical_veto"]:
        if len(reviews) != 1:
            raise ValueError("extra review after nonrecoverable failure")
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
    _record_review(result, second)
    if second["status"] != "pass" or not second["coverage_complete"] or result["mechanical_veto"]:
        return _veto(result, original, candidate, ["final re-review did not provide a complete pass; recovery limit reached"])
    result["recovery_eligible"] = False
    result["recovery_status"] = "passed"
    _summary(result, candidate, required=True, passed=True)
    return result


def schema_contract():
    contract = copy.deepcopy(v5.schema_contract())
    contract["gate_version"] = GATE_VERSION
    contract["limitation"] = LIMITATION
    contract["guard"]["final_review"] = {
        "selection_binding": "Copy every field of preview.selection exactly; operator may attach these after a blinded semantic review.",
        "status": "pass|fail|unknown", "coverage_complete": "boolean", "reason": "nonempty full-report independent assessment",
        "patch_localization": {"status": "none|definite|ambiguous|unknown", "patch_ids": "unique selected patch IDs",
            "all_failure_causes_localized": "boolean; true only if ALL failure causes are definitely attributable to the listed patches",
            "reason": "nonempty explanation; none requires empty IDs and false attestation; pass requires none"}}
    contract["guard"]["final_reviews"] = "Alternative to final_review: ordered array of one or two strict reviews; round 1 is permitted only after a recoverable round-0 failure. Never provide both fields."
    contract["gate_rules"] = contract["gate_rules"][:-2] + [
        "Reuse v5 eligibility, original information protection and source retention without changing frozen inputs or labels.",
        "Screen assembled selected text, not candidate fragments alone. Each bare decimal source-ID token requires a uniquely occurring numeric-only citation-unit/addition quote (one number per physical line, horizontal whitespace allowed). Token boundaries come from the annotation itself, so a declared citation already joined inside a replacement is screened. Original anchors additionally use original standalone lines and numeric-only citation before_quote declarations. Repeated quotes and broad contextual citation quotes do not establish precise numeric intent; record repeated-quote ambiguity and leave it for full semantic review. Numeric source membership alone never classifies a number. Untouched original number lines without citation intent declarations remain for independent review. Complete bracketed citations may appear inline; malformed bracket syntax remains for independent review.",
        "Track exact equal-character original provenance and edit seams. Existing unchanged original joins are not blamed. Counterfactually revert each potential responsible patch: an adjacent edit is excluded from blame if the same anchor-side join remains. Reject definite responsible patches and transitive dependents; reassemble and rescreen to a fixed point without editing any replacement. Ambiguous mechanical attribution vetoes the whole selection.",
        "Every selection binds exact text/sources, input JSON, packet, guard evaluation, exclusion trail and mechanical diagnostics. Final reviews copy all selection fields exactly; stale or malformed reviews raise before applying.",
        "A complete failed full-report review can authorize one removal round only if localization is definite, every ID is selected, the list is nonempty and all failure causes are attested localized. Unknown, incomplete, absent or ambiguous localization cannot recover. Remove only these patches plus dependent/mechanically implicated patches; never rewrite.",
        "The changed full report requires a second independently produced complete pass with its new exact selection binding. No old pass transfers. Failed/incomplete/unknown second review or missing re-review vetoes all remaining patches. Maximum recovery rounds is one.",
        "If localized removal and closure leave no selected patches, recovery terminates with recovery_status=no_remainder, recovery_eligible=false and the original report. No second review is requested and no recovery pass is claimed; an unexpected second review is rejected.",
        "A blinded reviewer can return only status, coverage_complete, reason, patch_localization; the operator attaches the coded selection's binding fields. The code validates structure and provenance, not reviewer independence or truth."]
    contract["cli"].append("recover --original PATH --candidate PATH --ledger PATH --guard-packet PATH --guard PATH --final-review PATH --output-dir DIR")
    contract["report_interface"] = "v5-compatible result fields plus mechanical_diagnostics, exclusion_trail, review_history, mechanical_veto, recovery_eligible; selected_applied is the last reviewed/provisional selection, final_applied is original on full veto. Selection is immutable and remains bound to the reviewed selection even when final veto changes effective_repair."
    return contract
