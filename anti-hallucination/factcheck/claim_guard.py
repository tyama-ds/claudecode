"""Deterministic claim-ledger validation and a conservative preservation gate.

Models supply all semantic judgments. This module checks their structure,
provenance, exact quotes and declared preservation constraints, never truth.
Block coverage and unit coverage attestations do not prove exhaustive atomic
claim extraction. No baseline or gold labels are accepted by this interface.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter

import replication as rep
from .repair import _https_url


CHECKS = ("quantity", "metric", "population", "time", "comparison", "modality", "citation")
CHECK_STATUSES = {"supported", "contradicted", "missing_material", "unknown", "not_applicable"}
CLAIM_STATUSES = {"supported", "problem", "uncertain", "nonfactual"}
LIMITATION = ("Block coverage and exact unit mappings are structural checks, not proof of "
              "exhaustive atomic extraction or factual truth; semantic labels are supplied by reviewers.")


def sha256_json(value):
    """Hash JSON content independently of indentation and object-key ordering."""
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _version(record, label):
    rep._object(record, label)
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError(f"{label}.schema_version must be 1")


def _reason(value, label):
    return rep._string(value, label, nonempty=True)


def _refs(value, known, label, *, nonempty=False):
    result = rep._id_refs(value, known, label)
    if nonempty and not result:
        raise ValueError(f"{label} must contain at least one ID")
    return result


def _evidence(rows, label):
    result = rep._rows_by_id(rows, "evidence_id", label)
    for identity, row in result.items():
        name = f"{label}.{identity}"
        _https_url(row.get("url"), f"{name}.url")
        for field in ("locator", "summary"):
            _reason(row.get(field), f"{name}.{field}")
        rep._enum(row.get("access"), {"full_section", "partial", "unavailable"}, f"{name}.access")
        rep._boolean(row.get("primary"), f"{name}.primary")
    return result


def _primary_accessible(refs, evidence):
    # A search snippet/partial section does not establish a supported finding.
    return any(evidence[identity]["primary"] and evidence[identity]["access"] == "full_section"
               for identity in refs)


def _accessible(refs, evidence):
    return any(evidence[identity]["access"] == "full_section" for identity in refs)


def build_block_packet(original):
    """Enumerate every nonblank physical line, retaining exact character offsets.

    A one-line paragraph is one block; a paragraph with hard line breaks has
    several blocks. Line terminators are excluded. Offsets count Python Unicode
    characters, not UTF-8 bytes or UTF-16 code units.
    """
    rep.validate_original(original)
    blocks = []
    offset = 0
    for line in original["text"].splitlines(keepends=True):
        content = line.rstrip("\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029")
        if content.strip():
            blocks.append({"block_id": f"B{len(blocks) + 1:04d}", "start_char": offset,
                           "end_char": offset + len(content), "sha256_utf8": _text_hash(content),
                           "text": content})
        offset += len(line)
    return {"schema_version": 1, "original_sha256": sha256_json(original),
            "original_text_sha256": _text_hash(original["text"]),
            "original": copy.deepcopy(original), "blocks": blocks}


def validate_ledger(original, ledger, block_packet=None):
    """Validate explicit block coverage and seven checks; return summary counts."""
    expected = build_block_packet(original)
    if block_packet is not None and block_packet != expected:
        raise ValueError("block packet does not match original")
    _version(ledger, "ledger")
    if ledger.get("original_sha256") != expected["original_sha256"]:
        raise ValueError("ledger original_sha256 mismatch")
    if ledger.get("block_packet_sha256") != sha256_json(expected):
        raise ValueError("ledger block_packet_sha256 mismatch")
    evidence = _evidence(ledger.get("evidence"), "ledger.evidence")
    blocks = {row["block_id"]: row for row in expected["blocks"]}
    coverage = rep._rows_by_id(ledger.get("blocks"), "block_id", "ledger.blocks")
    rep._all_ids(coverage, blocks, "ledger block coverage")
    claims = rep._rows_by_id(ledger.get("claims"), "claim_id", "ledger.claims")
    mapped = []
    for identity, row in coverage.items():
        name = f"ledger.blocks.{identity}"
        refs = _refs(row.get("claim_ids"), claims, f"{name}.claim_ids")
        reason = rep._string(row.get("nonfactual_reason"), f"{name}.nonfactual_reason")
        if (not refs and not reason.strip()) or (refs and reason != ""):
            raise ValueError(f"{name} must list claims or an explicit nonfactual_reason, exclusively")
        for claim_id in refs:
            if claims[claim_id].get("block_id") != identity:
                raise ValueError(f"{name}: claim {claim_id} belongs to another block")
        mapped.extend(refs)
    if len(mapped) != len(set(mapped)) or set(mapped) != set(claims):
        raise ValueError("Every ledger claim must be mapped exactly once to its block")
    for identity, row in claims.items():
        name = f"ledger.claims.{identity}"
        block_id = row.get("block_id")
        if not isinstance(block_id, str) or block_id not in blocks:
            raise ValueError(f"{name}.block_id is unknown")
        rep._quote(row.get("quote"), blocks[block_id]["text"], f"{name}.quote")
        status = rep._enum(row.get("status"), CLAIM_STATUSES, f"{name}.status")
        _reason(row.get("reason"), f"{name}.reason")
        context = _refs(row.get("context_block_ids"), blocks, f"{name}.context_block_ids", nonempty=True)
        if block_id not in context:
            raise ValueError(f"{name}.context_block_ids must include its own block")
        for field in ("report_scope", "source_scope"):
            rep._string(row.get(field), f"{name}.{field}", nonempty=status != "nonfactual")
        checks = rep._object(row.get("checks"), f"{name}.checks")
        rep._all_ids(checks, CHECKS, f"{name}.checks")
        applicable = []
        for check in CHECKS:
            item = rep._object(checks[check], f"{name}.checks.{check}")
            label = f"{name}.checks.{check}"
            check_status = rep._enum(item.get("status"), CHECK_STATUSES, f"{label}.status")
            _reason(item.get("rationale"), f"{label}.rationale")
            refs = _refs(item.get("evidence_ids"), evidence, f"{label}.evidence_ids")
            if check_status != "not_applicable":
                applicable.append(check_status)
            accessible = _accessible(refs, evidence) if check == "citation" else _primary_accessible(refs, evidence)
            if check_status == "supported" and not accessible:
                raise ValueError(f"{label}: supported requires {'accessible' if check == 'citation' else 'primary'} full_section evidence")
        if status == "supported" and (not applicable or any(s != "supported" for s in applicable)):
            raise ValueError(f"{name}: supported requires at least one applicable check and all applicable checks supported")
        if status == "nonfactual" and applicable:
            raise ValueError(f"{name}: nonfactual requires every check to be not_applicable")
        if status == "problem" and not any(s in {"contradicted", "missing_material"} for s in applicable):
            raise ValueError(f"{name}: problem requires a contradicted or missing_material check")
    counts = Counter(row["status"] for row in claims.values())
    return {"block_count": len(blocks), "covered_block_count": len(coverage), "block_coverage_complete": True,
            "claim_count": len(claims), "claim_status_counts": {key: counts[key] for key in sorted(CLAIM_STATUSES)},
            "supported_claim_count": counts["supported"], "uncertain_claim_count": counts["uncertain"],
            "limitation": LIMITATION}


def _occurrences(text, quote):
    start = text.find(quote)
    while start >= 0:
        yield start, start + len(quote)
        start = text.find(quote, start + 1)


def _claim_spans(ledger, block_packet):
    blocks = {row["block_id"]: row for row in block_packet["blocks"]}
    return {claim["claim_id"]: [(a + blocks[claim["block_id"]]["start_char"],
                                 b + blocks[claim["block_id"]]["start_char"])
                                for a, b in _occurrences(blocks[claim["block_id"]]["text"], claim["quote"])]
            for claim in ledger["claims"]}


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def build_guard_packet(original, candidate, ledger, *, input_file_sha256=None):
    """Freeze exactly the original, candidate application and ledger for review."""
    block_packet = build_block_packet(original)
    validate_ledger(original, ledger, block_packet)
    applied = rep.validate_repair(original, candidate)
    spans = _claim_spans(ledger, block_packet)
    patch_ids = {patch["patch_id"] for patch in applied["patches"]}
    dependencies = {}
    for patch in applied["patches"]:
        label = f"candidate.patches.{patch['patch_id']}.claim_ids"
        refs = _refs(patch.get("claim_ids"), spans, label, nonempty=True)
        touched = {identity for identity, locations in spans.items()
                   if any(_overlap(location, (patch["start_char"], patch["end_char"])) for location in locations)}
        rep._all_ids(refs, touched, label)
        dependencies[patch["patch_id"]] = _refs(patch.get("depends_on_patch_ids", []), patch_ids,
                                               f"candidate.patches.{patch['patch_id']}.depends_on_patch_ids")
    def visit(identity, active, done):
        if identity in active:
            raise ValueError("candidate patch dependencies contain a cycle")
        if identity not in done:
            active.add(identity)
            for dependency in dependencies[identity]:
                visit(dependency, active, done)
            active.remove(identity)
            done.add(identity)
    done = set()
    for identity in dependencies:
        visit(identity, set(), done)
    packet = {"schema_version": 1, "input_sha256": {"original": sha256_json(original),
              "candidate": sha256_json(candidate), "ledger": sha256_json(ledger)},
              "original": copy.deepcopy(original), "candidate_repair": copy.deepcopy(candidate), "candidate_applied": applied,
              "ledger": copy.deepcopy(ledger), "block_packet": block_packet,
              "limitation": LIMITATION}
    if input_file_sha256 is not None:
        rep._object(input_file_sha256, "input_file_sha256")
        rep._all_ids(input_file_sha256, ("original", "candidate", "ledger"), "input_file_sha256")
        for value in input_file_sha256.values():
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("input_file_sha256 values must be SHA256 hex strings")
        packet["input_file_sha256"] = copy.deepcopy(input_file_sha256)
    return packet


def _retained_sources(candidate, accepted):
    """Keep all source dependencies, including those introduced by rejected peers.

    Never renumber. URLs and standalone numeric tokens in accepted replacements
    conservatively count as references. A number used as data may retain an
    extra source, but this deliberately avoids deleting an ambiguous dependency.
    """
    urls = {url for patch in accepted for url in patch["source_urls"]}
    result = []
    for source in candidate["added_sources"]:
        identity = str(source["citation_id"])
        pattern = r"(?<![0-9])" + re.escape(identity) + r"(?![0-9])"
        if source["url"] in urls or any(source["url"] in patch["after"] or re.search(pattern, patch["after"])
                                         for patch in accepted):
            result.append(copy.deepcopy(source))
    return result


def _select_repair(original, candidate, ledger, guard, *, guard_packet=None, require_final_review=True):
    """Keep only necessary, fully mapped, source-backed preservation-safe patches.

    Invalid records raise ValueError. Valid but uncertain/optional/unsafe reviews
    reject the affected patch, leaving the original span untouched. There is no
    rewrite or retry. The returned effective_repair uses the existing schema.
    """
    file_hashes = guard_packet.get("input_file_sha256") if isinstance(guard_packet, dict) else None
    expected = build_guard_packet(original, candidate, ledger, input_file_sha256=file_hashes)
    if guard_packet is not None and guard_packet != expected:
        raise ValueError("guard packet does not match original/candidate/ledger")
    _version(guard, "guard")
    if guard.get("guard_packet_sha256") != sha256_json(expected):
        raise ValueError("guard guard_packet_sha256 mismatch")
    if guard.get("input_sha256") != expected["input_sha256"]:
        raise ValueError("guard input_sha256 mismatch")
    evidence = _evidence(ledger["evidence"], "ledger.evidence")
    for identity, row in _evidence(guard.get("evidence"), "guard.evidence").items():
        if identity in evidence and row != evidence[identity]:
            raise ValueError(f"Conflicting evidence_id in guard: {identity}")
        evidence[identity] = row
    patches = {row["patch_id"]: row for row in expected["candidate_applied"]["patches"]}
    reviews = rep._rows_by_id(guard.get("patch_reviews"), "patch_id", "guard.patch_reviews")
    rep._all_ids(reviews, patches, "guard patch coverage")
    spans = _claim_spans(ledger, expected["block_packet"])
    claims = {row["claim_id"]: row for row in ledger["claims"]}
    decisions = []
    for identity, patch in patches.items():
        row = reviews[identity]
        label = f"guard.patch_reviews.{identity}"
        necessity = rep._enum(row.get("necessity"), {"necessary", "optional", "unnecessary", "uncertain"}, f"{label}.necessity")
        _reason(row.get("rationale"), f"{label}.rationale")
        refs = _refs(row.get("evidence_ids"), evidence, f"{label}.evidence_ids")
        complete = rep._boolean(row.get("coverage_complete"), f"{label}.coverage_complete")
        _reason(row.get("coverage_reason"), f"{label}.coverage_reason")
        units = rep._rows_by_id(row.get("units"), "unit_id", f"{label}.units")
        reasons = []
        if necessity != "necessary":
            reasons.append(f"necessity is {necessity}")
        if not complete:
            reasons.append("reviewer did not attest complete original-assertion coverage")
        if not units:
            reasons.append("no original information units mapped")
        if not _accessible(refs, evidence):
            reasons.append("necessity lacks full_section evidence")
        if patch["before"] == patch["after"]:
            reasons.append("patch makes no change")
        mapped_claims = set()
        has_correction = False
        for unit_id, unit in units.items():
            name = f"{label}.units.{unit_id}"
            before = unit.get("before_quote")
            after = unit.get("after_quote")
            rep._quote(before, patch["before"], f"{name}.before_quote")
            rep._quote(after, patch["after"], f"{name}.after_quote", allow_empty=True)
            unit_refs = _refs(unit.get("claim_ids"), patch["claim_ids"], f"{name}.claim_ids")
            locations = [(a + patch["start_char"], b + patch["start_char"])
                         for a, b in _occurrences(patch["before"], before)]
            for claim_id in unit_refs:
                if not any(_overlap(a, b) for a in locations for b in spans[claim_id]):
                    raise ValueError(f"{name}: before_quote does not overlap claim {claim_id}")
            mapped_claims.update(unit_refs)
            status = rep._enum(unit.get("status"), {"supported", "uncertain", "contradicted"}, f"{name}.status")
            kind = rep._enum(unit.get("content_kind", "factual"), {"factual", "citation", "nonfactual"}, f"{name}.content_kind")
            source_refs = _refs(unit.get("evidence_ids"), evidence, f"{name}.evidence_ids")
            _reason(unit.get("reason"), f"{name}.reason")
            disposition = rep._enum(unit.get("disposition"),
                {"preserved", "corrected_with_evidence", "removed_unsupported", "lost", "uncertain"}, f"{name}.disposition")
            _reason(unit.get("rationale"), f"{name}.rationale")
            primary = _primary_accessible(source_refs, evidence)
            source_backed = (_accessible(source_refs, evidence)
                             if kind == "citation" and patch["action"] == "relink" else primary)
            # A reviewer cannot evade retention by simply relabeling a ledger's
            # supported/uncertain assertion as contradicted and deleting it.
            protected = status in {"supported", "uncertain"} or any(
                claims[claim_id]["status"] in {"supported", "uncertain"} for claim_id in unit_refs)
            if status in {"supported", "contradicted"} and kind != "nonfactual" and not source_backed:
                reasons.append(f"{unit_id}: {status} unit lacks required full_section evidence")
            if disposition in {"lost", "uncertain"}:
                reasons.append(f"{unit_id}: disposition is {disposition}")
            elif disposition == "preserved":
                if before != after:
                    reasons.append(f"{unit_id}: preservation requires an identical exact quote")
                if status == "contradicted":
                    reasons.append(f"{unit_id}: contradicted information remains uncorrected")
            elif disposition == "corrected_with_evidence":
                has_correction = True
                if status != "contradicted" or not source_backed or not after or before == after or kind == "nonfactual":
                    reasons.append(f"{unit_id}: correction requires a contradicted unit, appropriate full_section evidence and a nonempty changed factual/citation quote")
            elif disposition == "removed_unsupported":
                has_correction = True
                if protected or status != "contradicted" or not source_backed or after or kind == "nonfactual":
                    reasons.append(f"{unit_id}: removal requires an unprotected contradicted unit, primary evidence and empty after_quote")
        missing = set(patch["claim_ids"]) - mapped_claims
        if missing:
            reasons.append(f"unmapped claim IDs: {', '.join(sorted(missing))}")
        for index, addition in enumerate(rep._array(row.get("additions"), f"{label}.additions")):
            name = f"{label}.additions[{index}]"
            rep._object(addition, name)
            rep._quote(addition.get("after_quote"), patch["after"], f"{name}.after_quote")
            kind = rep._enum(addition.get("content_kind"), {"factual", "citation", "nonfactual"}, f"{name}.content_kind")
            status = rep._enum(addition.get("status"), {"supported", "uncertain", "contradicted"}, f"{name}.status")
            source_refs = _refs(addition.get("evidence_ids"), evidence, f"{name}.evidence_ids")
            _refs(addition.get("claim_ids", []), claims, f"{name}.claim_ids")
            _reason(addition.get("reason"), f"{name}.reason")
            for field in ("report_scope", "source_scope"):
                rep._string(addition.get(field), f"{name}.{field}", nonempty=kind != "nonfactual")
            if status != "supported":
                reasons.append(f"addition {index}: status is {status}")
            if kind == "factual" and not _primary_accessible(source_refs, evidence):
                reasons.append(f"addition {index}: factual addition lacks primary full_section evidence")
            if kind == "citation" and not _accessible(source_refs, evidence):
                reasons.append(f"addition {index}: citation addition lacks full_section evidence")
        if not has_correction:
            reasons.append("no source-backed correction or contradicted-unit removal; added detail alone is insufficient")
        decisions.append({"patch_id": identity, "accepted": not reasons, "necessity": necessity,
                          "reasons": reasons, "review_rationale": row["rationale"]})
    # Filtering is monotone: rejected dependencies can only reject further
    # patches. A dependency never causes an unsafe patch to be admitted.
    accepted_ids = {row["patch_id"] for row in decisions if row["accepted"]}
    changed = True
    while changed:
        changed = False
        for decision in decisions:
            identity = decision["patch_id"]
            missing = set(patches[identity].get("depends_on_patch_ids", [])) - accepted_ids
            if decision["accepted"] and missing:
                decision["accepted"] = False
                decision["reasons"].append(f"rejected patch dependencies: {', '.join(sorted(missing))}")
                accepted_ids.remove(identity)
                changed = True
    effective = copy.deepcopy(candidate)
    effective["patches"] = [copy.deepcopy(patch) for patch in candidate["patches"] if patch["patch_id"] in accepted_ids]
    effective["added_sources"] = _retained_sources(candidate, effective["patches"])
    for patch in effective["patches"]:
        patch.pop("claim_ids", None)
        patch.pop("depends_on_patch_ids", None)
    final_applied = rep.validate_repair(original, effective)
    selected_applied = copy.deepcopy(final_applied)
    selection = {"accepted_patch_ids": sorted(accepted_ids),
                 "final_text_sha256_utf8": final_applied["revised_sha256_utf8"],
                 "final_sources_sha256": sha256_json(final_applied["sources"])}
    final_blockers = []
    if require_final_review:
        final_review = rep._object(guard.get("final_review"), "guard.final_review")
        final_status = rep._enum(final_review.get("status"), {"pass", "fail"}, "guard.final_review.status")
        final_complete = rep._boolean(final_review.get("coverage_complete"), "guard.final_review.coverage_complete")
        _reason(final_review.get("reason"), "guard.final_review.reason")
        final_ids = _refs(final_review.get("accepted_patch_ids"), patches, "guard.final_review.accepted_patch_ids")
        if set(final_ids) != accepted_ids:
            raise ValueError("guard.final_review accepted_patch_ids mismatch with selected report")
        for field in ("final_text_sha256_utf8", "final_sources_sha256"):
            if final_review.get(field) != selection[field]:
                raise ValueError(f"guard.final_review {field} mismatch with selected report")
        if final_status != "pass":
            final_blockers.append("independent final combined-report review failed")
        if not final_complete:
            final_blockers.append("independent final combined-report coverage is incomplete")
        if final_blockers:
            for decision in decisions:
                if decision["accepted"]:
                    decision["accepted"] = False
                    decision["reasons"].extend(final_blockers)
            accepted_ids.clear()
            effective["patches"] = []
            effective["added_sources"] = []
            final_applied = rep.validate_repair(original, effective)
    summary = validate_ledger(original, ledger)
    summary.update({"candidate_patch_count": len(patches), "accepted_patch_count": len(accepted_ids),
                    "rejected_patch_count": len(patches) - len(accepted_ids),
                    "rejected_patches": [{"patch_id": row["patch_id"], "reasons": row["reasons"]}
                                         for row in decisions if not row["accepted"]],
                    "candidate_added_source_count": len(candidate["added_sources"]),
                    "retained_added_source_count": len(effective["added_sources"]),
                    "source_filter_policy": "Keep URLs/numeric tokens referenced by accepted patches; never renumber; ambiguous numeric tokens retain a source.",
                    "rewrite_retries": 0, "final_review_required": require_final_review,
                    "final_review_passed": require_final_review and not final_blockers,
                    "final_review_blockers": final_blockers})
    return {"schema_version": 1, "input_sha256": expected["input_sha256"],
            "guard_packet_sha256": sha256_json(expected), "guard_sha256": sha256_json(guard),
            "candidate_applied": copy.deepcopy(expected["candidate_applied"]),
            "selected_applied": selected_applied, "selection": selection,
            "effective_repair": effective, "final_applied": final_applied,
            "decisions": decisions, "summary": summary}


def preview_repair(original, candidate, ledger, guard, *, guard_packet=None):
    """Produce the deterministic selected report for the guard's final review."""
    return _select_repair(original, candidate, ledger, guard, guard_packet=guard_packet, require_final_review=False)


def gate_repair(original, candidate, ledger, guard, *, guard_packet=None):
    """Apply the gate only after an exact-hash review of the combined selection."""
    return _select_repair(original, candidate, ledger, guard, guard_packet=guard_packet, require_final_review=True)


def schema_contract():
    """Return the versioned, JSON-serializable authoring contract used by workers."""
    return {
        "schema_version": 1,
        "hash_algorithm": "SHA256 of UTF-8 JSON sorted by object keys, ensure_ascii=false, separators=(',', ':'), allow_nan=false; text hashes use exact UTF-8 text.",
        "blocks": "Every nonblank physical line becomes B0001 etc.; offsets count Python Unicode characters, end exclusive; line terminators excluded.",
        "evidence": {"evidence_id": "unique string", "url": "absolute HTTPS URL", "locator": "nonempty string",
                     "summary": "nonempty string", "access": "full_section|partial|unavailable", "primary": "boolean"},
        "ledger": {"schema_version": 1, "block_packet_sha256": "canonical block packet hash", "original_sha256": "canonical original JSON hash",
                   "evidence": "array of evidence", "blocks": [{"block_id": "every packet block exactly once", "claim_ids": "array of claim IDs", "nonfactual_reason": "nonempty iff claim_ids empty; otherwise empty string"}],
                   "claims": [{"claim_id": "unique string", "block_id": "own block", "quote": "nonempty exact substring of own block",
                               "context_block_ids": "known block IDs including own block and relevant headings/table headers/nearby context",
                               "report_scope": "nonempty for factual claims: contextual original claim scope",
                               "source_scope": "nonempty for factual claims: evidenced scope or explicit unknown with reason",
                               "status": "supported|problem|uncertain|nonfactual", "reason": "nonempty string",
                               "checks": {key: {"status": "supported|contradicted|missing_material|unknown|not_applicable", "rationale": "nonempty string", "evidence_ids": "array of known evidence IDs"} for key in CHECKS}}]},
        "ledger_rules": ["All seven checks are mandatory. Supported requires at least one applicable check, all applicable checks supported, and primary full_section evidence for each; citation alone allows secondary full_section evidence.",
                         "Use unknown when scope is unknown; supported must not conceal scope uncertainty. Arithmetic may cite primary input data and explain the calculation in evidence.summary.",
                         "nonfactual requires all checks not_applicable. problem requires at least one contradicted or missing_material check.",
                         "Block claim membership is reciprocal and complete. Quote matches are exact. Complete block coverage is not proof of atomic completeness."],
        "candidate": "Existing replication repair schema, plus mandatory nonempty patch.claim_ids exactly covering every overlapping ledger claim; optional patch.depends_on_patch_ids referencing existing patches with no cycles. Original sources remain unchanged.",
        "guard_packet": "schema_version,input_sha256:{original,candidate,ledger},original,candidate_repair,candidate_applied,ledger,block_packet,limitation; CLI also includes input_file_sha256 for exact raw files. No baseline/gold inputs.",
        "guard": {"schema_version": 1, "guard_packet_sha256": "canonical entire guard packet hash", "input_sha256": {"original": "from packet", "candidate": "from packet", "ledger": "from packet"},
                  "evidence": "array; may be empty and reference ledger evidence; overlapping IDs must be identical",
                  "patch_reviews": [{"patch_id": "each candidate patch exactly once", "necessity": "necessary|optional|unnecessary|uncertain", "rationale": "nonempty independent judgment",
                                     "evidence_ids": "known IDs", "coverage_complete": "boolean; attest all original assertions, including units omitted from ledger", "coverage_reason": "nonempty string",
                                     "additions": [{"after_quote": "nonempty exact substring of patch.after; enumerate every added factual/citation unit beyond the mapped corrections",
                                                    "content_kind": "factual|citation|nonfactual", "claim_ids": "optional known ledger IDs for context", "status": "supported|uncertain|contradicted",
                                                    "evidence_ids": "known IDs", "reason": "nonempty independent assessment", "report_scope": "nonempty unless nonfactual", "source_scope": "nonempty unless nonfactual"}],
                                     "units": [{"unit_id": "unique within patch", "claim_ids": "array of patch claim IDs; collectively map all; may be empty for an additional original unit",
                                                "before_quote": "nonempty exact substring of patch.before, overlapping each referenced claim",
                                                "status": "supported|uncertain|contradicted", "content_kind": "factual|citation|nonfactual (default factual)", "evidence_ids": "known IDs", "reason": "nonempty independent source assessment",
                                                "after_quote": "exact substring of patch.after or empty", "disposition": "preserved|corrected_with_evidence|removed_unsupported|lost|uncertain", "rationale": "nonempty independent explanation"}]}],
                  "final_review": {"status": "pass|fail", "accepted_patch_ids": "exact preview selection IDs", "final_text_sha256_utf8": "from preview selection",
                                   "final_sources_sha256": "from preview selection", "coverage_complete": "boolean", "reason": "nonempty independent review of actual full selected report"}},
        "gate_rules": ["Only necessary patches with complete coverage, all mapped claims, accessible full_section necessity evidence, and at least one source-backed correction/removal survive.",
                       "preserved requires identical before_quote and after_quote; even semantically equivalent paraphrases can be held by this conservative rule.",
                       "Corrected factual units require primary full_section evidence and nonempty changed after_quote. Citation units in relink patches may use secondary full_section evidence. Nonfactual units may be preserved but not source-corrected.",
                       "additions is mandatory even when empty; coverage_reason must attest that all introduced assertions beyond mapped corrections were reviewed, or that none exist. Uncertain/contradicted additions hold the patch. Supported factual additions require primary full_section evidence; citation additions allow secondary full_section evidence.",
                       "Any lost/uncertain disposition holds the patch. Supported or uncertain units/ledger claims cannot be removed. Removal requires a contradicted unprotected unit, evidence and empty after_quote.",
                       "Rejected dependencies cascade. Added sources referenced by accepted replacements or source_urls survive even if introduced for a rejected patch; never renumber. Numeric token ambiguity conservatively retains sources.",
                       "preview omits final review and is provisional. apply requires exact selected text/source hashes and IDs. A failed or incomplete final review holds all patches. No rewrite retry.",
                       "All semantic correctness, necessity, atomic coverage and source interpretation remain reviewer judgments, not code guarantees."],
        "cli": ["blocks --original PATH --output PATH", "describe --output PATH", "prepare --original PATH --candidate PATH --ledger PATH --output-dir DIR",
                "preview --original PATH --candidate PATH --ledger PATH --guard-packet PATH --guard PATH --output-dir DIR",
                "apply --original PATH --candidate PATH --ledger PATH --guard-packet PATH --guard PATH --output-dir DIR"],
        "limitation": LIMITATION,
    }
