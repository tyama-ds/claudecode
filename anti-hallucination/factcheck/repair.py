"""Apply explicit, exact-match report repairs without interpreting their content."""
from __future__ import annotations

import copy
import hashlib
from urllib.parse import urlsplit


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _array(value, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _string(value, name, *, nonempty=False):
    if not isinstance(value, str) or (nonempty and not value.strip()):
        qualifier = "a non-empty string" if nonempty else "a string"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _https_url(value, name):
    url = _string(value, name, nonempty=True)
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url) or "\\" in url:
        raise ValueError(f"{name} must be an HTTPS URL without whitespace or control characters")
    try:
        parts = urlsplit(url)
        valid = (
            parts.scheme == "https"
            and bool(parts.hostname)
            and parts.username is None
            and parts.password is None
        )
        # Accessing port also rejects nonnumeric or out-of-range ports.
        parts.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an absolute HTTPS URL without credentials")
    return url


def _source(source, name, *, added):
    _object(source, name)
    citation_id = source.get("citation_id")
    if isinstance(citation_id, bool) or not isinstance(citation_id, (int, str)):
        raise ValueError(f"{name}.citation_id must be an integer or a non-empty string")
    if isinstance(citation_id, str):
        _string(citation_id, f"{name}.citation_id", nonempty=True)
    _string(source.get("title"), f"{name}.title")
    if added:
        _https_url(source.get("url"), f"{name}.url")
    else:
        _string(source.get("url"), f"{name}.url")
    # Compare the displayed identifier while preserving its original input type
    # in the returned source. Thus numeric 20 and string "20" cannot coexist.
    return str(citation_id)


def _utf8_hash(text, name):
    try:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        raise ValueError(f"{name} must be encodable as UTF-8") from None


def apply_repair(original: dict, repair: dict) -> dict:
    """Return a new repaired text, source list, patch audit and UTF-8 hashes.

    Every patch requires ``patch_id``, ``before``, ``after``, ``finding_ids``,
    ``reason``, ``source_urls`` and ``action``. Metadata is recorded, not
    semantically evaluated. Empty replacements and empty metadata arrays are
    allowed. ``before`` must occur exactly once, including overlapping matches,
    in the *original* text. Patch spans cannot overlap; adjacent spans are valid.

    Returned patches are ordered by their original positions and include
    ``start_char``, ``end_char``, ``after_start_char`` and ``after_end_char``.
    These are zero-based Python string offsets with exclusive end positions,
    not byte offsets or UTF-16 code units. Only exact replacements are made;
    whitespace, line endings and Unicode normalization are otherwise preserved.

    Original sources are retained in order, followed by added sources in input
    order. Citation IDs may be integers (excluding booleans) or non-empty
    strings, retain their input types, and must be unique across both lists
    after conversion to strings (so 20 and "20" collide). Added source URLs and
    patch source URLs must use HTTPS and contain no credentials or whitespace.
    Existing source URLs are retained without imposing that new-URL policy.
    Extra source and patch metadata is preserved. Neither input is modified,
    and returned mutable data does not alias either input. Invalid input raises
    ``ValueError``. This function performs no network access or file writes.
    """
    _object(original, "original")
    _object(repair, "repair")
    text = _string(original.get("text"), "original.text")
    original_hash = _utf8_hash(text, "original.text")
    sources = _array(original.get("sources"), "original.sources")
    added_sources = _array(repair.get("added_sources"), "repair.added_sources")
    patches = _array(repair.get("patches"), "repair.patches")

    citation_ids = set()
    for added, rows, label in (
        (False, sources, "original.sources"),
        (True, added_sources, "repair.added_sources"),
    ):
        for index, source in enumerate(rows):
            citation_id = _source(source, f"{label}[{index}]", added=added)
            if citation_id in citation_ids:
                raise ValueError(f"duplicate citation_id: {citation_id!r}")
            citation_ids.add(citation_id)

    patch_ids = set()
    located = []
    for index, patch in enumerate(patches):
        name = f"repair.patches[{index}]"
        _object(patch, name)
        patch_id = _string(patch.get("patch_id"), f"{name}.patch_id", nonempty=True)
        if patch_id in patch_ids:
            raise ValueError(f"duplicate patch_id: {patch_id!r}")
        patch_ids.add(patch_id)
        before = _string(patch.get("before"), f"{name}.before")
        if not before:
            raise ValueError(f"{name}.before must not be empty")
        _string(patch.get("after"), f"{name}.after")
        _string(patch.get("reason"), f"{name}.reason")
        _string(patch.get("action"), f"{name}.action")
        for item_index, finding_id in enumerate(_array(patch.get("finding_ids"), f"{name}.finding_ids")):
            _string(finding_id, f"{name}.finding_ids[{item_index}]", nonempty=True)
        for item_index, url in enumerate(_array(patch.get("source_urls"), f"{name}.source_urls")):
            _https_url(url, f"{name}.source_urls[{item_index}]")
        start = text.find(before)
        if start < 0:
            raise ValueError(f"{name}.before was not found in original.text")
        if text.find(before, start + 1) >= 0:
            raise ValueError(f"{name}.before is ambiguous in original.text")
        located.append((start, start + len(before), patch))

    located.sort(key=lambda entry: entry[0])
    previous_end = 0
    for start, end, patch in located:
        if start < previous_end:
            raise ValueError(f"patch ranges overlap at patch_id {patch['patch_id']!r}")
        previous_end = end

    pieces = []
    applied = []
    cursor = 0
    offset = 0
    for start, end, patch in located:
        pieces.extend((text[cursor:start], patch["after"]))
        after_start = start + offset
        after_end = after_start + len(patch["after"])
        applied.append({
            **copy.deepcopy(patch),
            "start_char": start,
            "end_char": end,
            "after_start_char": after_start,
            "after_end_char": after_end,
        })
        cursor = end
        offset += len(patch["after"]) - (end - start)
    pieces.append(text[cursor:])
    revised_text = "".join(pieces)
    return {
        "text": revised_text,
        "sources": copy.deepcopy(sources + added_sources),
        "patches": applied,
        "original_sha256_utf8": original_hash,
        "revised_sha256_utf8": _utf8_hash(revised_text, "revised text"),
    }
