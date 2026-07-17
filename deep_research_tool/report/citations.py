"""
Citation manager - stable evidence-backed citations across regeneration.

Citations must survive rewrite / compression / targeted re-research
(spec section 8):

- Each section keeps a STABLE ordered mapping [SOURCE N] -> evidence_id
  built once from the section's evidence list. The canonical key is the
  Evidence Locker id (``Evidence.id``), NOT the URL: URLs are neither
  unique (reposts) nor stable (redirects), while locker ids survive
  re-research and deduplication.
- Rewrites reuse the same mapping; newly researched evidence APPENDS,
  so existing numbers never shift and other sections are never
  renumbered.
- After every LLM edit the citations in the output are machine-checked:
  an unknown [SOURCE N] (no mapping entry / evidence id absent from the
  locker) fails the edit, as does an edit that leaves a citation whose
  claim disappeared (orphan) or deletes every citation from a section
  that previously had them.
- Both ``[SOURCE N]`` and the legacy ``[SOURCE: N]`` spelling follow the
  same rules everywhere (validation, rendering, counting).
- Display numbering happens ONLY at final rendering via
  ``render_numbering``; the pipeline itself never renumbers.
"""

import re
from typing import Dict, List, Optional, Tuple

# Both "[SOURCE 3]" and "[SOURCE: 3]" are citations (item: same rules)
_CITATION_RE = re.compile(r"\[SOURCE:?\s*(\d+)\]")


class CitationManager:
    """Stable per-section citation registry with machine validation.

    Registry values are canonical Evidence ids. ``evidence_ids_exist``
    should check LIVE against the Evidence Locker (no snapshot), e.g.::

        CitationManager(lambda eid: locker.get_evidence(eid) is not None)
    """

    def __init__(self, evidence_ids_exist=None):
        self._sections: Dict[str, List[str]] = {}
        self._exists = evidence_ids_exist or (lambda _id: True)

    # -- registry -----------------------------------------------------

    def register_section(self, section_id: str,
                         evidence_ids: List[str]) -> None:
        """Register the stable citation order for a section (once)."""
        self._sections[section_id] = list(evidence_ids or [])

    def append_evidence(self, section_id: str, evidence_id: str) -> int:
        """Add newly researched evidence; existing numbers stay stable.

        Returns the 1-based [SOURCE N] number assigned to the new item.
        """
        ids = self._sections.setdefault(section_id, [])
        if evidence_id in ids:
            return ids.index(evidence_id) + 1
        ids.append(evidence_id)
        return len(ids)

    def mapping(self, section_id: str) -> Dict[int, str]:
        """{source_number: evidence_id} for one section."""
        return {i + 1: eid
                for i, eid in enumerate(self._sections.get(section_id, []))}

    def evidence_ids(self, section_id: str) -> List[str]:
        return list(self._sections.get(section_id, []))

    def sections(self) -> List[str]:
        return list(self._sections.keys())

    def evidence_id_for(self, section_id: str,
                        source_number: int) -> Optional[str]:
        """Resolve one [SOURCE N] number to its canonical evidence id."""
        return self.mapping(section_id).get(source_number)

    # -- validation -----------------------------------------------------

    @staticmethod
    def cited_numbers(text: str) -> List[int]:
        return [int(m) for m in _CITATION_RE.findall(text or "")]

    def invalid_citations(self, section_id: str, text: str) -> List[int]:
        """Citation numbers that do not resolve to known evidence.

        The evidence-id existence check runs LIVE against the locker via
        the injected callable — evidence deleted after registration is
        detected here.
        """
        known = self.mapping(section_id)
        bad = []
        for n in self.cited_numbers(text):
            eid = known.get(n)
            if eid is None or not self._exists(eid):
                bad.append(n)
        return sorted(set(bad))

    @staticmethod
    def orphan_citation_lines(text: str) -> List[str]:
        """Lines that are citations without a surviving claim.

        Compression must not strip a sentence while keeping its citation:
        a line whose content (minus citation tags and punctuation) is
        effectively empty is an orphan.
        """
        orphans = []
        for line in (text or "").split("\n"):
            if not _CITATION_RE.search(line):
                continue
            rest = _CITATION_RE.sub("", line)
            rest = re.sub(r"[\s。、．，,.\-*#>|]+", "", rest)
            if len(rest) < 8:
                orphans.append(line.strip())
        return orphans

    def validate(self, section_id: str, text: str,
                 previous_text: Optional[str] = None) -> bool:
        """Machine check after any LLM edit; False fails the edit.

        Rejected:
        - citation numbers with no registry entry / dead evidence id
        - orphan citation lines (citation kept, claim deleted)
        - edits that delete EVERY citation from a section that had them
          (an "all citations removed" body is a defect, not a rewrite)
        """
        if self.invalid_citations(section_id, text):
            return False
        if self.orphan_citation_lines(text):
            return False
        if previous_text is not None:
            if self.cited_numbers(previous_text) and \
                    not self.cited_numbers(text):
                return False
        return True

    def validate_report(self, chapters: Dict[str, str]) -> Dict[str, List[int]]:
        """{section_id: invalid citation numbers} across the whole body."""
        problems = {}
        for sid, text in chapters.items():
            bad = self.invalid_citations(sid, text)
            if bad:
                problems[sid] = bad
        return problems

    def report_orphans(self, chapters: Dict[str, str]) -> Dict[str, List[str]]:
        """Report-wide orphan citation lines: {section_id: [lines]}."""
        orphans = {}
        for sid, text in chapters.items():
            lines = self.orphan_citation_lines(text)
            if lines:
                orphans[sid] = lines
        return orphans

    def sections_that_lost_all_citations(
            self, chapters: Dict[str, str]) -> List[str]:
        """Sections with a registered evidence list but zero citations left.

        Only sections that HAVE registered evidence are reported — a
        section that legitimately never cited anything is not a defect.
        """
        lost = []
        for sid, ids in self._sections.items():
            if not ids:
                continue
            text = chapters.get(sid)
            if text is None:
                continue
            if not self.cited_numbers(text):
                lost.append(sid)
        return lost

    def uncited_evidence_ids(self, section_id: str, text: str) -> List[str]:
        """Registered evidence never cited in the section body."""
        cited = set(self.cited_numbers(text or ""))
        return [eid for n, eid in self.mapping(section_id).items()
                if n not in cited]

    # -- rendering -------------------------------------------------------

    def render_numbering(self, chapters: Dict[str, str]
                         ) -> Tuple[Dict[str, str], List[str]]:
        """Deterministic display numbering for the final render ONLY.

        Converts per-section [SOURCE N] / [SOURCE: N] to document-wide
        [n] in first-use order and returns
        (rendered_chapters, ordered_evidence_ids). The ordered id list is
        what the references section must be built from so that [n] in the
        text and entry n in the reference list always agree.
        """
        global_order: List[str] = []

        def number_for(eid: str) -> int:
            if eid not in global_order:
                global_order.append(eid)
            return global_order.index(eid) + 1

        rendered = {}
        for sid, text in chapters.items():
            local = self.mapping(sid)

            def _sub(match):
                eid = local.get(int(match.group(1)))
                if eid is None:
                    return match.group(0)   # left as-is; validation catches it
                return f"[{number_for(eid)}]"

            rendered[sid] = _CITATION_RE.sub(_sub, text or "")
        return rendered, global_order
