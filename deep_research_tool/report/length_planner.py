"""
Length planner - information-unit-based adaptive length ranges.

Length is decided in three stages (spec section 5):

1. BEFORE research: a provisional allocation from the research plan
   (section importance, audience, purpose).
2. AFTER research: per-section recommended ranges recomputed from the
   UNIQUE information units actually collected (claims, numbers,
   comparisons, contradictions, uncertainties). Duplicate sources and
   raw scraped character counts are explicitly NOT information.
3. AFTER drafting: the draft is compared against evidence to decide
   between RESEARCH / REWRITE_FROM_EVIDENCE / COMPRESS_FROM_EVIDENCE /
   ACCEPT (the decision itself lives in finalization.decide*).

The planner never imposes a fixed quota: preferred_body_chars is a soft
wish, hard_min/hard_max are absolute only when the user explicitly set
them, and everything defaults to adaptive ranges with a tolerance band.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Characters of well-written prose one information unit typically needs
CHARS_PER_UNIT_JA = 180
CHARS_PER_UNIT_EN = 280
SECTION_OVERHEAD_CHARS = 300      # intro/transition per section
MIN_SECTION_CHARS = 400           # floor so a section is a paragraph at least
DUP_SIMILARITY = 0.6              # bigram-jaccard threshold for duplicates


def _jaccard_bigram(a: str, b: str) -> float:
    """Character-bigram Jaccard similarity (CJK-safe)."""
    a, b = (a or "").strip(), (b or "").strip()
    if len(a) < 2 or len(b) < 2:
        return 1.0 if a == b else 0.0
    sa = {a[i:i + 2] for i in range(len(a) - 1)}
    sb = {b[i:i + 2] for i in range(len(b) - 1)}
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def dedupe_units(units: List[str]) -> List[str]:
    """Collapse near-identical statements: reposts count once."""
    unique: List[str] = []
    for u in units:
        u = (u or "").strip()
        if not u:
            continue
        if any(_jaccard_bigram(u, seen) > DUP_SIMILARITY for seen in unique):
            continue
        unique.append(u)
    return unique


_NUMBER_RE = re.compile(r"\d[\d,.]*\s*(?:%|％|億|兆|万|円|ドル|人|件|台|年)?")

# Information units beyond key points and numbers: comparisons,
# contradictions, uncertainties and causal explanations all take prose
# space and are counted as units (capped per section).
_COMPARISON_RE = re.compile(
    r"比較|対比|に対して|より[も高低多少大小]|上回|下回|versus|vs\.?|compared",
    re.IGNORECASE)
_CONTRADICTION_RE = re.compile(
    r"一方で?|矛盾|相反|逆に|とは異なり|however|in contrast|contradict",
    re.IGNORECASE)
_UNCERTAINTY_RE = re.compile(
    r"不確実|不透明|不明|未確定|可能性|見込み|推定|とみられる|uncertain|estimated|likely",
    re.IGNORECASE)
_CAUSALITY_RE = re.compile(
    r"のため|ことにより|要因|背景には|起因|につながる|therefore|because|due to|leads? to",
    re.IGNORECASE)
_UNIT_PATTERN_CAP = 6      # per-section cap per pattern type


@dataclass
class SectionUnits:
    """Unique information units collected for one section."""
    section_id: str
    key_points: List[str] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)
    comparisons: int = 0
    contradictions: int = 0
    uncertainties: int = 0
    causal_links: int = 0
    unique_sources: int = 0

    @property
    def total_units(self) -> int:
        return (len(self.key_points) + len(self.numbers)
                + self.comparisons + self.contradictions
                + self.uncertainties + self.causal_links)


@dataclass
class SectionLengthPlan:
    section_id: str
    importance: str = "normal"          # high / normal / low
    provisional_chars: int = 0          # stage 1
    recommended_min_chars: int = 0      # stage 2
    recommended_chars: int = 0
    recommended_max_chars: int = 0
    units: int = 0

    def to_dict(self):
        return {
            "section_id": self.section_id,
            "importance": self.importance,
            "provisional_chars": self.provisional_chars,
            "recommended_min_chars": self.recommended_min_chars,
            "recommended_chars": self.recommended_chars,
            "recommended_max_chars": self.recommended_max_chars,
            "units": self.units,
        }


IMPORTANCE_WEIGHT = {"high": 1.5, "normal": 1.0, "low": 0.6}


class LengthPlanner:
    """Adaptive, information-driven length planning."""

    def __init__(
        self,
        language: str = "ja",
        length_mode: str = "adaptive",
        preferred_body_chars: Optional[int] = None,
        hard_min_body_chars: Optional[int] = None,
        hard_max_body_chars: Optional[int] = None,
        length_tolerance: float = 0.20,
    ):
        self.language = language
        self.length_mode = (length_mode or "adaptive").lower()
        self.preferred_body_chars = preferred_body_chars
        self.hard_min_body_chars = hard_min_body_chars
        self.hard_max_body_chars = hard_max_body_chars
        self.length_tolerance = max(0.0, float(length_tolerance))
        self.chars_per_unit = (CHARS_PER_UNIT_JA if language == "ja"
                               else CHARS_PER_UNIT_EN)
        self.plans: Dict[str, SectionLengthPlan] = {}

    # ------------------------------------------------------------------
    # Stage 1: provisional allocation before research
    # ------------------------------------------------------------------

    def initial_allocation(
        self,
        sections: List[Dict],
    ) -> Dict[str, SectionLengthPlan]:
        """Distribute a provisional budget by section importance.

        sections: [{"section": id, "title": ..., "importance": optional}]
        With no preferred total, a neutral per-section default is used —
        it is provisional and gets replaced by evidence-based ranges.
        """
        n = max(1, len(sections))
        default_total = (1500 if self.language == "ja" else 2500) * n
        total = self.preferred_body_chars or default_total

        weights = {}
        for s in sections:
            sid = s.get("section", "")
            importance = s.get("importance") or self._guess_importance(s)
            weights[sid] = (IMPORTANCE_WEIGHT.get(importance, 1.0), importance)
        weight_sum = sum(w for w, _ in weights.values()) or 1.0

        self.plans = {}
        for s in sections:
            sid = s.get("section", "")
            w, importance = weights[sid]
            chars = int(total * w / weight_sum)
            self.plans[sid] = SectionLengthPlan(
                section_id=sid,
                importance=importance,
                provisional_chars=max(MIN_SECTION_CHARS, chars),
            )
        return self.plans

    @staticmethod
    def _guess_importance(section: Dict) -> str:
        """Heuristic importance when the plan does not specify one."""
        sid = str(section.get("section", ""))
        if "." in sid:
            return "normal"       # subsection
        return "high"             # top-level chapters carry the report

    # ------------------------------------------------------------------
    # Stage 2: evidence-based recomputation after research
    # ------------------------------------------------------------------

    def extract_units(self, section_id: str,
                      extracted_content: List[Dict]) -> SectionUnits:
        """Unique information units from a section's evidence.

        Duplicate statements (reposts, quotes of the same fact) collapse
        into one unit; the number of sources or the raw text volume is
        never used as an information measure.
        """
        key_points: List[str] = []
        numbers: List[str] = []
        contents: List[str] = []
        for ec in extracted_content or []:
            key_points.extend(ec.get("key_points") or [])
            content = (ec.get("content") or "")
            contents.append(content)
            numbers.extend(_NUMBER_RE.findall(content)[:20])

        unique_points = dedupe_units(key_points)
        unique_numbers = sorted({n.strip() for n in numbers
                                 if len(n.strip()) >= 2})[:15]

        # unique sources: near-identical source contents count once
        unique_source_contents = dedupe_units([c[:400] for c in contents])

        # comparisons / contradictions / uncertainties / causal links are
        # information units too — each needs explanatory prose
        all_text = "\n".join(unique_points) + "\n" + \
            "\n".join(c[:2000] for c in contents)
        return SectionUnits(
            section_id=section_id,
            key_points=unique_points,
            numbers=unique_numbers,
            comparisons=min(_UNIT_PATTERN_CAP,
                            len(_COMPARISON_RE.findall(all_text))),
            contradictions=min(_UNIT_PATTERN_CAP,
                               len(_CONTRADICTION_RE.findall(all_text))),
            uncertainties=min(_UNIT_PATTERN_CAP,
                              len(_UNCERTAINTY_RE.findall(all_text))),
            causal_links=min(_UNIT_PATTERN_CAP,
                             len(_CAUSALITY_RE.findall(all_text))),
            unique_sources=len(unique_source_contents),
        )

    def recalc_after_research(
        self,
        section_units: Dict[str, SectionUnits],
    ) -> Dict[str, SectionLengthPlan]:
        """Recompute per-section recommended ranges from unique units."""
        raw: Dict[str, int] = {}
        for sid, units in section_units.items():
            need = (units.total_units * self.chars_per_unit
                    + SECTION_OVERHEAD_CHARS)
            raw[sid] = max(MIN_SECTION_CHARS, need)

        # Soft-fit to the preferred total when one was given: scale, but
        # never below what the information actually needs at minimum
        scale = 1.0
        total_raw = sum(raw.values()) or 1
        if self.preferred_body_chars:
            scale = self.preferred_body_chars / total_raw
            scale = max(0.5, min(2.0, scale))    # soft, not a hard quota

        tol = self.length_tolerance
        for sid, need in raw.items():
            plan = self.plans.get(sid) or SectionLengthPlan(section_id=sid)
            rec = int(need * scale)
            plan.recommended_chars = rec
            plan.recommended_min_chars = int(rec * (1 - tol))
            plan.recommended_max_chars = int(rec * (1 + tol))
            plan.units = section_units[sid].total_units
            self.plans[sid] = plan
        return self.plans

    # ------------------------------------------------------------------
    # Rebalancing (spec section 6: shift budget between sections)
    # ------------------------------------------------------------------

    def rebalance(self) -> Dict[str, SectionLengthPlan]:
        """Shift budget from low-importance to high-importance sections
        without changing the overall total."""
        highs = [p for p in self.plans.values() if p.importance == "high"]
        lows = [p for p in self.plans.values() if p.importance == "low"]
        if not highs or not lows:
            return self.plans
        transfer = 0
        for p in lows:
            give = int(p.recommended_chars * 0.15)
            p.recommended_chars -= give
            p.recommended_min_chars = int(p.recommended_chars
                                          * (1 - self.length_tolerance))
            p.recommended_max_chars = int(p.recommended_chars
                                          * (1 + self.length_tolerance))
            transfer += give
        per_high = transfer // len(highs)
        for p in highs:
            p.recommended_chars += per_high
            p.recommended_min_chars = int(p.recommended_chars
                                          * (1 - self.length_tolerance))
            p.recommended_max_chars = int(p.recommended_chars
                                          * (1 + self.length_tolerance))
        return self.plans

    # ------------------------------------------------------------------
    # Document-level range
    # ------------------------------------------------------------------

    def document_range(self) -> Dict[str, Optional[int]]:
        rec = sum(p.recommended_chars for p in self.plans.values())
        rec_min = sum(p.recommended_min_chars for p in self.plans.values())
        rec_max = sum(p.recommended_max_chars for p in self.plans.values())
        return {
            "recommended_min_chars": rec_min,
            "recommended_chars": rec,
            "recommended_max_chars": rec_max,
            "preferred_body_chars": self.preferred_body_chars,
            "hard_min_body_chars": self.hard_min_body_chars,
            "hard_max_body_chars": self.hard_max_body_chars,
        }

    def fixed_quota(self) -> Optional[int]:
        """fixed mode: legacy behavior wants an explicit total; adaptive
        mode never returns a quota."""
        if self.length_mode == "fixed":
            return self.preferred_body_chars
        return None
