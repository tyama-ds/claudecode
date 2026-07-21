"""
Claim-level full-text verifier.

Replaces the head-truncated verification (first 6,000 chars of the body,
first 20 evidence items at 4,000 chars) with per-section, per-claim
processing:

- every section is verified, chunked so no text is silently dropped
  (claims after the 6,000th character ARE extracted);
- claims carry stable ids (C-1, C-2, ...) and importance levels;
- EVIDENCE IS CHUNKED TOO: every evidence body is split into ~900-char
  chunks and the search runs over ALL chunks of ALL evidence, so a fact
  on the 5th page of a long source is found (no 800/1,500-char head
  caps). Selected chunks keep their provenance (evidence id + character
  offset).
- a claim that actually cites evidence ([SOURCE N]) is judged FIRST
  against the evidence it cites, then against other relevant chunks;
- unsupported / contradicted / uncertain are counted SEPARATELY, and an
  unknown judge status maps to uncertain (never silently supported);
- "supported" with zero valid supporting sources is downgraded;
- the verifier is FAIL-CLOSED: zero claims extracted from a factual
  body, or wholesale extraction failure, is a verification FAILURE
  (score 0.0, verification_failed=True), never a perfect score;
- extraction errors and chunk success/failure counts are recorded in
  the metrics;
- critical-question coverage distinguishes "answered" from "the words
  merely appear", covers ALL sections, and produces
  ISSUE_UNANSWERED_QUESTION entries with non-empty search queries;
- primary/freshness is a 3-state gate: pass / fail / not_required.

The LLM only produces structured judgements; every aggregation and
threshold lives in Python (finalization.decide).
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..utils.helpers import extract_json_from_response
from ..report.finalization import (
    FRESHNESS_FAIL,
    FRESHNESS_NOT_REQUIRED,
    FRESHNESS_PASS,
    ISSUE_CONTRADICTED,
    ISSUE_INVALID_CITATION,
    ISSUE_ORPHAN_CITATION,
    ISSUE_REDUNDANCY,
    ISSUE_ALL_CITATIONS_DELETED,
    ISSUE_STALE_OR_NON_PRIMARY,
    ISSUE_UNANSWERED_QUESTION,
    ISSUE_UNCERTAIN,
    ISSUE_UNCITED_CLAIM,
    ISSUE_UNSUPPORTED,
    ISSUE_VERIFICATION_FAILURE,
    SectionAssessment,
    StructuredVerdict,
    VerificationIssue,
    VerificationMetrics,
    count_body_chars,
    decide_section_action,
)
from ..report.length_planner import _jaccard_bigram

# Section text is processed in chunks of this size so that long chapters
# are fully covered without overloading a single prompt
CLAIM_CHUNK_CHARS = 4000
# Evidence bodies are chunked at this size for retrieval; the whole text
# of every evidence item is searchable, not just its head
EVIDENCE_CHUNK_CHARS = 900
MAX_CHUNKS_PER_EVIDENCE = 60
EVIDENCE_PER_CLAIM = 5          # top-k relevant evidence chunks per claim
MAX_CHUNKS_PER_SOURCE_IN_JUDGE = 2
IMPORTANCE_WEIGHT = {"critical": 3.0, "important": 2.0, "minor": 1.0}

_VALID_STATUSES = {"supported", "unsupported", "contradicted", "uncertain"}

# body below this (after markup removal) is considered non-factual filler
MIN_FACTUAL_BODY_CHARS = 300


def _bigrams(text: str) -> set:
    text = (text or "").strip()
    return {text[i:i + 2] for i in range(len(text) - 1)} if len(text) > 1 \
        else ({text} if text else set())


def _jaccard_sets(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _containment(needle: set, haystack: set) -> float:
    """Fraction of the needle's bigrams present in the haystack."""
    if not needle:
        return 0.0
    return len(needle & haystack) / len(needle)


@dataclass
class EvidenceChunk:
    """One retrievable slice of an evidence body, with provenance."""
    evidence_id: str
    offset: int                 # character offset in the source text
    text: str
    title: str = ""
    url: str = ""
    _bigram_cache: Optional[set] = field(default=None, repr=False)

    @property
    def id(self) -> str:        # evidence id (for callers scoring by source)
        return self.evidence_id

    @property
    def provenance(self) -> str:
        return f"{self.evidence_id}@{self.offset}"

    def bigrams(self) -> set:
        if self._bigram_cache is None:
            self._bigram_cache = _bigrams(self.text)
        return self._bigram_cache


@dataclass
class Claim:
    claim_id: str
    section_id: str
    text: str
    importance: str = "important"      # critical / important / minor
    status: str = "supported"          # supported/unsupported/contradicted/uncertain
    reason: str = ""
    supporting_source_ids: List[str] = field(default_factory=list)
    # [SOURCE N] numbers the extraction saw next to this claim; None when
    # the extraction did not report the field (cannot distinguish
    # "uncited" from "unreported")
    cited_source_numbers: Optional[List[int]] = None
    # (evidence_id, char_offset) pairs of the chunks the judgement used
    evidence_provenance: List[Tuple[str, int]] = field(default_factory=list)
    # citation integrity (cited-first verification)
    citation_mismatch: bool = False    # cited evidence does NOT support
    citation_missing: bool = False     # critical/important claim w/o citation
    # uncited evidence that DOES support the claim — used ONLY as
    # citation-replacement candidates for the rewrite, never as support
    replacement_source_ids: List[str] = field(default_factory=list)


class ClaimVerifier:
    """Full-text, claim-level verification against the evidence locker."""

    EXTRACT_RETRIES = 2      # extra attempts per extraction chunk

    def __init__(self, llm_client, language: str = "ja",
                 evidence_per_claim: int = EVIDENCE_PER_CLAIM,
                 chunk_chars: int = CLAIM_CHUNK_CHARS,
                 evidence_chunk_chars: int = EVIDENCE_CHUNK_CHARS):
        self.llm = llm_client
        self.language = language
        self.evidence_per_claim = evidence_per_claim
        self.chunk_chars = chunk_chars
        self.evidence_chunk_chars = evidence_chunk_chars
        # per-verify_report accounting (fail-closed bookkeeping)
        self.extraction_errors: List[str] = []
        self.chunks_total = 0
        self.chunks_failed = 0

    # ------------------------------------------------------------------
    # Claim extraction (whole section, chunked — no head truncation)
    # ------------------------------------------------------------------

    def extract_claims(self, section_id: str, text: str,
                       start_index: int = 0) -> List[Claim]:
        claims: List[Claim] = []
        chunks = self._chunk(text)
        counter = start_index
        for ci, chunk in enumerate(chunks):
            self.chunks_total += 1
            prompt = self._claim_prompt(section_id, chunk, ci + 1, len(chunks))
            data = None
            last_error = None
            # bounded per-chunk retry: a transient LLM failure must not
            # silently leave part of the body unverified
            for _attempt in range(self.EXTRACT_RETRIES + 1):
                try:
                    response = self.llm.generate(prompt)
                    data = extract_json_from_response(response.content)
                    break
                except Exception as e:
                    last_error = e
            if data is None:
                self.chunks_failed += 1
                self.extraction_errors.append(
                    f"claim extraction failed after "
                    f"{self.EXTRACT_RETRIES + 1} attempts ({section_id} "
                    f"chunk {ci + 1}/{len(chunks)}, chars "
                    f"{ci * self.chunk_chars}-"
                    f"{min(len(text), (ci + 1) * self.chunk_chars)}): "
                    f"{last_error}")
                print(f"[ClaimVerifier] claim extraction failed "
                      f"({section_id} chunk {ci + 1}): {last_error}")
                continue
            for item in data.get("claims", []):
                if not isinstance(item, dict):
                    continue
                claim_text = (item.get("claim") or "").strip()
                if not claim_text:
                    continue
                counter += 1
                importance = item.get("importance", "important")
                if importance not in IMPORTANCE_WEIGHT:
                    importance = "important"
                cited = None
                if "source_numbers" in item:
                    raw = item.get("source_numbers") or []
                    cited = [int(n) for n in raw
                             if isinstance(n, (int, str))
                             and str(n).isdigit()]
                claims.append(Claim(
                    claim_id=f"C-{counter}",
                    section_id=section_id,
                    text=claim_text,
                    importance=importance,
                    cited_source_numbers=cited,
                ))
        return claims

    def _chunk(self, text: str) -> List[str]:
        text = text or ""
        if len(text) <= self.chunk_chars:
            return [text] if text.strip() else []
        chunks = []
        pos = 0
        while pos < len(text):
            chunks.append(text[pos:pos + self.chunk_chars])
            pos += self.chunk_chars
        return chunks

    def _claim_prompt(self, section_id, chunk, part, total) -> str:
        if self.language == "ja":
            return f"""以下はレポートのセクション{section_id}の本文です（{part}/{total}分割）。
検証が必要な事実主張（数値・日付・固有名詞・因果関係・比較を含む記述）を抽出してください。

【本文】
{chunk}

JSONで回答:
{{"claims": [{{"claim": "主張の要約（原文の内容を保持）",
  "importance": "critical/important/minor",
  "source_numbers": [その主張の近くで引用されている [SOURCE N] のN。無ければ空リスト]}}]}}

- critical: レポートの結論を左右する主張
- important: 章の内容を支える主張
- minor: 補足的な記述
- 意見・一般論は含めない。JSON以外は出力しない。"""
        return f"""Below is section {section_id} of a report (part {part}/{total}).
Extract factual claims that require verification (statements with numbers,
dates, named entities, causal links, comparisons).

[TEXT]
{chunk}

Respond as JSON:
{{"claims": [{{"claim": "the claim", "importance": "critical/important/minor",
  "source_numbers": [the N of [SOURCE N] citations next to the claim, or []]}}]}}

Do not include opinions or generic statements. JSON only."""

    # ------------------------------------------------------------------
    # Deterministic claim -> [SOURCE N] association
    # ------------------------------------------------------------------

    CITATION_TAG_RE = re.compile(r"\[SOURCE:?\s*(\d+)\]")

    @classmethod
    def parse_cited_numbers(cls, claim_text: str,
                            section_text: str) -> List[int]:
        """Deterministically associate a claim with the [SOURCE N]
        citations of its best-matching sentence (fallback: paragraph).

        This parser — not the LLM — is the primary authority on which
        citations a claim carries; the LLM-reported numbers are merged in
        as a secondary signal.
        """
        text = section_text or ""
        if not text or not claim_text:
            return []
        claim_bi = _bigrams(claim_text)
        best_sent, best_para, best_score = "", "", 0.0
        for para in text.split("\n\n"):
            for sent in re.split(r"(?<=[。．！？!?.])\s*", para):
                if len(sent.strip()) < 8:
                    continue
                score = _containment(claim_bi, _bigrams(sent))
                if score > best_score:
                    best_score, best_sent, best_para = score, sent, para
        if best_score < 0.3:
            return []
        nums = [int(n) for n in cls.CITATION_TAG_RE.findall(best_sent)]
        if not nums:
            nums = [int(n) for n in cls.CITATION_TAG_RE.findall(best_para)]
        return sorted(set(nums))

    # ------------------------------------------------------------------
    # Evidence chunk index (whole locker, whole texts)
    # ------------------------------------------------------------------

    def build_chunk_index(self, evidence_list: List) -> List[EvidenceChunk]:
        """Split every evidence body into retrievable chunks."""
        index: List[EvidenceChunk] = []
        step = self.evidence_chunk_chars
        for ev in evidence_list:
            text = (getattr(ev, "extracted_text", "") or
                    getattr(ev, "content_excerpt", "") or "")
            eid = getattr(ev, "id", "") or ""
            title = getattr(ev, "title", "") or ""
            url = getattr(ev, "url", "") or ""
            if not text.strip():
                continue
            count = 0
            for pos in range(0, len(text), step):
                index.append(EvidenceChunk(
                    evidence_id=eid, offset=pos,
                    text=text[pos:pos + step + 100],   # slight overlap
                    title=title, url=url))
                count += 1
                if count >= MAX_CHUNKS_PER_EVIDENCE:
                    break
        return index

    # ------------------------------------------------------------------
    # Evidence selection from the WHOLE locker (chunked, no head caps)
    # ------------------------------------------------------------------

    def select_evidence(self, claim: Claim, evidence_list: List,
                        chunk_index: Optional[List[EvidenceChunk]] = None,
                        cited_evidence_ids: Optional[List[str]] = None,
                        ) -> List[EvidenceChunk]:
        """Most relevant evidence chunks for this claim.

        Chunks of the evidence the claim actually CITES come first
        (verification checks the cited sources before anything else);
        the remaining slots are filled by the best-scoring chunks from
        the entire locker.
        """
        if chunk_index is None:
            chunk_index = self.build_chunk_index(evidence_list)

        claim_bi = _bigrams(claim.text)
        numbers = re.findall(r"\d[\d,.]*", claim.text)[:5]

        def score(chunk: EvidenceChunk) -> float:
            s = _jaccard_sets(claim_bi, chunk.bigrams())
            if chunk.offset == 0 and chunk.title:
                s = max(s, _jaccard_bigram(claim.text, chunk.title) * 0.8)
            for token in numbers:
                if token and token in chunk.text:
                    s += 0.15
            return s

        scored = sorted(((score(c), c) for c in chunk_index),
                        key=lambda pair: pair[0], reverse=True)

        selected: List[EvidenceChunk] = []
        per_source: Dict[str, int] = {}

        # 1) cited evidence first: best chunk of each cited source
        cited = [c for c in (cited_evidence_ids or []) if c]
        for eid in cited:
            best = None
            best_score = -1.0
            for s, c in scored:
                if c.evidence_id == eid and s > best_score:
                    best, best_score = c, s
            if best is not None:
                selected.append(best)
                per_source[eid] = per_source.get(eid, 0) + 1

        # 2) fill with the best remaining chunks across the whole locker
        for s, c in scored:
            if len(selected) >= self.evidence_per_claim:
                break
            if s <= 0.02:
                break
            if c in selected:
                continue
            if per_source.get(c.evidence_id, 0) >= \
                    MAX_CHUNKS_PER_SOURCE_IN_JUDGE:
                continue
            selected.append(c)
            per_source[c.evidence_id] = per_source.get(c.evidence_id, 0) + 1

        return selected[:max(self.evidence_per_claim, len(cited))]

    # ------------------------------------------------------------------
    # Per-claim verification
    # ------------------------------------------------------------------

    def verify_claims(self, claims: List[Claim], evidence_list: List,
                      chunk_index: Optional[List[EvidenceChunk]] = None,
                      cited_ids_by_claim: Optional[Dict[str, List[str]]] = None,
                      enforce_citations: bool = False,
                      ) -> List[Claim]:
        """Judge each claim against ITS OWN evidence — CITED FIRST.

        With ``enforce_citations`` (the production path, i.e. whenever a
        citation registry exists):
        - a claim WITH citations is judged first against ONLY the
          evidence it cites; "supported" requires at least one id that is
          BOTH cited in the body AND judged to actually support it;
        - cited-but-unsupporting is a citation mismatch — uncited
          evidence that happens to support the claim NEVER turns it into
          a pass; it is recorded only as a replacement candidate for the
          rewrite;
        - critical/important factual claims WITHOUT citations are never
          supported (citation required, regardless of registry size).
        Without ``enforce_citations`` (no registry available), the legacy
        whole-locker judgement applies.
        """
        if chunk_index is None:
            chunk_index = self.build_chunk_index(evidence_list)
        cited_ids_by_claim = cited_ids_by_claim or {}
        for claim in claims:
            cited_ids = cited_ids_by_claim.get(claim.claim_id) or []
            if enforce_citations and cited_ids:
                self._verify_cited_claim(claim, cited_ids, chunk_index)
            elif enforce_citations and claim.importance in ("critical",
                                                            "important"):
                claim.status = "unsupported"
                claim.citation_missing = True
                claim.reason = (
                    "引用がありません（critical/important の事実主張には"
                    "引用が必須です）" if self.language == "ja" else
                    "no citation (critical/important factual claims "
                    "require one)")
                claim.supporting_source_ids = []
                self._find_replacements(claim, chunk_index, exclude=set())
            else:
                self._verify_open_claim(claim, evidence_list, chunk_index,
                                        cited_ids)
        return claims

    def _verify_open_claim(self, claim, evidence_list, chunk_index,
                           cited_ids) -> None:
        """Legacy whole-locker judgement (minor claims / no registry)."""
        selected = self.select_evidence(
            claim, evidence_list, chunk_index=chunk_index,
            cited_evidence_ids=cited_ids)
        claim.evidence_provenance = [
            (c.evidence_id, c.offset) for c in selected]
        claim.supporting_source_ids = list(dict.fromkeys(
            c.evidence_id for c in selected))
        if not selected:
            claim.status = "unsupported"
            claim.reason = ("関連するエビデンスが見つかりません"
                            if self.language == "ja"
                            else "no relevant evidence found")
            return
        try:
            verdict = self._judge_claim(claim, selected)
        except Exception as e:
            claim.status = "uncertain"
            claim.reason = f"verification error: {e}"
            self.extraction_errors.append(
                f"claim judgement failed ({claim.claim_id}): {e}")
            return
        status = verdict.get("status", "uncertain")
        if status not in _VALID_STATUSES:
            status = "uncertain"     # unknown status is never "fine"
        claim.status = status
        claim.reason = verdict.get("reason", "")
        ids = verdict.get("supporting_source_ids")
        valid = {c.evidence_id for c in selected}
        if isinstance(ids, list):
            claim.supporting_source_ids = [i for i in ids if i in valid]
        # "supported" with zero valid supporting sources is a
        # contradiction in terms -> downgrade to uncertain
        if claim.status == "supported" and not claim.supporting_source_ids:
            claim.status = "uncertain"
            claim.reason = (claim.reason + " / 有効な支持ソースなし"
                            if self.language == "ja"
                            else claim.reason + " / no valid sources")

    def _verify_cited_claim(self, claim, cited_ids, chunk_index) -> None:
        """Cited-first judgement: ONLY the evidence the body cites."""
        cited_set = set(cited_ids)
        cited_chunks = [c for c in chunk_index if c.evidence_id in cited_set]
        selected = self.select_evidence(
            claim, [], chunk_index=cited_chunks,
            cited_evidence_ids=cited_ids)
        claim.evidence_provenance = [
            (c.evidence_id, c.offset) for c in selected]
        if not selected:
            claim.status = "unsupported"
            claim.citation_mismatch = True
            claim.supporting_source_ids = []
            claim.reason = ("引用された出典に該当する本文が見つかりません"
                            if self.language == "ja" else
                            "cited evidence has no matching content")
            self._find_replacements(claim, chunk_index, exclude=cited_set)
            return
        try:
            verdict = self._judge_claim(claim, selected)
        except Exception as e:
            claim.status = "uncertain"
            claim.reason = f"verification error: {e}"
            self.extraction_errors.append(
                f"claim judgement failed ({claim.claim_id}): {e}")
            return
        status = verdict.get("status", "uncertain")
        if status not in _VALID_STATUSES:
            status = "uncertain"
        ids = verdict.get("supporting_source_ids")
        valid = {c.evidence_id for c in selected}
        supporting = [i for i in ids if i in valid] \
            if isinstance(ids, list) else []
        # supported REQUIRES: cited-in-body ∩ actually-supporting ≠ ∅
        if status == "supported" and any(i in cited_set for i in supporting):
            claim.status = "supported"
            claim.supporting_source_ids = supporting
            claim.reason = verdict.get("reason", "")
            return
        claim.supporting_source_ids = []
        claim.reason = verdict.get("reason", "")
        if status in ("unsupported", "contradicted"):
            # the citation is real but its content does NOT support the
            # claim -> invalid/unsupported citation
            claim.status = status
            claim.citation_mismatch = True
            if not claim.reason:
                claim.reason = ("引用された出典は主張を支持しません"
                                if self.language == "ja" else
                                "the cited evidence does not support "
                                "the claim")
            self._find_replacements(claim, chunk_index, exclude=cited_set)
        else:
            # uncertain / unknown / supported-without-valid-ids
            claim.status = "uncertain"
            claim.reason = (claim.reason + " / 有効な支持ソースなし"
                            if self.language == "ja"
                            else claim.reason + " / no valid sources")

    def _find_replacements(self, claim, chunk_index, exclude) -> None:
        """Search UNCITED evidence for citation-replacement candidates.

        The result is stored on the claim for the rewrite step only; it
        never changes the claim's verification status (a misattributed
        citation must not pass because some other source agrees).
        """
        other = [c for c in chunk_index if c.evidence_id not in exclude]
        selected = self.select_evidence(claim, [], chunk_index=other)
        if not selected:
            return
        try:
            verdict = self._judge_claim(claim, selected)
        except Exception:
            return
        if verdict.get("status") == "supported":
            valid = {c.evidence_id for c in selected}
            ids = verdict.get("supporting_source_ids")
            if isinstance(ids, list):
                claim.replacement_source_ids = [
                    i for i in ids if i in valid]

    def _judge_claim(self, claim: Claim, chunks: List[EvidenceChunk]) -> Dict:
        blocks = []
        for c in chunks:
            pos = ("冒頭" if c.offset == 0 else f"{c.offset}字目以降") \
                if self.language == "ja" else \
                ("start" if c.offset == 0 else f"offset {c.offset}")
            blocks.append(f"[{c.evidence_id}] {c.title}（{pos}）\n{c.text}")
        evidence_text = "\n---\n".join(blocks)

        if self.language == "ja":
            prompt = f"""次の主張を、提示されたエビデンスだけで検証してください。

【主張】（{claim.importance}）
{claim.text}

【エビデンス】
{evidence_text}

JSONで回答:
{{"status": "supported/unsupported/contradicted/uncertain",
 "reason": "判定理由（1文）",
 "supporting_source_ids": ["支持するエビデンスのID"]}}

- supported: エビデンスが主張を実際に支持している
- unsupported: どのエビデンスも主張を支持しない
- contradicted: エビデンスが主張と矛盾する
- uncertain: 判断材料が不十分
JSON以外は出力しない。"""
        else:
            prompt = f"""Verify the claim using ONLY the evidence provided.

[CLAIM] ({claim.importance})
{claim.text}

[EVIDENCE]
{evidence_text}

Respond as JSON:
{{"status": "supported/unsupported/contradicted/uncertain",
 "reason": "one sentence",
 "supporting_source_ids": ["ids of supporting evidence"]}}
JSON only."""
        response = self.llm.generate(prompt)
        return extract_json_from_response(response.content)

    # ------------------------------------------------------------------
    # Critical-question coverage (all sections; answered vs mentioned)
    # ------------------------------------------------------------------

    # The limitations block QUOTES unresolved questions verbatim — it must
    # never make a question look answered (lexically or to the LLM)
    _LIMITATIONS_BLOCK_RE = re.compile(
        r"###\s*(?:調査上の限界|Research Limitations)[\s\S]*\Z")

    @classmethod
    def _strip_limitations(cls, text: str) -> str:
        return cls._LIMITATIONS_BLOCK_RE.sub("", text or "")

    def judge_coverage(self, questions: List[str],
                       chapters: Dict[str, str]) -> List[Dict]:
        """LLM-judged coverage with a lexical fallback.

        Returns one dict per question:
        {"question", "answered", "section_id", "missing", "search_queries"}
        """
        # a limitations note about an UNRESOLVED question is not an answer
        chapters = {sid: self._strip_limitations(text)
                    for sid, text in chapters.items()}
        results = []
        llm_results = None
        try:
            llm_results = self._judge_coverage_llm(questions, chapters)
        except Exception as e:
            self.extraction_errors.append(f"coverage judgement failed: {e}")

        # lexical fallback data: FULL body of EVERY section (no 20k cap)
        section_bigrams = {sid: _bigrams(text)
                           for sid, text in chapters.items()}

        for qi, q in enumerate(questions):
            entry = None
            if llm_results:
                entry = llm_results.get(qi)
            if entry is None:
                # fallback: containment of the question's bigrams in each
                # section — a high overlap approximates "answered", a low
                # one is at best a mention
                q_bi = _bigrams(q)
                best_sid, best = "", 0.0
                for sid, bi in section_bigrams.items():
                    c = _containment(q_bi, bi)
                    if c > best:
                        best_sid, best = sid, c
                entry = {
                    "answered": best >= 0.6,
                    "section_id": best_sid,
                    "missing": "" if best >= 0.6 else q,
                    "search_queries": [q[:60]],
                }
            elif not entry.get("answered"):
                # lexical CROSS-CHECK of an LLM "unanswered" verdict: a
                # very strong lexical presence in some section suggests
                # the LLM missed it (e.g. window boundaries) — overturn
                # and note the disagreement
                q_bi = _bigrams(q)
                best_sid, best = "", 0.0
                for sid, bi in section_bigrams.items():
                    c = _containment(q_bi, bi)
                    if c > best:
                        best_sid, best = sid, c
                if best >= 0.7:
                    self.extraction_errors.append(
                        f"coverage cross-check: LLM marked unanswered but "
                        f"lexical containment {best:.2f} in section "
                        f"{best_sid} — treated as answered")
                    entry = {"answered": True, "section_id": best_sid,
                             "missing": "", "search_queries": []}
            queries = [s for s in (entry.get("search_queries") or []) if s]
            if not queries:
                queries = [q[:60]]
            results.append({
                "question": q,
                "answered": bool(entry.get("answered")),
                "section_id": entry.get("section_id", "") or "",
                "missing": entry.get("missing", "") or "",
                "search_queries": queries,
            })
        return results

    # Per-LLM-call body budget for coverage judging; the FULL text is
    # covered via map-reduce over windows of this size (no 3,000-char
    # per-section or 24,000-char whole-report truncation)
    COVERAGE_WINDOW_CHARS = 20000

    def _judge_coverage_llm(self, questions: List[str],
                            chapters: Dict[str, str]) -> Dict[int, Dict]:
        """Map-reduce coverage over the WHOLE body.

        The body is split into windows; every window is judged for every
        question; a question is answered if ANY window answers it — a
        critical question addressed only at the end of a long report is
        found.
        """
        # map: build windows covering ALL text of ALL sections
        windows: List[str] = []
        current: List[str] = []
        used = 0
        for sid, text in chapters.items():
            text = text or ""
            pos = 0
            while pos < len(text) or (pos == 0 and not text):
                room = self.COVERAGE_WINDOW_CHARS - used
                piece = text[pos:pos + max(room, 1000)]
                header = (f"### セクション {sid}"
                          f"（{pos}字目〜）" if self.language == "ja"
                          else f"### Section {sid} (from char {pos})")
                current.append(f"{header}\n{piece}")
                used += len(piece)
                pos += len(piece)
                if used >= self.COVERAGE_WINDOW_CHARS:
                    windows.append("\n\n".join(current))
                    current, used = [], 0
                if not text:
                    break
        if current:
            windows.append("\n\n".join(current))

        merged: Dict[int, Dict] = {}
        for window in windows:
            try:
                result = self._judge_coverage_window(questions, window)
            except Exception as e:
                self.extraction_errors.append(
                    f"coverage window judgement failed: {e}")
                continue
            if not result:
                continue
            for idx, entry in result.items():
                prev = merged.get(idx)
                if prev is None or (entry.get("answered")
                                    and not prev.get("answered")):
                    merged[idx] = entry
        return merged or None

    def _judge_coverage_window(self, questions: List[str],
                               body: str) -> Optional[Dict[int, Dict]]:
        q_lines = "\n".join(f"{i}. {q}" for i, q in enumerate(questions))

        if self.language == "ja":
            prompt = f"""以下のレポート本文が、各「重要な問い」に実質的に回答しているか判定してください。
単語が出てくるだけ（言及）では「回答」になりません。問いに対する説明・数値・結論が本文にあるかで判断してください。

【重要な問い】
{q_lines}

【レポート本文】
{body}

JSONで回答:
{{"coverage": [{{"question": 問い番号, "answered": true/false,
  "section_id": "回答があるセクション番号（無ければ空）",
  "missing": "不足している内容（answered=falseのとき）",
  "search_queries": ["不足を埋める検索クエリ（answered=falseのとき1つ以上）"]}}]}}
JSON以外は出力しない。"""
        else:
            prompt = f"""Judge whether the report body SUBSTANTIVELY ANSWERS each critical
question. A mere mention of the words is NOT an answer; look for an
actual explanation, figures, or a conclusion.

[CRITICAL QUESTIONS]
{q_lines}

[REPORT BODY]
{body}

Respond as JSON:
{{"coverage": [{{"question": index, "answered": true/false,
  "section_id": "section that answers it (or empty)",
  "missing": "what is missing (when answered=false)",
  "search_queries": ["queries to fill the gap (>=1 when answered=false)"]}}]}}
JSON only."""
        response = self.llm.generate(prompt)
        data = extract_json_from_response(response.content)
        out: Dict[int, Dict] = {}
        for item in data.get("coverage", []):
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("question"))
            except (TypeError, ValueError):
                continue
            out[idx] = item
        return out or None

    # ------------------------------------------------------------------
    # primary/freshness gate (3-state)
    # ------------------------------------------------------------------

    @staticmethod
    def assess_primary_freshness(evidence_list: List,
                                 required: bool,
                                 max_age_years: int = 3) -> Tuple[str, str]:
        """Return (state, reason): pass / fail / not_required."""
        if not required:
            return FRESHNESS_NOT_REQUIRED, ""
        now_year = datetime.now().year
        fresh = False
        primary = False
        for ev in evidence_list:
            date = str(getattr(ev, "published_date", "") or "")
            m = re.search(r"(19|20)\d{2}", date)
            if m and now_year - int(m.group(0)) <= max_age_years:
                fresh = True
            try:
                qi = getattr(ev, "quality_indicators", None)
                if qi is not None and bool(getattr(qi, "is_primary_source",
                                                   False)):
                    primary = True
            except Exception:
                pass
            st = getattr(ev, "source_type", None)
            if st is not None and str(getattr(st, "value", st)) in (
                    "official", "academic"):
                primary = True
            qc = getattr(ev, "quality_category", None)
            if qc is not None and str(getattr(qc, "value", qc)) == \
                    "authoritative":
                primary = True
        if fresh and primary:
            return FRESHNESS_PASS, ""
        missing = []
        if not fresh:
            missing.append(f"published within {max_age_years}y")
        if not primary:
            missing.append("primary/official source")
        return FRESHNESS_FAIL, "missing: " + ", ".join(missing)

    # ------------------------------------------------------------------
    # Whole-report verification -> StructuredVerdict
    # ------------------------------------------------------------------

    def verify_report(
        self,
        chapters: Dict[str, str],
        evidence_list: List,
        citation_manager=None,
        critical_questions: Optional[List[str]] = None,
        length_plans: Optional[Dict] = None,
        preferred_body_chars: Optional[int] = None,
        exclude_references: bool = True,
        primary_freshness_required: bool = False,
    ) -> StructuredVerdict:
        """Verify the FINAL candidate body, section by section."""
        verdict = StructuredVerdict()
        self.extraction_errors = []
        self.chunks_total = 0
        self.chunks_failed = 0
        all_claims: List[Claim] = []

        counter = 0
        for sid, text in chapters.items():
            claims = self.extract_claims(sid, text, start_index=counter)
            counter += len(claims)
            all_claims.extend(claims)

        # resolve each claim's cited [SOURCE N] numbers to evidence ids.
        # The DETERMINISTIC parser (sentence/paragraph association) is the
        # primary source; LLM-reported numbers are merged in on top.
        enforce_citations = citation_manager is not None
        cited_ids_by_claim: Dict[str, List[str]] = {}
        if enforce_citations:
            for claim in all_claims:
                parsed = self.parse_cited_numbers(
                    claim.text, chapters.get(claim.section_id, ""))
                nums = sorted(set(parsed)
                              | set(claim.cited_source_numbers or []))
                claim.cited_source_numbers = nums
                if nums:
                    mapping = citation_manager.mapping(claim.section_id)
                    ids = [mapping[n] for n in nums if n in mapping]
                    if ids:
                        cited_ids_by_claim[claim.claim_id] = ids

        chunk_index = self.build_chunk_index(evidence_list)
        self.verify_claims(all_claims, evidence_list,
                           chunk_index=chunk_index,
                           cited_ids_by_claim=cited_ids_by_claim,
                           enforce_citations=enforce_citations)

        # --- aggregate: unsupported / contradicted / uncertain SEPARATELY ---
        weighted_total = 0.0
        weighted_supported = 0.0
        for claim in all_claims:
            weight = IMPORTANCE_WEIGHT.get(claim.importance, 2.0)
            weighted_total += weight
            if claim.status == "supported":
                weighted_supported += weight
            elif claim.status == "unsupported":
                fixable = bool(claim.replacement_source_ids)
                if (claim.citation_mismatch or claim.citation_missing) \
                        and fixable:
                    # the correct source already exists in the locker:
                    # this is a citation defect fixable by REWRITE (the
                    # rewrite swaps in the replacement candidates); it is
                    # NOT counted as an evidence gap needing research
                    verdict.issues.append(VerificationIssue(
                        section_id=claim.section_id,
                        claim_id=claim.claim_id,
                        type=ISSUE_INVALID_CITATION,
                        severity=claim.importance,
                        claim=claim.text,
                        reason=(claim.reason or "") + (
                            "／正しい出典候補あり" if self.language == "ja"
                            else " / replacement candidates available"),
                        supporting_source_ids=claim.replacement_source_ids,
                        fallback_action="replace_citation",
                    ))
                else:
                    verdict.metrics.unsupported_count += 1
                    if claim.importance == "critical":
                        verdict.metrics.unsupported_critical_claims += 1
                    verdict.issues.append(VerificationIssue(
                        section_id=claim.section_id,
                        claim_id=claim.claim_id,
                        type=ISSUE_UNSUPPORTED,
                        severity=claim.importance,
                        claim=claim.text,
                        reason=claim.reason,
                        supporting_source_ids=claim.supporting_source_ids,
                        needed_evidence=claim.text,
                        search_queries=[claim.text[:60]],
                    ))
            elif claim.status == "contradicted":
                verdict.metrics.contradicted_count += 1
                verdict.metrics.unresolved_contradictions += 1
                verdict.issues.append(VerificationIssue(
                    section_id=claim.section_id,
                    claim_id=claim.claim_id,
                    type=ISSUE_CONTRADICTED,
                    severity=claim.importance,
                    claim=claim.text,
                    reason=claim.reason,
                    supporting_source_ids=claim.supporting_source_ids,
                    needed_evidence=claim.text,
                    search_queries=[claim.text[:60]],
                ))
            else:   # uncertain (including downgrades and judge errors)
                verdict.metrics.uncertain_count += 1
                verdict.issues.append(VerificationIssue(
                    section_id=claim.section_id,
                    claim_id=claim.claim_id,
                    type=ISSUE_UNCERTAIN,
                    severity=claim.importance,
                    claim=claim.text,
                    reason=claim.reason,
                    supporting_source_ids=claim.supporting_source_ids,
                ))
            # claim without citations -> integrity issue. Required for
            # critical/important claims REGARDLESS of whether the section
            # registry happens to be empty.
            if (enforce_citations
                    and not claim.cited_source_numbers
                    and claim.importance in ("critical", "important")):
                verdict.issues.append(VerificationIssue(
                    section_id=claim.section_id,
                    claim_id=claim.claim_id,
                    type=ISSUE_UNCITED_CLAIM,
                    severity=claim.importance,
                    claim=claim.text,
                    reason=("引用のない事実主張"
                            if self.language == "ja"
                            else "factual claim without citation"),
                ))

        verdict.metrics.claims_total = len(all_claims)
        verdict.metrics.chunks_total = self.chunks_total
        verdict.metrics.chunks_failed = self.chunks_failed
        # ALWAYS transcribed: callers read metrics, not verifier state
        verdict.metrics.extraction_errors = list(self.extraction_errors)

        # any unverified chunk after retries is a CRITICAL failure — a
        # partial extraction success is never treated as full-body success
        if self.chunks_failed > 0:
            verdict.issues.append(VerificationIssue(
                type=ISSUE_VERIFICATION_FAILURE, severity="critical",
                reason=(f"未検証の本文範囲が残っています（抽出チャンク "
                        f"{self.chunks_failed}/{self.chunks_total} 失敗、"
                        f"リトライ{self.EXTRACT_RETRIES}回後）"
                        if self.language == "ja" else
                        f"unverified body ranges remain: "
                        f"{self.chunks_failed}/{self.chunks_total} "
                        f"extraction chunks failed after "
                        f"{self.EXTRACT_RETRIES} retries"),
            ))

        # citation mismatches invalidate the citation gate even when every
        # [SOURCE N] number technically resolves
        has_mismatch = any(c.citation_mismatch or c.citation_missing
                           for c in all_claims)
        verdict.metrics.claim_support_score = (
            weighted_supported / weighted_total if weighted_total else 0.0)

        # --- fail-closed: an unverifiable factual body never scores 1.0 ---
        body_text = "\n\n".join(chapters.values())
        factual_chars = count_body_chars(
            body_text, exclude_references=exclude_references)
        if not all_claims and factual_chars >= MIN_FACTUAL_BODY_CHARS:
            verdict.metrics.verification_failed = True
            verdict.metrics.claim_support_score = 0.0
            reason = (f"事実的な本文（{factual_chars}字）から主張を1件も"
                      f"抽出できませんでした（チャンク失敗 "
                      f"{self.chunks_failed}/{self.chunks_total}）"
                      if self.language == "ja" else
                      f"no claims extracted from a factual body "
                      f"({factual_chars} chars; chunk failures "
                      f"{self.chunks_failed}/{self.chunks_total})")
            verdict.issues.append(VerificationIssue(
                type=ISSUE_VERIFICATION_FAILURE, severity="critical",
                reason=reason))
        elif self.chunks_total and self.chunks_failed == self.chunks_total:
            verdict.metrics.verification_failed = True
            verdict.metrics.claim_support_score = 0.0
            verdict.issues.append(VerificationIssue(
                type=ISSUE_VERIFICATION_FAILURE, severity="critical",
                reason="all extraction chunks failed"))
        elif not all_claims:
            # trivially short / non-factual body: nothing to support
            verdict.metrics.claim_support_score = 1.0

        # --- critical question coverage (all sections, answered vs
        #     mentioned, with actionable queries) ---
        if critical_questions:
            coverage = self.judge_coverage(critical_questions, chapters)
            answered = sum(1 for c in coverage if c["answered"])
            verdict.metrics.critical_question_coverage = (
                answered / len(critical_questions))
            for c in coverage:
                if c["answered"]:
                    continue
                verdict.issues.append(VerificationIssue(
                    section_id=c["section_id"],
                    type=ISSUE_UNANSWERED_QUESTION,
                    severity="critical",
                    claim=c["question"],
                    reason=c["missing"] or c["question"],
                    needed_evidence=c["missing"] or c["question"],
                    search_queries=c["search_queries"],
                ))

        # --- citation machine validation (registry + orphans + deletion) ---
        if citation_manager is not None:
            problems = citation_manager.validate_report(chapters)
            orphans = citation_manager.report_orphans(chapters)
            lost = citation_manager.sections_that_lost_all_citations(chapters)
            verdict.metrics.citations_valid = not (
                problems or orphans or lost or has_mismatch)
            for sid, numbers in problems.items():
                verdict.issues.append(VerificationIssue(
                    section_id=sid,
                    type=ISSUE_INVALID_CITATION,
                    severity="critical",
                    reason=f"存在しない引用番号: {numbers}",
                ))
            for sid, lines in orphans.items():
                verdict.issues.append(VerificationIssue(
                    section_id=sid,
                    type=ISSUE_ORPHAN_CITATION,
                    severity="important",
                    reason=(f"主張が削除され引用だけが残っています: "
                            f"{lines[0][:80]}"),
                ))
            for sid in lost:
                verdict.issues.append(VerificationIssue(
                    section_id=sid,
                    type=ISSUE_ALL_CITATIONS_DELETED,
                    severity="critical",
                    reason="登録済みエビデンスがあるのに引用が全て削除されています",
                ))

        # --- primary / freshness (3-state), tied to the evidence that the
        #     IMPORTANT claims actually rely on (cited + supporting), not
        #     to the whole locker — one fresh but irrelevant source can
        #     never satisfy the gate. Falls back to all evidence when no
        #     claim-level linkage exists. ---
        relevant_ids: set = set()
        for claim in all_claims:
            if claim.importance in ("critical", "important"):
                relevant_ids.update(claim.supporting_source_ids or [])
                relevant_ids.update(
                    cited_ids_by_claim.get(claim.claim_id, []))
        freshness_evidence = [ev for ev in evidence_list
                              if getattr(ev, "id", "") in relevant_ids] \
            if relevant_ids else evidence_list
        state, why = self.assess_primary_freshness(
            freshness_evidence, required=primary_freshness_required)
        verdict.metrics.primary_freshness = state
        if state == FRESHNESS_FAIL:
            verdict.issues.append(VerificationIssue(
                type=ISSUE_STALE_OR_NON_PRIMARY, severity="critical",
                reason=why,
                needed_evidence=why,
                search_queries=["一次情報 最新 統計"
                                if self.language == "ja"
                                else "primary source latest statistics"],
            ))

        # --- length metrics (adaptive ranges, references excluded) ---
        verdict.metrics.actual_body_chars = factual_chars
        verdict.metrics.preferred_body_chars = preferred_body_chars
        if length_plans:
            verdict.metrics.recommended_min_chars = sum(
                p.recommended_min_chars for p in length_plans.values())
            verdict.metrics.recommended_chars = sum(
                p.recommended_chars for p in length_plans.values())
            verdict.metrics.recommended_max_chars = sum(
                p.recommended_max_chars for p in length_plans.values())

        # --- per-section assessments from REAL verifier/registry data ---
        verdict.section_assessments = self._build_section_assessments(
            chapters, verdict, citation_manager, length_plans,
            evidence_list, exclude_references)
        # Aggregate the per-section assessments into DOCUMENT metrics —
        # these drive the Python decision logic (REWRITE / RESEARCH /
        # COMPRESS / ACCEPT), they are not informational output.
        unused_all: List[str] = []
        missing_all: List[str] = []
        redundant_all: List[str] = []
        for a in verdict.section_assessments.values():
            unused_all.extend(a.unused_evidence_ids)
            missing_all.extend(a.missing_content_units)
            redundant_all.extend(a.redundant_passages)
        verdict.metrics.unused_high_importance_evidence_ids = unused_all
        verdict.metrics.missing_content_units = missing_all
        verdict.metrics.redundant_passages = redundant_all
        for a in verdict.section_assessments.values():
            if a.redundant_passages:
                verdict.issues.append(VerificationIssue(
                    section_id=a.section_id,
                    type=ISSUE_REDUNDANCY,
                    severity="minor",
                    reason=(f"重複記述: {a.redundant_passages[0][:80]}"
                            if self.language == "ja" else
                            f"redundant passage: "
                            f"{a.redundant_passages[0][:80]}"),
                ))

        return verdict

    # ------------------------------------------------------------------
    # Section assessments (real data: verifier + citation registry)
    # ------------------------------------------------------------------

    def _build_section_assessments(
        self, chapters, verdict, citation_manager, length_plans,
        evidence_list, exclude_references,
    ) -> Dict[str, SectionAssessment]:
        by_id = {getattr(ev, "id", ""): ev for ev in evidence_list}
        assessments: Dict[str, SectionAssessment] = {}
        for sid, text in chapters.items():
            plan = (length_plans or {}).get(sid)
            a = SectionAssessment(
                section_id=sid,
                importance=getattr(plan, "importance", "normal"),
                actual_chars=count_body_chars(
                    text, exclude_references=exclude_references),
                recommended_min_chars=getattr(plan, "recommended_min_chars", 0),
                recommended_chars=getattr(plan, "recommended_chars", 0),
                recommended_max_chars=getattr(plan, "recommended_max_chars", 0),
            )
            # missing units: unanswered questions / unsupported claims
            # located in this section
            for issue in verdict.issues:
                if issue.section_id != sid:
                    continue
                if issue.type == ISSUE_UNANSWERED_QUESTION:
                    a.missing_content_units.append(issue.claim or issue.reason)
            # unused high-importance evidence: registered for the section
            # but never cited in its body
            has_evidence = False
            if citation_manager is not None:
                registered = citation_manager.evidence_ids(sid)
                has_evidence = bool(registered)
                for eid in citation_manager.uncited_evidence_ids(sid, text):
                    ev = by_id.get(eid)
                    try:
                        importance = float(
                            getattr(ev, "importance_score", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        importance = 0.0
                    if importance >= 0.6:
                        a.unused_evidence_ids.append(eid)
            # redundancy: near-duplicate paragraphs inside the section
            a.redundant_passages = self._redundant_paragraphs(text)
            a.recommended_action = decide_section_action(
                a, evidence_sufficient=has_evidence)
            assessments[sid] = a
        return assessments

    @staticmethod
    def _redundant_paragraphs(text: str, threshold: float = 0.75,
                              max_paragraphs: int = 60) -> List[str]:
        paras = [p.strip() for p in (text or "").split("\n\n")
                 if len(p.strip()) >= 80 and not p.strip().startswith("#")]
        paras = paras[:max_paragraphs]
        sets = [_bigrams(p) for p in paras]
        redundant = []
        for i in range(len(paras)):
            for j in range(i + 1, len(paras)):
                if _jaccard_sets(sets[i], sets[j]) > threshold:
                    redundant.append(paras[j][:120])
        return redundant
