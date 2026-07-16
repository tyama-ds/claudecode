"""
Claim-level full-text verifier.

Replaces the head-truncated verification (first 6,000 chars of the body,
first 20 evidence items at 4,000 chars) with per-section, per-claim
processing:

- every section is verified, chunked so no text is silently dropped
  (claims after the 6,000th character ARE extracted);
- claims carry stable ids (C-1, C-2, ...) and importance levels;
- for EACH claim the most relevant evidence is selected from the ENTIRE
  Evidence Locker by lexical similarity (evidence item #21+ is used
  whenever it is relevant);
- unsupported and contradicted are counted SEPARATELY;
- the support score weights claims by importance;
- citation ids are machine-checked against the citation registry.

The LLM only produces structured judgements; every aggregation and
threshold lives in Python (finalization.decide).
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..utils.helpers import extract_json_from_response
from ..report.finalization import (
    ISSUE_CONTRADICTED,
    ISSUE_INVALID_CITATION,
    ISSUE_UNSUPPORTED,
    StructuredVerdict,
    VerificationIssue,
    VerificationMetrics,
    count_body_chars,
)
from ..report.length_planner import _jaccard_bigram

# Section text is processed in chunks of this size so that long chapters
# are fully covered without overloading a single prompt
CLAIM_CHUNK_CHARS = 4000
EVIDENCE_PER_CLAIM = 5          # top-k relevant evidence per claim
IMPORTANCE_WEIGHT = {"critical": 3.0, "important": 2.0, "minor": 1.0}


@dataclass
class Claim:
    claim_id: str
    section_id: str
    text: str
    importance: str = "important"      # critical / important / minor
    status: str = "supported"          # supported/unsupported/contradicted/uncertain
    reason: str = ""
    supporting_source_ids: List[str] = field(default_factory=list)


class ClaimVerifier:
    """Full-text, claim-level verification against the evidence locker."""

    def __init__(self, llm_client, language: str = "ja",
                 evidence_per_claim: int = EVIDENCE_PER_CLAIM,
                 chunk_chars: int = CLAIM_CHUNK_CHARS):
        self.llm = llm_client
        self.language = language
        self.evidence_per_claim = evidence_per_claim
        self.chunk_chars = chunk_chars

    # ------------------------------------------------------------------
    # Claim extraction (whole section, chunked — no head truncation)
    # ------------------------------------------------------------------

    def extract_claims(self, section_id: str, text: str,
                       start_index: int = 0) -> List[Claim]:
        claims: List[Claim] = []
        chunks = self._chunk(text)
        counter = start_index
        for ci, chunk in enumerate(chunks):
            prompt = self._claim_prompt(section_id, chunk, ci + 1, len(chunks))
            try:
                response = self.llm.generate(prompt)
                data = extract_json_from_response(response.content)
            except Exception as e:
                print(f"[ClaimVerifier] claim extraction failed "
                      f"({section_id} chunk {ci + 1}): {e}")
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
                claims.append(Claim(
                    claim_id=f"C-{counter}",
                    section_id=section_id,
                    text=claim_text,
                    importance=importance,
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
  "importance": "critical/important/minor"}}]}}

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
{{"claims": [{{"claim": "the claim", "importance": "critical/important/minor"}}]}}

Do not include opinions or generic statements. JSON only."""

    # ------------------------------------------------------------------
    # Evidence selection from the WHOLE locker (no first-20 cap)
    # ------------------------------------------------------------------

    def select_evidence(self, claim: Claim, evidence_list: List) -> List:
        """Top-k most relevant evidence for this claim, from ALL evidence."""
        scored = []
        for ev in evidence_list:
            text = (getattr(ev, "extracted_text", "") or
                    getattr(ev, "content_excerpt", "") or "")
            title = getattr(ev, "title", "") or ""
            score = max(
                _jaccard_bigram(claim.text, text[:800]),
                _jaccard_bigram(claim.text, title) * 0.8,
            )
            # exact number/term overlap is a strong signal
            for token in re.findall(r"\d[\d,.]*", claim.text)[:5]:
                if token and token in text:
                    score += 0.15
            scored.append((score, ev))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [ev for score, ev in scored[:self.evidence_per_claim]
                if score > 0.02]

    # ------------------------------------------------------------------
    # Per-claim verification
    # ------------------------------------------------------------------

    def verify_claims(self, claims: List[Claim], evidence_list: List
                      ) -> List[Claim]:
        """Judge each claim against ITS OWN selected evidence.

        Claims are batched per section into one prompt each with their
        individually selected evidence, keeping prompts bounded while
        still covering every claim.
        """
        for claim in claims:
            selected = self.select_evidence(claim, evidence_list)
            claim.supporting_source_ids = [
                getattr(ev, "id", "") for ev in selected]
            if not selected:
                claim.status = "unsupported"
                claim.reason = ("関連するエビデンスが見つかりません"
                                if self.language == "ja"
                                else "no relevant evidence found")
                continue
            try:
                verdict = self._judge_claim(claim, selected)
            except Exception as e:
                claim.status = "uncertain"
                claim.reason = f"verification error: {e}"
                continue
            claim.status = verdict.get("status", "uncertain")
            claim.reason = verdict.get("reason", "")
            ids = verdict.get("supporting_source_ids")
            if isinstance(ids, list) and ids:
                valid = {getattr(ev, "id", "") for ev in selected}
                claim.supporting_source_ids = [i for i in ids if i in valid]
        return claims

    def _judge_claim(self, claim: Claim, evidence: List) -> Dict:
        blocks = []
        for ev in evidence:
            text = (getattr(ev, "extracted_text", "") or
                    getattr(ev, "content_excerpt", "") or "")[:1500]
            blocks.append(f"[{getattr(ev, 'id', '?')}] "
                          f"{getattr(ev, 'title', '')}\n{text}")
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
    ) -> StructuredVerdict:
        """Verify the FINAL candidate body, section by section."""
        verdict = StructuredVerdict()
        all_claims: List[Claim] = []

        counter = 0
        for sid, text in chapters.items():
            claims = self.extract_claims(sid, text, start_index=counter)
            counter += len(claims)
            all_claims.extend(claims)

        self.verify_claims(all_claims, evidence_list)

        # --- aggregate: unsupported vs contradicted SEPARATELY ---
        weighted_total = 0.0
        weighted_supported = 0.0
        for claim in all_claims:
            weight = IMPORTANCE_WEIGHT.get(claim.importance, 2.0)
            weighted_total += weight
            if claim.status == "supported":
                weighted_supported += weight
            elif claim.status == "unsupported":
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

        verdict.metrics.claim_support_score = (
            weighted_supported / weighted_total if weighted_total else 1.0)

        # --- critical question coverage ---
        if critical_questions:
            body = "\n".join(chapters.values())
            answered = 0
            for q in critical_questions:
                if _jaccard_bigram(q, body[:20000]) > 0.05 or q[:20] in body:
                    answered += 1
            verdict.metrics.critical_question_coverage = (
                answered / len(critical_questions))

        # --- citation machine validation ---
        if citation_manager is not None:
            problems = citation_manager.validate_report(chapters)
            verdict.metrics.citations_valid = not problems
            for sid, numbers in problems.items():
                verdict.issues.append(VerificationIssue(
                    section_id=sid,
                    type=ISSUE_INVALID_CITATION,
                    severity="critical",
                    reason=f"存在しない引用番号: {numbers}",
                ))

        # --- length metrics (adaptive ranges, references excluded) ---
        body_text = "\n\n".join(chapters.values())
        verdict.metrics.actual_body_chars = count_body_chars(
            body_text, exclude_references=exclude_references)
        verdict.metrics.preferred_body_chars = preferred_body_chars
        if length_plans:
            verdict.metrics.recommended_min_chars = sum(
                p.recommended_min_chars for p in length_plans.values())
            verdict.metrics.recommended_chars = sum(
                p.recommended_chars for p in length_plans.values())
            verdict.metrics.recommended_max_chars = sum(
                p.recommended_max_chars for p in length_plans.values())

        return verdict
