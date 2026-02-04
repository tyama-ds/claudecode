"""
Hallucination Checker - Advanced hallucination detection and content verification.

Extends the base Verifier with:
- Detailed claim categorization
- Evidence quality-based weighting
- Cross-reference verification
- Source traceability
- Enhanced visualization
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from ..evidence.locker import EvidenceLocker, Evidence, QualityCategory
from .verifier import (
    Verifier,
    VerificationResult,
    ClaimVerification,
    ConfidenceLevel,
)


class ClaimType(str, Enum):
    """Types of claims that can be verified."""
    STATISTICAL = "statistical"  # Numbers, percentages, data
    TEMPORAL = "temporal"  # Dates, timelines, events
    QUOTATION = "quotation"  # Attributed quotes
    CAUSAL = "causal"  # Cause-effect relationships
    COMPARATIVE = "comparative"  # Comparisons
    ABSOLUTE = "absolute"  # Always, never, all, none
    FACTUAL = "factual"  # General factual claims
    OPINION = "opinion"  # Opinions presented as facts
    PREDICTION = "prediction"  # Future predictions
    HISTORICAL = "historical"  # Historical facts


class HallucinationRisk(str, Enum):
    """Risk levels for hallucination."""
    CRITICAL = "critical"  # Very likely hallucination
    HIGH = "high"  # Probable hallucination
    MEDIUM = "medium"  # Possible hallucination
    LOW = "low"  # Unlikely hallucination
    NONE = "none"  # Verified correct


@dataclass
class DetailedClaim:
    """Detailed claim with type classification and evidence links."""
    id: str
    text: str
    claim_type: ClaimType
    section: str = ""
    context: str = ""  # Surrounding text for context

    # Verification results
    confidence: ConfidenceLevel = ConfidenceLevel.UNSUPPORTED
    hallucination_risk: HallucinationRisk = HallucinationRisk.MEDIUM
    risk_score: float = 0.5  # 0-1, higher = more risky

    # Evidence links
    supporting_evidence: List[str] = field(default_factory=list)  # Evidence IDs
    contradicting_evidence: List[str] = field(default_factory=list)
    evidence_quality_score: float = 0.0  # Average quality of supporting evidence

    # Verification details
    reasoning: str = ""
    suggestions: str = ""
    verified_facts: List[str] = field(default_factory=list)
    issues_found: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "claim_type": self.claim_type.value,
            "section": self.section,
            "context": self.context,
            "confidence": self.confidence.value,
            "hallucination_risk": self.hallucination_risk.value,
            "risk_score": self.risk_score,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "evidence_quality_score": self.evidence_quality_score,
            "reasoning": self.reasoning,
            "suggestions": self.suggestions,
            "verified_facts": self.verified_facts,
            "issues_found": self.issues_found,
        }


@dataclass
class HallucinationCheckResult:
    """Complete result of hallucination checking."""
    document_title: str
    checked_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Summary statistics
    total_claims: int = 0
    verified_claims: int = 0
    suspicious_claims: int = 0
    likely_hallucinations: int = 0

    # Detailed results
    claims: List[DetailedClaim] = field(default_factory=list)

    # By type breakdown
    claims_by_type: Dict[str, int] = field(default_factory=dict)
    risks_by_type: Dict[str, int] = field(default_factory=dict)

    # Scores
    overall_accuracy_score: float = 0.0  # 0-1
    evidence_coverage_score: float = 0.0  # How much content is backed by evidence
    source_quality_score: float = 0.0  # Quality of supporting sources

    # Issues summary
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_title": self.document_title,
            "checked_at": self.checked_at,
            "total_claims": self.total_claims,
            "verified_claims": self.verified_claims,
            "suspicious_claims": self.suspicious_claims,
            "likely_hallucinations": self.likely_hallucinations,
            "claims": [c.to_dict() for c in self.claims],
            "claims_by_type": self.claims_by_type,
            "risks_by_type": self.risks_by_type,
            "overall_accuracy_score": self.overall_accuracy_score,
            "evidence_coverage_score": self.evidence_coverage_score,
            "source_quality_score": self.source_quality_score,
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }

    def get_summary(self, language: str = "ja") -> str:
        """Get human-readable summary."""
        if language == "ja":
            return f"""
== ハルシネーションチェック結果 ==
文書: {self.document_title}
検証日時: {self.checked_at}

総主張数: {self.total_claims}
検証済み: {self.verified_claims}
要確認: {self.suspicious_claims}
ハルシネーションの可能性: {self.likely_hallucinations}

全体精度スコア: {self.overall_accuracy_score:.1%}
エビデンスカバレッジ: {self.evidence_coverage_score:.1%}
ソース品質スコア: {self.source_quality_score:.1%}

重大な問題: {len(self.critical_issues)}件
警告: {len(self.warnings)}件
"""
        else:
            return f"""
== Hallucination Check Result ==
Document: {self.document_title}
Checked: {self.checked_at}

Total Claims: {self.total_claims}
Verified: {self.verified_claims}
Suspicious: {self.suspicious_claims}
Likely Hallucinations: {self.likely_hallucinations}

Overall Accuracy: {self.overall_accuracy_score:.1%}
Evidence Coverage: {self.evidence_coverage_score:.1%}
Source Quality: {self.source_quality_score:.1%}

Critical Issues: {len(self.critical_issues)}
Warnings: {len(self.warnings)}
"""


class HallucinationChecker:
    """
    Advanced hallucination detection and verification.

    Features:
    - Claim extraction and categorization
    - Evidence-based verification with quality weighting
    - Cross-reference checking
    - Detailed risk assessment
    - Visual HTML report generation
    """

    # Patterns for detecting claim types
    CLAIM_PATTERNS = {
        ClaimType.STATISTICAL: [
            r'\d+\.?\d*\s*%',  # Percentages
            r'\d+\.?\d*\s*(million|billion|trillion|万|億|兆)',  # Large numbers
            r'(approximately|about|around|roughly|約|およそ)\s*\d+',
            r'\d+\s*(times|倍|回)',
            r'(average|mean|median|平均|中央値)',
        ],
        ClaimType.TEMPORAL: [
            r'\d{4}年',  # Japanese year
            r'(19|20)\d{2}',  # Years
            r'(January|February|March|April|May|June|July|August|September|October|November|December)',
            r'(1月|2月|3月|4月|5月|6月|7月|8月|9月|10月|11月|12月)',
            r'(yesterday|today|tomorrow|last year|next year|昨日|今日|明日|昨年|来年)',
        ],
        ClaimType.QUOTATION: [
            r'"[^"]{10,}"',  # English quotes
            r'「[^」]{10,}」',  # Japanese quotes
            r'(said|stated|claimed|mentioned|と述べた|と語った)',
        ],
        ClaimType.CAUSAL: [
            r'(because|due to|as a result|therefore|consequently|hence)',
            r'(なぜなら|そのため|したがって|結果として|原因)',
            r'(leads to|causes|results in|につながる|を引き起こす)',
        ],
        ClaimType.COMPARATIVE: [
            r'(more than|less than|greater than|fewer than)',
            r'(より多い|より少ない|以上|以下)',
            r'(compared to|in contrast|unlike|と比較して)',
            r'(best|worst|largest|smallest|最大|最小|最高|最低)',
        ],
        ClaimType.ABSOLUTE: [
            r'\b(always|never|all|none|every|no one)\b',
            r'(常に|決して|すべて|全て|誰も)',
            r'(definitely|certainly|absolutely|確実に|必ず)',
        ],
    }

    def __init__(self, llm_client, language: str = "ja"):
        """
        Initialize HallucinationChecker.

        Args:
            llm_client: LLM API client
            language: Target language
        """
        self.llm = llm_client
        self.language = language
        self.base_verifier = Verifier(llm_client, language)

    def check_content(
        self,
        content: str,
        evidence_locker: EvidenceLocker,
        document_title: str = "Research Report",
        strictness: str = "medium",
    ) -> HallucinationCheckResult:
        """
        Perform comprehensive hallucination check on content.

        Args:
            content: The content to check
            evidence_locker: Evidence locker with sources
            document_title: Title of the document
            strictness: Check strictness (low, medium, high)

        Returns:
            HallucinationCheckResult with detailed findings
        """
        result = HallucinationCheckResult(document_title=document_title)

        # Step 1: Extract and categorize claims
        claims = self._extract_and_categorize_claims(content)
        result.total_claims = len(claims)

        # Step 2: Build evidence index for efficient lookup
        evidence_index = self._build_evidence_index(evidence_locker)

        # Step 3: Verify each claim against evidence
        for claim in claims:
            verified_claim = self._verify_claim_with_evidence(
                claim, evidence_locker, evidence_index, strictness
            )
            result.claims.append(verified_claim)

            # Update type counts
            claim_type = verified_claim.claim_type.value
            result.claims_by_type[claim_type] = result.claims_by_type.get(claim_type, 0) + 1

            # Update risk counts
            if verified_claim.hallucination_risk in [HallucinationRisk.CRITICAL, HallucinationRisk.HIGH]:
                result.likely_hallucinations += 1
                result.risks_by_type[claim_type] = result.risks_by_type.get(claim_type, 0) + 1
            elif verified_claim.hallucination_risk == HallucinationRisk.MEDIUM:
                result.suspicious_claims += 1
            else:
                result.verified_claims += 1

        # Step 4: Calculate overall scores
        self._calculate_scores(result, evidence_locker)

        # Step 5: Generate issues and recommendations
        self._generate_issues_and_recommendations(result)

        return result

    def _extract_and_categorize_claims(self, content: str) -> List[DetailedClaim]:
        """Extract claims and categorize them by type."""
        prompt = f"""Analyze this content and extract all verifiable claims with their types.

Content:
{content[:8000]}

For each claim, identify:
1. The exact claim text
2. The claim type: statistical, temporal, quotation, causal, comparative, absolute, factual, opinion, prediction, historical
3. The section it belongs to (if identifiable)
4. A brief context (surrounding text summary)

Return as JSON array:
[
    {{
        "text": "The exact claim",
        "type": "statistical/temporal/quotation/causal/comparative/absolute/factual/opinion/prediction/historical",
        "section": "Section name or empty",
        "context": "Brief context"
    }},
    ...
]

Focus on verifiable factual claims. Extract 15-30 significant claims."""

        response = self.llm.generate(prompt)
        claims = []

        try:
            content_str = response.content
            start = content_str.find("[")
            end = content_str.rfind("]") + 1
            if start != -1 and end > start:
                raw_claims = json.loads(content_str[start:end])

                for i, raw in enumerate(raw_claims):
                    claim_type_str = raw.get("type", "factual").lower()
                    try:
                        claim_type = ClaimType(claim_type_str)
                    except ValueError:
                        claim_type = ClaimType.FACTUAL

                    claims.append(DetailedClaim(
                        id=f"claim_{i+1}",
                        text=raw.get("text", ""),
                        claim_type=claim_type,
                        section=raw.get("section", ""),
                        context=raw.get("context", ""),
                    ))
        except (json.JSONDecodeError, ValueError):
            # Fallback: use pattern matching to extract claims
            claims = self._extract_claims_by_pattern(content)

        return claims

    def _extract_claims_by_pattern(self, content: str) -> List[DetailedClaim]:
        """Fallback claim extraction using patterns."""
        claims = []
        sentences = re.split(r'[。.!?！？]', content)
        claim_id = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue

            # Detect claim type by pattern matching
            claim_type = ClaimType.FACTUAL
            for ctype, patterns in self.CLAIM_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        claim_type = ctype
                        break
                if claim_type != ClaimType.FACTUAL:
                    break

            claim_id += 1
            claims.append(DetailedClaim(
                id=f"claim_{claim_id}",
                text=sentence,
                claim_type=claim_type,
            ))

            if claim_id >= 30:  # Limit claims
                break

        return claims

    def _build_evidence_index(self, evidence_locker: EvidenceLocker) -> Dict[str, Any]:
        """Build an index of evidence for efficient lookup."""
        index = {
            "by_id": {},
            "by_keywords": {},
            "all_text": "",
        }

        for evidence in evidence_locker.get_all_evidence():
            index["by_id"][evidence.id] = evidence

            # Index by keywords in title and content
            text = f"{evidence.title} {evidence.content_excerpt}".lower()
            words = re.findall(r'\b\w{4,}\b', text)
            for word in words:
                if word not in index["by_keywords"]:
                    index["by_keywords"][word] = []
                index["by_keywords"][word].append(evidence.id)

            index["all_text"] += f"\n[{evidence.id}] {evidence.title}: {evidence.content_excerpt[:500]}"

        return index

    def _verify_claim_with_evidence(
        self,
        claim: DetailedClaim,
        evidence_locker: EvidenceLocker,
        evidence_index: Dict[str, Any],
        strictness: str,
    ) -> DetailedClaim:
        """Verify a single claim against evidence."""
        # Find potentially relevant evidence
        relevant_evidence = self._find_relevant_evidence(claim, evidence_index)

        # Prepare evidence summaries
        evidence_text = ""
        for eid in relevant_evidence[:5]:  # Limit to top 5
            evidence = evidence_index["by_id"].get(eid)
            if evidence:
                quality = evidence.quality_category.value
                evidence_text += f"\n[{eid}] (Quality: {quality})\n"
                evidence_text += f"Title: {evidence.title}\n"
                evidence_text += f"Content: {evidence.content_excerpt[:400]}\n---"

        # LLM verification
        strictness_instruction = {
            "low": "Be lenient - only flag clearly false claims",
            "medium": "Be balanced - flag unsupported and suspicious claims",
            "high": "Be strict - require strong evidence for all claims",
        }.get(strictness, "Be balanced")

        prompt = f"""Verify this claim against the available evidence.

Claim: {claim.text}
Claim Type: {claim.claim_type.value}
Context: {claim.context}

Available Evidence:
{evidence_text if evidence_text else "No directly relevant evidence found"}

Strictness: {strictness_instruction}

Analyze thoroughly:
1. Is this claim supported, partially supported, or contradicted by evidence?
2. Are there any factual errors or inaccuracies?
3. Is this the type of claim prone to AI hallucination?
4. What is the confidence level based on evidence quality?

Return as JSON:
{{
    "confidence": "high/medium/low/unsupported/contradicted",
    "hallucination_risk": "critical/high/medium/low/none",
    "risk_score": 0.0-1.0,
    "supporting_evidence": ["evidence_id_1"],
    "contradicting_evidence": ["evidence_id_2"],
    "reasoning": "Detailed explanation",
    "verified_facts": ["Confirmed fact 1"],
    "issues_found": ["Issue 1"],
    "suggestions": "How to improve or verify"
}}"""

        response = self.llm.generate(prompt)

        try:
            content_str = response.content
            start = content_str.find("{")
            end = content_str.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content_str[start:end])

                claim.confidence = ConfidenceLevel(data.get("confidence", "low"))
                claim.hallucination_risk = HallucinationRisk(data.get("hallucination_risk", "medium"))
                claim.risk_score = float(data.get("risk_score", 0.5))
                claim.supporting_evidence = data.get("supporting_evidence", [])
                claim.contradicting_evidence = data.get("contradicting_evidence", [])
                claim.reasoning = data.get("reasoning", "")
                claim.verified_facts = data.get("verified_facts", [])
                claim.issues_found = data.get("issues_found", [])
                claim.suggestions = data.get("suggestions", "")

                # Calculate evidence quality score
                if claim.supporting_evidence:
                    quality_scores = []
                    for eid in claim.supporting_evidence:
                        evidence = evidence_index["by_id"].get(eid)
                        if evidence:
                            quality_map = {
                                QualityCategory.AUTHORITATIVE: 1.0,
                                QualityCategory.HIGH: 0.8,
                                QualityCategory.MEDIUM: 0.6,
                                QualityCategory.LOW: 0.3,
                                QualityCategory.UNVERIFIED: 0.1,
                            }
                            quality_scores.append(quality_map.get(evidence.quality_category, 0.5))
                    if quality_scores:
                        claim.evidence_quality_score = sum(quality_scores) / len(quality_scores)

        except (json.JSONDecodeError, ValueError, KeyError):
            # Fallback - mark as needing verification
            claim.confidence = ConfidenceLevel.LOW
            claim.hallucination_risk = HallucinationRisk.MEDIUM
            claim.risk_score = 0.5
            claim.reasoning = "Automatic verification incomplete"
            claim.suggestions = "Manual verification recommended"

        return claim

    def _find_relevant_evidence(
        self,
        claim: DetailedClaim,
        evidence_index: Dict[str, Any],
    ) -> List[str]:
        """Find evidence relevant to a claim."""
        relevant = set()

        # Extract keywords from claim
        claim_text = claim.text.lower()
        keywords = re.findall(r'\b\w{4,}\b', claim_text)

        # Find evidence with matching keywords
        for keyword in keywords:
            if keyword in evidence_index["by_keywords"]:
                for eid in evidence_index["by_keywords"][keyword]:
                    relevant.add(eid)

        return list(relevant)

    def _calculate_scores(
        self,
        result: HallucinationCheckResult,
        evidence_locker: EvidenceLocker,
    ) -> None:
        """Calculate overall scores for the result."""
        if not result.claims:
            return

        # Overall accuracy score
        confidence_scores = []
        for claim in result.claims:
            score_map = {
                ConfidenceLevel.HIGH: 1.0,
                ConfidenceLevel.MEDIUM: 0.7,
                ConfidenceLevel.LOW: 0.3,
                ConfidenceLevel.UNSUPPORTED: 0.0,
                ConfidenceLevel.CONTRADICTED: -0.5,
            }
            confidence_scores.append(score_map.get(claim.confidence, 0.5))
        result.overall_accuracy_score = max(0, sum(confidence_scores) / len(confidence_scores))

        # Evidence coverage score
        claims_with_evidence = sum(1 for c in result.claims if c.supporting_evidence)
        result.evidence_coverage_score = claims_with_evidence / len(result.claims)

        # Source quality score
        quality_scores = [c.evidence_quality_score for c in result.claims if c.evidence_quality_score > 0]
        if quality_scores:
            result.source_quality_score = sum(quality_scores) / len(quality_scores)

    def _generate_issues_and_recommendations(self, result: HallucinationCheckResult) -> None:
        """Generate issues and recommendations based on findings."""
        # Critical issues
        for claim in result.claims:
            if claim.hallucination_risk == HallucinationRisk.CRITICAL:
                result.critical_issues.append(
                    f"[{claim.id}] Critical hallucination risk: {claim.text[:100]}..."
                )
            if claim.contradicting_evidence:
                result.critical_issues.append(
                    f"[{claim.id}] Contradicted by evidence: {claim.text[:100]}..."
                )

        # Warnings
        for claim in result.claims:
            if claim.hallucination_risk == HallucinationRisk.HIGH:
                result.warnings.append(
                    f"[{claim.id}] High risk ({claim.claim_type.value}): {claim.text[:80]}..."
                )
            if not claim.supporting_evidence and claim.confidence != ConfidenceLevel.HIGH:
                result.warnings.append(
                    f"[{claim.id}] No supporting evidence: {claim.text[:80]}..."
                )

        # Recommendations
        if result.likely_hallucinations > 0:
            result.recommendations.append(
                f"Review and verify {result.likely_hallucinations} claims flagged as likely hallucinations"
            )

        if result.evidence_coverage_score < 0.5:
            result.recommendations.append(
                "Evidence coverage is low - consider adding more source citations"
            )

        if result.source_quality_score < 0.5:
            result.recommendations.append(
                "Source quality is low - consider using more authoritative sources"
            )

        # Type-specific recommendations
        if result.risks_by_type.get("statistical", 0) > 2:
            result.recommendations.append(
                "Multiple statistical claims need verification - double-check all numbers"
            )
        if result.risks_by_type.get("temporal", 0) > 2:
            result.recommendations.append(
                "Multiple date/time claims need verification - verify all dates"
            )

    def generate_detailed_html_report(
        self,
        result: HallucinationCheckResult,
        evidence_locker: EvidenceLocker = None,
        output_path: Path = None,
    ) -> str:
        """
        Generate detailed HTML hallucination check report.

        Args:
            result: HallucinationCheckResult
            evidence_locker: Optional evidence locker for source details
            output_path: Optional path to save HTML

        Returns:
            HTML content string
        """
        # Risk colors
        risk_colors = {
            "critical": "#dc3545",
            "high": "#fd7e14",
            "medium": "#ffc107",
            "low": "#20c997",
            "none": "#28a745",
        }

        confidence_colors = {
            "high": "#28a745",
            "medium": "#ffc107",
            "low": "#fd7e14",
            "unsupported": "#dc3545",
            "contradicted": "#6f42c1",
        }

        # Build claims HTML
        claims_html = []
        for claim in result.claims:
            risk_color = risk_colors.get(claim.hallucination_risk.value, "#6c757d")
            conf_color = confidence_colors.get(claim.confidence.value, "#6c757d")

            # Evidence links
            evidence_links = ""
            if claim.supporting_evidence:
                evidence_links = "<div class='evidence-links'><strong>Supporting:</strong> "
                evidence_links += ", ".join(
                    f'<span class="evidence-tag support">{eid}</span>'
                    for eid in claim.supporting_evidence
                )
                evidence_links += "</div>"
            if claim.contradicting_evidence:
                evidence_links += "<div class='evidence-links'><strong>Contradicting:</strong> "
                evidence_links += ", ".join(
                    f'<span class="evidence-tag contradict">{eid}</span>'
                    for eid in claim.contradicting_evidence
                )
                evidence_links += "</div>"

            # Issues and facts
            issues_html = ""
            if claim.issues_found:
                issues_html = "<div class='issues'><strong>Issues:</strong><ul>"
                issues_html += "".join(f"<li>{issue}</li>" for issue in claim.issues_found)
                issues_html += "</ul></div>"

            claims_html.append(f"""
            <div class="claim-card" data-risk="{claim.hallucination_risk.value}"
                 data-confidence="{claim.confidence.value}" data-type="{claim.claim_type.value}">
                <div class="claim-header">
                    <span class="claim-id">{claim.id}</span>
                    <span class="claim-type-badge">{claim.claim_type.value}</span>
                    <span class="risk-badge" style="background-color: {risk_color};">
                        Risk: {claim.hallucination_risk.value.upper()}
                    </span>
                    <span class="conf-badge" style="background-color: {conf_color};">
                        {claim.confidence.value.upper()}
                    </span>
                    <span class="risk-score">Score: {claim.risk_score:.2f}</span>
                </div>
                <div class="claim-text">{claim.text}</div>
                {f'<div class="claim-context">Context: {claim.context}</div>' if claim.context else ''}
                <div class="claim-analysis">
                    <div class="reasoning"><strong>Analysis:</strong> {claim.reasoning}</div>
                    {evidence_links}
                    {issues_html}
                    {f'<div class="suggestions"><strong>Suggestions:</strong> {claim.suggestions}</div>' if claim.suggestions else ''}
                </div>
            </div>
            """)

        # Type distribution chart data
        type_data = json.dumps(result.claims_by_type)
        risk_data = json.dumps(result.risks_by_type)

        # Critical issues HTML
        critical_html = ""
        if result.critical_issues:
            critical_items = "".join(f"<li>{issue}</li>" for issue in result.critical_issues)
            critical_html = f"""
            <div class="alert alert-danger">
                <h4>Critical Issues ({len(result.critical_issues)})</h4>
                <ul>{critical_items}</ul>
            </div>
            """

        # Warnings HTML
        warnings_html = ""
        if result.warnings:
            warning_items = "".join(f"<li>{w}</li>" for w in result.warnings[:10])
            warnings_html = f"""
            <div class="alert alert-warning">
                <h4>Warnings ({len(result.warnings)})</h4>
                <ul>{warning_items}</ul>
                {f'<p>...and {len(result.warnings) - 10} more</p>' if len(result.warnings) > 10 else ''}
            </div>
            """

        # Recommendations HTML
        rec_html = ""
        if result.recommendations:
            rec_items = "".join(f"<li>{r}</li>" for r in result.recommendations)
            rec_html = f"""
            <div class="alert alert-info">
                <h4>Recommendations</h4>
                <ul>{rec_items}</ul>
            </div>
            """

        html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hallucination Check Report - {result.document_title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f0f2f5;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header .meta {{ opacity: 0.9; }}

        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            transition: transform 0.2s;
        }}
        .metric-card:hover {{ transform: translateY(-5px); }}
        .metric-value {{ font-size: 2.5em; font-weight: bold; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .metric-good {{ color: #28a745; }}
        .metric-warning {{ color: #ffc107; }}
        .metric-danger {{ color: #dc3545; }}

        .score-bars {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }}
        .score-bar {{
            margin: 15px 0;
        }}
        .score-bar-label {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }}
        .score-bar-track {{
            height: 12px;
            background: #e9ecef;
            border-radius: 6px;
            overflow: hidden;
        }}
        .score-bar-fill {{
            height: 100%;
            border-radius: 6px;
            transition: width 0.5s ease;
        }}

        .alerts {{ margin-bottom: 30px; }}
        .alert {{
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
        }}
        .alert h4 {{ margin-bottom: 10px; }}
        .alert ul {{ padding-left: 20px; }}
        .alert li {{ margin: 5px 0; }}
        .alert-danger {{ background: #f8d7da; border-left: 5px solid #dc3545; }}
        .alert-warning {{ background: #fff3cd; border-left: 5px solid #ffc107; }}
        .alert-info {{ background: #d1ecf1; border-left: 5px solid #17a2b8; }}

        .filters {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }}
        .filter-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 15px;
        }}
        .filter-btn {{
            padding: 8px 16px;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
        }}
        .filter-btn:hover {{ transform: scale(1.05); }}
        .filter-btn.active {{ box-shadow: 0 0 0 3px rgba(0,0,0,0.2); }}

        .claims-container {{
            display: grid;
            gap: 20px;
        }}
        .claim-card {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            border-left: 5px solid #6c757d;
            transition: all 0.2s;
        }}
        .claim-card:hover {{ box-shadow: 0 8px 25px rgba(0,0,0,0.12); }}
        .claim-header {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            margin-bottom: 15px;
        }}
        .claim-id {{ font-weight: bold; color: #666; }}
        .claim-type-badge {{
            background: #e9ecef;
            padding: 3px 10px;
            border-radius: 10px;
            font-size: 0.85em;
        }}
        .risk-badge, .conf-badge {{
            padding: 3px 10px;
            border-radius: 10px;
            color: white;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .risk-score {{
            margin-left: auto;
            font-size: 0.9em;
            color: #666;
        }}
        .claim-text {{
            font-size: 1.1em;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 15px;
        }}
        .claim-context {{
            font-size: 0.9em;
            color: #666;
            font-style: italic;
            margin-bottom: 15px;
        }}
        .claim-analysis {{
            font-size: 0.95em;
        }}
        .claim-analysis > div {{
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .evidence-links {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            align-items: center;
        }}
        .evidence-tag {{
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.85em;
        }}
        .evidence-tag.support {{ background: #d4edda; color: #155724; }}
        .evidence-tag.contradict {{ background: #f8d7da; color: #721c24; }}
        .issues ul {{ padding-left: 20px; margin-top: 5px; }}

        @media (max-width: 768px) {{
            .header {{ padding: 20px; }}
            .header h1 {{ font-size: 1.8em; }}
            .dashboard {{ grid-template-columns: 1fr 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Hallucination Check Report</h1>
            <p class="meta">{result.document_title}</p>
            <p class="meta">Checked: {result.checked_at}</p>
        </div>

        <div class="dashboard">
            <div class="metric-card">
                <div class="metric-value">{result.total_claims}</div>
                <div class="metric-label">Total Claims</div>
            </div>
            <div class="metric-card">
                <div class="metric-value metric-good">{result.verified_claims}</div>
                <div class="metric-label">Verified</div>
            </div>
            <div class="metric-card">
                <div class="metric-value metric-warning">{result.suspicious_claims}</div>
                <div class="metric-label">Suspicious</div>
            </div>
            <div class="metric-card">
                <div class="metric-value metric-danger">{result.likely_hallucinations}</div>
                <div class="metric-label">Likely Hallucinations</div>
            </div>
        </div>

        <div class="score-bars">
            <h3>Quality Scores</h3>
            <div class="score-bar">
                <div class="score-bar-label">
                    <span>Overall Accuracy</span>
                    <span>{result.overall_accuracy_score:.1%}</span>
                </div>
                <div class="score-bar-track">
                    <div class="score-bar-fill" style="width: {result.overall_accuracy_score*100}%; background: {'#28a745' if result.overall_accuracy_score >= 0.7 else '#ffc107' if result.overall_accuracy_score >= 0.5 else '#dc3545'};"></div>
                </div>
            </div>
            <div class="score-bar">
                <div class="score-bar-label">
                    <span>Evidence Coverage</span>
                    <span>{result.evidence_coverage_score:.1%}</span>
                </div>
                <div class="score-bar-track">
                    <div class="score-bar-fill" style="width: {result.evidence_coverage_score*100}%; background: {'#28a745' if result.evidence_coverage_score >= 0.7 else '#ffc107' if result.evidence_coverage_score >= 0.5 else '#dc3545'};"></div>
                </div>
            </div>
            <div class="score-bar">
                <div class="score-bar-label">
                    <span>Source Quality</span>
                    <span>{result.source_quality_score:.1%}</span>
                </div>
                <div class="score-bar-track">
                    <div class="score-bar-fill" style="width: {result.source_quality_score*100}%; background: {'#28a745' if result.source_quality_score >= 0.7 else '#ffc107' if result.source_quality_score >= 0.5 else '#dc3545'};"></div>
                </div>
            </div>
        </div>

        <div class="alerts">
            {critical_html}
            {warnings_html}
            {rec_html}
        </div>

        <div class="filters">
            <h3>Filter Claims</h3>
            <div class="filter-group">
                <strong>By Risk:</strong>
                <button class="filter-btn active" style="background: #6c757d; color: white;" onclick="filterBy('risk', 'all')">All</button>
                <button class="filter-btn" style="background: #dc3545; color: white;" onclick="filterBy('risk', 'critical')">Critical</button>
                <button class="filter-btn" style="background: #fd7e14; color: white;" onclick="filterBy('risk', 'high')">High</button>
                <button class="filter-btn" style="background: #ffc107;" onclick="filterBy('risk', 'medium')">Medium</button>
                <button class="filter-btn" style="background: #20c997; color: white;" onclick="filterBy('risk', 'low')">Low</button>
                <button class="filter-btn" style="background: #28a745; color: white;" onclick="filterBy('risk', 'none')">None</button>
            </div>
            <div class="filter-group">
                <strong>By Type:</strong>
                <button class="filter-btn" style="background: #e9ecef;" onclick="filterBy('type', 'all')">All</button>
                <button class="filter-btn" style="background: #e9ecef;" onclick="filterBy('type', 'statistical')">Statistical</button>
                <button class="filter-btn" style="background: #e9ecef;" onclick="filterBy('type', 'temporal')">Temporal</button>
                <button class="filter-btn" style="background: #e9ecef;" onclick="filterBy('type', 'factual')">Factual</button>
                <button class="filter-btn" style="background: #e9ecef;" onclick="filterBy('type', 'causal')">Causal</button>
            </div>
        </div>

        <div class="claims-container" id="claims">
            {''.join(claims_html)}
        </div>
    </div>

    <script>
        function filterBy(filterType, value) {{
            const cards = document.querySelectorAll('.claim-card');
            cards.forEach(card => {{
                if (value === 'all') {{
                    card.style.display = 'block';
                }} else {{
                    const attr = card.getAttribute('data-' + filterType);
                    card.style.display = attr === value ? 'block' : 'none';
                }}
            }});

            // Update active button
            const buttons = document.querySelectorAll('.filter-btn');
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
"""

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

        return html_content

    def quick_check(self, content: str) -> Dict[str, Any]:
        """
        Quick hallucination check without full evidence.

        Args:
            content: Content to check

        Returns:
            Quick check result dictionary
        """
        prompt = f"""Quickly analyze this content for potential hallucinations and factual accuracy.

Content:
{content[:5000]}

Identify:
1. Claims that seem fabricated or unsupported
2. Statistics or numbers that look suspicious
3. Quotes that may be misattributed
4. Logical inconsistencies
5. Overly specific details that are hallucination-prone

Return as JSON:
{{
    "overall_risk": "critical/high/medium/low",
    "accuracy_estimate": 0.0-1.0,
    "suspicious_claims": [
        {{"text": "claim text", "reason": "why suspicious", "risk": "high/medium/low"}}
    ],
    "likely_hallucinations": ["claim 1", "claim 2"],
    "verification_priorities": ["most important to verify"],
    "summary": "Brief overall assessment"
}}"""

        response = self.llm.generate(prompt)

        try:
            content_str = response.content
            start = content_str.find("{")
            end = content_str.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content_str[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        return {
            "overall_risk": "unknown",
            "accuracy_estimate": 0.5,
            "suspicious_claims": [],
            "likely_hallucinations": [],
            "verification_priorities": ["Full verification needed"],
            "summary": "Quick check incomplete - run full verification",
        }
