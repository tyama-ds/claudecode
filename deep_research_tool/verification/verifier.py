"""
Verifier - Detect potential hallucinations and verify content accuracy.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..evidence.locker import EvidenceLocker


class ConfidenceLevel(str, Enum):
    """Confidence levels for verified claims."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""
    claim_text: str
    confidence: ConfidenceLevel
    source_support: List[str] = field(default_factory=list)
    reasoning: str = ""
    suggestions: str = ""
    is_hallucination_risk: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "claim_text": self.claim_text,
            "confidence": self.confidence.value,
            "source_support": self.source_support,
            "reasoning": self.reasoning,
            "suggestions": self.suggestions,
            "is_hallucination_risk": self.is_hallucination_risk,
        }


@dataclass
class VerificationResult:
    """Complete verification result for a document."""
    document_title: str
    total_claims: int = 0
    verified_claims: List[ClaimVerification] = field(default_factory=list)
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0
    unsupported_count: int = 0
    hallucination_risk_count: int = 0
    overall_reliability_score: float = 0.0
    verification_notes: str = ""
    verified_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_title": self.document_title,
            "total_claims": self.total_claims,
            "verified_claims": [c.to_dict() for c in self.verified_claims],
            "high_confidence_count": self.high_confidence_count,
            "medium_confidence_count": self.medium_confidence_count,
            "low_confidence_count": self.low_confidence_count,
            "unsupported_count": self.unsupported_count,
            "hallucination_risk_count": self.hallucination_risk_count,
            "overall_reliability_score": self.overall_reliability_score,
            "verification_notes": self.verification_notes,
            "verified_at": self.verified_at,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_claims": self.total_claims,
            "reliability_score": f"{self.overall_reliability_score:.1%}",
            "breakdown": {
                "high_confidence": self.high_confidence_count,
                "medium_confidence": self.medium_confidence_count,
                "low_confidence": self.low_confidence_count,
                "unsupported": self.unsupported_count,
            },
            "hallucination_risks": self.hallucination_risk_count,
        }


class Verifier:
    """
    Verify research content for potential hallucinations and accuracy.

    Identifies:
    - Claims without source support
    - Statistical claims that may be inaccurate
    - Temporal claims (dates, events) that are error-prone
    - Causal claims that may be oversimplified
    - Comparative claims that lack evidence
    """

    # Categories of claims prone to hallucination
    HALLUCINATION_PRONE_PATTERNS = [
        "statistics",      # Numbers, percentages, data points
        "dates",           # Specific dates and timelines
        "quotes",          # Attributed quotes
        "causal",          # Cause-effect relationships
        "comparative",     # Comparisons without evidence
        "absolute",        # Absolute statements (always, never, all)
        "predictions",     # Future predictions
        "historical",      # Historical facts
    ]

    def __init__(self, llm_client, language: str = "ja"):
        """
        Initialize Verifier.

        Args:
            llm_client: LLM API client
            language: Target language
        """
        self.llm = llm_client
        self.language = language

    def verify_content(
        self,
        content: str,
        evidence_locker: EvidenceLocker,
        document_title: str = "Research Report",
        strictness: str = "medium",
    ) -> VerificationResult:
        """
        Verify content against evidence and identify potential hallucinations.

        Args:
            content: The content to verify
            evidence_locker: Evidence locker with source information
            document_title: Title of the document being verified
            strictness: Verification strictness (low, medium, high)

        Returns:
            VerificationResult with detailed findings
        """
        result = VerificationResult(document_title=document_title)

        # Step 1: Extract claims from content
        claims = self._extract_claims(content)
        result.total_claims = len(claims)

        # Step 2: Get evidence summaries
        evidence_summaries = self._prepare_evidence_summaries(evidence_locker)

        # Step 3: Verify each claim
        for claim in claims:
            verification = self._verify_claim(
                claim, evidence_summaries, strictness
            )
            result.verified_claims.append(verification)

            # Update counts
            if verification.confidence == ConfidenceLevel.HIGH:
                result.high_confidence_count += 1
            elif verification.confidence == ConfidenceLevel.MEDIUM:
                result.medium_confidence_count += 1
            elif verification.confidence == ConfidenceLevel.LOW:
                result.low_confidence_count += 1
            else:
                result.unsupported_count += 1

            if verification.is_hallucination_risk:
                result.hallucination_risk_count += 1

        # Calculate overall reliability
        if result.total_claims > 0:
            weighted_score = (
                result.high_confidence_count * 1.0 +
                result.medium_confidence_count * 0.7 +
                result.low_confidence_count * 0.3 +
                result.unsupported_count * 0.0
            ) / result.total_claims
            result.overall_reliability_score = weighted_score

        # Generate verification notes
        result.verification_notes = self._generate_verification_notes(result)

        return result

    def _extract_claims(self, content: str) -> List[str]:
        """Extract verifiable claims from content."""
        if self.language == "ja":
            prompt = f"""以下の文章を分析し、検証可能な事実主張をすべて抽出してください。

文章:
{content[:6000]}

各主張について以下を特定してください:
1. 具体的な事実記述
2. 統計的・数値的な主張
3. 引用文
4. 因果関係の主張
5. 歴史的・時間的な主張

JSON配列で返してください:
["主張1", "主張2", ...]

検証可能な事実主張（意見ではなく）に焦点を当ててください。
意味のある主張を10〜20個以上抽出してください。"""
        else:
            prompt = f"""Analyze this content and extract all verifiable factual claims.

Content:
{content[:6000]}

For each claim, identify:
1. Specific factual statements
2. Statistical or numerical claims
3. Quoted statements
4. Causal relationships claimed
5. Historical or temporal claims

Return as a JSON array of claim strings:
["claim 1", "claim 2", ...]

Focus on claims that CAN be verified (factual statements, not opinions).
Extract at least 10-20 meaningful claims if available."""

        response = self.llm.generate(prompt)

        try:
            content = response.content
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: split into sentences as basic claims
        sentences = content.split(".")
        return [s.strip() for s in sentences if len(s.strip()) > 20][:20]

    def _prepare_evidence_summaries(
        self,
        evidence_locker: EvidenceLocker
    ) -> str:
        """Prepare evidence summaries for verification."""
        summaries = []
        for evidence in evidence_locker.get_all_evidence():
            summary = f"""
[{evidence.citation_key}]
URL: {evidence.url}
Title: {evidence.title}
Content: {evidence.content_excerpt[:500]}
Reliability: {evidence.reliability_score:.2f}
"""
            summaries.append(summary)

        return "\n---\n".join(summaries[:20])  # Limit to 20 sources

    def _verify_claim(
        self,
        claim: str,
        evidence_summaries: str,
        strictness: str,
    ) -> ClaimVerification:
        """Verify a single claim against evidence."""
        if self.language == "ja":
            strictness_instruction = {
                "low": "寛容に判定 - 明らかに根拠のない主張のみ指摘する",
                "medium": "バランスよく判定 - 明確な根拠がない主張を指摘する",
                "high": "厳格に判定 - すべての主張に強い根拠を要求する",
            }.get(strictness, "バランスよく判定")

            prompt = f"""以下の主張をエビデンスと照合して検証してください。

主張: {claim}

利用可能なエビデンス:
{evidence_summaries[:4000]}

検証の厳格さ: {strictness_instruction}

分析内容:
1. この主張はエビデンスによって裏付けられていますか？
2. ハルシネーションが起きやすいタイプの主張ですか（統計、日付、引用）？
3. 確信度はどのレベルですか？
4. どのソースがこの主張を支持または矛盾していますか？

JSONで返してください:
{{
    "confidence": "high/medium/low/unsupported/contradicted",
    "source_support": ["ソースキー1", "ソースキー2"],
    "reasoning": "この確信度の理由（日本語）",
    "is_hallucination_risk": true/false,
    "hallucination_type": "statistics/dates/quotes/causal/none",
    "suggestions": "この主張を改善・検証する方法（日本語）"
}}"""
        else:
            strictness_instruction = {
                "low": "Be lenient - only flag clearly unsupported claims",
                "medium": "Be balanced - flag claims without clear evidence",
                "high": "Be strict - require strong evidence for all claims",
            }.get(strictness, "Be balanced")

            prompt = f"""Verify this claim against the available evidence.

Claim: {claim}

Available Evidence:
{evidence_summaries[:4000]}

Verification Strictness: {strictness_instruction}

Analyze:
1. Is this claim supported by the evidence?
2. Is this the type of claim prone to hallucination (statistics, dates, quotes)?
3. What is the confidence level?
4. What sources support or contradict this claim?

Return as JSON:
{{
    "confidence": "high/medium/low/unsupported/contradicted",
    "source_support": ["source_key_1", "source_key_2"],
    "reasoning": "Why this confidence level",
    "is_hallucination_risk": true/false,
    "hallucination_type": "statistics/dates/quotes/causal/none",
    "suggestions": "How to improve or verify this claim"
}}"""

        response = self.llm.generate(prompt)

        try:
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])

                return ClaimVerification(
                    claim_text=claim,
                    confidence=ConfidenceLevel(data.get("confidence", "low")),
                    source_support=data.get("source_support", []),
                    reasoning=data.get("reasoning", ""),
                    suggestions=data.get("suggestions", ""),
                    is_hallucination_risk=data.get("is_hallucination_risk", False),
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback
        return ClaimVerification(
            claim_text=claim,
            confidence=ConfidenceLevel.LOW,
            reasoning="Automatic verification failed",
            is_hallucination_risk=True,
        )

    def _generate_verification_notes(
        self,
        result: VerificationResult
    ) -> str:
        """Generate summary notes about verification."""
        notes = []

        if self.language == "ja":
            if result.overall_reliability_score >= 0.8:
                notes.append("全体信頼度: 高 — 大半の主張が十分に裏付けられています")
            elif result.overall_reliability_score >= 0.6:
                notes.append("全体信頼度: 中 — 多くの主張に追加検証が必要です")
            else:
                notes.append("全体信頼度: 低 — 大幅な検証が必要です")

            if result.hallucination_risk_count > 0:
                notes.append(
                    f"ハルシネーションリスク: {result.hallucination_risk_count}件の主張に"
                    "不正確な情報が含まれている可能性があります"
                )

            if result.unsupported_count > 0:
                notes.append(
                    f"根拠不足の主張: {result.unsupported_count}件の主張にソースエビデンスがありません"
                )
        else:
            if result.overall_reliability_score >= 0.8:
                notes.append("Overall reliability: HIGH - Most claims are well-supported")
            elif result.overall_reliability_score >= 0.6:
                notes.append("Overall reliability: MEDIUM - Many claims need additional verification")
            else:
                notes.append("Overall reliability: LOW - Significant verification needed")

            if result.hallucination_risk_count > 0:
                notes.append(
                    f"Hallucination risks identified: {result.hallucination_risk_count} claims "
                    "may contain inaccurate information"
                )

            if result.unsupported_count > 0:
                notes.append(
                    f"Unsupported claims: {result.unsupported_count} claims lack source evidence"
                )

        return "\n".join(notes)

    def generate_verification_report_html(
        self,
        result: VerificationResult,
        output_path: Path = None,
    ) -> str:
        """
        Generate an HTML verification report.

        Args:
            result: VerificationResult to visualize
            output_path: Optional path to save the HTML file

        Returns:
            HTML content string
        """
        # Color coding for confidence levels
        confidence_colors = {
            "high": "#28a745",      # Green
            "medium": "#ffc107",    # Yellow
            "low": "#fd7e14",       # Orange
            "unsupported": "#dc3545",  # Red
            "contradicted": "#6f42c1",  # Purple
        }

        # Labels
        ja = self.language == "ja"
        lbl_reasoning = "分析理由" if ja else "Reasoning"
        lbl_sources = "ソース" if ja else "Sources"
        lbl_suggestions = "改善提案" if ja else "Suggestions"
        lbl_none_cited = "引用なし" if ja else "None cited"
        lbl_risk_badge = "ハルシネーションリスク" if ja else "Hallucination Risk"

        # Build claims HTML
        claims_html = []
        for claim in result.verified_claims:
            color = confidence_colors.get(claim.confidence.value, "#6c757d")
            risk_badge = (
                f'<span class="badge bg-danger">{lbl_risk_badge}</span>'
                if claim.is_hallucination_risk else ""
            )

            claims_html.append(f"""
            <div class="claim-card" style="border-left: 4px solid {color};">
                <div class="claim-header">
                    <span class="confidence-badge" style="background-color: {color};">
                        {claim.confidence.value.upper()}
                    </span>
                    {risk_badge}
                </div>
                <div class="claim-text">{claim.claim_text}</div>
                <div class="claim-details">
                    <p><strong>{lbl_reasoning}:</strong> {claim.reasoning}</p>
                    <p><strong>{lbl_sources}:</strong> {', '.join(claim.source_support) or lbl_none_cited}</p>
                    <p><strong>{lbl_suggestions}:</strong> {claim.suggestions}</p>
                </div>
            </div>
            """)

        # Calculate percentages for chart
        total = result.total_claims or 1
        chart_data = {
            "high": (result.high_confidence_count / total) * 100,
            "medium": (result.medium_confidence_count / total) * 100,
            "low": (result.low_confidence_count / total) * 100,
            "unsupported": (result.unsupported_count / total) * 100,
        }

        # Section labels
        lbl_title = "検証レポート" if ja else "Verification Report"
        lbl_verified = "検証日時" if ja else "Verified"
        lbl_total = "総主張数" if ja else "Total Claims"
        lbl_reliability = "信頼性スコア" if ja else "Reliability Score"
        lbl_hal_risks = "ハルシネーションリスク" if ja else "Hallucination Risks"
        lbl_unsupported = "根拠不足" if ja else "Unsupported Claims"
        lbl_conf_dist = "確信度分布" if ja else "Confidence Distribution"
        lbl_high_conf = "高確信" if ja else "High Confidence"
        lbl_medium_conf = "中程度" if ja else "Medium"
        lbl_low_conf = "低確信" if ja else "Low"
        lbl_unsup_conf = "根拠不足" if ja else "Unsupported"
        lbl_verified_claims = "検証済み主張" if ja else "Verified Claims"
        lbl_all = "すべて" if ja else "All"
        lbl_notes = "検証ノート" if ja else "Verification Notes"

        html_content = f"""
<!DOCTYPE html>
<html lang="{'ja' if ja else 'en'}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{lbl_title} - {result.document_title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{ margin: 0 0 10px 0; }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-card h3 {{ margin: 0; font-size: 2em; }}
        .summary-card p {{ margin: 5px 0 0 0; color: #666; }}
        .reliability-meter {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .meter-bar {{
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            display: flex;
        }}
        .meter-segment {{
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
            font-weight: bold;
        }}
        .claims-section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .claim-card {{
            background: #f8f9fa;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        .claim-header {{
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }}
        .confidence-badge {{
            padding: 3px 10px;
            border-radius: 3px;
            color: white;
            font-size: 12px;
            font-weight: bold;
        }}
        .badge {{
            padding: 3px 10px;
            border-radius: 3px;
            font-size: 12px;
        }}
        .bg-danger {{ background-color: #dc3545; color: white; }}
        .claim-text {{
            font-size: 1.1em;
            margin-bottom: 10px;
            padding: 10px;
            background: white;
            border-radius: 3px;
        }}
        .claim-details {{
            font-size: 0.9em;
            color: #666;
        }}
        .claim-details p {{ margin: 5px 0; }}
        .notes-section {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }}
        .filter-buttons {{
            margin-bottom: 20px;
        }}
        .filter-btn {{
            padding: 8px 16px;
            margin-right: 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }}
        .filter-btn:hover {{ opacity: 0.8; }}
        @media (max-width: 768px) {{
            .summary-cards {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{lbl_title}</h1>
        <p>{result.document_title}</p>
        <p>{lbl_verified}: {result.verified_at}</p>
    </div>

    <div class="summary-cards">
        <div class="summary-card">
            <h3>{result.total_claims}</h3>
            <p>{lbl_total}</p>
        </div>
        <div class="summary-card">
            <h3 style="color: #28a745;">{result.overall_reliability_score:.1%}</h3>
            <p>{lbl_reliability}</p>
        </div>
        <div class="summary-card">
            <h3 style="color: #dc3545;">{result.hallucination_risk_count}</h3>
            <p>{lbl_hal_risks}</p>
        </div>
        <div class="summary-card">
            <h3 style="color: #fd7e14;">{result.unsupported_count}</h3>
            <p>{lbl_unsupported}</p>
        </div>
    </div>

    <div class="reliability-meter">
        <h3>{lbl_conf_dist}</h3>
        <div class="meter-bar">
            <div class="meter-segment" style="width: {chart_data['high']}%; background-color: #28a745;">
                {result.high_confidence_count}
            </div>
            <div class="meter-segment" style="width: {chart_data['medium']}%; background-color: #ffc107;">
                {result.medium_confidence_count}
            </div>
            <div class="meter-segment" style="width: {chart_data['low']}%; background-color: #fd7e14;">
                {result.low_confidence_count}
            </div>
            <div class="meter-segment" style="width: {chart_data['unsupported']}%; background-color: #dc3545;">
                {result.unsupported_count}
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 10px; font-size: 12px;">
            <span><span style="color: #28a745;">&#9632;</span> {lbl_high_conf}</span>
            <span><span style="color: #ffc107;">&#9632;</span> {lbl_medium_conf}</span>
            <span><span style="color: #fd7e14;">&#9632;</span> {lbl_low_conf}</span>
            <span><span style="color: #dc3545;">&#9632;</span> {lbl_unsup_conf}</span>
        </div>
    </div>

    <div class="claims-section">
        <h2>{lbl_verified_claims}</h2>

        <div class="filter-buttons">
            <button class="filter-btn" style="background: #28a745; color: white;"
                onclick="filterClaims('high')">{lbl_high_conf} ({result.high_confidence_count})</button>
            <button class="filter-btn" style="background: #ffc107;"
                onclick="filterClaims('medium')">{lbl_medium_conf} ({result.medium_confidence_count})</button>
            <button class="filter-btn" style="background: #fd7e14; color: white;"
                onclick="filterClaims('low')">{lbl_low_conf} ({result.low_confidence_count})</button>
            <button class="filter-btn" style="background: #dc3545; color: white;"
                onclick="filterClaims('unsupported')">{lbl_unsup_conf} ({result.unsupported_count})</button>
            <button class="filter-btn" style="background: #6c757d; color: white;"
                onclick="filterClaims('all')">{lbl_all}</button>
        </div>

        <div id="claims-container">
            {''.join(claims_html)}
        </div>

        <div class="notes-section">
            <h4>{lbl_notes}</h4>
            <pre>{result.verification_notes}</pre>
        </div>
    </div>

    <script>
        function filterClaims(level) {{
            const cards = document.querySelectorAll('.claim-card');
            cards.forEach(card => {{
                if (level === 'all') {{
                    card.style.display = 'block';
                }} else {{
                    const badge = card.querySelector('.confidence-badge');
                    if (badge && badge.textContent.trim().toLowerCase() === level) {{
                        card.style.display = 'block';
                    }} else {{
                        card.style.display = 'none';
                    }}
                }}
            }});
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

    def quick_verify(
        self,
        content: str,
        evidence_summaries: str = "",
    ) -> Dict[str, Any]:
        """
        Quick verification without full evidence locker.

        Args:
            content: Content to verify
            evidence_summaries: Optional evidence text

        Returns:
            Quick verification result dictionary
        """
        if self.language == "ja":
            prompt = f"""以下の文章について、ハルシネーションや正確性の問題がないか簡易分析してください。

文章:
{content[:4000]}

{f"エビデンス要約: {evidence_summaries[:2000]}" if evidence_summaries else ""}

以下を特定してください:
1. 根拠が不十分または疑わしい主張
2. 検証が必要な統計・数値
3. 出典確認が必要な引用文
4. 過度に単純化された因果関係の主張
5. 全体的な信頼性評価

JSONで返してください:
{{
    "overall_reliability": "high/medium/low",
    "suspicious_claims": ["疑わしい主張1", "疑わしい主張2"],
    "verification_needed": ["検証が必要な項目"],
    "recommendations": ["推奨事項1"]
}}"""
        else:
            prompt = f"""Quickly analyze this content for potential hallucinations and accuracy issues.

Content:
{content[:4000]}

{f"Evidence summaries: {evidence_summaries[:2000]}" if evidence_summaries else ""}

Identify:
1. Claims that seem unsupported or suspicious
2. Statistics or numbers that should be verified
3. Quotes that need attribution verification
4. Causal claims that may be oversimplified
5. Overall reliability assessment

Return as JSON:
{{
    "overall_reliability": "high/medium/low",
    "suspicious_claims": ["claim 1", "claim 2"],
    "verification_needed": ["item needing verification"],
    "recommendations": ["recommendation 1"]
}}"""

        response = self.llm.generate(prompt)

        try:
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        return {
            "overall_reliability": "unknown",
            "suspicious_claims": [],
            "verification_needed": ["Full verification required"],
            "recommendations": ["Run full verification"],
        }
