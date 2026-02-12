"""
Fact verification engine for assessing claim accuracy.

Uses a separate LLM session for verification to ensure independence.
Provides detailed labeling and correction suggestions.

Labels:
- SUPPORTED: Claim is supported by gathered evidence
- NOT_TRUE: Claim contradicts gathered evidence
- NOT_VERIFIED: Could not find evidence to verify/refute
- SUSPICIOUS: Claim appears questionable but not definitively wrong
- PARTIALLY_TRUE: Claim is partially correct with some inaccuracies
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from .claim_extractor import Claim, ClaimType
from .web_crawler import CrawlResult


class VerificationLabel(str, Enum):
    """Verification result labels."""
    SUPPORTED = "supported"
    NOT_TRUE = "not_true"
    NOT_VERIFIED = "not_verified"
    SUSPICIOUS = "suspicious"
    PARTIALLY_TRUE = "partially_true"
    OPINION = "opinion"  # Not verifiable (opinion/subjective)


class EvidenceStrength(str, Enum):
    """Strength of supporting evidence."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"
    CONTRADICTORY = "contradictory"


@dataclass
class EvidenceMatch:
    """Evidence that relates to a claim."""
    source_url: str
    source_title: str
    relevant_text: str
    support_type: str  # "supports", "contradicts", "partially_supports", "unrelated"
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrectionSuggestion:
    """Suggested correction for an inaccurate claim."""
    original_claim: str
    corrected_claim: str
    correction_type: str  # "factual", "precision", "context", "wording"
    explanation: str
    confidence: float
    sources: List[str] = field(default_factory=list)


@dataclass
class SentenceVerificationResult:
    """Verification result for a single sentence/claim."""
    claim: Claim
    label: VerificationLabel
    confidence: float
    evidence_strength: EvidenceStrength
    evidence_matches: List[EvidenceMatch] = field(default_factory=list)
    correction: Optional[CorrectionSuggestion] = None
    reasoning: str = ""
    search_queries_used: List[str] = field(default_factory=list)
    verification_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "claim_text": self.claim.text,
            "claim_type": self.claim.claim_type.value,
            "source_sentence": self.claim.source_sentence.text,
            "sentence_index": self.claim.source_sentence.index,
            "label": self.label.value,
            "confidence": self.confidence,
            "evidence_strength": self.evidence_strength.value,
            "evidence_matches": [
                {
                    "source_url": e.source_url,
                    "source_title": e.source_title,
                    "relevant_text": e.relevant_text,
                    "support_type": e.support_type,
                    "relevance_score": e.relevance_score,
                }
                for e in self.evidence_matches
            ],
            "correction": {
                "original": self.correction.original_claim,
                "corrected": self.correction.corrected_claim,
                "type": self.correction.correction_type,
                "explanation": self.correction.explanation,
                "confidence": self.correction.confidence,
                "sources": self.correction.sources,
            } if self.correction else None,
            "reasoning": self.reasoning,
            "search_queries_used": self.search_queries_used,
            "timestamp": self.verification_timestamp,
        }


@dataclass
class FactCheckReport:
    """Complete fact-checking report for a document."""
    document_title: str
    total_sentences: int
    total_claims: int
    results: List[SentenceVerificationResult] = field(default_factory=list)

    # Summary statistics
    supported_count: int = 0
    not_true_count: int = 0
    not_verified_count: int = 0
    suspicious_count: int = 0
    partially_true_count: int = 0
    opinion_count: int = 0

    overall_accuracy_score: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_title": self.document_title,
            "total_sentences": self.total_sentences,
            "total_claims": self.total_claims,
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "supported": self.supported_count,
                "not_true": self.not_true_count,
                "not_verified": self.not_verified_count,
                "suspicious": self.suspicious_count,
                "partially_true": self.partially_true_count,
                "opinion": self.opinion_count,
            },
            "overall_accuracy_score": self.overall_accuracy_score,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    def calculate_statistics(self):
        """Calculate summary statistics from results."""
        self.supported_count = sum(1 for r in self.results if r.label == VerificationLabel.SUPPORTED)
        self.not_true_count = sum(1 for r in self.results if r.label == VerificationLabel.NOT_TRUE)
        self.not_verified_count = sum(1 for r in self.results if r.label == VerificationLabel.NOT_VERIFIED)
        self.suspicious_count = sum(1 for r in self.results if r.label == VerificationLabel.SUSPICIOUS)
        self.partially_true_count = sum(1 for r in self.results if r.label == VerificationLabel.PARTIALLY_TRUE)
        self.opinion_count = sum(1 for r in self.results if r.label == VerificationLabel.OPINION)

        # Calculate accuracy score (excluding opinions)
        verifiable_count = self.total_claims - self.opinion_count
        if verifiable_count > 0:
            accurate = self.supported_count + (self.partially_true_count * 0.5)
            self.overall_accuracy_score = accurate / verifiable_count


class FactVerifier:
    """
    Fact verification engine.

    Uses a separate LLM session to verify claims against gathered evidence.
    Supports multiple LLM providers (OpenAI, Anthropic).
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,  # Lower temperature for more consistent verification
        language: str = "ja",
    ):
        """
        Initialize fact verifier with a separate LLM session.

        Args:
            llm_provider: LLM provider ("openai" or "anthropic")
            api_key: API key (uses env var if not provided)
            model: Model name (uses default for provider if not provided)
            temperature: Sampling temperature (lower = more consistent)
            language: Output language
        """
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.language = language
        self._llm_client = None

    def _get_llm_client(self):
        """Get or create LLM client (separate session)."""
        if self._llm_client is None:
            self._llm_client = self._create_llm_client()
        return self._llm_client

    def _create_llm_client(self):
        """Create a new LLM client instance."""
        if self.llm_provider == "openai":
            return self._create_openai_client()
        elif self.llm_provider == "anthropic":
            return self._create_anthropic_client()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _create_openai_client(self):
        """Create OpenAI client."""
        from ..api.openai_client import OpenAIClient

        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        model = self.model or "gpt-4o-mini"

        return OpenAIClient(
            api_key=api_key,
            model=model,
            temperature=self.temperature,
        )

    def _create_anthropic_client(self):
        """Create Anthropic client."""
        from ..api.anthropic_client import AnthropicClient

        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        model = self.model or "claude-3-5-sonnet-20241022"

        return AnthropicClient(
            api_key=api_key,
            model=model,
            temperature=self.temperature,
        )

    def verify_claim(
        self,
        claim: Claim,
        evidence: List[CrawlResult],
    ) -> SentenceVerificationResult:
        """
        Verify a single claim against gathered evidence.

        Args:
            claim: The claim to verify
            evidence: List of crawled pages as evidence

        Returns:
            SentenceVerificationResult with verification details
        """
        # Skip non-verifiable claims
        if not claim.is_verifiable:
            return SentenceVerificationResult(
                claim=claim,
                label=VerificationLabel.OPINION,
                confidence=0.9,
                evidence_strength=EvidenceStrength.NONE,
                reasoning="This claim is not verifiable (opinion/subjective statement).",
            )

        # Prepare evidence text
        evidence_text = self._prepare_evidence_text(evidence)

        # Use LLM to verify
        llm = self._get_llm_client()

        prompt = self._build_verification_prompt(claim, evidence_text)
        response = llm.generate(prompt)

        # Parse response
        return self._parse_verification_response(response.content, claim, evidence)

    def _prepare_evidence_text(self, evidence: List[CrawlResult], max_chars: int = 8000) -> str:
        """Prepare evidence text for verification prompt."""
        evidence_parts = []
        total_chars = 0

        for i, ev in enumerate(evidence):
            if total_chars >= max_chars:
                break

            part = f"""
[Source {i+1}]
URL: {ev.url}
Title: {ev.title}
Content:
{ev.clean_text[:2000]}
"""
            evidence_parts.append(part)
            total_chars += len(part)

        return "\n---\n".join(evidence_parts)

    def _build_verification_prompt(self, claim: Claim, evidence_text: str) -> str:
        """Build the verification prompt for LLM."""
        claim_type_hint = {
            ClaimType.FACTUAL: "a factual statement",
            ClaimType.STATISTICAL: "a statistical/numerical claim",
            ClaimType.QUOTATION: "a quote or attributed statement",
            ClaimType.TEMPORAL: "a temporal/date-related claim",
            ClaimType.CAUSAL: "a cause-effect relationship claim",
            ClaimType.COMPARATIVE: "a comparative claim",
            ClaimType.DEFINITION: "a definition or explanation",
        }.get(claim.claim_type, "a claim")

        lang_instruction = ""
        if self.language == "ja":
            lang_instruction = "Respond in Japanese for the reasoning and correction fields."
        elif self.language == "en":
            lang_instruction = "Respond in English for the reasoning and correction fields."

        prompt = f"""You are a fact-checker. Verify the following claim against the provided evidence.

CLAIM TO VERIFY:
"{claim.text}"

This appears to be {claim_type_hint}.

ORIGINAL SENTENCE:
"{claim.source_sentence.text}"

GATHERED EVIDENCE:
{evidence_text}

VERIFICATION TASK:
1. Determine if the claim is SUPPORTED, NOT_TRUE, NOT_VERIFIED, SUSPICIOUS, or PARTIALLY_TRUE
2. Identify relevant evidence that supports or contradicts the claim
3. If the claim is inaccurate or suspicious, suggest a correction
4. Provide clear reasoning for your verdict

{lang_instruction}

Return your analysis as JSON:
{{
    "label": "supported|not_true|not_verified|suspicious|partially_true",
    "confidence": 0.0-1.0,
    "evidence_strength": "strong|moderate|weak|none|contradictory",
    "evidence_matches": [
        {{
            "source_index": 1,
            "relevant_text": "The relevant quote from the source",
            "support_type": "supports|contradicts|partially_supports|unrelated",
            "relevance_score": 0.0-1.0
        }}
    ],
    "reasoning": "Detailed explanation of why this verdict was given",
    "correction": {{
        "needed": true|false,
        "corrected_claim": "The corrected version of the claim (if needed)",
        "correction_type": "factual|precision|context|wording",
        "explanation": "Why this correction is needed",
        "confidence": 0.0-1.0
    }}
}}

Be strict and skeptical. Only mark as SUPPORTED if there is clear evidence.
If evidence is ambiguous or insufficient, use NOT_VERIFIED.
If the claim appears false but you're not 100% certain, use SUSPICIOUS."""

        return prompt

    def _parse_verification_response(
        self,
        response_content: str,
        claim: Claim,
        evidence: List[CrawlResult],
    ) -> SentenceVerificationResult:
        """Parse LLM response into verification result."""
        try:
            # Extract JSON from response
            start = response_content.find('{')
            end = response_content.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(response_content[start:end])
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError):
            # Fallback: could not parse, mark as not verified
            return SentenceVerificationResult(
                claim=claim,
                label=VerificationLabel.NOT_VERIFIED,
                confidence=0.5,
                evidence_strength=EvidenceStrength.NONE,
                reasoning="Failed to parse verification response. Manual review recommended.",
            )

        # Parse label
        try:
            label = VerificationLabel(data.get('label', 'not_verified'))
        except ValueError:
            label = VerificationLabel.NOT_VERIFIED

        # Parse evidence strength
        try:
            evidence_strength = EvidenceStrength(data.get('evidence_strength', 'none'))
        except ValueError:
            evidence_strength = EvidenceStrength.NONE

        # Parse evidence matches
        evidence_matches = []
        for match_data in data.get('evidence_matches', []):
            source_idx = match_data.get('source_index', 0) - 1
            if 0 <= source_idx < len(evidence):
                ev = evidence[source_idx]
                evidence_matches.append(EvidenceMatch(
                    source_url=ev.url,
                    source_title=ev.title,
                    relevant_text=match_data.get('relevant_text', ''),
                    support_type=match_data.get('support_type', 'unrelated'),
                    relevance_score=match_data.get('relevance_score', 0.0),
                ))

        # Parse correction
        correction = None
        correction_data = data.get('correction', {})
        if correction_data.get('needed', False):
            correction = CorrectionSuggestion(
                original_claim=claim.text,
                corrected_claim=correction_data.get('corrected_claim', ''),
                correction_type=correction_data.get('correction_type', 'factual'),
                explanation=correction_data.get('explanation', ''),
                confidence=correction_data.get('confidence', 0.0),
                sources=[em.source_url for em in evidence_matches if em.support_type != 'unrelated'],
            )

        return SentenceVerificationResult(
            claim=claim,
            label=label,
            confidence=data.get('confidence', 0.5),
            evidence_strength=evidence_strength,
            evidence_matches=evidence_matches,
            correction=correction,
            reasoning=data.get('reasoning', ''),
            search_queries_used=claim.search_queries,
        )

    def verify_claims(
        self,
        claims: List[Claim],
        evidence_map: Dict[int, List[CrawlResult]],
    ) -> List[SentenceVerificationResult]:
        """
        Verify multiple claims.

        Args:
            claims: List of claims to verify
            evidence_map: Dictionary mapping claim index to its evidence

        Returns:
            List of verification results
        """
        results = []

        for i, claim in enumerate(claims):
            evidence = evidence_map.get(i, [])
            result = self.verify_claim(claim, evidence)
            results.append(result)

        return results

    def generate_report(
        self,
        results: List[SentenceVerificationResult],
        document_title: str = "Fact Check Report",
        total_sentences: int = 0,
    ) -> FactCheckReport:
        """
        Generate a fact-check report from verification results.

        Args:
            results: List of verification results
            document_title: Title of the document being checked
            total_sentences: Total number of sentences in document

        Returns:
            FactCheckReport with summary statistics
        """
        report = FactCheckReport(
            document_title=document_title,
            total_sentences=total_sentences or len(results),
            total_claims=len(results),
            results=results,
        )

        report.calculate_statistics()

        return report

    def export_report_json(self, report: FactCheckReport, filepath: str):
        """Export report to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    def export_report_html(self, report: FactCheckReport, filepath: str):
        """Export report to HTML file."""
        html_content = self._generate_html_report(report)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_html_report(self, report: FactCheckReport) -> str:
        """Generate HTML report."""
        label_colors = {
            VerificationLabel.SUPPORTED: "#28a745",
            VerificationLabel.NOT_TRUE: "#dc3545",
            VerificationLabel.NOT_VERIFIED: "#6c757d",
            VerificationLabel.SUSPICIOUS: "#fd7e14",
            VerificationLabel.PARTIALLY_TRUE: "#ffc107",
            VerificationLabel.OPINION: "#17a2b8",
        }

        label_icons = {
            VerificationLabel.SUPPORTED: "✓",
            VerificationLabel.NOT_TRUE: "✗",
            VerificationLabel.NOT_VERIFIED: "?",
            VerificationLabel.SUSPICIOUS: "⚠",
            VerificationLabel.PARTIALLY_TRUE: "~",
            VerificationLabel.OPINION: "💭",
        }

        # Build results HTML
        results_html = []
        for result in report.results:
            color = label_colors.get(result.label, "#6c757d")
            icon = label_icons.get(result.label, "?")

            evidence_html = ""
            if result.evidence_matches:
                evidence_items = "".join([
                    f'<li><a href="{e.source_url}" target="_blank">{e.source_title}</a>: '
                    f'<em>"{e.relevant_text[:200]}..."</em> ({e.support_type})</li>'
                    for e in result.evidence_matches[:3]
                ])
                evidence_html = f'<ul class="evidence-list">{evidence_items}</ul>'

            correction_html = ""
            if result.correction:
                correction_html = f'''
                <div class="correction">
                    <strong>提案された訂正:</strong><br>
                    <del>{result.correction.original_claim}</del><br>
                    <ins>{result.correction.corrected_claim}</ins><br>
                    <small>{result.correction.explanation}</small>
                </div>
                '''

            results_html.append(f'''
            <div class="result-card" style="border-left: 4px solid {color};">
                <div class="result-header">
                    <span class="label-badge" style="background-color: {color};">
                        {icon} {result.label.value.upper()}
                    </span>
                    <span class="confidence">信頼度: {result.confidence:.0%}</span>
                </div>
                <div class="claim-text">
                    <strong>主張:</strong> {result.claim.text}
                </div>
                <div class="source-sentence">
                    <strong>元の文:</strong> {result.claim.source_sentence.text}
                </div>
                <div class="reasoning">
                    <strong>判定理由:</strong> {result.reasoning}
                </div>
                {evidence_html}
                {correction_html}
            </div>
            ''')

        # Calculate chart data
        total = report.total_claims or 1
        chart_data = {
            'supported': (report.supported_count / total) * 100,
            'not_true': (report.not_true_count / total) * 100,
            'not_verified': (report.not_verified_count / total) * 100,
            'suspicious': (report.suspicious_count / total) * 100,
            'partially_true': (report.partially_true_count / total) * 100,
            'opinion': (report.opinion_count / total) * 100,
        }

        html = f'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ファクトチェックレポート - {report.document_title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Hiragino Sans', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{ margin: 0 0 10px 0; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
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
        .summary-card p {{ margin: 5px 0 0 0; color: #666; font-size: 0.9em; }}
        .accuracy-score {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .accuracy-score h2 {{
            font-size: 3em;
            color: {self._get_accuracy_color(report.overall_accuracy_score)};
            margin: 0;
        }}
        .meter-bar {{
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            display: flex;
            margin: 20px 0;
        }}
        .meter-segment {{
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 11px;
            font-weight: bold;
        }}
        .results-section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .result-card {{
            background: #f8f9fa;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        .result-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .label-badge {{
            padding: 5px 15px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .confidence {{
            color: #666;
            font-size: 0.9em;
        }}
        .claim-text, .source-sentence, .reasoning {{
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-radius: 5px;
        }}
        .evidence-list {{
            font-size: 0.9em;
            color: #555;
            padding-left: 20px;
        }}
        .evidence-list li {{
            margin: 5px 0;
        }}
        .correction {{
            margin-top: 15px;
            padding: 15px;
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 5px;
        }}
        .correction del {{
            color: #dc3545;
            background: #ffe6e6;
            padding: 2px 5px;
        }}
        .correction ins {{
            color: #28a745;
            background: #e6ffe6;
            padding: 2px 5px;
            text-decoration: none;
        }}
        .filter-buttons {{
            margin-bottom: 20px;
        }}
        .filter-btn {{
            padding: 8px 16px;
            margin: 5px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            color: white;
        }}
        .filter-btn:hover {{ opacity: 0.8; }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 15px;
            margin-top: 10px;
            font-size: 0.85em;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 15px;
            height: 15px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📋 ファクトチェックレポート</h1>
        <p>{report.document_title}</p>
        <p>作成日時: {report.created_at}</p>
    </div>

    <div class="accuracy-score">
        <p>全体の正確性スコア</p>
        <h2>{report.overall_accuracy_score:.1%}</h2>
        <p>（意見・主観的主張を除く{report.total_claims - report.opinion_count}件の主張を評価）</p>
    </div>

    <div class="summary-grid">
        <div class="summary-card">
            <h3 style="color: #28a745;">{report.supported_count}</h3>
            <p>✓ 裏付けあり</p>
        </div>
        <div class="summary-card">
            <h3 style="color: #dc3545;">{report.not_true_count}</h3>
            <p>✗ 事実と異なる</p>
        </div>
        <div class="summary-card">
            <h3 style="color: #ffc107;">{report.partially_true_count}</h3>
            <p>~ 部分的に正確</p>
        </div>
        <div class="summary-card">
            <h3 style="color: #fd7e14;">{report.suspicious_count}</h3>
            <p>⚠ 疑わしい</p>
        </div>
        <div class="summary-card">
            <h3 style="color: #6c757d;">{report.not_verified_count}</h3>
            <p>? 未検証</p>
        </div>
        <div class="summary-card">
            <h3 style="color: #17a2b8;">{report.opinion_count}</h3>
            <p>💭 意見・主観</p>
        </div>
    </div>

    <div class="accuracy-score">
        <h4>判定分布</h4>
        <div class="meter-bar">
            <div class="meter-segment" style="width: {chart_data['supported']}%; background-color: #28a745;">
                {report.supported_count if report.supported_count else ''}
            </div>
            <div class="meter-segment" style="width: {chart_data['partially_true']}%; background-color: #ffc107;">
                {report.partially_true_count if report.partially_true_count else ''}
            </div>
            <div class="meter-segment" style="width: {chart_data['suspicious']}%; background-color: #fd7e14;">
                {report.suspicious_count if report.suspicious_count else ''}
            </div>
            <div class="meter-segment" style="width: {chart_data['not_true']}%; background-color: #dc3545;">
                {report.not_true_count if report.not_true_count else ''}
            </div>
            <div class="meter-segment" style="width: {chart_data['not_verified']}%; background-color: #6c757d;">
                {report.not_verified_count if report.not_verified_count else ''}
            </div>
            <div class="meter-segment" style="width: {chart_data['opinion']}%; background-color: #17a2b8;">
                {report.opinion_count if report.opinion_count else ''}
            </div>
        </div>
        <div class="legend">
            <span class="legend-item"><span class="legend-color" style="background: #28a745;"></span> 裏付けあり</span>
            <span class="legend-item"><span class="legend-color" style="background: #ffc107;"></span> 部分的に正確</span>
            <span class="legend-item"><span class="legend-color" style="background: #fd7e14;"></span> 疑わしい</span>
            <span class="legend-item"><span class="legend-color" style="background: #dc3545;"></span> 事実と異なる</span>
            <span class="legend-item"><span class="legend-color" style="background: #6c757d;"></span> 未検証</span>
            <span class="legend-item"><span class="legend-color" style="background: #17a2b8;"></span> 意見</span>
        </div>
    </div>

    <div class="results-section">
        <h2>詳細な検証結果</h2>

        <div class="filter-buttons">
            <button class="filter-btn" style="background: #28a745;" onclick="filterResults('supported')">
                裏付けあり ({report.supported_count})
            </button>
            <button class="filter-btn" style="background: #dc3545;" onclick="filterResults('not_true')">
                事実と異なる ({report.not_true_count})
            </button>
            <button class="filter-btn" style="background: #ffc107; color: #333;" onclick="filterResults('partially_true')">
                部分的 ({report.partially_true_count})
            </button>
            <button class="filter-btn" style="background: #fd7e14;" onclick="filterResults('suspicious')">
                疑わしい ({report.suspicious_count})
            </button>
            <button class="filter-btn" style="background: #6c757d;" onclick="filterResults('not_verified')">
                未検証 ({report.not_verified_count})
            </button>
            <button class="filter-btn" style="background: #333;" onclick="filterResults('all')">
                すべて表示
            </button>
        </div>

        <div id="results-container">
            {''.join(results_html)}
        </div>
    </div>

    <script>
        function filterResults(label) {{
            const cards = document.querySelectorAll('.result-card');
            cards.forEach(card => {{
                if (label === 'all') {{
                    card.style.display = 'block';
                }} else {{
                    const badge = card.querySelector('.label-badge');
                    if (badge && badge.textContent.toLowerCase().includes(label.replace('_', ' '))) {{
                        card.style.display = 'block';
                    }} else {{
                        card.style.display = 'none';
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>'''

        return html

    def _get_accuracy_color(self, score: float) -> str:
        """Get color based on accuracy score."""
        if score >= 0.8:
            return "#28a745"  # Green
        elif score >= 0.6:
            return "#ffc107"  # Yellow
        elif score >= 0.4:
            return "#fd7e14"  # Orange
        else:
            return "#dc3545"  # Red
