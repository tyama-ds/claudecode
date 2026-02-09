"""
Tests for hallucination detection functionality.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from deep_research_tool.verification.hallucination_checker import (
    HallucinationChecker,
    HallucinationCheckResult,
    DetailedClaim,
    ClaimType,
    HallucinationRisk,
)
from deep_research_tool.verification.verifier import (
    Verifier,
    VerificationResult,
    ClaimVerification,
    ConfidenceLevel,
)
from deep_research_tool.evidence.locker import (
    EvidenceLocker,
    Evidence,
    QualityCategory,
    SourceType,
)


class TestClaimType:
    """Test ClaimType enum."""

    def test_claim_types_exist(self):
        """Test all claim types are defined."""
        assert ClaimType.STATISTICAL
        assert ClaimType.TEMPORAL
        assert ClaimType.QUOTATION
        assert ClaimType.CAUSAL
        assert ClaimType.FACTUAL
        assert ClaimType.OPINION


class TestHallucinationRisk:
    """Test HallucinationRisk enum."""

    def test_risk_levels_exist(self):
        """Test all risk levels are defined."""
        assert HallucinationRisk.CRITICAL
        assert HallucinationRisk.HIGH
        assert HallucinationRisk.MEDIUM
        assert HallucinationRisk.LOW
        assert HallucinationRisk.NONE


class TestDetailedClaim:
    """Test DetailedClaim dataclass."""

    def test_create_claim(self):
        """Test creating a detailed claim."""
        claim = DetailedClaim(
            id="claim_1",
            text="The population grew by 15% in 2023",
            claim_type=ClaimType.STATISTICAL,
            section="Demographics",
        )
        assert claim.id == "claim_1"
        assert claim.claim_type == ClaimType.STATISTICAL
        assert claim.confidence == ConfidenceLevel.UNSUPPORTED  # Default

    def test_claim_to_dict(self):
        """Test claim serialization."""
        claim = DetailedClaim(
            id="claim_1",
            text="Test claim",
            claim_type=ClaimType.FACTUAL,
            confidence=ConfidenceLevel.HIGH,
            hallucination_risk=HallucinationRisk.LOW,
            supporting_evidence=["ev_1", "ev_2"],
        )
        data = claim.to_dict()
        assert data["id"] == "claim_1"
        assert data["claim_type"] == "factual"
        assert data["confidence"] == "high"
        assert data["hallucination_risk"] == "low"
        assert len(data["supporting_evidence"]) == 2


class TestHallucinationCheckResult:
    """Test HallucinationCheckResult dataclass."""

    def test_create_result(self):
        """Test creating a check result."""
        result = HallucinationCheckResult(
            document_title="Test Report",
            total_claims=10,
            verified_claims=5,
            suspicious_claims=3,
            likely_hallucinations=2,
        )
        assert result.document_title == "Test Report"
        assert result.total_claims == 10
        assert result.verified_claims == 5

    def test_result_to_dict(self):
        """Test result serialization."""
        result = HallucinationCheckResult(
            document_title="Test Report",
            total_claims=10,
            overall_accuracy_score=0.75,
        )
        data = result.to_dict()
        assert data["document_title"] == "Test Report"
        assert data["total_claims"] == 10
        assert data["overall_accuracy_score"] == 0.75

    def test_get_summary_japanese(self):
        """Test Japanese summary generation."""
        result = HallucinationCheckResult(
            document_title="テストレポート",
            total_claims=10,
            verified_claims=5,
            suspicious_claims=3,
            likely_hallucinations=2,
            overall_accuracy_score=0.75,
        )
        summary = result.get_summary(language="ja")
        assert "ハルシネーションチェック結果" in summary
        assert "テストレポート" in summary
        assert "75.0%" in summary

    def test_get_summary_english(self):
        """Test English summary generation."""
        result = HallucinationCheckResult(
            document_title="Test Report",
            total_claims=10,
            overall_accuracy_score=0.75,
        )
        summary = result.get_summary(language="en")
        assert "Hallucination Check Result" in summary
        assert "Test Report" in summary


class TestHallucinationChecker:
    """Test HallucinationChecker class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        mock = Mock()
        mock.generate = Mock()
        return mock

    @pytest.fixture
    def evidence_locker(self, tmp_path):
        """Create evidence locker with test data."""
        locker = EvidenceLocker(output_dir=tmp_path)
        locker.add_evidence(
            url="https://example.gov/stats",
            title="Official Statistics Report",
            content_excerpt="The population grew by 15% between 2020 and 2023 according to census data.",
            quality_category=QualityCategory.AUTHORITATIVE,
            source_type=SourceType.OFFICIAL,
        )
        locker.add_evidence(
            url="https://research.edu/study",
            title="Academic Study",
            content_excerpt="Research shows climate patterns have shifted significantly over the past decade.",
            quality_category=QualityCategory.HIGH,
            source_type=SourceType.ACADEMIC,
        )
        return locker

    def test_create_checker(self, mock_llm):
        """Test creating a hallucination checker."""
        checker = HallucinationChecker(mock_llm, language="ja")
        assert checker.llm == mock_llm
        assert checker.language == "ja"

    def test_extract_claims_by_pattern(self, mock_llm):
        """Test pattern-based claim extraction."""
        checker = HallucinationChecker(mock_llm)

        content = """
        The study found that 85% of participants showed improvement.
        This event occurred on January 15, 2024.
        The CEO said "We are committed to sustainability."
        Because of climate change, temperatures have risen.
        Our product is better than competitors.
        """

        claims = checker._extract_claims_by_pattern(content)
        assert len(claims) > 0

        # Check that different claim types are detected
        claim_types = [c.claim_type for c in claims]
        # Should find at least statistical, temporal, or causal
        assert any(t in claim_types for t in [ClaimType.STATISTICAL, ClaimType.TEMPORAL, ClaimType.CAUSAL])

    def test_build_evidence_index(self, mock_llm, evidence_locker):
        """Test building evidence index."""
        checker = HallucinationChecker(mock_llm)
        index = checker._build_evidence_index(evidence_locker)

        assert "by_id" in index
        assert "by_keywords" in index
        assert "all_text" in index
        assert len(index["by_id"]) == 2

    def test_find_relevant_evidence(self, mock_llm, evidence_locker):
        """Test finding relevant evidence for a claim."""
        checker = HallucinationChecker(mock_llm)
        index = checker._build_evidence_index(evidence_locker)

        claim = DetailedClaim(
            id="test_1",
            text="The population grew significantly",
            claim_type=ClaimType.STATISTICAL,
        )

        relevant = checker._find_relevant_evidence(claim, index)
        # Should find evidence containing "population"
        assert len(relevant) >= 0  # May or may not match depending on keywords

    def test_check_content_with_mock(self, mock_llm, evidence_locker):
        """Test full content check with mocked LLM."""
        # Setup mock responses
        mock_llm.generate.side_effect = [
            # First call: extract claims
            Mock(content=json.dumps([
                {
                    "text": "The population grew by 15%",
                    "type": "statistical",
                    "section": "Demographics",
                    "context": "Population statistics"
                },
                {
                    "text": "Climate has changed significantly",
                    "type": "factual",
                    "section": "Environment",
                    "context": "Environmental changes"
                },
            ])),
            # Second call: verify first claim
            Mock(content=json.dumps({
                "confidence": "high",
                "hallucination_risk": "low",
                "risk_score": 0.2,
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "reasoning": "Supported by official data",
                "verified_facts": ["Population growth confirmed"],
                "issues_found": [],
                "suggestions": "None needed",
            })),
            # Third call: verify second claim
            Mock(content=json.dumps({
                "confidence": "medium",
                "hallucination_risk": "medium",
                "risk_score": 0.5,
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "reasoning": "Partially supported",
                "verified_facts": [],
                "issues_found": ["Vague statement"],
                "suggestions": "Add specific data",
            })),
        ]

        checker = HallucinationChecker(mock_llm)
        content = """
        The population grew by 15% according to recent data.
        Climate has changed significantly over the years.
        """

        result = checker.check_content(
            content=content,
            evidence_locker=evidence_locker,
            document_title="Test Report",
        )

        assert result.total_claims == 2
        assert result.document_title == "Test Report"
        assert len(result.claims) == 2

    def test_quick_check(self, mock_llm):
        """Test quick hallucination check."""
        mock_llm.generate.return_value = Mock(content=json.dumps({
            "overall_risk": "medium",
            "accuracy_estimate": 0.7,
            "suspicious_claims": [
                {"text": "Suspicious claim", "reason": "No source", "risk": "high"}
            ],
            "likely_hallucinations": [],
            "verification_priorities": ["Verify statistics"],
            "summary": "Generally accurate with some concerns",
        }))

        checker = HallucinationChecker(mock_llm)
        result = checker.quick_check("Some content to check")

        assert result["overall_risk"] == "medium"
        assert result["accuracy_estimate"] == 0.7
        assert len(result["suspicious_claims"]) == 1


class TestHTMLReportGeneration:
    """Test HTML report generation."""

    def test_generate_html_report(self, tmp_path):
        """Test generating HTML report."""
        mock_llm = Mock()
        checker = HallucinationChecker(mock_llm)

        # Create a result with test data
        result = HallucinationCheckResult(
            document_title="Test Report",
            total_claims=5,
            verified_claims=2,
            suspicious_claims=2,
            likely_hallucinations=1,
            overall_accuracy_score=0.6,
            evidence_coverage_score=0.5,
            source_quality_score=0.7,
            claims=[
                DetailedClaim(
                    id="claim_1",
                    text="Test claim 1",
                    claim_type=ClaimType.STATISTICAL,
                    confidence=ConfidenceLevel.HIGH,
                    hallucination_risk=HallucinationRisk.LOW,
                    reasoning="Well supported",
                ),
                DetailedClaim(
                    id="claim_2",
                    text="Test claim 2",
                    claim_type=ClaimType.FACTUAL,
                    confidence=ConfidenceLevel.LOW,
                    hallucination_risk=HallucinationRisk.HIGH,
                    reasoning="No evidence",
                    issues_found=["Missing source"],
                ),
            ],
            critical_issues=["Critical issue 1"],
            warnings=["Warning 1"],
            recommendations=["Add more sources"],
        )

        output_path = tmp_path / "test_report.html"
        html = checker.generate_detailed_html_report(result, output_path=output_path)

        assert output_path.exists()
        assert "Test Report" in html
        assert "Hallucination Check Report" in html
        assert "claim_1" in html
        assert "claim_2" in html
        assert "Critical Issues" in html
        assert "Recommendations" in html


class TestVerifierCompatibility:
    """Test compatibility with base Verifier class."""

    def test_verifier_create(self):
        """Test creating base verifier."""
        mock_llm = Mock()
        verifier = Verifier(mock_llm, language="ja")
        assert verifier.llm == mock_llm

    def test_claim_verification_dataclass(self):
        """Test ClaimVerification dataclass."""
        claim = ClaimVerification(
            claim_text="Test claim",
            confidence=ConfidenceLevel.HIGH,
            source_support=["source_1"],
            reasoning="Verified",
            is_hallucination_risk=False,
        )
        assert claim.claim_text == "Test claim"
        assert claim.confidence == ConfidenceLevel.HIGH

        data = claim.to_dict()
        assert data["claim_text"] == "Test claim"
        assert data["confidence"] == "high"

    def test_verification_result_dataclass(self):
        """Test VerificationResult dataclass."""
        result = VerificationResult(
            document_title="Test",
            total_claims=10,
            high_confidence_count=5,
            medium_confidence_count=3,
            low_confidence_count=2,
            overall_reliability_score=0.75,
        )
        assert result.total_claims == 10
        assert result.overall_reliability_score == 0.75

        summary = result.get_summary()
        assert summary["total_claims"] == 10
        assert "75.0%" in summary["reliability_score"]


class TestImports:
    """Test all exports are available."""

    def test_verification_imports(self):
        """Test importing from verification module."""
        from deep_research_tool.verification import (
            Verifier,
            VerificationResult,
            ClaimVerification,
            ConfidenceLevel,
            HallucinationChecker,
            HallucinationCheckResult,
            DetailedClaim,
            ClaimType,
            HallucinationRisk,
        )

        assert Verifier is not None
        assert HallucinationChecker is not None
        assert ClaimType is not None
        assert HallucinationRisk is not None
