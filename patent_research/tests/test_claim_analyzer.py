"""Tests for claim analyzer (unit tests with mock LLM)."""

import pytest
from unittest.mock import MagicMock

from patent_research.research.claim_analyzer import ClaimAnalyzer
from patent_research.models.patent import Patent, PatentClaim


class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class TestClaimAnalyzer:
    def setup_method(self):
        self.mock_llm = MagicMock()
        self.analyzer = ClaimAnalyzer(llm_client=self.mock_llm, language="ja")

    def test_extract_technical_elements(self):
        self.mock_llm.generate.return_value = MockLLMResponse(
            '["リチウムイオン電池", "正極材料", "コバルト酸リチウム"]'
        )

        patent = Patent(
            patent_number="JP2024123456A",
            title="リチウムイオン電池の正極材料",
            claims=[
                PatentClaim(
                    claim_number=1,
                    claim_text="コバルト酸リチウムを含む正極材料",
                    claim_type="independent",
                ),
            ],
        )

        elements = self.analyzer.extract_technical_elements(patent)
        assert len(elements) == 3
        assert "リチウムイオン電池" in elements
        self.mock_llm.generate.assert_called_once()

    def test_extract_elements_no_claims(self):
        patent = Patent(patent_number="JP123", title="Test")
        elements = self.analyzer.extract_technical_elements(patent)
        assert elements == []

    def test_generate_claim_chart(self):
        self.mock_llm.generate.return_value = MockLLMResponse(
            '{"entries": [{"claim_element": "正極材料", "patent_number": "JP456", '
            '"mapping": "対応あり", "confidence": 0.8, "source_excerpt": ""}], '
            '"summary": "テストサマリー"}'
        )

        target = Patent(
            patent_number="JP123",
            title="Target Patent",
            claims=[PatentClaim(1, "Test claim", "independent")],
        )
        refs = [
            Patent(
                patent_number="JP456",
                title="Reference Patent",
                claims=[PatentClaim(1, "Ref claim", "independent")],
            )
        ]

        chart = self.analyzer.generate_claim_chart(target, refs)
        assert chart.target_patent == "JP123"
        assert len(chart.entries) == 1
        assert chart.entries[0].confidence == 0.8
        assert chart.summary == "テストサマリー"

    def test_compare_claims(self):
        self.mock_llm.generate.return_value = MockLLMResponse(
            '{"similarity_score": 0.7, "overlapping_elements": ["要素A"], '
            '"unique_to_a": ["要素B"], "unique_to_b": ["要素C"], '
            '"analysis": "分析結果"}'
        )

        claim_a = PatentClaim(1, "Claim A text", "independent")
        claim_b = PatentClaim(1, "Claim B text", "independent")

        result = self.analyzer.compare_claims(claim_a, claim_b)
        assert result["similarity_score"] == 0.7
        assert "要素A" in result["overlapping_elements"]
