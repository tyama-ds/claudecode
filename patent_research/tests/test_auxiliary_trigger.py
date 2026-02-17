"""Tests for auxiliary search trigger logic."""

import pytest
from unittest.mock import MagicMock

from patent_research.research.auxiliary_trigger import (
    AuxiliaryTrigger,
    TriggerResult,
    TriggerItem,
)
from patent_research.config import AuxiliarySearchConfig
from patent_research.models.patent import Patent, PatentClaim


class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class TestTriggerResult:
    def test_needs_academic_search_with_technical_terms(self):
        result = TriggerResult(
            patent_number="JP123",
            technical_terms=[
                TriggerItem(term="CNT", context="Carbon nanotube", search_query="CNT 論文"),
            ],
        )
        assert result.needs_academic_search is True
        assert result.needs_business_search is False

    def test_needs_business_search(self):
        result = TriggerResult(
            patent_number="JP123",
            business_indicators=[
                TriggerItem(term="市場規模", context="Test", search_query="CNT 市場規模"),
            ],
        )
        assert result.needs_business_search is True

    def test_no_triggers_needed(self):
        result = TriggerResult(patent_number="JP123")
        assert result.needs_academic_search is False
        assert result.needs_business_search is False

    def test_get_academic_queries(self):
        result = TriggerResult(
            technical_terms=[
                TriggerItem(term="A", context="", search_query="query A"),
            ],
            academic_references=[
                TriggerItem(term="B", context="", search_query="query B"),
            ],
        )
        queries = result.get_academic_queries()
        assert "query A" in queries
        assert "query B" in queries


class TestAuxiliaryTrigger:
    def setup_method(self):
        self.mock_llm = MagicMock()
        self.config = AuxiliarySearchConfig()
        self.trigger = AuxiliaryTrigger(
            llm_client=self.mock_llm,
            config=self.config,
            language="ja",
        )

    def test_analyze_patent_with_triggers(self):
        self.mock_llm.generate.return_value = MockLLMResponse(
            '{"technical_terms": [{"term": "CNT", "context": "Carbon nanotube", '
            '"search_query": "カーボンナノチューブ 論文", "confidence": 0.8}], '
            '"academic_references": [], '
            '"business_indicators": [{"term": "市場規模", "context": "EV市場", '
            '"search_query": "EV電池 市場規模", "confidence": 0.7}], '
            '"standards_references": []}'
        )

        patent = Patent(
            patent_number="JP123",
            title="CNT電極材料",
            abstract="カーボンナノチューブを用いた電極材料",
            claims=[PatentClaim(1, "CNTを含む電極", "independent")],
        )

        result = self.trigger.analyze_patent(patent)
        assert result.needs_academic_search is True
        assert result.needs_business_search is True
        assert len(result.technical_terms) == 1
        assert result.technical_terms[0].term == "CNT"

    def test_analyze_patent_below_threshold(self):
        self.mock_llm.generate.return_value = MockLLMResponse(
            '{"technical_terms": [{"term": "Tech", "context": "Low confidence", '
            '"search_query": "query", "confidence": 0.3}], '
            '"academic_references": [], '
            '"business_indicators": [{"term": "Sales", "context": "Low", '
            '"search_query": "query", "confidence": 0.2}], '
            '"standards_references": []}'
        )

        patent = Patent(patent_number="JP123", title="Test")
        result = self.trigger.analyze_patent(patent)
        # Below thresholds, so should not trigger
        assert len(result.technical_terms) == 0  # 0.3 < 0.6 threshold
        assert len(result.business_indicators) == 0  # 0.2 < 0.5 threshold

    def test_aggregate_triggers(self):
        results = [
            TriggerResult(
                patent_number="JP1",
                technical_terms=[
                    TriggerItem(term="A", context="", search_query="query A"),
                ],
            ),
            TriggerResult(
                patent_number="JP2",
                technical_terms=[
                    TriggerItem(term="B", context="", search_query="query B"),
                    TriggerItem(term="A", context="", search_query="query A"),  # Duplicate
                ],
            ),
        ]

        combined = self.trigger.aggregate_triggers(results)
        # Should deduplicate by search_query
        assert len(combined.technical_terms) == 2

    def test_analyze_patent_llm_failure(self):
        self.mock_llm.generate.side_effect = Exception("LLM error")
        patent = Patent(patent_number="JP123", title="Test")
        result = self.trigger.analyze_patent(patent)
        # Should return empty result, not raise
        assert result.patent_number == "JP123"
        assert len(result.technical_terms) == 0


class TestPatentResearcherConfig:
    def test_default_patent_researcher_config(self):
        """Test that PatentResearchSession can be created."""
        from patent_research.research.patent_researcher import PatentResearchSession

        session = PatentResearchSession(query="test query")
        assert session.query == "test query"
        assert session.state == "initialized"
        assert len(session.patents_found) == 0

    def test_session_to_dict(self):
        from patent_research.research.patent_researcher import PatentResearchSession

        session = PatentResearchSession(query="test", requirements="req")
        data = session.to_dict()
        assert data["query"] == "test"
        assert data["requirements"] == "req"
        assert data["state"] == "initialized"
