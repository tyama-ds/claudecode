"""Tests for patent data models."""

import pytest
from patent_research.models.patent import (
    Patent,
    PatentClaim,
    IPCClassification,
    PatentFamily,
    normalize_patent_number,
)
from patent_research.models.analysis import (
    ClaimChart,
    ClaimChartEntry,
    TechnologyLandscape,
    PriorArtRecord,
)
from patent_research.models.search_result import PatentSearchResult


class TestIPCClassification:
    def test_parse_full_code(self):
        ipc = IPCClassification(full_code="H01L21/027")
        assert ipc.section == "H"
        assert ipc.class_code == "H01"
        assert ipc.subclass == "H01L"
        assert ipc.main_group == "H01L21"
        assert ipc.subgroup == "H01L21/027"

    def test_to_dict_round_trip(self):
        ipc = IPCClassification(full_code="G06F3/041")
        data = ipc.to_dict()
        restored = IPCClassification.from_dict(data)
        assert restored.full_code == "G06F3/041"
        assert restored.section == "G"


class TestPatentClaim:
    def test_independent_claim(self):
        claim = PatentClaim(
            claim_number=1,
            claim_text="A method for...",
            claim_type="independent",
        )
        assert claim.claim_type == "independent"
        assert claim.depends_on is None

    def test_dependent_claim(self):
        claim = PatentClaim(
            claim_number=2,
            claim_text="The method of claim 1...",
            claim_type="dependent",
            depends_on=1,
        )
        assert claim.claim_type == "dependent"
        assert claim.depends_on == 1


class TestNormalizePatentNumber:
    def test_jp_with_hyphen(self):
        assert normalize_patent_number("JP2024-123456A") == "JP2024123456A"

    def test_jp_with_spaces(self):
        assert normalize_patent_number("JP 2024-123456 A") == "JP2024123456A"

    def test_us_with_commas(self):
        assert normalize_patent_number("US11,234,567B2") == "US11234567B2"

    def test_already_normalized(self):
        assert normalize_patent_number("JP2024123456A") == "JP2024123456A"

    def test_lowercase(self):
        assert normalize_patent_number("jp2024123456a") == "JP2024123456A"


class TestPatent:
    def test_normalized_number(self):
        patent = Patent(patent_number="JP2024-123456A", title="Test")
        assert patent.normalized_number == "JP2024123456A"

    def test_independent_claims(self):
        patent = Patent(
            patent_number="JP123",
            title="Test",
            claims=[
                PatentClaim(1, "Independent claim", "independent"),
                PatentClaim(2, "Dependent claim", "dependent", depends_on=1),
                PatentClaim(3, "Another independent", "independent"),
            ],
        )
        assert len(patent.independent_claims) == 2
        assert len(patent.dependent_claims) == 1

    def test_to_dict_round_trip(self):
        patent = Patent(
            patent_number="JP2024123456A",
            title="Test Patent",
            abstract="An abstract",
            applicant="Test Corp",
            ipc_classifications=[IPCClassification(full_code="H01L21/027")],
        )
        data = patent.to_dict()
        restored = Patent.from_dict(data)
        assert restored.patent_number == "JP2024123456A"
        assert restored.title == "Test Patent"
        assert len(restored.ipc_classifications) == 1


class TestClaimChart:
    def test_creation(self):
        chart = ClaimChart(
            target_patent="JP123",
            comparison_type="prior_art",
            entries=[
                ClaimChartEntry(
                    claim_element="Element A",
                    patent_number="JP456",
                    mapping="Corresponds to claim 1",
                    confidence=0.8,
                ),
            ],
        )
        assert len(chart.entries) == 1
        assert chart.entries[0].confidence == 0.8


class TestTechnologyLandscape:
    def test_creation(self):
        landscape = TechnologyLandscape(
            topic="Battery technology",
            total_patents_analyzed=50,
            ipc_distribution={"H01M10": 20, "H01M4": 15},
            filing_trend={"2022": 10, "2023": 20, "2024": 20},
        )
        assert landscape.total_patents_analyzed == 50
        assert landscape.ipc_distribution["H01M10"] == 20


class TestPatentSearchResult:
    def test_creation(self):
        result = PatentSearchResult(
            patent_number="JP2024123456A",
            title="Test Patent",
            source_database="google_patents",
        )
        assert result.source_database == "google_patents"
