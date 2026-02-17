"""Tests for patent search clients and merger."""

import pytest
from patent_research.search.patent_merger import PatentMerger
from patent_research.models.patent import Patent, IPCClassification
from patent_research.models.search_result import PatentSearchResult


class TestPatentMerger:
    def setup_method(self):
        self.merger = PatentMerger()

    def test_merge_no_duplicates(self):
        results = {
            "google_patents": [
                PatentSearchResult(patent_number="JP2024123456A", title="Patent 1", source_database="google_patents"),
            ],
            "jplatpat": [
                PatentSearchResult(patent_number="JP2024789012A", title="Patent 2", source_database="jplatpat"),
            ],
        }
        merged = self.merger.merge_search_results(results)
        assert len(merged) == 2

    def test_merge_with_duplicates(self):
        results = {
            "google_patents": [
                PatentSearchResult(
                    patent_number="JP2024123456A",
                    title="Patent Title EN",
                    source_database="google_patents",
                ),
            ],
            "jplatpat": [
                PatentSearchResult(
                    patent_number="JP2024-123456A",  # Same patent, different format
                    title="特許タイトル",
                    applicant="テスト株式会社",
                    source_database="jplatpat",
                ),
            ],
        }
        merged = self.merger.merge_search_results(results)
        assert len(merged) == 1
        # Should have both sources tracked
        assert len(merged[0].metadata.get("sources", [])) == 2

    def test_merge_patents_enrichment(self):
        patents = {
            "google_patents": [
                Patent(
                    patent_number="JP2024123456A",
                    title="Title from Google",
                    abstract="Abstract from Google",
                    source_database="google_patents",
                    source_databases=["google_patents"],
                ),
            ],
            "espacenet": [
                Patent(
                    patent_number="JP2024123456A",
                    title="",  # Missing title
                    applicant="Applicant from Espacenet",
                    ipc_classifications=[IPCClassification(full_code="H01L21/027")],
                    source_database="espacenet",
                    source_databases=["espacenet"],
                ),
            ],
        }
        merged = self.merger.merge_patents(patents)
        assert len(merged) == 1

        patent = merged[0]
        assert patent.title == "Title from Google"  # Kept from first source
        assert patent.applicant == "Applicant from Espacenet"  # Filled from second
        assert len(patent.ipc_classifications) == 1  # Merged from second
        assert "google_patents" in patent.source_databases
        assert "espacenet" in patent.source_databases

    def test_merge_empty(self):
        merged = self.merger.merge_search_results({})
        assert len(merged) == 0
