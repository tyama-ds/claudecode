"""
Patent merger for deduplicating and merging results from multiple databases.

When searching Google Patents, J-PlatPat, and Espacenet simultaneously,
the same patent may appear in multiple databases. This module normalizes
patent numbers and merges metadata from all sources.
"""

import logging
from typing import List, Dict, Optional

from ..models.patent import Patent, normalize_patent_number
from ..models.search_result import PatentSearchResult

logger = logging.getLogger(__name__)


class PatentMerger:
    """Merges and deduplicates patent results from multiple databases."""

    def merge_search_results(
        self,
        results_by_source: Dict[str, List[PatentSearchResult]],
    ) -> List[PatentSearchResult]:
        """
        Merge patent search results from multiple sources.

        Deduplicates by normalized patent number, preserving the best
        metadata from each source.

        Args:
            results_by_source: Dict mapping source name to list of results.
                e.g., {"google_patents": [...], "jplatpat": [...], "espacenet": [...]}

        Returns:
            Merged and deduplicated list of PatentSearchResult
        """
        # Group by normalized patent number
        merged: Dict[str, PatentSearchResult] = {}

        for source, results in results_by_source.items():
            for result in results:
                normalized = normalize_patent_number(result.patent_number)

                if normalized in merged:
                    # Merge: enrich existing result with new source data
                    existing = merged[normalized]
                    self._enrich_search_result(existing, result)
                else:
                    # New patent, add to merged dict
                    result.metadata["sources"] = [result.source_database]
                    merged[normalized] = result

        result_list = list(merged.values())
        logger.info(
            f"[PatentMerger] Merged {sum(len(r) for r in results_by_source.values())} "
            f"results into {len(result_list)} unique patents"
        )
        return result_list

    def merge_patents(
        self,
        patents_by_source: Dict[str, List[Patent]],
    ) -> List[Patent]:
        """
        Merge full Patent objects from multiple sources.

        Combines metadata, claims, classifications from all sources.

        Args:
            patents_by_source: Dict mapping source name to list of patents

        Returns:
            Merged and deduplicated list of Patent objects
        """
        merged: Dict[str, Patent] = {}

        for source, patents in patents_by_source.items():
            for patent in patents:
                normalized = normalize_patent_number(patent.patent_number)

                if normalized in merged:
                    self._enrich_patent(merged[normalized], patent)
                else:
                    if source not in patent.source_databases:
                        patent.source_databases.append(source)
                    merged[normalized] = patent

        result_list = list(merged.values())
        logger.info(
            f"[PatentMerger] Merged {sum(len(p) for p in patents_by_source.values())} "
            f"patents into {len(result_list)} unique patents"
        )
        return result_list

    def _enrich_search_result(
        self,
        existing: PatentSearchResult,
        new: PatentSearchResult,
    ) -> None:
        """Enrich an existing search result with data from another source."""
        # Track all sources
        sources = existing.metadata.get("sources", [])
        if new.source_database not in sources:
            sources.append(new.source_database)
            existing.metadata["sources"] = sources

        # Fill in missing fields
        if not existing.title and new.title:
            existing.title = new.title
        if not existing.snippet and new.snippet:
            existing.snippet = new.snippet
        if not existing.applicant and new.applicant:
            existing.applicant = new.applicant
        if not existing.filing_date and new.filing_date:
            existing.filing_date = new.filing_date
        if not existing.publication_date and new.publication_date:
            existing.publication_date = new.publication_date
        if new.ipc_codes:
            existing_codes = set(existing.ipc_codes)
            for code in new.ipc_codes:
                if code not in existing_codes:
                    existing.ipc_codes.append(code)

    def _enrich_patent(self, existing: Patent, new: Patent) -> None:
        """Enrich an existing Patent with data from another source."""
        # Track all source databases
        if new.source_database and new.source_database not in existing.source_databases:
            existing.source_databases.append(new.source_database)

        # Fill in missing basic fields
        if not existing.title and new.title:
            existing.title = new.title
        if not existing.abstract and new.abstract:
            existing.abstract = new.abstract
        if not existing.applicant and new.applicant:
            existing.applicant = new.applicant
        if not existing.inventor and new.inventor:
            existing.inventor = new.inventor
        if not existing.filing_date and new.filing_date:
            existing.filing_date = new.filing_date
        if not existing.publication_date and new.publication_date:
            existing.publication_date = new.publication_date
        if not existing.grant_date and new.grant_date:
            existing.grant_date = new.grant_date
        if not existing.status and new.status:
            existing.status = new.status
        if not existing.family_id and new.family_id:
            existing.family_id = new.family_id

        # Merge claims (prefer the source with more claims)
        if len(new.claims) > len(existing.claims):
            existing.claims = new.claims

        # Merge IPC classifications
        existing_ipc_codes = {c.full_code for c in existing.ipc_classifications}
        for ipc in new.ipc_classifications:
            if ipc.full_code not in existing_ipc_codes:
                existing.ipc_classifications.append(ipc)
                existing_ipc_codes.add(ipc.full_code)

        # Merge CPC classifications
        existing_cpc = set(existing.cpc_classifications)
        for cpc in new.cpc_classifications:
            if cpc not in existing_cpc:
                existing.cpc_classifications.append(cpc)

        # Merge citations
        existing_cited = set(existing.cited_patents)
        for cited in new.cited_patents:
            if cited not in existing_cited:
                existing.cited_patents.append(cited)

        existing_citing = set(existing.citing_patents)
        for citing in new.citing_patents:
            if citing not in existing_citing:
                existing.citing_patents.append(citing)

        # Merge family members
        existing_family = set(existing.family_members)
        for member in new.family_members:
            if member not in existing_family:
                existing.family_members.append(member)

        # Keep the longer full_text
        if len(new.full_text) > len(existing.full_text):
            existing.full_text = new.full_text
