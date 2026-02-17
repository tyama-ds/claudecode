"""
Multi-layer search orchestrator.

Coordinates the three search layers:
1. Primary: Patent database searches (Google Patents, J-PlatPat, Espacenet)
2. Secondary: Academic/technical papers (CiNii, J-STAGE, Google Scholar) + examination docs
3. Tertiary: Business evidence (market data, revenue, etc.)

The secondary and tertiary layers are triggered based on LLM analysis of
patent content from the primary layer.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple

from ..models.patent import Patent
from ..models.search_result import PatentSearchResult
from ..config import PatentSearchConfig, AuxiliarySearchConfig
from .patent_search import PatentSearchClient
from .patent_merger import PatentMerger
from .academic_search import AcademicSearchClient, AcademicPaper
from .examination_search import ExaminationSearchClient, ExaminationDocument
from .business_search import BusinessSearchClient, BusinessEvidence
from ..research.auxiliary_trigger import AuxiliaryTrigger, TriggerResult

logger = logging.getLogger(__name__)


@dataclass
class MultiLayerSearchResult:
    """Results from the multi-layer search process."""

    # Layer 1: Patent results
    patents: List[Patent] = field(default_factory=list)
    patent_search_results: List[PatentSearchResult] = field(default_factory=list)

    # Layer 2: Academic/technical results
    academic_papers: List[AcademicPaper] = field(default_factory=list)
    examination_documents: List[ExaminationDocument] = field(default_factory=list)

    # Layer 3: Business evidence
    business_evidence: List[BusinessEvidence] = field(default_factory=list)

    # Trigger analysis
    trigger_results: List[TriggerResult] = field(default_factory=list)
    aggregated_triggers: Optional[TriggerResult] = None

    # Metadata
    search_queries_used: Dict[str, List[str]] = field(default_factory=dict)
    sources_searched: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patents_found": len(self.patents),
            "academic_papers_found": len(self.academic_papers),
            "examination_documents_found": len(self.examination_documents),
            "business_evidence_found": len(self.business_evidence),
            "search_queries_used": self.search_queries_used,
            "sources_searched": self.sources_searched,
        }


class SearchOrchestrator:
    """
    Orchestrates multi-layer patent search.

    Layer 1 (Primary): Patent database searches across all configured DBs
    Layer 2 (Secondary): Academic/technical searches — triggered by technical elements
    Layer 3 (Tertiary): Business evidence searches — triggered by commercial claims
    """

    def __init__(
        self,
        patent_clients: List[PatentSearchClient],
        llm_client,
        patent_config: PatentSearchConfig,
        auxiliary_config: AuxiliarySearchConfig,
        language: str = "ja",
        proxies: dict = None,
        verify_ssl: bool = True,
        progress_callback: Callable[[str, float], None] = None,
    ):
        self.patent_clients = patent_clients
        self.llm_client = llm_client
        self.patent_config = patent_config
        self.auxiliary_config = auxiliary_config
        self.language = language
        self.progress_callback = progress_callback

        # Patent result merger
        self.merger = PatentMerger()

        # Auxiliary trigger
        self.trigger = AuxiliaryTrigger(
            llm_client=llm_client,
            config=auxiliary_config,
            language=language,
        )

        # Secondary layer clients
        self.academic_client = AcademicSearchClient(
            sources=auxiliary_config.academic_sources,
            max_results_per_source=auxiliary_config.max_papers_per_trigger,
            language=language,
            proxies=proxies,
            verify_ssl=verify_ssl,
        )
        self.examination_client = ExaminationSearchClient(
            language=language,
            max_results=auxiliary_config.max_examination_docs,
            proxies=proxies,
            verify_ssl=verify_ssl,
        )

        # Tertiary layer client
        self.business_client = BusinessSearchClient(
            language=language,
            max_results=auxiliary_config.max_business_sources,
            proxies=proxies,
            verify_ssl=verify_ssl,
        )

    def _report_progress(self, message: str, percentage: float) -> None:
        if self.progress_callback:
            self.progress_callback(message, percentage)
        logger.info(f"[{percentage:.0f}%] {message}")

    def execute_search(
        self,
        queries: List[str],
        ipc_codes: List[str] = None,
        target_patents: List[str] = None,
    ) -> MultiLayerSearchResult:
        """
        Execute the full multi-layer search.

        Args:
            queries: Patent search queries
            ipc_codes: IPC codes to filter by
            target_patents: Specific patents to look up

        Returns:
            MultiLayerSearchResult with all layers' results
        """
        result = MultiLayerSearchResult()

        # =============================================
        # Layer 1: Patent Search (Primary)
        # =============================================
        self._report_progress("Layer 1: 特許DB検索中...", 10)
        result.search_queries_used["patent"] = queries

        # Search across all patent databases in parallel (conceptually)
        results_by_source: Dict[str, List[PatentSearchResult]] = {}
        for client in self.patent_clients:
            source_results = []
            for query in queries:
                try:
                    search_results = client.search_patents(
                        query=query,
                        ipc_codes=ipc_codes,
                        jurisdictions=self.patent_config.patent_jurisdictions,
                        date_range=(
                            self.patent_config.date_range_start,
                            self.patent_config.date_range_end,
                        ) if self.patent_config.date_range_start else None,
                        max_results=self.patent_config.max_patents_per_query,
                    )
                    source_results.extend(search_results)
                except Exception as e:
                    logger.error(
                        f"[SearchOrchestrator] {client.source_name} search failed: {e}"
                    )
            results_by_source[client.source_name] = source_results
            result.sources_searched.append(client.source_name)

        # Merge results from all sources
        merged_results = self.merger.merge_search_results(results_by_source)
        result.patent_search_results = merged_results

        self._report_progress(
            f"Layer 1: {len(merged_results)}件の特許を発見", 25
        )

        # Fetch detailed patent information for top results
        self._report_progress("特許詳細情報を取得中...", 30)
        patents = self._fetch_patent_details(merged_results)
        result.patents = patents

        # Also look up specific target patents if provided
        if target_patents:
            for patent_num in target_patents:
                if not any(
                    p.patent_number == patent_num or p.normalized_number == patent_num
                    for p in patents
                ):
                    detail = self._fetch_single_patent(patent_num)
                    if detail:
                        patents.append(detail)

        self._report_progress(
            f"Layer 1完了: {len(patents)}件の特許詳細を取得", 40
        )

        # =============================================
        # Layer 2: Auxiliary Trigger Analysis
        # =============================================
        self._report_progress("補助検索トリガー分析中...", 45)
        trigger_results = self.trigger.analyze_patents_batch(patents[:10])
        result.trigger_results = trigger_results
        aggregated = self.trigger.aggregate_triggers(trigger_results)
        result.aggregated_triggers = aggregated

        # =============================================
        # Layer 2: Academic/Technical Search (Secondary)
        # =============================================
        if self.auxiliary_config.enable_academic_search and aggregated.needs_academic_search:
            self._report_progress("Layer 2: 学術論文検索中...", 55)
            academic_queries = aggregated.get_academic_queries()
            result.search_queries_used["academic"] = academic_queries

            for query in academic_queries[:self.auxiliary_config.max_papers_per_trigger]:
                try:
                    papers = self.academic_client.search_and_extract(query)
                    result.academic_papers.extend(papers)
                except Exception as e:
                    logger.error(f"[SearchOrchestrator] Academic search failed: {e}")

            self._report_progress(
                f"Layer 2: {len(result.academic_papers)}件の論文を発見", 65
            )

        # Examination document search
        if self.auxiliary_config.enable_examination_search:
            self._report_progress("Layer 2: 審査資料検索中...", 68)
            for patent in patents[:5]:
                try:
                    docs = self.examination_client.search_examination_history(
                        patent.patent_number
                    )
                    result.examination_documents.extend(docs)
                except Exception as e:
                    logger.error(
                        f"[SearchOrchestrator] Examination search failed: {e}"
                    )

            self._report_progress(
                f"Layer 2: {len(result.examination_documents)}件の審査資料を発見", 72
            )

        # =============================================
        # Layer 3: Business Evidence Search (Tertiary)
        # =============================================
        if self.auxiliary_config.enable_business_search and aggregated.needs_business_search:
            self._report_progress("Layer 3: ビジネスエビデンス検索中...", 78)
            business_queries = aggregated.get_business_queries()
            result.search_queries_used["business"] = business_queries

            for query in business_queries[:self.auxiliary_config.max_business_sources]:
                try:
                    evidence = self.business_client.search_and_extract(query)
                    result.business_evidence.extend(evidence)
                except Exception as e:
                    logger.error(f"[SearchOrchestrator] Business search failed: {e}")

            self._report_progress(
                f"Layer 3: {len(result.business_evidence)}件のビジネスエビデンスを発見",
                85,
            )

        self._report_progress("全層検索完了", 90)
        return result

    def _fetch_patent_details(
        self,
        search_results: List[PatentSearchResult],
        max_fetch: int = 15,
    ) -> List[Patent]:
        """Fetch detailed patent information for search results."""
        patents = []
        for sr in search_results[:max_fetch]:
            patent = self._fetch_single_patent(sr.patent_number)
            if patent:
                patents.append(patent)
        return patents

    def _fetch_single_patent(self, patent_number: str) -> Optional[Patent]:
        """Try to fetch a patent from any available client."""
        for client in self.patent_clients:
            try:
                patent = client.get_patent_detail(patent_number)
                if patent:
                    return patent
            except Exception as e:
                logger.warning(
                    f"[SearchOrchestrator] {client.source_name} detail fetch "
                    f"failed for {patent_number}: {e}"
                )
        return None
