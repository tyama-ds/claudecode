"""
Patent Research Tool - Main interface.

Provides the PatentResearchTool class and run_patent_research convenience function.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable

from deep_research_tool.api.openai_client import OpenAIClient
from deep_research_tool.api.anthropic_client import AnthropicClient
from deep_research_tool.config import LLMProvider

from .config import PatentResearchConfig, create_patent_config
from .search.google_patents import GooglePatentsClient
from .search.jplatpat import JPlatPatClient
from .search.espacenet import EspacenetClient
from .search.search_orchestrator import SearchOrchestrator
from .research.patent_researcher import PatentResearcher, PatentResearchSession
from .report.patent_report_generator import PatentReportGenerator

logger = logging.getLogger(__name__)


class PatentResearchTool:
    """
    Main interface for Patent Research Mode.

    Orchestrates the complete patent research workflow:
    1. Creates LLM and search clients based on configuration
    2. Builds the search orchestrator with 3-layer search capability
    3. Runs the patent researcher
    4. Generates patent-specific reports
    """

    def __init__(self, config: PatentResearchConfig = None):
        """
        Initialize Patent Research Tool.

        Args:
            config: Patent research configuration. If None, uses defaults.
        """
        self.config = config or PatentResearchConfig()

    def _create_llm_client(self):
        """Create LLM client based on configuration."""
        if self.config.api.provider == LLMProvider.OPENAI:
            return OpenAIClient(
                api_key=self.config.api.openai_api_key,
                model=self.config.api.openai_model,
                max_tokens=self.config.api.max_tokens,
                temperature=self.config.api.temperature,
            )
        elif self.config.api.provider == LLMProvider.ANTHROPIC:
            return AnthropicClient(
                api_key=self.config.api.anthropic_api_key,
                model=self.config.api.anthropic_model,
                max_tokens=self.config.api.max_tokens,
                temperature=self.config.api.temperature,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.config.api.provider}")

    def _create_patent_clients(self):
        """Create patent search clients based on configuration."""
        clients = []
        proxies = self.config.proxy.get_proxies_dict()
        verify_ssl = self.config.proxy.verify_ssl

        if self.config.patent_search.enable_google_patents:
            clients.append(
                GooglePatentsClient(
                    language=self.config.language,
                    proxies=proxies,
                    verify_ssl=verify_ssl,
                )
            )

        if self.config.patent_search.enable_jplatpat:
            clients.append(
                JPlatPatClient(
                    language=self.config.language,
                    proxies=proxies,
                    verify_ssl=verify_ssl,
                )
            )

        if self.config.patent_search.enable_espacenet:
            clients.append(
                EspacenetClient(
                    language=self.config.language,
                    consumer_key=self.config.patent_search.ops_consumer_key,
                    consumer_secret=self.config.patent_search.ops_consumer_secret,
                    proxies=proxies,
                    verify_ssl=verify_ssl,
                )
            )

        return clients

    def run(
        self,
        query: str,
        requirements: str = "",
        target_patents: List[str] = None,
        ipc_focus: List[str] = None,
        progress_callback: Callable[[str, float], None] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete patent research workflow.

        Args:
            query: Research query/topic
            requirements: Specific research requirements
            target_patents: Specific patent numbers to analyze
            ipc_focus: IPC codes to focus on
            progress_callback: Callback for progress updates (message, percentage)

        Returns:
            Dict with:
                - session_id: Research session ID
                - report_path: Path to generated report
                - evidence_json: Path to evidence JSON
                - evidence_csv: Path to evidence CSV
                - session: PatentResearchSession object
                - patents_found: Number of patents found
                - academic_papers: Number of academic papers found
                - business_evidence: Number of business evidence items
        """
        # Validate configuration
        errors = self.config.validate()
        if errors:
            logger.warning(f"Configuration warnings: {errors}")

        # Create clients
        llm_client = self._create_llm_client()
        patent_clients = self._create_patent_clients()

        if not patent_clients:
            raise ValueError(
                "No patent search clients configured. "
                "Enable at least one of: google_patents, jplatpat, espacenet"
            )

        # Create search orchestrator
        proxies = self.config.proxy.get_proxies_dict()
        orchestrator = SearchOrchestrator(
            patent_clients=patent_clients,
            llm_client=llm_client,
            patent_config=self.config.patent_search,
            auxiliary_config=self.config.auxiliary,
            language=self.config.language,
            proxies=proxies,
            verify_ssl=self.config.proxy.verify_ssl,
            progress_callback=progress_callback,
        )

        # Create researcher
        researcher = PatentResearcher(
            llm_client=llm_client,
            search_orchestrator=orchestrator,
            config=self.config,
            progress_callback=progress_callback,
        )

        # Conduct research
        session = researcher.conduct_research(
            query=query,
            requirements=requirements,
            target_patents=target_patents,
            ipc_focus=ipc_focus,
        )

        # Generate report
        output_dir = self.config.report.output_dir
        report_generator = PatentReportGenerator(
            llm_client=llm_client,
            language=self.config.language,
            output_dir=output_dir,
        )

        report_path = report_generator.generate_report(
            session=session,
            format_type=self.config.report.format.value,
        )

        # Build result
        result = {
            "session_id": session.session_id,
            "report_path": str(report_path),
            "evidence_json": str(
                output_dir / "evidence" / f"evidence_{session.session_id}.json"
            ),
            "evidence_csv": str(
                output_dir / "evidence" / f"evidence_{session.session_id}.csv"
            ),
            "session": session,
            "patents_found": len(session.patents_found),
            "academic_papers": len(session.academic_papers),
            "examination_documents": len(session.examination_documents),
            "business_evidence": len(session.business_evidence),
            "claim_charts": len(session.claim_charts),
        }

        return result


def run_patent_research(
    query: str,
    provider: str = "openai",
    model: str = None,
    language: str = "ja",
    output_format: str = "markdown",
    output_dir: str = "./output/patent_research",
    requirements: str = "",
    target_patents: List[str] = None,
    ipc_codes: List[str] = None,
    enable_google_patents: bool = True,
    enable_jplatpat: bool = True,
    enable_espacenet: bool = True,
    enable_academic_search: bool = True,
    enable_examination_search: bool = True,
    enable_business_search: bool = True,
    generate_claim_chart: bool = True,
    generate_landscape: bool = True,
    verbose: bool = False,
    progress_callback: Callable[[str, float], None] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for patent research.

    Args:
        query: Research query/topic
        provider: LLM provider ('openai' or 'anthropic')
        model: Model name
        language: Output language
        output_format: Report format ('markdown', 'docx', 'pdf', 'html')
        output_dir: Output directory
        requirements: Research requirements
        target_patents: Specific patent numbers to analyze
        ipc_codes: IPC codes to focus on
        enable_google_patents: Enable Google Patents search
        enable_jplatpat: Enable J-PlatPat search
        enable_espacenet: Enable Espacenet search
        enable_academic_search: Enable academic paper search
        enable_examination_search: Enable examination document search
        enable_business_search: Enable business evidence search
        generate_claim_chart: Generate claim chart
        generate_landscape: Generate technology landscape
        verbose: Enable verbose logging
        progress_callback: Progress callback

    Returns:
        Dict with research results
    """
    config = create_patent_config(
        provider=provider,
        model=model,
        language=language,
        output_format=output_format,
        output_dir=output_dir,
        verbose=verbose,
        enable_google_patents=enable_google_patents,
        enable_jplatpat=enable_jplatpat,
        enable_espacenet=enable_espacenet,
        ipc_codes=ipc_codes,
        enable_academic_search=enable_academic_search,
        enable_examination_search=enable_examination_search,
        enable_business_search=enable_business_search,
        generate_claim_chart=generate_claim_chart,
        generate_landscape=generate_landscape,
        **kwargs,
    )

    tool = PatentResearchTool(config=config)
    return tool.run(
        query=query,
        requirements=requirements,
        target_patents=target_patents,
        ipc_focus=ipc_codes,
        progress_callback=progress_callback,
    )
