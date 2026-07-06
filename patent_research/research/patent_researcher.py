"""
Patent researcher - Main patent research orchestration.

Coordinates the full patent research workflow:
1. Query analysis and search plan creation
2. Multi-layer patent search
3. Claim analysis and technical element extraction
4. Auxiliary trigger evaluation and supplementary searches
5. Results synthesis
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from uuid import uuid4

from deep_research_tool.evidence.locker import EvidenceLocker, EvidenceType

from ..models.patent import Patent, PatentFamily
from ..models.analysis import ClaimChart, TechnologyLandscape, PriorArtRecord
from ..config import PatentResearchConfig
from ..search.search_orchestrator import SearchOrchestrator, MultiLayerSearchResult
from .patent_query_generator import PatentQueryGenerator, PatentSearchPlan
from .claim_analyzer import ClaimAnalyzer
from .auxiliary_trigger import TriggerResult

logger = logging.getLogger(__name__)


@dataclass
class PatentResearchSession:
    """Complete patent research session data."""

    session_id: str = field(default_factory=lambda: str(uuid4())[:8])
    query: str = ""
    requirements: str = ""
    state: str = "initialized"

    # Search plan
    search_plan: Optional[PatentSearchPlan] = None

    # Layer 1 results
    patents_found: List[Patent] = field(default_factory=list)
    patent_families: List[PatentFamily] = field(default_factory=list)

    # Analysis results
    claim_charts: List[ClaimChart] = field(default_factory=list)
    technology_landscape: Optional[TechnologyLandscape] = None
    prior_art_records: List[PriorArtRecord] = field(default_factory=list)

    # Layer 2/3 results
    academic_papers: List[Dict[str, Any]] = field(default_factory=list)
    examination_documents: List[Dict[str, Any]] = field(default_factory=list)
    business_evidence: List[Dict[str, Any]] = field(default_factory=list)

    # Trigger analysis
    trigger_results: List[Dict[str, Any]] = field(default_factory=list)

    # Report sections
    section_contents: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Metadata
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "requirements": self.requirements,
            "state": self.state,
            "search_plan": self.search_plan.to_dict() if self.search_plan else None,
            "patents_found": [p.to_dict() for p in self.patents_found],
            "claim_charts": [c.to_dict() for c in self.claim_charts],
            "technology_landscape": (
                self.technology_landscape.to_dict()
                if self.technology_landscape
                else None
            ),
            "academic_papers": self.academic_papers,
            "examination_documents": self.examination_documents,
            "business_evidence": self.business_evidence,
            "section_contents": self.section_contents,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
        }

    def save(self, filepath: Path) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class PatentResearcher:
    """
    Main patent research orchestrator.

    Coordinates the full patent research workflow including multi-layer
    search, claim analysis, and report content generation.
    """

    def __init__(
        self,
        llm_client,
        search_orchestrator: SearchOrchestrator,
        config: PatentResearchConfig,
        progress_callback: Callable[[str, float], None] = None,
    ):
        self.llm = llm_client
        self.search_orchestrator = search_orchestrator
        self.config = config
        self.progress_callback = progress_callback

        self.query_generator = PatentQueryGenerator(llm_client, config.language)
        self.claim_analyzer = ClaimAnalyzer(llm_client, config.language)

        self.session: Optional[PatentResearchSession] = None
        self.evidence_locker: Optional[EvidenceLocker] = None

    def _report_progress(self, message: str, percentage: float) -> None:
        if self.progress_callback:
            self.progress_callback(message, percentage)
        print(f"[{percentage:.1f}%] {message}")

    def conduct_research(
        self,
        query: str,
        requirements: str = "",
        target_patents: List[str] = None,
        ipc_focus: List[str] = None,
    ) -> PatentResearchSession:
        """
        Conduct complete patent research.

        Args:
            query: Research query/topic
            requirements: Specific research requirements
            target_patents: Specific patent numbers to analyze
            ipc_focus: IPC codes to focus on

        Returns:
            Completed PatentResearchSession
        """
        # Initialize session
        self.session = PatentResearchSession(query=query, requirements=requirements)
        output_dir = self.config.report.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_locker = EvidenceLocker(
            research_id=self.session.session_id,
            output_dir=output_dir / "evidence",
        )

        try:
            # Phase 1: Planning
            self._report_progress("検索計画を作成中...", 5)
            self.session.state = "planning"

            self.session.search_plan = self.query_generator.create_patent_search_plan(
                query=query,
                requirements=requirements,
                ipc_codes=ipc_focus,
                target_patents=target_patents,
            )

            self._report_progress(
                f"検索計画作成完了: {len(self.session.search_plan.patent_queries)}件のクエリ",
                10,
            )

            # Phase 2: Multi-layer search
            self.session.state = "searching"
            self._report_progress("3層検索を開始...", 12)

            search_result = self.search_orchestrator.execute_search(
                queries=self.session.search_plan.patent_queries,
                ipc_codes=(
                    self.session.search_plan.suggested_ipc_codes
                    + (ipc_focus or [])
                ),
                target_patents=target_patents,
            )

            # Store results in session
            self.session.patents_found = search_result.patents
            self.session.academic_papers = [
                p.to_dict() for p in search_result.academic_papers
            ]
            self.session.examination_documents = [
                d.to_dict() for d in search_result.examination_documents
            ]
            self.session.business_evidence = [
                e.to_dict() for e in search_result.business_evidence
            ]
            self.session.trigger_results = [
                t.to_dict() for t in search_result.trigger_results
            ]

            # Add patents to evidence locker
            for patent in search_result.patents:
                self.evidence_locker.add_evidence(
                    url=patent.source_url,
                    title=f"{patent.patent_number}: {patent.title}",
                    content_excerpt=patent.abstract[:500],
                    evidence_type=EvidenceType.OTHER,
                    search_query=query,
                    relevance_score=1.0,
                )

            # Add academic papers to evidence locker
            for paper in search_result.academic_papers:
                self.evidence_locker.add_evidence(
                    url=paper.url,
                    title=paper.title,
                    content_excerpt=paper.abstract[:500],
                    evidence_type=EvidenceType.RESEARCH_PAPER,
                    relevance_score=paper.relevance_score,
                )

            # Add business evidence to evidence locker
            for evidence in search_result.business_evidence:
                self.evidence_locker.add_evidence(
                    url=evidence.url,
                    title=evidence.title,
                    content_excerpt=evidence.content_excerpt[:500],
                    evidence_type=EvidenceType.OTHER,
                    relevance_score=0.7,
                )

            self._report_progress(
                f"検索完了: 特許{len(search_result.patents)}件, "
                f"論文{len(search_result.academic_papers)}件, "
                f"審査資料{len(search_result.examination_documents)}件, "
                f"ビジネスエビデンス{len(search_result.business_evidence)}件",
                50,
            )

            # Phase 3: Analysis
            self.session.state = "analyzing"
            self._perform_analysis(search_result)

            # Phase 4: Content synthesis
            self.session.state = "synthesizing"
            self._report_progress("レポートコンテンツを生成中...", 80)
            self._synthesize_content(search_result)

            # Mark completion
            self.session.state = "completed"
            self.session.completed_at = datetime.now().isoformat()
            self._report_progress("特許調査完了", 100)

            # Save session and evidence
            session_path = output_dir / f"patent_session_{self.session.session_id}.json"
            self.session.save(session_path)
            self.evidence_locker.export_to_json()
            self.evidence_locker.export_to_csv()

        except Exception as e:
            self.session.state = "error"
            self.session.error_message = str(e)
            self._report_progress(f"エラー: {e}", -1)
            raise

        return self.session

    def _perform_analysis(self, search_result: MultiLayerSearchResult) -> None:
        """Perform claim analysis and related analyses."""
        patents = search_result.patents

        if not patents:
            return

        # Extract technical elements for each patent
        self._report_progress("クレーム分析中...", 55)
        for patent in patents[:10]:
            if patent.claims:
                elements = self.claim_analyzer.extract_technical_elements(patent)
                for claim in patent.claims:
                    if not claim.technical_elements:
                        claim.technical_elements = elements

        # Generate claim chart if configured
        if self.config.report.generate_claim_chart and len(patents) >= 2:
            self._report_progress("クレームチャート生成中...", 60)
            target = patents[0]
            references = patents[1:self.config.report.claim_chart_max_patents + 1]
            chart = self.claim_analyzer.generate_claim_chart(
                target_patent=target,
                reference_patents=references,
                detail_level=self.config.report.claim_chart_detail_level,
            )
            self.session.claim_charts.append(chart)

        # Generate technology landscape if configured
        if self.config.report.generate_landscape:
            self._report_progress("技術ランドスケープ分析中...", 70)
            self.session.technology_landscape = self._generate_landscape(patents)

        self._report_progress("分析完了", 78)

    def _generate_landscape(self, patents: List[Patent]) -> TechnologyLandscape:
        """Generate technology landscape from patent data."""
        landscape = TechnologyLandscape(
            topic=self.session.query,
            total_patents_analyzed=len(patents),
        )

        # IPC distribution
        for patent in patents:
            for ipc in patent.ipc_classifications:
                code = ipc.subclass or ipc.class_code or ipc.full_code
                landscape.ipc_distribution[code] = (
                    landscape.ipc_distribution.get(code, 0) + 1
                )

        # Top applicants
        applicant_counts: Dict[str, int] = {}
        for patent in patents:
            if patent.applicant:
                applicant_counts[patent.applicant] = (
                    applicant_counts.get(patent.applicant, 0) + 1
                )
        landscape.top_applicants = [
            {"name": name, "count": count}
            for name, count in sorted(
                applicant_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]
        ]

        # Filing trend
        for patent in patents:
            year = ""
            date_str = patent.filing_date or patent.publication_date
            if date_str:
                # Extract year from various date formats
                for fmt_len in [4, 10]:
                    try:
                        year = date_str[:4]
                        break
                    except (ValueError, IndexError):
                        continue
            if year and year.isdigit():
                landscape.filing_trend[year] = (
                    landscape.filing_trend.get(year, 0) + 1
                )

        # Geographic distribution
        for patent in patents:
            if patent.jurisdiction:
                landscape.geographic_distribution[patent.jurisdiction] = (
                    landscape.geographic_distribution.get(patent.jurisdiction, 0) + 1
                )

        # Generate summary with LLM
        try:
            summary_prompt = f"""以下の特許ランドスケープデータを要約してください。

テーマ: {self.session.query}
分析特許数: {len(patents)}
IPC分布: {dict(sorted(landscape.ipc_distribution.items(), key=lambda x: x[1], reverse=True)[:10])}
上位出願人: {landscape.top_applicants[:5]}
出願トレンド: {dict(sorted(landscape.filing_trend.items()))}

150-300文字で技術ランドスケープの概要を記述してください。"""

            response = self.llm.generate(summary_prompt)
            landscape.summary = response.content.strip()
        except Exception as e:
            logger.warning(f"[PatentResearcher] Landscape summary failed: {e}")

        return landscape

    def _synthesize_content(self, search_result: MultiLayerSearchResult) -> None:
        """Synthesize all research results into report sections."""
        patents = search_result.patents

        # Section 1: Executive Summary
        self._generate_section(
            "1",
            "エグゼクティブサマリー",
            self._build_executive_summary_prompt(search_result),
        )

        # Section 2: Search Methodology
        self._generate_section(
            "2",
            "調査方法",
            self._build_methodology_section(search_result),
        )

        # Section 3: Patent Landscape
        if self.session.technology_landscape:
            self.session.section_contents["3"] = {
                "title": "特許ランドスケープ概観",
                "content": self._format_landscape_section(),
            }

        # Section 4: Key Patent Analysis
        self._generate_section(
            "4",
            "主要特許分析",
            self._build_patent_analysis_prompt(patents[:10]),
        )

        # Section 5: Claim Chart
        if self.session.claim_charts:
            self.session.section_contents["5"] = {
                "title": "クレームチャート",
                "content": self._format_claim_chart_section(),
            }

        # Section 6: Prior Art Analysis
        if len(patents) >= 2:
            self._generate_section(
                "6",
                "先行技術分析",
                self._build_prior_art_prompt(patents),
            )

        # Section 7: Technical Background (Layer 2 results)
        if search_result.academic_papers:
            self._generate_section(
                "7",
                "技術的背景",
                self._build_technical_bg_prompt(search_result.academic_papers),
            )

        # Section 8: Business Context (Layer 3 results)
        if search_result.business_evidence:
            self._generate_section(
                "8",
                "市場・ビジネスコンテキスト",
                self._build_business_context_prompt(search_result.business_evidence),
            )

        # Section 9: Examination History
        if search_result.examination_documents:
            self._generate_section(
                "9",
                "審査経過情報",
                self._build_examination_prompt(search_result.examination_documents),
            )

        # Section 10: Conclusions
        self._generate_section(
            "10",
            "結論・推奨事項",
            self._build_conclusions_prompt(search_result),
        )

    def _generate_section(self, section_id: str, title: str, prompt: str) -> None:
        """Generate a report section using LLM."""
        try:
            response = self.llm.generate(prompt)
            self.session.section_contents[section_id] = {
                "title": title,
                "content": response.content.strip(),
            }
        except Exception as e:
            logger.error(f"[PatentResearcher] Section {section_id} generation failed: {e}")
            self.session.section_contents[section_id] = {
                "title": title,
                "content": f"セクション「{title}」の生成に失敗しました: {e}",
            }

    def _build_executive_summary_prompt(self, result: MultiLayerSearchResult) -> str:
        patents_summary = "\n".join(
            f"- {p.patent_number}: {p.title} ({p.applicant})"
            for p in result.patents[:5]
        )
        return f"""以下の特許調査結果のエグゼクティブサマリーを作成してください。

【調査テーマ】{self.session.query}
【要件】{self.session.requirements or 'なし'}
【発見特許数】{len(result.patents)}件
【主要特許】
{patents_summary}
【学術論文】{len(result.academic_papers)}件
【審査資料】{len(result.examination_documents)}件
【ビジネスエビデンス】{len(result.business_evidence)}件

500-800文字でエグゼクティブサマリーを記述してください。主要な発見事項、技術トレンド、注目すべき特許を含めてください。"""

    def _build_methodology_section(self, result: MultiLayerSearchResult) -> str:
        queries_used = result.search_queries_used
        return f"""以下の調査方法を説明するセクションを作成してください。

【検索DB】{', '.join(result.sources_searched)}
【特許検索クエリ】{queries_used.get('patent', [])}
【学術検索クエリ】{queries_used.get('academic', [])}
【ビジネス検索クエリ】{queries_used.get('business', [])}
【管轄区域】{self.config.patent_search.patent_jurisdictions}

調査方法（使用したDB、検索戦略、3層検索の概要）を300-500文字で記述してください。"""

    def _build_patent_analysis_prompt(self, patents: List[Patent]) -> str:
        details = []
        for p in patents[:10]:
            claims_text = "\n".join(
                f"  請求項{c.claim_number}: {c.claim_text[:200]}"
                for c in p.claims[:3]
            )
            ipc_text = ", ".join(c.full_code for c in p.ipc_classifications[:5])
            details.append(
                f"【{p.patent_number}】{p.title}\n"
                f"出願人: {p.applicant}\n"
                f"IPC: {ipc_text}\n"
                f"要約: {p.abstract[:300]}\n"
                f"主要請求項:\n{claims_text}"
            )

        return f"""以下の主要特許を分析してください。

{chr(10).join(details)}

各特許について以下を含めた分析を記述してください：
1. 技術概要
2. 主要な技術的特徴
3. 請求項の要点
4. 注目すべきポイント"""

    def _build_prior_art_prompt(self, patents: List[Patent]) -> str:
        summaries = "\n".join(
            f"- {p.patent_number}: {p.title} (出願日: {p.filing_date})"
            for p in patents[:10]
        )
        return f"""以下の特許群に基づいて、先行技術分析を行ってください。

【発見特許】
{summaries}

以下を含めてください：
1. 技術の発展経緯
2. 主要な先行技術の特定
3. 技術的進歩のポイント
4. 先行技術との差別化要素"""

    def _build_technical_bg_prompt(self, papers) -> str:
        paper_list = "\n".join(
            f"- {p.title} ({p.source}): {p.abstract[:200]}"
            for p in papers[:10]
        )
        return f"""以下の学術論文・技術資料に基づいて、技術的背景セクションを作成してください。

【調査テーマ】{self.session.query}
【関連論文・技術資料】
{paper_list}

特許技術の学術的・技術的な裏付けとなる情報を整理して記述してください。"""

    def _build_business_context_prompt(self, evidence) -> str:
        evidence_list = "\n".join(
            f"- {e.title} ({e.evidence_type}): {e.content_excerpt[:200]}"
            for e in evidence[:10]
        )
        return f"""以下のビジネスエビデンスに基づいて、市場・ビジネスコンテキストセクションを作成してください。

【調査テーマ】{self.session.query}
【ビジネスエビデンス】
{evidence_list}

市場規模、成長性、競合状況など、特許技術のビジネス面での位置付けを記述してください。"""

    def _build_examination_prompt(self, documents) -> str:
        doc_list = "\n".join(
            f"- {d.patent_number} [{d.document_type}]: {d.title} - {d.content_excerpt[:200]}"
            for d in documents[:10]
        )
        return f"""以下の審査経過情報に基づいて、審査経過セクションを作成してください。

【審査資料】
{doc_list}

拒絶理由、補正の経緯、審査上の論点などを整理して記述してください。"""

    def _build_conclusions_prompt(self, result: MultiLayerSearchResult) -> str:
        return f"""以下の特許調査結果に基づいて、結論と推奨事項を作成してください。

【調査テーマ】{self.session.query}
【発見特許数】{len(result.patents)}件
【学術論文数】{len(result.academic_papers)}件
【ビジネスエビデンス数】{len(result.business_evidence)}件

以下を含めてください：
1. 調査の主要な結論
2. 技術的な推奨事項
3. 今後の調査方向
4. リスクと注意事項"""

    def _format_landscape_section(self) -> str:
        """Format technology landscape data as report content."""
        ls = self.session.technology_landscape
        if not ls:
            return ""

        lines = [ls.summary, ""]

        if ls.ipc_distribution:
            lines.append("### IPC分類分布")
            for code, count in sorted(
                ls.ipc_distribution.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                lines.append(f"- {code}: {count}件")
            lines.append("")

        if ls.top_applicants:
            lines.append("### 上位出願人")
            for app in ls.top_applicants[:10]:
                lines.append(f"- {app['name']}: {app['count']}件")
            lines.append("")

        if ls.filing_trend:
            lines.append("### 出願トレンド")
            for year, count in sorted(ls.filing_trend.items()):
                lines.append(f"- {year}年: {count}件")

        return "\n".join(lines)

    def _format_claim_chart_section(self) -> str:
        """Format claim charts as report content."""
        lines = []
        for chart in self.session.claim_charts:
            lines.append(f"### 対象特許: {chart.target_patent}")
            lines.append(f"比較タイプ: {chart.comparison_type}")
            lines.append("")
            if chart.summary:
                lines.append(chart.summary)
                lines.append("")
            lines.append("| クレーム要素 | 参照特許 | 対応関係 | 確信度 |")
            lines.append("|---|---|---|---|")
            for entry in chart.entries:
                lines.append(
                    f"| {entry.claim_element} | {entry.patent_number} "
                    f"| {entry.mapping} | {entry.confidence:.0%} |"
                )
            lines.append("")
        return "\n".join(lines)

    def get_session(self) -> Optional[PatentResearchSession]:
        return self.session

    def get_evidence_locker(self) -> Optional[EvidenceLocker]:
        return self.evidence_locker
