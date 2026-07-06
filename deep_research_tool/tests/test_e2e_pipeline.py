"""
End-to-end pipeline audit (no network, no real LLM).

Runs the full research flow — plan → per-section collection → importance
scoring → synthesis → executive summary — through Researcher for each
source mode (web / local / hybrid), then generates a V1 markdown report
and a V2 report from the resulting session. A rule-based FakeLLM answers
each prompt type; a FakeSearchClient serves canned pages.
"""

import json
from types import SimpleNamespace

import pytest
from unittest.mock import Mock

from deep_research_tool.config import CrawlMode, ResearchSourceMode
from deep_research_tool.report.generator import ReportGenerator
from deep_research_tool.report.v2.generator import (
    CHAPTER_META_DELIMITER,
    ReportGeneratorV2,
)
from deep_research_tool.research.researcher import Researcher, ResearchState


PLAN_JSON = json.dumps({
    "title": "炭素繊維市場の調査レポート",
    "summary": "国内炭素繊維市場の概観",
    "table_of_contents": [
        {"section": "1", "title": "市場規模の推移", "description": "国内市場規模の推移",
         "subsections": []},
        {"section": "2", "title": "主要メーカーの動向", "description": "主要企業の戦略",
         "subsections": []},
    ],
    "search_queries": [
        "炭素繊維 市場規模 国内",
        "炭素繊維 メーカー 動向",
        "炭素繊維 生産能力",
        "炭素繊維 用途 航空機",
    ],
    "key_terms": ["炭素繊維", "PAN系"],
    "suggested_sources": [],
    "methodology_notes": "",
    "estimated_complexity": "low",
}, ensure_ascii=False)


class FakeLLM:
    """Rule-based LLM stub: answers by prompt markers."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt, system_prompt=None, **kwargs):
        self.calls.append(prompt)
        return Mock(content=self._answer(prompt))

    def _answer(self, prompt):
        if "estimated_complexity" in prompt:
            return PLAN_JSON
        if "===SECTION_META===" in prompt:
            return (
                "国内の炭素繊維市場は堅調に拡大している。[SOURCE 1] "
                "航空機向け需要の回復が背景にあり、主要メーカーは増産投資を進めている。"
                "一方で原料価格の上昇が収益を圧迫する懸念も残る。[SOURCE 2]\n\n"
                "===SECTION_META===\n"
                '{"summary": "市場は拡大傾向にある。", '
                '"information_gaps": [], "confidence_level": "high"}'
            )
        if CHAPTER_META_DELIMITER in prompt:
            return (
                "## 1. 章本文\n\n炭素繊維市場は拡大している。\n\n"
                f"{CHAPTER_META_DELIMITER}\n"
                '{"key_points": ["拡大"], "terms_used": ["炭素繊維"], "facts_stated": []}'
            )
        if '"scores"' in prompt:
            return json.dumps({"scores": [
                {"index": i, "importance": 0.9 - 0.1 * (i - 1)} for i in range(1, 9)
            ]})
        if "is_coherent" in prompt:
            return '{"is_coherent": true, "issues": [], "suggestions": []}'
        if "executive_summary" in prompt:
            return json.dumps({
                "executive_summary": "本調査では炭素繊維市場の拡大を確認した。",
                "key_findings": ["市場拡大"],
                "recommendations": ["継続調査"],
                "overall_confidence": "high",
            }, ensure_ascii=False)
        if "processed_content" in prompt:
            return json.dumps({
                "processed_content": "炭素繊維市場は年率10%で成長しているという情報である。",
                "key_points": ["年率10%成長"],
                "quotes": [],
                "relevance_score": 0.8,
                "extraction_notes": "",
            }, ensure_ascii=False)
        return "0.7"


class FakeSearchClient:
    """Canned search results and page content; counts accesses."""

    def __init__(self):
        self.search_calls = 0
        self.fetch_calls = 0

    def search(self, query, max_results=10):
        self.search_calls += 1
        return [
            SimpleNamespace(
                url=f"https://example.com/{abs(hash(query)) % 1000}/{i}",
                title=f"検索結果 {i}: {query[:20]}",
                snippet="炭素繊維に関する情報",
            )
            for i in range(min(2, max_results))
        ]

    def get_page_content(self, url):
        self.fetch_calls += 1
        return SimpleNamespace(
            url=url,
            title="炭素繊維市場レポート",
            text_content="炭素繊維の国内市場は拡大を続けている。" * 30,
            html_content="<html><body></body></html>",
            images=[],
            links=[],
            metadata={},
        )


LOCAL_DOCS = [
    {
        "path": "internal_report.pdf",
        "title": "社内市場調査レポート",
        "content": "炭素繊維の国内市場規模は2,000億円に達した。主要メーカーは増産投資を進めている。" * 25,
    },
]


def run_pipeline(tmp_path, source_mode, documents=None):
    llm = FakeLLM()
    search = FakeSearchClient()
    researcher = Researcher(
        llm_client=llm,
        search_client=search,
        output_dir=tmp_path,
        source_mode=source_mode,
        crawl_mode=CrawlMode.STANDARD,
        use_enhanced_synthesis=False,
        filter_mode="none",
        max_gap_fill_rounds=0,
        max_queries_per_iteration=2,
        max_pages_per_query=2,
    )
    session = researcher.conduct_research(
        query="炭素繊維の市場調査",
        # 「目次」を含めてToC検証をスキップし、2章の固定プランで流す
        requirements="目次は2章構成でよい",
        additional_documents=documents,
    )
    return researcher, session, llm, search


class TestEndToEndPipeline:
    """全体監査: 各ソースモードで計画→調査→執筆→レポートまで通す。"""

    def test_web_mode_full_pipeline(self, tmp_path):
        researcher, session, llm, search = run_pipeline(
            tmp_path, ResearchSourceMode.WEB,
        )

        assert session.state == ResearchState.COMPLETED
        # Both sections written with citations
        for num in ("1", "2"):
            content = session.section_contents[num]["content"]
            assert "[SOURCE" in content
            assert session.section_contents[num]["sources"]
        # Web was actually used
        assert search.search_calls > 0
        assert search.fetch_calls > 0
        # Importance propagated to evidence
        evidence = researcher.evidence_locker.get_all_evidence()
        assert evidence
        assert any(e.importance_score > 0 for e in evidence)
        # Executive summary generated
        assert "_executive_summary" in session.section_contents

    def test_local_mode_full_pipeline_no_web_access(self, tmp_path):
        researcher, session, llm, search = run_pipeline(
            tmp_path, ResearchSourceMode.LOCAL, documents=LOCAL_DOCS,
        )

        assert session.state == ResearchState.COMPLETED
        # No web access at all
        assert search.search_calls == 0
        assert search.fetch_calls == 0
        # Evidence comes only from local documents
        research_evidence = [
            e for e in researcher.evidence_locker.get_all_evidence()
            if e.section_reference  # per-section evidence
        ]
        assert research_evidence
        assert all(e.url.startswith("local://") for e in research_evidence)
        for num in ("1", "2"):
            assert session.section_contents[num]["content"]

    def test_hybrid_mode_uses_both_sources(self, tmp_path):
        researcher, session, llm, search = run_pipeline(
            tmp_path, ResearchSourceMode.HYBRID, documents=LOCAL_DOCS,
        )

        assert session.state == ResearchState.COMPLETED
        assert search.search_calls > 0  # web used
        urls = [e.url for e in researcher.evidence_locker.get_all_evidence()]
        assert any(u.startswith("local://") for u in urls)  # local used
        assert any(u.startswith("https://") for u in urls)

    def test_v1_report_generated_from_web_session(self, tmp_path):
        researcher, session, llm, search = run_pipeline(
            tmp_path, ResearchSourceMode.WEB,
        )
        generator = ReportGenerator(output_dir=tmp_path / "reports", language="ja")
        report_path = generator.generate_report(
            session=session,
            evidence_locker=researcher.evidence_locker,
        )
        assert report_path.exists()
        text = report_path.read_text(encoding="utf-8")
        assert "市場規模の推移" in text
        assert "主要メーカーの動向" in text
        assert "References" in text or "参考文献" in text

    def test_v2_report_generated_from_session(self, tmp_path):
        researcher, session, llm, search = run_pipeline(
            tmp_path, ResearchSourceMode.WEB,
        )
        generator = ReportGeneratorV2(
            llm_client=llm,
            language="ja",
            enable_consistency_check=False,
            enable_polish=False,
        )
        result = generator.generate_report(
            research_topic=session.query,
            research_plan=session.research_plan,
            section_contents=session.section_contents,
        )
        assert result.chapters
        document = generator.generate_final_document(result)
        assert "章本文" in document
        assert "目次" in document

    def test_session_and_evidence_exported(self, tmp_path):
        researcher, session, llm, search = run_pipeline(
            tmp_path, ResearchSourceMode.WEB,
        )
        # conduct_research exports these at completion
        assert list(tmp_path.glob("session_*.json"))
        assert list((tmp_path / "evidence").glob("*.json"))
        csv_files = list((tmp_path / "evidence").glob("*.csv"))
        assert csv_files
        assert "importance_score" in csv_files[0].read_text(encoding="utf-8")
