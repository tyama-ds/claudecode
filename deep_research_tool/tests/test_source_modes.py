"""
Tests for research source modes (web / local / hybrid) and the local
document store.
"""

import json

import pytest
from unittest.mock import Mock

from deep_research_tool.config import ResearchSourceMode, create_config
from deep_research_tool.evidence.locker import EvidenceLocker
from deep_research_tool.research.local_store import LocalDocumentStore
from deep_research_tool.research.query_generator import TableOfContentsItem
from deep_research_tool.research.researcher import Researcher, ResearchSession


class TestLocalDocumentStore:
    def test_add_and_search(self):
        store = LocalDocumentStore(chunk_size=200, overlap=20)
        store.add_document(
            title="市場レポート",
            content="炭素繊維の市場規模は拡大している。" * 20,
            path="report.pdf",
        )
        store.add_document(
            title="無関係文書",
            content="今日の天気は晴れである。" * 20,
        )
        assert store.document_count == 2
        assert store.chunk_count >= 2

        results = store.search("炭素繊維 市場規模", top_k=3, keywords=["炭素繊維"])
        assert results
        assert all(c.doc_title == "市場レポート" for c in results)
        assert results[0].score > 0
        assert results[0].source_url.startswith("local://report.pdf#chunk")

    def test_empty_store(self):
        store = LocalDocumentStore()
        assert store.is_empty()
        assert store.search("anything") == []

    def test_empty_content_ignored(self):
        store = LocalDocumentStore()
        assert store.add_document(title="empty", content="   ") == 0
        assert store.is_empty()


class TestSourceModeConfig:
    def test_source_mode_values(self):
        for value in ("web", "local", "hybrid"):
            config = create_config(source_mode=value)
            assert config.research.source_mode == ResearchSourceMode(value)

    def test_default_is_web(self):
        assert create_config().research.source_mode == ResearchSourceMode.WEB

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            create_config(source_mode="offline")


def make_researcher(tmp_path, source_mode, llm=None, search=None, **kwargs):
    return Researcher(
        llm_client=llm or Mock(),
        search_client=search or Mock(),
        output_dir=tmp_path,
        source_mode=source_mode,
        max_gap_fill_rounds=0,
        **kwargs,
    )


def extraction_json(content="抽出済みの関連コンテンツである。", relevance=0.7):
    return json.dumps({
        "processed_content": content,
        "key_points": ["要点"],
        "relevance_score": relevance,
    }, ensure_ascii=False)


class TestLocalMode:
    def test_local_mode_requires_documents(self, tmp_path):
        r = make_researcher(tmp_path, ResearchSourceMode.LOCAL)
        with pytest.raises(ValueError, match="local"):
            r.conduct_research(query="テーマ", additional_documents=None)

    def test_collect_local_parts_extracts_and_stores_evidence(self, tmp_path):
        llm = Mock()
        llm.generate = Mock(return_value=Mock(content=extraction_json()))
        r = make_researcher(tmp_path, ResearchSourceMode.LOCAL, llm=llm)
        r.session = ResearchSession(query="炭素繊維の市場調査")
        r.evidence_locker = EvidenceLocker(output_dir=tmp_path / "ev")
        r.local_store.add_document(
            title="社内資料",
            content="炭素繊維の市場は年率10%で成長している。" * 30,
            path="internal.pdf",
        )

        section = TableOfContentsItem(section="1", title="市場動向", description="炭素繊維")
        parts = r._collect_local_parts(section, ["炭素繊維 市場"])

        assert parts
        assert all(p.source_url.startswith("local://") for p in parts)
        evidence = r.evidence_locker.get_section_evidence("1")
        assert evidence
        assert evidence[0].access_method == "local"
        # No web access at all
        r.search.search.assert_not_called()
        r.search.get_page_content.assert_not_called()

    def test_gap_fill_in_local_mode_stays_local(self, tmp_path):
        llm = Mock()
        llm.generate = Mock(return_value=Mock(content=extraction_json()))
        search = Mock()
        r = make_researcher(tmp_path, ResearchSourceMode.LOCAL, llm=llm, search=search)
        r.session = ResearchSession(query="炭素繊維")
        r.evidence_locker = EvidenceLocker(output_dir=tmp_path / "ev")
        r.local_store.add_document(title="doc", content="炭素繊維の情報。" * 50)

        section = TableOfContentsItem(section="1", title="炭素繊維", description="")
        parts = r._collect_additional_parts(section, ["炭素繊維 詳細"])

        assert parts  # found locally
        search.search.assert_not_called()
        search.get_page_content.assert_not_called()


class TestHybridMode:
    def test_hybrid_merges_local_and_web_parts(self, tmp_path):
        """Hybrid: local parts are passed as extra_parts into the web path."""
        llm = Mock()
        llm.generate = Mock(return_value=Mock(content=extraction_json()))
        search = Mock()
        search.search = Mock(return_value=[])  # web finds nothing
        r = make_researcher(tmp_path, ResearchSourceMode.HYBRID, llm=llm, search=search)
        r.session = ResearchSession(query="炭素繊維")
        r.evidence_locker = EvidenceLocker(output_dir=tmp_path / "ev")
        r.local_store.add_document(title="doc", content="炭素繊維の市場情報。" * 50)
        r._score_importance = Mock()
        r.content_extractor.synthesize_section_content = Mock(return_value={
            "content": "本文である。", "summary": "s",
            "confidence_level": "high", "information_gaps": [],
        })
        r.use_enhanced_synthesis = False

        section = TableOfContentsItem(section="1", title="炭素繊維", description="")
        local_parts = r._collect_local_parts(section, ["q"])
        assert local_parts

        r._process_section_with_immediate_generation(
            section=section, available_queries=["q"],
            section_idx=0, total_sections=1,
            extra_parts=local_parts,
        )

        # Web search WAS attempted (hybrid) and local parts reached synthesis
        search.search.assert_called()
        synth_parts = r.content_extractor.synthesize_section_content.call_args.kwargs[
            "extracted_contents"
        ]
        assert any(p.source_url.startswith("local://") for p in synth_parts)

    def test_web_mode_skips_local_store(self, tmp_path):
        r = make_researcher(tmp_path, ResearchSourceMode.WEB)
        r.session = ResearchSession(query="q")
        # web mode: documents are not indexed into the local store
        assert r.local_store.is_empty()
