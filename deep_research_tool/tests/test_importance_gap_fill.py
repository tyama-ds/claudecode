"""
Tests for evidence importance scoring and gap-fill re-search.
"""

import json

import pytest
from unittest.mock import Mock, patch

from deep_research_tool.evidence.locker import EvidenceLocker
from deep_research_tool.research.content_extractor import ExtractedContent
from deep_research_tool.research.query_generator import TableOfContentsItem
from deep_research_tool.research.researcher import Researcher, ResearchSession


def make_researcher(tmp_path, llm=None, **kwargs):
    defaults = dict(
        importance_threshold=0.6,
        min_high_importance_sources=2,
        max_gap_fill_rounds=1,
    )
    defaults.update(kwargs)
    r = Researcher(
        llm_client=llm or Mock(),
        search_client=Mock(),
        output_dir=tmp_path,
        **defaults,
    )
    r.session = ResearchSession(query="炭素繊維の市場調査", requirements="国内市場中心")
    r.evidence_locker = EvidenceLocker(output_dir=tmp_path / "evidence")
    return r


def make_part(url="https://example.com/a", title="Source A",
              content="内容", relevance=0.5, importance=0.0):
    return ExtractedContent(
        source_url=url,
        source_title=title,
        raw_content=content,
        processed_content=content,
        relevance_score=relevance,
        importance_score=importance,
    )


def make_section():
    return TableOfContentsItem(section="1", title="市場動向", description="説明")


class TestEvidenceLockerImportance:
    def test_update_importance_by_url(self, tmp_path):
        locker = EvidenceLocker(output_dir=tmp_path)
        ev = locker.add_evidence(
            url="https://example.com/a", title="A",
            content_excerpt="x", section_reference="1",
        )
        updated = locker.update_importance_by_url(
            "https://example.com/a", 0.9, section="1",
        )
        assert updated == 1
        assert locker.get_evidence(ev.id).importance_score == 0.9

    def test_update_importance_by_url_respects_section(self, tmp_path):
        locker = EvidenceLocker(output_dir=tmp_path)
        locker.add_evidence(
            url="https://example.com/a", title="A",
            content_excerpt="x", section_reference="1",
        )
        updated = locker.update_importance_by_url(
            "https://example.com/a", 0.9, section="2",
        )
        assert updated == 0

    def test_get_evidence_by_importance_sorted(self, tmp_path):
        locker = EvidenceLocker(output_dir=tmp_path)
        low = locker.add_evidence(url="https://a.com", title="low", content_excerpt="1")
        high = locker.add_evidence(url="https://b.com", title="high", content_excerpt="2")
        locker.update_importance(low.id, 0.3)
        locker.update_importance(high.id, 0.9)

        result = locker.get_evidence_by_importance(min_importance=0.0)
        assert [e.id for e in result] == [high.id, low.id]

        filtered = locker.get_evidence_by_importance(min_importance=0.5)
        assert [e.id for e in filtered] == [high.id]

    def test_importance_in_csv_export(self, tmp_path):
        locker = EvidenceLocker(output_dir=tmp_path)
        ev = locker.add_evidence(url="https://a.com", title="t", content_excerpt="c")
        locker.update_importance(ev.id, 0.75)
        csv_path = locker.export_to_csv()
        text = csv_path.read_text(encoding="utf-8")
        assert "importance_score" in text
        assert "0.75" in text


class TestScoreImportance:
    def test_scores_parts_and_propagates_to_evidence(self, tmp_path):
        llm = Mock()
        llm.generate = Mock(return_value=Mock(content=json.dumps({
            "scores": [
                {"index": 1, "importance": 0.9},
                {"index": 2, "importance": 0.3},
            ]
        })))
        r = make_researcher(tmp_path, llm=llm)
        ev = r.evidence_locker.add_evidence(
            url="https://example.com/a", title="A",
            content_excerpt="x", section_reference="1",
        )
        parts = [
            make_part(url="https://example.com/a", title="A"),
            make_part(url="https://example.com/b", title="B"),
        ]

        r._score_importance(make_section(), parts)

        assert parts[0].importance_score == 0.9
        assert parts[1].importance_score == 0.3
        assert r.evidence_locker.get_evidence(ev.id).importance_score == 0.9
        # Prompt mentions the research purpose
        prompt = llm.generate.call_args.args[0]
        assert "炭素繊維の市場調査" in prompt

    def test_llm_failure_falls_back_to_relevance(self, tmp_path):
        llm = Mock()
        llm.generate = Mock(return_value=Mock(content="not json"))
        r = make_researcher(tmp_path, llm=llm)
        parts = [make_part(relevance=0.45)]

        r._score_importance(make_section(), parts)

        assert parts[0].importance_score == 0.45


class TestNeedsGapFill:
    def test_triggers_when_too_few_high_importance(self, tmp_path):
        r = make_researcher(tmp_path)
        parts = [make_part(importance=0.9), make_part(importance=0.3)]
        assert r._needs_gap_fill(parts, {"confidence_level": "high"}) is True

    def test_no_trigger_with_enough_high_importance(self, tmp_path):
        r = make_researcher(tmp_path)
        parts = [make_part(importance=0.9), make_part(importance=0.7)]
        assert r._needs_gap_fill(parts, {"confidence_level": "high"}) is False

    def test_triggers_on_low_confidence(self, tmp_path):
        r = make_researcher(tmp_path)
        parts = [make_part(importance=0.9), make_part(importance=0.7)]
        assert r._needs_gap_fill(parts, {"confidence_level": "low"}) is True


class TestGapFillFlow:
    def test_gap_fill_collects_and_resynthesizes(self, tmp_path):
        r = make_researcher(tmp_path)
        r.use_enhanced_synthesis = False
        section = make_section()

        # Initial part: low importance -> triggers gap fill
        parts = [make_part(url="https://example.com/a", importance=0.0, relevance=0.3)]

        # Importance scoring: keep low on first pass, high for the new source after
        def score(section_arg, parts_arg):
            for p in parts_arg:
                p.importance_score = 0.9 if "new" in p.source_url else 0.3
        r._score_importance = Mock(side_effect=score)

        synth_result = {
            "content": "本文である。[SOURCE 1]",
            "summary": "要約",
            "confidence_level": "medium",
            "information_gaps": ["価格データ"],
        }
        r.content_extractor.synthesize_section_content = Mock(return_value=synth_result)
        r.query_generator.generate_follow_up_queries = Mock(return_value=["価格 データ"])
        new_part = make_part(url="https://example.com/new", title="New", content="新情報")
        r._collect_additional_parts = Mock(return_value=[new_part])

        r._generate_and_save_section_content(section, parts)

        # Gap fill ran: follow-up queries generated, new source added, re-synthesized
        r.query_generator.generate_follow_up_queries.assert_called_once()
        r._collect_additional_parts.assert_called_once()
        assert r.content_extractor.synthesize_section_content.call_count == 2
        assert any("new" in p.source_url for p in parts)
        # High-importance source sorted first
        assert parts[0].source_url == "https://example.com/new"
        # Saved section reflects the merged sources
        saved = r.session.section_contents["1"]
        assert "https://example.com/new" in saved["sources"]

    def test_no_gap_fill_when_sufficient(self, tmp_path):
        r = make_researcher(tmp_path)
        r.use_enhanced_synthesis = False
        section = make_section()
        parts = [
            make_part(url="https://a.com", importance=0.0),
            make_part(url="https://b.com", importance=0.0),
        ]

        def score(section_arg, parts_arg):
            for p in parts_arg:
                p.importance_score = 0.9
        r._score_importance = Mock(side_effect=score)
        r.content_extractor.synthesize_section_content = Mock(return_value={
            "content": "十分な本文である。",
            "summary": "s",
            "confidence_level": "high",
            "information_gaps": [],
        })
        r.query_generator.generate_follow_up_queries = Mock()
        r._collect_additional_parts = Mock()

        r._generate_and_save_section_content(section, parts)

        r.query_generator.generate_follow_up_queries.assert_not_called()
        r._collect_additional_parts.assert_not_called()
        assert r.content_extractor.synthesize_section_content.call_count == 1

    def test_gap_fill_rounds_zero_disables(self, tmp_path):
        r = make_researcher(tmp_path, max_gap_fill_rounds=0)
        r.use_enhanced_synthesis = False
        section = make_section()
        parts = [make_part(importance=0.0, relevance=0.2)]

        r._score_importance = Mock()
        r.content_extractor.synthesize_section_content = Mock(return_value={
            "content": "本文である。", "summary": "s",
            "confidence_level": "low", "information_gaps": ["x"],
        })
        r.query_generator.generate_follow_up_queries = Mock()

        r._generate_and_save_section_content(section, parts)

        r.query_generator.generate_follow_up_queries.assert_not_called()

    def test_gap_fill_dedupes_same_url_same_content(self, tmp_path):
        r = make_researcher(tmp_path)
        r.use_enhanced_synthesis = False
        section = make_section()
        parts = [make_part(url="https://a.com", content="同一内容", importance=0.0)]

        r._score_importance = Mock(side_effect=lambda s, ps: [
            setattr(p, "importance_score", 0.3) for p in ps
        ])
        r.content_extractor.synthesize_section_content = Mock(return_value={
            "content": "本文である。", "summary": "s",
            "confidence_level": "medium", "information_gaps": ["x"],
        })
        r.query_generator.generate_follow_up_queries = Mock(return_value=["q2"])
        # Re-crawled page returns the SAME url and content -> duplicate, dropped
        dup = make_part(url="https://a.com", content="同一内容")
        r._collect_additional_parts = Mock(return_value=[dup])

        r._generate_and_save_section_content(section, parts)

        assert len(parts) == 1  # duplicate not added
        assert r.content_extractor.synthesize_section_content.call_count == 1

    def test_gap_fill_keeps_same_url_different_content(self, tmp_path):
        """既訪問サイトでも詳細が違えば再クロール結果を採用する。"""
        r = make_researcher(tmp_path)
        r.use_enhanced_synthesis = False
        section = make_section()
        parts = [make_part(url="https://a.com", content="旧内容", importance=0.0)]

        r._score_importance = Mock(side_effect=lambda s, ps: [
            setattr(p, "importance_score", 0.3) for p in ps
        ])
        r.content_extractor.synthesize_section_content = Mock(return_value={
            "content": "本文である。", "summary": "s",
            "confidence_level": "medium", "information_gaps": ["x"],
        })
        r.query_generator.generate_follow_up_queries = Mock(return_value=["q2"])
        revisit = make_part(url="https://a.com", content="新しい詳細内容")
        r._collect_additional_parts = Mock(return_value=[revisit])

        r._generate_and_save_section_content(section, parts)

        assert len(parts) == 2  # same URL, different detail -> kept
