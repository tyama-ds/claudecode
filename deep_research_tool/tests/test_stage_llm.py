"""
Tests for per-stage LLM routing (planning / crawling / evaluation / writing).
"""

import json

import pytest
from unittest.mock import Mock

from deep_research_tool.api.stage_router import LLM_STAGES, StageLLMRouter
from deep_research_tool.config import CrawlMode, create_config
from deep_research_tool.research.content_extractor import ExtractedContent
from deep_research_tool.research.query_generator import TableOfContentsItem
from deep_research_tool.research.researcher import Researcher, ResearchSession


class TestStageLLMRouter:
    def test_fallback_to_default(self):
        default = Mock()
        router = StageLLMRouter(default)
        assert router.for_stage("planning") is default
        assert router.has_override("planning") is False

    def test_stage_override(self):
        default, writer = Mock(), Mock()
        router = StageLLMRouter(default, {"writing": writer})
        assert router.for_stage("writing") is writer
        assert router.for_stage("planning") is default
        assert router.has_override("writing") is True

    def test_unknown_stage_raises(self):
        with pytest.raises(ValueError):
            StageLLMRouter(Mock(), {"drawing": Mock()})


class TestConfigStageOverrides:
    def test_stage_llm_flows_into_api_config(self):
        overrides = {
            "planning": {"provider": "openai", "model": "gpt-5-mini"},
            "writing": {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
        }
        config = create_config(stage_llm=overrides)
        assert config.api.stage_overrides == overrides

    def test_default_no_overrides(self):
        config = create_config()
        assert config.api.stage_overrides == {}

    def test_invalid_stage_name_raises(self):
        with pytest.raises(ValueError):
            create_config(stage_llm={"drawing": {"provider": "openai"}})

    def test_all_valid_stage_names_accepted(self):
        overrides = {s: {"provider": "openai", "model": "gpt-5-mini"} for s in LLM_STAGES}
        config = create_config(stage_llm=overrides)
        assert set(config.api.stage_overrides) == set(LLM_STAGES)


class TestResearcherStageDistribution:
    def make_researcher(self, tmp_path, **kwargs):
        return Researcher(
            llm_client=Mock(name="default"),
            search_client=Mock(),
            output_dir=tmp_path,
            **kwargs,
        )

    def test_defaults_to_single_client(self, tmp_path):
        default = Mock(name="default")
        r = Researcher(llm_client=default, search_client=Mock(), output_dir=tmp_path)
        assert r.planning_llm is default
        assert r.crawling_llm is default
        assert r.evaluation_llm is default
        assert r.writing_llm is default
        assert r.query_generator.llm is default
        assert r.content_extractor.llm is default
        assert r.content_extractor.eval_llm is default

    def test_stage_clients_distributed_to_components(self, tmp_path):
        planning, crawling, evaluation, writing = (
            Mock(name="planning"), Mock(name="crawling"),
            Mock(name="evaluation"), Mock(name="writing"),
        )
        r = self.make_researcher(
            tmp_path,
            crawl_mode=CrawlMode.AI_CRAWL,
            planning_llm=planning,
            crawling_llm=crawling,
            evaluation_llm=evaluation,
            writing_llm=writing,
        )
        assert r.query_generator.llm is planning
        assert r.ai_crawler.llm is crawling
        assert r.content_extractor.llm is writing        # prose synthesis
        assert r.content_extractor.eval_llm is evaluation  # extraction/quality

    def test_importance_scoring_uses_evaluation_client(self, tmp_path):
        evaluation = Mock(name="evaluation")
        evaluation.generate = Mock(return_value=Mock(content=json.dumps({
            "scores": [{"index": 1, "importance": 0.8}]
        })))
        default = Mock(name="default")
        r = Researcher(
            llm_client=default, search_client=Mock(), output_dir=tmp_path,
            evaluation_llm=evaluation,
        )
        r.session = ResearchSession(query="q")
        r.evidence_locker = None

        part = ExtractedContent(
            source_url="https://a.com", source_title="A",
            raw_content="x", processed_content="x",
        )
        r._score_importance(
            TableOfContentsItem(section="1", title="t"), [part],
        )

        evaluation.generate.assert_called_once()
        default.generate.assert_not_called()
        assert part.importance_score == 0.8


class TestContentExtractorEvalClient:
    def test_extraction_uses_eval_client_synthesis_uses_writing(self):
        from deep_research_tool.research.content_extractor import ContentExtractor

        writing = Mock(name="writing")
        writing.generate = Mock(return_value=Mock(
            content="本文。\n===SECTION_META===\n{\"summary\": \"s\"}"
        ))
        evaluation = Mock(name="evaluation")
        evaluation.generate = Mock(return_value=Mock(content=json.dumps({
            "processed_content": "抽出済みの内容である。",
            "relevance_score": 0.7,
        }, ensure_ascii=False)))

        extractor = ContentExtractor(writing, evaluation_llm_client=evaluation)

        extractor.extract_relevant_content(
            raw_content="raw", source_url="https://a.com",
            source_title="A", section_context="1", research_query="q",
        )
        evaluation.generate.assert_called_once()
        writing.generate.assert_not_called()

        extractor.synthesize_section_content(
            section_title="t", section_description="d",
            extracted_contents=[ExtractedContent(
                source_url="https://a.com", source_title="A",
                raw_content="x", processed_content="x",
            )],
        )
        writing.generate.assert_called_once()
