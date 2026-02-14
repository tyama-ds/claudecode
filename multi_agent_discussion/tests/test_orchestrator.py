"""Tests for the orchestrator module."""

import json
import pytest
import tempfile
from pathlib import Path

from multi_agent_discussion.orchestrator.context import (
    PipelineContext,
    ResearchResult,
    StageResult,
)
from multi_agent_discussion.orchestrator.config import (
    OrchestratorConfig,
    PipelineMode,
    ResearchAgentConfig,
    SynthesisConfig,
    RefinementConfig,
    DebateConfig,
    CompetitiveConfig,
    FactCheckConfig,
    ReportConfig,
    create_orchestrator_config,
)
from multi_agent_discussion.orchestrator.pipeline import Pipeline
from multi_agent_discussion.orchestrator.stages import (
    BaseStage,
    ParallelResearchStage,
    SynthesisStage,
    RefinementStage,
    DebateStage,
    CompetitiveStage,
    FactCheckStage,
    ReportStage,
)
from multi_agent_discussion.orchestrator.presets import (
    get_preset,
    PRESET_DESCRIPTIONS,
    build_multi_perspective,
    build_debate_research,
    build_iterative_refinement,
    build_competitive_analysis,
    build_full_pipeline,
)


class TestPipelineContext:
    """Tests for PipelineContext."""

    def test_create_context(self):
        """Test creating a new context."""
        ctx = PipelineContext(topic="テストトピック")
        assert ctx.topic == "テストトピック"
        assert ctx.pipeline_id  # auto-generated
        assert len(ctx.research_results) == 0
        assert ctx.final_report == ""

    def test_add_research_result(self):
        """Test adding a research result."""
        ctx = PipelineContext(topic="test")
        result = ResearchResult(
            agent_name="agent1",
            perspective="技術視点",
            report_content="レポート内容",
        )
        ctx.add_research_result(result)
        assert "agent1" in ctx.research_results
        assert ctx.research_results["agent1"].perspective == "技術視点"

    def test_get_all_evidence(self):
        """Test getting all evidence from results."""
        ctx = PipelineContext(topic="test")
        r1 = ResearchResult(
            agent_name="a1", perspective="p1", report_content="c1",
            evidence=[{"url": "http://a.com"}],
        )
        r2 = ResearchResult(
            agent_name="a2", perspective="p2", report_content="c2",
            evidence=[{"url": "http://b.com"}, {"url": "http://c.com"}],
        )
        ctx.add_research_result(r1)
        ctx.add_research_result(r2)
        evidence = ctx.get_all_evidence()
        assert len(evidence) == 3

    def test_get_all_report_contents(self):
        """Test getting all report contents."""
        ctx = PipelineContext(topic="test")
        ctx.add_research_result(ResearchResult(
            agent_name="a1", perspective="p1", report_content="content1",
        ))
        ctx.add_research_result(ResearchResult(
            agent_name="a2", perspective="p2", report_content="content2",
        ))
        contents = ctx.get_all_report_contents()
        assert contents["a1"] == "content1"
        assert contents["a2"] == "content2"

    def test_get_latest_report_priority(self):
        """Test report priority: final > refined > best > synthesized > research."""
        ctx = PipelineContext(topic="test")

        # With research results only
        ctx.add_research_result(ResearchResult(
            agent_name="a1", perspective="p1", report_content="research",
        ))
        assert "research" in ctx.get_latest_report()

        # Synthesized takes priority
        ctx.synthesized_report = "synthesized"
        assert ctx.get_latest_report() == "synthesized"

        # Best report takes priority
        ctx.best_report = "best"
        assert ctx.get_latest_report() == "best"

        # Refined report takes priority
        ctx.refined_reports = ["refined1", "refined2"]
        assert ctx.get_latest_report() == "refined2"

        # Final report takes priority
        ctx.final_report = "final"
        assert ctx.get_latest_report() == "final"

    def test_record_stage(self):
        """Test recording stage results."""
        ctx = PipelineContext(topic="test")
        stage = StageResult(stage_name="research", stage_type="parallel_research")
        stage.complete()
        stage.metadata["status"] = "success"
        ctx.record_stage(stage)
        assert len(ctx.stage_results) == 1
        assert ctx.stage_results[0].stage_name == "research"
        assert ctx.stage_results[0].completed_at is not None

    def test_serialization(self):
        """Test context to_dict and from_dict."""
        ctx = PipelineContext(topic="シリアライゼーションテスト")
        ctx.add_research_result(ResearchResult(
            agent_name="agent1",
            perspective="技術",
            report_content="テスト内容",
            evidence=[{"url": "http://test.com"}],
        ))
        ctx.synthesized_report = "統合レポート"
        ctx.discussion_transcript = "議論記録"

        data = ctx.to_dict()
        restored = PipelineContext.from_dict(data)

        assert restored.topic == ctx.topic
        assert restored.pipeline_id == ctx.pipeline_id
        assert "agent1" in restored.research_results
        assert restored.synthesized_report == "統合レポート"
        assert restored.discussion_transcript == "議論記録"

    def test_save_and_load(self):
        """Test saving and loading from file."""
        ctx = PipelineContext(topic="ファイルテスト")
        ctx.add_research_result(ResearchResult(
            agent_name="a1", perspective="p1", report_content="c1",
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_context.json"
            ctx.save(filepath=filepath)

            assert filepath.exists()

            loaded = PipelineContext.load(filepath)
            assert loaded.topic == "ファイルテスト"
            assert "a1" in loaded.research_results

    def test_save_auto_dir(self):
        """Test saving with auto-generated filepath."""
        ctx = PipelineContext(topic="autodir test")

        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = ctx.save(output_dir=tmpdir)
            assert result_path.exists()
            assert ctx.pipeline_id in str(result_path)


class TestResearchResult:
    """Tests for ResearchResult."""

    def test_create(self):
        """Test creating a research result."""
        r = ResearchResult(
            agent_name="test",
            perspective="技術",
            report_content="内容",
            evidence=[{"url": "http://test.com"}],
            session_id="abc",
        )
        assert r.agent_name == "test"
        assert r.perspective == "技術"
        assert len(r.evidence) == 1

    def test_serialization(self):
        """Test to_dict and from_dict."""
        r = ResearchResult(
            agent_name="test",
            perspective="技術",
            report_content="内容",
            evidence=[{"url": "http://test.com"}],
        )
        data = r.to_dict()
        restored = ResearchResult.from_dict(data)
        assert restored.agent_name == r.agent_name
        assert restored.report_content == r.report_content
        assert restored.evidence == r.evidence


class TestStageResult:
    """Tests for StageResult."""

    def test_create_and_complete(self):
        """Test creating and completing a stage result."""
        sr = StageResult(stage_name="test", stage_type="test_type")
        assert sr.completed_at is None
        sr.complete()
        assert sr.completed_at is not None

    def test_serialization(self):
        """Test to_dict."""
        sr = StageResult(stage_name="test", stage_type="test_type")
        sr.complete()
        sr.metadata["status"] = "success"
        data = sr.to_dict()
        assert data["stage_name"] == "test"
        assert data["stage_type"] == "test_type"
        assert data["completed_at"] is not None
        assert data["metadata"]["status"] == "success"


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""

    def test_create_default(self):
        """Test creating default config."""
        config = OrchestratorConfig(topic="テスト")
        assert config.topic == "テスト"
        assert isinstance(config.synthesis, SynthesisConfig)
        assert isinstance(config.refinement, RefinementConfig)
        assert isinstance(config.debate, DebateConfig)
        assert isinstance(config.competitive, CompetitiveConfig)
        assert isinstance(config.fact_check, FactCheckConfig)
        assert isinstance(config.report, ReportConfig)

    def test_validate_empty_topic(self):
        """Test validation with empty topic."""
        config = OrchestratorConfig()
        errors = config.validate()
        assert any("topic" in e.lower() or "トピック" in e for e in errors)

    def test_validate_no_agents(self):
        """Test validation with no agents."""
        config = OrchestratorConfig(topic="test")
        errors = config.validate()
        assert any("agent" in e.lower() for e in errors)

    def test_validate_agent_missing_perspective(self):
        """Test validation with agent missing perspective."""
        config = OrchestratorConfig(
            topic="test",
            research_agents=[
                ResearchAgentConfig(name="agent1", perspective=""),
            ],
        )
        errors = config.validate()
        assert len(errors) > 0

    def test_validate_agent_with_file(self):
        """Test validation allows agent with from_file."""
        config = OrchestratorConfig(
            topic="test",
            research_agents=[
                ResearchAgentConfig(name="agent1", perspective="", from_file="/path/to/file.json"),
            ],
        )
        errors = config.validate()
        assert len(errors) == 0

    def test_get_llm_config_fallback(self):
        """Test LLM config fallback to global."""
        from multi_agent_discussion.config import LLMConfig
        global_llm = LLMConfig()
        config = OrchestratorConfig(topic="test", llm_config=global_llm)

        # With None, should return global
        assert config.get_llm_config(None) is global_llm

        # With specific config, should return it
        specific = LLMConfig()
        assert config.get_llm_config(specific) is specific


class TestCreateOrchestratorConfig:
    """Tests for create_orchestrator_config factory."""

    def test_default_perspectives(self):
        """Test factory creates default perspectives."""
        config = create_orchestrator_config("AIの未来")
        assert config.topic == "AIの未来"
        assert len(config.research_agents) == 3

    def test_custom_perspectives(self):
        """Test factory with custom perspectives."""
        perspectives = [
            {"name": "技術者", "perspective": "技術的観点"},
            {"name": "経営者", "perspective": "ビジネス観点"},
        ]
        config = create_orchestrator_config("テスト", perspectives=perspectives)
        assert len(config.research_agents) == 2
        assert config.research_agents[0].name == "技術者"

    def test_proxy_url(self):
        """Test factory with proxy URL."""
        config = create_orchestrator_config("test", proxy_url="http://proxy:8080")
        assert config.llm_config.proxy_url == "http://proxy:8080"

    def test_provider_model(self):
        """Test factory with provider and model."""
        config = create_orchestrator_config(
            "test", provider="anthropic", model="claude-3-5-sonnet-20241022",
        )
        assert config.llm_config.provider.value == "anthropic"
        assert config.llm_config.anthropic_model == "claude-3-5-sonnet-20241022"


class TestPipelineMode:
    """Tests for PipelineMode enum."""

    def test_modes(self):
        """Test all modes are defined."""
        assert PipelineMode.SYNTHESIS == "synthesis"
        assert PipelineMode.REFINEMENT == "refinement"
        assert PipelineMode.DEBATE == "debate"
        assert PipelineMode.COMPETITIVE == "competitive"


class TestResearchAgentConfig:
    """Tests for ResearchAgentConfig."""

    def test_default_values(self):
        """Test default config values."""
        agent = ResearchAgentConfig(name="test", perspective="技術")
        assert agent.research_query == ""
        assert agent.llm_config is None
        assert agent.search_method == "duckduckgo"
        assert agent.max_iterations == 3
        assert agent.from_file is None

    def test_from_file_config(self):
        """Test file-based config."""
        agent = ResearchAgentConfig(
            name="loaded",
            perspective="",
            from_file="/path/to/session.json",
        )
        assert agent.from_file == "/path/to/session.json"


class TestPipeline:
    """Tests for Pipeline engine."""

    def test_create_pipeline(self):
        """Test creating a pipeline."""
        pipeline = Pipeline(topic="テスト")
        assert pipeline.topic == "テスト"
        assert len(pipeline.stages) == 0

    def test_add_stage(self):
        """Test adding stages."""
        config = OrchestratorConfig(topic="test")

        class DummyStage(BaseStage):
            stage_type = "dummy"
            def execute(self, context):
                return context

        pipeline = Pipeline(topic="test", config=config)
        pipeline.add_stage(DummyStage(name="stage1"))
        assert len(pipeline.stages) == 1

    def test_add_stage_chaining(self):
        """Test method chaining for add_stage."""
        config = OrchestratorConfig(topic="test")

        class DummyStage(BaseStage):
            stage_type = "dummy"
            def execute(self, context):
                return context

        pipeline = Pipeline(topic="test", config=config)
        result = pipeline.add_stage(DummyStage(name="s1")).add_stage(DummyStage(name="s2"))
        assert result is pipeline
        assert len(pipeline.stages) == 2

    def test_run_empty_pipeline_raises(self):
        """Test running empty pipeline raises ValueError."""
        pipeline = Pipeline(topic="test")
        with pytest.raises(ValueError, match="No stages"):
            pipeline.run()

    def test_run_pipeline(self):
        """Test running a pipeline with a dummy stage."""
        config = OrchestratorConfig(topic="test")

        class DummyStage(BaseStage):
            stage_type = "dummy"
            def execute(self, context):
                context.metadata["dummy_ran"] = True
                return context

        pipeline = Pipeline(topic="test", config=config)
        pipeline.add_stage(DummyStage(name="stage1", config=config))

        result = pipeline.run()
        assert result.metadata.get("dummy_ran") is True
        assert result.topic == "test"
        assert len(result.stage_results) == 1

    def test_progress_callback(self):
        """Test progress callback is called."""
        config = OrchestratorConfig(topic="test")
        progress_log = []

        class DummyStage(BaseStage):
            stage_type = "dummy"
            def execute(self, context):
                return context

        pipeline = Pipeline(topic="test", config=config)
        pipeline.add_stage(DummyStage(name="s1", config=config))
        pipeline.set_progress_callback(lambda s, p: progress_log.append((s, p)))

        pipeline.run()
        assert len(progress_log) > 0

    def test_stage_callback(self):
        """Test stage completion callback."""
        config = OrchestratorConfig(topic="test")
        stage_log = []

        class DummyStage(BaseStage):
            stage_type = "dummy"
            def execute(self, context):
                return context

        pipeline = Pipeline(topic="test", config=config)
        pipeline.add_stage(DummyStage(name="s1", config=config))
        pipeline.set_stage_callback(lambda name, ctx: stage_log.append(name))

        pipeline.run()
        assert stage_log == ["s1"]

    def test_multi_stage_pipeline(self):
        """Test pipeline with multiple stages."""
        config = OrchestratorConfig(topic="multi-test")

        class AddMetaStage(BaseStage):
            stage_type = "add_meta"
            def execute(self, context):
                key = f"stage_{self.name}"
                context.metadata[key] = True
                return context

        pipeline = Pipeline(topic="multi-test", config=config)
        pipeline.add_stage(AddMetaStage(name="first", config=config))
        pipeline.add_stage(AddMetaStage(name="second", config=config))
        pipeline.add_stage(AddMetaStage(name="third", config=config))

        result = pipeline.run()
        assert result.metadata["stage_first"] is True
        assert result.metadata["stage_second"] is True
        assert result.metadata["stage_third"] is True
        assert len(result.stage_results) == 3

    def test_repr(self):
        """Test pipeline repr."""
        pipeline = Pipeline(topic="テスト")
        assert "Pipeline" in repr(pipeline)

    def test_stages_property_returns_copy(self):
        """Test that stages property returns a copy."""
        config = OrchestratorConfig(topic="test")

        class DummyStage(BaseStage):
            stage_type = "dummy"
            def execute(self, context):
                return context

        pipeline = Pipeline(topic="test", config=config)
        pipeline.add_stage(DummyStage(name="s1", config=config))

        stages = pipeline.stages
        stages.append(DummyStage(name="extra", config=config))
        assert len(pipeline.stages) == 1  # original unchanged


class TestBaseStage:
    """Tests for BaseStage."""

    def test_stage_type(self):
        """Test stage type."""
        class TestStage(BaseStage):
            stage_type = "test"
            def execute(self, context):
                return context

        stage = TestStage(name="test_stage")
        assert stage.stage_type == "test"
        assert stage.name == "test_stage"

    def test_default_name(self):
        """Test default name from stage_type."""
        class TestStage(BaseStage):
            stage_type = "my_type"
            def execute(self, context):
                return context

        stage = TestStage()
        assert stage.name == "my_type"

    def test_run_records_result(self):
        """Test run method records stage result."""
        config = OrchestratorConfig(topic="test")

        class TestStage(BaseStage):
            stage_type = "test"
            def execute(self, context):
                return context

        stage = TestStage(name="tracked", config=config)
        ctx = PipelineContext(topic="test")
        result = stage.run(ctx)

        assert len(result.stage_results) == 1
        assert result.stage_results[0].stage_name == "tracked"
        assert result.stage_results[0].metadata["status"] == "success"

    def test_run_records_error(self):
        """Test run method records errors."""
        config = OrchestratorConfig(topic="test")

        class FailStage(BaseStage):
            stage_type = "fail"
            def execute(self, context):
                raise RuntimeError("test error")

        stage = FailStage(name="failing", config=config)
        ctx = PipelineContext(topic="test")

        with pytest.raises(RuntimeError, match="test error"):
            stage.run(ctx)

    def test_progress_callback(self):
        """Test progress reporting."""
        config = OrchestratorConfig(topic="test")
        progress_log = []

        class TestStage(BaseStage):
            stage_type = "test"
            def execute(self, context):
                self._report_progress("halfway", 0.5)
                return context

        stage = TestStage(name="prog", config=config)
        stage.set_progress_callback(lambda s, p: progress_log.append((s, p)))

        ctx = PipelineContext(topic="test")
        stage.run(ctx)

        # Should have: 開始, halfway, 完了
        assert len(progress_log) == 3
        assert "halfway" in progress_log[1][0]

    def test_repr(self):
        """Test stage repr."""
        class TestStage(BaseStage):
            stage_type = "test"
            def execute(self, context):
                return context

        stage = TestStage(name="my_stage")
        assert "TestStage" in repr(stage)
        assert "my_stage" in repr(stage)


class TestStageTypes:
    """Tests for stage type attributes."""

    def test_research_stage_type(self):
        """Test ParallelResearchStage type."""
        assert ParallelResearchStage.stage_type == "parallel_research"

    def test_synthesis_stage_type(self):
        """Test SynthesisStage type."""
        assert SynthesisStage.stage_type == "synthesis"

    def test_refinement_stage_type(self):
        """Test RefinementStage type."""
        assert RefinementStage.stage_type == "refinement"

    def test_debate_stage_type(self):
        """Test DebateStage type."""
        assert DebateStage.stage_type == "debate"

    def test_competitive_stage_type(self):
        """Test CompetitiveStage type."""
        assert CompetitiveStage.stage_type == "competitive"

    def test_fact_check_stage_type(self):
        """Test FactCheckStage type."""
        assert FactCheckStage.stage_type == "fact_check"

    def test_report_stage_type(self):
        """Test ReportStage type."""
        assert ReportStage.stage_type == "report"


class TestResearchStageFileLoading:
    """Tests for research stage file loading."""

    def test_load_deep_research_format(self):
        """Test loading a deep_research_tool session file."""
        config = OrchestratorConfig(
            topic="test",
            research_agents=[
                ResearchAgentConfig(name="test_agent", perspective="技術視点"),
            ],
        )
        stage = ParallelResearchStage(name="research", config=config)

        # Create a mock session file
        session_data = {
            "section_contents": {
                "sec1": {
                    "title": "Introduction",
                    "content": "This is the introduction.",
                    "sources": ["http://source1.com"],
                },
                "sec2": {
                    "title": "Analysis",
                    "content": "This is the analysis.",
                    "sources": ["http://source2.com"],
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(session_data, f, ensure_ascii=False)
            temp_path = f.name

        try:
            agent_config = ResearchAgentConfig(
                name="file_agent",
                perspective="技術視点",
                from_file=temp_path,
            )
            result = stage._load_from_file(agent_config)
            assert result.agent_name == "file_agent"
            assert "Introduction" in result.report_content
            assert "Analysis" in result.report_content
            assert len(result.evidence) == 2
            assert result.metadata["source"] == "file"
        finally:
            Path(temp_path).unlink()

    def test_load_own_format(self):
        """Test loading our own format."""
        config = OrchestratorConfig(
            topic="test",
            research_agents=[],
        )
        stage = ParallelResearchStage(name="research", config=config)

        data = {
            "report_content": "テストレポート内容",
            "evidence": [{"url": "http://test.com"}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, ensure_ascii=False)
            temp_path = f.name

        try:
            agent_config = ResearchAgentConfig(
                name="own_format",
                perspective="p",
                from_file=temp_path,
            )
            result = stage._load_from_file(agent_config)
            assert result.report_content == "テストレポート内容"
            assert len(result.evidence) == 1
        finally:
            Path(temp_path).unlink()


class TestRefinementScoreExtraction:
    """Tests for refinement stage score extraction."""

    def test_extract_score(self):
        """Test extracting score from review text."""
        config = OrchestratorConfig(topic="test")
        stage = RefinementStage(name="refinement", config=config)

        review = "色々なフィードバック...\n総合スコア: 0.85"
        score = stage._extract_score(review)
        assert score == 0.85

    def test_extract_score_colon(self):
        """Test extracting score with full-width colon."""
        config = OrchestratorConfig(topic="test")
        stage = RefinementStage(name="refinement", config=config)

        review = "総合スコア：0.72"
        score = stage._extract_score(review)
        assert score == 0.72

    def test_extract_score_no_match(self):
        """Test score extraction returns 0.0 when no match."""
        config = OrchestratorConfig(topic="test")
        stage = RefinementStage(name="refinement", config=config)

        review = "スコアが見つからない"
        score = stage._extract_score(review)
        assert score == 0.0


class TestPresets:
    """Tests for preset pipeline configurations."""

    def test_get_preset_multi_perspective(self):
        """Test multi_perspective preset."""
        pipeline = get_preset("multi_perspective", "テスト")
        assert isinstance(pipeline, Pipeline)
        stage_types = [s.stage_type for s in pipeline.stages]
        assert "parallel_research" in stage_types
        assert "synthesis" in stage_types
        assert "report" in stage_types

    def test_get_preset_debate_research(self):
        """Test debate_research preset."""
        pipeline = get_preset("debate_research", "テスト")
        stage_types = [s.stage_type for s in pipeline.stages]
        assert "parallel_research" in stage_types
        assert "debate" in stage_types
        assert "fact_check" in stage_types
        assert "report" in stage_types

    def test_get_preset_iterative_refinement(self):
        """Test iterative_refinement preset."""
        pipeline = get_preset("iterative_refinement", "テスト")
        stage_types = [s.stage_type for s in pipeline.stages]
        assert "parallel_research" in stage_types
        assert "synthesis" in stage_types
        assert "refinement" in stage_types
        assert "fact_check" in stage_types
        assert "report" in stage_types

    def test_get_preset_competitive_analysis(self):
        """Test competitive_analysis preset."""
        pipeline = get_preset("competitive_analysis", "テスト")
        stage_types = [s.stage_type for s in pipeline.stages]
        assert "parallel_research" in stage_types
        assert "competitive" in stage_types
        assert "refinement" in stage_types
        assert "report" in stage_types

    def test_get_preset_full(self):
        """Test full preset."""
        pipeline = get_preset("full", "テスト")
        stage_types = [s.stage_type for s in pipeline.stages]
        assert len(stage_types) == 7
        assert "parallel_research" in stage_types
        assert "synthesis" in stage_types
        assert "debate" in stage_types
        assert "competitive" in stage_types
        assert "refinement" in stage_types
        assert "fact_check" in stage_types
        assert "report" in stage_types

    def test_get_preset_unknown(self):
        """Test unknown preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent", "テスト")

    def test_get_preset_with_config(self):
        """Test preset with custom config."""
        config = create_orchestrator_config("カスタム")
        pipeline = get_preset("multi_perspective", "カスタム", config=config)
        assert pipeline.config is config

    def test_preset_descriptions(self):
        """Test preset descriptions contain all presets."""
        assert "multi_perspective" in PRESET_DESCRIPTIONS
        assert "debate_research" in PRESET_DESCRIPTIONS
        assert "iterative_refinement" in PRESET_DESCRIPTIONS
        assert "competitive_analysis" in PRESET_DESCRIPTIONS
        assert "full" in PRESET_DESCRIPTIONS

        for key, desc in PRESET_DESCRIPTIONS.items():
            assert "name" in desc
            assert "description" in desc
            assert "stages" in desc

    def test_pipeline_preset_classmethod(self):
        """Test Pipeline.preset() classmethod."""
        pipeline = Pipeline.preset("multi_perspective", "テスト")
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.stages) >= 3


class TestOrchestratorImports:
    """Tests for orchestrator module imports."""

    def test_import_from_orchestrator(self):
        """Test importing from orchestrator package."""
        from multi_agent_discussion.orchestrator import (
            OrchestratorConfig,
            PipelineMode,
            ResearchAgentConfig,
            PipelineContext,
            ResearchResult,
            StageResult,
            Pipeline,
            BaseStage,
            ParallelResearchStage,
            SynthesisStage,
            RefinementStage,
            DebateStage,
            CompetitiveStage,
            FactCheckStage,
            ReportStage,
            create_orchestrator_config,
        )
        # All imports should be non-None
        assert OrchestratorConfig is not None
        assert Pipeline is not None
        assert BaseStage is not None

    def test_import_stages(self):
        """Test importing from stages subpackage."""
        from multi_agent_discussion.orchestrator.stages import (
            BaseStage,
            ParallelResearchStage,
            SynthesisStage,
            RefinementStage,
            DebateStage,
            CompetitiveStage,
            FactCheckStage,
            ReportStage,
        )
        assert len([
            BaseStage, ParallelResearchStage, SynthesisStage,
            RefinementStage, DebateStage, CompetitiveStage,
            FactCheckStage, ReportStage,
        ]) == 8

    def test_import_presets(self):
        """Test importing presets."""
        from multi_agent_discussion.orchestrator.presets import (
            get_preset,
            PRESET_DESCRIPTIONS,
        )
        assert get_preset is not None
        assert isinstance(PRESET_DESCRIPTIONS, dict)

    def test_orchestrator_accessible_from_main_package(self):
        """Test orchestrator is accessible from main package."""
        import multi_agent_discussion
        assert hasattr(multi_agent_discussion, "orchestrator")
        from multi_agent_discussion import orchestrator
        assert hasattr(orchestrator, "Pipeline")
        assert hasattr(orchestrator, "PipelineContext")


class TestConfigDataclasses:
    """Tests for various config dataclasses."""

    def test_synthesis_config_defaults(self):
        """Test SynthesisConfig defaults."""
        c = SynthesisConfig()
        assert c.llm_config is None
        assert c.focus_areas == []
        assert c.max_length == 5000
        assert c.style == "academic"

    def test_refinement_config_defaults(self):
        """Test RefinementConfig defaults."""
        c = RefinementConfig()
        assert c.max_iterations == 3
        assert c.quality_threshold == 0.8
        assert len(c.review_criteria) == 4

    def test_debate_config_defaults(self):
        """Test DebateConfig defaults."""
        c = DebateConfig()
        assert c.max_rounds == 3
        assert c.include_fact_check is True

    def test_competitive_config_defaults(self):
        """Test CompetitiveConfig defaults."""
        c = CompetitiveConfig()
        assert len(c.evaluation_criteria) == 4
        assert c.merge_top_n == 0

    def test_fact_check_config_defaults(self):
        """Test FactCheckConfig defaults."""
        c = FactCheckConfig()
        assert c.search_engine == "duckduckgo"
        assert c.max_claims == 20

    def test_report_config_defaults(self):
        """Test ReportConfig defaults."""
        c = ReportConfig()
        assert c.output_format == "markdown"
        assert c.include_sources is True
        assert c.include_discussion is True
        assert c.include_fact_check is True
        assert c.language == "ja"
