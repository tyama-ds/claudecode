"""Preset pipeline configurations for common workflows."""

from typing import Optional

from .config import OrchestratorConfig
from .pipeline import Pipeline
from .stages import (
    ParallelResearchStage,
    SynthesisStage,
    RefinementStage,
    DebateStage,
    CompetitiveStage,
    FactCheckStage,
    ReportStage,
)


def get_preset(
    name: str,
    topic: str,
    config: Optional[OrchestratorConfig] = None,
    **kwargs,
) -> Pipeline:
    """
    Get a preset pipeline by name.

    Args:
        name: Preset name
        topic: Research topic
        config: Optional custom config
        **kwargs: Additional arguments

    Returns:
        Configured Pipeline

    Raises:
        ValueError: If preset name is unknown
    """
    presets = {
        "multi_perspective": build_multi_perspective,
        "debate_research": build_debate_research,
        "iterative_refinement": build_iterative_refinement,
        "competitive_analysis": build_competitive_analysis,
        "full": build_full_pipeline,
    }

    builder = presets.get(name)
    if builder is None:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    return builder(topic, config, **kwargs)


def build_multi_perspective(
    topic: str,
    config: Optional[OrchestratorConfig] = None,
    **kwargs,
) -> Pipeline:
    """
    Multi-perspective synthesis pipeline.

    Research → Synthesis → Report

    Multiple agents research the topic from different perspectives,
    then results are synthesized into a unified report.
    """
    if config is None:
        from .config import create_orchestrator_config
        config = create_orchestrator_config(topic)

    pipeline = Pipeline(topic=topic, config=config)
    pipeline.add_stage(ParallelResearchStage(name="調査", config=config))
    pipeline.add_stage(SynthesisStage(name="統合", config=config))
    pipeline.add_stage(ReportStage(name="レポート", config=config))

    return pipeline


def build_debate_research(
    topic: str,
    config: Optional[OrchestratorConfig] = None,
    **kwargs,
) -> Pipeline:
    """
    Debate-based research pipeline.

    Research → Debate → Fact-check → Report

    Multiple agents research independently, then debate their findings.
    Results are fact-checked before final report generation.
    """
    if config is None:
        from .config import create_orchestrator_config
        config = create_orchestrator_config(topic)

    pipeline = Pipeline(topic=topic, config=config)
    pipeline.add_stage(ParallelResearchStage(name="調査", config=config))
    pipeline.add_stage(DebateStage(name="議論", config=config))
    pipeline.add_stage(FactCheckStage(name="ファクトチェック", config=config))
    pipeline.add_stage(ReportStage(name="レポート", config=config))

    return pipeline


def build_iterative_refinement(
    topic: str,
    config: Optional[OrchestratorConfig] = None,
    **kwargs,
) -> Pipeline:
    """
    Iterative refinement pipeline.

    Research → Synthesis → Refinement → Fact-check → Report

    Researches the topic, synthesizes findings, then iteratively
    refines the output through writer-reviewer cycles.
    """
    if config is None:
        from .config import create_orchestrator_config
        config = create_orchestrator_config(topic)

    pipeline = Pipeline(topic=topic, config=config)
    pipeline.add_stage(ParallelResearchStage(name="調査", config=config))
    pipeline.add_stage(SynthesisStage(name="統合", config=config))
    pipeline.add_stage(RefinementStage(name="洗練", config=config))
    pipeline.add_stage(FactCheckStage(name="ファクトチェック", config=config))
    pipeline.add_stage(ReportStage(name="レポート", config=config))

    return pipeline


def build_competitive_analysis(
    topic: str,
    config: Optional[OrchestratorConfig] = None,
    **kwargs,
) -> Pipeline:
    """
    Competitive analysis pipeline.

    Research → Competitive Evaluation → Refinement → Report

    Multiple agents research independently, their outputs are
    evaluated competitively, and the best result is refined.
    """
    if config is None:
        from .config import create_orchestrator_config
        config = create_orchestrator_config(topic)

    pipeline = Pipeline(topic=topic, config=config)
    pipeline.add_stage(ParallelResearchStage(name="調査", config=config))
    pipeline.add_stage(CompetitiveStage(name="競争評価", config=config))
    pipeline.add_stage(RefinementStage(name="洗練", config=config))
    pipeline.add_stage(ReportStage(name="レポート", config=config))

    return pipeline


def build_full_pipeline(
    topic: str,
    config: Optional[OrchestratorConfig] = None,
    **kwargs,
) -> Pipeline:
    """
    Full pipeline with all stages.

    Research → Synthesis → Debate → Competitive → Refinement → Fact-check → Report

    The most comprehensive pipeline: researches from multiple perspectives,
    synthesizes results, debates findings, evaluates competitively,
    refines the output, fact-checks, and generates the final report.
    """
    if config is None:
        from .config import create_orchestrator_config
        config = create_orchestrator_config(topic)

    pipeline = Pipeline(topic=topic, config=config)
    pipeline.add_stage(ParallelResearchStage(name="調査", config=config))
    pipeline.add_stage(SynthesisStage(name="統合", config=config))
    pipeline.add_stage(DebateStage(name="議論", config=config))
    pipeline.add_stage(CompetitiveStage(name="競争評価", config=config))
    pipeline.add_stage(RefinementStage(name="洗練", config=config))
    pipeline.add_stage(FactCheckStage(name="ファクトチェック", config=config))
    pipeline.add_stage(ReportStage(name="レポート", config=config))

    return pipeline


# Preset descriptions for help/documentation
PRESET_DESCRIPTIONS = {
    "multi_perspective": {
        "name": "多角的統合",
        "description": "複数の視点から調査し、統合レポートを生成",
        "stages": ["調査", "統合", "レポート"],
    },
    "debate_research": {
        "name": "議論型調査",
        "description": "調査結果をエージェント間で議論し、ファクトチェック後にレポートを生成",
        "stages": ["調査", "議論", "ファクトチェック", "レポート"],
    },
    "iterative_refinement": {
        "name": "反復洗練",
        "description": "統合レポートをライター・レビューアーのサイクルで洗練",
        "stages": ["調査", "統合", "洗練", "ファクトチェック", "レポート"],
    },
    "competitive_analysis": {
        "name": "競争分析",
        "description": "各調査結果を競争的に評価し、最良の結果を洗練",
        "stages": ["調査", "競争評価", "洗練", "レポート"],
    },
    "full": {
        "name": "フルパイプライン",
        "description": "全ステージを網羅する包括的なパイプライン",
        "stages": ["調査", "統合", "議論", "競争評価", "洗練", "ファクトチェック", "レポート"],
    },
}
