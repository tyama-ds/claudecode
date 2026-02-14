"""
Orchestrator module for multi-stage research pipelines.

Combines deep_research_tool and multi_agent_discussion into
configurable pipelines with multiple processing modes.
"""

from .config import (
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
from .context import PipelineContext, ResearchResult, StageResult
from .pipeline import Pipeline
from .stages import (
    BaseStage,
    ParallelResearchStage,
    SynthesisStage,
    RefinementStage,
    DebateStage,
    CompetitiveStage,
    FactCheckStage,
    ReportStage,
)

__all__ = [
    # Config
    "OrchestratorConfig",
    "PipelineMode",
    "ResearchAgentConfig",
    "SynthesisConfig",
    "RefinementConfig",
    "DebateConfig",
    "CompetitiveConfig",
    "FactCheckConfig",
    "ReportConfig",
    "create_orchestrator_config",
    # Context
    "PipelineContext",
    "ResearchResult",
    "StageResult",
    # Pipeline
    "Pipeline",
    # Stages
    "BaseStage",
    "ParallelResearchStage",
    "SynthesisStage",
    "RefinementStage",
    "DebateStage",
    "CompetitiveStage",
    "FactCheckStage",
    "ReportStage",
]
