"""Configuration for the orchestrator pipeline."""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any

from ..config import LLMConfig, LLMProvider


class PipelineMode(str, Enum):
    """Available pipeline modes."""
    SYNTHESIS = "synthesis"
    REFINEMENT = "refinement"
    DEBATE = "debate"
    COMPETITIVE = "competitive"


@dataclass
class ResearchAgentConfig:
    """Configuration for a research agent in the pipeline."""
    name: str
    perspective: str
    research_query: str = ""  # Auto-generated from topic if empty
    llm_config: Optional[LLMConfig] = None
    # deep_research_tool specific settings
    search_method: str = "duckduckgo"
    max_iterations: int = 3
    # Alternatively, load from existing file
    from_file: Optional[str] = None  # Path to existing session JSON


@dataclass
class SynthesisConfig:
    """Configuration for synthesis stage."""
    llm_config: Optional[LLMConfig] = None
    focus_areas: List[str] = field(default_factory=list)
    max_length: int = 5000
    style: str = "academic"  # academic, business, casual


@dataclass
class RefinementConfig:
    """Configuration for refinement stage."""
    llm_config: Optional[LLMConfig] = None
    reviewer_llm_config: Optional[LLMConfig] = None  # Can use different LLM for review
    max_iterations: int = 3
    quality_threshold: float = 0.8
    review_criteria: List[str] = field(default_factory=lambda: [
        "論理的整合性",
        "根拠の充実度",
        "網羅性",
        "読みやすさ",
    ])


@dataclass
class DebateConfig:
    """Configuration for debate stage."""
    max_rounds: int = 3
    include_fact_check: bool = True


@dataclass
class CompetitiveConfig:
    """Configuration for competitive evaluation stage."""
    evaluator_llm_config: Optional[LLMConfig] = None
    evaluation_criteria: List[str] = field(default_factory=lambda: [
        "正確性",
        "網羅性",
        "論理性",
        "独自の視点",
    ])
    merge_top_n: int = 0  # 0 = pick best, >0 = merge top N


@dataclass
class FactCheckConfig:
    """Configuration for fact-check stage."""
    llm_config: Optional[LLMConfig] = None
    search_engine: str = "duckduckgo"
    max_claims: int = 20


@dataclass
class ReportConfig:
    """Configuration for report generation stage."""
    llm_config: Optional[LLMConfig] = None
    output_format: str = "markdown"  # markdown, docx, pdf, html
    output_dir: str = "./pipeline_output"
    include_sources: bool = True
    include_discussion: bool = True
    include_fact_check: bool = True
    language: str = "ja"


@dataclass
class OrchestratorConfig:
    """Main configuration for the orchestrator pipeline."""
    topic: str = ""
    # Global LLM config (fallback for stages without specific config)
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    # Research agents
    research_agents: List[ResearchAgentConfig] = field(default_factory=list)
    # Stage configurations
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    competitive: CompetitiveConfig = field(default_factory=CompetitiveConfig)
    fact_check: FactCheckConfig = field(default_factory=FactCheckConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    # Pipeline control
    save_intermediate: bool = True
    output_dir: str = "./pipeline_output"

    def get_llm_config(self, stage_config_llm: Optional[LLMConfig] = None) -> LLMConfig:
        """Get LLM config with fallback to global."""
        return stage_config_llm or self.llm_config

    def validate(self) -> List[str]:
        """Validate the configuration."""
        errors = []
        if not self.topic:
            errors.append("Pipeline topic is not set.")
        if not self.research_agents:
            errors.append("No research agents configured.")
        for agent in self.research_agents:
            if not agent.name:
                errors.append("Research agent name is required.")
            if not agent.from_file and not agent.perspective:
                errors.append(f"Agent '{agent.name}': perspective or from_file is required.")
        return errors


def create_orchestrator_config(
    topic: str,
    perspectives: Optional[List[Dict[str, str]]] = None,
    provider: str = "openai",
    model: Optional[str] = None,
    proxy_url: Optional[str] = None,
) -> OrchestratorConfig:
    """
    Create an orchestrator config with sensible defaults.

    Args:
        topic: Research topic
        perspectives: List of dicts with 'name' and 'perspective' keys
        provider: Default LLM provider
        model: Default model name
        proxy_url: Proxy URL for all LLM clients

    Returns:
        Configured OrchestratorConfig
    """
    llm_config = LLMConfig(provider=LLMProvider(provider), proxy_url=proxy_url)
    if model:
        model_attr_map = {
            "openai": "openai_model",
            "anthropic": "anthropic_model",
            "google": "google_model",
            "ollama": "ollama_model",
            "xai": "xai_model",
        }
        attr = model_attr_map.get(provider)
        if attr:
            setattr(llm_config, attr, model)

    agents = []
    if perspectives:
        for p in perspectives:
            agents.append(ResearchAgentConfig(
                name=p["name"],
                perspective=p["perspective"],
                research_query=p.get("query", ""),
            ))
    else:
        # Default: 3 perspectives
        agents = [
            ResearchAgentConfig(name="技術視点", perspective="技術的な観点から調査する"),
            ResearchAgentConfig(name="社会視点", perspective="社会的影響の観点から調査する"),
            ResearchAgentConfig(name="経済視点", perspective="経済・ビジネスの観点から調査する"),
        ]

    return OrchestratorConfig(
        topic=topic,
        llm_config=llm_config,
        research_agents=agents,
        report=ReportConfig(llm_config=llm_config),
    )
