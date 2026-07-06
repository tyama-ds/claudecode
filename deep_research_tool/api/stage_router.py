"""
Stage-based LLM routing.

Different pipeline stages (planning, crawl decisions, evaluation, writing)
can use different LLM providers/models. A stage without an override falls
back to the default client, so single-model setups keep working unchanged.

Stages:
    planning   - research plan / TOC / query generation
    crawling   - crawl decisions and page relevance evaluation
    evaluation - importance scoring, source quality, coherence checks
    writing    - section synthesis, chapter writing, polish, summaries
"""

from typing import Dict, Optional

# Recognized pipeline stages
LLM_STAGES = ("planning", "crawling", "evaluation", "writing")


class StageLLMRouter:
    """Maps pipeline stages to LLM clients with a default fallback."""

    def __init__(self, default_client, stage_clients: Optional[Dict[str, object]] = None):
        """
        Args:
            default_client: Client used for stages without an override
            stage_clients: Optional {stage_name: client} overrides
        """
        unknown = set(stage_clients or {}) - set(LLM_STAGES)
        if unknown:
            raise ValueError(
                f"Unknown LLM stages: {sorted(unknown)}. "
                f"Valid stages: {list(LLM_STAGES)}"
            )
        self.default = default_client
        self.stage_clients = dict(stage_clients or {})

    def for_stage(self, stage: str):
        """Return the client for a stage (default when no override)."""
        return self.stage_clients.get(stage) or self.default

    def has_override(self, stage: str) -> bool:
        """Whether the stage has a dedicated client."""
        return stage in self.stage_clients
