"""Pipeline engine for orchestrating multi-stage research workflows."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

from .config import OrchestratorConfig
from .context import PipelineContext
from .stages.base import BaseStage


class Pipeline:
    """
    Orchestrates a sequence of stages to produce a final output.

    Stages are executed in order, each receiving and returning a PipelineContext.
    """

    def __init__(
        self,
        topic: str,
        config: Optional[OrchestratorConfig] = None,
    ):
        self.topic = topic
        self.config = config or OrchestratorConfig(topic=topic)
        self._stages: List[BaseStage] = []
        self._progress_callback: Optional[Callable[[str, float], None]] = None
        self._stage_callback: Optional[Callable[[str, PipelineContext], None]] = None

    def add_stage(self, stage: BaseStage) -> "Pipeline":
        """
        Add a stage to the pipeline.

        Args:
            stage: Stage to add

        Returns:
            Self for chaining
        """
        if stage.config is None:
            stage.config = self.config
        self._stages.append(stage)
        return self

    def set_progress_callback(self, callback: Callable[[str, float], None]) -> "Pipeline":
        """Set callback for progress updates."""
        self._progress_callback = callback
        return self

    def set_stage_callback(self, callback: Callable[[str, PipelineContext], None]) -> "Pipeline":
        """Set callback called after each stage completes."""
        self._stage_callback = callback
        return self

    def run(
        self,
        context: Optional[PipelineContext] = None,
    ) -> PipelineContext:
        """
        Execute the pipeline.

        Args:
            context: Optional pre-populated context (for resuming or loading files)

        Returns:
            Final pipeline context with all results
        """
        if not self._stages:
            raise ValueError("No stages configured in the pipeline.")

        # Initialize context
        if context is None:
            context = PipelineContext(topic=self.topic)

        total_stages = len(self._stages)

        for i, stage in enumerate(self._stages):
            stage_progress_base = i / total_stages
            stage_progress_range = 1.0 / total_stages

            # Wire up progress callback with stage-aware scaling
            if self._progress_callback:
                def make_callback(base, range_):
                    def cb(status, progress):
                        overall = base + (progress * range_)
                        self._progress_callback(status, overall)
                    return cb
                stage.set_progress_callback(make_callback(stage_progress_base, stage_progress_range))

            # Execute stage
            context = stage.run(context)

            # Save intermediate results if configured
            if self.config.save_intermediate:
                context.save(output_dir=self.config.output_dir)

            # Notify stage completion
            if self._stage_callback:
                self._stage_callback(stage.name, context)

        return context

    @classmethod
    def preset(
        cls,
        preset_name: str,
        topic: str,
        config: Optional[OrchestratorConfig] = None,
        **kwargs,
    ) -> "Pipeline":
        """
        Create a pipeline from a preset configuration.

        Args:
            preset_name: Name of the preset
            topic: Research topic
            config: Optional custom config
            **kwargs: Additional arguments passed to preset builder

        Returns:
            Configured Pipeline
        """
        from .presets import get_preset
        return get_preset(preset_name, topic, config, **kwargs)

    @property
    def stages(self) -> List[BaseStage]:
        """Get the list of stages."""
        return list(self._stages)

    def __repr__(self) -> str:
        stage_names = [s.name for s in self._stages]
        return f"Pipeline(topic='{self.topic[:30]}...', stages={stage_names})"
