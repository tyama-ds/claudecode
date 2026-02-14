"""Base stage class for pipeline stages."""

from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..context import PipelineContext, StageResult
from ..config import OrchestratorConfig


class BaseStage(ABC):
    """Abstract base class for all pipeline stages."""

    stage_type: str = "base"

    def __init__(self, name: str = "", config: Optional[OrchestratorConfig] = None):
        self.name = name or self.stage_type
        self.config = config
        self._progress_callback: Optional[Callable[[str, float], None]] = None

    def set_progress_callback(self, callback: Callable[[str, float], None]):
        self._progress_callback = callback

    def _report_progress(self, status: str, progress: float):
        if self._progress_callback:
            self._progress_callback(f"[{self.name}] {status}", progress)

    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Execute this stage. Wraps execute() with tracking.

        Args:
            context: Pipeline context

        Returns:
            Updated pipeline context
        """
        stage_result = StageResult(stage_name=self.name, stage_type=self.stage_type)
        self._report_progress("開始", 0.0)

        try:
            context = self.execute(context)
            stage_result.complete()
            stage_result.metadata["status"] = "success"
        except Exception as e:
            stage_result.complete()
            stage_result.metadata["status"] = "error"
            stage_result.metadata["error"] = str(e)
            raise

        context.record_stage(stage_result)
        self._report_progress("完了", 1.0)
        return context

    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute the stage logic.

        Args:
            context: Pipeline context with data from previous stages

        Returns:
            Updated pipeline context
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
