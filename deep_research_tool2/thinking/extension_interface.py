"""
Extension interface for DeepThink integration.

This module provides interfaces and base classes for extending DeepThink
functionality to other modules in the Deep Research Tool.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field

from .reasoning_chain import DeepThinkResult, ConsistencyCheckResult
from .deep_think import DeepThinkConfig, DeepThinkProcessor
from .metrics import DeepThinkMetrics


T = TypeVar('T')  # Generic type for processed content


@dataclass
class ExtensionConfig:
    """
    Configuration for DeepThink extension.

    Attributes:
        enabled: Whether the extension is active
        inherit_level: Inherit level from parent processor
        custom_level: Custom level override
        custom_threshold: Custom consistency threshold
        pre_hooks: Functions to run before processing
        post_hooks: Functions to run after processing
    """
    enabled: bool = True
    inherit_level: bool = True
    custom_level: Optional[float] = None
    custom_threshold: Optional[float] = None
    pre_hooks: List[Callable] = field(default_factory=list)
    post_hooks: List[Callable] = field(default_factory=list)


class DeepThinkExtension(ABC, Generic[T]):
    """
    Abstract base class for DeepThink extensions.

    Implement this class to add DeepThink capabilities to a new module.
    The extension handles preprocessing, DeepThink processing, and
    postprocessing of content.
    """

    def __init__(
        self,
        processor: DeepThinkProcessor = None,
        config: ExtensionConfig = None
    ):
        """
        Initialize the extension.

        Args:
            processor: Parent DeepThink processor
            config: Extension configuration
        """
        self.processor = processor
        self.config = config or ExtensionConfig()
        self._results_cache: Dict[str, DeepThinkResult] = {}

    @abstractmethod
    def preprocess(self, content: T) -> str:
        """
        Preprocess content before DeepThink.

        Convert module-specific content format to string for processing.

        Args:
            content: Module-specific content

        Returns:
            String representation for processing
        """
        pass

    @abstractmethod
    def postprocess(self, result: DeepThinkResult, original: T) -> T:
        """
        Postprocess DeepThink result.

        Convert processed result back to module-specific format.

        Args:
            result: DeepThink processing result
            original: Original content

        Returns:
            Processed content in module-specific format
        """
        pass

    def process(
        self,
        content: T,
        source_texts: Dict[str, str] = None,
        **kwargs
    ) -> T:
        """
        Process content through DeepThink.

        Args:
            content: Content to process
            source_texts: Source texts for reference
            **kwargs: Additional arguments

        Returns:
            Processed content
        """
        if not self.config.enabled or self.processor is None:
            return content

        # Run pre-hooks
        for hook in self.config.pre_hooks:
            content = hook(content)

        # Preprocess
        text_content = self.preprocess(content)

        # Process with DeepThink
        result = self.processor.process(
            content=text_content,
            source_texts=source_texts,
            **kwargs
        )

        # Cache result
        cache_key = str(hash(text_content[:100]))
        self._results_cache[cache_key] = result

        # Postprocess
        processed = self.postprocess(result, content)

        # Run post-hooks
        for hook in self.config.post_hooks:
            processed = hook(processed, result)

        return processed

    def get_last_result(self) -> Optional[DeepThinkResult]:
        """Get the most recent processing result."""
        if not self._results_cache:
            return None
        return list(self._results_cache.values())[-1]

    def clear_cache(self) -> None:
        """Clear the results cache."""
        self._results_cache.clear()


class SectionExtension(DeepThinkExtension[Dict[str, Any]]):
    """
    Extension for processing research sections.

    Handles section content dictionaries with 'content', 'sources', etc.
    """

    def preprocess(self, content: Dict[str, Any]) -> str:
        """Extract text content from section dictionary."""
        if isinstance(content, dict):
            return content.get("content", "")
        return str(content)

    def postprocess(
        self,
        result: DeepThinkResult,
        original: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update section dictionary with processed content."""
        if not isinstance(original, dict):
            return {"content": result.processed_content}

        updated = original.copy()
        updated["content"] = result.processed_content
        updated["deep_think_result"] = {
            "is_valid": result.is_valid,
            "confidence": result.overall_confidence,
            "metrics": result.metrics_summary,
        }
        return updated


class ReportExtension(DeepThinkExtension[str]):
    """
    Extension for processing report content.

    Handles full report text with section markers.
    """

    def preprocess(self, content: str) -> str:
        """Pass through string content."""
        return content

    def postprocess(self, result: DeepThinkResult, original: str) -> str:
        """Return processed content."""
        return result.processed_content


class VerificationExtension(DeepThinkExtension[Dict[str, Any]]):
    """
    Extension for enhancing verification with DeepThink metrics.

    Adds quantitative metrics to verification results.
    """

    def __init__(
        self,
        processor: DeepThinkProcessor = None,
        config: ExtensionConfig = None,
        metrics: DeepThinkMetrics = None
    ):
        super().__init__(processor, config)
        self.metrics = metrics or DeepThinkMetrics()

    def preprocess(self, content: Dict[str, Any]) -> str:
        """Extract content for verification."""
        return content.get("content", "")

    def postprocess(
        self,
        result: DeepThinkResult,
        original: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance verification result with metrics."""
        updated = original.copy()

        if result.consistency_result:
            updated["deep_think_verification"] = {
                "is_consistent": result.consistency_result.is_consistent,
                "deviation_score": result.consistency_result.deviation_score,
                "confidence_score": result.consistency_result.confidence_score,
                "semantic_deviation": result.consistency_result.semantic_deviation,
                "logical_gap": result.consistency_result.logical_gap,
                "contradiction_score": result.consistency_result.contradiction_score,
                "problematic_areas": result.consistency_result.problematic_areas,
                "suggestions": result.consistency_result.suggestions,
            }

        return updated


def create_extension_for_module(
    module_name: str,
    processor: DeepThinkProcessor,
    config: ExtensionConfig = None
) -> DeepThinkExtension:
    """
    Factory function to create appropriate extension for a module.

    Args:
        module_name: Name of the module to extend
        processor: DeepThink processor instance
        config: Extension configuration

    Returns:
        Appropriate DeepThinkExtension subclass instance
    """
    extensions = {
        "section": SectionExtension,
        "report": ReportExtension,
        "verification": VerificationExtension,
    }

    extension_class = extensions.get(module_name.lower(), SectionExtension)
    return extension_class(processor=processor, config=config)


class DeepThinkMixin:
    """
    Mixin class to add DeepThink capabilities to existing classes.

    Example usage:
        class MyModule(BaseModule, DeepThinkMixin):
            def process(self, content):
                if self.deep_think_enabled:
                    content = self.apply_deep_think(content)
                return super().process(content)
    """

    _deep_think_processor: Optional[DeepThinkProcessor] = None
    _deep_think_config: Optional[DeepThinkConfig] = None

    @property
    def deep_think_enabled(self) -> bool:
        """Check if DeepThink is enabled."""
        return (
            self._deep_think_config is not None and
            self._deep_think_config.enabled and
            self._deep_think_processor is not None
        )

    def setup_deep_think(
        self,
        llm_client,
        config: DeepThinkConfig = None
    ) -> None:
        """
        Set up DeepThink for this module.

        Args:
            llm_client: LLM client for processing
            config: DeepThink configuration
        """
        self._deep_think_config = config or DeepThinkConfig()
        if self._deep_think_config.enabled:
            self._deep_think_processor = DeepThinkProcessor(
                llm_client=llm_client,
                config=self._deep_think_config
            )

    def apply_deep_think(
        self,
        content: str,
        source_texts: Dict[str, str] = None
    ) -> str:
        """
        Apply DeepThink processing to content.

        Args:
            content: Content to process
            source_texts: Source texts for reference

        Returns:
            Processed content
        """
        if not self.deep_think_enabled:
            return content

        result = self._deep_think_processor.process(
            content=content,
            source_texts=source_texts
        )
        return result.processed_content

    def get_deep_think_metrics(self) -> Optional[Dict[str, float]]:
        """Get metrics from last DeepThink processing."""
        if not hasattr(self, '_deep_think_processor') or self._deep_think_processor is None:
            return None

        # This would need access to the last result
        # Implementation depends on processor state management
        return None


class DeepThinkHook:
    """
    Hook interface for injecting DeepThink at specific points.

    Allows fine-grained control over when DeepThink is applied.
    """

    def __init__(
        self,
        processor: DeepThinkProcessor,
        trigger_condition: Callable[[Any], bool] = None
    ):
        """
        Initialize the hook.

        Args:
            processor: DeepThink processor
            trigger_condition: Function to determine if hook should trigger
        """
        self.processor = processor
        self.trigger_condition = trigger_condition or (lambda x: True)

    def __call__(
        self,
        content: str,
        context: Dict[str, Any] = None
    ) -> str:
        """
        Execute the hook.

        Args:
            content: Content to process
            context: Additional context

        Returns:
            Processed content
        """
        if not self.trigger_condition(content):
            return content

        source_texts = {}
        if context and "sources" in context:
            source_texts = context["sources"]

        result = self.processor.process(
            content=content,
            source_texts=source_texts
        )

        return result.processed_content
