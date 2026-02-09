"""
DeepThink module for enhanced reasoning in Deep Research Tool.

This module provides deep iterative reasoning capabilities with:
- Quantitative metrics evaluation using embeddings
- Consistency checking against original facts
- Extensibility for integration with other modules
- Configurable reasoning depth and parameters

Example usage:
    from deep_research_tool2.thinking import DeepThinkProcessor, DeepThinkConfig

    # Configure DeepThink
    config = DeepThinkConfig(
        enabled=True,
        level=0.5,
        reasoning_iterations=3,
        consistency_threshold=0.3,
    )

    # Create processor
    processor = DeepThinkProcessor(llm_client=llm_client, config=config)

    # Process content
    result = processor.process(content, source_texts=sources)

    # Check results
    print(f"Confidence: {result.overall_confidence}")
    print(f"Is valid: {result.is_valid}")

For extension to other modules, see extension_interface.py or the
extension guide in docs/deep_think_extension_guide.md
"""

from .reasoning_chain import (
    ReasoningType,
    ReasoningStep,
    ReasoningChain,
    ConsistencyMode,
    ConsistencyCheckResult,
    DeepThinkResult,
)

from .metrics import (
    DeepThinkMetrics,
    MetricsConfig,
    HAS_SENTENCE_TRANSFORMERS,
)

from .logic_validator import (
    LogicValidator,
    ValidationConfig,
    ValidationLevel,
)

from .deep_think import (
    DeepThinkConfig,
    DeepThinkProcessor,
)

from .extension_interface import (
    ExtensionConfig,
    DeepThinkExtension,
    SectionExtension,
    ReportExtension,
    VerificationExtension,
    DeepThinkMixin,
    DeepThinkHook,
    create_extension_for_module,
)

__all__ = [
    # Reasoning chain classes
    "ReasoningType",
    "ReasoningStep",
    "ReasoningChain",
    "ConsistencyMode",
    "ConsistencyCheckResult",
    "DeepThinkResult",
    # Metrics
    "DeepThinkMetrics",
    "MetricsConfig",
    "HAS_SENTENCE_TRANSFORMERS",
    # Validation
    "LogicValidator",
    "ValidationConfig",
    "ValidationLevel",
    # Main processor
    "DeepThinkConfig",
    "DeepThinkProcessor",
    # Extensions
    "ExtensionConfig",
    "DeepThinkExtension",
    "SectionExtension",
    "ReportExtension",
    "VerificationExtension",
    "DeepThinkMixin",
    "DeepThinkHook",
    "create_extension_for_module",
]
