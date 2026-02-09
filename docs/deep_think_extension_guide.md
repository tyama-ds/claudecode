# DeepThink Extension Guide

This guide explains how to extend and integrate DeepThink functionality into other modules of the Deep Research Tool.

## Overview

DeepThink provides enhanced reasoning capabilities through:
- **Iterative reasoning chains** - Structured fact extraction, inference, and conclusion
- **Quantitative metrics** - Embedding-based similarity and statistical evaluation
- **Consistency checking** - Verification that conclusions don't deviate from source facts
- **Extensibility** - Interfaces for integration with other modules

## Architecture

```
thinking/
├── __init__.py              # Module exports
├── reasoning_chain.py       # Data classes (ReasoningStep, ReasoningChain, etc.)
├── metrics.py               # Quantitative evaluation (DeepThinkMetrics)
├── logic_validator.py       # Logic validation (LogicValidator)
├── deep_think.py            # Main processor (DeepThinkProcessor)
└── extension_interface.py   # Extension interfaces
```

## Core Components

### DeepThinkConfig

Configuration for DeepThink processing:

```python
from deep_research_tool.thinking import DeepThinkConfig

config = DeepThinkConfig(
    enabled=True,
    level=0.5,                    # 0.0 (conservative) to 1.0 (exploratory)
    reasoning_iterations=3,       # Number of reasoning iterations
    consistency_threshold=0.3,    # Max acceptable deviation
    consistency_mode="warn",      # "warn", "revise", or "strict"
    fidelity_threshold=0.7,       # Min source fidelity score
)
```

### DeepThinkProcessor

Main processor for DeepThink operations:

```python
from deep_research_tool.thinking import DeepThinkProcessor

processor = DeepThinkProcessor(
    llm_client=llm_client,
    config=config,
    language="ja"
)

result = processor.process(
    content="Text to process...",
    source_texts={"source1": "Original source text..."}
)

print(f"Confidence: {result.overall_confidence}")
print(f"Is valid: {result.is_valid}")
```

### DeepThinkMetrics

Quantitative evaluation using embeddings:

```python
from deep_research_tool.thinking import DeepThinkMetrics

metrics = DeepThinkMetrics()

# Source fidelity
fidelity = metrics.calc_source_fidelity(
    fact_text="Extracted fact...",
    source_text="Original source..."
)

# Logical coherence
coherence = metrics.calc_logical_coherence(
    premises=["Premise 1", "Premise 2"],
    conclusion="Conclusion..."
)

# Deviation score
deviation = metrics.calc_deviation_score(
    initial_facts=["Fact 1", "Fact 2"],
    conclusion="Final conclusion...",
    intermediate_conclusions=["Intermediate 1"]
)
```

## Extension Methods

### 1. Using DeepThinkExtension Base Class

Create a custom extension by inheriting from `DeepThinkExtension`:

```python
from deep_research_tool.thinking import DeepThinkExtension, DeepThinkResult
from typing import Dict, Any

class CustomExtension(DeepThinkExtension[Dict[str, Any]]):
    """Extension for custom content type."""

    def preprocess(self, content: Dict[str, Any]) -> str:
        """Convert custom format to string."""
        return content.get("text", "")

    def postprocess(
        self,
        result: DeepThinkResult,
        original: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert result back to custom format."""
        updated = original.copy()
        updated["text"] = result.processed_content
        updated["deep_think_metrics"] = result.metrics_summary
        return updated
```

### 2. Using DeepThinkMixin

Add DeepThink capabilities to existing classes:

```python
from deep_research_tool.thinking import DeepThinkMixin, DeepThinkConfig

class MyProcessor(BaseProcessor, DeepThinkMixin):
    """Custom processor with DeepThink support."""

    def __init__(self, llm_client, enable_deep_think=False):
        super().__init__()

        if enable_deep_think:
            config = DeepThinkConfig(enabled=True, level=0.5)
            self.setup_deep_think(llm_client, config)

    def process(self, content: str) -> str:
        # Apply DeepThink if enabled
        if self.deep_think_enabled:
            content = self.apply_deep_think(content)

        # Continue with normal processing
        return super().process(content)
```

### 3. Using DeepThinkHook

Inject DeepThink at specific points:

```python
from deep_research_tool.thinking import DeepThinkHook, DeepThinkProcessor

processor = DeepThinkProcessor(llm_client, config)

# Create hook with trigger condition
hook = DeepThinkHook(
    processor=processor,
    trigger_condition=lambda x: len(x) > 1000  # Only for long content
)

# Use hook
processed = hook(content, context={"sources": source_texts})
```

### 4. Using Factory Function

Create appropriate extension for known module types:

```python
from deep_research_tool.thinking import create_extension_for_module

# For section processing
section_ext = create_extension_for_module("section", processor)

# For report processing
report_ext = create_extension_for_module("report", processor)

# For verification enhancement
verify_ext = create_extension_for_module("verification", processor)
```

## Metrics Reference

### Source Fidelity
Measures how faithfully facts are extracted from sources.
- Uses cosine similarity between embeddings
- Range: 0.0 (no relation) to 1.0 (identical)
- Threshold: `fidelity_threshold` (default: 0.7)

### Logical Coherence
Measures logical flow from premises to conclusion.
- Combines semantic similarity (70%) and vocabulary overlap (30%)
- Range: 0.0 (no coherence) to 1.0 (perfect coherence)

### Expansion Degree
Measures new information introduced in conclusion.
- Based on vocabulary not present in premises
- Range: 0.0 (no new info) to 1.0 (all new)
- Control: `_expansion_tolerance` (default: 0.2)

### Deviation Score
Comprehensive deviation measure combining:
- Semantic deviation (weight: 0.4)
- Logical gap (weight: 0.4)
- Contradiction score (weight: 0.3)
- Weights configurable via `_deviation_weights`

### Confidence Score
Final confidence using sigmoid normalization:
- Combines fidelity, coherence, deviation, expansion
- Range: 0.0 to 1.0
- Used for pass/fail determination

## Consistency Modes

| Mode | Behavior |
|------|----------|
| `warn` | Log warning, continue with original conclusion |
| `revise` | Attempt to revise conclusion to fix issues |
| `strict` | Reject conclusions that fail consistency check |

## Advanced Configuration

### Hidden Parameters

These parameters are for advanced tuning:

```python
config = DeepThinkConfig(
    # ... standard params ...
    _expansion_tolerance=0.2,        # Max acceptable expansion
    _deviation_weights=(0.4, 0.4, 0.2),  # (semantic, logical, contradiction)
)
```

### Custom Embedding Model

```python
from deep_research_tool.thinking import MetricsConfig, DeepThinkMetrics

metrics_config = MetricsConfig(
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
    fidelity_threshold=0.7,
    coherence_threshold=0.6,
)

metrics = DeepThinkMetrics(metrics_config)
```

## Integration Examples

### With Researcher Module

```python
from deep_research_tool import DeepResearchTool, create_config

config = create_config(
    provider="anthropic",
    deep_think=True,
    deep_think_level=0.6,
    consistency_mode="revise",
)

tool = DeepResearchTool(config)
result = tool.run("Complex research topic")

# Access DeepThink results
for section, dt_result in result["deep_think_results"].items():
    print(f"Section {section}: confidence={dt_result['confidence']:.2f}")
```

### With Report Generator

```python
from deep_research_tool.thinking import SectionExtension

# Process sections before report generation
section_ext = SectionExtension(processor=processor)

for section_num, section_data in session.section_contents.items():
    processed = section_ext.process(section_data)
    session.section_contents[section_num] = processed
```

### Standalone Metrics Evaluation

```python
from deep_research_tool.thinking import DeepThinkMetrics

metrics = DeepThinkMetrics()

# Evaluate a reasoning step
evaluation = metrics.evaluate_reasoning_step(
    premises=["Premise 1", "Premise 2"],
    conclusion="Conclusion text",
    source_text="Original source"
)

print(f"Fidelity: {evaluation['source_fidelity']:.2f}")
print(f"Coherence: {evaluation['logical_coherence']:.2f}")
print(f"Confidence: {evaluation['confidence_score']:.2f}")
```

## Best Practices

1. **Choose appropriate level**: Use lower levels (0.2-0.4) for scientific/technical content, higher levels (0.6-0.8) for creative/exploratory content.

2. **Set consistency mode based on use case**:
   - `warn`: For exploratory research
   - `revise`: For reports requiring accuracy
   - `strict`: For critical applications

3. **Monitor metrics**: Track confidence scores to identify problematic sections.

4. **Cache embeddings**: The metrics class caches embeddings automatically, but clear the cache between unrelated processing tasks.

5. **Fallback handling**: When sentence-transformers is not available, metrics fall back to vocabulary-based methods.

## Troubleshooting

### Low Confidence Scores
- Check source text quality
- Verify facts are being extracted correctly
- Consider lowering `fidelity_threshold`

### High Deviation Scores
- Review intermediate reasoning steps
- Add more source material
- Lower `deep_think_level` for more conservative reasoning

### Performance Issues
- Reduce `reasoning_iterations`
- Use smaller embedding model
- Enable caching for repeated content
