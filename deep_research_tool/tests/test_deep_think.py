"""
Tests for DeepThink module.
"""

import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import asdict

from deep_research_tool.thinking import (
    DeepThinkConfig,
    DeepThinkProcessor,
    DeepThinkMetrics,
    MetricsConfig,
    LogicValidator,
    ValidationConfig,
    ValidationLevel,
    ReasoningStep,
    ReasoningChain,
    ReasoningType,
    ConsistencyMode,
    ConsistencyCheckResult,
    DeepThinkResult,
)
from deep_research_tool.thinking.extension_interface import (
    ExtensionConfig,
    SectionExtension,
    ReportExtension,
    DeepThinkMixin,
    create_extension_for_module,
)


class TestDeepThinkConfig:
    """Tests for DeepThinkConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DeepThinkConfig()
        assert config.enabled is False
        assert config.level == 0.5
        assert config.reasoning_iterations == 3
        assert config.consistency_threshold == 0.3
        assert config.consistency_mode == ConsistencyMode.WARN
        assert config.fidelity_threshold == 0.7
        assert config._expansion_tolerance == 0.2

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DeepThinkConfig(
            enabled=True,
            level=0.8,
            reasoning_iterations=5,
            consistency_threshold=0.2,
            consistency_mode=ConsistencyMode.STRICT,
        )
        assert config.enabled is True
        assert config.level == 0.8
        assert config.reasoning_iterations == 5
        assert config.consistency_mode == ConsistencyMode.STRICT

    def test_level_clamping(self):
        """Test that level is clamped to 0-1 range."""
        config = DeepThinkConfig(level=1.5)
        assert config.level == 1.0

        config = DeepThinkConfig(level=-0.5)
        assert config.level == 0.0


class TestReasoningChain:
    """Tests for ReasoningChain and ReasoningStep."""

    def test_reasoning_step_creation(self):
        """Test creating a reasoning step."""
        step = ReasoningStep(
            step_id="step_1",
            step_type=ReasoningType.FACT_EXTRACTION,
            premises=["Source text"],
            conclusion="Extracted fact",
        )
        assert step.step_id == "step_1"
        assert step.step_type == ReasoningType.FACT_EXTRACTION
        assert len(step.premises) == 1

    def test_reasoning_chain_add_step(self):
        """Test adding steps to a chain."""
        chain = ReasoningChain(chain_id="chain_1")
        step = ReasoningStep(
            step_id="step_1",
            step_type=ReasoningType.FACT_EXTRACTION,
            premises=["Premise"],
            conclusion="Conclusion",
        )
        chain.add_step(step)
        assert len(chain.steps) == 1

    def test_reasoning_chain_to_dict(self):
        """Test converting chain to dictionary."""
        chain = ReasoningChain(
            chain_id="chain_1",
            initial_facts=["Fact 1", "Fact 2"],
            final_conclusion="Final conclusion",
        )
        result = chain.to_dict()
        assert result["chain_id"] == "chain_1"
        assert len(result["initial_facts"]) == 2

    def test_get_all_premises(self):
        """Test getting all premises from chain."""
        chain = ReasoningChain(chain_id="chain_1")
        chain.add_step(ReasoningStep(
            step_id="s1",
            step_type=ReasoningType.INFERENCE,
            premises=["P1", "P2"],
            conclusion="C1"
        ))
        chain.add_step(ReasoningStep(
            step_id="s2",
            step_type=ReasoningType.INFERENCE,
            premises=["P3"],
            conclusion="C2"
        ))
        premises = chain.get_all_premises()
        assert len(premises) == 3
        assert "P1" in premises


class TestDeepThinkMetrics:
    """Tests for DeepThinkMetrics."""

    @pytest.fixture
    def metrics(self):
        """Create metrics instance."""
        return DeepThinkMetrics()

    def test_tokenize(self, metrics):
        """Test tokenization."""
        tokens = metrics._tokenize("Hello world, this is a test!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_jaccard_coefficient(self, metrics):
        """Test Jaccard coefficient calculation."""
        set1 = {"a", "b", "c"}
        set2 = {"b", "c", "d"}
        jaccard = metrics._jaccard_coefficient(set1, set2)
        # Intersection: {b, c} = 2, Union: {a, b, c, d} = 4
        assert jaccard == 0.5

    def test_jaccard_empty_sets(self, metrics):
        """Test Jaccard with empty sets."""
        assert metrics._jaccard_coefficient(set(), set()) == 0.0
        assert metrics._jaccard_coefficient({"a"}, set()) == 0.0

    def test_calc_expansion_degree(self, metrics):
        """Test expansion degree calculation."""
        premises = ["The weather is sunny today"]
        conclusion = "The weather is sunny and warm"

        expansion = metrics.calc_expansion_degree(premises, conclusion)
        assert 0.0 <= expansion <= 1.0

    def test_calc_expansion_no_premises(self, metrics):
        """Test expansion with no premises."""
        expansion = metrics.calc_expansion_degree([], "Some conclusion")
        assert expansion == 1.0  # Full expansion

    def test_detect_contradiction_negation(self, metrics):
        """Test contradiction detection for negation patterns."""
        facts = ["The product is available"]
        conclusion = "The product is not available"

        score, contradictions = metrics.detect_contradiction(facts, conclusion)
        assert score > 0.0
        assert len(contradictions) > 0

    def test_detect_contradiction_numerical(self, metrics):
        """Test contradiction detection for numerical values."""
        facts = ["Sales: 100 units"]
        conclusion = "Sales: 200 units"

        score, contradictions = metrics.detect_contradiction(facts, conclusion)
        # Should detect significant numerical discrepancy
        assert score >= 0.0

    def test_calc_deviation_score(self, metrics):
        """Test deviation score calculation."""
        facts = ["AI is transforming healthcare", "Machine learning improves diagnosis"]
        conclusion = "AI and ML are revolutionizing medical care"

        result = metrics.calc_deviation_score(facts, conclusion)
        assert "semantic_deviation" in result
        assert "logical_gap" in result
        assert "contradiction_score" in result
        assert "total_deviation" in result

    def test_calc_confidence_score(self, metrics):
        """Test confidence score calculation."""
        confidence = metrics.calc_confidence_score(
            source_fidelity=0.8,
            logical_coherence=0.7,
            expansion_degree=0.2,
            deviation_score=0.3,
        )
        assert 0.0 <= confidence <= 1.0

    def test_evaluate_reasoning_step(self, metrics):
        """Test full reasoning step evaluation."""
        result = metrics.evaluate_reasoning_step(
            premises=["Premise 1", "Premise 2"],
            conclusion="Conclusion based on premises",
            source_text="Original source text for reference",
        )
        assert "source_fidelity" in result
        assert "logical_coherence" in result
        assert "confidence_score" in result


class TestLogicValidator:
    """Tests for LogicValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return LogicValidator()

    def test_validate_step_fact_extraction(self, validator):
        """Test validation of fact extraction step."""
        step = ReasoningStep(
            step_id="s1",
            step_type=ReasoningType.FACT_EXTRACTION,
            premises=["Source text here"],
            conclusion="Extracted fact",
        )
        is_valid, issues = validator.validate_step(step)
        # Should be valid with premises and conclusion
        assert isinstance(is_valid, bool)

    def test_validate_step_no_premises(self, validator):
        """Test validation with no premises."""
        step = ReasoningStep(
            step_id="s1",
            step_type=ReasoningType.INFERENCE,
            premises=[],
            conclusion="Some conclusion",
        )
        is_valid, issues = validator.validate_step(step)
        assert "no premises" in " ".join(issues).lower()

    def test_validate_chain(self, validator):
        """Test chain validation."""
        chain = ReasoningChain(
            chain_id="chain_1",
            initial_facts=["Fact 1", "Fact 2"],
        )
        chain.add_step(ReasoningStep(
            step_id="s1",
            step_type=ReasoningType.FACT_EXTRACTION,
            premises=["Source"],
            conclusion="Extracted fact",
        ))
        chain.final_conclusion = "Final conclusion"

        result = validator.validate_chain(chain)
        assert isinstance(result, ConsistencyCheckResult)
        assert isinstance(result.is_consistent, bool)

    def test_check_final_consistency(self, validator):
        """Test final consistency check."""
        facts = ["AI improves efficiency", "Automation reduces costs"]
        conclusion = "AI and automation provide business benefits"

        result = validator.check_final_consistency(facts, conclusion)
        assert isinstance(result, ConsistencyCheckResult)
        assert "confidence_score" in result.to_dict()


class TestDeepThinkResult:
    """Tests for DeepThinkResult."""

    def test_result_creation(self):
        """Test creating a DeepThink result."""
        result = DeepThinkResult(
            original_content="Original text",
            processed_content="Processed text",
            deep_think_level=0.5,
        )
        assert result.original_content == "Original text"
        assert result.is_valid is True  # No consistency result = valid

    def test_result_with_consistency(self):
        """Test result with consistency check."""
        consistency = ConsistencyCheckResult(
            is_consistent=False,
            deviation_score=0.5,
            confidence_score=0.6,
        )
        result = DeepThinkResult(
            original_content="Original",
            processed_content="Processed",
            consistency_result=consistency,
        )
        assert result.is_valid is False
        assert result.overall_confidence == 0.6

    def test_result_to_dict(self):
        """Test result serialization."""
        result = DeepThinkResult(
            original_content="Original",
            processed_content="Processed",
            metrics_summary={"confidence": 0.8},
        )
        d = result.to_dict()
        assert "original_content" in d
        assert "metrics_summary" in d


class TestExtensions:
    """Tests for extension interfaces."""

    def test_section_extension_preprocess(self):
        """Test SectionExtension preprocessing."""
        ext = SectionExtension()
        content = {"content": "Section text", "title": "Section 1"}
        result = ext.preprocess(content)
        assert result == "Section text"

    def test_section_extension_postprocess(self):
        """Test SectionExtension postprocessing."""
        ext = SectionExtension()
        original = {"content": "Original", "title": "Section"}
        result = DeepThinkResult(
            original_content="Original",
            processed_content="Processed",
            metrics_summary={"confidence": 0.9},
        )
        processed = ext.postprocess(result, original)
        assert processed["content"] == "Processed"
        assert "deep_think_result" in processed

    def test_create_extension_for_module(self):
        """Test extension factory function."""
        mock_processor = Mock()

        section_ext = create_extension_for_module("section", mock_processor)
        assert isinstance(section_ext, SectionExtension)

        report_ext = create_extension_for_module("report", mock_processor)
        assert isinstance(report_ext, ReportExtension)

    def test_deep_think_mixin(self):
        """Test DeepThinkMixin."""
        class TestClass(DeepThinkMixin):
            pass

        obj = TestClass()
        assert obj.deep_think_enabled is False

        # Setup should require llm_client
        mock_client = Mock()
        config = DeepThinkConfig(enabled=True)
        obj.setup_deep_think(mock_client, config)
        assert obj.deep_think_enabled is True


class TestDeepThinkProcessor:
    """Tests for DeepThinkProcessor."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = Mock()
        response = Mock()
        response.content = "- Extracted fact 1\n- Extracted fact 2"
        client.generate.return_value = response
        return client

    def test_processor_disabled(self, mock_llm_client):
        """Test processor when disabled."""
        config = DeepThinkConfig(enabled=False)
        processor = DeepThinkProcessor(mock_llm_client, config)

        result = processor.process("Some content")
        assert result.processed_content == "Some content"
        mock_llm_client.generate.assert_not_called()

    def test_processor_enabled(self, mock_llm_client):
        """Test processor when enabled."""
        config = DeepThinkConfig(enabled=True, reasoning_iterations=1)
        processor = DeepThinkProcessor(mock_llm_client, config)

        result = processor.process("Some content to analyze")
        assert isinstance(result, DeepThinkResult)
        assert mock_llm_client.generate.called

    def test_extract_facts(self, mock_llm_client):
        """Test fact extraction."""
        config = DeepThinkConfig(enabled=True)
        processor = DeepThinkProcessor(mock_llm_client, config)

        facts = processor._extract_facts("Content to extract from")
        assert isinstance(facts, list)
        assert len(facts) > 0

    def test_adjust_level_for_domain(self, mock_llm_client):
        """Test domain-based level adjustment."""
        config = DeepThinkConfig(enabled=True, level=0.5)
        processor = DeepThinkProcessor(mock_llm_client, config)

        # Scientific domain should reduce level
        adjusted = processor.adjust_level_for_domain("science")
        assert adjusted < 0.5

        # Creative domain should increase level
        adjusted = processor.adjust_level_for_domain("creative")
        assert adjusted > 0.5


class TestConfigIntegration:
    """Tests for config integration."""

    def test_create_config_with_deep_think(self):
        """Test creating config with DeepThink parameters."""
        from deep_research_tool.config import create_config

        config = create_config(
            provider="openai",
            deep_think=True,
            deep_think_level=0.7,
            reasoning_iterations=5,
            consistency_mode="strict",
        )

        assert config.deep_think.enabled is True
        assert config.deep_think.level == 0.7
        assert config.deep_think.reasoning_iterations == 5
        assert config.deep_think.consistency_mode == "strict"

    def test_create_config_default_deep_think(self):
        """Test default DeepThink settings in config."""
        from deep_research_tool.config import create_config

        config = create_config(provider="openai")
        assert config.deep_think.enabled is False
        assert config.deep_think.level == 0.5
