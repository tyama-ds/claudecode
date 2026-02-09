"""
DeepThink processor for enhanced reasoning.

This module provides the main DeepThink processor that performs
iterative reasoning with quantitative evaluation and consistency checking.
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

from .reasoning_chain import (
    ReasoningStep,
    ReasoningChain,
    ReasoningType,
    ConsistencyMode,
    ConsistencyCheckResult,
    DeepThinkResult,
)
from .metrics import DeepThinkMetrics, MetricsConfig
from .logic_validator import LogicValidator, ValidationConfig, ValidationLevel


@dataclass
class DeepThinkConfig:
    """
    Configuration for DeepThink processing.

    Attributes:
        enabled: Whether DeepThink is enabled
        level: Thinking depth level (0.0-1.0)
        reasoning_iterations: Number of reasoning iterations
        consistency_threshold: Threshold for consistency check
        consistency_mode: How to handle consistency issues
        fidelity_threshold: Minimum source fidelity score
        _expansion_tolerance: Hidden parameter for expansion control
        _deviation_weights: Hidden weights for deviation calculation
    """
    enabled: bool = False
    level: float = 0.5
    reasoning_iterations: int = 3
    consistency_threshold: float = 0.3
    consistency_mode: ConsistencyMode = ConsistencyMode.WARN
    fidelity_threshold: float = 0.7
    # Hidden parameters (prefixed with _)
    _expansion_tolerance: float = 0.2
    _deviation_weights: tuple = (0.4, 0.4, 0.2)

    def __post_init__(self):
        """Validate configuration values."""
        self.level = max(0.0, min(1.0, self.level))
        self.reasoning_iterations = max(1, min(10, self.reasoning_iterations))
        self.consistency_threshold = max(0.0, min(1.0, self.consistency_threshold))
        self.fidelity_threshold = max(0.0, min(1.0, self.fidelity_threshold))
        self._expansion_tolerance = max(0.0, min(1.0, self._expansion_tolerance))


class DeepThinkProcessor:
    """
    Main processor for DeepThink functionality.

    Provides deep iterative reasoning with:
    - Structured reasoning chains (fact -> inference -> conclusion)
    - Quantitative metrics evaluation
    - Consistency checking against original facts
    - Support for multiple reasoning iterations
    """

    # Prompt templates for different reasoning steps
    FACT_EXTRACTION_PROMPT = """以下のテキストから客観的な事実を抽出してください。
意見や推測ではなく、確認可能な事実のみを箇条書きで列挙してください。

テキスト:
{source_text}

抽出した事実:"""

    INFERENCE_PROMPT = """以下の前提（事実）に基づいて、論理的な推論を行ってください。
推論は前提から導き出せる範囲に限定し、過度な飛躍は避けてください。
推論の深さレベル: {level} (0=保守的, 1=探索的)

前提:
{premises}

論理的推論:"""

    SYNTHESIS_PROMPT = """以下の情報を統合して、一貫した結論を導出してください。
各情報源からの事実と推論を考慮し、矛盾がないか確認してください。

情報:
{information}

統合された結論:"""

    CONCLUSION_PROMPT = """以下の分析に基づいて、最終的な結論を述べてください。
結論は事実に基づき、推論の根拠を明確にしてください。

分析内容:
{analysis}

最終結論:"""

    REVISION_PROMPT = """以下の結論が元の事実と矛盾している可能性があります。
矛盾点を確認し、事実に基づいて結論を修正してください。

元の事実:
{facts}

現在の結論:
{conclusion}

検出された問題:
{issues}

修正された結論:"""

    def __init__(
        self,
        llm_client,
        config: DeepThinkConfig = None,
        language: str = "ja"
    ):
        """
        Initialize the DeepThink processor.

        Args:
            llm_client: LLM client for text generation
            config: DeepThink configuration
            language: Output language
        """
        self.llm_client = llm_client
        self.config = config or DeepThinkConfig()
        self.language = language

        # Initialize metrics and validator
        metrics_config = MetricsConfig(
            fidelity_threshold=self.config.fidelity_threshold,
            expansion_tolerance=self.config._expansion_tolerance,
            deviation_weights=self.config._deviation_weights,
        )
        self.metrics = DeepThinkMetrics(metrics_config)

        validation_config = ValidationConfig(
            level=self._get_validation_level(),
            consistency_threshold=self.config.consistency_threshold,
            fidelity_threshold=self.config.fidelity_threshold,
        )
        self.validator = LogicValidator(validation_config, self.metrics)

        # Processing state
        self._current_chain: Optional[ReasoningChain] = None
        self._all_chains: List[ReasoningChain] = []

    def _get_validation_level(self) -> ValidationLevel:
        """Map consistency mode to validation level."""
        if self.config.consistency_mode == ConsistencyMode.STRICT:
            return ValidationLevel.STRICT
        elif self.config.consistency_mode == ConsistencyMode.REVISE:
            return ValidationLevel.STANDARD
        else:
            return ValidationLevel.RELAXED

    def process(
        self,
        content: str,
        source_texts: Dict[str, str] = None,
        progress_callback: Callable[[str, float], None] = None
    ) -> DeepThinkResult:
        """
        Process content with DeepThink reasoning.

        Args:
            content: Content to process
            source_texts: Mapping of source IDs to their texts
            progress_callback: Progress callback function

        Returns:
            DeepThinkResult with processed content and metrics
        """
        if not self.config.enabled:
            return DeepThinkResult(
                original_content=content,
                processed_content=content,
                deep_think_level=self.config.level
            )

        start_time = time.time()
        source_texts = source_texts or {}

        if progress_callback:
            progress_callback("DeepThink: Starting reasoning process...", 0)

        # Step 1: Extract facts from content
        initial_facts = self._extract_facts(content, progress_callback)

        # Step 2: Perform iterative reasoning
        reasoning_chains = []
        for iteration in range(self.config.reasoning_iterations):
            if progress_callback:
                progress = (iteration + 1) / self.config.reasoning_iterations * 70
                progress_callback(f"DeepThink: Reasoning iteration {iteration + 1}/{self.config.reasoning_iterations}", progress)

            chain = self._perform_reasoning_iteration(
                initial_facts=initial_facts,
                iteration=iteration,
                source_texts=source_texts
            )
            reasoning_chains.append(chain)

        # Step 3: Synthesize final conclusion
        if progress_callback:
            progress_callback("DeepThink: Synthesizing conclusions...", 80)

        final_conclusion = self._synthesize_conclusions(reasoning_chains)

        # Step 4: Consistency check
        if progress_callback:
            progress_callback("DeepThink: Running consistency check...", 90)

        consistency_result = self.validator.check_final_consistency(
            initial_facts=initial_facts,
            final_conclusion=final_conclusion,
            threshold=self.config.consistency_threshold
        )

        # Step 5: Handle inconsistency based on mode
        processed_content = final_conclusion
        if not consistency_result.is_consistent:
            if self.config.consistency_mode == ConsistencyMode.REVISE:
                processed_content = self._revise_conclusion(
                    initial_facts=initial_facts,
                    conclusion=final_conclusion,
                    issues=consistency_result.problematic_areas
                )
                # Re-check consistency after revision
                consistency_result = self.validator.check_final_consistency(
                    initial_facts=initial_facts,
                    final_conclusion=processed_content,
                    threshold=self.config.consistency_threshold
                )
            elif self.config.consistency_mode == ConsistencyMode.STRICT:
                # In strict mode, return original content if consistency fails
                processed_content = content

        # Compile metrics summary
        metrics_summary = self._compile_metrics_summary(reasoning_chains, consistency_result)

        processing_time = time.time() - start_time

        if progress_callback:
            progress_callback("DeepThink: Complete", 100)

        return DeepThinkResult(
            original_content=content,
            processed_content=processed_content,
            reasoning_chains=reasoning_chains,
            consistency_result=consistency_result,
            metrics_summary=metrics_summary,
            processing_time=processing_time,
            deep_think_level=self.config.level
        )

    def _extract_facts(
        self,
        content: str,
        progress_callback: Callable[[str, float], None] = None
    ) -> List[str]:
        """Extract facts from content using LLM."""
        if progress_callback:
            progress_callback("DeepThink: Extracting facts...", 10)

        prompt = self.FACT_EXTRACTION_PROMPT.format(source_text=content[:4000])
        response = self.llm_client.generate(prompt)

        # Parse bullet points from response
        facts = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('・') or line.startswith('*')):
                fact = line.lstrip('-・* ').strip()
                if fact:
                    facts.append(fact)

        # If no bullet points found, split by sentences
        if not facts:
            import re
            sentences = re.split(r'[。.！!？?]', response.content)
            facts = [s.strip() for s in sentences if s.strip()]

        return facts[:20]  # Limit to 20 facts

    def _perform_reasoning_iteration(
        self,
        initial_facts: List[str],
        iteration: int,
        source_texts: Dict[str, str] = None
    ) -> ReasoningChain:
        """Perform one iteration of reasoning."""
        chain_id = f"chain_{iteration}_{uuid.uuid4().hex[:8]}"
        chain = ReasoningChain(chain_id=chain_id, initial_facts=initial_facts.copy())

        # Step 1: Fact extraction step
        fact_step = ReasoningStep(
            step_id=f"{chain_id}_fact",
            step_type=ReasoningType.FACT_EXTRACTION,
            premises=initial_facts,
            conclusion="\n".join(initial_facts),
            source_references=list(source_texts.keys()) if source_texts else []
        )

        # Validate and add metrics
        self.validator.validate_step(fact_step)
        chain.add_step(fact_step)

        # Step 2: Inference step
        premises_text = "\n".join(f"- {f}" for f in initial_facts)
        inference_prompt = self.INFERENCE_PROMPT.format(
            premises=premises_text,
            level=self.config.level
        )
        inference_response = self.llm_client.generate(inference_prompt)

        inference_step = ReasoningStep(
            step_id=f"{chain_id}_inference",
            step_type=ReasoningType.INFERENCE,
            premises=initial_facts,
            conclusion=inference_response.content
        )
        self.validator.validate_step(inference_step)
        chain.add_step(inference_step)

        # Step 3: Conclusion step
        analysis_text = f"事実:\n{premises_text}\n\n推論:\n{inference_response.content}"
        conclusion_prompt = self.CONCLUSION_PROMPT.format(analysis=analysis_text)
        conclusion_response = self.llm_client.generate(conclusion_prompt)

        conclusion_step = ReasoningStep(
            step_id=f"{chain_id}_conclusion",
            step_type=ReasoningType.CONCLUSION,
            premises=[inference_response.content],
            conclusion=conclusion_response.content
        )
        self.validator.validate_step(conclusion_step)
        chain.add_step(conclusion_step)

        chain.final_conclusion = conclusion_response.content

        # Validate entire chain
        self.validator.validate_chain(chain, source_texts or {})

        return chain

    def _synthesize_conclusions(self, chains: List[ReasoningChain]) -> str:
        """Synthesize conclusions from multiple reasoning chains."""
        if not chains:
            return ""

        if len(chains) == 1:
            return chains[0].final_conclusion

        # Combine conclusions from all chains
        conclusions = [chain.final_conclusion for chain in chains if chain.final_conclusion]

        if not conclusions:
            return ""

        # Use synthesis prompt to combine
        information = "\n\n".join(f"[分析{i+1}]\n{c}" for i, c in enumerate(conclusions))
        synthesis_prompt = self.SYNTHESIS_PROMPT.format(information=information)
        response = self.llm_client.generate(synthesis_prompt)

        return response.content

    def _revise_conclusion(
        self,
        initial_facts: List[str],
        conclusion: str,
        issues: List[str]
    ) -> str:
        """Revise conclusion to address consistency issues."""
        facts_text = "\n".join(f"- {f}" for f in initial_facts)
        issues_text = "\n".join(f"- {i}" for i in issues)

        revision_prompt = self.REVISION_PROMPT.format(
            facts=facts_text,
            conclusion=conclusion,
            issues=issues_text
        )
        response = self.llm_client.generate(revision_prompt)

        return response.content

    def _compile_metrics_summary(
        self,
        chains: List[ReasoningChain],
        consistency_result: ConsistencyCheckResult
    ) -> Dict[str, float]:
        """Compile summary of all metrics."""
        if not chains:
            return {}

        # Collect metrics from all chains
        all_fidelity = []
        all_coherence = []
        all_expansion = []

        for chain in chains:
            for step in chain.steps:
                if "source_fidelity" in step.metrics:
                    all_fidelity.append(step.metrics["source_fidelity"])
                if "logical_coherence" in step.metrics:
                    all_coherence.append(step.metrics["logical_coherence"])
                if "expansion_degree" in step.metrics:
                    all_expansion.append(step.metrics["expansion_degree"])

        return {
            "avg_source_fidelity": sum(all_fidelity) / len(all_fidelity) if all_fidelity else 0.0,
            "avg_logical_coherence": sum(all_coherence) / len(all_coherence) if all_coherence else 0.0,
            "avg_expansion_degree": sum(all_expansion) / len(all_expansion) if all_expansion else 0.0,
            "semantic_deviation": consistency_result.semantic_deviation,
            "logical_gap": consistency_result.logical_gap,
            "contradiction_score": consistency_result.contradiction_score,
            "total_deviation": consistency_result.deviation_score,
            "confidence_score": consistency_result.confidence_score,
            "is_consistent": float(consistency_result.is_consistent),
            "num_chains": len(chains),
            "num_steps": sum(len(chain.steps) for chain in chains),
        }

    def process_section(
        self,
        section_content: str,
        section_title: str = "",
        sources: List[Dict[str, str]] = None
    ) -> DeepThinkResult:
        """
        Process a single section with DeepThink.

        This is a convenience method for processing individual sections.

        Args:
            section_content: Content of the section
            section_title: Title of the section
            sources: List of source dictionaries with 'id' and 'text'

        Returns:
            DeepThinkResult for the section
        """
        source_texts = {}
        if sources:
            for src in sources:
                if "id" in src and "text" in src:
                    source_texts[src["id"]] = src["text"]

        return self.process(
            content=section_content,
            source_texts=source_texts
        )

    def adjust_level_for_domain(
        self,
        domain: str,
        base_level: float = None
    ) -> float:
        """
        Adjust thinking level based on domain characteristics.

        Scientific/technical domains may need more conservative reasoning,
        while creative domains can allow more exploration.

        Args:
            domain: Domain identifier (e.g., "science", "business", "creative")
            base_level: Base level to adjust from

        Returns:
            Adjusted level value
        """
        base_level = base_level or self.config.level

        # Domain-specific adjustments
        conservative_domains = ["science", "medical", "legal", "financial", "技術", "医学", "法律", "金融"]
        exploratory_domains = ["creative", "marketing", "brainstorm", "創作", "マーケティング"]

        domain_lower = domain.lower()

        for d in conservative_domains:
            if d in domain_lower:
                return max(0.0, base_level - 0.2)

        for d in exploratory_domains:
            if d in domain_lower:
                return min(1.0, base_level + 0.2)

        return base_level
