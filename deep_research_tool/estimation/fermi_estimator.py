"""
Main Fermi Estimator for Deep Research Tool.

Orchestrates the complete Fermi estimation workflow:
decomposition -> assumption resolution -> calculation -> validation.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .assumptions import Assumption, AssumptionManager, AssumptionSource
from .calculator import Calculator, ScenarioResult, SensitivityAnalysis
from .decomposer import Decomposer, DecompositionTree, TreeNode
from .validator import ValidationResult, Validator

logger = logging.getLogger(__name__)


@dataclass
class FermiEstimationConfig:
    """Configuration for Fermi estimation."""
    enabled: bool = False

    # Target metrics
    target_metrics: List[str] = field(default_factory=list)
    auto_detect_targets: bool = True

    # Decomposition settings
    max_tree_depth: int = 4
    max_leaf_nodes: int = 10

    # Calculation settings
    monte_carlo_iterations: int = 1000

    # Validation settings
    validate_with_llm: bool = True
    min_confidence_threshold: float = 0.3

    # Output settings
    write_to_data_store: bool = True
    include_sensitivity: bool = True

    # Sub-decomposition settings
    enable_sub_decomposition: bool = True
    sub_decomposition_confidence_threshold: float = 0.65
    sub_decomposition_max_iterations: int = 3
    sub_decomposition_min_sensitivity_pct: float = 10.0

    def __post_init__(self):
        self.max_tree_depth = max(1, min(6, self.max_tree_depth))
        self.max_leaf_nodes = max(2, min(20, self.max_leaf_nodes))
        self.min_confidence_threshold = max(0.0, min(1.0, self.min_confidence_threshold))
        self.sub_decomposition_confidence_threshold = max(
            0.0, min(1.0, self.sub_decomposition_confidence_threshold)
        )
        self.sub_decomposition_max_iterations = max(
            0, min(10, self.sub_decomposition_max_iterations)
        )
        self.sub_decomposition_min_sensitivity_pct = max(
            0.0, min(100.0, self.sub_decomposition_min_sensitivity_pct)
        )


@dataclass
class FermiEstimationResult:
    """Complete result of a Fermi estimation."""

    estimation_id: str = ""
    target_metric: str = ""

    base_estimate: float = 0.0
    low_estimate: float = 0.0
    high_estimate: float = 0.0
    unit: str = ""

    decomposition_tree: Optional[DecompositionTree] = None
    assumptions: List[Assumption] = field(default_factory=list)
    scenarios: Dict[str, ScenarioResult] = field(default_factory=dict)
    sensitivity: Optional[SensitivityAnalysis] = None
    monte_carlo: Optional[Dict[str, Any]] = None
    validation: Optional[ValidationResult] = None

    overall_confidence: float = 0.0
    evidence_backed_ratio: float = 0.0
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimation_id": self.estimation_id,
            "target_metric": self.target_metric,
            "base_estimate": self.base_estimate,
            "low_estimate": self.low_estimate,
            "high_estimate": self.high_estimate,
            "unit": self.unit,
            "decomposition_tree": self.decomposition_tree.to_dict() if self.decomposition_tree else None,
            "assumptions": [a.to_dict() for a in self.assumptions],
            "scenarios": {k: v.to_dict() for k, v in self.scenarios.items()},
            "sensitivity": self.sensitivity.to_dict() if self.sensitivity else None,
            "monte_carlo": self.monte_carlo,
            "validation": self.validation.to_dict() if self.validation else None,
            "overall_confidence": self.overall_confidence,
            "evidence_backed_ratio": self.evidence_backed_ratio,
            "processing_time": self.processing_time,
        }

    def to_summary(self, language: str = "ja") -> str:
        """Generate a human-readable summary for report inclusion."""
        if language == "ja":
            return self._to_summary_ja()
        return self._to_summary_en()

    def _to_summary_ja(self) -> str:
        lines = []
        lines.append(f"### フェルミ推定: {self.target_metric}")
        lines.append("")

        # Main result
        lines.append("#### 推定結果")
        lines.append("")
        lines.append(f"| シナリオ | 推定値 |")
        lines.append(f"|----------|--------|")
        lines.append(f"| 悲観 | {self._format_number(self.low_estimate)} {self.unit} |")
        lines.append(f"| **ベース** | **{self._format_number(self.base_estimate)} {self.unit}** |")
        lines.append(f"| 楽観 | {self._format_number(self.high_estimate)} {self.unit} |")
        lines.append("")

        # Confidence
        lines.append(f"**信頼度:** {self.overall_confidence:.0%}  ")
        lines.append(f"**エビデンス裏付け率:** {self.evidence_backed_ratio:.0%}")
        lines.append("")

        # Decomposition
        if self.decomposition_tree and self.decomposition_tree.root:
            lines.append("#### 分解構造")
            lines.append("")
            lines.append(self.decomposition_tree.decomposition_reasoning)
            lines.append("")
            self._render_tree(self.decomposition_tree.root, lines, indent=0)
            lines.append("")

        # Key assumptions
        if self.assumptions:
            lines.append("#### 主要な前提条件")
            lines.append("")
            lines.append("| パラメータ | 値 | 出典 | 信頼度 |")
            lines.append("|------------|-----|------|--------|")
            for a in sorted(self.assumptions, key=lambda x: x.sensitivity_rank or 999):
                source_label = {
                    AssumptionSource.EVIDENCE_DIRECT: "エビデンス",
                    AssumptionSource.EVIDENCE_DERIVED: "派生データ",
                    AssumptionSource.LLM_ESTIMATE: "LLM推定",
                    AssumptionSource.USER_INPUT: "ユーザー入力",
                    AssumptionSource.COMMON_KNOWLEDGE: "一般常識",
                    AssumptionSource.DEFAULT: "デフォルト",
                }.get(a.source, str(a.source))
                lines.append(
                    f"| {a.parameter_name} | {self._format_number(a.value)} {a.unit} "
                    f"| {source_label} | {a.confidence:.0%} |"
                )
            lines.append("")

        # Sensitivity
        if self.sensitivity and self.sensitivity.items:
            lines.append("#### 感度分析")
            lines.append("")
            lines.append(f"最も影響の大きいパラメータ: **{self.sensitivity.most_sensitive_parameter}**")
            lines.append("")
            lines.append("| パラメータ | 悲観時の結果 | 楽観時の結果 | 影響度 |")
            lines.append("|------------|-------------|-------------|--------|")
            for item in self.sensitivity.items[:5]:
                lines.append(
                    f"| {item.node_name} | {self._format_number(item.result_at_low)} "
                    f"| {self._format_number(item.result_at_high)} | {item.sensitivity_pct:.1f}% |"
                )
            lines.append("")

        # Validation
        if self.validation:
            if self.validation.issues:
                lines.append("#### 検証結果")
                lines.append("")
                for issue in self.validation.issues:
                    icon = {"error": "x", "warning": "!", "info": "i"}.get(issue.severity, "-")
                    lines.append(f"- [{icon}] {issue.description}")
                    if issue.suggestion:
                        lines.append(f"  - 提案: {issue.suggestion}")
                lines.append("")

        return "\n".join(lines)

    def _to_summary_en(self) -> str:
        lines = []
        lines.append(f"### Fermi Estimation: {self.target_metric}")
        lines.append("")

        lines.append("#### Estimation Result")
        lines.append("")
        lines.append(f"| Scenario | Estimate |")
        lines.append(f"|----------|----------|")
        lines.append(f"| Pessimistic | {self._format_number(self.low_estimate)} {self.unit} |")
        lines.append(f"| **Base** | **{self._format_number(self.base_estimate)} {self.unit}** |")
        lines.append(f"| Optimistic | {self._format_number(self.high_estimate)} {self.unit} |")
        lines.append("")
        lines.append(f"**Confidence:** {self.overall_confidence:.0%}  ")
        lines.append(f"**Evidence-backed ratio:** {self.evidence_backed_ratio:.0%}")
        lines.append("")

        if self.assumptions:
            lines.append("#### Key Assumptions")
            lines.append("")
            lines.append("| Parameter | Value | Source | Confidence |")
            lines.append("|-----------|-------|--------|------------|")
            for a in sorted(self.assumptions, key=lambda x: x.sensitivity_rank or 999):
                lines.append(
                    f"| {a.parameter_name} | {self._format_number(a.value)} {a.unit} "
                    f"| {a.source.value} | {a.confidence:.0%} |"
                )
            lines.append("")

        return "\n".join(lines)

    def _render_tree(self, node, lines: list, indent: int) -> None:
        """Render tree structure as indented text."""
        prefix = "  " * indent + ("- " if indent > 0 else "")
        value_str = ""
        if node.value is not None:
            value_str = f" = {self._format_number(node.value)} {node.unit}"
            if node.is_evidence_backed:
                value_str += " [Evidence]"
        lines.append(f"{prefix}**{node.name}**{value_str}")
        for child in node.children:
            self._render_tree(child, lines, indent + 1)

    @staticmethod
    def _format_number(value: float) -> str:
        """Format number with appropriate precision."""
        if value == 0:
            return "0"
        abs_val = abs(value)
        if abs_val >= 1_000_000_000_000:
            return f"{value/1_000_000_000_000:,.1f}兆"
        if abs_val >= 100_000_000:
            return f"{value/100_000_000:,.1f}億"
        if abs_val >= 10_000:
            return f"{value/10_000:,.1f}万"
        if abs_val >= 1:
            return f"{value:,.2f}"
        return f"{value:.4g}"


# LLM prompt for auto-detecting target metrics
TARGET_DETECTION_PROMPT_JA = """以下のリサーチトピックと収集されたエビデンスから、
フェルミ推定で推定すべき重要な定量指標を抽出してください。

## リサーチトピック
{research_topic}

## 収集されたデータの概要
{data_summary}

## 指示
- 直接的にはデータが不足しているが、推定できれば価値のある指標を特定してください
- 市場規模、需要量、人口、コスト等の定量指標に限定してください
- 最大3個まで

## 出力形式 (JSON)
{{
    "target_metrics": [
        {{
            "metric": "指標名（具体的に）",
            "unit": "単位",
            "reason": "なぜこの推定が有用か"
        }}
    ]
}}

JSONのみを出力:"""

TARGET_DETECTION_PROMPT_EN = """From the following research topic and collected evidence,
extract important quantitative metrics that should be estimated via Fermi estimation.

## Research Topic
{research_topic}

## Data Summary
{data_summary}

## Instructions
- Identify metrics where data is insufficient but estimation would be valuable
- Limit to quantitative metrics (market size, demand, population, cost, etc.)
- Maximum 3 metrics

## Output (JSON only)
{{
    "target_metrics": [
        {{
            "metric": "Specific metric name",
            "unit": "unit",
            "reason": "Why this estimation is useful"
        }}
    ]
}}

Output only JSON:"""


class FermiEstimator:
    """
    Main Fermi estimation orchestrator.

    Workflow:
    1. (Optional) Auto-detect target metrics from research data
    2. Decompose each target into a multiplication tree
    3. Resolve leaf node values from evidence or LLM estimation
    4. Calculate three scenarios (base, pessimistic, optimistic)
    5. Run sensitivity analysis
    6. Validate results
    7. Write back to NumericalDataStore
    """

    def __init__(
        self,
        llm_client,
        config: FermiEstimationConfig = None,
        language: str = "ja",
    ):
        self.llm_client = llm_client
        self.config = config or FermiEstimationConfig()
        self.language = language
        self._decomposer = Decomposer(llm_client, language)
        self._calculator = Calculator(
            monte_carlo_iterations=self.config.monte_carlo_iterations,
        )

    def estimate(
        self,
        target_metric: str,
        data_store,
        evidence_locker=None,
        context: str = "",
        progress_callback: Callable[[str, float], None] = None,
    ) -> FermiEstimationResult:
        """Perform Fermi estimation for a single target metric."""
        start_time = time.time()

        if progress_callback:
            progress_callback("Fermi Estimation: Starting...", 0)

        # Step 1: Summarize available data
        available_summary = self._summarize_available_data(data_store)

        # Step 2: Decompose
        if progress_callback:
            progress_callback("Fermi Estimation: Decomposing problem...", 10)
        tree = self._decomposer.decompose(
            target_metric=target_metric,
            available_data_summary=available_summary,
            context=context,
        )
        logger.info(
            f"Decomposition complete: {len(tree.get_all_leaves())} leaf nodes"
        )

        # Step 3: Resolve assumptions
        if progress_callback:
            progress_callback("Fermi Estimation: Resolving assumptions...", 30)
        assumption_mgr = AssumptionManager(
            llm_client=self.llm_client,
            data_store=data_store,
            language=self.language,
        )
        assumptions = self._resolve_all_leaves(tree, assumption_mgr, context)

        # Step 4: Calculate scenarios
        if progress_callback:
            progress_callback("Fermi Estimation: Calculating scenarios...", 60)
        scenarios = self._calculator.evaluate_tree(tree)

        base = scenarios["base"].final_value
        low = scenarios["pessimistic"].final_value
        high = scenarios["optimistic"].final_value
        unit = scenarios["base"].unit

        # Step 5: Sensitivity analysis
        sensitivity = None
        if self.config.include_sensitivity:
            if progress_callback:
                progress_callback("Fermi Estimation: Sensitivity analysis...", 70)
            sensitivity = self._calculator.sensitivity_analysis(tree)
            # Update assumption sensitivity ranks
            if sensitivity and sensitivity.items:
                for item in sensitivity.items:
                    for a in assumptions:
                        if a.node_id == item.node_id:
                            a.sensitivity_rank = getattr(item, 'sensitivity_rank', 0)

        # Step 5b: Sub-decomposition of low-confidence, high-sensitivity leaves
        if (
            self.config.enable_sub_decomposition
            and sensitivity
            and self.config.sub_decomposition_max_iterations > 0
        ):
            sub_decomposition_applied = False
            for iteration in range(self.config.sub_decomposition_max_iterations):
                candidates = self._identify_sub_decomposition_candidates(
                    tree, sensitivity
                )
                if not candidates:
                    break

                if progress_callback:
                    progress_callback(
                        f"Fermi Estimation: Sub-decomposing "
                        f"'{candidates[0].name}'...",
                        72 + iteration * 2,
                    )

                success = self._apply_sub_decomposition(
                    tree=tree,
                    target_node=candidates[0],
                    data_store=data_store,
                    context=context,
                )

                if success:
                    sub_decomposition_applied = True
                    scenarios = self._calculator.evaluate_tree(tree)
                    base = scenarios["base"].final_value
                    low = scenarios["pessimistic"].final_value
                    high = scenarios["optimistic"].final_value
                    unit = scenarios["base"].unit
                    sensitivity = self._calculator.sensitivity_analysis(tree)
                else:
                    break

            if sub_decomposition_applied:
                # Rebuild assumption list from current leaves
                assumptions = []
                for leaf in tree.get_all_leaves():
                    source = (
                        AssumptionSource.EVIDENCE_DIRECT
                        if leaf.is_evidence_backed
                        else AssumptionSource.LLM_ESTIMATE
                    )
                    assumptions.append(Assumption(
                        node_id=leaf.node_id,
                        parameter_name=leaf.name,
                        value=leaf.value or 0.0,
                        value_low=leaf.value_low or 0.0,
                        value_high=leaf.value_high or 0.0,
                        unit=leaf.unit,
                        source=source,
                        evidence_data_id=leaf.evidence_data_id,
                        reasoning=leaf.estimation_reasoning,
                        confidence=leaf.confidence,
                    ))
                # Re-run sensitivity with updated assumptions
                if self.config.include_sensitivity and sensitivity:
                    for item in sensitivity.items:
                        for a in assumptions:
                            if a.node_id == item.node_id:
                                a.sensitivity_rank = getattr(
                                    item, 'sensitivity_rank', 0
                                )

        # Step 6: Monte Carlo
        monte_carlo = None
        if self.config.monte_carlo_iterations > 0:
            if progress_callback:
                progress_callback("Fermi Estimation: Monte Carlo simulation...", 80)
            # Save original values before Monte Carlo (it modifies leaf values)
            leaf_originals = {}
            for leaf in tree.get_all_leaves():
                leaf_originals[leaf.node_id] = (leaf.value, leaf.value_low, leaf.value_high)

            monte_carlo = self._calculator.monte_carlo_simulation(tree)

            # Restore original values
            for leaf in tree.get_all_leaves():
                if leaf.node_id in leaf_originals:
                    leaf.value, leaf.value_low, leaf.value_high = leaf_originals[leaf.node_id]

        # Step 7: Validation
        if progress_callback:
            progress_callback("Fermi Estimation: Validating...", 90)
        validator = Validator(
            llm_client=self.llm_client if self.config.validate_with_llm else None,
            data_store=data_store,
            evidence_locker=evidence_locker,
            language=self.language,
        )
        validation = validator.validate(
            target_metric=target_metric,
            base_value=base,
            low_value=low,
            high_value=high,
            unit=unit,
            assumptions=assumptions,
        )

        # Step 8: Write to data store
        if self.config.write_to_data_store and data_store:
            self._write_to_data_store(
                data_store=data_store,
                target_metric=target_metric,
                base=base, unit=unit,
                tree=tree,
            )

        processing_time = time.time() - start_time
        evidence_backed = assumption_mgr.get_evidence_backed_count()

        if progress_callback:
            progress_callback("Fermi Estimation: Complete", 100)

        return FermiEstimationResult(
            estimation_id=tree.tree_id,
            target_metric=target_metric,
            base_estimate=base,
            low_estimate=low,
            high_estimate=high,
            unit=unit,
            decomposition_tree=tree,
            assumptions=assumptions,
            scenarios=scenarios,
            sensitivity=sensitivity,
            monte_carlo=monte_carlo,
            validation=validation,
            overall_confidence=self._compute_overall_confidence(assumptions, validation),
            evidence_backed_ratio=(
                evidence_backed / len(assumptions) if assumptions else 0.0
            ),
            processing_time=processing_time,
        )

    def estimate_multiple(
        self,
        target_metrics: List[str],
        data_store,
        evidence_locker=None,
        context: str = "",
        progress_callback: Callable[[str, float], None] = None,
    ) -> List[FermiEstimationResult]:
        """Perform Fermi estimation for multiple target metrics."""
        results = []
        total = len(target_metrics)
        for i, metric in enumerate(target_metrics):
            def sub_callback(msg, pct):
                if progress_callback:
                    overall = (i / total + pct / 100 / total) * 100
                    progress_callback(msg, overall)

            result = self.estimate(
                target_metric=metric,
                data_store=data_store,
                evidence_locker=evidence_locker,
                context=context,
                progress_callback=sub_callback,
            )
            results.append(result)
        return results

    def detect_target_metrics(
        self,
        research_topic: str,
        data_store,
    ) -> List[Dict[str, str]]:
        """Auto-detect target metrics using LLM."""
        data_summary = self._summarize_available_data(data_store)
        template = (
            TARGET_DETECTION_PROMPT_JA
            if self.language == "ja"
            else TARGET_DETECTION_PROMPT_EN
        )
        prompt = template.format(
            research_topic=research_topic,
            data_summary=data_summary,
        )

        try:
            response = self.llm_client.generate(prompt)
            if not response or not response.content:
                return []

            data = self._parse_json(response.content)
            if not data:
                return []

            return data.get("target_metrics", [])

        except Exception as e:
            logger.error(f"Target metric detection failed: {e}")
            return []

    def _summarize_available_data(self, data_store) -> str:
        """Create text summary of available data."""
        if not data_store or not data_store.data_points:
            return "なし" if self.language == "ja" else "None"

        lines = []
        seen = set()
        for dp in sorted(
            data_store.data_points,
            key=lambda x: x.combined_confidence or 0,
            reverse=True,
        )[:20]:
            key = f"{dp.metric_name}_{dp.subject}_{dp.year}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"- {dp.metric_name} ({dp.subject}): "
                f"{dp.raw_text} [{dp.date_context}]"
            )

        return "\n".join(lines) if lines else (
            "なし" if self.language == "ja" else "None"
        )

    def _resolve_all_leaves(
        self,
        tree: DecompositionTree,
        assumption_mgr: AssumptionManager,
        context: str,
    ) -> List[Assumption]:
        """Resolve values for all leaf nodes."""
        leaves = tree.get_all_leaves()
        for leaf in leaves:
            assumption_mgr.resolve_leaf_node(leaf, context)
        return assumption_mgr.get_all_assumptions()

    def _identify_sub_decomposition_candidates(
        self,
        tree: DecompositionTree,
        sensitivity: SensitivityAnalysis,
    ) -> List[TreeNode]:
        """
        Identify leaf nodes that should be sub-decomposed.

        Criteria: low confidence AND high sensitivity impact.
        Returns nodes sorted by priority (highest impact first).
        """
        threshold = self.config.sub_decomposition_confidence_threshold
        min_sensitivity = self.config.sub_decomposition_min_sensitivity_pct
        max_depth = self.config.max_tree_depth

        sensitivity_map = {
            item.node_id: item.sensitivity_pct
            for item in (sensitivity.items if sensitivity else [])
        }

        candidates = []
        for leaf in tree.get_all_leaves():
            if leaf.confidence >= threshold:
                continue
            sens_pct = sensitivity_map.get(leaf.node_id, 0.0)
            if sens_pct < min_sensitivity:
                continue
            node_depth = tree.get_node_depth(leaf.node_id)
            if node_depth >= max_depth - 1:
                continue
            candidates.append((leaf, sens_pct))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in candidates]

    def _apply_sub_decomposition(
        self,
        tree: DecompositionTree,
        target_node: TreeNode,
        data_store,
        context: str,
    ) -> bool:
        """
        Sub-decompose a single leaf node and re-resolve its new children.

        Modifies the tree in-place. Returns True if the sub-decomposition
        improved confidence (or at least produced children).
        """
        available_summary = self._summarize_available_data(data_store)
        parent_context = f"{tree.target_metric}: {tree.decomposition_reasoning}"

        node_depth = tree.get_node_depth(target_node.node_id)
        max_sub_depth = min(self.config.max_tree_depth - node_depth, 2)

        # Save original state for rollback
        original_value = target_node.value
        original_value_low = target_node.value_low
        original_value_high = target_node.value_high
        original_confidence = target_node.confidence
        original_is_leaf = target_node.is_leaf
        original_children = target_node.children
        original_operation = target_node.operation
        original_is_evidence_backed = target_node.is_evidence_backed

        new_node = self._decomposer.sub_decompose(
            node=target_node,
            available_data_summary=available_summary,
            parent_context=parent_context,
            max_sub_depth=max_sub_depth,
        )

        if new_node is None:
            return False

        # Graft: replace the leaf's properties with the sub-tree's properties
        target_node.children = new_node.children
        target_node.operation = new_node.operation
        target_node.is_leaf = False
        target_node.description = new_node.description or target_node.description

        # Resolve new leaves
        assumption_mgr = AssumptionManager(
            llm_client=self.llm_client,
            data_store=data_store,
            language=self.language,
        )
        new_leaves = tree.get_all_leaves()
        sub_leaves = [
            leaf for leaf in new_leaves
            if leaf.value is None or leaf.confidence == 0.5  # unresolved leaves
        ]
        # If no sub_leaves detected via value check, resolve all leaves under the target
        if not sub_leaves:
            sub_leaves = []
            self._collect_leaves(target_node, sub_leaves)

        for leaf in sub_leaves:
            assumption_mgr.resolve_leaf_node(leaf, context)

        new_assumptions = assumption_mgr.get_all_assumptions()

        # Check if sub-decomposition improved confidence
        if new_assumptions:
            avg_new_confidence = (
                sum(a.confidence for a in new_assumptions) / len(new_assumptions)
            )
            if avg_new_confidence <= original_confidence:
                logger.info(
                    f"Sub-decomposition of '{target_node.name}' did not improve "
                    f"confidence ({avg_new_confidence:.2f} <= "
                    f"{original_confidence:.2f}), rolling back"
                )
                target_node.children = original_children
                target_node.operation = original_operation
                target_node.is_leaf = original_is_leaf
                target_node.value = original_value
                target_node.value_low = original_value_low
                target_node.value_high = original_value_high
                target_node.confidence = original_confidence
                target_node.is_evidence_backed = original_is_evidence_backed
                return False

            logger.info(
                f"Sub-decomposed '{target_node.name}': "
                f"{len(sub_leaves)} new leaves, avg confidence {avg_new_confidence:.2f}"
            )
        return True

    @staticmethod
    def _collect_leaves(node: TreeNode, result: List[TreeNode]) -> None:
        """Collect all leaf nodes under a node."""
        if node.is_leaf:
            result.append(node)
        else:
            for child in node.children:
                FermiEstimator._collect_leaves(child, result)

    def _write_to_data_store(
        self,
        data_store,
        target_metric: str,
        base: float,
        unit: str,
        tree: DecompositionTree,
    ) -> None:
        """Write estimation results as derived NumericalDataPoints."""
        try:
            from ..evidence.numerical_extractor import NumericalDataPoint, DataType, MetricCategory

            leaf_ids = [leaf.evidence_data_id for leaf in tree.get_all_leaves() if leaf.evidence_data_id]

            dp = NumericalDataPoint(
                value=base,
                normalized_value=base,
                raw_text=f"Fermi estimation: {target_metric}",
                unit=unit,
                metric_name=target_metric,
                subject=target_metric,
                data_type=DataType.MEASUREMENT,
                category=MetricCategory.OTHER,
                extraction_confidence=0.6,
                source_reliability=0.5,
                is_derived=True,
                derived_from=leaf_ids,
            )
            data_store.add(dp)
            logger.info(f"Wrote Fermi estimation result to data store: {target_metric}")

        except Exception as e:
            logger.error(f"Failed to write to data store: {e}")

    @staticmethod
    def _compute_overall_confidence(
        assumptions: List[Assumption],
        validation: ValidationResult,
    ) -> float:
        """Compute overall confidence."""
        if not assumptions:
            return 0.0

        # Average assumption confidence
        avg_conf = sum(a.confidence for a in assumptions) / len(assumptions)

        # Validation adjustment
        if validation:
            val_conf = validation.overall_confidence
            # Weighted average: 60% assumptions, 40% validation
            return avg_conf * 0.6 + val_conf * 0.4

        return avg_conf

    @staticmethod
    def _parse_json(content: str) -> Optional[Dict]:
        """Parse JSON from LLM response."""
        content = content.strip()
        if "```" in content:
            parts = content.split("```")
            for part in parts[1:]:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
                if cleaned.startswith("{"):
                    content = cleaned
                    break
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(content[start:end])
                except json.JSONDecodeError:
                    return None
        return None
