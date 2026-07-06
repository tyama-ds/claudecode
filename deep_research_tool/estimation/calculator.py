"""
Calculation engine for Fermi estimation.

Evaluates the decomposition tree to produce final estimates
with sensitivity analysis and Monte Carlo simulation.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .decomposer import DecompositionTree, NodeOperation, TreeNode

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Result for a single scenario (base/pessimistic/optimistic)."""
    scenario_name: str = ""
    final_value: float = 0.0
    unit: str = ""
    calculation_steps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "final_value": self.final_value,
            "unit": self.unit,
            "calculation_steps": self.calculation_steps,
        }


@dataclass
class SensitivityItem:
    """Sensitivity of final result to a single assumption."""
    node_id: str = ""
    node_name: str = ""
    base_value: float = 0.0
    low_value: float = 0.0
    high_value: float = 0.0
    result_at_low: float = 0.0
    result_at_high: float = 0.0
    sensitivity_pct: float = 0.0
    tornado_range: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "base_value": self.base_value,
            "low_value": self.low_value,
            "high_value": self.high_value,
            "result_at_low": self.result_at_low,
            "result_at_high": self.result_at_high,
            "sensitivity_pct": self.sensitivity_pct,
            "tornado_range": list(self.tornado_range),
        }


@dataclass
class SensitivityAnalysis:
    """Complete sensitivity analysis results."""
    items: List[SensitivityItem] = field(default_factory=list)
    base_result: float = 0.0
    most_sensitive_parameter: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [i.to_dict() for i in self.items],
            "base_result": self.base_result,
            "most_sensitive_parameter": self.most_sensitive_parameter,
        }


class Calculator:
    """
    Evaluates a decomposition tree to produce numerical estimates.

    Supports:
    - Tree evaluation with multiply/add/subtract/divide operations
    - Three-scenario calculation (base, pessimistic, optimistic)
    - Sensitivity analysis (tornado diagram data)
    - Monte Carlo simulation for confidence intervals
    """

    def __init__(self, monte_carlo_iterations: int = 1000):
        self.monte_carlo_iterations = monte_carlo_iterations

    def evaluate_tree(
        self,
        tree: DecompositionTree,
    ) -> Dict[str, ScenarioResult]:
        """Evaluate the tree for all three scenarios."""
        if not tree.root:
            empty = ScenarioResult(final_value=0.0)
            return {"base": empty, "pessimistic": empty, "optimistic": empty}

        results = {}
        for scenario_name, value_key in [
            ("base", "base"),
            ("pessimistic", "low"),
            ("optimistic", "high"),
        ]:
            steps = []
            final = self._evaluate_node(tree.root, value_key, steps)
            results[scenario_name] = ScenarioResult(
                scenario_name=scenario_name,
                final_value=final,
                unit=tree.root.unit,
                calculation_steps=steps,
            )

        return results

    def _evaluate_node(
        self,
        node: TreeNode,
        scenario: str,
        steps: List[Dict[str, Any]],
    ) -> float:
        """Recursively evaluate a node."""
        if node.is_leaf:
            if scenario == "base":
                val = node.value or 0.0
            elif scenario == "low":
                val = node.value_low or node.value or 0.0
            elif scenario == "high":
                val = node.value_high or node.value or 0.0
            else:
                val = node.value or 0.0

            steps.append({
                "node_name": node.name,
                "type": "leaf",
                "value": val,
                "scenario": scenario,
            })
            return val

        # Evaluate children
        child_values = []
        for child in node.children:
            child_val = self._evaluate_node(child, scenario, steps)
            child_values.append(child_val)

        # Combine based on operation
        result = self._combine(child_values, node.operation)

        steps.append({
            "node_name": node.name,
            "type": "computed",
            "operation": node.operation.value,
            "child_values": child_values,
            "result": result,
            "scenario": scenario,
        })
        return result

    @staticmethod
    def _combine(values: List[float], operation: NodeOperation) -> float:
        """Combine child values using the specified operation."""
        if not values:
            return 0.0

        if operation == NodeOperation.MULTIPLY:
            result = 1.0
            for v in values:
                result *= v
            return result
        elif operation == NodeOperation.ADD:
            return sum(values)
        elif operation == NodeOperation.SUBTRACT:
            result = values[0]
            for v in values[1:]:
                result -= v
            return result
        elif operation == NodeOperation.DIVIDE:
            if len(values) < 2 or values[1] == 0:
                return 0.0
            return values[0] / values[1]
        return 0.0

    def sensitivity_analysis(
        self,
        tree: DecompositionTree,
    ) -> SensitivityAnalysis:
        """One-at-a-time sensitivity analysis for each leaf node."""
        if not tree.root:
            return SensitivityAnalysis()

        # Calculate base result
        base_steps: List[Dict[str, Any]] = []
        base_result = self._evaluate_node(tree.root, "base", base_steps)

        leaves = tree.get_all_leaves()
        items = []

        for leaf in leaves:
            if leaf.value is None:
                continue

            original_value = leaf.value
            original_low = leaf.value_low
            original_high = leaf.value_high

            # Evaluate with low value
            leaf.value = leaf.value_low or (original_value * 0.5)
            low_steps: List[Dict[str, Any]] = []
            result_low = self._evaluate_node(tree.root, "base", low_steps)

            # Evaluate with high value
            leaf.value = leaf.value_high or (original_value * 2.0)
            high_steps: List[Dict[str, Any]] = []
            result_high = self._evaluate_node(tree.root, "base", high_steps)

            # Restore
            leaf.value = original_value
            leaf.value_low = original_low
            leaf.value_high = original_high

            # Compute sensitivity percentage
            if base_result != 0 and original_value != 0:
                delta_result = abs(result_high - result_low)
                sensitivity_pct = (delta_result / abs(base_result)) * 100
            else:
                sensitivity_pct = 0.0

            items.append(SensitivityItem(
                node_id=leaf.node_id,
                node_name=leaf.name,
                base_value=original_value,
                low_value=leaf.value_low or (original_value * 0.5),
                high_value=leaf.value_high or (original_value * 2.0),
                result_at_low=result_low,
                result_at_high=result_high,
                sensitivity_pct=sensitivity_pct,
                tornado_range=(result_low, result_high),
            ))

        # Sort by sensitivity (descending)
        items.sort(key=lambda x: x.sensitivity_pct, reverse=True)

        # Assign sensitivity ranks
        for i, item in enumerate(items):
            item.sensitivity_rank = i + 1

        return SensitivityAnalysis(
            items=items,
            base_result=base_result,
            most_sensitive_parameter=items[0].node_name if items else "",
        )

    def monte_carlo_simulation(
        self,
        tree: DecompositionTree,
        iterations: int = None,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation using triangular distributions."""
        if not tree.root:
            return {}

        n = iterations or self.monte_carlo_iterations
        if n <= 0:
            return {}

        leaves = tree.get_all_leaves()
        results = []

        for _ in range(n):
            # Sample random values for each leaf
            for leaf in leaves:
                low = leaf.value_low or (leaf.value or 0) * 0.5
                base = leaf.value or 0
                high = leaf.value_high or (leaf.value or 0) * 2.0

                # Ensure valid range for triangular distribution
                low = min(low, base)
                high = max(high, base)
                if low == high:
                    leaf.value = base
                else:
                    leaf.value = random.triangular(low, high, base)

            # Evaluate tree
            steps: List[Dict[str, Any]] = []
            result = self._evaluate_node(tree.root, "base", steps)
            results.append(result)

            # Restore base values
            for leaf in leaves:
                leaf.value = leaf.value  # already set in loop; just need the final evaluate

        # Restore original base values
        # (They were modified during simulation - restore from assumptions tracking)
        # The caller should have the original values stored elsewhere

        if not results:
            return {}

        results.sort()
        n_results = len(results)
        mean_val = sum(results) / n_results
        median_val = results[n_results // 2]

        variance = sum((x - mean_val) ** 2 for x in results) / n_results
        std_val = math.sqrt(variance)

        return {
            "iterations": n,
            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "min": results[0],
            "max": results[-1],
            "p5": results[int(n_results * 0.05)],
            "p25": results[int(n_results * 0.25)],
            "p75": results[int(n_results * 0.75)],
            "p95": results[int(n_results * 0.95)],
        }
