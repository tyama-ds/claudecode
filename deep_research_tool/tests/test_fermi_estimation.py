"""Tests for Fermi estimation module."""

import pytest

from deep_research_tool.estimation.decomposer import (
    TreeNode,
    DecompositionTree,
    NodeOperation,
    Decomposer,
)
from deep_research_tool.estimation.assumptions import (
    Assumption,
    AssumptionSource,
    AssumptionManager,
)
from deep_research_tool.estimation.calculator import (
    Calculator,
    ScenarioResult,
    SensitivityAnalysis,
)
from deep_research_tool.estimation.validator import (
    Validator,
    ValidationResult,
    ValidationIssue,
)
from deep_research_tool.estimation.fermi_estimator import (
    FermiEstimator,
    FermiEstimationConfig,
    FermiEstimationResult,
)


# ========================================================================
# TreeNode / DecompositionTree Tests
# ========================================================================


class TestTreeNode:
    def test_leaf_node(self):
        node = TreeNode(name="Test", value=100, is_leaf=True)
        assert node.is_leaf is True
        assert node.value == 100
        assert node.children == []

    def test_non_leaf_node(self):
        child1 = TreeNode(name="Child1", value=10)
        child2 = TreeNode(name="Child2", value=20)
        parent = TreeNode(
            name="Parent",
            children=[child1, child2],
            operation=NodeOperation.MULTIPLY,
            is_leaf=False,
        )
        assert parent.is_leaf is False
        assert len(parent.children) == 2

    def test_to_dict(self):
        node = TreeNode(name="Test", value=42, unit="USD")
        d = node.to_dict()
        assert d["name"] == "Test"
        assert d["value"] == 42
        assert d["unit"] == "USD"
        assert d["is_leaf"] is True

    def test_nested_to_dict(self):
        child = TreeNode(name="Child", value=5)
        parent = TreeNode(name="Parent", children=[child], is_leaf=False)
        d = parent.to_dict()
        assert len(d["children"]) == 1
        assert d["children"][0]["name"] == "Child"


class TestDecompositionTree:
    def _make_tree(self):
        leaf1 = TreeNode(name="A", value=10, value_low=5, value_high=20, is_leaf=True)
        leaf2 = TreeNode(name="B", value=3, value_low=2, value_high=5, is_leaf=True)
        leaf3 = TreeNode(name="C", value=100, value_low=80, value_high=120, is_leaf=True)
        mid = TreeNode(
            name="A*B", children=[leaf1, leaf2],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        root = TreeNode(
            name="Root", children=[mid, leaf3],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        return DecompositionTree(target_metric="Test", root=root)

    def test_get_all_leaves(self):
        tree = self._make_tree()
        leaves = tree.get_all_leaves()
        assert len(leaves) == 3
        names = {l.name for l in leaves}
        assert names == {"A", "B", "C"}

    def test_get_all_nodes(self):
        tree = self._make_tree()
        nodes = tree.get_all_nodes()
        assert len(nodes) == 5  # Root, A*B, C, A, B

    def test_empty_tree(self):
        tree = DecompositionTree(target_metric="Empty")
        assert tree.get_all_leaves() == []
        assert tree.get_all_nodes() == []

    def test_to_dict(self):
        tree = self._make_tree()
        d = tree.to_dict()
        assert d["target_metric"] == "Test"
        assert d["root"]["name"] == "Root"


# ========================================================================
# Calculator Tests
# ========================================================================


class TestCalculator:
    def _make_simple_tree(self):
        """Market Size = Volume * Price"""
        volume = TreeNode(
            name="Volume", value=1000, value_low=800, value_high=1200,
            unit="units", is_leaf=True,
        )
        price = TreeNode(
            name="Price", value=50, value_low=40, value_high=60,
            unit="USD", is_leaf=True,
        )
        root = TreeNode(
            name="Market Size", children=[volume, price],
            operation=NodeOperation.MULTIPLY, is_leaf=False, unit="USD",
        )
        return DecompositionTree(target_metric="Market Size", root=root)

    def test_evaluate_multiply(self):
        tree = self._make_simple_tree()
        calc = Calculator()
        results = calc.evaluate_tree(tree)

        assert results["base"].final_value == 50000  # 1000 * 50
        assert results["pessimistic"].final_value == 32000  # 800 * 40
        assert results["optimistic"].final_value == 72000  # 1200 * 60

    def test_evaluate_add(self):
        a = TreeNode(name="A", value=100, value_low=80, value_high=120, is_leaf=True)
        b = TreeNode(name="B", value=200, value_low=150, value_high=250, is_leaf=True)
        root = TreeNode(
            name="Total", children=[a, b],
            operation=NodeOperation.ADD, is_leaf=False,
        )
        tree = DecompositionTree(root=root)
        calc = Calculator()
        results = calc.evaluate_tree(tree)

        assert results["base"].final_value == 300
        assert results["pessimistic"].final_value == 230

    def test_evaluate_divide(self):
        a = TreeNode(name="Revenue", value=1000, value_low=800, value_high=1200, is_leaf=True)
        b = TreeNode(name="Units", value=100, value_low=80, value_high=120, is_leaf=True)
        root = TreeNode(
            name="Price", children=[a, b],
            operation=NodeOperation.DIVIDE, is_leaf=False,
        )
        tree = DecompositionTree(root=root)
        calc = Calculator()
        results = calc.evaluate_tree(tree)

        assert results["base"].final_value == 10.0

    def test_empty_tree(self):
        tree = DecompositionTree()
        calc = Calculator()
        results = calc.evaluate_tree(tree)
        assert results["base"].final_value == 0.0

    def test_sensitivity_analysis(self):
        tree = self._make_simple_tree()
        calc = Calculator()
        analysis = calc.sensitivity_analysis(tree)

        assert analysis.base_result == 50000
        assert len(analysis.items) == 2
        assert analysis.most_sensitive_parameter != ""

        # Check that items are sorted by sensitivity
        for i in range(len(analysis.items) - 1):
            assert analysis.items[i].sensitivity_pct >= analysis.items[i+1].sensitivity_pct

    def test_monte_carlo(self):
        tree = self._make_simple_tree()
        calc = Calculator(monte_carlo_iterations=100)
        mc = calc.monte_carlo_simulation(tree)

        assert mc["iterations"] == 100
        assert mc["min"] <= mc["p5"] <= mc["median"] <= mc["p95"] <= mc["max"]
        # Mean should be roughly near 50000 (triangular distributions)
        assert 20000 < mc["mean"] < 100000

    def test_deep_tree(self):
        """Test 3-level tree: (A * B) * (C + D)"""
        a = TreeNode(name="A", value=10, value_low=8, value_high=12, is_leaf=True)
        b = TreeNode(name="B", value=5, value_low=4, value_high=6, is_leaf=True)
        c = TreeNode(name="C", value=100, value_low=80, value_high=120, is_leaf=True)
        d = TreeNode(name="D", value=200, value_low=150, value_high=250, is_leaf=True)

        ab = TreeNode(name="A*B", children=[a, b], operation=NodeOperation.MULTIPLY, is_leaf=False)
        cd = TreeNode(name="C+D", children=[c, d], operation=NodeOperation.ADD, is_leaf=False)
        root = TreeNode(name="Root", children=[ab, cd], operation=NodeOperation.MULTIPLY, is_leaf=False)

        tree = DecompositionTree(root=root)
        calc = Calculator()
        results = calc.evaluate_tree(tree)

        # Base: (10 * 5) * (100 + 200) = 50 * 300 = 15000
        assert results["base"].final_value == 15000


# ========================================================================
# Validator Tests
# ========================================================================


class TestValidator:
    def test_sanity_checks_pass(self):
        v = Validator()
        result = v.validate(
            target_metric="Test", base_value=100, low_value=80,
            high_value=120, unit="USD",
        )
        assert result.is_valid is True
        errors = [i for i in result.issues if i.severity == "error"]
        assert len(errors) == 0

    def test_sanity_checks_low_gt_base(self):
        v = Validator()
        result = v.validate(
            target_metric="Test", base_value=100, low_value=200,
            high_value=300, unit="USD",
        )
        assert result.is_valid is False

    def test_sanity_checks_base_gt_high(self):
        v = Validator()
        result = v.validate(
            target_metric="Test", base_value=500, low_value=100,
            high_value=200, unit="USD",
        )
        assert result.is_valid is False

    def test_wide_range_warning(self):
        v = Validator()
        result = v.validate(
            target_metric="Test", base_value=100, low_value=1,
            high_value=10000, unit="USD",
        )
        warnings = [i for i in result.issues if i.severity == "warning" and i.category == "range"]
        assert len(warnings) == 1


# ========================================================================
# Assumption Tests
# ========================================================================


class TestAssumption:
    def test_to_dict(self):
        a = Assumption(
            node_id="n1", parameter_name="price",
            value=100, value_low=80, value_high=120,
            unit="USD", source=AssumptionSource.EVIDENCE_DIRECT,
        )
        d = a.to_dict()
        assert d["parameter_name"] == "price"
        assert d["value"] == 100
        assert d["source"] == "evidence_direct"


# ========================================================================
# FermiEstimationConfig Tests
# ========================================================================


class TestFermiEstimationConfig:
    def test_defaults(self):
        c = FermiEstimationConfig()
        assert c.enabled is False
        assert c.max_tree_depth == 4
        assert c.monte_carlo_iterations == 1000

    def test_clamping(self):
        c = FermiEstimationConfig(max_tree_depth=100, max_leaf_nodes=0)
        assert c.max_tree_depth == 6  # clamped to max
        assert c.max_leaf_nodes == 2  # clamped to min


# ========================================================================
# FermiEstimationResult Tests
# ========================================================================


class TestFermiEstimationResult:
    def test_to_dict(self):
        r = FermiEstimationResult(
            estimation_id="test",
            target_metric="Market Size",
            base_estimate=50000,
            low_estimate=30000,
            high_estimate=70000,
            unit="USD",
        )
        d = r.to_dict()
        assert d["base_estimate"] == 50000
        assert d["target_metric"] == "Market Size"

    def test_to_summary_ja(self):
        r = FermiEstimationResult(
            target_metric="日本の市場規模",
            base_estimate=50000,
            low_estimate=30000,
            high_estimate=70000,
            unit="億円",
            overall_confidence=0.6,
            evidence_backed_ratio=0.5,
        )
        summary = r.to_summary("ja")
        assert "フェルミ推定" in summary
        assert "ベース" in summary
        assert "悲観" in summary

    def test_to_summary_en(self):
        r = FermiEstimationResult(
            target_metric="Market Size",
            base_estimate=50000,
            low_estimate=30000,
            high_estimate=70000,
            unit="USD",
        )
        summary = r.to_summary("en")
        assert "Fermi Estimation" in summary
        assert "Base" in summary

    def test_format_number(self):
        assert FermiEstimationResult._format_number(0) == "0"
        assert "兆" in FermiEstimationResult._format_number(5_000_000_000_000)
        assert "億" in FermiEstimationResult._format_number(500_000_000)
        assert "万" in FermiEstimationResult._format_number(50_000)


# ========================================================================
# Integration Test (Calculator + Tree)
# ========================================================================


class TestIntegration:
    def test_full_estimation_flow_no_llm(self):
        """Test the calculation flow without LLM (manual tree setup)."""
        # Build a tree manually
        population = TreeNode(
            name="Japan Population",
            value=125_000_000, value_low=124_000_000, value_high=126_000_000,
            unit="people", is_leaf=True, is_evidence_backed=True, confidence=0.95,
        )
        ownership_rate = TreeNode(
            name="Smartphone Ownership Rate",
            value=0.85, value_low=0.80, value_high=0.90,
            unit="%", is_leaf=True, confidence=0.7,
        )
        avg_spend = TreeNode(
            name="Average Annual App Spend",
            value=5000, value_low=3000, value_high=8000,
            unit="JPY", is_leaf=True, confidence=0.5,
        )

        users = TreeNode(
            name="Smartphone Users",
            children=[population, ownership_rate],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        root = TreeNode(
            name="Japan App Market Size",
            children=[users, avg_spend],
            operation=NodeOperation.MULTIPLY, is_leaf=False, unit="JPY",
        )

        tree = DecompositionTree(
            target_metric="Japan App Market Size",
            root=root,
        )

        # Calculate
        calc = Calculator(monte_carlo_iterations=100)
        scenarios = calc.evaluate_tree(tree)

        base = scenarios["base"].final_value
        # 125M * 0.85 * 5000 = 531,250,000,000
        assert abs(base - 531_250_000_000) < 1

        # Sensitivity
        sensitivity = calc.sensitivity_analysis(tree)
        assert len(sensitivity.items) == 3
        assert sensitivity.base_result == base

        # Monte Carlo
        mc = calc.monte_carlo_simulation(tree)
        assert mc["iterations"] == 100

        # Build result
        result = FermiEstimationResult(
            target_metric="Japan App Market Size",
            base_estimate=base,
            low_estimate=scenarios["pessimistic"].final_value,
            high_estimate=scenarios["optimistic"].final_value,
            unit="JPY",
            scenarios=scenarios,
            sensitivity=sensitivity,
            monte_carlo=mc,
            overall_confidence=0.6,
        )

        # Verify summary is generated
        summary = result.to_summary("ja")
        assert "フェルミ推定" in summary
        assert len(summary) > 100

        # Verify serialization
        d = result.to_dict()
        assert d["base_estimate"] == base
        assert "base" in d["scenarios"]
