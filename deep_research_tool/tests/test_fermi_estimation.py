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
    DomainPrior,
    DomainPriorProvider,
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


# ========================================================================
# TreeNode.get_depth / DecompositionTree depth tests
# ========================================================================


class TestTreeNodeDepth:
    def test_leaf_depth(self):
        node = TreeNode(name="Leaf", is_leaf=True)
        assert node.get_depth() == 1

    def test_two_level_depth(self):
        child = TreeNode(name="Child", is_leaf=True)
        parent = TreeNode(
            name="Parent", children=[child],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        assert parent.get_depth() == 2

    def test_three_level_depth(self):
        leaf = TreeNode(name="Leaf", is_leaf=True)
        mid = TreeNode(
            name="Mid", children=[leaf],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        root = TreeNode(
            name="Root", children=[mid],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        assert root.get_depth() == 3

    def test_asymmetric_tree(self):
        """Deeper left branch should determine depth."""
        deep_leaf = TreeNode(name="DeepLeaf", is_leaf=True)
        mid = TreeNode(
            name="Mid", children=[deep_leaf],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        shallow_leaf = TreeNode(name="ShallowLeaf", is_leaf=True)
        root = TreeNode(
            name="Root", children=[mid, shallow_leaf],
            operation=NodeOperation.ADD, is_leaf=False,
        )
        assert root.get_depth() == 3


class TestDecompositionTreeNodeDepth:
    def test_get_node_depth(self):
        leaf = TreeNode(node_id="leaf1", name="Leaf", is_leaf=True)
        mid = TreeNode(
            node_id="mid1", name="Mid", children=[leaf],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        root = TreeNode(
            node_id="root1", name="Root", children=[mid],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        tree = DecompositionTree(root=root)

        assert tree.get_node_depth("root1") == 0
        assert tree.get_node_depth("mid1") == 1
        assert tree.get_node_depth("leaf1") == 2
        assert tree.get_node_depth("nonexistent") == -1

    def test_get_tree_depth(self):
        leaf = TreeNode(name="Leaf", is_leaf=True)
        root = TreeNode(
            name="Root", children=[leaf],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        tree = DecompositionTree(root=root)
        assert tree.get_tree_depth() == 2

    def test_empty_tree_depth(self):
        tree = DecompositionTree()
        assert tree.get_tree_depth() == 0
        assert tree.get_node_depth("any") == -1


# ========================================================================
# Sub-decomposition candidate identification tests
# ========================================================================


class TestSubDecompositionCandidates:
    def test_identifies_low_confidence_high_sensitivity(self):
        leaf_low_conf = TreeNode(
            node_id="lc", name="LowConf", value=10,
            value_low=5, value_high=20,
            is_leaf=True, confidence=0.4,
        )
        leaf_high_conf = TreeNode(
            node_id="hc", name="HighConf", value=100,
            value_low=90, value_high=110,
            is_leaf=True, confidence=0.9,
        )
        root = TreeNode(
            name="Root", children=[leaf_low_conf, leaf_high_conf],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        tree = DecompositionTree(root=root)

        from deep_research_tool.estimation.calculator import SensitivityItem
        sensitivity = SensitivityAnalysis(
            items=[
                SensitivityItem(
                    node_id="lc", node_name="LowConf",
                    sensitivity_pct=50.0,
                ),
                SensitivityItem(
                    node_id="hc", node_name="HighConf",
                    sensitivity_pct=5.0,
                ),
            ],
            base_result=1000,
        )

        config = FermiEstimationConfig(
            sub_decomposition_confidence_threshold=0.65,
            sub_decomposition_min_sensitivity_pct=10.0,
        )
        estimator = FermiEstimator(llm_client=None, config=config)
        candidates = estimator._identify_sub_decomposition_candidates(
            tree, sensitivity
        )

        assert len(candidates) == 1
        assert candidates[0].node_id == "lc"

    def test_skips_high_confidence_leaves(self):
        leaf = TreeNode(
            node_id="hc", name="HighConf", value=10,
            is_leaf=True, confidence=0.9,
        )
        root = TreeNode(
            name="Root", children=[leaf],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        tree = DecompositionTree(root=root)

        from deep_research_tool.estimation.calculator import SensitivityItem
        sensitivity = SensitivityAnalysis(
            items=[SensitivityItem(
                node_id="hc", node_name="HighConf",
                sensitivity_pct=80.0,
            )],
            base_result=10,
        )

        config = FermiEstimationConfig(
            sub_decomposition_confidence_threshold=0.65,
        )
        estimator = FermiEstimator(llm_client=None, config=config)
        candidates = estimator._identify_sub_decomposition_candidates(
            tree, sensitivity
        )
        assert len(candidates) == 0

    def test_skips_low_sensitivity_leaves(self):
        leaf = TreeNode(
            node_id="ls", name="LowSens", value=10,
            is_leaf=True, confidence=0.3,
        )
        root = TreeNode(
            name="Root", children=[leaf],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        tree = DecompositionTree(root=root)

        from deep_research_tool.estimation.calculator import SensitivityItem
        sensitivity = SensitivityAnalysis(
            items=[SensitivityItem(
                node_id="ls", node_name="LowSens",
                sensitivity_pct=5.0,
            )],
            base_result=10,
        )

        config = FermiEstimationConfig(
            sub_decomposition_min_sensitivity_pct=10.0,
        )
        estimator = FermiEstimator(llm_client=None, config=config)
        candidates = estimator._identify_sub_decomposition_candidates(
            tree, sensitivity
        )
        assert len(candidates) == 0

    def test_skips_when_depth_exceeded(self):
        """Leaf at max_tree_depth-1 has no room for children."""
        leaf = TreeNode(
            node_id="deep", name="Deep", value=10,
            is_leaf=True, confidence=0.3,
        )
        mid = TreeNode(
            node_id="mid", name="Mid", children=[leaf],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        root = TreeNode(
            node_id="root", name="Root", children=[mid],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        tree = DecompositionTree(root=root)

        from deep_research_tool.estimation.calculator import SensitivityItem
        sensitivity = SensitivityAnalysis(
            items=[SensitivityItem(
                node_id="deep", node_name="Deep",
                sensitivity_pct=80.0,
            )],
            base_result=10,
        )

        # max_tree_depth=3 means leaf at depth 2 is at limit (3-1=2)
        config = FermiEstimationConfig(
            max_tree_depth=3,
            sub_decomposition_confidence_threshold=0.65,
        )
        estimator = FermiEstimator(llm_client=None, config=config)
        candidates = estimator._identify_sub_decomposition_candidates(
            tree, sensitivity
        )
        assert len(candidates) == 0


# ========================================================================
# Sub-decomposition: Pet example (Calculator integration)
# ========================================================================


class TestPetSubDecompositionExample:
    def test_sub_decomposed_pet_count_calculation(self):
        """
        Verify that a sub-decomposed pet count tree
        produces the correct arithmetic result.

        Original: 1世帯あたりペット数 = 1.3 (single leaf)
        Sub-decomposed:
          = dog_ratio * dog_count + cat_ratio * cat_count
          = 0.55 * 1.24 + 0.45 * 1.74
          = 0.682 + 0.783 = 1.465
        """
        # Dog contribution
        dog_ratio = TreeNode(
            name="犬飼育世帯の比率", value=0.55,
            value_low=0.50, value_high=0.60,
            unit="ratio", is_leaf=True, is_evidence_backed=True,
            confidence=0.85,
        )
        dog_count = TreeNode(
            name="犬の平均飼育頭数", value=1.24,
            value_low=1.10, value_high=1.40,
            unit="頭", is_leaf=True, is_evidence_backed=True,
            confidence=0.90,
        )
        dog = TreeNode(
            name="犬の寄与", children=[dog_ratio, dog_count],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )

        # Cat contribution
        cat_ratio = TreeNode(
            name="猫飼育世帯の比率", value=0.45,
            value_low=0.40, value_high=0.50,
            unit="ratio", is_leaf=True, is_evidence_backed=True,
            confidence=0.85,
        )
        cat_count = TreeNode(
            name="猫の平均飼育頭数", value=1.74,
            value_low=1.50, value_high=2.00,
            unit="頭", is_leaf=True, is_evidence_backed=True,
            confidence=0.90,
        )
        cat = TreeNode(
            name="猫の寄与", children=[cat_ratio, cat_count],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )

        # Combined: ADD
        pet_node = TreeNode(
            name="1世帯あたりペット数",
            children=[dog, cat],
            operation=NodeOperation.ADD,
            is_leaf=False,
            unit="頭/世帯",
        )

        tree = DecompositionTree(root=pet_node)
        calc = Calculator()
        results = calc.evaluate_tree(tree)

        expected = 0.55 * 1.24 + 0.45 * 1.74
        assert abs(results["base"].final_value - expected) < 0.001

    def test_full_market_with_sub_decomposed_pet_count(self):
        """
        Full market size estimation with sub-decomposed pet count.

        Market = Households * OwnershipRate * PetsPerHH * AnnualSpend
        where PetsPerHH is sub-decomposed.
        """
        households = TreeNode(
            name="世帯数", value=55_000_000,
            value_low=54_000_000, value_high=56_000_000,
            unit="世帯", is_leaf=True, confidence=0.95,
        )
        ownership = TreeNode(
            name="飼育率", value=0.158,
            value_low=0.14, value_high=0.17,
            unit="%", is_leaf=True, confidence=0.80,
        )

        # Sub-decomposed pet count
        dog_ratio = TreeNode(
            name="犬比率", value=0.55,
            value_low=0.50, value_high=0.60,
            is_leaf=True, confidence=0.85,
        )
        dog_count = TreeNode(
            name="犬頭数", value=1.24,
            value_low=1.10, value_high=1.40,
            is_leaf=True, confidence=0.90,
        )
        dog = TreeNode(
            name="犬の寄与", children=[dog_ratio, dog_count],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        cat_ratio = TreeNode(
            name="猫比率", value=0.45,
            value_low=0.40, value_high=0.50,
            is_leaf=True, confidence=0.85,
        )
        cat_count = TreeNode(
            name="猫頭数", value=1.74,
            value_low=1.50, value_high=2.00,
            is_leaf=True, confidence=0.90,
        )
        cat = TreeNode(
            name="猫の寄与", children=[cat_ratio, cat_count],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        pet_count = TreeNode(
            name="ペット数/世帯", children=[dog, cat],
            operation=NodeOperation.ADD, is_leaf=False,
        )

        annual_spend = TreeNode(
            name="年間フード費", value=50_000,
            value_low=35_000, value_high=70_000,
            unit="円", is_leaf=True, confidence=0.50,
        )

        # Build the full tree
        owning_hh = TreeNode(
            name="飼育世帯数", children=[households, ownership],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        total_pets = TreeNode(
            name="ペット総頭数", children=[owning_hh, pet_count],
            operation=NodeOperation.MULTIPLY, is_leaf=False,
        )
        root = TreeNode(
            name="市場規模", children=[total_pets, annual_spend],
            operation=NodeOperation.MULTIPLY, is_leaf=False, unit="円",
        )

        tree = DecompositionTree(root=root)
        calc = Calculator()
        results = calc.evaluate_tree(tree)

        pet_per_hh = 0.55 * 1.24 + 0.45 * 1.74  # 1.465
        expected = 55_000_000 * 0.158 * pet_per_hh * 50_000
        assert abs(results["base"].final_value - expected) < 1

        # Verify sensitivity analysis works on deeper tree
        sensitivity = calc.sensitivity_analysis(tree)
        assert len(sensitivity.items) == 7  # 7 leaf nodes now
        assert sensitivity.base_result == results["base"].final_value


# ========================================================================
# FermiEstimationConfig sub-decomposition settings tests
# ========================================================================


class TestFermiEstimationConfigSubDecomp:
    def test_sub_decomp_defaults(self):
        c = FermiEstimationConfig()
        assert c.enable_sub_decomposition is True
        assert c.sub_decomposition_confidence_threshold == 0.65
        assert c.sub_decomposition_max_iterations == 3
        assert c.sub_decomposition_min_sensitivity_pct == 10.0

    def test_sub_decomp_clamping(self):
        c = FermiEstimationConfig(
            sub_decomposition_confidence_threshold=2.0,
            sub_decomposition_max_iterations=100,
            sub_decomposition_min_sensitivity_pct=-5.0,
        )
        assert c.sub_decomposition_confidence_threshold == 1.0
        assert c.sub_decomposition_max_iterations == 10
        assert c.sub_decomposition_min_sensitivity_pct == 0.0


# ========================================================================
# DomainPrior / DomainPriorProvider Tests
# ========================================================================


class TestDomainPrior:
    def test_to_dict(self):
        prior = DomainPrior(
            expected_order_low=10,
            expected_order_high=13,
            source="evidence",
            evidence_count=5,
            reasoning="test",
        )
        d = prior.to_dict()
        assert d["expected_order_low"] == 10
        assert d["expected_order_high"] == 13
        assert d["source"] == "evidence"

    def test_margin_evidence_many(self):
        prior = DomainPrior(source="evidence", evidence_count=5)
        assert prior.margin == 0.5

    def test_margin_evidence_few(self):
        prior = DomainPrior(source="evidence", evidence_count=1)
        assert prior.margin == 1.0

    def test_margin_static_rule(self):
        prior = DomainPrior(source="static_rule")
        assert prior.margin == 1.0

    def test_margin_llm(self):
        prior = DomainPrior(source="llm")
        assert prior.margin == 1.5

    def test_margin_fallback(self):
        prior = DomainPrior(source="unit_fallback")
        assert prior.margin == 3.0


class TestDomainPriorProviderLayer0:
    """Tests for Layer 0: Evidence-based prior."""

    def _make_data_store_with_points(self, points):
        """Helper to create a data store with given data points."""
        from deep_research_tool.evidence.numerical_extractor import (
            NumericalDataStore, NumericalDataPoint,
        )
        store = NumericalDataStore(research_topic="test")
        for p in points:
            store.add(NumericalDataPoint(**p))
        return store

    def test_prior_from_single_data_point(self):
        store = self._make_data_store_with_points([{
            "value": 2_500_000_000_000,  # 2.5 trillion
            "normalized_value": 2_500_000_000_000,
            "metric_name": "アプリ市場規模",
            "subject": "日本 スマホアプリ",
            "unit": "円",
            "raw_text": "約2.5兆円",
            "extraction_confidence": 0.8,
            "source_reliability": 0.9,
        }])
        provider = DomainPriorProvider()
        prior = provider.get_domain_prior(
            "日本のスマホアプリ市場の年間売上", "円", data_store=store,
        )
        assert prior.source == "evidence"
        assert prior.evidence_count == 1
        # 2.5 trillion = ~10^12.4, expected range ~10^10..10^14
        assert prior.expected_order_low <= 12
        assert prior.expected_order_high >= 12

    def test_prior_from_multiple_data_points(self):
        store = self._make_data_store_with_points([
            {
                "value": 2_500_000_000_000,
                "normalized_value": 2_500_000_000_000,
                "metric_name": "アプリ市場規模",
                "subject": "日本 アプリ市場",
                "unit": "円",
                "raw_text": "約2.5兆円",
                "extraction_confidence": 0.8,
                "source_reliability": 0.9,
            },
            {
                "value": 1_200_000_000_000,
                "normalized_value": 1_200_000_000_000,
                "metric_name": "モバイルゲーム市場",
                "subject": "日本 アプリ ゲーム",
                "unit": "円",
                "raw_text": "1.2兆円",
                "extraction_confidence": 0.7,
                "source_reliability": 0.8,
            },
            {
                "value": 800_000_000_000,
                "normalized_value": 800_000_000_000,
                "metric_name": "Google Play 売上",
                "subject": "日本 アプリ",
                "unit": "円",
                "raw_text": "8000億円",
                "extraction_confidence": 0.6,
                "source_reliability": 0.7,
            },
        ])
        provider = DomainPriorProvider()
        prior = provider.get_domain_prior(
            "日本のアプリ市場の年間売上", "円", data_store=store,
        )
        assert prior.source == "evidence"
        assert prior.evidence_count == 3
        # All values in 10^11..10^12 range
        assert prior.expected_order_low <= 11
        assert prior.expected_order_high >= 12

    def test_no_match_returns_none(self):
        store = self._make_data_store_with_points([{
            "value": 100,
            "metric_name": "unrelated metric",
            "subject": "something else",
            "unit": "kg",
            "raw_text": "100kg",
            "extraction_confidence": 0.9,
            "source_reliability": 0.9,
        }])
        provider = DomainPriorProvider()
        # This should fall through to static/fallback (not evidence)
        prior = provider.get_domain_prior(
            "日本の人口", "人", data_store=store,
        )
        # Should NOT be from evidence since there's no keyword match
        assert prior.source != "evidence"

    def test_empty_data_store_falls_through(self):
        from deep_research_tool.evidence.numerical_extractor import NumericalDataStore
        store = NumericalDataStore()
        provider = DomainPriorProvider()
        prior = provider.get_domain_prior(
            "some metric", "USD", data_store=store,
        )
        assert prior.source != "evidence"


class TestDomainPriorProviderLayer1:
    """Tests for Layer 1: Static rules."""

    def test_japan_population_match(self):
        provider = DomainPriorProvider()
        prior = provider._prior_from_static_rules("日本の人口", "人")
        assert prior is not None
        assert prior.source == "static_rule"
        assert prior.expected_order_low == 7
        assert prior.expected_order_high == 9

    def test_japan_gdp_yen(self):
        provider = DomainPriorProvider()
        prior = provider._prior_from_static_rules("日本のGDP", "円")
        assert prior is not None
        assert prior.expected_order_low == 14

    def test_no_match(self):
        provider = DomainPriorProvider()
        prior = provider._prior_from_static_rules("something random xyz", "abc")
        assert prior is None


class TestDomainPriorProviderLayer3:
    """Tests for Layer 3: Unit-based fallback."""

    def test_yen_fallback(self):
        provider = DomainPriorProvider()
        prior = provider._prior_from_unit_fallback("unknown metric", "円")
        assert prior.source == "unit_fallback"
        assert prior.expected_order_low == 0
        assert prior.expected_order_high == 16

    def test_usd_fallback(self):
        provider = DomainPriorProvider()
        prior = provider._prior_from_unit_fallback("unknown metric", "USD")
        assert prior.source == "unit_fallback"
        assert prior.expected_order_high == 14

    def test_percentage_fallback(self):
        provider = DomainPriorProvider()
        prior = provider._prior_from_unit_fallback("some rate", "%")
        assert prior.expected_order_low == -2
        assert prior.expected_order_high == 2

    def test_universal_fallback(self):
        provider = DomainPriorProvider()
        prior = provider._prior_from_unit_fallback("metric", "unknown_unit_xyz")
        assert prior.expected_order_low == 0
        assert prior.expected_order_high == 20


class TestDomainPriorProviderFullChain:
    """Tests for the full 4-layer fallback chain."""

    def _make_data_store(self, points=None):
        from deep_research_tool.evidence.numerical_extractor import (
            NumericalDataStore, NumericalDataPoint,
        )
        store = NumericalDataStore(research_topic="test")
        for p in (points or []):
            store.add(NumericalDataPoint(**p))
        return store

    def test_evidence_takes_priority_over_static(self):
        """When evidence exists, it should be used instead of static rules."""
        store = self._make_data_store([{
            "value": 125_000_000,
            "normalized_value": 125_000_000,
            "metric_name": "日本の人口",
            "subject": "日本 人口",
            "unit": "人",
            "raw_text": "1億2500万人",
            "extraction_confidence": 0.9,
            "source_reliability": 0.95,
        }])
        provider = DomainPriorProvider()
        prior = provider.get_domain_prior("日本の人口", "人", data_store=store)
        # Should use evidence, not static rule
        assert prior.source == "evidence"

    def test_static_rule_when_no_evidence(self):
        """Falls back to static rule when evidence doesn't match."""
        from deep_research_tool.evidence.numerical_extractor import NumericalDataStore
        store = NumericalDataStore()
        provider = DomainPriorProvider()
        prior = provider.get_domain_prior("日本の人口", "人", data_store=store)
        assert prior.source == "static_rule"

    def test_unit_fallback_when_nothing_matches(self):
        """Falls all the way to unit fallback."""
        from deep_research_tool.evidence.numerical_extractor import NumericalDataStore
        store = NumericalDataStore()
        provider = DomainPriorProvider()
        prior = provider.get_domain_prior("obscure metric xyz", "円", data_store=store)
        assert prior.source == "unit_fallback"


class TestValidatorDomainPriorIntegration:
    """Tests for domain prior integration in Validator."""

    def _make_data_store(self, points=None):
        from deep_research_tool.evidence.numerical_extractor import (
            NumericalDataStore, NumericalDataPoint,
        )
        store = NumericalDataStore(research_topic="test")
        for p in (points or []):
            store.add(NumericalDataPoint(**p))
        return store

    def test_valid_estimate_passes_prior(self):
        """Estimate within expected range should pass."""
        store = self._make_data_store([{
            "value": 2_500_000_000_000,
            "normalized_value": 2_500_000_000_000,
            "metric_name": "アプリ市場規模",
            "subject": "日本 アプリ 市場",
            "unit": "円",
            "raw_text": "約2.5兆円",
            "extraction_confidence": 0.8,
            "source_reliability": 0.9,
        }])
        v = Validator(data_store=store)
        result = v.validate(
            target_metric="日本のアプリ市場の年間売上",
            base_value=3_000_000_000_000,  # 3 trillion, close to reference
            low_value=2_000_000_000_000,
            high_value=4_000_000_000_000,
            unit="円",
        )
        prior_issues = [i for i in result.issues if i.category == "domain_prior"]
        assert len(prior_issues) == 0
        assert result.domain_prior is not None
        assert result.domain_prior.source == "evidence"

    def test_wildly_wrong_estimate_flagged(self):
        """Estimate way outside expected range should produce error."""
        store = self._make_data_store([{
            "value": 2_500_000_000_000,
            "normalized_value": 2_500_000_000_000,
            "metric_name": "アプリ市場規模",
            "subject": "日本 アプリ 市場",
            "unit": "円",
            "raw_text": "約2.5兆円",
            "extraction_confidence": 0.8,
            "source_reliability": 0.9,
        }])
        v = Validator(data_store=store)
        result = v.validate(
            target_metric="日本のアプリ市場の年間売上",
            base_value=1e20,  # Absurdly large
            low_value=5e19,
            high_value=2e20,
            unit="円",
        )
        prior_issues = [i for i in result.issues if i.category == "domain_prior"]
        assert len(prior_issues) >= 1
        # 10^20 vs expected ~10^12 = 8 orders off, well beyond any margin
        assert any(i.severity == "error" for i in prior_issues)

    def test_domain_prior_in_validation_result(self):
        """ValidationResult should include domain_prior field."""
        v = Validator()
        result = v.validate(
            target_metric="some metric", base_value=100,
            low_value=80, high_value=120, unit="USD",
        )
        assert result.domain_prior is not None
        d = result.to_dict()
        assert "domain_prior" in d

    def test_sanity_checks_total_includes_prior(self):
        """sanity_checks_total should include domain prior check."""
        v = Validator()
        result = v.validate(
            target_metric="test", base_value=100,
            low_value=80, high_value=120, unit="USD",
        )
        assert result.sanity_checks_total == 5
