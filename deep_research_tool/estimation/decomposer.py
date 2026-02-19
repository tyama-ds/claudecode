"""
Problem decomposition for Fermi estimation.

Uses LLM to decompose a target metric into a multiplication tree
of estimable sub-components.
"""

import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NodeOperation(str, Enum):
    """How child nodes combine to produce parent value."""
    MULTIPLY = "multiply"
    ADD = "add"
    SUBTRACT = "subtract"
    DIVIDE = "divide"


@dataclass
class TreeNode:
    """A single node in the decomposition tree."""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    name_en: str = ""
    description: str = ""
    unit: str = ""

    # Values (filled during estimation phase)
    value: Optional[float] = None
    value_low: Optional[float] = None
    value_high: Optional[float] = None

    # Tree structure
    children: List["TreeNode"] = field(default_factory=list)
    operation: NodeOperation = NodeOperation.MULTIPLY

    # Estimation metadata
    is_leaf: bool = True
    is_evidence_backed: bool = False
    evidence_data_id: str = ""
    estimation_reasoning: str = ""
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "name_en": self.name_en,
            "description": self.description,
            "unit": self.unit,
            "value": self.value,
            "value_low": self.value_low,
            "value_high": self.value_high,
            "children": [c.to_dict() for c in self.children],
            "operation": self.operation.value,
            "is_leaf": self.is_leaf,
            "is_evidence_backed": self.is_evidence_backed,
            "evidence_data_id": self.evidence_data_id,
            "estimation_reasoning": self.estimation_reasoning,
            "confidence": self.confidence,
        }


@dataclass
class DecompositionTree:
    """Complete decomposition tree for a Fermi estimation."""
    tree_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    target_metric: str = ""
    root: Optional[TreeNode] = None
    decomposition_reasoning: str = ""
    alternative_decompositions: List[str] = field(default_factory=list)
    language: str = "ja"

    def get_all_leaves(self) -> List[TreeNode]:
        """Get all leaf nodes that need estimation."""
        if not self.root:
            return []
        leaves = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                leaves.append(node)
            else:
                stack.extend(node.children)
        return leaves

    def get_all_nodes(self) -> List[TreeNode]:
        """Get all nodes in the tree (BFS order)."""
        if not self.root:
            return []
        nodes = []
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            nodes.append(node)
            queue.extend(node.children)
        return nodes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "target_metric": self.target_metric,
            "root": self.root.to_dict() if self.root else None,
            "decomposition_reasoning": self.decomposition_reasoning,
            "alternative_decompositions": self.alternative_decompositions,
            "language": self.language,
        }


class Decomposer:
    """Decomposes a target metric into a multiplication tree using LLM."""

    DECOMPOSITION_PROMPT_JA = """あなたはフェルミ推定の専門家です。
以下のターゲット指標を、推定可能なサブコンポーネントに分解してください。

## ターゲット指標
{target_metric}

## 追加コンテキスト
{context}

## 利用可能なデータ
以下の数値データが既に存在します。これらを可能な限り活用してください：
{available_data_summary}

## 指示
1. ターゲット指標を、掛け算（または足し算）で組み合わせると元の指標になるサブコンポーネントに分解
2. 各サブコンポーネントは、データから直接取得可能か、合理的に推定可能なものにする
3. 分解の深さは2-4階層が適切
4. 代替的な分解アプローチも1-2個提示

## 出力形式 (JSON)
{{
    "decomposition_reasoning": "分解の考え方の説明",
    "tree": {{
        "name": "ターゲット指標名",
        "name_en": "English name",
        "description": "説明",
        "unit": "単位",
        "operation": "multiply",
        "children": [
            {{
                "name": "サブコンポーネント名",
                "name_en": "English name",
                "description": "説明",
                "unit": "単位",
                "children": []
            }}
        ]
    }},
    "alternative_decompositions": ["代替分解1の概要", "代替分解2の概要"]
}}

JSONのみを出力:"""

    DECOMPOSITION_PROMPT_EN = """You are a Fermi estimation expert.
Decompose the following target metric into estimable sub-components.

## Target Metric
{target_metric}

## Additional Context
{context}

## Available Data
The following numerical data is already available. Use them where possible:
{available_data_summary}

## Instructions
1. Decompose the target into sub-components that combine (multiply/add) to produce the target
2. Each sub-component should be either directly available or reasonably estimable
3. Tree depth of 2-4 levels is ideal
4. Provide 1-2 alternative decomposition approaches

## Output (JSON only)
{{
    "decomposition_reasoning": "Explanation of decomposition approach",
    "tree": {{
        "name": "Target metric name",
        "name_en": "English name",
        "description": "Description",
        "unit": "unit",
        "operation": "multiply",
        "children": [
            {{
                "name": "Sub-component",
                "name_en": "English name",
                "description": "Description",
                "unit": "unit",
                "children": []
            }}
        ]
    }},
    "alternative_decompositions": ["Alternative 1", "Alternative 2"]
}}

Output only JSON:"""

    def __init__(self, llm_client, language: str = "ja"):
        self.llm_client = llm_client
        self.language = language

    def decompose(
        self,
        target_metric: str,
        available_data_summary: str = "",
        context: str = "",
    ) -> DecompositionTree:
        """Decompose a target metric into a tree."""
        template = (
            self.DECOMPOSITION_PROMPT_JA
            if self.language == "ja"
            else self.DECOMPOSITION_PROMPT_EN
        )
        prompt = template.format(
            target_metric=target_metric,
            available_data_summary=available_data_summary or "なし" if self.language == "ja" else "None",
            context=context or "なし" if self.language == "ja" else "None",
        )

        try:
            response = self.llm_client.generate(prompt)
            if not response or not response.content:
                logger.warning("Empty LLM response for decomposition")
                return self._fallback_tree(target_metric)
            return self._parse_tree_response(response.content, target_metric)
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return self._fallback_tree(target_metric)

    def _parse_tree_response(self, content: str, target_metric: str) -> DecompositionTree:
        """Parse LLM JSON response into DecompositionTree."""
        content = content.strip()

        # Extract JSON from markdown code blocks if present
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
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON object in the content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(content[start:end])
                except json.JSONDecodeError:
                    logger.error("Failed to parse decomposition JSON")
                    return self._fallback_tree(target_metric)
            else:
                return self._fallback_tree(target_metric)

        tree_data = data.get("tree", {})
        root = self._build_tree_node(tree_data)

        return DecompositionTree(
            target_metric=target_metric,
            root=root,
            decomposition_reasoning=data.get("decomposition_reasoning", ""),
            alternative_decompositions=data.get("alternative_decompositions", []),
            language=self.language,
        )

    def _build_tree_node(self, node_dict: Dict[str, Any]) -> TreeNode:
        """Recursively build TreeNode from parsed dictionary."""
        children_data = node_dict.get("children", [])
        children = [self._build_tree_node(c) for c in children_data]

        operation_str = node_dict.get("operation", "multiply")
        try:
            operation = NodeOperation(operation_str)
        except ValueError:
            operation = NodeOperation.MULTIPLY

        return TreeNode(
            name=node_dict.get("name", ""),
            name_en=node_dict.get("name_en", ""),
            description=node_dict.get("description", ""),
            unit=node_dict.get("unit", ""),
            children=children,
            operation=operation,
            is_leaf=len(children) == 0,
        )

    def _fallback_tree(self, target_metric: str) -> DecompositionTree:
        """Create a simple fallback tree when LLM decomposition fails."""
        root = TreeNode(
            name=target_metric,
            description=target_metric,
            is_leaf=True,
        )
        return DecompositionTree(
            target_metric=target_metric,
            root=root,
            decomposition_reasoning="Fallback: direct estimation (decomposition failed)",
            language=self.language,
        )
