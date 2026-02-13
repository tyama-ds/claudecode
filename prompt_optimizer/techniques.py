"""
Prompt Engineering technique definitions.

Each technique includes metadata and instruction templates
used by the optimization engine to transform user prompts.
"""

from dataclasses import dataclass, field


@dataclass
class Technique:
    """A prompt engineering technique."""
    id: str
    name: str
    name_ja: str
    description: str
    description_ja: str
    category: str
    instruction: str
    example_pattern: str = ""
    tags: list[str] = field(default_factory=list)


# ── Technique Registry ──────────────────────────────────────────────

TECHNIQUES: dict[str, Technique] = {}


def _register(t: Technique) -> Technique:
    TECHNIQUES[t.id] = t
    return t


# ── 1. Chain of Thought (CoT) ──────────────────────────────────────

_register(Technique(
    id="cot",
    name="Chain of Thought (CoT)",
    name_ja="Chain of Thought (CoT) - 思考の連鎖",
    description="Guides the model to reason step-by-step before arriving at an answer.",
    description_ja="回答に至る前にステップバイステップで推論するよう誘導します。",
    category="reasoning",
    instruction=(
        "Structure the prompt so the model is explicitly asked to think through "
        "the problem step by step. Insert a section like 'Let's think step by step:' "
        "before the final answer request. Break the task into sequential reasoning stages."
    ),
    example_pattern="Let's think step by step:\nStep 1: ...\nStep 2: ...\nTherefore: ...",
    tags=["reasoning", "step-by-step"],
))

# ── 2. Zero-Shot CoT ───────────────────────────────────────────────

_register(Technique(
    id="zero_shot_cot",
    name="Zero-Shot CoT",
    name_ja="Zero-Shot CoT - ゼロショット思考連鎖",
    description="Adds 'Let's think step by step' without any examples, triggering reasoning.",
    description_ja="例を示さずに「ステップバイステップで考えよう」を追加し推論を引き出します。",
    category="reasoning",
    instruction=(
        "Append the phrase 'Let's think step by step.' at the end of the user's "
        "query or just before the answer request. This simple addition significantly "
        "improves reasoning quality without requiring examples."
    ),
    example_pattern="[Original prompt]\n\nLet's think step by step.",
    tags=["reasoning", "zero-shot"],
))

# ── 3. Tree of Thought (ToT) ──────────────────────────────────────

_register(Technique(
    id="tot",
    name="Tree of Thought (ToT)",
    name_ja="Tree of Thought (ToT) - 思考の木",
    description="Explores multiple reasoning paths in parallel, evaluating each branch.",
    description_ja="複数の推論パスを並行して探索し、各分岐を評価します。",
    category="reasoning",
    instruction=(
        "Restructure the prompt to ask the model to: "
        "(1) Generate 3 or more distinct approaches/hypotheses to the problem. "
        "(2) For each approach, reason through the implications and potential outcomes. "
        "(3) Evaluate which approach is strongest based on logical consistency. "
        "(4) Select the best path and provide the final answer with justification. "
        "Use clear headings like 'Approach A:', 'Approach B:', 'Evaluation:', 'Best Answer:'."
    ),
    example_pattern=(
        "Consider multiple approaches:\n"
        "Approach A: ...\nApproach B: ...\nApproach C: ...\n"
        "Evaluation of each approach: ...\n"
        "Best answer based on evaluation: ..."
    ),
    tags=["reasoning", "parallel", "evaluation"],
))

# ── 4. Graph of Thought (GoT) ─────────────────────────────────────

_register(Technique(
    id="got",
    name="Graph of Thought (GoT)",
    name_ja="Graph of Thought (GoT) - 思考のグラフ",
    description="Builds a network of interconnected reasoning nodes, allowing merging and refining.",
    description_ja="相互接続された推論ノードのネットワークを構築し、統合・洗練します。",
    category="reasoning",
    instruction=(
        "Restructure the prompt to instruct the model to: "
        "(1) Decompose the problem into sub-problems (nodes). "
        "(2) Solve each sub-problem independently. "
        "(3) Identify connections and dependencies between sub-solutions. "
        "(4) Merge compatible partial solutions into a coherent whole. "
        "(5) Refine the merged solution by resolving any conflicts. "
        "Encourage the model to revisit and revise earlier conclusions as new insights emerge."
    ),
    example_pattern=(
        "Sub-problem decomposition:\n"
        "Node 1: ... → Solution 1\n"
        "Node 2: ... → Solution 2\n"
        "Connections: Node 1 ↔ Node 2\n"
        "Merged solution: ...\n"
        "Refined final answer: ..."
    ),
    tags=["reasoning", "network", "merging"],
))

# ── 5. Few-Shot Prompting ─────────────────────────────────────────

_register(Technique(
    id="few_shot",
    name="Few-Shot Prompting",
    name_ja="Few-Shot Prompting - 少数例プロンプティング",
    description="Provides concrete input-output examples before the actual task.",
    description_ja="実際のタスクの前に具体的な入出力例を提示します。",
    category="examples",
    instruction=(
        "Add 2-3 high-quality examples before the actual query. Each example should "
        "demonstrate the desired input format and output format/style. "
        "Use a clear delimiter between examples (e.g., '---'). "
        "Ensure the examples cover different aspects or edge cases of the task."
    ),
    example_pattern=(
        "Example 1:\nInput: ...\nOutput: ...\n---\n"
        "Example 2:\nInput: ...\nOutput: ...\n---\n"
        "Now, your task:\nInput: [user's actual query]"
    ),
    tags=["examples", "demonstration"],
))

# ── 6. Self-Consistency ───────────────────────────────────────────

_register(Technique(
    id="self_consistency",
    name="Self-Consistency",
    name_ja="Self-Consistency - 自己一貫性",
    description="Generates multiple independent reasoning paths, then selects the most consistent answer.",
    description_ja="複数の独立した推論パスを生成し、最も一貫性のある回答を選択します。",
    category="reasoning",
    instruction=(
        "Instruct the model to: "
        "(1) Solve the problem using 3 completely different reasoning approaches. "
        "(2) Compare the conclusions from each approach. "
        "(3) Identify the answer that appears most frequently or is most logically consistent. "
        "(4) Present the consensus answer with confidence level."
    ),
    example_pattern=(
        "Solve this problem 3 different ways:\n"
        "Method 1: ...\nConclusion 1: ...\n"
        "Method 2: ...\nConclusion 2: ...\n"
        "Method 3: ...\nConclusion 3: ...\n"
        "Consensus answer: ..."
    ),
    tags=["reasoning", "voting", "reliability"],
))

# ── 7. ReAct (Reasoning + Acting) ─────────────────────────────────

_register(Technique(
    id="react",
    name="ReAct (Reasoning + Acting)",
    name_ja="ReAct - 推論と行動の交互実行",
    description="Interleaves reasoning (Thought) with actions (Act) and observations (Obs).",
    description_ja="推論（Thought）と行動（Act）と観察（Obs）を交互に実行します。",
    category="agent",
    instruction=(
        "Structure the prompt using the Thought-Action-Observation loop: "
        "(1) Thought: The model reasons about what to do next. "
        "(2) Action: The model specifies a concrete action to take. "
        "(3) Observation: The result of the action is provided. "
        "Repeat until the task is solved. "
        "This is especially useful for tasks requiring external information or multi-step execution."
    ),
    example_pattern=(
        "Thought 1: I need to ...\n"
        "Action 1: [action description]\n"
        "Observation 1: [result]\n"
        "Thought 2: Based on the observation, ...\n"
        "Action 2: [action description]\n"
        "Final Answer: ..."
    ),
    tags=["agent", "iterative", "tool-use"],
))

# ── 8. Role Prompting / Expert Persona ────────────────────────────

_register(Technique(
    id="role_prompting",
    name="Role Prompting / Expert Persona",
    name_ja="Role Prompting - 役割・専門家ペルソナ",
    description="Assigns a specific expert role to the model to leverage domain knowledge.",
    description_ja="モデルに特定の専門家の役割を割り当て、ドメイン知識を活用します。",
    category="framing",
    instruction=(
        "Prefix the prompt with a detailed role definition: "
        "'You are a [domain] expert with [N] years of experience in [specific area]. "
        "You specialize in [specifics].' "
        "The role should be relevant to the task. Be specific about the expertise level "
        "and domain to activate the model's relevant knowledge patterns."
    ),
    example_pattern=(
        "You are a senior [domain] expert with 20 years of experience. "
        "You specialize in [specific area] and are known for [specific trait].\n\n"
        "[Original prompt]"
    ),
    tags=["framing", "persona", "expertise"],
))

# ── 9. Structured Output ──────────────────────────────────────────

_register(Technique(
    id="structured_output",
    name="Structured Output / Format Control",
    name_ja="Structured Output - 構造化出力",
    description="Specifies an explicit output format (JSON, table, sections, etc.).",
    description_ja="明示的な出力形式（JSON、表、セクション等）を指定します。",
    category="format",
    instruction=(
        "Add explicit formatting instructions to the prompt. Specify: "
        "(1) The exact output structure (e.g., JSON schema, markdown sections, table format). "
        "(2) Required fields or sections. "
        "(3) Data types and constraints for each field. "
        "This reduces ambiguity and ensures the output is machine-parseable when needed."
    ),
    example_pattern=(
        "Respond in the following format:\n"
        "## Summary\n[1-2 sentence summary]\n"
        "## Key Points\n- Point 1\n- Point 2\n"
        "## Detailed Analysis\n[detailed content]\n"
        "## Conclusion\n[conclusion]"
    ),
    tags=["format", "structure", "parseable"],
))

# ── 10. Self-Refine ───────────────────────────────────────────────

_register(Technique(
    id="self_refine",
    name="Self-Refine / Iterative Improvement",
    name_ja="Self-Refine - 自己改善・反復改良",
    description="Generates a draft, critiques it, then produces an improved version.",
    description_ja="ドラフトを生成し、自己批評した上で改善版を出力します。",
    category="refinement",
    instruction=(
        "Instruct the model to follow a 3-phase process: "
        "(1) Draft: Generate an initial response. "
        "(2) Critique: Identify weaknesses, gaps, errors, or areas for improvement. "
        "(3) Refine: Produce an improved version addressing all identified issues. "
        "Optionally repeat the critique-refine cycle for higher quality."
    ),
    example_pattern=(
        "Phase 1 - Draft:\n[initial response]\n\n"
        "Phase 2 - Self-Critique:\n- Weakness 1: ...\n- Gap: ...\n\n"
        "Phase 3 - Refined Response:\n[improved response]"
    ),
    tags=["refinement", "iterative", "quality"],
))

# ── 11. Meta-Prompting ────────────────────────────────────────────

_register(Technique(
    id="meta_prompting",
    name="Meta-Prompting",
    name_ja="Meta-Prompting - メタプロンプティング",
    description="Asks the model to first design the optimal prompt strategy, then execute it.",
    description_ja="まず最適なプロンプト戦略を設計させ、その後実行させます。",
    category="meta",
    instruction=(
        "Add a meta-reasoning step: "
        "(1) Ask the model to first analyze what kind of task this is. "
        "(2) Have it determine the best approach/strategy for this type of task. "
        "(3) Then have it execute that strategy. "
        "This leverages the model's ability to reason about its own reasoning process."
    ),
    example_pattern=(
        "Before answering, analyze this task:\n"
        "1. What type of task is this?\n"
        "2. What's the best strategy for this task type?\n"
        "3. Now execute that strategy:\n[actual response]"
    ),
    tags=["meta", "strategy", "self-aware"],
))

# ── 12. Constraint-Based Prompting ────────────────────────────────

_register(Technique(
    id="constraint_based",
    name="Constraint-Based Prompting",
    name_ja="Constraint-Based Prompting - 制約ベースプロンプティング",
    description="Defines explicit constraints, boundaries, and rules the response must follow.",
    description_ja="レスポンスが従うべき明示的な制約・境界・ルールを定義します。",
    category="framing",
    instruction=(
        "Add a 'Constraints' or 'Rules' section that explicitly lists: "
        "(1) What the response MUST include. "
        "(2) What the response MUST NOT include. "
        "(3) Quality criteria (accuracy, completeness, conciseness). "
        "(4) Scope boundaries. "
        "This reduces hallucination and keeps the response focused."
    ),
    example_pattern=(
        "Constraints:\n"
        "- MUST: Be factually accurate and cite reasoning\n"
        "- MUST: Stay within the scope of [topic]\n"
        "- MUST NOT: Speculate beyond available information\n"
        "- MUST NOT: Exceed [N] words\n\n"
        "[Original prompt]"
    ),
    tags=["framing", "constraints", "precision"],
))

# ── 13. Step-Back Prompting ───────────────────────────────────────

_register(Technique(
    id="step_back",
    name="Step-Back Prompting",
    name_ja="Step-Back Prompting - 一歩引いた抽象化",
    description="First asks a higher-level abstract question, then uses that to answer the specific question.",
    description_ja="まず高レベルの抽象的な質問をし、その回答を使って具体的な質問に答えます。",
    category="reasoning",
    instruction=(
        "Restructure the prompt in two phases: "
        "(1) Step-Back Question: Ask a broader, more abstract version of the original question "
        "to establish foundational principles or context. "
        "(2) Original Question: Use the foundational understanding to answer the specific question. "
        "This helps the model ground its reasoning in first principles."
    ),
    example_pattern=(
        "Step-Back: What are the general principles of [broader topic]?\n"
        "[Answer to step-back question]\n\n"
        "Now, using these principles, answer the specific question:\n"
        "[Original specific question]"
    ),
    tags=["reasoning", "abstraction", "principles"],
))

# ── 14. RISEN Framework ──────────────────────────────────────────

_register(Technique(
    id="risen",
    name="RISEN Framework",
    name_ja="RISEN フレームワーク",
    description="Role, Instructions, Steps, End goal, Narrowing - a comprehensive prompting framework.",
    description_ja="Role, Instructions, Steps, End goal, Narrowing - 包括的プロンプティングフレームワーク。",
    category="framework",
    instruction=(
        "Structure the prompt using the RISEN framework: "
        "(R) Role: Define who the model should be. "
        "(I) Instructions: State the core task clearly. "
        "(S) Steps: Break down the process into sequential steps. "
        "(E) End goal: Describe the desired outcome explicitly. "
        "(N) Narrowing: Add constraints to focus the response."
    ),
    example_pattern=(
        "Role: You are a ...\n"
        "Instructions: Your task is to ...\n"
        "Steps:\n1. First, ...\n2. Then, ...\n3. Finally, ...\n"
        "End Goal: The output should ...\n"
        "Narrowing: Focus only on ... Do not include ..."
    ),
    tags=["framework", "structured", "comprehensive"],
))

# ── 15. Emotional Prompting ───────────────────────────────────────

_register(Technique(
    id="emotional",
    name="Emotional Prompting",
    name_ja="Emotional Prompting - 感情プロンプティング",
    description="Adds emotional context or urgency to improve response quality and engagement.",
    description_ja="感情的な文脈や緊急性を追加してレスポンス品質を向上させます。",
    category="framing",
    instruction=(
        "Add emotional context that communicates the importance and impact of the task: "
        "'This is very important for [reason]. The quality of this response will directly "
        "impact [stakeholder/outcome].' "
        "Research shows this can improve model output quality by conveying the stakes involved."
    ),
    example_pattern=(
        "This task is critically important because [reason]. "
        "A high-quality response will [positive impact], while errors could [negative impact].\n\n"
        "[Original prompt]\n\n"
        "Please give this your best effort as the outcome matters significantly."
    ),
    tags=["framing", "motivation", "quality"],
))


def get_all_techniques() -> list[dict]:
    """Return all techniques as serializable dicts for the frontend."""
    return [
        {
            "id": t.id,
            "name": t.name,
            "name_ja": t.name_ja,
            "description": t.description,
            "description_ja": t.description_ja,
            "category": t.category,
            "tags": t.tags,
        }
        for t in TECHNIQUES.values()
    ]


def get_technique_by_id(technique_id: str) -> Technique | None:
    return TECHNIQUES.get(technique_id)


def get_techniques_by_ids(technique_ids: list[str]) -> list[Technique]:
    return [TECHNIQUES[tid] for tid in technique_ids if tid in TECHNIQUES]
