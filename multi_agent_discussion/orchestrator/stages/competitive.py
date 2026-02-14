"""Competitive evaluation stage - compares and ranks research results."""

import json

from .base import BaseStage
from ..context import PipelineContext


class CompetitiveStage(BaseStage):
    """
    Evaluates and ranks multiple research results competitively.

    An evaluator agent reads all results, scores them against criteria,
    and either selects the best or merges top-N results.
    """

    stage_type = "competitive"

    def execute(self, context: PipelineContext) -> PipelineContext:
        if not context.research_results:
            raise ValueError("No research results available for competitive evaluation.")

        comp_config = self.config.competitive
        llm_config = self.config.get_llm_config(comp_config.evaluator_llm_config)

        self._report_progress("競争評価中...", 0.2)

        # Step 1: Evaluate each result
        rankings = self._evaluate_all(context, llm_config)
        context.competitive_rankings = rankings

        # Step 2: Either pick best or merge top-N
        if comp_config.merge_top_n > 0 and comp_config.merge_top_n < len(rankings):
            self._report_progress("上位結果を統合中...", 0.7)
            context.best_report = self._merge_top_n(context, rankings, comp_config.merge_top_n, llm_config)
        else:
            # Pick the best
            best = max(rankings, key=lambda r: r.get("total_score", 0))
            best_name = best["agent_name"]
            context.best_report = context.research_results[best_name].report_content

        self._report_progress("競争評価完了", 1.0)
        return context

    def _evaluate_all(self, context: PipelineContext, llm_config) -> list:
        """Evaluate all research results against criteria."""
        criteria = self.config.competitive.evaluation_criteria
        criteria_text = "\n".join(f"- {c}" for c in criteria)

        reports_text = ""
        for name, result in context.research_results.items():
            reports_text += f"\n### レポート: {name}\n視点: {result.perspective}\n\n{result.report_content}\n\n---\n"

        prompt = f"""
以下は「{context.topic}」について、異なる視点から作成された複数のレポートです。

{reports_text}

各レポートを以下の評価基準で0.0〜1.0のスコアで評価してください:
{criteria_text}

以下のJSON形式で結果を出力してください:
[
  {{
    "agent_name": "レポート名",
    "scores": {{"基準1": 0.8, "基準2": 0.7}},
    "total_score": 0.75,
    "strengths": ["強み1", "強み2"],
    "weaknesses": ["弱み1"]
  }}
]

JSONのみを出力してください。
"""
        agent = self._create_temp_agent(llm_config)
        response = agent._call_llm(
            [{"role": "user", "content": prompt}],
            system_prompt="あなたは厳格で公正なレポート評価者です。",
        )

        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Fallback: equal scores
            return [
                {"agent_name": name, "total_score": 0.5, "scores": {}, "strengths": [], "weaknesses": []}
                for name in context.research_results
            ]

    def _merge_top_n(self, context: PipelineContext, rankings: list, n: int, llm_config) -> str:
        """Merge the top N research results."""
        sorted_rankings = sorted(rankings, key=lambda r: r.get("total_score", 0), reverse=True)
        top_names = [r["agent_name"] for r in sorted_rankings[:n]]

        parts = []
        for name in top_names:
            if name in context.research_results:
                result = context.research_results[name]
                strengths = next((r.get("strengths", []) for r in rankings if r["agent_name"] == name), [])
                parts.append(
                    f"### {name}（視点: {result.perspective}）\n"
                    f"強み: {', '.join(strengths)}\n\n{result.report_content}"
                )

        prompt = f"""
以下は「{context.topic}」について上位評価を得たレポートです。

{chr(10).join(parts)}

これらのレポートの最も優れた部分を取り合わせ、一つの最高品質のレポートを作成してください。
各レポートの強みを活かし、弱みを補完してください。

Markdown形式で出力してください。
"""
        agent = self._create_temp_agent(llm_config)
        return agent._call_llm(
            [{"role": "user", "content": prompt}],
            system_prompt="あなたは複数のレポートの最良の要素を組み合わせる専門家です。",
        )

    def _create_temp_agent(self, llm_config):
        from ...agents.base import BaseAgent
        from ...config import AgentConfig, AgentRole

        temp_config = AgentConfig(name="evaluator", role=AgentRole.EVALUATOR)

        class _TempAgent(BaseAgent):
            def generate_response(self, topic, conversation_history, context=None):
                pass

        return _TempAgent(temp_config, llm_config)
