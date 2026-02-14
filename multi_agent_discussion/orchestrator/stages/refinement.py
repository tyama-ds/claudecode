"""Refinement stage - iteratively improves a report through review cycles."""

from .base import BaseStage
from ..context import PipelineContext


class RefinementStage(BaseStage):
    """
    Iteratively refines a report through writer-reviewer cycles.

    A reviewer agent critiques the report, then a writer agent
    incorporates the feedback. This repeats until quality threshold
    is met or max iterations reached.
    """

    stage_type = "refinement"

    def execute(self, context: PipelineContext) -> PipelineContext:
        current_report = context.get_latest_report()
        if not current_report:
            raise ValueError("No report available for refinement.")

        refinement_config = self.config.refinement
        writer_llm = self.config.get_llm_config(refinement_config.llm_config)
        reviewer_llm = self.config.get_llm_config(refinement_config.reviewer_llm_config)

        criteria_text = "\n".join(f"- {c}" for c in refinement_config.review_criteria)

        for iteration in range(refinement_config.max_iterations):
            progress = iteration / refinement_config.max_iterations
            self._report_progress(f"洗練ループ {iteration + 1}/{refinement_config.max_iterations}", progress)

            # Step 1: Review
            review = self._review(current_report, context.topic, criteria_text, reviewer_llm)

            # Step 2: Check if quality is sufficient
            score = self._extract_score(review)
            if score >= refinement_config.quality_threshold:
                self._report_progress(f"品質基準達成 (スコア: {score:.2f})", 1.0)
                break

            # Step 3: Revise based on review
            current_report = self._revise(current_report, review, context.topic, writer_llm)
            context.refined_reports.append(current_report)

        context.synthesized_report = current_report
        return context

    def _review(self, report: str, topic: str, criteria: str, llm_config) -> str:
        """Generate a review of the report."""
        prompt = f"""
以下のレポートをレビューしてください。

トピック: {topic}

レポート:
{report}

評価基準:
{criteria}

以下の形式でレビューを記述してください:
1. 各評価基準に対するスコア（0.0〜1.0）と具体的なフィードバック
2. 改善が必要な具体的な箇所の指摘
3. 総合スコア（0.0〜1.0）

最後の行に「総合スコア: X.X」の形式で総合スコアを記載してください。
"""
        agent = self._create_temp_agent(llm_config)
        return agent._call_llm(
            [{"role": "user", "content": prompt}],
            system_prompt="あなたは厳格なレポートレビュアーです。建設的で具体的なフィードバックを提供してください。",
        )

    def _revise(self, report: str, review: str, topic: str, llm_config) -> str:
        """Revise the report based on review feedback."""
        prompt = f"""
以下のレポートを、レビューのフィードバックに基づいて改善してください。

トピック: {topic}

現在のレポート:
{report}

レビューフィードバック:
{review}

フィードバックの指摘を全て反映し、改善されたレポートを出力してください。
レポートのみを出力してください（メタコメントは不要です）。
"""
        agent = self._create_temp_agent(llm_config)
        return agent._call_llm(
            [{"role": "user", "content": prompt}],
            system_prompt="あなたはフィードバックを的確に反映してレポートを改善するライターです。",
        )

    def _extract_score(self, review: str) -> float:
        """Extract the quality score from a review."""
        import re
        # Look for "総合スコア: X.X" pattern
        match = re.search(r"総合スコア[：:]\s*(\d+\.?\d*)", review)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.0

    def _create_temp_agent(self, llm_config):
        """Create a temporary agent for LLM access."""
        from ...agents.base import BaseAgent
        from ...config import AgentConfig, AgentRole

        temp_config = AgentConfig(name="refiner", role=AgentRole.EVALUATOR)

        class _TempAgent(BaseAgent):
            def generate_response(self, topic, conversation_history, context=None):
                pass

        return _TempAgent(temp_config, llm_config)
