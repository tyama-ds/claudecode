"""Synthesis stage - merges multiple research results into one report."""

from .base import BaseStage
from ..context import PipelineContext


class SynthesisStage(BaseStage):
    """
    Synthesizes multiple research results into a unified report.

    Reads all research results from context, identifies common themes,
    complementary findings, and contradictions, then produces a single
    multi-perspective report.
    """

    stage_type = "synthesis"

    def execute(self, context: PipelineContext) -> PipelineContext:
        if not context.research_results:
            raise ValueError("No research results available for synthesis.")

        llm_config = self.config.get_llm_config(self.config.synthesis.llm_config)

        self._report_progress("調査結果を統合中...", 0.3)

        # Build input from all research results
        research_summaries = []
        for name, result in context.research_results.items():
            research_summaries.append(
                f"### {name}（視点: {result.perspective}）\n\n{result.report_content}"
            )
        all_research = "\n\n---\n\n".join(research_summaries)

        # Build focus areas instruction
        focus_instruction = ""
        if self.config.synthesis.focus_areas:
            areas = "\n".join(f"- {a}" for a in self.config.synthesis.focus_areas)
            focus_instruction = f"\n\n特に以下の観点に焦点を当ててください:\n{areas}"

        prompt = f"""
以下は「{context.topic}」について、複数の視点から行われた調査結果です。

{all_research}

---

上記の調査結果を統合し、一つの包括的なレポートを作成してください。

要件:
1. 各視点からの主要な発見を統合する
2. 共通して指摘されている点を強調する
3. 視点間で矛盾する点があれば明示し、分析する
4. 各視点が補完し合う部分を活かす
5. 統合的な結論と提言を述べる{focus_instruction}

レポートはMarkdown形式で、以下の構成で作成してください:
- エグゼクティブサマリー
- 各視点からの主要な発見
- 統合分析
- 結論と提言

最大文字数: {self.config.synthesis.max_length}文字程度
"""
        from ...agents.base import BaseAgent
        from ...config import AgentConfig, AgentRole

        temp_config = AgentConfig(name="synthesizer", role=AgentRole.EVALUATOR)

        class _SynthAgent(BaseAgent):
            def generate_response(self, topic, conversation_history, context=None):
                pass

        agent = _SynthAgent(temp_config, llm_config)

        self._report_progress("統合レポート生成中...", 0.6)

        content = agent._call_llm(
            [{"role": "user", "content": prompt}],
            system_prompt="あなたは複数の調査結果を統合して高品質なレポートを作成する専門家です。",
        )

        context.synthesized_report = content
        self._report_progress("統合完了", 1.0)
        return context
