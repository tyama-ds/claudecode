"""Fact-check stage - verifies claims in the report."""

from .base import BaseStage
from ..context import PipelineContext


class FactCheckStage(BaseStage):
    """
    Verifies factual claims in the report using deep_research_tool's fact checker.

    Falls back to LLM-based verification if fact_checker is unavailable.
    """

    stage_type = "fact_check"

    def execute(self, context: PipelineContext) -> PipelineContext:
        report = context.get_latest_report()
        if not report:
            raise ValueError("No report available for fact-checking.")

        fc_config = self.config.fact_check
        llm_config = self.config.get_llm_config(fc_config.llm_config)

        self._report_progress("ファクトチェック中...", 0.2)

        try:
            result = self._run_fact_checker(report, llm_config, fc_config)
        except ImportError:
            self._report_progress("LLMベースの検証にフォールバック...", 0.4)
            result = self._fallback_fact_check(report, context.topic, llm_config)

        context.fact_check_results = result
        self._report_progress("ファクトチェック完了", 1.0)
        return context

    def _run_fact_checker(self, text: str, llm_config, fc_config) -> dict:
        """Run deep_research_tool's fact checker."""
        from deep_research_tool.fact_checker import FactChecker, FactCheckerConfig

        config = FactCheckerConfig(
            extraction_llm_provider=llm_config.provider.value,
            verification_llm_provider=llm_config.provider.value,
            output_language="ja",
        )

        checker = FactChecker(config)
        report = checker.check(text)

        return {
            "source": "fact_checker",
            "accuracy_score": getattr(report, "accuracy_score", 0.0),
            "total_claims": getattr(report, "total_claims", 0),
            "verified_claims": getattr(report, "verified_claims", 0),
            "results": [
                {
                    "claim": getattr(r, "claim", ""),
                    "verdict": getattr(r, "verdict", ""),
                    "confidence": getattr(r, "confidence", ""),
                }
                for r in getattr(report, "results", [])
            ],
        }

    def _fallback_fact_check(self, report: str, topic: str, llm_config) -> dict:
        """LLM-based fact check when deep_research_tool unavailable."""
        prompt = f"""
以下のレポートに含まれる主要な事実主張を検証してください。

トピック: {topic}

レポート:
{report[:4000]}

以下の形式で検証結果を述べてください:
1. 主張の抽出（最大{self.config.fact_check.max_claims}件）
2. 各主張の信頼性評価（高/中/低/検証不能）
3. 問題のある主張の指摘

最後に全体の信頼性スコア（0.0〜1.0）を記載してください。
"""
        agent = self._create_temp_agent(llm_config)
        response = agent._call_llm(
            [{"role": "user", "content": prompt}],
            system_prompt="あなたは事実検証の専門家です。主張の正確性を厳密に評価してください。",
        )

        # Extract score
        import re
        score_match = re.search(r"(?:信頼性スコア|スコア)[：:]\s*(\d+\.?\d*)", response)
        score = float(score_match.group(1)) if score_match else 0.5

        return {
            "source": "llm_fallback",
            "accuracy_score": score,
            "verification_text": response,
            "provider": llm_config.provider.value,
        }

    def _create_temp_agent(self, llm_config):
        from ...agents.base import BaseAgent
        from ...config import AgentConfig, AgentRole

        temp_config = AgentConfig(name="fact_checker", role=AgentRole.EVALUATOR)

        class _TempAgent(BaseAgent):
            def generate_response(self, topic, conversation_history, context=None):
                pass

        return _TempAgent(temp_config, llm_config)
